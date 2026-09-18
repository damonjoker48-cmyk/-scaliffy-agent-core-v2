"""Color confirmation for the order paper (deterministic, no LLM).

Before checkout, the merchant needs the customer's COLOR choice on the
order paper. This module reports the fact state only:

- confirmed: a thread-grounded color/variant (current message wins, then
  recent window, then already-stored state variant).
- missing: the product has several colors and none is grounded → Luna will
  naturally ask (evidence + existing clarification signal); checkout waits.
- single_option / unknown: nothing to ask → checkout may proceed.

No sentences, no templates: Luna stays fully free. Backend owns the gate.
"""
from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or "").casefold())
    return " ".join(
        "".join(c for c in nfkd if not unicodedata.combining(c)).split()
    )


def available_colors(catalogue: dict | None) -> list[dict]:
    """Exact color options from catalogue structures (verbatim, no invention)."""
    catalogue = catalogue if isinstance(catalogue, dict) else {}
    found: dict[str, dict] = {}
    variants = catalogue.get("visual_variants")
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            vid = str(variant.get("id") or variant.get("color") or "").strip()
            color = str(variant.get("color") or vid).strip()
            if not color:
                continue
            aliases = [str(a).strip() for a in (variant.get("aliases") or [])
                       if str(a or "").strip()]
            found.setdefault(color, {"color": color[:120], "variant_id": vid[:180],
                                     "aliases": aliases[:16]})
    for key in ("variants", "colors", "available_colors"):
        values = catalogue.get(key)
        if isinstance(values, list):
            for item in values:
                label = str(item if isinstance(item, str) else
                              (item.get("color") or item.get("name") or "")).strip()
                if label:
                    found.setdefault(label, {"color": label[:120], "variant_id": "",
                                             "aliases": []})
    return list(found.values())[:24]


def _mentions(text: str, labels: list[str]) -> str:
    """Return the first listed label named in text (canonical match)."""
    norm = " " + _norm(text) + " "
    for label in labels:
        needle = _norm(label)
        if needle and f" {needle} " in norm:
            return label
    return ""


def resolve_color(*, message_text: str, history: tuple | list = (),
                  catalogue: dict | None = None,
                  stored_variant: str = "") -> dict:
    """Decide color status. Never invents: unknown stays unknown."""
    options = available_colors(catalogue)
    if len(options) <= 1:
        single = options[0] if options else {}
        return {"status": "single_option" if options else "unknown",
                "variant": str(single.get("variant_id") or single.get("color") or ""),
                "color": str(single.get("color") or ""),
                "available_colors": [o["color"] for o in options]}
    labels: list[str] = []
    alias_to_option: dict[str, dict] = {}
    for option in options:
        labels.append(option["color"])
        alias_to_option[_norm(option["color"])] = option
        for alias in option.get("aliases") or []:
            labels.append(alias)
            alias_to_option.setdefault(_norm(alias), option)
    # Current message wins.
    hit = _mentions(message_text, labels)
    source = "explicit_current_turn" if hit else ""
    # Then the recent thread window (latest first).
    if not hit:
        for turn in reversed(list(history or [])[-12:]):
            role = str(getattr(turn, "role", "") or "").lower()
            if role not in ("customer", "assistant", "human"):
                continue
            hit = _mentions(getattr(turn, "text", ""), labels)
            if hit:
                source = "recent_thread"
                break
    if hit:
        option = alias_to_option.get(_norm(hit), {})
        return {"status": "confirmed", "source": source,
                "variant": str(option.get("variant_id") or hit)[:180],
                "color": str(option.get("color") or hit)[:120],
                "available_colors": [o["color"] for o in options]}
    if stored_variant:
        return {"status": "confirmed", "source": "stored_state",
                "variant": stored_variant[:180], "color": stored_variant[:120],
                "available_colors": [o["color"] for o in options]}
    return {"status": "missing", "source": "none", "variant": "",
            "color": "", "available_colors": [o["color"] for o in options]}
