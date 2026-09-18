"""BuildAgentInput V2 — ONE pure function.

BuildAgentInput(merchant_brain, session_state, recent_messages, evidence,
               current_message) -> ONE compact final Luna payload.

MUST NOT: call database, mutate state, fetch catalogue, query media,
persist anything, append conversation, or call Luna. Everything needed
is passed explicitly. Output is reproducible and replayable: same inputs
=> byte-identical output.

Final Luna structure (conceptual):
  SYSTEM + compact Merchant Brain + compact SessionState +
  current deterministic Evidence + 4-6 recent messages + CURRENT message.
Recent messages appear EXACTLY ONCE (assertion + debug trace).
"""
from __future__ import annotations

import hashlib
import json


def _cap(text: str, limit: int) -> str:
    cleaned = str(text or "")
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"


def _estimate_tokens(text: str) -> int:
    return max(0, len(str(text or "")) // 4)


def build_agent_input(
    *,
    merchant_brain: dict,
    session_state: dict | str,
    recent_messages: list | tuple,
    evidence: dict,
    current_message: str,
) -> dict:
    # ---- normalize inputs (no I/O, no mutation of caller objects) ---------
    brain = merchant_brain if isinstance(merchant_brain, dict) else {}
    brain_text = _cap(str(brain.get("content") or ""), 4400)
    if isinstance(session_state, dict):
        state_text = _cap(str(session_state.get("fragment") or session_state.get("text") or ""), 1000)
        if not state_text or state_text in ("{}", "STATE: (no prior context)") and session_state:
            # Accept raw state dicts too: render allowlisted fields only.
            parts: list[str] = []
            for key in (
                "active_product_id", "selected_variant_id", "selected_color",
                "quantity", "customer_city", "purchase_intent", "current_offer_id",
                "open_question", "draft_order_id", "recent_media_id",
                "resolved_media_product_id",
            ):
                value = str(session_state.get(key) or "").strip()
                if value:
                    parts.append(f"{key}={value[:200]}")
            state_text = ("STATE: " + "; ".join(parts))[:1000] if parts else "STATE: (no prior context)"
    else:
        state_text = _cap(str(session_state or ""), 1000) or "STATE: (no prior context)"

    recent: list[dict] = []
    for item in list(recent_messages or [])[-6:]:
        if isinstance(item, dict):
            role = str(item.get("role") or "").strip()
            text = str(item.get("text") or item.get("content") or "").strip()[:600]
        else:
            role = str(getattr(item, "role", "") or "").strip()
            text = str(getattr(item, "text", "") or getattr(item, "content", "") or "").strip()[:600]
        if role not in ("customer", "assistant"):
            continue
        if len(text) < 1 or not any(c.isalpha() or c.isdigit() for c in text):
            continue
        recent.append({"role": role, "content": text})
    recent = recent[-6:]

    ev = evidence if isinstance(evidence, dict) else {}
    evidence_text = _cap(
        "EVIDENCE (exact facts for this turn):\n"
        + json.dumps(ev, ensure_ascii=False, separators=(",", ":")),
        1600,
    )
    current = _cap(str(current_message or ""), 4000)

    # ---- duplicate-history guard ------------------------------------------
    # Recent messages must appear exactly once in the final payload: current
    # message is separate; runtime block must NOT re-embed recent turns.
    seen_contents = [m["content"] for m in recent]
    if current and seen_contents.count(current) > 1:
        raise ValueError("build_agent_input_duplicate_history")
    if len(set(seen_contents)) != len(seen_contents):
        # Exact duplicate blocks (retries) collapse to one — deterministic.
        deduped: list[dict] = []
        seen: set[str] = set()
        for msg in recent:
            digest = hashlib.sha256(
                f"{msg['role']}\n{msg['content']}".encode("utf-8")
            ).hexdigest()
            if digest in seen and deduped and deduped[-1]["content"] == msg["content"]:
                continue
            if digest in seen:
                continue
            seen.add(digest)
            deduped.append(msg)
        recent = deduped[-6:]

    system_text = (
        "SCALIFFY CORE V2 (test): backend owns truth, Luna owns language. "
        "Answer the current intent directly and naturally; avoid unnecessary "
        "repetition; ask clarification only when truly needed. "
        "HISTORY IS CONTEXT ONLY (what the customer refers to). EVIDENCE IS "
        "TRUTH (price, shipping, offers, stock, colors, order actions). "
        "A previous assistant message NEVER establishes a commercial fact."
    )
    history_block_count = 1 if recent else 0

    payload = {
        "system": system_text,
        "merchant_brain": brain_text,
        "session_state": state_text,
        "evidence": evidence_text,
        "recent_messages": [dict(m) for m in recent],
        "current_message": current,
        "debug": {
            "history_blocks": history_block_count,
            "recent_count": len(recent),
            "duplicate_history": False,
        },
    }
    # Reproducibility seal (pure: derived only from the payload itself).
    payload["debug"]["payload_sha256"] = hashlib.sha256(
        json.dumps(
            {k: payload[k] for k in ("system", "merchant_brain", "session_state",
                                     "evidence", "recent_messages", "current_message")},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload["token_estimate"] = {
        "system_tokens": _estimate_tokens(system_text),
        "brain_tokens": _estimate_tokens(brain_text),
        "state_tokens": _estimate_tokens(state_text),
        "history_tokens": _estimate_tokens(" ".join(m["content"] for m in recent)),
        "evidence_tokens": _estimate_tokens(evidence_text),
        "current_tokens": _estimate_tokens(current),
    }
    payload["token_estimate"]["total_input_estimate"] = sum(
        payload["token_estimate"][k] for k in (
            "system_tokens", "brain_tokens", "state_tokens",
            "history_tokens", "evidence_tokens", "current_tokens",
        )
    )
    return payload
