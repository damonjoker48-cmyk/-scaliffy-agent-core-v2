"""City provenance for shipping answers.

A shipping city is CURRENT evidence only when grounded in the live thread:
the current message, the recent 4-6 raw turns, or an explicit active-order
city corroborated by the thread. Anything else (adapter-persisted
conversation metadata, 24h state cache, catalogue defaults, first table
row) is STALE and must never enter Luna context as a current fact.

This mirrors the adapter's deterministic geo matching
(`commerce_context.location_from_text`) so both sides canonicalize alike.
"""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime, timezone


# Canonical Moroccan city table (same semantics as the adapter).
CITY_ALIASES = {
    "casa": "Casablanca", "casablanca": "Casablanca",
    "rabat": "Rabat", "rbat": "Rabat",
    "marrakech": "Marrakech", "marakech": "Marrakech",
    "fes": "Fes", "fez": "Fes", "fas": "Fes",
    "tanger": "Tanger", "tangier": "Tanger",
    "agadir": "Agadir", "oujda": "Oujda",
    "kenitra": "Kenitra", "tetouan": "Tetouan",
    "mohammedia": "Mohammedia", "meknes": "Meknes",
    "meknas": "Meknes", "eljadida": "El Jadida",
    "nouaceur": "Nouaceur", "berrechid": "Berrechid",
    "temara": "Temara", "mohamadia": "Mohammedia",
    "ouarzazate": "Ouarzazate", "nador": "Nador",
    "houceima": "Al Hoceima", "benimellal": "Beni Mellal",
    "khemisset": "Khemisset", "berkane": "Berkane",
    "taza": "Taza", "settat": "Settat",
    "essaouira": "Essaouira", "safi": "Safi",
    "ifrane": "Ifrane", "azrou": "Azrou",
    "الدار البيضاء": "Casablanca", "كازا": "Casablanca",
    "الرباط": "Rabat", "فاس": "Fes",
    "مراكش": "Marrakech", "طنجة": "Tanger",
    "أكادير": "Agadir", "وجدة": "Oujda",
    "القنيطرة": "Kenitra", "تطوان": "Tetouan",
    "المحمدية": "Mohammedia", "مكناس": "Meknes",
}

CURRENT_SOURCES = frozenset({"explicit_current_turn", "recent_thread", "active_order_current"})


def _key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    return " ".join(
        "".join(ch for ch in normalized if not unicodedata.combining(ch)).split()
    )


def city_in_text(text: str) -> str:
    """Return the canonical city if the text names exactly one, else ''.

    Unique-match only (same rule as the adapter): two different cities in one
    text is ambiguous, not evidence.
    """
    normalized = _key(text)
    matches: dict[str, str] = {}
    for alias, city in CITY_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(_key(alias)) + r"(?!\w)", normalized):
            matches[city] = city
    if len(matches) == 1:
        return next(iter(matches.values()))
    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_shipping_city(
    *,
    message_text: str,
    history: tuple | list = (),
    catalogue_location: str = "",
    active_order_city: str = "",
    stored_city: str = "",
    stored_source: str = "",
    stored_updated_at: str = "",
) -> dict:
    """Decide the shipping city AND where it came from.

    Returns a provenance record:
    {value, source, updated_at, corroborated_by}. Only sources in
    CURRENT_SOURCES may enter Luna context as current facts.
    """
    current_hit = city_in_text(message_text)
    if current_hit:
        return {
            "value": current_hit,
            "source": "explicit_current_turn",
            "updated_at": _now_iso(),
            "corroborated_by": "current_message",
        }
    # Recent raw thread window (same 4-6 turn doctrine as Luna context).
    # Customer turns ONLY: a previous assistant message must NEVER establish
    # a commercial fact, and a city used for a shipping quote is commercial.
    # (An old assistant hallucination mentioning Marrakech promoted itself to
    # current-thread city and then to a quoted fee.) Customer-stated cities
    # keep full continuity through their own turns; assistant echoes add
    # nothing that the customer's own words do not already ground.
    recent_texts: list[str] = []
    for turn in list(history or [])[-12:]:
        role = str(getattr(turn, "role", "") or "").lower()
        text = str(getattr(turn, "text", "") or "")
        if role in ("customer", "human") and any(
            ch.isalpha() or ch.isdigit() for ch in text
        ):
            recent_texts.append(text)
    recent_hit = ""
    for text in reversed(recent_texts[-6:]):
        hit = city_in_text(text)
        if hit:
            recent_hit = hit
            break
    if recent_hit:
        return {
            "value": recent_hit,
            "source": "recent_thread",
            "updated_at": _now_iso(),
            "corroborated_by": "recent_raw_window",
        }
    order_city = str(active_order_city or "").strip()
    if order_city:
        # An active-order city counts only when the live thread corroborates
        # it; otherwise it is a previous-order leftover.
        canon = CITY_ALIASES.get(_key(order_city).replace(" ", ""), "")
        canon = canon or (order_city if order_city in set(CITY_ALIASES.values()) else "")
        if canon and (canon == recent_hit or canon == current_hit):
            return {
                "value": canon,
                "source": "active_order_current",
                "updated_at": _now_iso(),
                "corroborated_by": "thread",
            }
        return {
            "value": "",
            "source": "stale_active_order",
            "updated_at": stored_updated_at or "",
            "corroborated_by": "none",
            "dropped_value": order_city[:120],
        }
    catalog_city = str(catalogue_location or "").strip()
    if catalog_city:
        return {
            "value": "",
            "source": "stale_adapter_metadata",
            "updated_at": "",
            "corroborated_by": "none",
            "dropped_value": catalog_city[:120],
        }
    if stored_city:
        return {
            "value": "",
            "source": "stale_cache",
            "updated_at": stored_updated_at or "",
            "corroborated_by": "none",
            "dropped_value": str(stored_city)[:120],
        }
    return {"value": "", "source": "none", "updated_at": "", "corroborated_by": "none"}


def now_epoch() -> float:
    return time.time()
