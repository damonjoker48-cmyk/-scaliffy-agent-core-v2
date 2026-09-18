"""Deterministic sales-stage signals V2 (§7/§8).

Pure functions, no Luna, no I/O. The stage is BEHAVIORAL context only:
it never hardcodes replies and never carries history or catalogue data.

Stages: browsing < interested < selecting_variant, with objection as a
side state, and ready_to_order < collecting_order as terminal progression.
"""
from __future__ import annotations

import re

from .session_state import SALES_STAGES

_ORDER_RE = re.compile(
    r"n[kc]ommandi|commander|commande|كوموندي|نكوموندي|"
    r"\border\b|checkout|صافي\s*أكد|اكد\s*الطلب|bghit\s*n+[td]lob",
    re.IGNORECASE,
)
_OBJECTION_RE = re.compile(
    r"ghali|ghalia|nchof|nchouf|mazal|3lach|nfekr|nfakr|"
    r"trop\s*cher|\bcher\b|hésite|hesite|réfléch",
    re.IGNORECASE,
)
_PHOTO_RE = re.compile(
    r"tswira|tsawr|swira|photo|seft|send.*(pic|img)|voir.*(photo|image)",
    re.IGNORECASE,
)


def has_order_intent(text: str) -> bool:
    return bool(_ORDER_RE.search(str(text or "")))


def has_objection(text: str) -> bool:
    return bool(_OBJECTION_RE.search(str(text or "")))


def wants_photo(text: str) -> bool:
    return bool(_PHOTO_RE.search(str(text or "")))


_RANK = {"browsing": 0, "interested": 1, "selecting_variant": 2,
         "objection": 2, "ready_to_order": 3, "collecting_order": 4}


def next_stage(*, text: str = "", previous: str = "",
               color_decided: bool = False, quantity: int = 0,
               interest: bool = False, has_draft: bool = False) -> str:
    """Deterministic sticky stage progression for the current turn."""
    prev = str(previous or "").strip() or "browsing"
    if prev not in SALES_STAGES:
        prev = "browsing"
    message = str(text or "")
    if has_draft:
        return "collecting_order"
    if has_order_intent(message):
        return "ready_to_order"
    if has_objection(message) and prev not in ("ready_to_order", "collecting_order"):
        return "objection"
    if color_decided:
        return "selecting_variant"
    if interest and _RANK.get(prev, 0) < 1:
        return "interested"
    if prev == "objection" and (interest or color_decided):
        return "selecting_variant" if color_decided else "interested"
    return prev
