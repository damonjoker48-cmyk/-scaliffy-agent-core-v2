"""Deterministic evidence V2 — stateless, rebuilt from scratch every turn.

Answers: "What is objectively true right now?"

Rules:
- Inputs: merchant catalogue snapshot + SessionStateV2 + CURRENT message
  (+ deterministic resolver helpers). NEVER previous assistant text,
  NEVER RAG-as-truth, NEVER prior evidence.
- Wraps the proven V1 evidence builder so commercial truth stays
  identical to production Adam Luxe (99 / 179 / 35 MAD, no Marrakech
  exception), while guaranteeing the stateless contract with an
  explicit signature.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaliffy_agent import evidence as _evidence_v1  # noqa: E402

from .session_state import SessionStateV2  # noqa: E402


def build_evidence_v2(
    *,
    catalogue: dict | None = None,
    state: SessionStateV2 | None = None,
    current_message: str = "",
    resolver_status: str = "",
    resolver_product_id: str = "",
    requested_quantity: int = 0,
    resolved_city: str = "",
    city_source: str = "",
    color_status: dict | None = None,
    reel_status: str = "",
    reel_owner_is_merchant: bool = False,
    turso_data: dict | None = None,
) -> tuple[dict, str]:
    catalogue = catalogue if isinstance(catalogue, dict) else {}
    st = state if isinstance(state, SessionStateV2) else SessionStateV2()
    # Stateless: only the CURRENT message text + current catalogue + current
    # state fields enter. No history, no assistant text, no prior evidence.
    evidence, fragment = _evidence_v1.build_evidence(
        resolver_status=resolver_status or "NOT_FOUND",
        resolver_product_id=resolver_product_id or st.active_product_id,
        turso_data=turso_data if isinstance(turso_data, dict) else {},
        catalogue=catalogue,
        resolved_city=resolved_city,
        city_source=city_source,
        requested_quantity=int(requested_quantity or 0),
        state_product_id=st.active_product_id,
        reel_owner_is_merchant=bool(reel_owner_is_merchant),
        reel_status=reel_status or "",
        color_status=color_status if isinstance(color_status, dict) else None,
        message_text=str(current_message or ""),
    )
    if not isinstance(evidence, dict):
        evidence = {"resolver": "NOT_FOUND"}
    evidence.setdefault("resolver", resolver_status or "NOT_FOUND")
    return evidence, fragment
