"""Core V2 constants — single source of truth for the test gate."""
from __future__ import annotations

import os

TEST_STORE_ID = "625374849"
PRODUCTION_STORE_ID = "166510782"
AGENT_CORE_VERSION_V2 = "v2_test"
AGENT_CORE_VERSION_CANARY = "v2_canary"


def canary_stores() -> tuple[str, ...]:
    """Prod tenants explicitly allowed through V2 (env CSV, default off)."""
    try:
        raw = os.environ.get("V2_CANARY_STORES") or ""
    except Exception:
        raw = ""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def is_canary_store(store_id: str = "", merchant_account_id: str = "") -> bool:
    allowed = set(canary_stores())
    if not allowed:
        return False
    return (str(store_id or "").strip() in allowed
            or str(merchant_account_id or "").strip() in allowed)

# Chat-memory bounds: rolling window, MAX 10 recent messages (§10).
MEMORY_LIMIT = 10
MEMORY_MESSAGE_CHAR_CAP = 600
MEMORY_TOTAL_CHAR_CAP = 2400

# Canonical Adam Luxe commercial truth cloned into the test tenant.
# Single pack 99 MAD + 1 gourmetta; 2 packs 179 MAD + 2 gourmettas;
# delivery 35 MAD everywhere in Morocco; NO free delivery exceptions.
SINGLE_PACK_PRICE = "99"
TWO_PACK_TOTAL = "179"
SHIPPING_PRICE = "35"
CURRENCY = "MAD"
