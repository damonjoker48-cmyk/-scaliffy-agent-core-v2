"""Core V2 constants — single source of truth for the test gate."""
from __future__ import annotations

TEST_STORE_ID = "625374849"
PRODUCTION_STORE_ID = "166510782"
AGENT_CORE_VERSION_V2 = "v2_test"

# Chat-memory bounds (n8n style: small recent dialogue only).
MEMORY_LIMIT = 6
MEMORY_MESSAGE_CHAR_CAP = 600
MEMORY_TOTAL_CHAR_CAP = 2400

# Canonical Adam Luxe commercial truth cloned into the test tenant.
# Single pack 99 MAD + 1 gourmetta; 2 packs 179 MAD + 2 gourmettas;
# delivery 35 MAD everywhere in Morocco; NO free delivery exceptions.
SINGLE_PACK_PRICE = "99"
TWO_PACK_TOTAL = "179"
SHIPPING_PRICE = "35"
CURRENCY = "MAD"
