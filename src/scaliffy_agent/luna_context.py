"""Luna input assembly with enforced token budgets.

Typical input:
CORE AGENT INSTRUCTIONS + MERCHANT BRAIN (~500-1200 tok) + COMPACT STATE
(~100-250) + 4-6 RECENT RAW (~200-600) + EVIDENCE (~50-400) + CURRENT MESSAGE.
Never: full catalog, full dump, full conversation, large Pinecone chunks.
"""
from __future__ import annotations


def _cap(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def build_runtime_block(
    *,
    state_fragment: str,
    recent_fragment: str = "",
    evidence_fragment: str,
    current_message: str,
) -> str:
    # Single-representation rule: conversation history travels ONLY as real
    # conversation messages (providers._conversation, 4-6 recent, capped).
    # It is NEVER embedded again inside this runtime block. recent_fragment
    # is accepted for backwards compatibility + logging but is NOT included
    # in the Luna payload, so the same customer turn appears exactly once.
    # Budgets in chars (~4 chars/token): state 1000, evidence 1600.
    state = _cap(state_fragment, 1000)
    evidence = _cap(evidence_fragment, 1600)
    current = _cap(current_message, 4000)
    return (
        f"{state}\n\n{evidence}\n\n"
        f"CURRENT CUSTOMER MESSAGE (answer this now, RAW CHAT WINS):\n{current}\n\n"
        "BACKEND DIRECTIVES (behavior, not scripts): answer every explicit part "
        "in one natural reply; quote evidenced offer totals exactly and never "
        "add shipping to a free-delivery total; never claim an order was created; "
        "speak as our own store (عندنا). "
        "HISTORY IS CONTEXT ONLY: recent conversation tells you WHAT the customer "
        "refers to (pronouns, ellipsis, previous choices). Deterministic EVIDENCE "
        "tells you WHAT IS TRUE (price, shipping, promos, stock, colors, gift "
        "quantity, order status, policy). A previous assistant message NEVER "
        "establishes a commercial fact — current evidence always wins; correct "
        "earlier unsupported claims instead of repeating them."
    )
