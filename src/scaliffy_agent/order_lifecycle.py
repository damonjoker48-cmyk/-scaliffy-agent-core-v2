"""Post-order commerce session reset (deterministic, no LLM).

A completed/cancelled/expired order CLOSES its active commerce session.
Transactional fields must not leak into the next conversation:
old quantity/color/product/city/draft must never behave as current.

Durable history (identity, past orders) lives adapter-side in the orders
table and Turso rows — agent-core simply stops treating them as active.
Nothing here deletes history; it only stops confusing history with now.

Lifecycle handled from the adapter-supplied active_order status:
  NEW -> SHOPPING -> ORDER_DRAFT -> ORDER_SUBMITTED -> ORDER_CONFIRMED -> CLOSED
  ORDER_CANCELLED / ORDER_EXPIRED -> clear stale draft state.
"""
from __future__ import annotations


# Adapter status spellings observed across order/session payloads.
CLOSED_STATUSES = frozenset({
    "confirmed", "completed", "closed", "cancelled", "refused",
    "expired", "canceled",
})

# Single-order plausible quantity persisted into state. Larger asks still
# reach Luna via turn evidence; they just don't poison durable state
# (e.g. "50k followers" must never become quantity=50).
MAX_PERSISTED_QUANTITY = 12


def order_is_closed(active_order: dict | None) -> tuple[bool, str]:
    """Return (closed, status). Unknown/empty status is NOT closed."""
    if not isinstance(active_order, dict):
        return False, ""
    status = str(active_order.get("status") or "").strip().lower()
    return (status in CLOSED_STATUSES), status


def reset_closed_session(state) -> list[str]:
    """Clear transactional fields in place. Returns cleared field names."""
    cleared: list[str] = []
    for field in ("active_product_id", "active_variant_id", "quantity",
                  "customer_city", "customer_city_source",
                  "customer_city_updated_at", "purchase_intent",
                  "draft_order_id", "last_media_shown", "open_question"):
        if getattr(state, field, ""):
            setattr(state, field, "")
            cleared.append(field)
    return cleared


def apply_order_lifecycle(state, active_order: dict | None) -> dict:
    """Reset-on-close gate. Pure state transition report, no I/O."""
    closed, status = order_is_closed(active_order)
    if not closed:
        return {"reset": False, "status": status, "cleared": []}
    cleared = reset_closed_session(state)
    return {"reset": True, "status": status, "cleared": cleared}


def persistable_quantity(value: int) -> int:
    """Clamp turn quantity for durable state (evidence keeps the raw value)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    if 1 <= number <= MAX_PERSISTED_QUANTITY:
        return number
    return 0
