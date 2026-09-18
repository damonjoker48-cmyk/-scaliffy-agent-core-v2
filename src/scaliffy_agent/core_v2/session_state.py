"""SessionState V2 — strict structured commerce-episode state.

Answers: "Where is the customer in the current commerce episode?"

Allowed fields ONLY (everything else is rejected):
  active_product_id, selected_variant_id, selected_color, quantity,
  customer_city, purchase_intent, current_offer_id, open_question,
  draft_order_id, recent_media_id, resolved_media_product_id

FORBIDDEN inside SessionState: history, full catalogue, prompt, previous
evidence, Merchant Brain, previous SessionState snapshots, full media
descriptions. Updates PATCH/REPLACE fields — never append a whole new
state blob onto the old state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import durable as _durable

ALLOWED_FIELDS = (
    "active_product_id",
    "selected_variant_id",
    "selected_color",
    "quantity",
    "customer_city",
    "purchase_intent",
    "current_offer_id",
    "open_question",
    "draft_order_id",
    "recent_media_id",
    "resolved_media_product_id",
    "sales_stage",
)

SALES_STAGES = (
    "browsing",
    "interested",
    "selecting_variant",
    "objection",
    "ready_to_order",
    "collecting_order",
)

FORBIDDEN_KEYS = frozenset({
    "history", "catalogue", "catalog", "prompt", "evidence", "brain",
    "merchant_brain", "snapshots", "state_blob", "media_payload",
    "media_description", "full_media", "runtime", "system_prompt",
})


@dataclass
class SessionStateV2:
    active_product_id: str = ""
    selected_variant_id: str = ""
    selected_color: str = ""
    quantity: str = ""
    customer_city: str = ""
    purchase_intent: str = ""
    current_offer_id: str = ""
    open_question: str = ""
    draft_order_id: str = ""
    recent_media_id: str = ""
    resolved_media_product_id: str = ""
    sales_stage: str = ""

    def to_dict(self) -> dict:
        return dict(asdict(self))

    def to_luna_fragment(self) -> str:
        parts: list[str] = []
        for key in ALLOWED_FIELDS:
            value = str(getattr(self, key, "") or "").strip()
            if value:
                parts.append(f"{key}={value[:200]}")
        if not parts:
            return "STATE: (no prior context)"
        return ("STATE: " + "; ".join(parts))[:1000]

    def patch(self, updates: dict) -> list[str]:
        """PATCH/REPLACE allowed fields. Returns changed field names."""
        if not isinstance(updates, dict):
            raise ValueError("state_patch_must_be_dict")
        for key in updates:
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"state_forbidden_key:{key}")
            if key not in ALLOWED_FIELDS:
                raise ValueError(f"state_unknown_field:{key}")
        changed: list[str] = []
        for key in ALLOWED_FIELDS:
            if key in updates:
                value = str(updates[key] or "").strip()[:200]
                if getattr(self, key) != value:
                    setattr(self, key, value)
                    changed.append(key)
        return changed

    def reset_episode(self) -> list[str]:
        """Clear transactional episode fields after order close/cancel/etc."""
        cleared: list[str] = []
        for key in ALLOWED_FIELDS:
            if getattr(self, key):
                setattr(self, key, "")
                cleared.append(key)
        return cleared


def _str(value: object, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def load_state(
    *, store_id: str, channel: str, customer_id: str,
    active_order: dict | None = None, known_customer: dict | None = None,
) -> SessionStateV2:
    cached = _durable.session_load(
        store_id=str(store_id), channel=str(channel), customer_id=str(customer_id)
    )
    state = SessionStateV2()
    if isinstance(cached, dict):
        patch: dict[str, str] = {}
        for key in ALLOWED_FIELDS:
            value = _str(cached.get(key), 200)
            if value:
                patch[key] = value
        if patch:
            state.patch(patch)
    # Adapter freshness wins for the current turn (same rule as V1).
    order = active_order if isinstance(active_order, dict) else {}
    known = known_customer if isinstance(known_customer, dict) else {}
    fresh: dict[str, str] = {}
    product = _str(order.get("product") or order.get("product_id"))
    if product:
        fresh["active_product_id"] = product
    variant = _str(order.get("variant") or order.get("variant_id") or order.get("finish"))
    if variant:
        fresh["selected_variant_id"] = variant
    color = _str(order.get("color") or order.get("selected_color"))
    if color:
        fresh["selected_color"] = color
    qty = _str(order.get("quantity") or order.get("qty"), 20)
    if qty:
        fresh["quantity"] = qty
    city = _str(known.get("city") or known.get("customer_city") or order.get("city"), 120)
    if city and city != state.customer_city:
        fresh["customer_city"] = city
    intent = _str(order.get("purchase_intent") or order.get("intent"), 120)
    if intent:
        fresh["purchase_intent"] = intent
    draft = _str(order.get("draft_order_id") or order.get("order_id"), 120)
    if draft:
        fresh["draft_order_id"] = draft
    if fresh:
        state.patch(fresh)
    return state


def save_state(
    *, store_id: str, channel: str, customer_id: str, state: SessionStateV2,
) -> None:
    _durable.session_save(
        store_id=str(store_id), channel=str(channel), customer_id=str(customer_id),
        state=state.to_dict(),
    )
