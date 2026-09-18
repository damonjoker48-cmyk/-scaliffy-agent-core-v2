"""Compact structured SessionState per store_id + channel + customer.

Migrates useful behavior from current active_order / known_customer / history
instead of blindly replacing it. Only continuity-useful fields are kept.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import cache as tenant_cache


@dataclass
class SessionState:
    active_product_id: str = ""
    active_variant_id: str = ""
    quantity: str = ""
    customer_city: str = ""
    # Provenance of customer_city: only CURRENT_SOURCES values may enter Luna
    # context. Anything stale stays stored (order-form continuity) but is
    # never shown to Luna as a current fact.
    customer_city_source: str = ""
    customer_city_updated_at: str = ""
    purchase_intent: str = ""
    draft_order_id: str = ""
    last_media_shown: str = ""
    open_question: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    def to_luna_fragment(self, *, include_city: bool = False) -> str:
        parts: list[str] = []
        if self.active_product_id:
            parts.append(f"active_product_id={self.active_product_id}")
        if self.active_variant_id:
            parts.append(f"active_variant_id={self.active_variant_id}")
        if self.quantity:
            parts.append(f"quantity={self.quantity}")
        if self.customer_city and include_city:
            parts.append(f"customer_city={self.customer_city}")
        if self.purchase_intent:
            parts.append(f"purchase_intent={self.purchase_intent}")
        if self.draft_order_id:
            parts.append(f"draft_order_id={self.draft_order_id}")
        if self.last_media_shown:
            parts.append(f"last_media_shown={self.last_media_shown}")
        if self.open_question:
            parts.append(f"open_question={self.open_question}")
        for key in sorted(self.extras.keys())[:8]:
            value = str(self.extras.get(key) or "").strip()
            if value:
                parts.append(f"{key}={value[:200]}")
        if not parts:
            return "STATE: (no prior context)"
        # ~100-250 tokens: hard cap at ~1000 chars.
        text = "STATE: " + "; ".join(parts)
        return text[:1000]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _str(value: object) -> str:
    return str(value or "").strip()


def load_state(
    *,
    store_id: str,
    channel: str,
    customer_id: str,
    active_order: dict | None,
    known_customer: dict | None,
) -> SessionState:
    """Load cached state, then migrate current-turn adapter state over it.

    Adapter active_order/known_customer win for the current turn (freshness),
    cached state preserves continuity across turns (memory).
    """
    key = tenant_cache.state_key(store_id, channel, customer_id)
    cached = tenant_cache.cache_get(key)
    state = SessionState()
    if isinstance(cached, dict):
        for fname in (
            "active_product_id", "active_variant_id", "quantity", "customer_city",
            "customer_city_source", "customer_city_updated_at",
            "purchase_intent", "draft_order_id", "last_media_shown", "open_question",
        ):
            value = _str(cached.get(fname))
            if value:
                setattr(state, fname, value[:200])
        extras = cached.get("extras")
        if isinstance(extras, dict):
            state.extras = {str(k)[:80]: _str(v)[:200] for k, v in list(extras.items())[:12] if _str(v)}

    order = active_order if isinstance(active_order, dict) else {}
    known = known_customer if isinstance(known_customer, dict) else {}
    # Migrate useful current behavior (field names vary by adapter version).
    product = _str(order.get("product") or order.get("product_id") or order.get("active_product"))
    if product:
        state.active_product_id = product[:200]
    variant = _str(order.get("variant") or order.get("variant_id") or order.get("finish"))
    if variant:
        state.active_variant_id = variant[:200]
    qty = _str(order.get("quantity") or order.get("qty"))
    if qty:
        state.quantity = qty[:20]
    # City from adapter payloads is UNVERIFIED until the live thread
    # corroborates it (see city_provenance.resolve_shipping_city). A stored
    # or payload city is continuity data, never current evidence by itself.
    city = _str(known.get("city") or known.get("customer_city") or order.get("city"))
    if city and city != state.customer_city:
        state.customer_city = city[:120]
        state.customer_city_source = "adapter_unverified"
        state.customer_city_updated_at = ""
    intent = _str(order.get("purchase_intent") or order.get("intent"))
    if intent:
        state.purchase_intent = intent[:120]
    draft = _str(order.get("draft_order_id") or order.get("order_id"))
    if draft:
        state.draft_order_id = draft[:120]
    return state


def save_state(*, store_id: str, channel: str, customer_id: str, state: SessionState) -> None:
    key = tenant_cache.state_key(store_id, channel, customer_id)
    tenant_cache.cache_set(
        key, state.to_dict(),
        ttl_seconds=tenant_cache.cache_ttl_seconds("STATE_CACHE_TTL", 86400),
    )
