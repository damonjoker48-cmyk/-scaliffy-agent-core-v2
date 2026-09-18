"""Server-side tenant resolution. No LLM inference, no tenant guessing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TenantContext:
    store_id: str
    merchant_account_id: str
    customer_id: str
    conversation_id: str
    channel: str


def resolve_tenant(
    *,
    store_id: str,
    merchant_account_id: str,
    customer_id: str,
    channel: str,
    conversation_id: str = "",
) -> TenantContext:
    """Resolve runtime tenant facts from authenticated transport data.

    The channel adapter (Scaliffy transport/webhooks) already authenticated
    the Instagram/Messenger/WhatsApp account -> store binding. This function
    only normalizes and validates; it never asks Luna and never fuzzy-matches.
    """
    sid = str(store_id or "").strip()
    mid = str(merchant_account_id or "").strip()
    cid = str(customer_id or "").strip()
    ch = str(channel or "test").strip().lower() or "test"
    conv = str(conversation_id or cid).strip()
    if not sid:
        raise ValueError("tenant_missing_store_id")
    if not cid:
        raise ValueError("tenant_missing_customer_id")
    if not mid:
        # merchant_account_id is the durable merchant key (e.g. 166510782).
        # Fall back to store_id so older callers keep working, but never guess.
        mid = sid
    return TenantContext(
        store_id=sid,
        merchant_account_id=mid,
        customer_id=cid,
        conversation_id=conv or cid,
        channel=ch,
    )
