"""Execution ID: ONE durable id per logical customer message.

execution_id = store_id + channel + customer_id + source_message_id

Invariant: ONE source message = ONE execution = ONE Luna call =
maximum ONE outbound response. Webhook retries resolve to the same
execution (see pipeline + outbound idempotency).
"""
from __future__ import annotations


def build_execution_id(
    *,
    store_id: str,
    channel: str,
    customer_id: str,
    source_message_id: str,
) -> str:
    sid = str(store_id or "").strip()
    ch = str(channel or "test").strip().lower() or "test"
    cid = str(customer_id or "").strip()
    mid = str(source_message_id or "").strip()
    if not sid:
        raise ValueError("execution_missing_store_id")
    if not cid:
        raise ValueError("execution_missing_customer_id")
    if not mid:
        raise ValueError("execution_missing_message_id")
    return f"{sid}:{ch}:{cid}:{mid}"


def build_outbound_key(
    *,
    store_id: str,
    channel: str,
    conversation_id: str,
    source_message_id: str,
) -> str:
    sid = str(store_id or "").strip()
    ch = str(channel or "test").strip().lower() or "test"
    conv = str(conversation_id or "").strip()
    mid = str(source_message_id or "").strip()
    if not sid or not conv or not mid:
        raise ValueError("outbound_key_missing_field")
    return f"{sid}:{ch}:{conv}:{mid}"


def lock_namespace(*, store_id: str, channel: str, customer_id: str) -> str:
    return (
        f"conversation_lock:{str(store_id).strip()}:"
        f"{str(channel or 'test').strip().lower()}:"
        f"{str(customer_id).strip()}"
    )
