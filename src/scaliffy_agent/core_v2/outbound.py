"""Idempotent outbound V2 — durable, text-independent.

outbound_key = store_id + channel + conversation_id + source_message_id

- Before outbound: atomic claim (only the claim winner sends).
- After successful send: mark success.
- Retry with same key + success exists -> NO SEND.
- Duplicate detection is NEVER based on reply text.
"""
from __future__ import annotations

from . import durable as _durable
from .execution import build_outbound_key


def claim_outbound(
    *, store_id: str, channel: str, conversation_id: str, source_message_id: str,
) -> tuple[str, bool]:
    key = build_outbound_key(
        store_id=store_id, channel=channel, conversation_id=conversation_id,
        source_message_id=source_message_id,
    )
    owned = _durable.outbound_claim(outbound_key=key)
    return key, owned


def mark_sent(*, outbound_key: str, reply: str) -> None:
    _durable.outbound_mark_success(outbound_key=outbound_key, reply=reply)


def already_sent(*, outbound_key: str) -> tuple[bool, str]:
    status = _durable.outbound_status(outbound_key=outbound_key)
    if status and status.get("status") == "success":
        return True, str(status.get("reply") or "")
    return False, ""
