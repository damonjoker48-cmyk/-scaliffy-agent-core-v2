"""ConversationMemory V2 — n8n style.

ONE job: preserve a SMALL amount of recent dialogue so Luna knows what
the customer refers to (pronouns, ellipsis, previous choices).

- get_recent(store, channel, customer, limit=10): max 10 messages.
- append_user / append_assistant with per-message size cap.
- clear_episode for post-order resets.

Never stored here: Merchant Brain, catalogue, evidence, previous prompts,
previous runtime blocks, SessionState JSON, huge media payloads.

Assistant messages are DIALOGUE ONLY — never commercial truth. Evidence
always wins over an old assistant claim.
"""
from __future__ import annotations

from . import durable as _durable
from .config import MEMORY_LIMIT, MEMORY_MESSAGE_CHAR_CAP


def _cap(text: str) -> str:
    return str(text or "").strip()[:MEMORY_MESSAGE_CHAR_CAP]


def get_recent(
    *, store_id: str, channel: str, customer_id: str, limit: int = MEMORY_LIMIT,
) -> list[dict]:
    n = max(1, min(MEMORY_LIMIT, int(limit or MEMORY_LIMIT)))
    rows = _durable.memory_recent(
        store_id=str(store_id), channel=str(channel), customer_id=str(customer_id),
        limit=n,
    )
    out: list[dict] = []
    for row in rows:
        role = str(row.get("role") or "")
        text = _cap(row.get("text") or "")
        if role in ("customer", "assistant") and text:
            out.append({"role": role, "text": text})
    return out[-n:]


def append_user(*, store_id: str, channel: str, customer_id: str, text: str) -> int:
    cleaned = _cap(text)
    if not cleaned:
        return -1
    return _durable.memory_append(
        store_id=str(store_id), channel=str(channel), customer_id=str(customer_id),
        role="customer", text=cleaned, char_cap=MEMORY_MESSAGE_CHAR_CAP,
    )


def append_assistant(*, store_id: str, channel: str, customer_id: str, text: str) -> int:
    cleaned = _cap(text)
    if not cleaned:
        return -1
    return _durable.memory_append(
        store_id=str(store_id), channel=str(channel), customer_id=str(customer_id),
        role="assistant", text=cleaned, char_cap=MEMORY_MESSAGE_CHAR_CAP,
    )


def clear_episode(*, store_id: str, channel: str, customer_id: str) -> None:
    _durable.memory_clear(
        store_id=str(store_id), channel=str(channel), customer_id=str(customer_id)
    )
