"""Normalizer: one input shape regardless of channel.

Transport-specific parsing (Instagram/Messenger/WhatsApp/test harness)
ends BEFORE Core V2. The Core only sees NormalizedMessage.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedMessage:
    store_id: str
    channel: str
    customer_id: str
    conversation_id: str
    source_message_id: str
    text: str = ""
    attachments: tuple = field(default_factory=tuple)
    media_reference: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not str(self.store_id or "").strip():
            raise ValueError("normalized_missing_store_id")
        if not str(self.customer_id or "").strip():
            raise ValueError("normalized_missing_customer_id")
        if not str(self.source_message_id or "").strip():
            raise ValueError("normalized_missing_message_id")


def normalize(
    *,
    store_id: str,
    channel: str = "test",
    customer_id: str,
    source_message_id: str,
    text: str = "",
    attachments: tuple | list = (),
    media_reference: str = "",
    timestamp: str = "",
    conversation_id: str = "",
) -> NormalizedMessage:
    """Build the single Core V2 input shape from any channel payload."""
    ch = str(channel or "test").strip().lower() or "test"
    conv = str(conversation_id or "").strip() or str(customer_id or "").strip()
    files: tuple = ()
    if isinstance(attachments, (list, tuple)):
        cleaned: list = []
        for item in attachments:
            if isinstance(item, dict):
                url = str(item.get("url") or "").strip()
                mime = str(item.get("mime_type") or item.get("type") or "").strip()
                mid = str(item.get("media_id") or item.get("id") or "").strip()
                if url or mid:
                    cleaned.append({"url": url[:2000], "mime_type": mime[:120], "media_id": mid[:200]})
            else:
                url = str(getattr(item, "url", "") or "").strip()
                if url:
                    cleaned.append({
                        "url": url[:2000],
                        "mime_type": str(getattr(item, "mime_type", "") or "")[:120],
                        "media_id": str(getattr(item, "media_id", "") or "")[:200],
                    })
        files = tuple(cleaned)
    return NormalizedMessage(
        store_id=str(store_id).strip(),
        channel=ch,
        customer_id=str(customer_id).strip(),
        conversation_id=conv,
        source_message_id=str(source_message_id).strip(),
        text=str(text or "")[:4000],
        attachments=files,
        media_reference=str(media_reference or "")[:500],
        timestamp=str(timestamp or "")[:80],
    )
