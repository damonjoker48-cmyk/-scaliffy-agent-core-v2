from __future__ import annotations

from .types import Attachment


def visual_attachment_urls(attachments: tuple[Attachment, ...]) -> tuple[str, ...]:
    """Return only image payloads that the multimodal model can inspect."""
    urls: list[str] = []
    for attachment in attachments:
        url = str(attachment.url or "").strip()
        media_type = str(attachment.mime_type or "").strip().lower()
        is_image = (
            media_type.startswith("image/")
            or media_type in {"image", "photo", "sticker"}
            or url.startswith("data:image/")
        )
        if not is_image or not url.startswith(("https://", "http://", "data:image/")):
            continue
        if url not in urls:
            urls.append(url)
    return tuple(urls[:4])
