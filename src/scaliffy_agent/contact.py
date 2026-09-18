"""WhatsApp redirect intent: deterministic detection, no LLM, no number in chat.

When the customer asks for WhatsApp / the number / continuing there,
agent-core emits contact_action=whatsapp_redirect. The ADAPTER resolves the
CURRENT merchant WhatsApp destination and renders the clickable button.
Luna must never write or invent a phone number.
"""
from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or "").casefold())
    return " ".join(
        "".join(c for c in nfkd if not unicodedata.combining(c)).split()
    )


_WHATSAPP_RE = re.compile(
    r"whats\s?app|whatsap|watsap|watsapp|واتساب|وات\s?اب|وتساب|vhatsapp",
    re.IGNORECASE,
)
_NUMBER_ASK_RE = re.compile(
    r"\b(?:nmra|nemra|namra|numero|num[eé]ro|num|ra9m|rakm|رقم|نمرة|النمرة)\b"
    r"|3tini|3tina|اعطيني|عطيني|اعطينا|سيفط|صيفط",
    re.IGNORECASE,
)
_CONTINUE_THERE_RE = re.compile(
    r"nkemel|nkmel|nkamal|نكمل|نکمل|continue|continuer|كمّل",
    re.IGNORECASE,
)


def requests_whatsapp(text: str) -> bool:
    """True when the turn asks for WhatsApp / the number / moving there."""
    value = str(text or "")
    if not _WHATSAPP_RE.search(value):
        return False
    # "WhatsApp" alone can be ambiguous; require a request/continuation cue
    # OR an explicit number ask alongside it.
    norm = _norm(value)
    if _NUMBER_ASK_RE.search(value) or _CONTINUE_THERE_RE.search(value):
        return True
    if re.search(
        r"\b(?:fin|fayn|fyn|where|ou|فين|فاين)\b.{0,20}wh?a?ts?ap+|\bwh?a?ts?ap+.{0,20}\b(?:fin|dyalkom|dyalek|dyali|deyal)\b",
        norm,
    ):
        return True
    return False


_PHONE_LIKE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?wa\.me/\d[\d\s]*"
    r"|\b0[67]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\b"
    r"|\b\+212\s?\d(?:\s?\d{2}){4}\b"
)


def strip_phone_attempts(text: str) -> tuple[str, bool]:
    """Remove wa.me links / Moroccan phone numbers Luna may have written.

    The transport button owns the destination; chat must not carry numbers.
    Returns (cleaned_text, stripped_any).
    """
    cleaned, count = _PHONE_LIKE_RE.subn("", str(text or ""))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, count > 0
