"""Merchant Brain: compact persistent commercial understanding per store.

Derived from the CURRENT adapter-supplied brain content (never invented).
Target 500-1200 tokens when sent to Luna (~4 chars/token heuristic).
Turso holds facts; the Brain holds the mental map.
"""
from __future__ import annotations

import hashlib
import re

from . import cache as tenant_cache


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def validate_brain(*, brain: dict, merchant_account_id: str) -> tuple[str, str, str]:
    """Return (content, checksum, version) or raise. Preserves current contract."""
    content = str((brain or {}).get("content") or "")
    checksum = str((brain or {}).get("checksum") or "")
    version = str((brain or {}).get("version") or "")
    if str((brain or {}).get("merchant_id") or "") != str(merchant_account_id or ""):
        raise ValueError("Adam Luxe Store Brain is absent or invalid")
    if not content:
        raise ValueError("Adam Luxe Store Brain is absent or invalid")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != checksum:
        raise ValueError("Adam Luxe Store Brain is absent or invalid")
    if version != checksum[:16]:
        raise ValueError("Adam Luxe Store Brain is absent or invalid")
    return content, checksum, version


def _sanitize_stale_labels(text: str) -> str:
    """Neutralize known-stale SaaS labels before Luna (deterministic, no LLM).

    Current Adam Luxe truth: product = pack / الباك, gift = gourmetta /
    ݣورميطة, shipping = 35 MAD everywhere including Marrakech. Old
    catalogue/owner-fact/template sources may still carry طقم, Luxury Swan
    Set, إسورة/سوار, bracelet, or a Marrakech-free exception. Internal IDs
    are untouched; only customer-facing words in the Luna-bound brain text
    are normalised. This is a data sanitizer, not a prompt instruction.
    """
    result = str(text or "")
    # Stale shipping exception -> current truth (documented SaaS sources 8A/8B/8D).
    stale_shipping = [
        ("Marrakech delivery is free (0 MAD)",
         "delivery is 35 MAD throughout Morocco, including Marrakech"),
        ("Marrakech 0 MAD (free)", "35 MAD fixed everywhere, including Marrakech"),
        ("Marrakech: free", "Marrakech: 35 MAD"),
        ("Marrakech free", "Marrakech: 35 MAD"),
        ("gratuite à Marrakech", "35 DH fixes partout, y compris à Marrakech"),
        ("free in Marrakech", "fixed 35 DH everywhere, including Marrakech"),
        ("فمراكش مجاني", "حتى فمراكش بـ35 درهم"),
        ("f Marrakech gratuite", "7ta f Marrakech b 35 DH"),
    ]
    for old, new in stale_shipping:
        result = result.replace(old, new)
    # Stale product/gift labels -> current customer-facing vocabulary.
    result = re.sub(r"Luxury\s+Swan\s+Set", "pack", result, flags=re.IGNORECASE)
    result = result.replace("طقم", "الباك")
    for old, new in (
        ("إسوارات", "ݣورميطات"), ("أساور", "ݣورميطات"), ("سوارات", "ݣورميطات"),
        ("إسورة", "ݣورميطة"), ("سوار", "ݣورميطة"),
    ):
        result = result.replace(old, new)
    result = re.sub(r"\bbracelets\b", "gourmettas", result, flags=re.IGNORECASE)
    result = re.sub(r"\bbracelet\b", "gourmetta", result, flags=re.IGNORECASE)
    result = re.sub(r"\bgourmettes\b", "gourmettas", result, flags=re.IGNORECASE)
    result = re.sub(r"\bgourmette\b", "gourmetta", result, flags=re.IGNORECASE)
    return result


def compact_brain(content: str, *, max_tokens: int = 1100) -> str:
    """Compact CURRENT brain content to ~500-1200 tokens without losing behavior.

    Strategy (deterministic, no LLM):
    - Strip stale historical/Pinecone export tail (already treated as non-truth
      in response_safety: [CATALOGUE SEED / HISTORICAL SOURCE ...]).
    - Neutralize known-stale customer-facing labels (Marrakech-free, طقم,
      Luxury Swan Set, إسورة/bracelet) to current truth.
    - Keep identity/tone/offers/contact/policy head sections verbatim.
    - If still over budget, cut from the middle (keep head + tail), never
      invent or rephrase.
    """
    text = str(content or "")
    # Drop historical export tail; current canonical catalogue outranks it.
    head = re.split(r"\[(?:CATALOGUE SEED|HISTORICAL SOURCE)", text, maxsplit=1)[0]
    head = head.strip() or text.strip()
    head = _sanitize_stale_labels(head)
    budget_chars = max_tokens * 4
    if len(head) <= budget_chars:
        return head
    # Keep head (identity/rules) + tail (contact/offers often at end).
    keep_head = int(budget_chars * 0.7)
    keep_tail = budget_chars - keep_head - 30
    return head[:keep_head].rstrip() + "\n…\n" + head[-keep_tail:].lstrip()


def load_merchant_brain(*, store_id: str, merchant_account_id: str, brain: dict) -> dict:
    """Validate, compact, and cache. Returns metrics + compact text."""
    content, checksum, version = validate_brain(brain=brain, merchant_account_id=merchant_account_id)
    key = tenant_cache.brain_key(store_id, version)
    cached = tenant_cache.cache_get(key)
    if isinstance(cached, dict) and cached.get("checksum") == checksum:
        return cached
    compact = compact_brain(content)
    record = {
        "content": compact,
        "full_checksum": checksum,
        "version": version,
        "estimated_tokens": _estimate_tokens(compact),
        "full_tokens": int((brain or {}).get("estimated_tokens") or _estimate_tokens(content)),
    }
    tenant_cache.cache_set(key, record, ttl_seconds=tenant_cache.cache_ttl_seconds("BRAIN_CACHE_TTL", 3600))
    return record
