"""Reel/media -> product resolution (deterministic, store-scoped).

Priority (never a visual/semantic guess):
1. Adapter-resolved media_context (merchant_media + product_id from the
   exact catalogue match in the SaaS adapter).
2. Curated mapping file reel_products.json: {store_id: {media_key: product}}
   where media_key is an Instagram media_id, reel_id, permalink or URL.
3. Caption exact product-name match is the ADAPTER's job; agent-core only
   consumes its verdict.

Statuses mirror the product resolver: FOUND / NOT_FOUND / AMBIGUOUS.
Resolved product seeds SessionState.active_product_id so follow-ups
("chhal?", "bghit jouj", "seft lia tsawer") work without retyping names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReelResolution:
    status: str  # FOUND | NOT_FOUND | AMBIGUOUS
    product_id: str = ""
    variant_id: str = ""
    match_kind: str = ""
    # True when the media comes from the merchant's own account
    # (adam_luxe.mo): OUR store's Reel, never a third-party business.
    owner_is_merchant: bool = False


# Instagram account(s) owned by the merchant Luna represents.
MERCHANT_OWN_ACCOUNTS = frozenset({"adam_luxe.mo"})


def _str(value: object, limit: int = 300) -> str:
    return str(value or "").strip()[:limit]


def _mapping_for_store(store_id: str) -> dict:
    path = os.environ.get("REEL_MAP_PATH") or str(
        Path(__file__).resolve().parent / "reel_products.json"
    )
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    scoped = data.get(store_id)
    return dict(scoped) if isinstance(scoped, dict) else {}


def _attachment_keys(attachments: tuple | list) -> set[str]:
    keys: set[str] = set()
    for item in attachments or ():
        get = (lambda k: item.get(k)) if isinstance(item, dict) else (
            lambda k: getattr(item, k, ""))
        for field in ("media_id", "reel_id", "permalink", "url", "thumbnail_url"):
            value = _str(get(field), 600)
            if value:
                keys.add(value)
                keys.add(value.rstrip("/").rsplit("/", 1)[-1])
    return {k for k in keys if k}


def resolve_reel_product(
    *,
    store_id: str,
    attachments: tuple | list = (),
    media_context: dict | None = None,
) -> ReelResolution:
    media_context = media_context if isinstance(media_context, dict) else {}
    owner_is_merchant = _owner_is_merchant(attachments, media_context)

    # 1. Adapter verdict (exact catalogue match, already tenant-scoped).
    # merchant_media=True means OUR catalogue matched: merchant-owned by
    # construction.
    if media_context.get("merchant_media") is True:
        product_id = _str(media_context.get("product_id"), 180)
        if product_id:
            return ReelResolution(
                "FOUND", product_id,
                _str(media_context.get("variant_id"), 180),
                "adapter_merchant_media", True,
            )
        return ReelResolution("NOT_FOUND", "", "", "adapter_unresolved",
                              owner_is_merchant)

    # 2. Curated store mapping.
    keys = _attachment_keys(attachments)
    extra_keys: set[str] = set()
    for field in ("media_id", "reel_id", "permalink", "caption"):
        value = _str(media_context.get(field), 600)
        if value:
            extra_keys.add(value)
    keys |= extra_keys
    if keys:
        mapping = _mapping_for_store(store_id)
        hits = {(mapping[k], k) for k in keys if k in mapping and _str(mapping[k])}
        products = {product for product, _ in hits}
        if len(products) == 1:
            return ReelResolution("FOUND", next(iter(products)), "", "reel_map",
                                  owner_is_merchant)
        if len(products) > 1:
            return ReelResolution("AMBIGUOUS", "", "", "reel_map_multi",
                                  owner_is_merchant)

    # 3. Unknown media: explicit, never a guess. Owner identity is still
    # reported: our Reel with unknown product ≠ unrelated content.
    has_media = bool(keys) or bool(media_context)
    return ReelResolution(
        "NOT_FOUND", "", "",
        "no_media" if not has_media else "unknown_media",
        owner_is_merchant,
    )


def _owner_is_merchant(attachments: tuple | list,
                       media_context: dict) -> bool:
    """Detect the merchant's own account in media metadata (deterministic)."""
    candidates: list[str] = []
    for item in attachments or ():
        get = (lambda k: item.get(k)) if isinstance(item, dict) else (
            lambda k: getattr(item, k, ""))
        for field in ("owner_username", "username", "sender_username",
                      "account", "page_name"):
            candidates.append(_str(get(field), 120))
    for field in ("owner_username", "username", "sender_username",
                  "account", "owner_id"):
        candidates.append(_str(media_context.get(field), 120))
    for value in candidates:
        cleaned = value.strip().lstrip("@").casefold()
        if cleaned in MERCHANT_OWN_ACCOUNTS:
            return True
    return False
