"""Deterministic product-media mapping V2 (§14/§15).

The model NEVER decides whether a photo exists. The backend resolves:
  media reference -> canonical product_id (+ variant, type, priority)

Catalog entry shape:
  media_reference | product_id | variant | media_type | priority

Only compact references travel to the model (never URL blobs).
"""
from __future__ import annotations


MEDIA_CATALOG: tuple[dict, ...] = (
    # Known merchant Reels -> pack (test tenant 625374849).
    {"media_reference": "reel_adam_001", "product_id": "pack-1",
     "variant": "", "media_type": "reel", "priority": 1},
    {"media_reference": "media_pack_black_01", "product_id": "pack-1",
     "variant": "noir", "media_type": "reel", "priority": 1},
    {"media_reference": "reel_pack_duck_07", "product_id": "pack-1",
     "variant": "", "media_type": "reel", "priority": 2},
    # Deterministic photo inventory for the pack variants.
    {"media_reference": "black_photo_01", "product_id": "pack-1",
     "variant": "noir", "media_type": "photo", "priority": 1},
    {"media_reference": "black_photo_02", "product_id": "pack-1",
     "variant": "noir", "media_type": "photo", "priority": 2},
    {"media_reference": "white_photo_01", "product_id": "pack-1",
     "variant": "abyed", "media_type": "photo", "priority": 1},
    {"media_reference": "white_photo_02", "product_id": "pack-1",
     "variant": "abyed", "media_type": "photo", "priority": 2},
)

_BY_REF: dict[str, dict] = {m["media_reference"]: m for m in MEDIA_CATALOG}


def resolve_media(keys: object) -> dict:
    """Resolve incoming media keys -> deterministic status + product.

    Returns {"status", "product_id", "variant", "media_reference",
    "media_type"} with status in FOUND/AMBIGUOUS/unknown_media/no_media.
    """
    found: dict[str, dict] = {}
    if isinstance(keys, (str, bytes)):
        items = [keys]
    else:
        try:
            items = list(keys or ())
        except TypeError:
            items = []
    for key in items:
        ref = str(key or "").strip()
        if ref and ref in _BY_REF:
            found[ref] = _BY_REF[ref]
    if not found:
        return {"status": "unknown_media" if items else "no_media",
                "product_id": "", "variant": "", "media_reference": "",
                "media_type": ""}
    products = {m["product_id"] for m in found.values()}
    if len(products) > 1:
        return {"status": "AMBIGUOUS", "product_id": "", "variant": "",
                "media_reference": "", "media_type": ""}
    best = sorted(found.values(), key=lambda m: int(m.get("priority") or 99))[0]
    return {"status": "FOUND", "product_id": str(best["product_id"]),
            "variant": str(best.get("variant") or ""),
            "media_reference": str(best["media_reference"]),
            "media_type": str(best.get("media_type") or "")}


def available_media(*, product_id: str, variant: str = "",
                    media_type: str = "photo") -> list[str]:
    """Compact ordered media references for a product (+optional variant).

    Falls back to product-level entries when no variant entry exists.
    Returns references only (highest priority first).
    """
    pid = str(product_id or "").strip()
    var = str(variant or "").strip().lower()
    if not pid:
        return []
    exact = sorted(
        (m for m in MEDIA_CATALOG
         if m["product_id"] == pid and m.get("media_type") == media_type
         and str(m.get("variant") or "").lower() == var),
        key=lambda m: int(m.get("priority") or 99),
    )
    if exact or not var:
        return [str(m["media_reference"]) for m in exact]
    generic = sorted(
        (m for m in MEDIA_CATALOG
         if m["product_id"] == pid and m.get("media_type") == media_type
         and not str(m.get("variant") or "").strip()),
        key=lambda m: int(m.get("priority") or 99),
    )
    return [str(m["media_reference"]) for m in generic]
