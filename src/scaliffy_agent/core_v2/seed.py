"""Seed: clone CURRENT Adam Luxe merchant truth into test store 625374849.

Everything belongs to store_id = 625374849. NEVER writes to 166510782.

Customer-facing truth preserved:
  product terminology: pack / الباك
  gift: gourmetta / ݣورميطة
  single pack: 99 MAD + 1 gourmetta
  2 packs: 179 MAD + 2 gourmettas
  delivery: 35 MAD everywhere in Morocco
  free_delivery = false (no Marrakech exception, no city exception,
  no free shipping for 2 packs as a SHIPPING rule — the 2-pack offer
  itself is 179 total with free delivery as an OFFER total).
"""
from __future__ import annotations

import hashlib

from .config import TEST_STORE_ID

TEST_MERCHANT_ACCOUNT_ID = "625374849"


def adam_test_brain_content() -> str:
    return (
        "Adam Luxe (test clone for 625374849). Le pack (الباك) se vend 99 MAD "
        "avec 1 gourmetta (ݣورميطة) OFFERTE — offre temporaire en cours. "
        "Deux packs 179 MAD avec 2 gourmettas offertes (offre temporaire). "
        "Livraison 35 MAD partout au Maroc, y compris Marrakech — no free "
        "shipping, no city exception, no quantity shipping exception. "
        "Tone: warm Moroccan Darija, natural, concise, 1-3 short lines. "
        "Product words: pack / الباك; gift words: gourmetta / ݣورميطة."
    )


def test_brain(*, store_id: str = TEST_STORE_ID) -> dict:
    if str(store_id) != TEST_STORE_ID:
        raise ValueError("seed_refuses_non_test_store")
    content = adam_test_brain_content()
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "merchant_id": TEST_MERCHANT_ACCOUNT_ID,
        "store_id": TEST_STORE_ID,
        "content": content,
        "checksum": checksum,
        "version": checksum[:16],
        "estimated_tokens": max(1, len(content) // 4),
    }


def test_catalogue(*, store_id: str = TEST_STORE_ID) -> dict:
    if str(store_id) != TEST_STORE_ID:
        raise ValueError("seed_refuses_non_test_store")
    return {
        "store_id": TEST_STORE_ID,
        "name": "الباك",
        "name_ar": "الباك",
        "product_id": "pack-1",
        "price": "99",
        "currency": "MAD",
        "available": "in_stock",
        "delivery_price": "35",
        "delivery_currency": "MAD",
        # Offer structure (backend truth):
        "offer_min_quantity": "2",
        "offer_total_price": "179",
        "offer_free_delivery": "true",
        "offer_free_bracelets": "2",
        "offer_temporary": "true",
        "single_offer_price": "99",
        "single_offer_bracelets": "1",
        # Variants / colors:
        "variants": "noir,abyed",
        "colors": "noir,abyed",
        "available_colors": "noir,abyed",
        # Order fields:
        "order_fields": "name,phone,address,city,quantity,color",
        # Merchant tone:
        "tone": "warm Moroccan Darija, concise, natural",
    }


def test_media_map(*, store_id: str = TEST_STORE_ID) -> dict:
    if str(store_id) != TEST_STORE_ID:
        raise ValueError("seed_refuses_non_test_store")
    return {
        # Known Reel / media -> product mappings (deterministic).
        "reel_adam_001": "pack-1",
        "media_pack_black_01": "pack-1",
        "reel_pack_duck_07": "pack-1",
    }


def seed_test_store(*, store_id: str = TEST_STORE_ID) -> dict:
    if str(store_id) != TEST_STORE_ID:
        raise ValueError("seed_refuses_non_test_store")
    brain = test_brain(store_id=store_id)
    catalogue = test_catalogue(store_id=store_id)
    media = test_media_map(store_id=store_id)
    return {
        "store_id": TEST_STORE_ID,
        "environment": "test",
        "brain": brain,
        "catalogue": catalogue,
        "media_map": media,
    }
