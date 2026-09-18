"""Deterministic Evidence Builder BEFORE Luna.

Inspects store_id + SessionState + customer message + store data and retrieves
ONLY the commercial facts likely needed for this turn. Tiny object to Luna.

SHIPPING-CITY RULE (Fes-bug provenance fix): a `shipping_city` enters evidence
ONLY when the city is grounded in the live thread (current message, recent
4-6 raw turns, or corroborated active order). Adapter-persisted conversation
metadata, state cache, catalogue defaults and first-table-row fallbacks are
STALE and must never surface as current facts. A generic delivery question
gets the generic (city-free) fee, never a city.
"""
from __future__ import annotations

import json
import re


def _str(value: object, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


# Deterministic shipping-intent signal (no LLM, no history): when the CURRENT
# turn asks about delivery/shipping, evidence MUST carry the current shipping
# value before Luna. Never inferred from assistant history.
_SHIPPING_INTENT_RE = re.compile(
    r"livraison|delivery|shipping|tawsil|twsil|توصيل|شحن|شحال.*توصيل|chhal.*tawsil|"
    r"ch7al.*tawsil|bchhal.*tawsil|taman.*tawsil|prix.*livraison|combien.*livraison",
    re.IGNORECASE,
)

# Deterministic price-intent signal (no LLM): current turn asks about price.
_PRICE_INTENT_RE = re.compile(
    r"\b(?:price|prix|taman|thaman|chhal|ch7al|شحال|ثمن|سعر|بشحال|combien)\b",
    re.IGNORECASE,
)


def is_shipping_intent(message_text: str) -> bool:
    return bool(_SHIPPING_INTENT_RE.search(str(message_text or "")))


def is_price_intent(message_text: str) -> bool:
    return bool(_PRICE_INTENT_RE.search(str(message_text or "")))


# Sources allowed to place a city into evidence (see city_provenance).
CURRENT_CITY_SOURCES = frozenset({"explicit_current_turn", "recent_thread", "active_order_current"})


def build_evidence(
    *,
    resolver_status: str,
    resolver_product_id: str = "",
    turso_data: dict | None = None,
    catalogue: dict | None = None,
    resolved_city: str = "",
    city_source: str = "",
    requested_quantity: int = 0,
    state_product_id: str = "",
    reel_owner_is_merchant: bool = False,
    reel_status: str = "",
    color_status: dict | None = None,
    message_text: str = "",
) -> tuple[dict, str]:
    """Return (evidence_dict, evidence_fragment_for_luna ~50-400 tokens).

    Multi-intent: one compact object carries product + quantity + offer +
    shipping together so Luna answers every part of a big question in ONE
    reply. Nothing is invented: offer numbers come only from catalogue keys.
    """
    turso_data = turso_data if isinstance(turso_data, dict) else {}
    catalogue = catalogue if isinstance(catalogue, dict) else {}
    evidence: dict[str, object] = {"resolver": resolver_status}
    active_product = resolver_product_id or _str(state_product_id, 180)
    if active_product:
        evidence["active_product_id"] = active_product
    if requested_quantity > 0:
        evidence["requested_quantity"] = requested_quantity
    # Arabic display name wins when the adapter provides one (never Latin).
    default_name = _str(catalogue.get("name_ar") or "", 180)
    if default_name:
        evidence["default_pack_name"] = default_name
    # Size/material attributes, verbatim and capped (bracelet size questions
    # like "what size?" need exact facts, never guesses).
    for key in ("size", "measurements", "size_info", "size_guide", "material",
                "materials", "weight", "adjustable", "details",
                "order_specifications"):
        value = _str(catalogue.get(key), 400)
        if value:
            evidence[f"attr_{key}"] = value
    if reel_owner_is_merchant or reel_status:
        evidence["reel"] = {
            "owner": "merchant" if reel_owner_is_merchant else "unknown",
            "resolution": reel_status or "no_media",
        }
    if isinstance(color_status, dict) and color_status.get("status"):
        color_block: dict[str, object] = {
            "status": color_status.get("status"),
            "needed_for_order_paper": color_status.get("status") == "missing",
        }
        if color_status.get("variant"):
            color_block["variant"] = _str(color_status.get("variant"), 180)
        if color_status.get("color"):
            color_block["color"] = _str(color_status.get("color"), 120)
        available = color_status.get("available_colors")
        if isinstance(available, list) and available:
            color_block["available_colors"] = [str(c)[:120] for c in available[:12]]
        evidence["color"] = color_block

    rows = turso_data.get("product_rows") if isinstance(turso_data.get("product_rows"), list) else []
    row = None
    if resolver_product_id:
        for candidate in rows:
            if str(candidate.get("id") or "") == resolver_product_id:
                row = candidate
                break
    # No silent nearest: without an exact resolved product, no product row.
    # (Previous rows[0] fallback could ground Luna on the wrong product.)

    if row:
        evidence.update({
            "product_id": _str(row.get("id"), 180),
            "available": _str(row.get("available", row.get("stock")), 120),
            "price": _str(row.get("price"), 120),
            "currency": _str(row.get("currency", "MAD"), 16),
        })
    elif catalogue.get("name") or catalogue.get("product_id"):
        # Price fallback chain: exact `price` first, then the single-pack
        # offer price (adapter may carry 99 only as single_offer_price).
        # Never invented: empty stays empty and is handled as UNKNOWN below.
        fallback_price = (
            _str(catalogue.get("price"), 120)
            or _str(catalogue.get("single_offer_price"), 120)
            or _str(catalogue.get("single_pack_offer"), 120)
        )
        evidence.update({
            "product_id": _str(catalogue.get("product_id") or catalogue.get("sku") or catalogue.get("name"), 180),
            "available": _str(catalogue.get("available") or catalogue.get("stock"), 120),
            "price": fallback_price,
            "currency": _str(catalogue.get("currency", "MAD"), 16),
        })
        media_ids = catalogue.get("relevant_media_ids") or catalogue.get("image_asset_ids")
        if media_ids:
            evidence["relevant_media_ids"] = _str(media_ids, 500)

    city_is_current = bool(resolved_city) and city_source in CURRENT_CITY_SOURCES

    # Deterministic offer facts pass through verbatim (backend owns truth).
    # Built BEFORE shipping so the 2-pack free-delivery override applies.
    # Never compute totals here: no 35 added on top of a free-delivery offer.
    offer_keys = ("offer_id", "offer_name", "offer_total_price", "offer_free_delivery",
                  "offer_free_bracelets", "offer_min_quantity", "two_pack_total", "free_delivery",
                  "single_offer_price", "single_offer_bracelets", "single_pack_offer")
    offer: dict[str, str] = {}
    for key in offer_keys:
        value = _str(catalogue.get(key), 120)
        if value:
            offer[key] = value
    if offer:
        # Resolve the applicable offer ONLY from stated facts: when the
        # requested quantity meets an offer's minimum, surface it as one
        # resolved block so a "2 packs + shipping" question gets
        # 179 + free delivery, never 179+35.
        try:
            min_qty = int(str(catalogue.get("offer_min_quantity") or "0").strip() or 0)
        except ValueError:
            min_qty = 0
        applies = bool(min_qty) and requested_quantity >= min_qty
        evidence["offer"] = {
            **offer,
            "applies_to_this_turn": applies,
            "overrides_normal_shipping": applies and str(
                catalogue.get("offer_free_delivery") or catalogue.get("free_delivery") or ""
            ).strip().lower() in {"true", "1", "yes"},
        }
    offer_applies = bool(isinstance(evidence.get("offer"), dict)
                         and evidence["offer"].get("applies_to_this_turn")
                         and evidence["offer"].get("overrides_normal_shipping"))
    if offer_applies:
        # The 2-pack offer overrides normal city shipping: delivery is FREE
        # on this turn. Luna must quote 179 + free delivery, never 179 + 35.
        evidence["offer_shipping"] = "free"

    shipping_rows = turso_data.get("shipping_rows") if isinstance(turso_data.get("shipping_rows"), list) else []
    ship = None
    if city_is_current:
        for candidate in shipping_rows:
            if str(candidate.get("city") or "").strip().lower() == resolved_city.strip().lower():
                ship = candidate
                break
    # No first-row fallback: an unmatched city yields no city row, never a
    # neighbouring city's fee presented under the wrong name.
    # Flat keys only: the nested shipping{} object duplicated the same facts.
    if ship:
        ship_price = _str(ship.get("price"), 120)
        evidence.update({
            "shipping_city": _str(ship.get("city"), 120),
            "shipping_city_source": city_source,
            # Currency adjacent to the amount: the monetary guard only
            # grounds "35 MAD", never a bare "35" followed by other keys.
            # (A missing currency here turned correct shipping replies into
            # HTTP 500 unsafe_reply:ungrounded_price.)
            "delivery_price": ship_price,
            "delivery_currency": _str(
                ship.get("currency") or catalogue.get("delivery_currency", "MAD"), 16),
            "delivery_conditions": _str(ship.get("conditions"), 300),
        })
        if offer_applies:
            evidence["delivery_price"] = "0"
    elif _str(catalogue.get("delivery_price")):
        if city_is_current:
            evidence.update({
                "shipping_city": _str(resolved_city, 120),
                "shipping_city_source": city_source,
                "delivery_price": _str(catalogue.get("delivery_price"), 120),
                "delivery_currency": _str(catalogue.get("delivery_currency", "MAD"), 16),
            })
            if offer_applies:
                evidence["delivery_price"] = "0"
        else:
            # Generic fee, explicitly city-free: answer "35 MAD", never a city.
            # Unless an applying offer makes delivery free this turn.
            generic_price = _str(catalogue.get("delivery_price"), 120)
            if offer_applies:
                generic_price = "0"
            evidence.update({
                "delivery_price": generic_price,
                "delivery_currency": _str(catalogue.get("delivery_currency", "MAD"), 16),
                # Canonical Adam Luxe scope: 35 MAD everywhere in Morocco.
                "shipping_scope": "all_morocco",
            })
    if city_source and city_source not in CURRENT_CITY_SOURCES and city_source != "none":
        evidence["dropped_stale_city_source"] = city_source

    # SHIPPING COMPLETENESS (Adam Luxe truth: 35 MAD, all_morocco, no free
    # shipping — except an applying free-delivery offer): for any
    # shipping-related intent the evidence MUST contain the current shipping
    # value before Luna. Never leave shipping intent + empty price alongside
    # historical assistant claims. When the current fee cannot be loaded,
    # mark it UNKNOWN explicitly — never infer it from assistant history.
    if is_shipping_intent(message_text) and not _str(evidence.get("delivery_price")):
        if offer_applies:
            evidence.update({
                "delivery_price": "0",
                "delivery_currency": _str(catalogue.get("delivery_currency", "MAD"), 16),
                "shipping_scope": "all_morocco",
                "free_shipping": "true",
            })
        else:
            evidence.update({
                "delivery_price": "UNKNOWN",
                "delivery_currency": _str(catalogue.get("delivery_currency", "MAD"), 16),
                "shipping_scope": "all_morocco",
                "free_shipping": "false",
                "shipping_unresolved": "true",
            })
    # PRICE COMPLETENESS: on a price intent with no exact price evidence,
    # mark it explicitly so Luna asks/acknowledges instead of inventing.
    # Normal price turns with a resolved product must carry the exact price
    # (fixed by the fallback chain above); this marker is only the honest
    # empty state — never a guessed amount.
    if is_price_intent(message_text) and not _str(evidence.get("price")):
        if not _str((evidence.get("offer") or {}).get("offer_total_price") if isinstance(evidence.get("offer"), dict) else ""):
            evidence["price_unresolved"] = "true"
    # Normalise the canonical shipping scope vocabulary: the generic fee is
    # Morocco-wide. Old per-city exceptions (e.g. Marrakech free) are stale
    # and must never surface; a city fee enters evidence ONLY via the
    # city-current path above.
    if _str(evidence.get("delivery_price")) and not _str(evidence.get("shipping_scope")):
        if not _str(evidence.get("shipping_city")):
            evidence["shipping_scope"] = "all_morocco"
    if _str(evidence.get("delivery_price")) and evidence.get("delivery_price") != "UNKNOWN":
        if "free_shipping" not in evidence and not offer_applies:
            try:
                _fee_num = float(str(evidence.get("delivery_price")).split()[0].replace(",", "."))
                evidence["free_shipping"] = "true" if _fee_num == 0 else "false"
            except (ValueError, IndexError):
                pass

    # Pre-computed grounded total: when product price + delivery fee are both
    # known in the same currency, their SUM is grounded evidence too. Luna
    # answering "total with delivery" must not trip the ungrounded-price
    # guard for doing correct arithmetic (99 + 35 = 134). On applying-offer
    # turns the offer total itself is the grounded total (never +fee).
    try:
        from decimal import Decimal as _Decimal
        _offer_total = ""
        if offer_applies:
            _offer_total = str(evidence.get("offer", {}).get("offer_total_price") or "").strip()
        if _offer_total:
            evidence["total_with_delivery"] = f"{_offer_total} MAD"
        else:
            _price_raw = str(evidence.get("price") or "").split()[0].replace(",", ".")
            _ship_raw = str(evidence.get("delivery_price") or "").split()[0].replace(",", ".")
            _cur = str(evidence.get("currency") or "").strip().upper()
            if _price_raw and _ship_raw and _cur in {"MAD", "DH", "درهم", "دراهم", ""}:
                _total = _Decimal(_price_raw) + _Decimal(_ship_raw)
                _total_str = format(_total.normalize(), "f") if _total == _total.to_integral_value() else format(_total, "f")
                # Stored WITH currency so the monetary guard recognizes it.
                _cur_out = "MAD" if _cur in {"MAD", "DH", ""} else _cur
                evidence["total_with_delivery"] = f"{_total_str} {_cur_out}".strip()
    except Exception:
        pass

    # Drop empty/placeholder values: they cost tokens and invite invention.
    # Structural keys (resolver, offer predicates) always stay.
    for key in [k for k, v in evidence.items()
                if k not in {"resolver", "offer", "offer_shipping"}
                and (v is None or (isinstance(v, str) and v.strip() in {"", "{}", "[]", "none", "null", "unknown"}))]:
        del evidence[key]
    color_block = evidence.get("color")
    if isinstance(color_block, dict):
        for key in [k for k, v in color_block.items()
                    if isinstance(v, str) and v.strip() in {"", "{}", "none"}]:
            del color_block[key]

    fragment = "EVIDENCE (exact facts for this turn):\n" + json.dumps(
        evidence, ensure_ascii=False, separators=(",", ":")
    )
    return evidence, fragment[:1600]
