"""Post-Luna deterministic validation. Backend owns truth; Luna proposes.

Validates state_patch, business action, referenced product/variant IDs, order
mutations, and commercial invariants. Then persist + send reply.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal


_MONEY = re.compile(
    r"(?<![\w\d])([0-9]+(?:[.,][0-9]+)?)\s*[*_]*\s*"
    r"(dhs?|mad|dirhams?|درهم|دراهم|eur|euros?|€|usd|dollars?|\$)(?!\w)", re.I,
)

# Arabic-Indic + Extended Arabic-Indic digits -> Latin. All prices (delivery
# and every other price) must be written with Latin digits 0-9.
_ARABIC_DIGITS = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}
_DIGIT_RE = re.compile("[" + "".join(_ARABIC_DIGITS) + "]")


def latinize_digits(text: str) -> str:
    """Deterministically rewrite Eastern Arabic digits to Latin digits."""
    return _DIGIT_RE.sub(lambda match: _ARABIC_DIGITS[match.group(0)], str(text or ""))


_TRAILING_ZERO_DECIMALS_RE = re.compile(r"(\d+)\.00(?![\d])")


def normalize_price_decimals(text: str) -> str:
    """Merchant writes whole-dirham prices: 99.00 DH -> 99 DH. Real decimals kept."""
    return _TRAILING_ZERO_DECIMALS_RE.sub(r"\1", str(text or ""))


# Adam Luxe merchant vocabulary (current truth): the product is the pack /
# الباك and the free gift is the gourmetta / ݣورميطة. Old internal labels
# (طقم, Luxury Swan Set, إسورة/سوار family, bracelet) must never reach the
# customer. Whole-word only; Luna keeps full phrasing freedom otherwise.
_BRACELET_TERMS = (
    ("إسوارات", "ݣورميطات"),
    ("أساور", "ݣورميطات"),
    ("سوارات", "ݣورميطات"),
    ("إسورة", "ݣورميطة"),
    ("سوار", "ݣورميطة"),
    ("bracelets", "gourmettas"),
    ("bracelet", "gourmetta"),
    ("gourmette", "gourmetta"),
    ("gourmettes", "gourmettas"),
)
# Old product labels -> current customer-facing product words.
_PRODUCT_TERMS = (
    ("Luxury Swan Set", "pack"),
    ("luxury swan set", "pack"),
    ("طقم", "الباك"),
)
# Word-boundary over Unicode LETTERS only ([^\W\d_]): Arabic punctuation
# such as ؟ ، ؛ must count as boundaries, not as word characters.
_LETTER_BOUNDARY_BEFORE = r"(?<![^\W\d_])"
_LETTER_BOUNDARY_AFTER = r"(?![^\W\d_])"
# Arabic clitic prefixes (conjunction/preposition + definite article) attach
# directly to the word (وسوار، بالباك، الطقم): match them and keep them, so
# "الطقم" -> "الباك" (not "الالباك") and "وسوار" -> "وݣورميطة".
_AR_PREFIX = r"(?P<pre>(?:[وفبكل]|ال|لل)?)"


def _word_re(src: str) -> "re.Pattern":
    flags = re.IGNORECASE if re.search(r"[A-Za-z0-9]", src) else 0
    if flags:  # Latin terms never carry Arabic clitics.
        return re.compile(
            _LETTER_BOUNDARY_BEFORE + re.escape(src) + _LETTER_BOUNDARY_AFTER, flags)
    return re.compile(
        _LETTER_BOUNDARY_BEFORE + _AR_PREFIX + re.escape(src) + _LETTER_BOUNDARY_AFTER)


def _keep_prefix(match: "re.Match", dst: str) -> str:
    pre = match.groupdict().get("pre") or ""
    if pre and dst.startswith(pre):
        return dst
    return pre + dst


_BRACELET_RES = [(_word_re(src), dst) for src, dst in _BRACELET_TERMS]
_PRODUCT_RES = [(_word_re(src), dst) for src, dst in _PRODUCT_TERMS]


def merchant_vocabulary(text: str) -> str:
    """Enforce Adam Luxe preferred commercial terms without templating replies."""
    result = str(text or "")
    for pattern, replacement in (*_BRACELET_RES, *_PRODUCT_RES):
        result = pattern.sub(lambda m, dst=replacement: _keep_prefix(m, dst), result)
    return result


# Adam Luxe animal vocabulary: the pack animal is البط (duck/canard),
# never بجعة/swan/cygne (the technical name loses customers). When the
# customer does not name the pack, Luna calls it "le pack ta3 lbat".
_ANIMAL_TERMS = (
    ("البجعة", "البط"),
    ("البجعه", "البط"),
    ("بجعة", "بطة"),
    ("بجعه", "بطة"),
    ("cygnes", "canards"),
    ("cygne", "canard"),
    ("swans", "ducks"),
    ("swan", "duck"),
    ("baj3a", "batta"),
    ("bija3a", "batta"),
    ("bej3a", "batta"),
)
_ANIMAL_RES = [
    (re.compile(_LETTER_BOUNDARY_BEFORE + re.escape(src) + _LETTER_BOUNDARY_AFTER,
                re.IGNORECASE if re.search(r"[A-Za-z0-9]", src) else 0), dst)
    for src, dst in _ANIMAL_TERMS
]


def merchant_animal(text: str) -> str:
    """Enforce the merchant's animal word (duck), never the confusing one."""
    result = str(text or "")
    for pattern, replacement in _ANIMAL_RES:
        result = pattern.sub(replacement, result)
    return result


# Swan emoji next to every duck reference (البط/بطة/batta/canard/duck),
# whatever the wording — unless one is already there. No duplicates.
_DUCK_WORDS = ("البطة", "البط", "بطة", "batta", "canards", "canard", "ducks", "duck")
_EMOJI_AHEAD = r"(?!\s*[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F])"
_DUCK_EMOJI_RES = [
    re.compile(_LETTER_BOUNDARY_BEFORE + re.escape(word) + _LETTER_BOUNDARY_AFTER
               + _EMOJI_AHEAD,
               re.IGNORECASE if re.search(r"[A-Za-z]", word) else 0)
    for word in sorted(_DUCK_WORDS, key=len, reverse=True)
]


def merchant_animal_emoji(text: str) -> str:
    """Append 🦢 beside the duck reference (pack animal). Idempotent."""
    result = str(text or "")
    for pattern in _DUCK_EMOJI_RES:
        result = pattern.sub(lambda match: match.group(0) + " 🦢", result)
    return result


# Gender-neutral addressing: customers are often men buying for wife/mom.
# Address everyone with masculine/neutral 2nd-person forms (nta, بغيت),
# never feminine (nti, بغيتي). ONLY unambiguous past-tense 2nd-person verbs
# + the standalone pronoun are rewritten. Feminine adjectives/nouns are left
# alone: السلسلة/ݣورميطة ARE feminine, and ديالك/بيتي/بنتي must never change.
_FEM_TO_MASC_VERBS = (
    "بغيتي", "كنتي", "وليتي", "درتي", "قلتي", "شفتي", "عرفتي", "شريتي",
    "خديتي", "سولتي", "جيتي", "مشيتي", "دخلتي", "خرجتي", "صيفطتي",
    "جربتي", "حبيتي", "لقيتي", "نسيتي", "فهمتي", "سمعتي", "شكرتي",
    "نتي", "انتي",
)
_FEM_TO_MASC_RES = [
    (re.compile(_LETTER_BOUNDARY_BEFORE + re.escape(src) + _LETTER_BOUNDARY_AFTER),
     ("نت" if src in ("نتي", "انتي") else src[:-1]))
    for src in _FEM_TO_MASC_VERBS
]
# Feminine verb + attached pronoun (شريتيه→شريته, خديتيها→خديتها).
# Only listed verb stems; nouns like بيتي/بنتي never match these stems.
_PRONOUN_SUFFIX = r"(ه|ها|هم|هن|ك|كم|كن|نا|ني|ي)"
_FEM_TO_MASC_SUFFIX_RES = [
    (re.compile(_LETTER_BOUNDARY_BEFORE + re.escape(src[:-1]) + "ي"
                + "(" + _PRONOUN_SUFFIX + ")" + _LETTER_BOUNDARY_AFTER),
     src[:-1] + r"\1")
    for src in _FEM_TO_MASC_VERBS if src not in ("نتي", "انتي")
]


def neutral_masculine(text: str) -> str:
    """Rewrite feminine 2nd-person addressing to neutral masculine."""
    result = str(text or "")
    for pattern, replacement in _FEM_TO_MASC_SUFFIX_RES:
        result = pattern.sub(replacement, result)
    for pattern, replacement in _FEM_TO_MASC_RES:
        result = pattern.sub(replacement, result)
    return result

ORDER_UI_ACTIONS = {"start_order", "start_new_order", "resend_order_form"}
ORDER_ACTIONS = {"none", *ORDER_UI_ACTIONS, "confirm", "refuse", "modify"}


def _money(text: str) -> set:
    out = set()
    for amount, currency in _MONEY.findall(str(text or "")):
        c = currency.casefold()
        code = "MAD" if c in {"dh", "dhs", "mad", "dirham", "dirhams", "درهم", "دراهم"} else (
            "EUR" if c in {"eur", "euro", "euros", "€"} else "USD")
        try:
            out.add((Decimal(amount.replace(",", ".")), code))
        except Exception:
            pass
    return out


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or "").casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in nfkd if not unicodedata.combining(c))))


def validate_action(
    *,
    reply_text: str,
    order_action: str,
    order_draft: dict,
    media_action: str,
    evidence: dict,
    resolver_status: str,
    resolver_product_id: str,
) -> tuple[str, dict, str]:
    """Return (order_action, order_draft, media_action), raising on violation."""
    action = str(order_action or "none").lower()
    if action not in ORDER_ACTIONS:
        action = "none"
    draft = dict(order_draft) if isinstance(order_draft, dict) else {}
    media = str(media_action or "none").lower()
    if media not in {"none", "send_product_image"}:
        media = "none"

    # Color gate for the order paper: opening checkout without a grounded
    # color on a multi-color product would prepare a wrong paper. Downgrade
    # so the color question turn completes first; resend/confirm/modify on
    # existing papers are unaffected. Luna's wording stays entirely hers.
    if isinstance(evidence, dict) and action in {"start_order", "start_new_order"}:
        color_block = evidence.get("color") if isinstance(evidence.get("color"), dict) else {}
        available = color_block.get("available_colors")
        multi_colors = isinstance(available, list) and len(available) > 1
        if multi_colors and color_block.get("status") == "missing":
            action = "none"
            draft = {}
            evidence["color_checkout_deferred"] = True

    # Referenced product must match resolved evidence; never silently swap.
    if resolver_status == "NOT_FOUND" and action in ORDER_UI_ACTIONS:
        # No exact product: do not start checkout on a guessed product.
        # Keep the reply (Luna asks for the name) but drop the side effect.
        action = "none"
        draft = {}
    if resolver_product_id and isinstance(evidence, dict):
        ev_product = str(evidence.get("product_id") or "")
        draft_product = str(draft.get("product_id") or draft.get("product") or "")
        if draft_product and ev_product and _norm(draft_product) != _norm(ev_product):
            # Draft disagrees with exact evidence: trust evidence, drop draft id.
            draft = {k: v for k, v in draft.items() if k not in ("product_id", "product")}

    # Offer invariant: a free-delivery offer total (e.g. two-pack 179) must
    # never gain a shipping addition. If the draft total already equals the
    # evidence offer total, drop any shipping keys from the draft.
    if isinstance(evidence, dict):
        offer = evidence.get("offer") if isinstance(evidence.get("offer"), dict) else {}
        offer_totals = {
            str(evidence.get(key) or "").strip()
            for key in ("offer_total_price", "two_pack_total")
        } | {
            str(offer.get(key) or "").strip()
            for key in ("offer_total_price", "two_pack_total")
        } - {""}
        free_delivery = str(
            evidence.get("offer_free_delivery") or evidence.get("free_delivery")
            or offer.get("offer_free_delivery") or offer.get("free_delivery") or ""
        ).strip().lower() in {"true", "1", "yes"}
        draft_total = str(
            draft.get("total") or draft.get("total_price") or ""
        ).strip()
        if free_delivery and draft_total and draft_total in offer_totals:
            draft = {
                key: value for key, value in draft.items()
                if key not in ("shipping_price", "delivery_price", "shipping_fee")
            }

    # Commercial invariant: monetary claims must appear in evidence.
    if isinstance(evidence, dict):
        def _flatten(value) -> list[str]:
            if isinstance(value, dict):
                parts: list[str] = []
                for item in value.values():
                    parts.extend(_flatten(item))
                return parts
            if isinstance(value, (list, tuple)):
                parts = []
                for item in value:
                    parts.extend(_flatten(item))
                return parts
            if isinstance(value, (str, int, float)):
                return [str(value)]
            return []

        evidence_text = " ".join(_flatten(evidence))
        claimed = _money(reply_text)
        grounded = _money(evidence_text)
        # Only enforce when evidence actually contains a price (else adapter
        # catalogue_context is the fallback evidence and may be partial).
        # UNKNOWN markers carry no amount and never ground a claim.
        if grounded and (claimed - grounded):
            raise RuntimeError("unsafe_reply:ungrounded_price")

    # Never block checkout on missing contact fields: the order form collects
    # name/phone/address/city. So no validation error for missing fields here.
    return action, draft, media
