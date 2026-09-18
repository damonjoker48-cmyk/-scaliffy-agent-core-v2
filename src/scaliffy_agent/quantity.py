"""Requested quantity detection (deterministic, no LLM).

Understands explicit quantity mentions in Darija (Arabic/Latin), French and
digits: "joj/jouj/جوج/2", "wahd/واحد/1", "tlata/3", "reb3a/4", ...
Returns 0 when no explicit quantity is stated (never guessed).
"""
from __future__ import annotations

import re
import unicodedata


_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or "").casefold())
    ascii_digits = "".join(
        c for c in nfkd if not unicodedata.combining(c)).translate(_EASTERN_DIGITS)
    return " ".join(ascii_digits.split())


_WORD_NUMBERS = {
    1: ["wahd", "wahed", "wa7d", "wa7ed", "wahda", "واحد", "وحدة", "واحدة", "un", "une"],
    2: ["joj", "jouj", "jooj", "juj", "zouj", "zuj", "zouj", "جوج", "زوج", "deux", "2 packs", "two"],
    3: ["tlata", "tlat", "tlta", "ثلاثة", "تلاتة", "trois", "three"],
    4: ["reb3a", "reb3aa", "rb3a", "أربعة", "ربعة", "quatre", "four"],
    5: ["khmsa", "خمسة", "cinq", "five"],
}


def requested_quantity(text: str) -> int:
    """Return the explicitly requested quantity, else 0."""
    value = str(text or "")
    norm = _norm(value)
    # Letter boundaries: Arabic punctuation (؟ ، ؛) must not glue words.
    before, after = r"(?<![^\W\d_])", r"(?![^\W\d_])"
    digit = re.search(before + r"([1-9][0-9]?)" + after, norm)
    if digit:
        try:
            return max(0, min(99, int(digit.group(1))))
        except ValueError:
            pass
    for number in sorted(_WORD_NUMBERS, reverse=True):
        for word in _WORD_NUMBERS[number]:
            if re.search(before + re.escape(word) + after, norm):
                return number
    return 0
