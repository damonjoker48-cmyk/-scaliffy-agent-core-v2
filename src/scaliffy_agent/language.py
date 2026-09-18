from __future__ import annotations

import re

from .types import ReplyScript

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")

def detect_reply_script(text: str) -> ReplyScript:
    """Keep Darija in the script selected by the customer."""
    return ReplyScript.ARABIC_DARIJA if ARABIC_RE.search(text) else ReplyScript.LATIN_DARIJA


def enforce_reply_script(text: str, script: ReplyScript) -> str:
    """Preserve Luna's directly composed reply; never transliterate it.

    Script selection is a generation constraint.  Rewriting Arabic letters
    one by one after generation destroys Moroccan words, punctuation and
    code-switching (and was the source of outputs such as ``m0?adatha``).
    A second rewriting model would also violate the one-Luna-call contract.
    """
    del script
    return str(text or "").strip()
