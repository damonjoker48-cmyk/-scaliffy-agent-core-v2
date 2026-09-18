"""Recent raw conversation window: 4-6 useful messages, no full dump, no summary."""
from __future__ import annotations


def _useful(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False
    # Punctuation-only / emoji-only turns carry no continuity value.
    return any(ch.isalpha() or ch.isdigit() for ch in stripped)


def build_recent_window(history: tuple | list, *, limit: int = 6) -> str:
    """Return raw recent turns oldest->newest, capped ~200-600 tokens (~2400 chars)."""
    turns: list[tuple[str, str]] = []
    for turn in list(history or [])[-16:]:
        role = str(getattr(turn, "role", "") or "").strip().lower()
        text = str(getattr(turn, "text", "") or "").strip()
        if role not in ("customer", "assistant", "human"):
            continue
        if not _useful(text):
            continue
        who = "Customer" if role == "customer" else "Store"
        turns.append((who, text[:600]))
    # Dedupe exact consecutive repeats (retries/double sends).
    deduped: list[tuple[str, str]] = []
    for item in turns:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    window = deduped[-max(2, min(6, int(limit or 6))):]
    if not window:
        return "HOT (recent chat): (none)"
    lines = [f"{who}: {text}" for who, text in window]
    text = "HOT (recent chat, verbatim):\n" + "\n".join(lines)
    return text[:2400]
