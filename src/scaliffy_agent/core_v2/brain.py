"""Merchant Brain V2 — compact, stable, cached by store_id + brain_version.

Contains ONLY stable merchant identity/behavior (tone, language
preferences, general store behavior, stable operational rules).
NEVER: entire catalogue, order history, current conversation, dynamic
evidence, giant media inventory. Customer-specific data never enters
the cache: key = store_id + brain_version.
"""
from __future__ import annotations

import hashlib
import threading

_cache: dict[str, dict] = {}
_guard = threading.Lock()


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def brain_cache_key(*, store_id: str, brain_version: str) -> str:
    return f"brain_v2:{str(store_id).strip()}:v{str(brain_version).strip()}"


def load_brain(*, store_id: str, brain: dict) -> dict:
    content = str((brain or {}).get("content") or "")
    if not content.strip():
        raise ValueError("v2_brain_missing_content")
    version = str((brain or {}).get("version") or "").strip()
    if not version:
        version = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    key = brain_cache_key(store_id=store_id, brain_version=version)
    with _guard:
        cached = _cache.get(key)
        if cached and cached.get("version") == version and cached.get("store_id") == str(store_id):
            return dict(cached)
    compact = content.strip()[:4400]
    record = {
        "store_id": str(store_id),
        "version": version,
        "content": compact,
        "estimated_tokens": _estimate_tokens(compact),
    }
    with _guard:
        _cache[key] = dict(record)
        if len(_cache) > 200:
            for old in list(_cache.keys())[:50]:
                _cache.pop(old, None)
    return dict(record)
