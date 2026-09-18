"""Tenant-scoped hot cache. Every key includes store_id. No global merchant keys."""
from __future__ import annotations

import os
import threading
import time


_lock = threading.Lock()
_memory: dict[str, tuple[float, object]] = {}


def _now() -> float:
    return time.time()


def brain_key(store_id: str, version: str) -> str:
    return f"brain:{store_id}:v{version}"


def state_key(store_id: str, channel: str, customer_id: str) -> str:
    return f"state:{store_id}:{channel}:{customer_id}"


def product_key(store_id: str, product_id: str) -> str:
    return f"product:{store_id}:{product_id}"


def alias_key(store_id: str, normalized_alias: str) -> str:
    return f"alias:{store_id}:{normalized_alias}"


def cache_get(key: str):
    with _lock:
        item = _memory.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at and expires_at < _now():
            _memory.pop(key, None)
            return None
        return value


def cache_set(key: str, value: object, ttl_seconds: int = 600) -> None:
    # Never allow a tenant key without a store_id segment.
    if ":" not in key or len(key.split(":")) < 2:
        raise ValueError("cache_key_must_be_tenant_scoped")
    expires_at = _now() + max(1, int(ttl_seconds)) if ttl_seconds > 0 else 0.0
    with _lock:
        _memory[key] = (expires_at, value)
        # Bound memory on serverless warm instances.
        if len(_memory) > 2000:
            oldest = sorted(_memory.items(), key=lambda kv: kv[1][0] or 0.0)[:200]
            for k, _ in oldest:
                _memory.pop(k, None)


def cache_ttl_seconds(name: str, default: int) -> int:
    try:
        return max(30, int(os.environ.get(name) or str(default)))
    except ValueError:
        return default
