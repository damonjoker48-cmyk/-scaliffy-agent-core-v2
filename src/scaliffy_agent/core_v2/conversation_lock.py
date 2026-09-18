"""Distributed conversation lock (V2).

Namespace: conversation_lock:{store_id}:{channel}:{customer_id}

- Only messages of THE SAME conversation are serialized.
- Different customers NEVER block each other (no global lock).
- Durable across processes via the V2 SQLite/Turso store (atomic claim
  with lease + expiry), NOT asyncio.Lock / dict / process-local alone.
  An in-process threading lock per key is only a fast-path to reduce
  store contention; correctness comes from the durable claim.

Usage:
    with conversation_lock(store_id, channel, customer_id, timeout=15):
        ...
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from . import durable as _durable
from .execution import lock_namespace

_process_locks: dict[str, threading.Lock] = {}
_process_locks_guard = threading.Lock()


def _local_lock(key: str) -> threading.Lock:
    with _process_locks_guard:
        lock = _process_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _process_locks[key] = lock
        return lock


@contextmanager
def conversation_lock(
    *, store_id: str, channel: str, customer_id: str,
    timeout_seconds: float = 15.0, lease_seconds: float = 20.0,
) -> Iterator[str]:
    key = lock_namespace(store_id=store_id, channel=channel, customer_id=customer_id)
    owner = f"{uuid.uuid4().hex}"
    deadline = time.time() + max(0.5, float(timeout_seconds or 15.0))
    local = _local_lock(key)
    # Fast path in-process (non-blocking); durable claim is authoritative.
    local_acquired = local.acquire(blocking=False)
    try:
        while True:
            if _durable.lock_acquire(lock_key=key, owner=owner, lease_seconds=lease_seconds):
                try:
                    yield key
                finally:
                    _durable.lock_release(lock_key=key, owner=owner)
                return
            if time.time() >= deadline:
                raise TimeoutError(f"conversation_lock_timeout:{key}")
            time.sleep(0.02)
    finally:
        if local_acquired:
            try:
                local.release()
            except Exception:
                pass
