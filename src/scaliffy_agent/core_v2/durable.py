"""Durable V2 store: SQLite-backed, multi-process safe.

Uses the existing durable-database approach (Turso in production when
configured; local SQLite otherwise) — NO new external dependency.
SQLite with WAL + busy_timeout + IMMEDIATE transactions is atomic across
processes on one host, unlike asyncio.Lock / dict / process-local state.

Tables (all tenant-scoped by store_id):
  v2_session   (store_id, channel, customer_id) -> state JSON
  v2_memory    (store_id, channel, customer_id, seq) -> dialogue rows
  v2_outbound  (outbound_key) -> claimed/success + reply
  v2_exec      (execution_id) -> cached result
  v2_lock      (lock_key) -> owner + expiry (distributed conversation lock)

Vercel note: set CORE_V2_DB_PATH to a /tmp path for writable storage.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time

_lock = threading.Lock()
_DB_PATH: str | None = None


def _is_busy(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "locked" in text or "busy" in text


def _retryable(fn, *, attempts: int = 8):
    delay = 0.02
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            last = exc
            if not _is_busy(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(0.4, delay * 1.6)
        except Exception:
            raise
    if last is not None:
        raise last
    raise RuntimeError("durable_retry_exhausted")


def db_path() -> str:
    global _DB_PATH
    if _DB_PATH:
        return _DB_PATH
    env_path = (os.environ.get("CORE_V2_DB_PATH") or "").strip()
    if env_path:
        _DB_PATH = env_path
        return _DB_PATH
    _DB_PATH = os.path.join(tempfile.gettempdir(), "scaliffy_core_v2.db")
    return _DB_PATH


def set_db_path(path: str) -> None:
    global _DB_PATH
    _DB_PATH = str(path or "").strip() or None  # type: ignore[assignment]


def _connect() -> sqlite3.Connection:
    path = db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA busy_timeout=8000;")
    except Exception:
        pass
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS v2_session ("
            "store_id TEXT NOT NULL, channel TEXT NOT NULL, customer_id TEXT NOT NULL, "
            "state_json TEXT NOT NULL DEFAULT '{}', updated_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (store_id, channel, customer_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS v2_memory ("
            "store_id TEXT NOT NULL, channel TEXT NOT NULL, customer_id TEXT NOT NULL, "
            "seq INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, ts REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (store_id, channel, customer_id, seq))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_v2_memory_lookup "
            "ON v2_memory (store_id, channel, customer_id, seq)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS v2_outbound ("
            "outbound_key TEXT PRIMARY KEY, status TEXT NOT NULL, reply TEXT NOT NULL DEFAULT '', "
            "ts REAL NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS v2_exec ("
            "execution_id TEXT PRIMARY KEY, result_json TEXT NOT NULL DEFAULT '{}', "
            "ts REAL NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS v2_lock ("
            "lock_key TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT '', "
            "expires_at REAL NOT NULL DEFAULT 0)"
        )
    finally:
        conn.close()


def reset_store(*, store_id: str = "") -> None:
    """Test helper: clear V2 rows (optionally scoped to one store)."""
    init_db()
    conn = _connect()
    try:
        if store_id:
            for table, col in (
                ("v2_session", "store_id"),
                ("v2_memory", "store_id"),
            ):
                conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (store_id,))
            conn.execute(
                "DELETE FROM v2_outbound WHERE outbound_key LIKE ?", (f"{store_id}:%",)
            )
            conn.execute(
                "DELETE FROM v2_exec WHERE execution_id LIKE ?", (f"{store_id}:%",)
            )
        else:
            for table in ("v2_session", "v2_memory", "v2_outbound", "v2_exec", "v2_lock"):
                conn.execute(f"DELETE FROM {table}")
    finally:
        conn.close()


# ---------------------------------------------------------------- session ---
def session_load(*, store_id: str, channel: str, customer_id: str) -> dict:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT state_json FROM v2_session WHERE store_id=? AND channel=? AND customer_id=?",
            (store_id, channel, customer_id),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {}
    try:
        data = json.loads(row[0] or "{}")
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def session_save(*, store_id: str, channel: str, customer_id: str, state: dict) -> None:
    def _do() -> None:
        init_db()
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO v2_session (store_id, channel, customer_id, state_json, updated_at) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT (store_id, channel, customer_id) DO UPDATE SET "
                "state_json=excluded.state_json, updated_at=excluded.updated_at",
                (store_id, channel, customer_id, json.dumps(state, ensure_ascii=False), time.time()),
            )
        finally:
            conn.close()
    _retryable(_do)


# ---------------------------------------------------------------- memory ----
def memory_recent(
    *, store_id: str, channel: str, customer_id: str, limit: int = 6
) -> list[dict]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, text FROM v2_memory "
            "WHERE store_id=? AND channel=? AND customer_id=? "
            "ORDER BY seq DESC LIMIT ?",
            (store_id, channel, customer_id, max(1, int(limit or 6))),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": r[0], "text": r[1]} for r in reversed(rows)]


def memory_append(
    *, store_id: str, channel: str, customer_id: str, role: str, text: str,
    char_cap: int = 600,
) -> int:
    cleaned = str(text or "").strip()[: max(1, int(char_cap or 600))]
    if not cleaned:
        return -1
    if str(role) not in ("customer", "assistant"):
        raise ValueError("memory_role_invalid")
    def _do() -> int:
        init_db()
        with _lock:
            conn = _connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM v2_memory "
                        "WHERE store_id=? AND channel=? AND customer_id=?",
                        (store_id, channel, customer_id),
                    ).fetchone()
                    nxt = int(row[0] or 0) + 1
                    conn.execute(
                        "INSERT INTO v2_memory (store_id, channel, customer_id, seq, role, text, ts) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (store_id, channel, customer_id, nxt, role, cleaned, time.time()),
                    )
                    # Bound raw rows per conversation (Luna still only sees last 6).
                    conn.execute(
                        "DELETE FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=? "
                        "AND seq <= (SELECT COALESCE(MAX(seq),0) - 400 FROM v2_memory "
                        "WHERE store_id=? AND channel=? AND customer_id=?)",
                        (store_id, channel, customer_id, store_id, channel, customer_id),
                    )
                    conn.execute("COMMIT;")
                    return nxt
                except Exception:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise
            finally:
                conn.close()
    return _retryable(_do)


def memory_clear(*, store_id: str, channel: str, customer_id: str) -> None:
    init_db()
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=?",
            (store_id, channel, customer_id),
        )
    finally:
        conn.close()


# --------------------------------------------------------------- outbound ---
def outbound_claim(*, outbound_key: str) -> bool:
    """Atomic claim. True = caller owns the send; False = already claimed."""
    def _do() -> bool:
        init_db()
        conn = _connect()
        try:
            cur = conn.execute(
                "INSERT INTO v2_outbound (outbound_key, status, reply, ts) VALUES (?,?,?,?) "
                "ON CONFLICT (outbound_key) DO NOTHING",
                (outbound_key, "claimed", "", time.time()),
            )
            return (cur.rowcount or 0) == 1
        finally:
            conn.close()
    return _retryable(_do)


def outbound_status(*, outbound_key: str) -> dict | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT status, reply FROM v2_outbound WHERE outbound_key=?", (outbound_key,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {"status": row[0], "reply": row[1]}


def outbound_mark_success(*, outbound_key: str, reply: str) -> None:
    def _do() -> None:
        init_db()
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO v2_outbound (outbound_key, status, reply, ts) VALUES (?,?,?,?) "
                "ON CONFLICT (outbound_key) DO UPDATE SET status='success', reply=excluded.reply, "
                "ts=excluded.ts",
                (outbound_key, "success", str(reply or "")[:4000], time.time()),
            )
        finally:
            conn.close()
    _retryable(_do)


# -------------------------------------------------------------- executions --
def exec_get(*, execution_id: str) -> dict | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT result_json FROM v2_exec WHERE execution_id=?", (execution_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        data = json.loads(row[0] or "{}")
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def exec_put(*, execution_id: str, result: dict) -> None:
    def _do() -> None:
        init_db()
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO v2_exec (execution_id, result_json, ts) VALUES (?,?,?) "
                "ON CONFLICT (execution_id) DO UPDATE SET result_json=excluded.result_json, "
                "ts=excluded.ts",
                (execution_id, json.dumps(result, ensure_ascii=False), time.time()),
            )
        finally:
            conn.close()
    _retryable(_do)


# ------------------------------------------------------------------ locks ---
def lock_acquire(*, lock_key: str, owner: str, lease_seconds: float = 20.0) -> bool:
    def _do() -> bool:
        now = time.time()
        expiry = now + max(1.0, float(lease_seconds or 20.0))
        init_db()
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute(
                    "SELECT owner, expires_at FROM v2_lock WHERE lock_key=?", (lock_key,)
                ).fetchone()
                if row is None or float(row[1] or 0) <= now:
                    conn.execute(
                        "INSERT INTO v2_lock (lock_key, owner, expires_at) VALUES (?,?,?) "
                        "ON CONFLICT (lock_key) DO UPDATE SET owner=excluded.owner, "
                        "expires_at=excluded.expires_at",
                        (lock_key, owner, expiry),
                    )
                    conn.execute("COMMIT;")
                    return True
                conn.execute("ROLLBACK;")
                return False
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
        finally:
            conn.close()
    return _retryable(_do)


def lock_release(*, lock_key: str, owner: str) -> None:
    def _do() -> None:
        init_db()
        conn = _connect()
        try:
            conn.execute(
                "DELETE FROM v2_lock WHERE lock_key=? AND owner=?", (lock_key, owner)
            )
        finally:
            conn.close()
    try:
        _retryable(_do)
    except Exception:
        pass
