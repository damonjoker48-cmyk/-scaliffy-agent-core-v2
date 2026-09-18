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
import logging
import os
import sqlite3
import tempfile
import threading
import time

logger = logging.getLogger("scaliffy.core_v2")

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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            _t_reset(_tc, store_id=store_id)
            return
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=reset_store")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_session_load(_tc, store_id=store_id, channel=channel, customer_id=customer_id)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=session_load")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            _t_session_save(_tc, store_id=store_id, channel=channel, customer_id=customer_id, state=state)
            return
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=session_save")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_memory_recent(_tc, store_id=store_id, channel=channel, customer_id=customer_id, limit=limit)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=memory_recent")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_memory_append(_tc, store_id=store_id, channel=channel, customer_id=customer_id, role=str(role), text=cleaned)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=memory_append")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            _t_memory_clear(_tc, store_id=store_id, channel=channel, customer_id=customer_id)
            return
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=memory_clear")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_outbound_claim(_tc, outbound_key=outbound_key)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=outbound_claim")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_outbound_status(_tc, outbound_key=outbound_key)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=outbound_status")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            _t_outbound_mark_success(_tc, outbound_key=outbound_key, reply=reply)
            return
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=outbound_mark_success")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_exec_get(_tc, execution_id=execution_id)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=exec_get")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            _t_exec_put(_tc, execution_id=execution_id, result=result)
            return
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=exec_put")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            return _t_lock_acquire(_tc, lock_key=lock_key, owner=owner, lease_seconds=lease_seconds)
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=lock_acquire")
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
    _tc = _turso_creds()
    if _tc[0]:
        try:
            _t_lock_release(_tc, lock_key=lock_key, owner=owner)
            return
        except Exception:
            logger.warning("v2_turso_fallback=sqlite op=lock_release")
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


# ------------------------------------------------------- turso backend ---
# Active automatically when TURSO_DATABASE_URL + TURSO_AUTH_TOKEN are set
# (Vercel: shared durable state across instances/lambdas — never /tmp).
# Otherwise the local SQLite path above is used (dev/tests/single host).
# Same tables, same function signatures; multi-statement batches run as ONE
# Hrana pipeline round trip. Any Turso failure falls back to SQLite
# (fail-open: today's single-instance behavior, never a 500).
_TURSO_DDL_DONE: set[str] = set()

_DDL_STATEMENTS: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS v2_session (store_id TEXT NOT NULL, channel TEXT NOT NULL, customer_id TEXT NOT NULL, state_json TEXT NOT NULL DEFAULT '{}', updated_at REAL NOT NULL DEFAULT 0, PRIMARY KEY (store_id, channel, customer_id))",
    "CREATE TABLE IF NOT EXISTS v2_memory (store_id TEXT NOT NULL, channel TEXT NOT NULL, customer_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, ts REAL NOT NULL DEFAULT 0, PRIMARY KEY (store_id, channel, customer_id, seq))",
    "CREATE INDEX IF NOT EXISTS idx_v2_memory_lookup ON v2_memory (store_id, channel, customer_id, seq)",
    "CREATE TABLE IF NOT EXISTS v2_outbound (outbound_key TEXT PRIMARY KEY, status TEXT NOT NULL, reply TEXT NOT NULL DEFAULT '', ts REAL NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS v2_exec (execution_id TEXT PRIMARY KEY, result_json TEXT NOT NULL DEFAULT '{}', ts REAL NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS v2_lock (lock_key TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT '', expires_at REAL NOT NULL DEFAULT 0)",
)


def backend_name() -> str:
    """Secret-free backend label for observability (never a value)."""
    url, token = _turso_creds()
    return "turso" if (url and token) else "sqlite"


def _turso_creds() -> tuple[str, str]:
    try:
        url = (os.environ.get("TURSO_DATABASE_URL") or "").strip().rstrip("/")
        token = (os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
    except Exception:
        return "", ""
    if url.startswith("libsql://"):  # Hrana HTTP endpoint uses https
        url = "https://" + url[len("libsql://"):]
    if not url or not token or url == "[SENSITIVE]" or token == "[SENSITIVE]":
        return "", ""
    return url, token


def _t_arg(value: object) -> dict:
    # Hrana JSON protocol: integers ride as strings (i64-safe), floats as
    # JSON numbers (a float-as-string is a 400), null has no value key.
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": float(value)}
    return {"type": "text", "value": str(value)}


def _t_run(url: str, token: str, statements: list[tuple[str, list]]) -> list[dict]:
    """ONE Hrana pipeline round trip. Returns [{rows, affected}]."""
    import urllib.request as _urlreq

    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": [_t_arg(a) for a in args]}}
            for sql, args in statements
        ]
    }
    data = json.dumps(payload).encode("utf-8")

    def _once() -> dict:
        req = _urlreq.Request(
            url + "/v2/pipeline", data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {token}")
        with _urlreq.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    last: Exception | None = None
    body: dict | None = None
    delay = 0.15
    for _ in range(3):
        try:
            body = _once()
            break
        except Exception as exc:  # network only; SQL errors arrive as payload
            last = exc
            if "URLError" not in type(exc).__name__ and not isinstance(exc, OSError):
                raise
            time.sleep(delay)
            delay *= 2
    if body is None:
        raise last or RuntimeError("turso_unreachable")
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"turso_error:{str(body.get('error'))[:200]}")
    out: list[dict] = []
    for item in (body.get("results", []) if isinstance(body, dict) else []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "error" or item.get("error"):
            raise RuntimeError(f"turso_sql_error:{str(item.get('error'))[:200]}")
        res = ((item.get("response") or {}).get("result") or {}) if isinstance(item.get("response"), dict) else {}
        cols = [c.get("name", "") for c in (res.get("cols") or []) if isinstance(c, dict)]
        rows = []
        for row in res.get("rows") or []:
            vals = [(v.get("value") if isinstance(v, dict) else v) for v in row]
            rows.append(dict(zip(cols, vals)))
        try:
            affected = int(res.get("affected_row_count") or 0)
        except (TypeError, ValueError):
            affected = 0
        out.append({"rows": rows, "affected": affected})
    return out


def _t_init(url: str, token: str) -> None:
    if url in _TURSO_DDL_DONE:
        return
    with _lock:
        if url in _TURSO_DDL_DONE:
            return
        _t_run(url, token, [(sql, []) for sql in _DDL_STATEMENTS])
        _TURSO_DDL_DONE.add(url)


def _t_reset(creds: tuple[str, str], *, store_id: str = "") -> None:
    url, token = creds
    _t_init(url, token)
    if store_id:
        _t_run(url, token, [
            ("DELETE FROM v2_session WHERE store_id = ?", [store_id]),
            ("DELETE FROM v2_memory WHERE store_id = ?", [store_id]),
            ("DELETE FROM v2_outbound WHERE outbound_key LIKE ?", [f"{store_id}:%"]),
            ("DELETE FROM v2_exec WHERE execution_id LIKE ?", [f"{store_id}:%"]),
        ])
    else:
        _t_run(url, token, [
            ("DELETE FROM v2_session", []),
            ("DELETE FROM v2_memory", []),
            ("DELETE FROM v2_outbound", []),
            ("DELETE FROM v2_exec", []),
            ("DELETE FROM v2_lock", []),
        ])


def _t_session_load(creds: tuple[str, str], *, store_id: str, channel: str, customer_id: str) -> dict:
    url, token = creds
    _t_init(url, token)
    res = _t_run(url, token, [(
        "SELECT state_json FROM v2_session WHERE store_id=? AND channel=? AND customer_id=?",
        [store_id, channel, customer_id],
    )])
    rows = res[0]["rows"] if res else []
    if not rows:
        return {}
    try:
        data = json.loads(rows[0].get("state_json") or "{}")
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError, AttributeError):
        return {}


def _t_session_save(creds: tuple[str, str], *, store_id: str, channel: str, customer_id: str, state: dict) -> None:
    url, token = creds
    _t_init(url, token)
    _t_run(url, token, [(
        "INSERT INTO v2_session (store_id, channel, customer_id, state_json, updated_at) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT (store_id, channel, customer_id) DO UPDATE SET "
        "state_json=excluded.state_json, updated_at=excluded.updated_at",
        [store_id, channel, customer_id, json.dumps(state, ensure_ascii=False), time.time()],
    )])


def _t_memory_recent(creds: tuple[str, str], *, store_id: str, channel: str, customer_id: str, limit: int = 6) -> list[dict]:
    url, token = creds
    _t_init(url, token)
    res = _t_run(url, token, [(
        "SELECT role, text FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=? "
        "ORDER BY seq DESC LIMIT ?",
        [store_id, channel, customer_id, max(1, int(limit or 6))],
    )])
    rows = res[0]["rows"] if res else []
    return [{"role": r.get("role"), "text": r.get("text")} for r in reversed(rows)]


def _t_memory_append(creds: tuple[str, str], *, store_id: str, channel: str, customer_id: str, role: str, text: str) -> int:
    url, token = creds
    _t_init(url, token)
    now = time.time()
    batch = [
        ("INSERT INTO v2_memory (store_id, channel, customer_id, seq, role, text, ts) "
         "VALUES (?,?,?,(SELECT COALESCE(MAX(seq),0)+1 FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=?),?,?,?)",
         [store_id, channel, customer_id, store_id, channel, customer_id, role, text, now]),
        ("DELETE FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=? "
         "AND seq <= (SELECT COALESCE(MAX(seq),0) - 400 FROM v2_memory "
         "WHERE store_id=? AND channel=? AND customer_id=?)",
         [store_id, channel, customer_id, store_id, channel, customer_id]),
        ("SELECT COALESCE(MAX(seq),0) AS seq FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=?",
         [store_id, channel, customer_id]),
    ]
    last: Exception | None = None
    for _ in range(3):  # concurrent writers may collide on seq; recompute
        try:
            res = _t_run(url, token, batch)
            rows = res[2]["rows"] if len(res) > 2 else []
            try:
                return int((rows[0].get("seq") if rows else 0) or 0)
            except (TypeError, ValueError, AttributeError):
                return 0
        except Exception as exc:
            last = exc
            if "turso_sql_error" not in str(exc):
                raise
            time.sleep(0.05)
    raise last or RuntimeError("turso_memory_append_failed")


def _t_memory_clear(creds: tuple[str, str], *, store_id: str, channel: str, customer_id: str) -> None:
    url, token = creds
    _t_init(url, token)
    _t_run(url, token, [(
        "DELETE FROM v2_memory WHERE store_id=? AND channel=? AND customer_id=?",
        [store_id, channel, customer_id],
    )])


def _t_outbound_claim(creds: tuple[str, str], *, outbound_key: str) -> bool:
    url, token = creds
    _t_init(url, token)
    res = _t_run(url, token, [(
        "INSERT INTO v2_outbound (outbound_key, status, reply, ts) VALUES (?,?,?,?) "
        "ON CONFLICT (outbound_key) DO NOTHING",
        [outbound_key, "claimed", "", time.time()],
    )])
    return bool(res and res[0]["affected"] == 1)


def _t_outbound_status(creds: tuple[str, str], *, outbound_key: str) -> dict | None:
    url, token = creds
    _t_init(url, token)
    res = _t_run(url, token, [(
        "SELECT status, reply FROM v2_outbound WHERE outbound_key=?", [outbound_key],
    )])
    rows = res[0]["rows"] if res else []
    if not rows:
        return None
    return {"status": rows[0].get("status"), "reply": rows[0].get("reply")}


def _t_outbound_mark_success(creds: tuple[str, str], *, outbound_key: str, reply: str) -> None:
    url, token = creds
    _t_init(url, token)
    now = time.time()
    _t_run(url, token, [(
        "INSERT INTO v2_outbound (outbound_key, status, reply, ts) VALUES (?,?,?,?) "
        "ON CONFLICT (outbound_key) DO UPDATE SET status='success', reply=excluded.reply, ts=excluded.ts",
        [outbound_key, "success", str(reply or "")[:4000], now],
    )])


def _t_exec_get(creds: tuple[str, str], *, execution_id: str) -> dict | None:
    url, token = creds
    _t_init(url, token)
    res = _t_run(url, token, [(
        "SELECT result_json FROM v2_exec WHERE execution_id=?", [execution_id],
    )])
    rows = res[0]["rows"] if res else []
    if not rows:
        return None
    try:
        data = json.loads(rows[0].get("result_json") or "{}")
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError, AttributeError):
        return None


def _t_exec_put(creds: tuple[str, str], *, execution_id: str, result: dict) -> None:
    url, token = creds
    _t_init(url, token)
    _t_run(url, token, [(
        "INSERT INTO v2_exec (execution_id, result_json, ts) VALUES (?,?,?) "
        "ON CONFLICT (execution_id) DO UPDATE SET result_json=excluded.result_json, ts=excluded.ts",
        [execution_id, json.dumps(result, ensure_ascii=False), time.time()],
    )])


def _t_lock_acquire(creds: tuple[str, str], *, lock_key: str, owner: str, lease_seconds: float = 20.0) -> bool:
    url, token = creds
    _t_init(url, token)
    now = time.time()
    expiry = now + max(1.0, float(lease_seconds or 20.0))
    res = _t_run(url, token, [(
        "INSERT INTO v2_lock (lock_key, owner, expires_at) VALUES (?,?,?) "
        "ON CONFLICT (lock_key) DO UPDATE SET owner=excluded.owner, expires_at=excluded.expires_at "
        "WHERE v2_lock.expires_at <= ?",
        [lock_key, owner, expiry, now],
    )])
    return bool(res and res[0]["affected"] == 1)


def _t_lock_release(creds: tuple[str, str], *, lock_key: str, owner: str) -> None:
    url, token = creds
    _t_init(url, token)
    _t_run(url, token, [(
        "DELETE FROM v2_lock WHERE lock_key=? AND owner=?", [lock_key, owner],
    )])
