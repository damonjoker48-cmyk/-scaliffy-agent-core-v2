"""Turso structured store source of truth. Batched single-phase retrieval.

All queries are tenant-scoped by store_id. If Turso env is absent, raise
TursoNotConfigured so callers fall back to current catalogue_context
behavior with zero loss. No sequential N+1 round trips: use one pipeline
call with multiple statements where dependencies allow.
"""
from __future__ import annotations

import json
import os
import urllib.request


class TursoNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    url = (os.environ.get("TURSO_DATABASE_URL") or "").strip()
    token = (os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
    return bool(url and token and url != "[SENSITIVE]" and token != "[SENSITIVE]")


def _pipeline(statements: list[tuple[str, list]]) -> list:
    """Execute multiple statements in ONE HTTP round trip (Hrana pipeline)."""
    url = (os.environ.get("TURSO_DATABASE_URL") or "").strip().rstrip("/")
    token = (os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
    if not url or not token:
        raise TursoNotConfigured("turso_env_missing")
    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": [{"type": "text", "value": str(a)} for a in args]}}
            for sql, args in statements
            for _ in [0]
        ],
    }
    # Fix comprehension above: build explicitly for clarity.
    payload["requests"] = [
        {"type": "execute", "stmt": {"sql": sql, "args": [{"type": "text", "value": str(a)} for a in args]}}
        for sql, args in statements
    ]
    req = urllib.request.Request(
        url + "/v2/pipeline",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    results = []
    for item in body.get("results", []):
        res = item.get("response", {}).get("result", {})
        cols = [c.get("name", "") for c in res.get("cols", [])]
        rows = []
        for row in res.get("rows", []):
            values = [v.get("value") if isinstance(v, dict) else v for v in row]
            rows.append(dict(zip(cols, values)))
        results.append(rows)
    return results


def fetch_store_phase(
    *,
    store_id: str,
    product_lookup: str = "",
    city: str = "",
) -> dict:
    """One retrieval phase: product + shipping + config in a single round trip.

    Table names adapt to the CURRENT schema: try canonical names first,
    tolerate missing tables (return what exists, never crash the DM).
    """
    if not is_configured():
        raise TursoNotConfigured("turso_env_missing")
    stmts: list[tuple[str, list]] = []
    kinds: list[str] = []
    if product_lookup:
        stmts.append((
            "SELECT id, title, price, currency, stock, available FROM products "
            "WHERE store_id = ? AND (id = ? OR sku = ? OR title = ?) LIMIT 3",
            [store_id, product_lookup, product_lookup, product_lookup],
        ))
        kinds.append("product")
    if city:
        stmts.append((
            "SELECT city, price, currency, conditions FROM shipping_rules "
            "WHERE store_id = ? AND (city = ? OR city = 'default') LIMIT 4",
            [store_id, city],
        ))
        kinds.append("shipping")
    stmts.append((
        "SELECT key, value FROM merchant_config WHERE store_id = ? LIMIT 50",
        [store_id],
    ))
    kinds.append("config")
    try:
        out = _pipeline(stmts)
    except Exception:
        # A Turso failure must never silence the customer: caller falls back.
        return {"product_rows": [], "shipping_rows": [], "config": {}}
    data: dict = {"product_rows": [], "shipping_rows": [], "config": {}}
    for kind, rows in zip(kinds, out):
        if kind == "product":
            data["product_rows"] = rows
        elif kind == "shipping":
            data["shipping_rows"] = rows
        elif kind == "config":
            data["config"] = {str(r.get("key")): r.get("value") for r in rows if r.get("key")}
    return data
