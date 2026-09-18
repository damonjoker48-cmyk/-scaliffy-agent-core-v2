"""Token observability V2 — per-turn, secret-free."""
from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger("scaliffy.core_v2")


def estimate_tokens(text: str) -> int:
    return max(0, len(str(text or "")) // 4)


def log_turn(
    *,
    execution_id: str,
    store_id: str,
    channel: str,
    customer_id: str,
    agent_input: dict | None = None,
    output_text: str = "",
    context_build_ms: int = 0,
    luna_ms: int = 0,
    total_ms: int = 0,
    luna_call_count: int = 1,
    outbound_count: int = 1,
    extra: dict | None = None,
) -> dict:
    est = (agent_input or {}).get("token_estimate", {}) if isinstance(agent_input, dict) else {}
    record = {
        "event": "CORE_V2_TURN",
        "execution_id": execution_id,
        "store_id": store_id,
        "channel": channel,
        "customer_id": customer_id,
        "brain_tokens": int(est.get("brain_tokens") or 0),
        "state_tokens": int(est.get("state_tokens") or 0),
        "history_tokens": int(est.get("history_tokens") or 0),
        "evidence_tokens": int(est.get("evidence_tokens") or 0),
        "system_tokens": int(est.get("system_tokens") or 0),
        "total_input_tokens": int(est.get("total_input_estimate") or 0),
        "output_tokens": estimate_tokens(output_text),
        "context_build_ms": int(context_build_ms or 0),
        "luna_ms": int(luna_ms or 0),
        "total_ms": int(total_ms or 0),
        "luna_call_count": int(luna_call_count or 0),
        "outbound_count": int(outbound_count or 0),
        "ts": int(time.time()),
    }
    if isinstance(extra, dict):
        for key in ("resolver", "reason", "order_action", "agent_core_version"):
            if key in extra:
                record[key] = extra[key]
    try:
        logger.info("%s", json.dumps(record, ensure_ascii=False, sort_keys=True))
    except Exception:
        pass
    return record
