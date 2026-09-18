"""SCALIFFY CORE V2 — isolated test engine for store 625374849.

Target model (n8n-inspired):
Channel -> Normalize -> Execution ID -> Distributed lock ->
Load Chat Memory (max 6) -> Load SessionState -> Resolve evidence ->
BuildAgentInput (pure) -> ONE Luna -> Validate -> Persist state ->
Append memory -> Idempotent outbound.

Production store 166510782 (and every other merchant) NEVER enters this
package. The feature gate lives in `agent.py` and delegates ONLY when
store_id == TEST_STORE_ID.
"""
from __future__ import annotations
