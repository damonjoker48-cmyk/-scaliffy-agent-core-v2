"""Muse Spark 1.3 Contributor compact adapter V2 (§1/§2/§16/§18).

ONE OpenRouter generation per logical customer turn. The model name lives in
ONE central setting (AGENT_MODEL, default below); no business logic depends
on the model name. The backend owns truth; the model owns language.

Provider compatibility (§2): structured json_schema is attempted first; on a
provider rejection the adapter retries once with JSON-in-prompt (recorded in
the report, never silent). The Luna-only `reasoning` extra is not sent.
"""
from __future__ import annotations

import json
import os
import time

MODEL_DEFAULT = "meta/muse-spark-1.3-contributor"

ORDER_ACTIONS = frozenset({
    "none", "start_order", "start_new_order", "resend_order_form",
    "confirm", "refuse", "modify",
})
MEDIA_ACTIONS = frozenset({"none", "send_product_image"})

SYSTEM_KERNEL = (
    "You are a real Moroccan Instagram/WhatsApp seller for this store, "
    "chatting like a human, not an AI assistant. "
    "LANGUAGE: match the customer (Arabic-script Darija by default; French "
    "when they write French; never Arabizi output). Latin digits for prices "
    "(99, 179, 35). "
    "TRUTH: EVIDENCE below is the ONLY commercial truth (price, shipping, "
    "offer, stock, colors, photos). Older assistant messages are context "
    "only: when they contradict EVIDENCE, answer from EVIDENCE silently, "
    "never argue about it. Never invent quality claims, materials, reviews, "
    "urgency, discounts, stock pressure, or photos. "
    "STYLE: 1-3 short lines. Answer the asked question first, then at most "
    "ONE natural next step toward purchase (help pick a color, propose the "
    "relevant option, clarify only what is truly missing). No catalogue "
    "dumps, no repeated order pushes, no paragraphs. "
    "SALES_STAGE guides you: browsing=light help; interested=help choose; "
    "selecting_variant=help with color/options; objection=answer it once, "
    "no pressure; ready_to_order/collecting_order=STOP pitching, complete "
    "briefly. "
    "OUTPUT: one JSON object only: {\"reply\": \"...\", "
    "\"order_action\": \"none|start_order|start_new_order|resend_order_form|"
    "confirm|refuse|modify\", \"order_draft\": {}, "
    "\"media_action\": \"none|send_product_image\"}. "
    "order_action=start_order ONLY when the customer explicitly wants to "
    "order AND the product is identified; then reply is one short "
    "confirmation line. media_action=send_product_image ONLY with a "
    "media_reference copied exactly from EVIDENCE, else none."
)


def requested_model_name() -> str:
    """Single central model setting (env override, Spark default)."""
    try:
        name = (os.environ.get("AGENT_MODEL") or os.environ.get("RAG_LLM_MODEL") or "").strip()
    except Exception:
        name = ""
    return name or MODEL_DEFAULT


def build_compact_user(agent_input: dict) -> str:
    """Render the pure AgentInput payload as compact model text."""
    data = agent_input if isinstance(agent_input, dict) else {}
    lines: list[str] = []
    brain = str(data.get("merchant_brain") or "").strip()
    if brain:
        lines.append("MERCHANT: " + brain[:1200])
    state = str(data.get("session_state") or "").strip()
    if state:
        lines.append("STATE: " + state[:800])
    evidence = str(data.get("evidence") or "").strip()
    if evidence:
        lines.append("EVIDENCE: " + evidence[:1600])
    recent = data.get("recent_messages") or []
    if isinstance(recent, list) and recent:
        lines.append("RECENT (oldest first, context only):")
        for item in recent[-10:]:
            if not isinstance(item, dict):
                continue
            role = "C" if item.get("role") == "customer" else "V"
            lines.append(f"{role}: {str(item.get('content') or '')[:500]}")
    current = str(data.get("current_message") or "").strip()
    lines.append("CURRENT (answer this now): " + current[:2000])
    return "\n".join(lines)


def _extract_json(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for end in range(start, len(text)):
            char = text[end]
            if in_str:
                if esc:
                    esc = False
                elif char == "\\":
                    esc = True
                elif char == '"':
                    in_str = False
            elif char == '"':
                in_str = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:end + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except (ValueError, TypeError):
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _openai_client():  # lazy: avoid import cycles at module load
    from scaliffy_agent.providers import _openrouter

    return _openrouter()


def _response_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "v2_turn",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reply": {"type": "string"},
                    "order_action": {"type": "string"},
                    "order_draft": {"type": "object"},
                    "media_action": {"type": "string"},
                },
                "required": ["reply", "order_action"],
                "additionalProperties": False,
            },
        },
    }


def answer_once(*, agent_input: dict, evidence: dict | None = None,
                timeout_s: float = 60.0) -> tuple[str, dict, dict]:
    """ONE Muse generation. Returns (reply_text, extras, report).

    extras matches the pipeline contract (order_action/order_draft/
    media_action). report carries model/tokens/latency/compat diagnostics.
    Raises RuntimeError on unusable output (pipeline degrades safely).
    """
    _ = evidence  # truth already inside agent_input; kept for signature parity
    model = requested_model_name()
    user_text = build_compact_user(agent_input)
    report: dict = {
        "model_requested": model,
        "model_resolved": "",
        "response_format": "json_schema",
        "reasoning_used": False,
        "degraded": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
    }
    client = _openai_client()
    messages = [
        {"role": "system", "content": SYSTEM_KERNEL},
        {"role": "user", "content": user_text},
    ]
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=500,
            messages=messages,
            response_format=_response_schema(),
            timeout=timeout_s,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "response_format" in message or "json_schema" in message or "schema" in message or "400" in message:
            report["response_format"] = "json_prompt"
            response = client.chat.completions.create(
                model=model,
                temperature=0.3,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": SYSTEM_KERNEL},
                    {"role": "user", "content": user_text + '\nRespond with ONE JSON object only, e.g. {"reply": "...", "order_action": "none"}.'},
                ],
                timeout=timeout_s,
            )
        else:
            raise
    report["latency_ms"] = int((time.perf_counter() - started) * 1000)
    try:
        report["model_resolved"] = str(getattr(response, "model", "") or model)
    except Exception:
        report["model_resolved"] = model
    try:
        choices = getattr(response, "choices", []) or []
        raw = ""
        if choices:
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", "") if msg is not None else ""
            raw = content if isinstance(content, str) else "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
    except Exception:
        raw = ""
    try:
        usage = getattr(response, "usage", None)
        report["input_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
        report["output_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception:
        pass
    parsed = _extract_json(raw)
    if not parsed:
        raise RuntimeError("spark_unparseable_reply")
    reply = str(parsed.get("reply") or "").strip()
    if not reply:
        raise RuntimeError("spark_empty_reply")
    action = str(parsed.get("order_action") or "none").strip().lower()
    draft = parsed.get("order_draft")
    draft = {str(k)[:80]: str(v)[:500] for k, v in draft.items()} if isinstance(draft, dict) else {}
    media = str(parsed.get("media_action") or "none").strip().lower()
    extras = {
        "order_action": action if action in ORDER_ACTIONS else "none",
        "order_draft": draft,
        "media_action": media if media in MEDIA_ACTIONS else "none",
    }
    return reply, extras, report
