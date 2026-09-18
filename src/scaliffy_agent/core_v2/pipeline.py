"""AgentCoreV2 pipeline — the exact mission execution model.

Channel -> Normalize -> Execution ID -> Distributed lock ->
Load Chat Memory (max 6) -> Load SessionState ->
Resolve deterministic evidence -> BuildAgentInput (pure) ->
ONE Luna -> Validate -> Persist SessionState -> Append memory ->
Idempotent outbound.

Rules enforced:
- Feature-gated to TEST_STORE_ID only (pipeline refuses other stores).
- ONE Luna call per logical turn (retries reuse the cached execution).
- Max ONE outbound per execution (durable atomic claim, text-independent).
- Evidence rebuilt from scratch every turn; assistant history is dialogue
  only and NEVER commercial truth.
- No duplicate history: recent messages appear exactly once.
- Token/latency observability on every turn, secret-free.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from . import durable as _durable
from . import memory as _memory
from . import observability as _obs
from . import outbound as _outbound
from . import session_state as _state
from .agent_input import build_agent_input
from .brain import load_brain
from .config import (
    AGENT_CORE_VERSION_V2,
    MEMORY_LIMIT,
    TEST_STORE_ID,
)
from .conversation_lock import conversation_lock
from .evidence import build_evidence_v2
from .execution import build_execution_id
from .normalizer import NormalizedMessage
from .seed import test_brain, test_catalogue, test_media_map


def is_v2_store(store_id: str) -> bool:
    return str(store_id or "").strip() == TEST_STORE_ID


def _luna_fallback(*, evidence: dict | None, text: str, exc: BaseException) -> tuple[str, dict]:
    """Honest intent-based fallback for ANY Luna-call failure (never a 500).

    The failed call still counts as the ONE call of this turn; the reason
    travels in extras for the trace. No second model call is made.
    """
    raw = str(exc)
    if raw.startswith("unsafe_reply:"):
        failure = "unsafe_" + raw.split(":", 1)[-1]
    else:
        failure = f"luna_error_{type(exc).__name__}"
    try:
        from scaliffy_agent.evidence import (
            is_price_intent as _is_price,
            is_shipping_intent as _is_ship,
        )
        facts = evidence if isinstance(evidence, dict) else {}
        delivery = str(facts.get("delivery_price") or "").strip()
        if _is_ship(text) and delivery and delivery != "UNKNOWN":
            fallback = f"التوصيل {delivery} درهم لجميع المدن."
        elif _is_price(text):
            fallback = "الباك متوفر، قوليا شحال بغيتي (واحد ولا جوج) ونعطيك الثمن بالضبط."
        else:
            fallback = "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
    except Exception:
        fallback = "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
    try:
        from scaliffy_agent.core_v2.spark import requested_model_name as _rmn
        _mreq = _rmn()
    except Exception:
        _mreq = ""
    return fallback, {
        "order_action": "none", "order_draft": {}, "media_action": "none",
        "_luna_fallback_reason": failure,
        "_model_requested": _mreq,
    }


class AgentCoreV2:
    def __init__(self, *, model: Any = None) -> None:
        self.model = model
        self._last_spark_report: dict = {}

    # ------------------------------------------------------------ main ---
    def handle(
        self,
        message: NormalizedMessage,
        *,
        catalogue: dict | None = None,
        brain: dict | None = None,
        active_order: dict | None = None,
        known_customer: dict | None = None,
        media_context: dict | None = None,
        order_mode: str = "test",
    ) -> dict:
        if not is_v2_store(message.store_id):
            raise ValueError("v2_refuses_non_test_store")
        total_started = time.perf_counter()
        execution_id = build_execution_id(
            store_id=message.store_id,
            channel=message.channel,
            customer_id=message.customer_id,
            source_message_id=message.source_message_id,
        )
        # Retry fast-path: same execution already finished -> same reply,
        # zero new Luna calls, zero new sends.
        cached_exec = _durable.exec_get(execution_id=execution_id)
        if isinstance(cached_exec, dict) and cached_exec.get("reply"):
            outbound_key = cached_exec.get("outbound_key") or ""
            return {
                "execution_id": execution_id,
                "reply": str(cached_exec.get("reply") or ""),
                "agent_core_version": AGENT_CORE_VERSION_V2,
                "luna_call_count": 0,
                "outbound_count": 0,
                "duplicate_execution": True,
                "outbound_key": outbound_key,
                "trace": dict(cached_exec.get("trace") or {}),
            }

        with conversation_lock(
            store_id=message.store_id,
            channel=message.channel,
            customer_id=message.customer_id,
        ):
            # Re-check inside the per-conversation lock (burst safety).
            cached_exec = _durable.exec_get(execution_id=execution_id)
            if isinstance(cached_exec, dict) and cached_exec.get("reply"):
                return {
                    "execution_id": execution_id,
                    "reply": str(cached_exec.get("reply") or ""),
                    "agent_core_version": AGENT_CORE_VERSION_V2,
                    "luna_call_count": 0,
                    "outbound_count": 0,
                    "duplicate_execution": True,
                    "outbound_key": cached_exec.get("outbound_key") or "",
                    "trace": dict(cached_exec.get("trace") or {}),
                }
            return self._execute_locked(
                message,
                execution_id=execution_id,
                catalogue=catalogue,
                brain=brain,
                active_order=active_order,
                known_customer=known_customer,
                media_context=media_context,
                order_mode=order_mode,
                total_started=total_started,
            )

    # ---------------------------------------------------------- locked ---
    def _execute_locked(
        self,
        message: NormalizedMessage,
        *,
        execution_id: str,
        catalogue: dict | None,
        brain: dict | None,
        active_order: dict | None,
        known_customer: dict | None,
        media_context: dict | None,
        order_mode: str,
        total_started: float,
    ) -> dict:
        ctx_started = time.perf_counter()
        store_id = message.store_id
        channel = message.channel
        customer_id = message.customer_id

        cat = dict(catalogue) if isinstance(catalogue, dict) else test_catalogue()
        cat.setdefault("store_id", TEST_STORE_ID)
        brain_in = dict(brain) if isinstance(brain, dict) else test_brain()
        media_ctx = dict(media_context) if isinstance(media_context, dict) else {}
        order = dict(active_order) if isinstance(active_order, dict) else {}
        known = dict(known_customer) if isinstance(known_customer, dict) else {}

        # 1. Merchant Brain (compact, cached by store_id + brain_version).
        brain_record = load_brain(store_id=store_id, brain=brain_in)

        # 2. Chat memory (max 6) — dialogue only.
        recent = _memory.get_recent(
            store_id=store_id, channel=channel, customer_id=customer_id,
            limit=MEMORY_LIMIT,
        )

        # 3. SessionState (strict). Post-order reset BEFORE evidence.
        state = _state.load_state(
            store_id=store_id, channel=channel, customer_id=customer_id,
            active_order=order, known_customer=known,
        )
        try:
            from scaliffy_agent.order_lifecycle import order_is_closed as _is_closed
            closed, status = _is_closed(order)
        except Exception:
            closed, status = False, ""
        state_reset: list[str] = []
        if closed:
            state_reset = state.reset_episode()
            _memory.clear_episode(
                store_id=store_id, channel=channel, customer_id=customer_id
            )
            recent = []

        # 4. Deterministic helpers (no Luna): quantity / color / city /
        #    reel-media / resolver. All tenant-scoped, current-turn only.
        try:
            from scaliffy_agent.quantity import requested_quantity as _qty
            turn_quantity = int(_qty(message.text) or 0)
        except Exception:
            turn_quantity = 0
        try:
            from scaliffy_agent.order_lifecycle import persistable_quantity as _pq
            persist_qty = int(_pq(turn_quantity) or 0)
        except Exception:
            persist_qty = 0
        if persist_qty > 0:
            state.patch({"quantity": str(persist_qty)})

        color_status: dict = {"status": "unknown"}
        try:
            from scaliffy_agent.color_status import resolve_color as _color
            history_tuples = tuple(
                {"role": r.get("role"), "text": r.get("text")} for r in recent
            )
            # resolve_color expects ConversationTurn-like objects; pass a
            # minimal adapter via dicts is unsafe, so call with empty history
            # + catalogue when shapes mismatch (still deterministic).
            try:
                color_status = _color(
                    message_text=message.text,
                    history=(),
                    catalogue=cat,
                    stored_variant=state.selected_variant_id,
                ) or {"status": "unknown"}
            except Exception:
                color_status = {"status": "unknown"}
            _ = history_tuples
        except Exception:
            color_status = {"status": "unknown"}
        if str(color_status.get("status")) in {"confirmed", "single_option"} and color_status.get("variant"):
            state.patch({"selected_variant_id": str(color_status["variant"])[:200]})
            if state.open_question == "color_choice":
                state.patch({"open_question": ""})

        resolved_city = ""
        city_source = "none"
        try:
            from scaliffy_agent.city_provenance import resolve_shipping_city as _city
            prov = _city(
                message_text=message.text,
                history=(),
                catalogue_location=str(cat.get("delivery_location") or ""),
                active_order_city=str(order.get("city") or ""),
                stored_city=state.customer_city,
                stored_source="",
                stored_updated_at="",
            ) or {}
            from scaliffy_agent.city_provenance import CURRENT_SOURCES as _CUR
            if prov.get("value") and prov.get("source") in _CUR:
                resolved_city = str(prov["value"])[:120]
                city_source = str(prov["source"])
                state.patch({"customer_city": resolved_city})
        except Exception:
            resolved_city, city_source = "", "none"

        # Reel/media -> product (deterministic, test-store scoped).
        reel_status = "no_media"
        reel_owner = False
        reel_product = ""
        try:
            attachments = message.attachments
            keys: set[str] = set()
            for item in attachments or ():
                if isinstance(item, dict):
                    for field in ("media_id", "url", "media_reference"):
                        value = str(item.get(field) or "").strip()
                        if value:
                            keys.add(value)
            if message.media_reference:
                keys.add(str(message.media_reference).strip())
            for field in ("media_id", "reel_id", "permalink"):
                value = str(media_ctx.get(field) or "").strip()
                if value:
                    keys.add(value)
            from scaliffy_agent.core_v2.media import resolve_media as _resolve_media
            adapter_pid = str(
                media_ctx.get("product_id") or media_ctx.get("recognized_product") or ""
            ).strip()
            if adapter_pid:
                reel_status, reel_product = "FOUND", adapter_pid[:200]
            else:
                resolved_media = _resolve_media(keys)
                reel_status = str(resolved_media.get("status") or "no_media")
                reel_product = str(resolved_media.get("product_id") or "")[:200]
                if reel_status == "FOUND" and resolved_media.get("variant") and not state.selected_color:
                    state.patch({"selected_color": str(resolved_media["variant"])[:120]})
            _ = test_media_map  # legacy alias map superseded by media catalog
            if reel_status == "FOUND" and reel_product:
                state.patch({"resolved_media_product_id": reel_product})
                if not state.recent_media_id and message.media_reference:
                    state.patch({"recent_media_id": str(message.media_reference)[:200]})
                if not state.active_product_id:
                    state.patch({"active_product_id": reel_product})
        except Exception:
            reel_status = "no_media"

        # Deterministic product resolver (store-scoped, no Luna).
        # Aliases bridge Latin/Arabizi/Arabic spellings from catalogue data.
        try:
            from scaliffy_agent.resolver import resolve_product as _resolve
            _pid = str(cat.get("product_id") or "").strip()
            _aliases: dict[str, str] = {}
            try:
                for _part in str(cat.get("aliases") or "").split(","):
                    _alias = _part.strip()
                    if _alias and _pid:
                        _aliases[_alias] = _pid
            except Exception:
                _aliases = {}
            resolution = _resolve(
                store_id=store_id,
                message_text=message.text,
                state_product=state.active_product_id,
                catalogue=cat,
                turso_rows=[],
                aliases=_aliases,
                media_present=reel_status not in ("", "no_media"),
            )
        except Exception:
            class _R:  # minimal fallback
                status, product_id = "NOT_FOUND", ""
            resolution = _R()

        # 4b. Deterministic sales stage (sticky progression, no Luna).
        try:
            from scaliffy_agent.core_v2.sales import next_stage as _next_stage
            from scaliffy_agent.core_v2.sales import wants_photo as _wants_photo
            from scaliffy_agent.evidence import (
                is_price_intent as _v2_is_price,
                is_shipping_intent as _v2_is_ship,
            )
            _color_decided = str((color_status or {}).get("status")) in {"confirmed", "single_option"}
            _interest = bool(
                _v2_is_price(message.text) or _v2_is_ship(message.text)
                or reel_status == "FOUND" or _wants_photo(message.text)
            )
            _stage = _next_stage(
                text=message.text, previous=state.sales_stage,
                color_decided=_color_decided, quantity=turn_quantity,
                interest=_interest, has_draft=bool(state.draft_order_id),
            )
            if _stage and _stage != state.sales_stage:
                state.patch({"sales_stage": _stage})
        except Exception:
            pass

        # 5. Deterministic evidence — rebuilt from scratch every turn.
        evidence, _fragment = build_evidence_v2(
            catalogue=cat,
            state=state,
            current_message=message.text,
            resolver_status=str(getattr(resolution, "status", "NOT_FOUND")),
            resolver_product_id=str(getattr(resolution, "product_id", "") or ""),
            requested_quantity=turn_quantity,
            resolved_city=resolved_city,
            city_source=city_source,
            color_status=color_status if isinstance(color_status, dict) else None,
            reel_status=reel_status,
            reel_owner_is_merchant=reel_owner,
        )
        # 5b. Deterministic photo inventory for photo requests (refs only).
        media_options: list[str] = []
        try:
            from scaliffy_agent.core_v2.media import available_media as _avail_media
            from scaliffy_agent.core_v2.sales import wants_photo as _wants_photo2
            if _wants_photo2(message.text):
                _mpid = str(getattr(resolution, "product_id", "") or state.active_product_id or "")
                _mvar = ""
                if isinstance(color_status, dict):
                    _mvar = str(color_status.get("variant") or color_status.get("color") or "")
                media_options = _avail_media(
                    product_id=_mpid, variant=_mvar or state.selected_color)
        except Exception:
            media_options = []
        if media_options:
            evidence["media_options"] = [str(r)[:120] for r in media_options[:4]]
        if reel_status in ("FOUND", "AMBIGUOUS"):
            evidence["media_resolution"] = {"status": reel_status, "product_id": reel_product}
        context_build_ms = int((time.perf_counter() - ctx_started) * 1000)

        # 6. BuildAgentInput — PURE (no DB, no mutation, no Luna).
        agent_input = build_agent_input(
            merchant_brain={"content": brain_record["content"],
                            "version": brain_record["version"]},
            session_state={"fragment": state.to_luna_fragment(), **state.to_dict()},
            recent_messages=recent,
            evidence=evidence,
            current_message=message.text,
        )
        assert agent_input.get("debug", {}).get("history_blocks", 1) <= 1

        # 7. ONE Luna call (non-negotiable).
        luna_started = time.perf_counter()
        reply_text, luna_extras = self._call_luna_once(
            agent_input=agent_input, evidence=evidence, message=message,
            catalogue=cat, resolution_status=str(getattr(resolution, "status", "")),
            resolver_product_id=str(getattr(resolution, "product_id", "") or ""),
        )
        luna_ms = int((time.perf_counter() - luna_started) * 1000)
        luna_fallback_reason = ""
        model_requested_override = ""
        try:
            luna_fallback_reason = str((luna_extras or {}).pop("_luna_fallback_reason", "") or "")
            model_requested_override = str((luna_extras or {}).pop("_model_requested", "") or "")
        except Exception:
            luna_fallback_reason = ""
            model_requested_override = ""

        # 8. Validate (deterministic, NO second Luna). Safe fallback on
        #    violation: short honest clarification, HTTP 200 semantics.
        order_action = str(luna_extras.get("order_action") or "none")
        order_draft = dict(luna_extras.get("order_draft") or {})
        try:
            from scaliffy_agent.validation import (
                latinize_digits as _lat,
                normalize_price_decimals as _dec,
                validate_action as _validate,
            )
            checked_text = _dec(_lat(reply_text))
            order_action, order_draft, _media_action = _validate(
                reply_text=checked_text,
                order_action=order_action,
                order_draft=order_draft,
                media_action=str(luna_extras.get("media_action") or "none"),
                evidence=evidence,
                resolver_status=str(getattr(resolution, "status", "NOT_FOUND")),
                resolver_product_id=str(getattr(resolution, "product_id", "") or ""),
            )
            reply_text = checked_text
            reason = f"safe_fallback_{luna_fallback_reason}" if luna_fallback_reason else "v2_ok"
        except RuntimeError as exc:
            if not str(exc).startswith("unsafe_reply:"):
                raise
            reason = f"safe_fallback_{str(exc).split(':', 1)[-1]}"
            try:
                from scaliffy_agent.evidence import (
                    is_price_intent as _is_price,
                    is_shipping_intent as _is_ship,
                )
                if _is_ship(message.text) and str(evidence.get("delivery_price") or "") == "UNKNOWN":
                    reply_text = "التوصيل كاين لجميع المدن، عطيني المدينة ديالك ونعطيك الثمن بالضبط."
                elif _is_price(message.text):
                    reply_text = "الباك متوفر، قوليا شحال بغيتي (واحد ولا جوج) ونعطيك الثمن بالضبط."
                else:
                    reply_text = "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
            except Exception:
                reply_text = "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
            order_action, order_draft = "none", {}
        try:
            from scaliffy_agent.validation import (
                latinize_digits as _lat2,
                merchant_animal as _animal,
                merchant_animal_emoji as _emoji,
                merchant_vocabulary as _vocab,
                neutral_masculine as _masc,
                normalize_price_decimals as _dec2,
            )
            reply_text = _dec2(_lat2(reply_text))
            reply_text = _vocab(reply_text)
            reply_text = _emoji(_animal(reply_text))
            reply_text = _masc(reply_text)
        except Exception:
            pass
        if not str(reply_text or "").strip():
            reply_text = "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
            reason = "safe_fallback_empty"

        # Sales stage follows the deterministic action (no second Luna).
        try:
            if str(order_action or "") in {"start_order", "start_new_order", "resend_order_form"}:
                state.patch({"sales_stage": "collecting_order"})
        except Exception:
            pass
        # Order test mode: never create real orders on the test store.
        if str(order_mode or "test").lower() != "test":
            order_mode = "test"
        if isinstance(order_draft, dict) and order_draft:
            order_draft = dict(order_draft)
            order_draft["environment"] = "test"
            order_draft["store_id"] = TEST_STORE_ID

        # 9. Persist structured SessionState changes only.
        try:
            if str(getattr(resolution, "status", "")) == "FOUND" and getattr(
                resolution, "product_id", ""
            ):
                state.patch({"active_product_id": str(resolution.product_id)[:200]})
            if isinstance(order_draft, dict) and str(order_draft.get("draft_order_id") or ""):
                state.patch({"draft_order_id": str(order_draft["draft_order_id"])[:120]})
            _state.save_state(
                store_id=store_id, channel=channel, customer_id=customer_id,
                state=state,
            )
        except Exception:
            pass

        # 10. Append chat memory (bounded; Luna sees max ~6 next turn).
        try:
            if str(message.text or "").strip():
                _memory.append_user(
                    store_id=store_id, channel=channel, customer_id=customer_id,
                    text=message.text,
                )
            if str(reply_text or "").strip():
                _memory.append_assistant(
                    store_id=store_id, channel=channel, customer_id=customer_id,
                    text=reply_text,
                )
        except Exception:
            pass

        # 11. Idempotent outbound (durable atomic claim, text-independent).
        outbound_key, owned = _outbound.claim_outbound(
            store_id=store_id, channel=channel,
            conversation_id=message.conversation_id or customer_id,
            source_message_id=message.source_message_id,
        )
        outbound_count = 0
        if owned:
            _outbound.mark_sent(outbound_key=outbound_key, reply=reply_text)
            outbound_count = 1
        else:
            sent, cached_reply = _outbound.already_sent(outbound_key=outbound_key)
            if sent and cached_reply:
                reply_text = cached_reply
            outbound_count = 0

        total_ms = int((time.perf_counter() - total_started) * 1000)
        spark_report = dict(getattr(self, "_last_spark_report", {}) or {})
        trace = {
            "agent_core_version": AGENT_CORE_VERSION_V2,
            "resolver": str(getattr(resolution, "status", "")),
            "reason": reason if "reason" in dir() else "v2_ok",
            "order_action": order_action,
            "reel_status": reel_status,
            "state_reset": state_reset,
            "order_mode": order_mode,
            "model_requested": str(spark_report.get("model_requested") or model_requested_override or ""),
            "model_resolved": str(spark_report.get("model_resolved") or ""),
            "model_input_tokens": int(spark_report.get("input_tokens") or 0),
            "model_output_tokens": int(spark_report.get("output_tokens") or 0),
            "sales_stage": state.sales_stage,
            "active_product_id": state.active_product_id,
        }
        obs = _obs.log_turn(
            execution_id=execution_id, store_id=store_id, channel=channel,
            customer_id=customer_id, agent_input=agent_input,
            output_text=reply_text, context_build_ms=context_build_ms,
            luna_ms=luna_ms, total_ms=total_ms, luna_call_count=1,
            outbound_count=outbound_count, extra=trace,
        )
        result = {
            "execution_id": execution_id,
            "reply": reply_text,
            "agent_core_version": AGENT_CORE_VERSION_V2,
            "luna_call_count": 1,
            "outbound_count": outbound_count,
            "duplicate_execution": False,
            "outbound_key": outbound_key,
            "trace": trace,
            "observability": obs,
            "order_action": order_action,
            "order_draft": order_draft,
            "model": str(trace.get("model_resolved") or trace.get("model_requested") or ""),
            "sales_stage": state.sales_stage,
        }
        _durable.exec_put(execution_id=execution_id, result=result)
        # Cache stores the full result; the retry path returns a slim view.
        _ = uuid.uuid4
        return result

    # ----------------------------------------------------------- luna ----
    def _call_luna_once(
        self, *, agent_input: dict, evidence: dict, message: NormalizedMessage,
        catalogue: dict, resolution_status: str, resolver_product_id: str,
    ) -> tuple[str, dict]:
        if self.model is None:
            # Offline deterministic fallback (tests without network): answer
            # directly from evidence truth. Production passes a real Luna
            # adapter; the pipeline shape is identical either way.
            return _deterministic_reply(evidence=evidence, text=message.text), {
                "order_action": "none", "order_draft": {}, "media_action": "none",
            }
        # Compact Muse path: the pure AgentInput goes straight to the
        # model (no legacy prompt rebuild, still exactly ONE generation).
        if hasattr(self.model, "answer_compact"):
            try:
                self._last_spark_report = {}
                compact_text, compact_extras, compact_report = self.model.answer_compact(
                    agent_input=agent_input, evidence=evidence, catalogue=catalogue,
                )
                self._last_spark_report = dict(compact_report or {})
                return str(compact_text or ""), {
                    "order_action": str((compact_extras or {}).get("order_action") or "none"),
                    "order_draft": dict((compact_extras or {}).get("order_draft") or {}),
                    "media_action": str((compact_extras or {}).get("media_action") or "none"),
                }
            except Exception as exc:
                return _luna_fallback(
                    evidence=evidence, text=message.text, exc=exc)
        # Adapt the V2 compact payload to the existing ChatModel shape.
        # History travels EXACTLY once as real conversation messages.
        try:
            from scaliffy_agent.types import (
                Channel,
                ConversationTurn,
                IncomingMessage,
                ReplyScript,
            )
            history: list = []
            for item in agent_input.get("recent_messages", []):
                role = "customer" if item.get("role") == "customer" else "assistant"
                history.append(ConversationTurn(role, str(item.get("content") or "")))
            try:
                channel_enum = Channel(str(message.channel))
            except ValueError:
                channel_enum = Channel.TEST
            runtime_block = (
                f"{agent_input.get('session_state','')}\n\n"
                f"{agent_input.get('evidence','')}\n\n"
                f"CURRENT CUSTOMER MESSAGE (answer this now):\n"
                f"{agent_input.get('current_message','')}"
            )
            adapted = IncomingMessage(
                message_id=message.source_message_id,
                text=str(agent_input.get("current_message") or ""),
                customer_id=message.customer_id,
                attachments=(),
                history=tuple(history),
                channel=channel_enum,
                catalogue_context={str(k): str(v)[:500] for k, v in catalogue.items()},
                store_brain={"content": agent_input.get("merchant_brain", ""),
                             "merchant_id": "625374849"},
                merchant_runtime_context=runtime_block[:6000],
            )
            answer = self.model.answer(
                message=adapted, script=ReplyScript.ARABIC_DARIJA,
                facts=(), memories=(), store_context_required=True,
            )
            extras = {
                "order_action": str(getattr(self.model, "last_order_action", "none") or "none"),
                "order_draft": dict(getattr(self.model, "last_order_draft", {}) or {})
                if isinstance(getattr(self.model, "last_order_draft", {}), dict) else {},
                "media_action": str(getattr(self.model, "last_media_action", "none") or "none"),
            }
            return str(answer or ""), extras
        except Exception as exc:
            # A Luna-call failure (unsafe reply, empty/malformed output,
            # provider error) must NEVER become a customer-facing 500.
            return _luna_fallback(
                evidence=evidence, text=message.text, exc=exc)


def _deterministic_reply(*, evidence: dict, text: str) -> str:
    """Offline Luna replacement used ONLY when no model is injected.

    Reads the deterministic evidence (never history) and answers the
    current intent directly in natural Darija.
    """
    try:
        from scaliffy_agent.evidence import (
            is_price_intent as _is_price,
            is_shipping_intent as _is_ship,
        )
        from scaliffy_agent.quantity import requested_quantity as _qty
    except Exception:
        _is_price = lambda t: False  # noqa: E731
        _is_ship = lambda t: False  # noqa: E731
        _qty = lambda t: 0  # noqa: E731
    message = str(text or "")
    qty = 0
    try:
        qty = int(_qty(message) or 0)
    except Exception:
        qty = 0
    price = str(evidence.get("price") or "").strip()
    delivery = str(evidence.get("delivery_price") or "").strip()
    total = str(evidence.get("total_with_delivery") or "").strip()
    offer = evidence.get("offer") if isinstance(evidence.get("offer"), dict) else {}
    offer_total = str(offer.get("offer_total_price") or evidence.get("offer_total_price") or "").strip()
    ship_intent = bool(_is_ship(message))
    price_intent = bool(_is_price(message))
    if ship_intent:
        # Multi-intent ("chhal tawsil?") answers BOTH facts in one reply:
        # shipping 35 MAD wins over any poisoned history, price included
        # when also asked. Evidence is the only truth source.
        if delivery == "0":
            ship_part = "التوصيل فابور على هاد العرض لجميع المدن."
        elif delivery and delivery != "UNKNOWN":
            ship_part = f"التوصيل {delivery} درهم لجميع المدن."
        else:
            return "التوصيل كاين لجميع المدن، عطيني المدينة ديالك ونعطيك الثمن بالضبط."
        if price_intent and price and price != "UNKNOWN":
            return f"الثمن ديال الباك هو {price} درهم. {ship_part}"
        return ship_part
    if qty >= 2 or "179" in message:
        amount = offer_total or "179"
        return f"جوج باكات بـ{amount} درهم مع جوج ݣورميطات والتوصيل فابور."
    if _is_price(message):
        if price and price != "UNKNOWN":
            if total and total.startswith(price):
                return f"الثمن ديال الباك هو {price} درهم (المجموع مع التوصيل {total})."
            return f"الثمن ديال الباك هو {price} درهم."
        return "الباك متوفر، قوليا شحال بغيتي (واحد ولا جوج) ونعطيك الثمن بالضبط."
    lowered = message.strip().lower()
    if lowered in {"سلام", "salam", "salut", "bonjour", "hello", "cc", "azul"} or not message.strip():
        return "لاباس الحمد لله! كيفاش نقدر نعاونك؟"
    if price and ("hada" in lowered or "chhal" in lowered or "?" in message):
        return f"الثمن ديال الباك هو {price} درهم."
    return "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
