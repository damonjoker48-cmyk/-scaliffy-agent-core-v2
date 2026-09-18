from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import unicodedata
import uuid
from typing import Protocol

from .language import detect_reply_script, enforce_reply_script
from .media import visual_attachment_urls
from .store import KnowledgeStore
from .types import (
    AgentReply, ConversationBudget, ConversationCategory, ConversationSurface,
    IncomingMessage, KnowledgeDocument, KnowledgeHit, ReplyScript, RetrievalPlan, StoreContext,
)


logger = logging.getLogger("scaliffy.agent_core")

ORDER_UI_ACTIONS = {"start_order", "start_new_order", "resend_order_form"}
ORDER_ACTIONS = {
    "none", *ORDER_UI_ACTIONS, "confirm", "refuse", "modify",
}


def _fact_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    return " ".join(re.findall(
        r"[^\W_]+",
        "".join(char for char in normalized if not unicodedata.combining(char)),
        flags=re.UNICODE,
    ))


def _structured_catalogue_hit(
    store: StoreContext, message: IncomingMessage,
) -> KnowledgeHit | None:
    """Build one authoritative hit from the adapter's current-turn product.

    The adapter only supplies this object after resolving the current text
    against exactly one product in this merchant's catalogue.  Checking the
    store id again here keeps the optimization tenant-safe even if a caller
    violates the transport contract.
    """
    fact = message.catalogue_context if isinstance(message.catalogue_context, dict) else {}
    if str(fact.get("store_id") or "") != store.store_id:
        return None
    name = str(fact.get("name") or "").strip()
    if not name:
        return None
    product_id = str(fact.get("product_id") or fact.get("sku") or _fact_identity(name)).strip()
    price = str(fact.get("price") or "").strip()
    currency = str(fact.get("currency") or "").strip()
    stock = str(fact.get("stock") or "").strip()
    description = str(fact.get("description") or "").strip()
    details = str(fact.get("details") or "").strip()
    answer_scope = str(fact.get("answer_scope") or "").strip()
    delivery_context = {}
    try:
        decoded_delivery = json.loads(str(fact.get("delivery_context") or "{}"))
        if isinstance(decoded_delivery, dict):
            delivery_context = decoded_delivery
    except (TypeError, ValueError, json.JSONDecodeError):
        delivery_context = {}
    delivery_location = str(
        fact.get("delivery_location")
        or delivery_context.get("delivery_location")
        or ""
    ).strip()
    delivery_price = str(
        fact.get("delivery_price")
        or delivery_context.get("delivery_price")
        or ""
    ).strip()
    delivery_currency = str(
        fact.get("delivery_currency")
        or delivery_context.get("delivery_currency")
        or currency
        or "MAD"
    ).strip()
    delivery_available = str(
        fact.get("delivery_available")
        or delivery_context.get("delivery_available")
        or ""
    ).strip()
    clauses = [f"Produit: {name}."]
    if price:
        clauses.append(f"Prix: {price}{(' ' + currency) if currency else ''}.")
    if stock:
        clauses.append(f"Stock: {stock}.")
    available = str(fact.get("available") or "").strip()
    if available:
        clauses.append(f"Produit disponible: {available}.")
    for key, label in (
        ("variants", "Variantes du catalogue et disponibilité"),
        ("order_specifications", "Finitions et spécifications disponibles"),
        ("visual_policy", "Disponibilité des produits et photos"),
    ):
        value = str(fact.get(key) or "").strip()
        if value:
            clauses.append(f"{label}: {value[:3_000]}")
    if description:
        clauses.append(f"Description: {description[:1_200]}")
    if details:
        clauses.append(f"Détails: {details[:1_200]}")
    if delivery_price:
        destination = f" à {delivery_location}" if delivery_location else ""
        clauses.append(
            f"Livraison{destination}: {delivery_price} {delivery_currency}."
        )
    if delivery_available:
        clauses.append(f"Livraison disponible: {delivery_available}.")
    policy = delivery_context.get("policy")
    if isinstance(policy, dict):
        clauses.append("Politique de livraison: " + json.dumps(policy, ensure_ascii=False, separators=(",", ":"))[:3_000])
        policy_currency = str(policy.get("currency") or delivery_currency)
        if policy.get("default_price") is not None:
            clauses.append(f"Tarif de livraison par défaut: {policy['default_price']} {policy_currency}.")
        for zone in (policy.get("zones") or [])[:30]:
            if isinstance(zone, dict) and zone.get("price") is not None:
                clauses.append(f"Livraison à {zone.get('location') or ''}: {zone['price']} {zone.get('currency') or policy_currency}.")
    if answer_scope:
        clauses.append(f"Portée de la réponse: {answer_scope[:600]}")
    return KnowledgeHit(
        document=KnowledgeDocument(
            id=f"structured:{store.store_id}:product:{product_id}"[:500],
            store_id=store.store_id,
            source="scaliffy:structured-catalogue",
            text=" ".join(clauses),
            metadata={
                "kind": "product",
                "product_id": product_id[:180],
                "title": name[:500],
            },
        ),
        score=1.0,
    )


def _conversation_budget_threshold() -> float:
    try:
        return max(
            0.5,
            float(os.environ.get("SOCIAL_CONVERSATION_BUDGET_THRESHOLD") or "3"),
        )
    except ValueError:
        return 3.0

class ChatModel(Protocol):
    def plan(self, *, message: IncomingMessage, script: ReplyScript) -> RetrievalPlan: ...
    def answer(
        self, *, message: IncomingMessage, script: ReplyScript,
        facts: tuple[KnowledgeHit, ...], memories: tuple[str, ...] = (),
        store_context_required: bool = False,
        product_clarification_required: bool = False,
        conversation_budget: ConversationBudget = ConversationBudget(),
        restriction_required: bool = False,
    ) -> str: ...


class SafeRuleModel:
    """Test-only deterministic model. Production replaces this with a Luna adapter."""

    def plan(self, *, message: IncomingMessage, script: ReplyScript) -> RetrievalPlan:
        # Deterministic test double: retrieve for the in-memory fixture.
        return RetrievalPlan(required=True, reason="test", query=message.text)

    def answer(
        self, *, message: IncomingMessage, script: ReplyScript,
        facts: tuple[KnowledgeHit, ...], memories: tuple[str, ...] = (),
        store_context_required: bool = False,
        product_clarification_required: bool = False,
        conversation_budget: ConversationBudget = ConversationBudget(),
        restriction_required: bool = False,
    ) -> str:
        del memories, store_context_required, product_clarification_required, conversation_budget, restriction_required
        if not facts:
            if script is ReplyScript.ARABIC_DARIJA:
                return "لاباس الحمد لله! نتا لاباس؟ كيفاش نقدر نعاونك؟"
            return "Labas lhamdollah! Nta labas? Kifach n9der n3awnk?"
        best_fact = facts[0].document.text
        if script is ReplyScript.ARABIC_DARIJA:
            return f"حسب معلومات المتجر: {best_fact}"
        return f"Hassab ma3loumat store: {best_fact}"


class AgentCore:
    def __init__(self, *, knowledge_store: KnowledgeStore, model: ChatModel) -> None:
        self.knowledge_store = knowledge_store
        self.model = model
        self.deferred_persistence = None

    def reply(
        self, *, store: StoreContext, message: IncomingMessage,
        defer_persistence: bool = False,
    ) -> AgentReply | None:
        """Return None only when a human owns the conversation or the agent is off."""
        if not store.agent_enabled or store.human_takeover:
            return None

        # CORE V2 FEATURE GATE (test-only): store 625374849 uses the
        # isolated AgentCoreV2 engine. Every other store — including
        # production Adam Luxe 166510782 — keeps its CURRENT path
        # unchanged below. No global rollout, no fallback to V2.
        try:
            from .core_v2.config import TEST_STORE_ID as _V2_TEST_STORE
        except Exception:
            _V2_TEST_STORE = "625374849"
        try:
            from .core_v2.config import is_canary_store as _is_canary_store
            _canary = bool(_is_canary_store(
                str(store.store_id or ""),
                str(store.merchant_account_id or "")))
        except Exception:
            _canary = False
        if str(store.store_id or "").strip() == str(_V2_TEST_STORE) or _canary:
            return self._reply_with_core_v2(
                store=store, message=message,
                order_mode=("live" if _canary else "test"), canary=_canary)

        if store.merchant_account_id == "166510782":
            return self._reply_with_store_brain(store=store, message=message)

        trace_id = str(uuid.uuid4())
        written_customer_turns = [
            turn.text
            for turn in message.history
            if turn.role == "customer" and turn.source == "text" and any(char.isalpha() for char in turn.text)
        ]
        # Current written language wins over old history. Voice transcription
        # does not express a writing preference; use the latest written turn.
        script_source = message.text
        if message.source == "instagram_voice" and written_customer_turns:
            script_source = written_customer_turns[-1]
        elif not any(char.isalpha() for char in message.text) and written_customer_turns:
            script_source = written_customer_turns[-1]
        script = detect_reply_script(script_source)
        budget = self.knowledge_store.conversation_budget(
            store=store, customer_id=message.customer_id,
        )
        try:
            plan = self.model.plan(message=message, script=script)
        except Exception:
            # The semantic router is an optimisation/data-plane helper, not
            # the conversational brain. If embeddings or routing fail, Luna
            # still receives the full conversation and must answer normally.
            logger.exception(
                "Agent Core router failed open store=%s channel=%s message=%s",
                store.store_id, store.channel.value, message.message_id,
            )
            plan = RetrievalPlan(
                required=False,
                reason="router_failed_open",
                query=message.text,
            )
        # Photos and image thumbnails are first-class input to Luna. Only a
        # genuinely unreadable attachment (for example a video without a
        # thumbnail) suppresses factual retrieval based on that media.
        readable_visuals = visual_attachment_urls(message.attachments)
        media_unavailable = bool(
            message.attachments
            and message.source != "instagram_voice"
            and not readable_visuals
            and not message.text.strip()
        )
        restriction_required = bool(
            budget.social_cooldown and plan.cooldown_gate == "blocked"
        )
        if message.surface == ConversationSurface.INSTAGRAM_COMMENT:
            # Store-related comments receive the real answer as an Instagram
            # private reply. Public comments only receive a short “answered in
            # DM” acknowledgement in the channel dispatcher.
            if plan.comment_action != "move_to_private":
                return None
        # The public-comment surface changes delivery only: it gets a private
        # answer and a short public acknowledgement. It must not create a
        # second conversational brain or a hard-coded reply. An embedding can
        # safely suppress a catalogue lookup when a product may be
        # unidentified, but it cannot reliably decide the intent by itself.
        # Keep the no-RAG optimisation, then let the one existing Luna call
        # interpret the message and available conversation context exactly as
        # it would for a direct message.
        query = plan.query.strip() or message.text
        hits: tuple[KnowledgeHit, ...] = ()
        retrieval_latency_ms = 0
        retrieval_sources: tuple[str, ...] = ()
        if plan.required and not media_unavailable:
            structured_hit = _structured_catalogue_hit(store, message)
            if structured_hit is not None and plan.preferred_knowledge_kinds == ("product",):
                # Exact structured commerce data is both faster and stronger
                # than a nearest-neighbour round trip. Pinecone remains the
                # fallback for partial/typo queries that the adapter could not
                # uniquely resolve and for all policy/store knowledge.
                hits = (structured_hit,)
                retrieval_sources = ("structured_catalogue",)
            else:
                retrieval_started_at = time.perf_counter()
                try:
                    hits = tuple(self.knowledge_store.query(
                        store_id=store.store_id,
                        text=query,
                        limit=4,
                        vector=plan.query_embedding,
                        preferred_kinds=plan.preferred_knowledge_kinds,
                    ))
                    retrieval_sources = ("pinecone",)
                except Exception:
                    # A missing vector, Pinecone timeout or empty namespace must
                    # never silence the customer. Luna can ask for clarification
                    # from the same turn without inventing merchant facts.
                    logger.exception(
                        "Agent Core RAG failed open store=%s channel=%s message=%s",
                        store.store_id, store.channel.value, message.message_id,
                    )
                finally:
                    retrieval_latency_ms = round(
                        (time.perf_counter() - retrieval_started_at) * 1000
                    )
        fresh_hits = hits
        # Preserve the authoritative facts that grounded the immediately
        # preceding store exchange. A new factual question may retrieve a
        # different product/policy subset, but that must not make Luna
        # contradict a fact it established one turn ago. This is a bounded,
        # tenant-scoped data capability: no classifier, no second model call,
        # no product-specific rule, and it is never loaded for conversation.
        if plan.required and not media_unavailable:
            try:
                loader = getattr(self.knowledge_store, "recent_grounding_context", None)
                loaded = loader(store=store, message=message, limit=4) if callable(loader) else ()
                continuity_hits = tuple(loaded) if isinstance(loaded, (tuple, list)) else ()
                merged: list[KnowledgeHit] = []
                seen_ids: set[str] = set()
                for hit in (*hits, *continuity_hits):
                    document_id = str(hit.document.id or "")
                    if not document_id or document_id in seen_ids:
                        continue
                    seen_ids.add(document_id)
                    merged.append(hit)
                active_product = _fact_identity(
                    str((message.active_order or {}).get("product") or "")
                )

                def grounding_rank(hit: KnowledgeHit) -> tuple[int, float]:
                    title = _fact_identity(str(hit.document.metadata.get("title") or ""))
                    return (
                        int(bool(active_product and title == active_product)),
                        float(hit.score),
                    )

                ranked = sorted(merged, key=grounding_rank, reverse=True)
                exact_product_hits = [
                    hit for hit in ranked
                    if grounding_rank(hit)[0]
                ]
                if exact_product_hits:
                    # Once the adapter has resolved the active product, weak
                    # catalogue neighbours are noise. Keep that exact product
                    # plus store/policy facts, never competing products.
                    ranked = exact_product_hits + [
                        hit for hit in ranked
                        if str(hit.document.metadata.get("kind") or "") != "product"
                    ]
                hits = tuple(ranked[:6])
            except Exception:
                logger.exception(
                    "Agent Core recent grounding failed open store=%s channel=%s message=%s",
                    store.store_id, store.channel.value, message.message_id,
                )
        memories: tuple[str, ...] = ()
        if plan.memory_required and not media_unavailable:
            try:
                memories = tuple(self.knowledge_store.customer_memory_context(
                    store=store,
                    message=message,
                    vector=plan.query_embedding,
                    limit=3,
                ))
            except Exception:
                # Long-term memory enriches Luna but never gates a reply.
                logger.exception(
                    "Agent Core memory failed open store=%s channel=%s message=%s",
                    store.store_id, store.channel.value, message.message_id,
                )
        try:
            answer = self.model.answer(
                message=message, script=script, facts=hits, memories=memories,
                store_context_required=plan.required and not media_unavailable,
                product_clarification_required=plan.product_clarification_required and not media_unavailable,
                conversation_budget=budget,
                restriction_required=restriction_required,
            )
        except Exception:
            # Provider/parser failures remain internal. The channel adapter
            # owns durable retry/suppression and must not manufacture Darija.
            logger.exception(
                "Agent Core Luna generation failed "
                "store=%s channel=%s message=%s",
                store.store_id, store.channel.value, message.message_id,
            )
            raise
        raw_order_action = str(getattr(self.model, "last_order_action", "none") or "none").lower()
        order_action = raw_order_action if raw_order_action in ORDER_ACTIONS else "none"
        answer = enforce_reply_script(answer, script)
        if not answer and order_action not in ORDER_UI_ACTIONS:
            raise RuntimeError("agent_core_empty_non_action_reply")
        def metric(name: str) -> int:
            value = getattr(self.model, name, 0)
            return int(value) if isinstance(value, (int, float)) else 0
        def metric_float(name: str) -> float:
            value = getattr(self.model, name, 0.0)
            return float(value) if isinstance(value, (int, float)) else 0.0
        def metric_text(name: str) -> str:
            value = getattr(self.model, name, "")
            return value if isinstance(value, str) else ""
        raw_updates = getattr(self.model, "last_memory_updates", ())
        updates = tuple(raw_updates) if isinstance(raw_updates, (tuple, list)) else ()
        raw_category = getattr(self.model, "last_conversation_category", None)
        try:
            category = ConversationCategory(str(raw_category)) if raw_category else None
        except ValueError:
            category = None
        raw_order_draft = getattr(self.model, "last_order_draft", {})
        order_draft = dict(raw_order_draft) if isinstance(raw_order_draft, dict) else {}
        raw_media_action = str(
            getattr(self.model, "last_media_action", "none") or "none"
        ).lower()
        media_action = (
            raw_media_action
            if raw_media_action in {"none", "send_product_image"}
            else "none"
        )
        raw_media_selection = getattr(self.model, "last_media_selection", {})
        media_selection = (
            dict(raw_media_selection) if isinstance(raw_media_selection, dict) else {}
        )
        next_budget = (
            budget.apply(category, threshold=_conversation_budget_threshold())
            if category else budget
        )
        def persist_exchange() -> None:
            try:
                self.knowledge_store.persist_customer_exchange(
                    store=store,
                    message=message,
                    reply=answer,
                    script=script,
                    vector=plan.query_embedding,
                    updates=updates,
                    conversation_budget=next_budget,
                    grounded_facts=tuple(hit.document for hit in fresh_hits[:4]),
                )
            except Exception:
                # Persistence is retriable operational work. The already-created
                # customer reply remains valid and must still be delivered.
                logger.exception(
                    "Agent Core persistence failed after reply store=%s channel=%s message=%s",
                    store.store_id, store.channel.value, message.message_id,
                )

        self.deferred_persistence = persist_exchange if defer_persistence else None
        if not defer_persistence:
            persist_exchange()
        return AgentReply(
            text=answer,
            script=script,
            used_rag=bool(hits),
            reason=(
                "model_media_unavailable" if media_unavailable else
                (
                    "model_structured_catalogue_answer"
                    if "structured_catalogue" in retrieval_sources
                    else ("model_store_answer" if hits else ("model_store_clarification" if plan.required else "model_conversation"))
                )
            ),
            trace_id=trace_id,
            knowledge_ids=tuple(hit.document.id for hit in hits),
            retrieval_namespace=f"agent-core:{store.store_id}",
            retrieval_filter={"store_id": store.store_id},
            retrieval_sources=retrieval_sources,
            retrieved_chunks=tuple({
                "knowledge_id": hit.document.id,
                "source": hit.document.source,
                "kind": str(hit.document.metadata.get("kind") or ""),
                "title": str(hit.document.metadata.get("title") or "")[:180],
                "score": round(float(hit.score), 6),
            } for hit in hits),
            rag_called=plan.required and not media_unavailable,
            luna_call_count=1,
            embedding_call_count=(0 if "structured_catalogue" in retrieval_sources else 1),
            llm_input_tokens=metric("last_input_tokens"),
            llm_output_tokens=metric("last_output_tokens"),
            llm_latency_ms=metric("last_llm_latency_ms"),
            retrieval_latency_ms=retrieval_latency_ms,
            retrieval_store_score=metric_float("last_retrieval_store_score"),
            retrieval_other_score=metric_float("last_retrieval_other_score"),
            memory_called=plan.memory_required and not media_unavailable,
            memory_updates_count=len(updates),
            memory_relevance_score=metric_float("last_memory_relevance_score"),
            recommendation_score=metric_float("last_recommendation_score"),
            conversation_category=category.value if category else "",
            social_cooldown_activated=next_budget.social_cooldown,
            order_action=order_action,
            order_draft=order_draft,
            media_action=media_action,
            media_selection=media_selection,
            raw_model_output=metric_text("last_raw_model_output"),
            raw_model_reply=metric_text("last_raw_model_reply"),
            model_provider=metric_text("last_model_provider"),
            requested_model=metric_text("last_requested_model"),
            resolved_model=metric_text("last_resolved_model"),
            temperature=metric_float("last_temperature"),
            max_output_tokens=metric("last_max_output_tokens"),
            reasoning_effort=metric_text("last_reasoning_effort"),
            response_format=metric_text("last_response_format"),
            effective_prompt_sha256=metric_text("last_effective_prompt_sha256"),
        )

    def _reply_with_core_v2(
        self, *, store: StoreContext, message: IncomingMessage,
        order_mode: str = "test", canary: bool = False,
    ) -> AgentReply:
        """Core V2 path — test store 625374849 plus explicit canary tenants.

        Delegates to scaliffy_agent.core_v2.pipeline.AgentCoreV2 using the
        SAME model slot (no second AI implementation). Canary tenants use
        ONLY caller-supplied merchant data (never test seed). Production
        behavior for every other store is untouched.
        """
        from .core_v2.config import AGENT_CORE_VERSION_V2
        from .core_v2.normalizer import normalize as _normalize
        from .core_v2.pipeline import AgentCoreV2 as _CoreV2

        attachments: list[dict] = []
        for item in message.attachments or ():
            attachments.append({
                "url": str(getattr(item, "url", "") or "")[:2000],
                "mime_type": str(getattr(item, "mime_type", "") or "")[:120],
                "media_id": str(getattr(item, "media_id", "") or "")[:200],
            })
        media_ref = ""
        try:
            media_ctx = message.media_context if isinstance(message.media_context, dict) else {}
            media_ref = str(
                media_ctx.get("media_reference") or media_ctx.get("media_id")
                or media_ctx.get("reel_id") or ""
            )[:500]
        except Exception:
            media_ctx = {}
        normalized = _normalize(
            store_id=store.store_id,
            channel=message.channel.value if hasattr(message.channel, "value") else str(message.channel),
            customer_id=message.customer_id,
            source_message_id=message.message_id,
            text=message.text,
            attachments=tuple(attachments),
            media_reference=media_ref,
            conversation_id=message.customer_id,
        )
        # Seed durable V2 memory from adapter history ONLY when the V2
        # memory is empty (migration continuity, still bounded to 6).
        try:
            from .core_v2 import memory as _v2mem
            existing = _v2mem.get_recent(
                store_id=store.store_id,
                channel=normalized.channel,
                customer_id=message.customer_id,
            )
            if not existing and message.history:
                tail = list(message.history)[-6:]
                for turn in tail:
                    role = str(getattr(turn, "role", "") or "")
                    text = str(getattr(turn, "text", "") or "").strip()[:600]
                    if role == "customer" and text:
                        _v2mem.append_user(
                            store_id=store.store_id, channel=normalized.channel,
                            customer_id=message.customer_id, text=text,
                        )
                    elif role in ("assistant", "human") and text:
                        _v2mem.append_assistant(
                            store_id=store.store_id, channel=normalized.channel,
                            customer_id=message.customer_id, text=text,
                        )
        except Exception:
            logger.exception("Core V2 history seed failed open store=%s", store.store_id)
        engine = _CoreV2(model=self.model)
        out = engine.handle(
            normalized,
            catalogue=dict(message.catalogue_context or {}) or None,
            brain=dict(message.store_brain or {}) or None,
            active_order=dict(message.active_order or {}),
            known_customer=dict(message.known_customer or {}),
            media_context=dict(media_ctx),
            order_mode=order_mode,
            canary=canary,
        )
        script = detect_reply_script(message.text or "")
        trace = dict(out.get("trace") or {})
        obs = dict(out.get("observability") or {})
        order_action = str(out.get("order_action") or "none")
        if order_action not in ORDER_ACTIONS:
            order_action = "none"
        order_draft = out.get("order_draft") if isinstance(out.get("order_draft"), dict) else {}
        return AgentReply(
            text=str(out.get("reply") or ""),
            requested_model=str(trace.get("model_requested") or ""),
            resolved_model=str(trace.get("model_resolved") or ""),
            agent_core_version=str(out.get("agent_core_version") or AGENT_CORE_VERSION_V2),
            script=script,
            used_rag=False,
            reason=str(trace.get("reason") or "v2_ok"),
            trace_id=str(out.get("execution_id") or str(uuid.uuid4())),
            retrieval_sources=("store_brain_v2", "evidence_v2"),
            rag_called=False,
            luna_call_count=int(out.get("luna_call_count") or 1),
            embedding_call_count=0,
            llm_input_tokens=int(obs.get("total_input_tokens") or 0),
            llm_output_tokens=int(obs.get("output_tokens") or 0),
            llm_latency_ms=int(obs.get("luna_ms") or 0),
            retrieval_latency_ms=int(obs.get("context_build_ms") or 0),
            memory_called=True,
            order_action=order_action,  # type: ignore[arg-type]
            order_draft=dict(order_draft),
        )

    def _reply_with_store_brain(
        self, *, store: StoreContext, message: IncomingMessage,
    ) -> AgentReply:
        """Target stack for 166510782 (Adam Luxe): one Luna turn, no Pinecone.

        Flow: tenant resolve -> Merchant Brain (cache) + SessionState + 4-6
        recent raw -> deterministic resolver -> Turso single-phase evidence ->
        Luna ONE generation -> deterministic validation -> persist -> reply.
        """
        import time as _time

        from .tenant import resolve_tenant
        from .merchant_brain import load_merchant_brain
        from .session_state import load_state, save_state
        from .recent_messages import build_recent_window
        from .resolver import resolve_product
        from .evidence import build_evidence
        from .luna_context import build_runtime_block
        from .validation import validate_action, latinize_digits, normalize_price_decimals, merchant_vocabulary, merchant_animal, merchant_animal_emoji, neutral_masculine
        from .city_provenance import resolve_shipping_city, CURRENT_SOURCES
        from .contact import requests_whatsapp, strip_phone_attempts
        from .color_status import resolve_color
        from .order_lifecycle import apply_order_lifecycle, persistable_quantity
        from .quantity import requested_quantity
        from .reel_map import resolve_reel_product
        from . import turso as turso_client
        from . import cache as tenant_cache

        started = _time.perf_counter()
        tenant = resolve_tenant(
            store_id=store.store_id,
            merchant_account_id=store.merchant_account_id,
            customer_id=message.customer_id,
            channel=store.channel.value,
        )
        brain_in = message.store_brain if isinstance(message.store_brain, dict) else {}
        brain_record = load_merchant_brain(
            store_id=tenant.store_id,
            merchant_account_id=tenant.merchant_account_id,
            brain=brain_in,
        )
        state = load_state(
            store_id=tenant.store_id,
            channel=tenant.channel,
            customer_id=tenant.customer_id,
            active_order=message.active_order,
            known_customer=message.known_customer,
        )
        recent_fragment = build_recent_window(message.history, limit=6)
        catalogue = message.catalogue_context if isinstance(message.catalogue_context, dict) else {}
        media_ctx = message.media_context if isinstance(message.media_context, dict) else {}

        # Reel/media -> product (deterministic, store-scoped). A resolved
        # Reel seeds active_product_id so follow-ups need no product name.
        # Unknown media stays NOT_FOUND/AMBIGUOUS: never a visual guess.
        reel_status = "no_media"
        reel_owner_is_merchant = False
        try:
            reel = resolve_reel_product(
                store_id=tenant.store_id,
                attachments=message.attachments,
                media_context=media_ctx,
            )
            reel_status = reel.status
            reel_owner_is_merchant = reel.owner_is_merchant
            if reel.status == "FOUND" and reel.product_id:
                state.active_product_id = reel.product_id[:200]
                if reel.variant_id:
                    state.active_variant_id = reel.variant_id[:200]
        except Exception:
            logger.exception("Adam Luxe reel resolve failed open store=%s", tenant.store_id)

        # Post-order reset: a closed/cancelled/expired order ends its active
        # commerce session. Transactional fields (product, color, quantity,
        # city, draft, open question) must not leak into the next
        # conversation. Durable history stays adapter-side; nothing deleted.
        try:
            lifecycle = apply_order_lifecycle(state, message.active_order)
            if lifecycle.get("reset"):
                logger.info(
                    "ORDER_STATE_RESET store=%s customer=%s status=%s cleared=%s",
                    tenant.store_id, tenant.customer_id,
                    lifecycle.get("status"), ",".join(lifecycle.get("cleared") or []),
                )
        except Exception:
            logger.exception("Order lifecycle failed open store=%s", tenant.store_id)

        # Requested quantity for THIS turn (multi-intent evidence). Never
        # guessed: 0 means unstated. Persisted only when plausible — the raw
        # value still reaches Luna via turn evidence (e.g. "50k followers"
        # must never become durable quantity=50).
        try:
            turn_quantity = requested_quantity(message.text)
        except Exception:
            turn_quantity = 0
        persist_quantity = 0
        try:
            persist_quantity = persistable_quantity(turn_quantity)
        except Exception:
            persist_quantity = 0
        if persist_quantity > 0:
            state.quantity = str(persist_quantity)

        # Color confirmation for the order paper (deterministic facts only).
        # Luna stays free: evidence marks color confirmed/missing + exact
        # options; the existing soft clarification signal nudges when missing.
        # Checkout without a grounded color on a multi-color product waits.
        color_status: dict = {"status": "unknown"}
        try:
            color_status = resolve_color(
                message_text=message.text,
                history=message.history,
                catalogue=catalogue,
                stored_variant=state.active_variant_id,
            )
        except Exception:
            logger.exception("Adam Luxe color resolve failed open store=%s", tenant.store_id)
        color_confirmed = color_status.get("status") in {"confirmed", "single_option"}
        if color_confirmed and color_status.get("variant"):
            state.active_variant_id = str(color_status["variant"])[:200]
            if state.open_question == "color_choice":
                state.open_question = ""
        product_known_for_color = bool(
            state.active_product_id or catalogue.get("name") or catalogue.get("product_id"))
        if (color_status.get("status") == "missing" and product_known_for_color
                and state.open_question != "color_choice"):
            state.open_question = "color_choice"
        needs_color_clarification = (
            color_status.get("status") == "missing" and product_known_for_color)

        # WhatsApp redirect intent (deterministic). Luna replies naturally;
        # the adapter resolves the CURRENT merchant destination + button.
        wants_whatsapp = False
        try:
            wants_whatsapp = requests_whatsapp(message.text)
        except Exception:
            wants_whatsapp = False

        # Shipping-city provenance gate (Fes-bug fix): a city reaches Luna
        # ONLY when the live thread grounds it (current message, recent 4-6
        # raw turns, corroborated active order). Adapter-persisted metadata,
        # state cache and catalogue defaults without thread corroboration are
        # stale and must never surface as current facts.
        order_city = ""
        try:
            order_city = str((message.active_order or {}).get("city") or "").strip()
        except Exception:
            order_city = ""
        city_prov = resolve_shipping_city(
            message_text=message.text,
            history=message.history,
            catalogue_location=str(catalogue.get("delivery_location") or ""),
            active_order_city=order_city,
            stored_city=state.customer_city,
            stored_source=state.customer_city_source,
            stored_updated_at=state.customer_city_updated_at,
        )
        city_is_current = (
            bool(city_prov.get("value")) and city_prov.get("source") in CURRENT_SOURCES
        )
        if city_is_current:
            state.customer_city = str(city_prov["value"])[:120]
            state.customer_city_source = str(city_prov["source"])
            state.customer_city_updated_at = str(city_prov.get("updated_at") or "")
        resolved_city = state.customer_city if city_is_current else ""

        # Deterministic resolver (store-scoped, no LLM).
        media_present = reel_status not in ("", "no_media")
        resolution = resolve_product(
            store_id=tenant.store_id,
            message_text=message.text,
            state_product=state.active_product_id,
            catalogue=catalogue,
            turso_rows=[],
            aliases={},
            media_present=media_present,
        )

        # Turso single retrieval phase (0-1 round trips); fallback preserves
        # current catalogue_context behavior when Turso is not configured.
        turso_data: dict = {"product_rows": [], "shipping_rows": [], "config": {}}
        retrieval_ms = 0
        turso_used = False
        if turso_client.is_configured():
            lookup = resolution.product_id or state.active_product_id or str(catalogue.get("name") or "")
            try:
                t0 = _time.perf_counter()
                turso_data = turso_client.fetch_store_phase(
                    store_id=tenant.store_id,
                    product_lookup=lookup[:180],
                    city=resolved_city[:120],
                )
                retrieval_ms = round((_time.perf_counter() - t0) * 1000)
                turso_used = True
                # Re-resolve with exact Turso rows (canonical truth).
                if turso_data.get("product_rows"):
                    resolution = resolve_product(
                        store_id=tenant.store_id,
                        message_text=message.text,
                        state_product=state.active_product_id,
                        catalogue=catalogue,
                        turso_rows=turso_data["product_rows"],
                        aliases={},
                        media_present=media_present,
                    )
            except Exception:
                logger.exception(
                    "Adam Luxe Turso evidence failed open store=%s",
                    tenant.store_id,
                )
                turso_data = {"product_rows": [], "shipping_rows": [], "config": {}}

        evidence, evidence_fragment = build_evidence(
            resolver_status=resolution.status,
            resolver_product_id=resolution.product_id,
            turso_data=turso_data,
            catalogue=catalogue,
            resolved_city=resolved_city,
            city_source=state.customer_city_source if city_is_current else str(city_prov.get("source") or "none"),
            requested_quantity=turn_quantity,
            state_product_id=state.active_product_id,
            reel_owner_is_merchant=reel_owner_is_merchant,
            reel_status=reel_status,
            color_status=color_status,
            message_text=message.text,
        )
        # Resolved media seeds continuity without re-sending payloads.
        if reel.status == "FOUND" and reel.product_id and not state.last_media_shown:
            state.last_media_shown = reel.product_id[:200]
        state_before_luna = state.to_dict()
        # Single-representation rule: history travels ONLY as real
        # conversation messages. recent_fragment is kept for the provenance
        # log only — it is NOT embedded in the Luna-bound runtime block.
        runtime_block = build_runtime_block(
            state_fragment=state.to_luna_fragment(include_city=city_is_current),
            recent_fragment="",
            evidence_fragment=evidence_fragment,
            current_message=message.text,
        )
        try:
            logger.info(
                "SHIPPING_PROVENANCE %s",
                json.dumps({
                    "event": "SHIPPING_PROVENANCE",
                    "store_id": tenant.store_id,
                    "merchant_id": tenant.merchant_account_id,
                    "customer_id": tenant.customer_id,
                    "conversation_id": tenant.conversation_id,
                    "session_id": tenant_cache.state_key(
                        tenant.store_id, tenant.channel, tenant.customer_id),
                    "shipping_city": {
                        "value": city_prov.get("value") or "",
                        "source": city_prov.get("source") or "",
                        "updated_at": city_prov.get("updated_at") or "",
                        "corroborated_by": city_prov.get("corroborated_by") or "",
                        "dropped_value": city_prov.get("dropped_value") or "",
                        "current_for_luna": city_is_current,
                    },
                    "adapter_delivery_location": str(catalogue.get("delivery_location") or ""),
                    "adapter_delivery_price": str(catalogue.get("delivery_price") or ""),
                    "shipping_evidence": evidence,
                    "state_before_luna": state_before_luna,
                    "recent_messages_used": recent_fragment[:2400],
                    "luna_runtime_block": runtime_block[:6000],
                }, ensure_ascii=False, sort_keys=True)[:12000],
            )
        except Exception:
            logger.exception("SHIPPING_PROVENANCE log failed store=%s", tenant.store_id)
        # Grounding consistency: the providers-level money guard only sees the
        # brain + catalogue_context (facts=() on this path). Turn-specific
        # computed evidence (delivery override on free-delivery offers,
        # pre-computed total_with_delivery for 99+35=134) must therefore be
        # mirrored into the Luna-bound catalogue copy — otherwise a CORRECT
        # Luna total trips unsafe_reply before the agent-level check that
        # knows the total is grounded. Original adapter values stay in logs.
        catalogue_for_luna = dict(catalogue)
        for _ground_key in ("delivery_price", "total_with_delivery"):
            _ground_value = str(evidence.get(_ground_key) or "").strip()
            if _ground_value and _ground_value.upper() != "UNKNOWN":
                catalogue_for_luna[_ground_key] = _ground_value[:120]
        # Compact media for Luna: once resolved to product_id/media_reference,
        # only that compact reference travels — never the full prior media
        # payload on later text turns. Stale media stays out entirely.
        luna_media_context: dict = {}
        try:
            from .media import visual_attachment_urls as _visual_urls
            has_new_visual = bool(_visual_urls(message.attachments))
        except Exception:
            has_new_visual = False
        if isinstance(media_ctx, dict) and media_ctx:
            adapter_resolved = bool(str(
                media_ctx.get("product_id") or media_ctx.get("recognized_product") or ""
            ).strip())
            if has_new_visual or adapter_resolved or reel_status not in ("no_media", ""):
                luna_media_context = {
                    key: (value if isinstance(value, bool) else str(value)[:180])
                    for key in ("product_id", "recognized_product", "variant_id",
                                "media_reference", "media_id", "reel_id",
                                "resolution", "merchant_media", "video_analyzed")
                    if isinstance(media_ctx.get(key), bool) or str(media_ctx.get(key) or "").strip()
                }
            # else: stale media on a plain text turn -> send nothing.
        # Inject compact runtime (state+evidence+current; history travels ONLY
        # as real conversation messages) for this single Luna call. The full
        # versioned brain stays in store_brain (source of truth, cached).
        adapted = IncomingMessage(
            message_id=message.message_id,
            text=message.text,
            customer_id=message.customer_id,
            attachments=message.attachments,
            history=message.history,
            source=message.source,
            spoken_language=message.spoken_language,
            channel=message.channel,
            channel_account_id=message.channel_account_id,
            received_at=message.received_at,
            surface=message.surface,
            comment_id=message.comment_id,
            store_name=message.store_name,
            reply_context=message.reply_context,
            merchant_runtime_context=runtime_block[:6000],
            catalogue_context=catalogue_for_luna,
            active_order=message.active_order,
            known_customer=message.known_customer,
            store_brain={
                **brain_in,
                "content": brain_record["content"],
                "estimated_tokens": brain_record["estimated_tokens"],
            },
            media_context=luna_media_context,
        )
        script = ReplyScript.ARABIC_DARIJA
        try:
            answer = self.model.answer(
                message=adapted, script=script, facts=(), memories=(),
                store_context_required=True,
                product_clarification_required=needs_color_clarification,
            )
            raw_action = str(getattr(self.model, "last_order_action", "none") or "none").lower()
            raw_draft = getattr(self.model, "last_order_draft", {})
            raw_media_action = str(getattr(self.model, "last_media_action", "none") or "none").lower()
            raw_media_selection = getattr(self.model, "last_media_selection", {})
            order_action, order_draft, media_action = validate_action(
                reply_text=normalize_price_decimals(latinize_digits(answer)),
                order_action=raw_action,
                order_draft=raw_draft if isinstance(raw_draft, dict) else {},
                media_action=raw_media_action,
                evidence=evidence,
                resolver_status=resolution.status,
                resolver_product_id=resolution.product_id,
            )
        except RuntimeError as exc:
            # Fail safely (never HTTP 500 on a normal price/shipping turn):
            # the safety validator stays, but a rejected Luna output becomes
            # a short honest clarification — no invented amount, no second
            # Luna call, HTTP 200. Luna call count remains 1 (one generation
            # was attempted).
            if not str(exc).startswith("unsafe_reply:"):
                raise
            from .evidence import is_price_intent as _is_price, is_shipping_intent as _is_ship
            logger.warning(
                "ADAM_SAFE_FALLBACK store=%s customer=%s reason=%s resolver=%s",
                tenant.store_id, tenant.customer_id, str(exc),
                resolution.status,
            )
            if _is_ship(message.text) and str(evidence.get("delivery_price") or "") == "UNKNOWN":
                answer = "التوصيل كاين لجميع المدن، عطيني المدينة ديالك ونعطيك الثمن بالضبط."
            elif _is_price(message.text):
                answer = "الباك متوفر، قوليا شحال بغيتي (واحد ولا جوج) ونعطيك الثمن بالضبط."
            else:
                answer = "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط."
            try:
                save_state(
                    store_id=tenant.store_id,
                    channel=tenant.channel,
                    customer_id=tenant.customer_id,
                    state=state,
                )
            except Exception:
                logger.exception("Adam Luxe state persist failed open store=%s", tenant.store_id)
            return AgentReply(
                text=answer, script=script, used_rag=False,
                reason=f"safe_fallback_{str(exc).split(':', 1)[-1]}", trace_id=str(uuid.uuid4()),
                retrieval_sources=(("store_brain", "turso_evidence") if turso_used
                                   else ("store_brain", "catalogue_evidence")),
                rag_called=False, embedding_call_count=0, memory_called=False,
                llm_input_tokens=int(getattr(self.model, "last_input_tokens", 0) or 0),
                llm_output_tokens=int(getattr(self.model, "last_output_tokens", 0) or 0),
                llm_latency_ms=int(getattr(self.model, "last_llm_latency_ms", 0) or 0),
                llm_ttft_ms=None, retrieval_latency_ms=retrieval_ms,
                order_action="none", order_draft={},
                contact_action="whatsapp_redirect" if wants_whatsapp else "none",
                media_action="none", media_selection={},
                raw_model_output=str(getattr(self.model, "last_raw_model_output", "") or ""),
                raw_model_reply=str(getattr(self.model, "last_raw_model_reply", "") or ""),
                model_provider=str(getattr(self.model, "last_model_provider", "") or ""),
                requested_model=str(getattr(self.model, "last_requested_model", "") or ""),
                resolved_model=str(getattr(self.model, "last_resolved_model", "") or ""),
                store_brain_version=brain_record["version"],
                store_brain_checksum=brain_record["full_checksum"],
                store_brain_tokens=int(brain_record["estimated_tokens"] or 0),
                shipping_provenance={
                    "value": city_prov.get("value") or "",
                    "source": city_prov.get("source") or "",
                    "fresh": city_is_current,
                    "current_for_luna": city_is_current,
                    "session_id": tenant_cache.state_key(
                        tenant.store_id, tenant.channel, tenant.customer_id),
                    "conversation_id": tenant.conversation_id,
                },
            )
        if evidence.get("color_checkout_deferred"):
            logger.info(
                "COLOR_CHECKOUT_DEFERRED store=%s customer=%s available=%s",
                tenant.store_id, tenant.customer_id,
                (evidence.get("color") or {}).get("available_colors")
                if isinstance(evidence.get("color"), dict) else [],
            )
        answer = enforce_reply_script(answer, script)
        # All customer-visible numbers (prices, shipping, quantities, totals,
        # times, references) use Latin digits 0-9. Validated above post-latin.
        # Whole-dirham prices drop ".00" (99.00 DH -> 99 DH).
        answer = normalize_price_decimals(latinize_digits(answer))
        # Merchant vocabulary (gourmetta/ݣورميطة, never سوار/إسورة/bracelet).
        # Phrasing stays Luna's.
        try:
            answer = merchant_vocabulary(answer)
        except Exception:
            logger.exception("Merchant vocabulary failed open store=%s", tenant.store_id)
        # Pack animal (البط/duck, never بجعة/swan). Phrasing stays Luna's.
        try:
            answer = merchant_animal(answer)
            answer = merchant_animal_emoji(answer)
        except Exception:
            logger.exception("Merchant animal failed open store=%s", tenant.store_id)
        # Gender-neutral addressing (never nti/بغيتي; men buy for wife/mom).
        try:
            answer = neutral_masculine(answer)
        except Exception:
            logger.exception("Gender neutralize failed open store=%s", tenant.store_id)
        # WhatsApp: chat never carries numbers/links; the transport button
        # owns the CURRENT merchant destination. Strip model-written attempts.
        contact_action = "none"
        if wants_whatsapp:
            contact_action = "whatsapp_redirect"
            try:
                stripped, did_strip = strip_phone_attempts(answer)
                if did_strip:
                    logger.info(
                        "WHATSAPP_NUMBER_STRIPPED store=%s customer=%s",
                        tenant.store_id, tenant.customer_id,
                    )
                    answer = stripped
            except Exception:
                logger.exception("WhatsApp strip failed open store=%s", tenant.store_id)
        if not answer and order_action not in ORDER_UI_ACTIONS:
            raise RuntimeError("agent_core_empty_non_action_reply")

        # Persist compact state: continuity for next turn (tenant-scoped).
        try:
            if resolution.status == "FOUND" and resolution.product_id:
                state.active_product_id = resolution.product_id[:200]
            if isinstance(order_draft, dict) and str(order_draft.get("draft_order_id") or ""):
                state.draft_order_id = str(order_draft["draft_order_id"])[:120]
            save_state(
                store_id=tenant.store_id,
                channel=tenant.channel,
                customer_id=tenant.customer_id,
                state=state,
            )
        except Exception:
            logger.exception("Adam Luxe state persist failed open store=%s", tenant.store_id)

        def integer(name: str) -> int:
            value = getattr(self.model, name, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        def string(name: str) -> str:
            value = getattr(self.model, name, "")
            return value if isinstance(value, str) else ""

        sources = ("store_brain", "turso_evidence") if turso_used else ("store_brain", "catalogue_evidence")
        # Fixed-cost instrumentation (item 8/14): per-part estimates (~4
        # chars/token) plus provider-reported actuals. History appears here
        # ONCE (conversation messages only — runtime block no longer embeds
        # it). Used by the token regression test BEFORE/AFTER comparison.
        try:
            from .providers import OpenRouterLunaModel as _LunaModel
            _history_turns = _LunaModel._conversation(adapted)
            _history_chars = sum(len(str(t.get("content") or "")) for t in _history_turns)
            _history_count = len(_history_turns)
        except Exception:
            _history_chars, _history_count = 0, 0
        try:
            _token_breakdown = {
                "event": "ADAM_TOKEN_BREAKDOWN",
                "store_id": tenant.store_id,
                "brain_tokens": int(brain_record["estimated_tokens"] or 0),
                "state_tokens": len(state.to_luna_fragment(include_city=city_is_current)) // 4,
                "history_tokens": _history_chars // 4,
                "history_messages": _history_count,
                "runtime_tokens": len(runtime_block) // 4,
                "evidence_tokens": len(evidence_fragment) // 4,
                "catalogue_tokens": len(json.dumps(
                    catalogue, ensure_ascii=False, separators=(",", ":"))) // 4,
                "media_tokens": len(json.dumps(
                    luna_media_context, ensure_ascii=False, separators=(",", ":"))) // 4,
                "evidence_has_price": bool(str(evidence.get("price") or "").strip()),
                "evidence_has_shipping": bool(str(evidence.get("delivery_price") or "").strip()),
                "resolver": resolution.status,
                "llm_input_tokens": integer("last_input_tokens"),
                "llm_output_tokens": integer("last_output_tokens"),
                "llm_latency_ms": integer("last_llm_latency_ms"),
                "retrieval_latency_ms": retrieval_ms,
                "luna_calls": 1,
                "total_turn_ms": round((_time.perf_counter() - started) * 1000),
            }
            _token_breakdown["total_input_estimate"] = (
                _token_breakdown["brain_tokens"] + _token_breakdown["state_tokens"]
                + _token_breakdown["history_tokens"] + _token_breakdown["runtime_tokens"]
                + _token_breakdown["evidence_tokens"] + _token_breakdown["catalogue_tokens"]
                + _token_breakdown["media_tokens"]
            )
            logger.info("ADAM_TOKEN_BREAKDOWN %s", json.dumps(_token_breakdown, sort_keys=True))
        except Exception:
            logger.exception("ADAM_TOKEN_BREAKDOWN log failed store=%s", tenant.store_id)
        return AgentReply(
            text=answer, script=script, used_rag=False,
            reason="model_store_brain_answer", trace_id=str(uuid.uuid4()),
            retrieval_sources=sources, rag_called=False,
            embedding_call_count=0, memory_called=False,
            llm_input_tokens=integer("last_input_tokens"),
            llm_cached_input_tokens=integer("last_cached_input_tokens"),
            llm_output_tokens=integer("last_output_tokens"),
            llm_latency_ms=integer("last_llm_latency_ms"),
            llm_ttft_ms=None,
            retrieval_latency_ms=retrieval_ms,
            retrieved_chunks=({"knowledge_id": f"evidence:{tenant.store_id}:{resolution.status}",
                               "source": "turso" if turso_used else "catalogue",
                               "kind": "evidence",
                               "title": resolution.product_id[:180],
                               "score": 1.0},),
            order_action=order_action,
            order_draft=dict(order_draft) if isinstance(order_draft, dict) else {},
            contact_action=contact_action,
            media_action=media_action if media_action in {"none", "send_product_image"} else "none",
            media_selection=dict(raw_media_selection) if isinstance(raw_media_selection, dict) else {},
            raw_model_output=string("last_raw_model_output"),
            raw_model_reply=string("last_raw_model_reply"),
            model_provider=string("last_model_provider"),
            requested_model=string("last_requested_model"),
            resolved_model=string("last_resolved_model"),
            temperature=float(getattr(self.model, "last_temperature", 0.0) or 0.0),
            max_output_tokens=integer("last_max_output_tokens"),
            reasoning_effort=string("last_reasoning_effort"),
            response_format=string("last_response_format"),
            effective_prompt_sha256=string("last_effective_prompt_sha256"),
            store_brain_version=brain_record["version"],
            store_brain_checksum=brain_record["full_checksum"],
            store_brain_tokens=int(brain_record["estimated_tokens"] or 0),
            shipping_provenance={
                "value": city_prov.get("value") or "",
                "source": city_prov.get("source") or "",
                "fresh": city_is_current,
                "updated_at": city_prov.get("updated_at") or "",
                "corroborated_by": city_prov.get("corroborated_by") or "",
                "dropped_value": city_prov.get("dropped_value") or "",
                "current_for_luna": city_is_current,
                "session_id": tenant_cache.state_key(
                    tenant.store_id, tenant.channel, tenant.customer_id),
                "conversation_id": tenant.conversation_id,
            },
            conversation_memory_updates=tuple(
                vars(update) for update in getattr(self.model, "last_memory_updates", ())
                if hasattr(update, "operation")
            ),
        )
