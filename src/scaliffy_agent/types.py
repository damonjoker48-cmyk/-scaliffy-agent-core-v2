from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class ReplyScript(StrEnum):
    LATIN_DARIJA = "latin_darija"
    ARABIC_DARIJA = "arabic_darija"


class Channel(StrEnum):
    INSTAGRAM = "instagram"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    TEST = "test"


class ConversationSurface(StrEnum):
    DIRECT_MESSAGE = "direct_message"
    INSTAGRAM_COMMENT = "instagram_comment"


class ConversationCategory(StrEnum):
    ORDER = "ORDER"
    DELIVERY = "DELIVERY"
    PAYMENT = "PAYMENT"
    RETURN_REFUND_EXCHANGE = "RETURN_REFUND_EXCHANGE"
    SAV_SUPPORT = "SAV_SUPPORT"
    PRODUCT_FACTS = "PRODUCT_FACTS"
    PRODUCT_DISCOVERY = "PRODUCT_DISCOVERY"
    PURCHASE_INTENT = "PURCHASE_INTENT"
    RELATIONSHIP_USEFUL = "RELATIONSHIP_USEFUL"
    SOCIAL_NEUTRAL = "SOCIAL_NEUTRAL"
    UNRELATED = "UNRELATED"
    GENERAL_ASSISTANT_USE = "GENERAL_ASSISTANT_USE"
    REPEATED_OFFTOPIC = "REPEATED_OFFTOPIC"


NON_PENALIZED_CATEGORIES = frozenset({
    ConversationCategory.ORDER,
    ConversationCategory.DELIVERY,
    ConversationCategory.PAYMENT,
    ConversationCategory.RETURN_REFUND_EXCHANGE,
    ConversationCategory.SAV_SUPPORT,
    ConversationCategory.PRODUCT_FACTS,
    ConversationCategory.PRODUCT_DISCOVERY,
    ConversationCategory.PURCHASE_INTENT,
})


_CONVERSATION_CATEGORY_COST = {
    ConversationCategory.RELATIONSHIP_USEFUL: 0.25,
    ConversationCategory.SOCIAL_NEUTRAL: 0.5,
    ConversationCategory.UNRELATED: 1.0,
    ConversationCategory.GENERAL_ASSISTANT_USE: 2.0,
    ConversationCategory.REPEATED_OFFTOPIC: 2.0,
}


@dataclass(frozen=True)
class ConversationBudget:
    """Durable per-customer social-conversation state, never shown to customers."""

    score: float = 0.0
    social_cooldown: bool = False
    exit_sent: bool = False
    off_topic_streak: int = 0

    def apply(self, category: ConversationCategory, *, threshold: float) -> "ConversationBudget":
        if category in NON_PENALIZED_CATEGORIES:
            # A real customer need is always allowed. It clears only the
            # off-topic streak; the accumulated social budget stays intact.
            return ConversationBudget(
                score=self.score,
                social_cooldown=self.social_cooldown,
                exit_sent=self.exit_sent,
                off_topic_streak=0,
            )
        cost = _CONVERSATION_CATEGORY_COST[category]
        streak = self.off_topic_streak + 1 if category in {
            ConversationCategory.UNRELATED,
            ConversationCategory.GENERAL_ASSISTANT_USE,
            ConversationCategory.REPEATED_OFFTOPIC,
        } else 0
        score = round(self.score + cost, 2)
        crossed = not self.exit_sent and score >= threshold
        return ConversationBudget(
            score=score,
            social_cooldown=self.social_cooldown or crossed,
            exit_sent=self.exit_sent or crossed,
            off_topic_streak=streak,
        )


@dataclass(frozen=True)
class Attachment:
    url: str
    mime_type: str
    caption: str = ""
    media_id: str = ""


@dataclass(frozen=True)
class StoreContext:
    merchant_account_id: str
    store_id: str
    store_name: str
    channel: Channel = Channel.TEST
    agent_enabled: bool = True
    human_takeover: bool = False


@dataclass(frozen=True)
class ConversationTurn:
    role: Literal["customer", "assistant", "human"]
    text: str
    # A prior voice transcript must not silently override the script chosen by
    # the customer's first actual written message.
    source: Literal["text", "instagram_voice"] = "text"


@dataclass(frozen=True)
class IncomingMessage:
    message_id: str
    text: str
    customer_id: str
    attachments: tuple[Attachment, ...] = ()
    history: tuple[ConversationTurn, ...] = ()
    source: Literal["text", "instagram_voice"] = "text"
    spoken_language: str = ""
    channel: Channel = Channel.TEST
    channel_account_id: str = ""
    received_at: str = ""
    surface: ConversationSurface = ConversationSurface.DIRECT_MESSAGE
    comment_id: str = ""
    # Canonical merchant identity resolved by the authenticated Scaliffy
    # adapter.  It is data, never a tenant-authored instruction.
    store_name: str = ""
    # Exact native channel reply reference, resolved by Scaliffy's data plane
    # before the single Core call.  An unresolved reference intentionally
    # contains no guessed content and therefore fails open.
    reply_context: dict[str, object] = field(default_factory=dict)
    # Merchant-owned identity, tone and business instructions are runtime
    # data supplied by the Scaliffy adapter. They never create a
    # merchant-specific Core implementation.
    merchant_runtime_context: str = ""
    # Exact product fact resolved by Scaliffy's authenticated, tenant-scoped
    # catalogue adapter for the current turn. It is deliberately transient:
    # no stale product is inferred from conversation history.
    catalogue_context: dict[str, str] = field(default_factory=dict)
    # Structured commerce state supplied by the channel adapter.  This is
    # runtime customer/order data, never merchant prompt policy.
    active_order: dict[str, str] = field(default_factory=dict)
    known_customer: dict[str, str] = field(default_factory=dict)
    # Versioned, complete merchant facts supplied by the trusted Scaliffy
    # adapter. Only Adam Luxe currently uses this retrieval-free path.
    store_brain: dict[str, object] = field(default_factory=dict)
    media_context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    store_id: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeHit:
    document: KnowledgeDocument
    score: float


@dataclass(frozen=True)
class RetrievalPlan:
    required: bool
    reason: str
    query: str = ""
    query_embedding: tuple[float, ...] = ()
    # Semantic data-plane preference inferred by the existing embedding router.
    # It is a ranking hint, never a hard filter: older merchant snapshots may
    # still contain a useful fact under a less specific document kind.
    preferred_knowledge_kinds: tuple[str, ...] = ()
    # A merchant fact is requested, but no exact product can be resolved from
    # the current exchange. Ask for that identifier before retrieving a list.
    product_clarification_required: bool = False
    # Only evaluated while the durable social cooldown is active.
    cooldown_gate: Literal["commercial", "blocked", "ambiguous"] = "ambiguous"
    memory_required: bool = False
    memory_reason: str = ""
    # Public comments are an acquisition surface, not a support conversation:
    # route a relevant store enquiry to private messages or stay silent.
    comment_action: Literal["ignore", "move_to_private"] = "ignore"


@dataclass(frozen=True)
class MemoryUpdate:
    operation: Literal["set", "remove"]
    category: Literal["preference", "commercial_fact", "open_thread"]
    key: str
    value: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ConversationDecision:
    needs_store_context: bool
    reply: str = ""
    retrieval_query: str = ""


@dataclass(frozen=True)
class AgentReply:
    text: str
    script: ReplyScript
    used_rag: bool
    reason: str
    trace_id: str
    knowledge_ids: tuple[str, ...] = ()
    retrieval_namespace: str = ""
    retrieval_filter: dict[str, str] = field(default_factory=dict)
    retrieval_sources: tuple[str, ...] = ()
    retrieved_chunks: tuple[dict[str, str | float], ...] = ()
    rag_called: bool = False
    luna_call_count: int = 1
    embedding_call_count: int = 0
    llm_input_tokens: int = 0
    llm_cached_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_latency_ms: int = 0
    llm_ttft_ms: int | None = None
    retrieval_latency_ms: int = 0
    retrieval_store_score: float = 0.0
    retrieval_other_score: float = 0.0
    memory_called: bool = False
    memory_updates_count: int = 0
    memory_relevance_score: float = 0.0
    recommendation_score: float = 0.0
    conversation_category: str = ""
    social_cooldown_activated: bool = False
    # A generic side-effect contract consumed by Scaliffy after the single
    # Luna turn.  The Core never writes orders or calls channel APIs itself.
    order_action: Literal[
        "none", "start_order", "start_new_order", "resend_order_form",
        "confirm", "refuse", "modify",
    ] = "none"
    order_draft: dict[str, str | bool] = field(default_factory=dict)
    # The shared Core may request a catalogue image, but the channel adapter
    # validates these ids against its tenant-scoped canonical catalogue before
    # sending anything.
    media_action: Literal["none", "send_product_image"] = "none"
    media_selection: dict[str, str] = field(default_factory=dict)
    # Safe generation audit metadata. Raw output is the model's response
    # before parser/language cleanup; it never contains credentials.
    raw_model_output: str = ""
    raw_model_reply: str = ""
    model_provider: str = ""
    requested_model: str = ""
    resolved_model: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 0
    reasoning_effort: str = ""
    response_format: str = ""
    effective_prompt_sha256: str = ""
    store_brain_version: str = ""
    store_brain_checksum: str = ""
    store_brain_tokens: int = 0
    conversation_memory_updates: tuple[dict[str, object], ...] = ()
    # Shipping-city provenance for the Fes-bug trace: value/source/updated_at
    # plus the evidence/state/recent snapshots that produced the answer.
    shipping_provenance: dict[str, object] = field(default_factory=dict)
    # WhatsApp redirect: Luna replies naturally; the adapter resolves the
    # CURRENT merchant destination and renders the clickable button.
    contact_action: str = "none"
    # Core version trace: production paths report "v1"; the isolated test
    # engine for store 625374849 reports "v2_test". Never inferred.
    agent_core_version: str = "v1"


@dataclass(frozen=True)
class VoiceTranscription:
    transcript_original: str
    spoken_language: str = ""
    duration_seconds: float = 0.0
    model: str = "openai/gpt-transcribe"
    provider_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    media_download_ms: int = 0
    transcription_ms: int = 0
    completed: bool = False
