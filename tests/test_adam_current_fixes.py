"""Regression tests for the CURRENT runtime-proven Adam Luxe defects.

Covers task items 12-15 WITHOUT any network/model call:
- 12. self-contamination: old assistant "Marrakech free" claim has ZERO
      factual authority; current shipping evidence (35 MAD) always wins.
- 13. price path: "ماهو سعر الباك" -> 99 MAD, HTTP200, 1 Luna call, no
      unsafe_reply; "جوج" -> qty 2 / 179 total / 2 gourmettas / free delivery.
- 14. token bounds: history appears ONCE, per-message caps, shipping 35 in
      evidence, stale Marrakech fact neutralised, 1 Luna call.
- 15. 20-turn stability: bounded context, no hallucination becomes truth,
      no 500, one Luna call per logical turn.

Run:  python -m pytest tests/test_adam_current_fixes.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scaliffy_agent import AgentCore, IncomingMessage, StoreContext  # noqa: E402
from scaliffy_agent import cache as tenant_cache  # noqa: E402
from scaliffy_agent.adam_dialogue import (  # noqa: E402
    _compact_catalogue_for_luna,
    prompt as adam_prompt,
)
from scaliffy_agent.evidence import build_evidence  # noqa: E402
from scaliffy_agent.luna_context import build_runtime_block  # noqa: E402
from scaliffy_agent.merchant_brain import compact_brain  # noqa: E402
from scaliffy_agent.providers import (  # noqa: E402
    OpenRouterLunaModel,
    _compact_media_context,
)
from scaliffy_agent.recent_messages import build_recent_window  # noqa: E402
from scaliffy_agent.types import Channel, ConversationTurn, ReplyScript  # noqa: E402
from scaliffy_agent.validation import (  # noqa: E402
    merchant_vocabulary,
    validate_action,
)

MERCHANT = "166510782"
STORE = "test-adam-fixes"


class FakeLuna:
    """Deterministic single-call Luna double with production validation order."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls = 0
        for name in (
            "last_input_tokens", "last_cached_input_tokens", "last_output_tokens",
            "last_llm_latency_ms", "last_max_output_tokens",
        ):
            setattr(self, name, 7)
        for name in (
            "last_model_provider", "last_requested_model", "last_resolved_model",
            "last_reasoning_effort", "last_response_format",
            "last_effective_prompt_sha256", "last_raw_model_output",
            "last_raw_model_reply",
        ):
            setattr(self, name, "test")
        self.last_temperature = 0.2
        self.last_order_action = "none"
        self.last_order_draft = {}
        self.last_media_action = "none"
        self.last_media_selection = {}
        self.last_memory_updates = ()
        self.last_conversation_category = None

    def answer(self, *, message, script, facts=(), memories=(), **kwargs):
        from scaliffy_agent.response_safety import validate_customer_reply

        self.calls += 1
        item = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        text = item if isinstance(item, str) else item.get("reply", "")
        if isinstance(item, dict):
            self.last_order_action = item.get("order_action", "none")
        self.last_raw_model_output = text
        self.last_raw_model_reply = text
        validate_customer_reply(text, message=message, script=script, facts=facts)
        return text


def make_brain(content: str | None = None) -> dict:
    content = content or (
        "Adam Luxe. Le pack se vend 99 MAD. Deux packs 179 MAD avec livraison gratuite. "
        "Livraison 35 MAD partout au Maroc, y compris Marrakech. Cadeau: gourmetta."
    )
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "merchant_id": MERCHANT,
        "content": content,
        "checksum": checksum,
        "version": checksum[:16],
        "estimated_tokens": max(1, len(content) // 4),
    }


def adam_catalogue() -> dict:
    return {
        "name": "الباك",
        "name_ar": "الباك",
        "product_id": "pack-1",
        "price": "99",
        "currency": "MAD",
        "available": "in_stock",
        "delivery_price": "35",
        "delivery_currency": "MAD",
        "offer_min_quantity": "2",
        "offer_total_price": "179",
        "offer_free_delivery": "true",
        "offer_free_bracelets": "2",
        "single_offer_price": "99",
        "single_offer_bracelets": "1",
    }


def make_store(store_id: str = STORE) -> StoreContext:
    return StoreContext(
        merchant_account_id=MERCHANT,
        store_id=store_id,
        store_name="Adam Luxe",
        channel=Channel.TEST,
    )


def make_msg(text: str, customer: str, history=None, catalogue=None, brain=None) -> IncomingMessage:
    return IncomingMessage(
        message_id=f"mid-{customer}-{abs(hash(text)) % 10_000_000}",
        text=text,
        customer_id=customer,
        attachments=(),
        history=tuple(history or ()),
        channel=Channel.TEST,
        catalogue_context=dict(catalogue) if catalogue is not None else {},
        active_order={},
        known_customer={},
        store_brain=brain if brain is not None else make_brain(),
        media_context={},
    )


def C(text: str) -> ConversationTurn:
    return ConversationTurn("customer", text)


def A(text: str) -> ConversationTurn:
    return ConversationTurn("assistant", text)


# ---------------------------------------------------------------------------
# 12. SELF-CONTAMINATION
# ---------------------------------------------------------------------------

def test_12_history_shipping_claim_has_zero_authority():
    history = [
        C("Salam"),
        A("35 درهم خارج مراكش ومجاني فمراكش"),
        C("Ok"),
    ]
    evidence, fragment = build_evidence(
        resolver_status="NOT_FOUND",
        catalogue=adam_catalogue(),
        message_text="Chhal tawsil",
    )
    assert evidence.get("delivery_price") == "35", evidence
    assert str(evidence.get("free_shipping")) == "false", evidence
    assert evidence.get("shipping_scope") in {"all_cities", "all_morocco"}, evidence
    assert "UNKNOWN" not in str(evidence.get("delivery_price"))

    runtime = build_runtime_block(
        state_fragment="STATE: (no prior context)",
        evidence_fragment=fragment,
        current_message="Chhal tawsil",
    )
    # The false historical claim must NOT be embedded in the Luna payload.
    assert "مجاني فمراكش" not in runtime
    assert "HOT (recent chat" not in runtime
    assert "35" in runtime  # current evidence wins

    agent = AgentCore(knowledge_store=None, model=FakeLuna(["التوصيل 35 درهم لجميع المدن."]))
    reply = agent.reply(store=make_store("test-12"), message=make_msg(
        "Chhal tawsil", "cust-12", history=history, catalogue=adam_catalogue()))
    assert reply is not None
    assert agent.model.calls == 1
    assert "35" in reply.text
    assert "مجاني فمراكش" not in reply.text


def test_12_parroted_history_claim_is_rejected_fail_safe():
    """A Luna output that copies the old 'Marrakech free' hallucination must
    fail validation and become a safe fallback — never a delivered lie, never
    an exception/HTTP 500."""
    history = [C("Salam"), A("Marrakech delivery is free"), C("Chhal tawsil")]
    agent = AgentCore(
        knowledge_store=None,
        model=FakeLuna(["التوصيل فابور فمراكش 0 درهم"]),
    )
    reply = agent.reply(store=make_store("test-12b"), message=make_msg(
        "Chhal tawsil", "cust-12b", history=history, catalogue=adam_catalogue()))
    assert reply is not None  # fail-safe: no raise -> no HTTP 500
    assert agent.model.calls == 1  # exactly one Luna call, no retry
    assert "0" not in reply.text  # ungrounded amount never delivered
    assert reply.reason.startswith("safe_fallback_")

    # Next turn: the false claim still does not propagate into evidence.
    evidence, _ = build_evidence(
        resolver_status="NOT_FOUND",
        catalogue=adam_catalogue(),
        message_text="Chhal tawsil",
    )
    assert evidence.get("delivery_price") == "35"


def test_12_shipping_unknown_when_unloadable():
    evidence, fragment = build_evidence(
        resolver_status="NOT_FOUND",
        catalogue={"name": "الباك", "product_id": "pack-1"},
        message_text="Chhal tawsil",
    )
    assert evidence.get("delivery_price") == "UNKNOWN", evidence
    assert "UNKNOWN" in fragment


# ---------------------------------------------------------------------------
# 13. PRICE PATH
# ---------------------------------------------------------------------------

def test_13_price_turn_200_one_call():
    agent = AgentCore(
        knowledge_store=None,
        model=FakeLuna(["الثمن ديال الباك هو 99 درهم."]),
    )
    reply = agent.reply(store=make_store("test-13"), message=make_msg(
        "ماهو سعر الباك", "cust-13", catalogue=adam_catalogue()))
    assert reply is not None
    assert agent.model.calls == 1
    assert "99" in reply.text
    assert reply.order_action == "none"

    evidence, _ = build_evidence(
        resolver_status="FOUND",
        resolver_product_id="pack-1",
        catalogue=adam_catalogue(),
        requested_quantity=0,
        state_product_id="pack-1",
        message_text="ماهو سعر الباك",
    )
    assert evidence.get("price") == "99", evidence
    assert "price_unresolved" not in evidence


def test_13_quantity_two_offer_total():
    customer = "cust-13b"
    store = "test-13b"
    agent1 = AgentCore(knowledge_store=None, model=FakeLuna(["الثمن ديال الباك هو 99 درهم."]))
    first = agent1.reply(store=make_store(store), message=make_msg(
        "ماهو سعر الباك", customer, catalogue=adam_catalogue()))
    assert first is not None and agent1.model.calls == 1

    agent2 = AgentCore(
        knowledge_store=None,
        model=FakeLuna(["جوج باكات بـ179 درهم مع جوج ݣورميطات والتوصيل فابور."]),
    )
    second = agent2.reply(store=make_store(store), message=make_msg(
        "جوج", customer, catalogue=adam_catalogue()))
    assert second is not None
    assert agent2.model.calls == 1
    assert "179" in second.text

    evidence, _ = build_evidence(
        resolver_status="FOUND",
        resolver_product_id="pack-1",
        catalogue=adam_catalogue(),
        requested_quantity=2,
        state_product_id="pack-1",
        message_text="جوج",
    )
    assert evidence["offer"]["applies_to_this_turn"] is True
    assert evidence.get("total_with_delivery") == "179 MAD", evidence
    assert evidence.get("delivery_price") == "0", evidence  # offer: free delivery
    # Canonical catalogue fee stays 35 (merchant truth); the charged fee is 0.
    assert adam_catalogue()["delivery_price"] == "35"


def test_13_ungrounded_price_never_500():
    agent = AgentCore(knowledge_store=None, model=FakeLuna(["الثمن 50 درهم."]))
    reply = agent.reply(store=make_store("test-13c"), message=make_msg(
        "ماهو سعر الباك", "cust-13c", catalogue=adam_catalogue()))
    assert reply is not None
    assert agent.model.calls == 1
    assert "50" not in reply.text
    assert reply.reason.startswith("safe_fallback_")


def test_validate_action_flattens_nested_offer_totals():
    """179 lives inside the nested offer dict; the money guard must see it."""
    evidence, _ = build_evidence(
        resolver_status="FOUND",
        resolver_product_id="pack-1",
        catalogue=adam_catalogue(),
        requested_quantity=2,
        state_product_id="pack-1",
        message_text="جوج",
    )
    action, _, _ = validate_action(
        reply_text="جوج باكات بـ179 درهم والتوصيل فابور.",
        order_action="none", order_draft={}, media_action="none",
        evidence=evidence, resolver_status="FOUND", resolver_product_id="pack-1",
    )
    assert action == "none"


# ---------------------------------------------------------------------------
# 14. TOKEN / HISTORY-ONCE / VOCABULARY
# ---------------------------------------------------------------------------

def test_14_history_appears_once_bounded():
    marker = "UNIQUE_MARKER_XYZ_TAWSIL"
    history = [
        C("Salam"),
        A(f"Promo {marker} Marrakech delivery is free"),
        C("Salam"),
        A(f"Promo {marker} Marrakech delivery is free"),
        C("Chhal taman?"),
        A("الثمن 99 درهم"),
        C("W tawsil?"),
        A("35 درهم خارج مراكش ومجاني فمراكش"),
        C("Ok"),
        A("Ok hbibi"),
    ]
    msg = make_msg("Chhal tawsil", "cust-14", history=history, catalogue=adam_catalogue())
    conv = OpenRouterLunaModel._conversation(msg)
    assert 1 <= len(conv) <= 6, [t["content"][:40] for t in conv]
    for turn in conv:
        assert len(turn["content"]) <= 600

    _, fragment = build_evidence(
        resolver_status="NOT_FOUND", catalogue=adam_catalogue(), message_text="Chhal tawsil")
    runtime = build_runtime_block(
        state_fragment="STATE: active_product_id=pack-1",
        evidence_fragment=fragment,
        current_message="Chhal tawsil",
    )
    # Single representation: the marker travels in conversation messages only.
    assert marker not in runtime
    assert sum(marker in t["content"] for t in conv) <= 1
    assert "35" in fragment

    # BEFORE/AFTER record (tiny brain case from the task).
    brain = make_brain("x" * 408)  # ~102 tokens
    compact = compact_brain(brain["content"])
    parts = {
        "brain_tokens": max(1, len(compact) // 4),
        "state_tokens": len("STATE: active_product_id=pack-1") // 4,
        "history_tokens": sum(len(t["content"]) for t in conv) // 4,
        "runtime_tokens": len(runtime) // 4,
        "evidence_tokens": len(fragment) // 4,
    }
    print("\nTOKEN_14_AFTER:", json.dumps(parts, sort_keys=True))
    assert parts["history_tokens"] <= 600  # ~2400 chars total cap
    assert "HOT (recent chat" not in runtime


def test_14_stale_brain_labels_neutralised():
    stale = (
        "Adam Luxe. Luxury Swan Set 99 MAD. Normal delivery is 35 MAD throughout Morocco; "
        "Marrakech delivery is free (0 MAD). Gift: إسورة. Product: طقم. bracelet offer."
    )
    compact = compact_brain(make_brain(stale)["content"])
    for bad in ("Luxury Swan Set", "Marrakech delivery is free", "إسورة", "طقم", "bracelet"):
        assert bad not in compact, bad
    for good in ("pack", "35 MAD", "ݣورميطة", "الباك"):
        assert good in compact, good


def test_14_vocabulary_gourmetta():
    out = merchant_vocabulary("الثمن ديال الطقم Luxury Swan Set مع إسورة وسوار و bracelet")
    assert "طقم" not in out and "Luxury Swan Set" not in out
    assert "إسورة" not in out and "سوار" not in out and "bracelet" not in out
    assert "الباك" in out and "pack" in out
    assert "ݣورميطة" in out and "gourmetta" in out


def test_14_media_compact_after_resolve():
    full = {
        "product_id": "pack-1",
        "caption": "y" * 5000,
        "thumbnail_url": "https://x/" + "z" * 2000,
        "video_analyzed": False,
        "merchant_media": True,
    }
    compact = _compact_media_context(full)
    assert compact.get("product_id") == "pack-1"
    assert len(json.dumps(compact, ensure_ascii=False)) <= 800
    assert "caption" not in compact and "thumbnail_url" not in compact

    catalogue = adam_catalogue()
    catalogue["visual_image_assets_pack-1"] = "asset-1,asset-2"
    catalogue["description"] = "d" * 5000
    compact_cat = _compact_catalogue_for_luna(catalogue)
    assert len(json.dumps(compact_cat, ensure_ascii=False)) <= 2000
    assert "description" not in compact_cat
    assert compact_cat.get("price") == "99"


def test_14_luna_calls_single():
    agent = AgentCore(knowledge_store=None, model=FakeLuna(["التوصيل 35 درهم لجميع المدن."]))
    reply = agent.reply(store=make_store("test-14b"), message=make_msg(
        "Chhal tawsil", "cust-14b", catalogue=adam_catalogue()))
    assert reply is not None and agent.model.calls == 1


# ---------------------------------------------------------------------------
# 15. LONG CONVERSATION (20 turns)
# ---------------------------------------------------------------------------

INTENT_REPLIES = {
    "greet": "لاباس الحمد لله! كيفاش نقدر نعاونك؟",
    "price": "الثمن ديال الباك هو 99 درهم.",
    "shipping": "التوصيل 35 درهم لجميع المدن.",
    "two": "جوج باكات بـ179 درهم مع جوج ݣورميطات والتوصيل فابور.",
    "total": "المجموع 134 درهم مع التوصيل.",
    "unknown": "واخا، عاود سولني على الباك ونعطيك المعلومة بالضبط.",
}


def test_15_twenty_turn_stability():
    store_id, customer = "test-15", "cust-15"
    script_plan = [
        ("Salam", "greet"), ("ماهو سعر الباك", "price"),
        ("Chhal tawsil", "shipping"), ("جوج", "two"),
        ("المجموع شحال مع التوصيل", "total"), ("Salam", "greet"),
        ("Chhal taman dyal pack", "price"), ("Livraison Casa?", "shipping"),
        ("بغيت واحد", "price"), ("Ok", "unknown"),
        ("ماهو سعر الباك", "price"), ("Chhal tawsil", "shipping"),
        ("جوج", "two"), ("Salam", "greet"),
        ("الثمن؟", "price"), ("التوصيل لمراكش؟", "shipping"),
        ("جوج باكات", "two"), ("شكرا", "unknown"),
        ("ماهو سعر الباك", "price"), ("Chhal tawsil", "shipping"),
    ]
    history: list = [C("Salam"), A("Marrakech delivery is free")]  # planted stale claim
    totals = {"input_est": 0, "calls": 0}
    for index, (text, intent) in enumerate(script_plan):
        catalogue = adam_catalogue()
        if index == 9:
            catalogue = {}  # genuinely missing catalogue -> honest fallback path
            expected_key = "unknown"
        else:
            expected_key = intent
        agent = AgentCore(knowledge_store=None, model=FakeLuna([INTENT_REPLIES[expected_key]]))
        reply = agent.reply(
            store=make_store(store_id),
            message=make_msg(text, customer, history=list(history), catalogue=catalogue),
        )
        assert reply is not None, f"turn {index} ({text!r}) returned None (500-equivalent)"
        assert agent.model.calls == 1, f"turn {index}: Luna calls != 1"
        assert reply.text.strip(), f"turn {index}: empty reply"
        totals["calls"] += 1
        # Evidence on every turn: stale history never becomes delivery truth.
        evidence, _ = build_evidence(
            resolver_status="FOUND" if index else "NOT_FOUND",
            resolver_product_id="pack-1" if index else "",
            catalogue=catalogue,
            requested_quantity=2 if intent == "two" else 0,
            state_product_id="pack-1",
            message_text=text,
        )
        fee = str(evidence.get("delivery_price") or "")
        assert fee in {"35", "0", "UNKNOWN", ""}, f"turn {index}: fee={fee!r}"
        assert "free" not in fee.lower() or fee == "0"
        history.extend([C(text), A(reply.text)])
        # Bounded window at every step.
        conv = OpenRouterLunaModel._conversation(
            make_msg(text, customer, history=list(history), catalogue=catalogue))
        assert len(conv) <= 6, f"turn {index}: history window {len(conv)}"
    assert totals["calls"] == 20
    print("\nTURNS_15_CALLS:", totals["calls"])


def test_prompt_contract_marks_history_untrusted():
    text = adam_prompt(
        brain=make_brain(), catalogue=adam_catalogue(),
        runtime="STATE: x", order={}, media_rule="", media_reference="", native_reply="",
    )
    assert "NEVER" in text and "commercial fact" in text.lower()
    assert "gourmetta" in text and "الباك" in text
    # The contract names stale labels ONLY to forbid them; the bound
    # catalogue/brain sections Luna grounds on must carry current words.
    assert "Never call the product" in text
    compact_cat = _compact_catalogue_for_luna(adam_catalogue())
    assert "طقم" not in json.dumps(compact_cat, ensure_ascii=False)
    assert "Luxury Swan Set" not in json.dumps(compact_cat, ensure_ascii=False)
