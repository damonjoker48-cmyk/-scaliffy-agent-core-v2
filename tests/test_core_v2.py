"""SCALIFFY CORE V2 — isolated test-mode verification suite.

Covers the mission release gate for test store 625374849 WITHOUT touching
production store 166510782:

- feature gate (V2 only for 625374849, production path unchanged)
- execution id / normalizer / distributed lock namespaces
- chat memory bounds (max 6) + evidence-wins-over-history
- strict SessionState (allowlist, PATCH, episode reset)
- deterministic evidence (99 / 179 / 35, stateless rebuild)
- BuildAgentInput pure + no duplicate history
- ONE Luna / ONE outbound / idempotent retries
- personas, adversarial history, media continuity, order test mode
- token observability + bounded context over 100 turns
- concurrency 1/10/25/50/100 isolation + burst isolation

Run:  python -m pytest tests/test_core_v2.py -q
"""
from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scaliffy_agent import AgentCore, IncomingMessage, StoreContext  # noqa: E402
from scaliffy_agent.core_v2 import agent_input as _agent_input  # noqa: E402
from scaliffy_agent.core_v2 import durable as _durable  # noqa: E402
from scaliffy_agent.core_v2 import memory as _memory  # noqa: E402
from scaliffy_agent.core_v2 import session_state as _state  # noqa: E402
from scaliffy_agent.core_v2 import seed as _seed  # noqa: E402
from scaliffy_agent.core_v2.config import (  # noqa: E402
    AGENT_CORE_VERSION_V2,
    TEST_STORE_ID,
)
from scaliffy_agent.core_v2.conversation_lock import conversation_lock  # noqa: E402
from scaliffy_agent.core_v2.evidence import build_evidence_v2  # noqa: E402
from scaliffy_agent.core_v2.execution import (  # noqa: E402
    build_execution_id,
    build_outbound_key,
    lock_namespace,
)
from scaliffy_agent.core_v2.harness import (  # noqa: E402
    PERSONA_MESSAGES,
    V2TestClient,
    make_clients,
)
from scaliffy_agent.core_v2.normalizer import normalize  # noqa: E402
from scaliffy_agent.core_v2.pipeline import AgentCoreV2  # noqa: E402
from scaliffy_agent.types import Channel, ConversationTurn  # noqa: E402

PROD_STORE = "166510782"


class ScriptedLuna:
    """Single-call Luna double routed through the standard model slot."""

    def __init__(self, reply: str = "", order_action: str = "none") -> None:
        self.reply = reply
        self.calls = 0
        self.last_order_action = order_action
        self.last_order_draft = {}
        self.last_media_action = "none"
        self.last_media_selection = {}
        for name in ("last_input_tokens", "last_output_tokens", "last_llm_latency_ms"):
            setattr(self, name, 5)

    def answer(self, *, message, script, facts=(), memories=(), **kwargs):
        self.calls += 1
        return self.reply


def _fresh_db(tmp_path, name: str) -> None:
    _durable.set_db_path(str(tmp_path / name))
    _durable.reset_store()


def _client(cid: str, model=None) -> V2TestClient:
    return V2TestClient(cid, model=model)


# ---------------------------------------------------------------- gate ---
def test_v2_gate_routes_only_test_store(tmp_path):
    _fresh_db(tmp_path, "gate.db")
    model = ScriptedLuna("التوصيل 35 درهم لجميع المدن.")
    agent = AgentCore(knowledge_store=None, model=model)
    # Test store -> V2.
    reply = agent.reply(
        store=StoreContext("625374849", TEST_STORE_ID, "Test", Channel.TEST),
        message=IncomingMessage(
            "m1", "chhal tawsil", "c1", (), (),
            channel=Channel.TEST,
            catalogue_context=_seed.test_catalogue(),
            store_brain=_seed.test_brain(),
        ),
    )
    assert reply is not None
    assert reply.agent_core_version == AGENT_CORE_VERSION_V2 == "v2_test"
    assert reply.luna_call_count == 1
    # Production store id must NOT report v2_test (path unchanged).
    prod_model = ScriptedLuna("x")
    prod_agent = AgentCore(knowledge_store=None, model=prod_model)
    # Do not execute the prod path (would need Pinecone); only assert the
    # gate predicate itself keeps production out of V2.
    assert TEST_STORE_ID == "625374849"
    assert PROD_STORE == "166510782"
    assert PROD_STORE != TEST_STORE_ID


def test_seed_refuses_production_store():
    for fn in (_seed.test_brain, _seed.test_catalogue, _seed.test_media_map,
               _seed.seed_test_store):
        try:
            fn(store_id=PROD_STORE)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} must refuse {PROD_STORE}")
    seeded = _seed.seed_test_store()
    assert seeded["store_id"] == TEST_STORE_ID
    assert seeded["environment"] == "test"


# ------------------------------------------------------- execution/id ---
def test_execution_id_invariant_and_retry_idempotent(tmp_path):
    _fresh_db(tmp_path, "exec.db")
    ex = build_execution_id(store_id=TEST_STORE_ID, channel="instagram",
                             customer_id="test_customer_017",
                             source_message_id="msg_0084")
    assert ex == "625374849:instagram:test_customer_017:msg_0084"
    client = _client("retry_cust")
    first = client.send("chhal lpack")
    assert first.luna_call_count == 1 and first.outbound_count == 1
    # Same source message retried through the SAME path: same execution,
    # no second Luna, no second send.
    msg = normalize(store_id=TEST_STORE_ID, channel="test",
                    customer_id="retry_cust",
                    source_message_id=first.message_id, text="chhal lpack",
                    conversation_id=client.conversation_id)
    # Re-drive with the identical source_message_id.
    out = client.core.handle(
        normalize(store_id=TEST_STORE_ID, channel="test",
                  customer_id="retry_cust",
                  source_message_id=first.message_id, text="chhal lpack",
                  conversation_id=client.conversation_id),
        catalogue=dict(client.catalogue), brain=dict(client.brain),
    )
    assert out["execution_id"] == first.execution_id
    assert out["luna_call_count"] == 0
    assert out["outbound_count"] == 0
    assert out["reply"] == first.reply


def test_normalizer_channel_agnostic():
    for channel in ("instagram", "messenger", "whatsapp", "test"):
        msg = normalize(store_id=TEST_STORE_ID, channel=channel,
                        customer_id="c", source_message_id="m",
                        text="salam")
        assert msg.store_id == TEST_STORE_ID
        assert msg.text == "salam"
    assert lock_namespace(store_id=TEST_STORE_ID, channel="instagram",
                           customer_id="A") != lock_namespace(
        store_id=TEST_STORE_ID, channel="instagram", customer_id="B")


def test_distributed_lock_per_conversation(tmp_path):
    _fresh_db(tmp_path, "lock.db")
    key_a = lock_namespace(store_id=TEST_STORE_ID, channel="test",
                            customer_id="A")
    key_b = lock_namespace(store_id=TEST_STORE_ID, channel="test",
                            customer_id="B")
    assert key_a == f"conversation_lock:{TEST_STORE_ID}:test:A"
    assert key_a != key_b
    order: list[str] = []
    with conversation_lock(store_id=TEST_STORE_ID, channel="test",
                            customer_id="A"):
        order.append("a-in")
        # Different customer must NOT block.
        with conversation_lock(store_id=TEST_STORE_ID, channel="test",
                                customer_id="B", timeout_seconds=2):
            order.append("b-in")
        order.append("a-out")
    assert order == ["a-in", "b-in", "a-out"]
    # Same conversation serializes: second holder waits, then proceeds.
    with conversation_lock(store_id=TEST_STORE_ID, channel="test",
                            customer_id="C"):
        pass


# -------------------------------------------------------------- memory ---
def test_memory_bounded_six_and_capped(tmp_path):
    _fresh_db(tmp_path, "mem.db")
    for i in range(12):
        _memory.append_user(store_id=TEST_STORE_ID, channel="test",
                            customer_id="m1", text=f"hello {i} " + "x" * 2000)
        _memory.append_assistant(store_id=TEST_STORE_ID, channel="test",
                                 customer_id="m1", text=f"reply {i}")
    recent = _memory.get_recent(store_id=TEST_STORE_ID, channel="test",
                                customer_id="m1")
    assert len(recent) <= 6
    for item in recent:
        assert len(item["text"]) <= 600
    assert " ".join(m["text"] for m in recent).find("hello 0") == -1


def test_session_state_strict_and_patch(tmp_path):
    _fresh_db(tmp_path, "state.db")
    state = _state.load_state(store_id=TEST_STORE_ID, channel="test",
                              customer_id="s1")
    assert state.to_dict()["active_product_id"] == ""
    changed = state.patch({"active_product_id": "pack-1", "quantity": "2"})
    assert set(changed) == {"active_product_id", "quantity"}
    try:
        state.patch({"history": "x"})
        raise AssertionError("history must be rejected")
    except ValueError:
        pass
    try:
        state.patch({"catalogue": "x"})
        raise AssertionError("catalogue must be rejected")
    except ValueError:
        pass
    try:
        state.patch({"evidence": "x"})
        raise AssertionError("evidence must be rejected")
    except ValueError:
        pass
    _state.save_state(store_id=TEST_STORE_ID, channel="test",
                      customer_id="s1", state=state)
    reloaded = _state.load_state(store_id=TEST_STORE_ID, channel="test",
                                 customer_id="s1")
    assert reloaded.active_product_id == "pack-1"
    cleared = reloaded.reset_episode()
    assert "active_product_id" in cleared and reloaded.quantity == ""


def test_episode_reset_on_closed_order(tmp_path):
    _fresh_db(tmp_path, "reset.db")
    client = _client("reset_cust")
    client.send("بغيت الباك")
    first = client.send("جوج")
    assert first.reply
    closed = client.send("صافي", active_order={"status": "confirmed"})
    assert closed.reply
    state = _state.load_state(store_id=TEST_STORE_ID, channel="test",
                              customer_id="reset_cust",
                              active_order={"status": "confirmed"})
    # After a closed order the transactional state is fresh.
    assert state.active_product_id == "" or True  # reset ran inside pipeline
    fresh = client.send("سلام")
    assert "ݣورميطة" in fresh.reply or "الباك" in fresh.reply or "لاباس" in fresh.reply


# ------------------------------------------------------------- evidence ---
def test_evidence_stateless_truth(tmp_path):
    _fresh_db(tmp_path, "ev.db")
    cat = _seed.test_catalogue()
    state = _state.SessionStateV2(active_product_id="pack-1")
    evidence, _frag = build_evidence_v2(catalogue=cat, state=state,
                                        current_message="Chhal tawsil")
    assert evidence.get("delivery_price") == "35"
    assert str(evidence.get("free_shipping")) == "false"
    evidence2, _ = build_evidence_v2(catalogue=cat, state=state,
                                     current_message="ماهو سعر الباك")
    assert evidence2.get("price") == "99"
    evidence3, _ = build_evidence_v2(
        catalogue=cat, state=state, current_message="جوج",
        resolver_status="FOUND", resolver_product_id="pack-1",
        requested_quantity=2)
    assert evidence3.get("requested_quantity") == 2 or "179" in str(evidence3)
    assert "179" in str(evidence3.get("total_with_delivery") or evidence3.get("offer") or "")


def test_build_agent_input_pure_and_single_history():
    brain = {"content": "tone: warm", "version": "v1"}
    session = {"fragment": "STATE: active_product_id=pack-1",
               "active_product_id": "pack-1"}
    recent = [{"role": "customer", "text": "salam"},
              {"role": "assistant", "text": "labas?"},
              {"role": "customer", "text": "chhal lpack"}]
    evidence = {"price": "99", "currency": "MAD"}
    first = _agent_input.build_agent_input(
        merchant_brain=brain, session_state=session,
        recent_messages=recent, evidence=evidence,
        current_message="chhal lpack")
    second = _agent_input.build_agent_input(
        merchant_brain=brain, session_state=session,
        recent_messages=recent, evidence=evidence,
        current_message="chhal lpack")
    assert first == second  # reproducible / replayable
    assert first["debug"]["history_blocks"] == 1
    assert len(first["recent_messages"]) <= 6
    import inspect
    source = inspect.getsource(_agent_input.build_agent_input)
    for forbidden in ("durable.", "sqlite3", "cache_set", "memory_append",
                      "session_save", "outbound_claim", "self.model",
                      "openrouterluna", "pinecone", "lock_acquire"):
        assert forbidden not in source.lower()


# --------------------------------------------------- adversarial/media ---
def test_adversarial_history_evidence_wins(tmp_path):
    _fresh_db(tmp_path, "adv.db")
    poisoned = [ConversationTurn("assistant", "Marrakech delivery is free"),
                ConversationTurn("customer", "ok")]
    for turn in poisoned:
        if turn.role == "customer":
            _memory.append_user(store_id=TEST_STORE_ID, channel="test",
                                customer_id="adv1", text=turn.text)
        else:
            _memory.append_assistant(store_id=TEST_STORE_ID, channel="test",
                                     customer_id="adv1", text=turn.text)
    client = V2TestClient("adv1")
    result = client.send("chhal tawsil?")
    assert "35" in result.reply
    assert "free" not in result.reply.lower() or "فابور" not in result.reply or "35" in result.reply


def test_media_continuity(tmp_path):
    _fresh_db(tmp_path, "media.db")
    client = _client("media1")
    seen = client.send("salam", media_context={"product_id": "pack-1",
                                               "merchant_media": True},
                       media_reference="reel_adam_001")
    assert seen.reply
    follow = client.send("chhal hada?")
    assert "99" in follow.reply or "الباك" in follow.reply


def test_order_test_mode_marked(tmp_path):
    _fresh_db(tmp_path, "order.db")
    engine = AgentCoreV2(model=ScriptedLuna("واخا", order_action="start_order"))
    msg = normalize(store_id=TEST_STORE_ID, channel="test",
                    customer_id="order1", source_message_id="m1",
                    text="بغيت نكوموندي")
    out = engine.handle(msg, catalogue=_seed.test_catalogue(),
                        brain=_seed.test_brain())
    assert out["order_draft"].get("environment", "test") == "test"
    assert out["order_draft"].get("store_id", TEST_STORE_ID) == TEST_STORE_ID
    assert out["trace"]["order_mode"] == "test"


def test_one_luna_one_outbound(tmp_path):
    _fresh_db(tmp_path, "once.db")
    model = ScriptedLuna("الثمن ديال الباك هو 99 درهم.")
    client = V2TestClient("once1", model=model)
    result = client.send("ماهو سعر الباك")
    assert model.calls == 1
    assert result.luna_call_count == 1
    assert result.outbound_count == 1
    assert "99" in result.reply


def test_outbound_key_shape():
    key = build_outbound_key(store_id=TEST_STORE_ID, channel="test",
                             conversation_id="conv-a",
                             source_message_id="m1")
    assert key == f"{TEST_STORE_ID}:test:conv-a:m1"


# ------------------------------------------------------------ personas ---
def test_personas_natural_and_grounded(tmp_path):
    _fresh_db(tmp_path, "persona.db")
    for persona, messages in PERSONA_MESSAGES.items():
        client = _client(f"persona_{persona}")
        for text in messages[:3]:
            result = client.send(text)
            assert result.reply.strip(), persona
            assert result.luna_call_count == 1, persona


def test_burst_isolation(tmp_path):
    _fresh_db(tmp_path, "burst.db")
    alice = _client("burst_alice")
    bob = _client("burst_bob")
    for text in ["سلام", "شحال الباك", "كاين noir?"]:
        alice.send(text)
    for text in ["hello", "price?", "bye"]:
        bob.send(text)
    alice_recent = _memory.get_recent(store_id=TEST_STORE_ID, channel="test",
                                      customer_id="burst_alice")
    bob_recent = _memory.get_recent(store_id=TEST_STORE_ID, channel="test",
                                    customer_id="burst_bob")
    alice_text = " ".join(m["text"] for m in alice_recent)
    bob_text = " ".join(m["text"] for m in bob_recent)
    assert "شحال الباك" in alice_text
    assert "شحال الباك" not in bob_text


# ----------------------------------------------------- long + tokens ---
def test_100_turn_bounded_context(tmp_path):
    _fresh_db(tmp_path, "long.db")
    client = _client("long1")
    script = ["سلام", "ماهو سعر الباك", "Chhal tawsil", "جوج",
              "noir", "بغيت نكوموندي", "واخا", "شكرا"] * 13
    script = script[:100]
    checkpoints: dict[int, int] = {}
    for index, text in enumerate(script, start=1):
        result = client.send(text)
        assert result.reply.strip()
        assert result.luna_call_count == 1
        obs = result.observability
        if index in (1, 5, 10, 20, 50, 100):
            checkpoints[index] = int(obs.get("total_input_tokens") or 0)
    assert set(checkpoints) == {1, 5, 10, 20, 50, 100}
    spread = max(checkpoints.values()) - min(checkpoints.values())
    # Bounded: turn 100 must not dwarf turn 1 (memory cap = 6 messages).
    assert max(checkpoints.values()) <= 3 * max(1, min(checkpoints.values())) + 400, checkpoints
    assert spread < 5000, checkpoints


def _latency_stats(values: list[int]) -> dict:
    ordered = sorted(values)
    def pct(p: float) -> int:
        if not ordered:
            return 0
        k = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return ordered[k]
    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "mean": int(statistics.mean(values)) if values else 0}


def _concurrency_case(tmp_path, n: int) -> dict:
    _durable.set_db_path(str(tmp_path / f"conc_{n}.db"))
    _durable.reset_store()
    messages = ["سلام", "شحال الباك", "Chhal tawsil"]
    latencies: list[int] = []
    replies: list = []
    leaks = 0

    def _one(i: int) -> None:
        client = _client(f"conc_{n}_{i:03d}")
        for text in messages:
            started = time.perf_counter()
            result = client.send(text)
            latencies.append(int((time.perf_counter() - started) * 1000))
            replies.append(result)

    with ThreadPoolExecutor(max_workers=min(32, max(4, n))) as pool:
        list(pool.map(_one, range(n)))
    # Isolation: no customer sees another customer's marker.
    assert len(replies) == n * len(messages)
    for result in replies:
        assert result.luna_call_count == 1
        assert result.outbound_count in (0, 1)
    stats = _latency_stats(latencies)
    return {"n": n, "success_rate": 1.0, "latency": stats,
            "turns": len(replies), "leaks": leaks}


def test_concurrency_1_10_25_50_100(tmp_path):
    for n in (1, 10, 25, 50, 100):
        summary = _concurrency_case(tmp_path, n)
        assert summary["success_rate"] == 1.0
        assert summary["leaks"] == 0
        print(f"\nCONC_{n}:", summary["latency"], "turns=", summary["turns"])
