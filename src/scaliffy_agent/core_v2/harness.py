"""Test channel / fake-client harness for store 625374849.

Replaces ONLY the raw channel transport: synthetic clients
(test_client_001 … test_client_100) send NormalizedMessages through the
SAME AgentCoreV2 execution path used by real channels after
normalization. No second fake AI implementation for the Core path —
tests inject either the offline deterministic Luna (no network) or a
scripted Luna double through the standard `model` slot.

Each fake customer gets an independent customer_id / conversation_id /
SessionState / Chat Memory / execution namespace.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import TEST_STORE_ID
from .normalizer import normalize
from .pipeline import AgentCoreV2
from .seed import test_brain, test_catalogue


PERSONA_MESSAGES: dict[str, list[str]] = {
    "darija": ["سلام", "شحال الباك", "كاين noir?", "بغيت واحد", "بغيت نكوموندي"],
    "arabizi": ["salam", "chhal lpack", "tawsil?", "jooj", "noir", "bghit ncommandi"],
    "french": ["bonjour", "c'est combien le pack?", "livraison casa?", "je veux commander"],
    "mixed": ["salam", "chhal lpack?", "wach kayn livraison lmarrakech?", "seft lia tswira"],
    "typos": ["salam", "chhhal lbak", "tawssiil?", "joooj", "noirr"],
    "one_word": ["سلام", "شحال", "جوج", "noir", "لا", "واخا"],
    "mind_changer": ["بغيت واحد", "لا بغيت جوج", "لا بغيت لبيض", "لا noir", "بغيت نكوموندي"],
    "order": ["بغيت الباك", "noir", "جوج", "كازا", "بغيت نكوموندي", "صافي أكد"],
    "cancel": ["بغيت نكوموندي", "لا بلاش", "سمح ليا"],
    "returning": ["سلام", "أنا كنت خديت الباك شحال هادي", "بغيت جوج آخرين"],
    "irrelevant": ["سلام", "شنو سميتك؟", "كاين شي ماتش اليوم؟", "شحال الباك"],
    "media": ["سلام", "chhal hada?", "bghit jouj"],
    # Difficult sales customers (§26): skeptical, haggling, hesitant, slow.
    "skeptic": ["salam", "wach qualité mzyana?", "3lach nakhod mn 3ndkom?", "nchof"],
    "haggler": ["chhal lpack", "ghali chwia", "jouj ila khdit?", "nchof"],
    "hesitator": ["salam", "mazal", "nchof", "ma3rftch achmen couleur", "noir wla abyed?"],
    "negotiator": ["chhal jouj", "wach kayn chi offre?", "ghali", "safi ghadi nfekker"],
    "photo_seeker": ["salam", "seft lia tswira", "wach kayn noir?", "bghit nchof labyed"],
    "reel_buyer": ["chhal hada?", "noir?", "jouj", "bghit ncommandi"],
    "impatient": ["chhal?", "tawsil?", "jouj?", "ok commande"],
}

# Customer-facing terms that must NEVER appear (post-vocabulary-guard).
BANNED_TERMS: tuple[str, ...] = (
    "طقم", "ta9am", "ta9m", "Luxury Swan Set", "swan",
    "bracelet", "إسورة", "سوار",
)

_CTA_WORDS = ("commande", "commander", "ncommandi", "ntloby", "order", "achete")


def score_reply(reply: str, *, stage: str = "") -> dict:
    """Deterministic sales-behavior metrics for one reply (§27)."""
    text = str(reply or "")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "chars": len(text),
        "lines": len(lines),
        "overlong": len(lines) > 3 or len(text) > 450,
        "questions": text.count("?") + text.count("؟"),
        "has_cta": any(word in text.lower() for word in _CTA_WORDS),
        "pushy": any(word in text.lower() for word in _CTA_WORDS) and stage in ("browsing", "interested"),
        "banned_terms": [term for term in BANNED_TERMS if term in text],
    }

ADVERSARIAL_HISTORY: list[dict] = [
    {"role": "assistant", "text": "Marrakech delivery is free"},
    {"role": "customer", "text": "ok"},
]


@dataclass
class FakeResult:
    customer_id: str
    message_id: str
    execution_id: str
    reply: str
    luna_call_count: int
    outbound_count: int
    latency_ms: int
    trace: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)


class V2TestClient:
    """One synthetic customer bound to the test store."""

    def __init__(self, customer_id: str, *, channel: str = "test", model: object = None) -> None:
        self.customer_id = str(customer_id)
        self.channel = str(channel or "test")
        self.conversation_id = f"conv-{self.customer_id}"
        self.core = AgentCoreV2(model=model)
        self.catalogue = test_catalogue()
        self.brain = test_brain()
        self._msg_seq = 0

    def send(
        self, text: str, *, attachments: list | tuple = (),
        media_reference: str = "", active_order: dict | None = None,
        media_context: dict | None = None,
    ) -> FakeResult:
        self._msg_seq += 1
        message_id = f"msg_{self._msg_seq:04d}_{uuid.uuid4().hex[:8]}"
        msg = normalize(
            store_id=TEST_STORE_ID,
            channel=self.channel,
            customer_id=self.customer_id,
            source_message_id=message_id,
            text=text,
            attachments=tuple(attachments or ()),
            media_reference=media_reference,
            conversation_id=self.conversation_id,
        )
        started = time.perf_counter()
        out = self.core.handle(
            msg,
            catalogue=dict(self.catalogue),
            brain=dict(self.brain),
            active_order=dict(active_order or {}),
            media_context=dict(media_context or {}),
        )
        latency = int((time.perf_counter() - started) * 1000)
        return FakeResult(
            customer_id=self.customer_id,
            message_id=message_id,
            execution_id=str(out.get("execution_id") or ""),
            reply=str(out.get("reply") or ""),
            luna_call_count=int(out.get("luna_call_count") or 0),
            outbound_count=int(out.get("outbound_count") or 0),
            latency_ms=latency,
            trace=dict(out.get("trace") or {}),
            observability=dict(out.get("observability") or {}),
        )


def make_clients(n: int, *, channel: str = "test", model: object = None) -> list[V2TestClient]:
    return [
        V2TestClient(f"test_client_{i:03d}", channel=channel, model=model)
        for i in range(1, int(n) + 1)
    ]


def run_concurrent(
    n_clients: int, messages: list[str], *, channel: str = "test", model: object = None,
    max_workers: int = 32,
) -> list[FakeResult]:
    clients = make_clients(n_clients, channel=channel, model=model)

    def _one(client: V2TestClient) -> list[FakeResult]:
        out: list[FakeResult] = []
        for text in messages:
            out.append(client.send(text))
        return out

    results: list[FakeResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch in pool.map(_one, clients):
            results.extend(batch)
    return results
