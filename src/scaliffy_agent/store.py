from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

from .types import (
    ConversationBudget, ConversationTurn, KnowledgeDocument, KnowledgeHit,
    MemoryUpdate, ReplyScript, StoreContext, IncomingMessage,
)


class KnowledgeStore(Protocol):
    def upsert(self, documents: list[KnowledgeDocument]) -> None: ...
    def query(
        self, *, store_id: str, text: str, limit: int = 5,
        vector: tuple[float, ...] = (), preferred_kinds: tuple[str, ...] = (),
    ) -> list[KnowledgeHit]: ...
    def customer_memory_context(
        self, *, store: StoreContext, message: IncomingMessage,
        vector: tuple[float, ...], limit: int = 3,
    ) -> tuple[str, ...]: ...
    def recent_grounding_context(
        self, *, store: StoreContext, message: IncomingMessage, limit: int = 4,
    ) -> tuple[KnowledgeHit, ...]: ...
    def persist_customer_exchange(
        self, *, store: StoreContext, message: IncomingMessage, reply: str,
        script: ReplyScript, vector: tuple[float, ...], updates: tuple[MemoryUpdate, ...],
        conversation_budget: ConversationBudget | None = None,
        grounded_facts: tuple[KnowledgeDocument, ...] = (),
    ) -> None: ...
    def conversation_budget(
        self, *, store: StoreContext, customer_id: str,
    ) -> ConversationBudget: ...
    def recent_customer_history(
        self, *, store: StoreContext, customer_id: str, limit: int = 16,
    ) -> tuple[ConversationTurn, ...]: ...
    def customer_message_result(
        self, *, store: StoreContext, message: IncomingMessage,
    ) -> tuple[str, bool] | None: ...
    def mark_customer_message_sent(
        self, *, store: StoreContext, message: IncomingMessage,
    ) -> None: ...


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[\wÀ-ÿ]+", text.lower()) if len(term) > 2}


class InMemoryKnowledgeStore:
    """Reference store for deterministic tests; the production adapter keeps this same API."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, KnowledgeDocument]] = defaultdict(dict)
        self._customer_exchanges: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self._processed_messages: set[tuple[str, str, str, str]] = set()
        self._message_results: dict[tuple[str, str, str, str], tuple[str, bool]] = {}
        self._conversation_budgets: dict[tuple[str, str, str], ConversationBudget] = {}
        self._recent_grounding: dict[
            tuple[str, str, str], tuple[KnowledgeDocument, ...]
        ] = {}

    def upsert(self, documents: list[KnowledgeDocument]) -> None:
        for document in documents:
            self._documents[document.store_id][document.id] = document

    def query(
        self, *, store_id: str, text: str, limit: int = 5,
        vector: tuple[float, ...] = (), preferred_kinds: tuple[str, ...] = (),
    ) -> list[KnowledgeHit]:
        query_terms = _terms(text)
        matches: list[KnowledgeHit] = []
        for document in self._documents.get(store_id, {}).values():
            overlap = query_terms & _terms(document.text)
            if overlap:
                score = len(overlap) / max(1, len(query_terms))
                if preferred_kinds and document.metadata.get("kind") in preferred_kinds:
                    score += 0.08
                matches.append(KnowledgeHit(document=document, score=score))
        return sorted(matches, key=lambda hit: hit.score, reverse=True)[:limit]

    def customer_memory_context(
        self, *, store: StoreContext, message: IncomingMessage,
        vector: tuple[float, ...], limit: int = 3,
    ) -> tuple[str, ...]:
        del vector
        key = (store.merchant_account_id, store.channel.value, message.customer_id)
        return tuple(self._customer_exchanges.get(key, [])[-limit:])

    def recent_grounding_context(
        self, *, store: StoreContext, message: IncomingMessage, limit: int = 4,
    ) -> tuple[KnowledgeHit, ...]:
        key = (store.merchant_account_id, store.channel.value, message.customer_id)
        return tuple(
            KnowledgeHit(document=document, score=1.0)
            for document in self._recent_grounding.get(key, ())[:max(0, limit)]
        )

    def persist_customer_exchange(
        self, *, store: StoreContext, message: IncomingMessage, reply: str,
        script: ReplyScript, vector: tuple[float, ...], updates: tuple[MemoryUpdate, ...],
        conversation_budget: ConversationBudget | None = None,
        grounded_facts: tuple[KnowledgeDocument, ...] = (),
    ) -> None:
        del script, vector, updates
        key = (store.merchant_account_id, store.channel.value, message.customer_id)
        self._customer_exchanges[key].append(f"Customer: {message.text}\nStore representative: {reply}")
        self._processed_messages.add((*key, message.message_id))
        self._message_results[(*key, message.message_id)] = (reply, False)
        if conversation_budget is not None:
            self._conversation_budgets[key] = conversation_budget
        if grounded_facts:
            self._recent_grounding[key] = tuple(grounded_facts[:4])

    def conversation_budget(
        self, *, store: StoreContext, customer_id: str,
    ) -> ConversationBudget:
        return self._conversation_budgets.get(
            (store.merchant_account_id, store.channel.value, customer_id),
            ConversationBudget(),
        )

    def recent_customer_history(
        self, *, store: StoreContext, customer_id: str, limit: int = 16,
    ) -> tuple[ConversationTurn, ...]:
        key = (store.merchant_account_id, store.channel.value, customer_id)
        turns: list[ConversationTurn] = []
        for exchange in self._customer_exchanges.get(key, [])[-max(1, limit // 2):]:
            customer, _, assistant = exchange.partition("\nStore representative: ")
            turns.extend((
                ConversationTurn("customer", customer.removeprefix("Customer: ")),
                ConversationTurn("assistant", assistant),
            ))
        return tuple(turns[-limit:])

    def customer_message_result(
        self, *, store: StoreContext, message: IncomingMessage,
    ) -> tuple[str, bool] | None:
        return self._message_results.get((
            store.merchant_account_id,
            store.channel.value,
            message.customer_id,
            message.message_id,
        ))

    def mark_customer_message_sent(
        self, *, store: StoreContext, message: IncomingMessage,
    ) -> None:
        key = (store.merchant_account_id, store.channel.value, message.customer_id, message.message_id)
        result = self._message_results.get(key)
        if result:
            self._message_results[key] = (result[0], True)
