from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .store import KnowledgeStore
from .types import KnowledgeDocument


@dataclass(frozen=True)
class YouCanProduct:
    id: str
    title: str
    description: str
    price: str
    stock: str
    variants: tuple[str, ...] = ()
    images: tuple[str, ...] = ()


@dataclass(frozen=True)
class YouCanSnapshot:
    store_id: str
    store_name: str
    address: str
    policies: tuple[str, ...]
    products: tuple[YouCanProduct, ...]


class YouCanClient(Protocol):
    def fetch_snapshot(self, *, store_id: str) -> YouCanSnapshot: ...


class YouCanIndexer:
    """Converts a completed YouCan connection into traceable, idempotent RAG chunks."""

    def __init__(self, knowledge_store: KnowledgeStore) -> None:
        self.knowledge_store = knowledge_store

    def documents_for(self, snapshot: YouCanSnapshot) -> list[KnowledgeDocument]:
        documents = [
            KnowledgeDocument(
                id=f"store:{snapshot.store_id}:profile",
                store_id=snapshot.store_id,
                source="youcan:store",
                text=f"Boutique {snapshot.store_name}. Adresse: {snapshot.address}.",
                metadata={"kind": "store_profile"},
            )
        ]
        documents.extend(
            KnowledgeDocument(
                id=f"store:{snapshot.store_id}:policy:{index}",
                store_id=snapshot.store_id,
                source="youcan:policy",
                text=policy,
                metadata={"kind": "policy"},
            )
            for index, policy in enumerate(snapshot.policies)
        )
        documents.extend(
            KnowledgeDocument(
                id=f"store:{snapshot.store_id}:product:{product.id}",
                store_id=snapshot.store_id,
                source="youcan:product",
                text=(
                    f"Produit: {product.title}. Prix: {product.price}. Stock: {product.stock}. "
                    f"Variantes: {', '.join(product.variants) or 'aucune'}. Description: {product.description}"
                ),
                metadata={
                    "kind": "product",
                    "product_id": product.id,
                    "title": product.title[:500],
                    "image_urls": json.dumps(
                        list(dict.fromkeys(str(url).strip()[:1_600] for url in product.images if str(url).strip()))[:8],
                        ensure_ascii=False,
                    ),
                },
            )
            for product in snapshot.products
        )
        return documents

    def index_snapshot(self, snapshot: YouCanSnapshot) -> list[KnowledgeDocument]:
        documents = self.documents_for(snapshot)
        self.knowledge_store.upsert(documents)
        return documents
