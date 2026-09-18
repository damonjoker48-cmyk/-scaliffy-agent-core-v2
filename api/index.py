from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scaliffy_agent import AgentCore, IncomingMessage, StoreContext, YouCanIndexer
from scaliffy_agent.ingestion import YouCanProduct, YouCanSnapshot
from scaliffy_agent.providers import OpenRouterLunaModel, PineconeKnowledgeStore
from scaliffy_agent.types import (
    Attachment, Channel, ConversationSurface, ConversationTurn, KnowledgeDocument, ReplyScript,
    VoiceTranscription,
)
from scaliffy_agent.language import detect_reply_script

app = FastAPI(title="Scaliffy Agent Core", version="0.1.0")
logger = logging.getLogger("scaliffy.agent_core")
# httpx logs full query strings at INFO. Meta access tokens are query
# parameters for some subscription endpoints, so keep third-party request
# URLs out of runtime logs entirely.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.middleware("http")
async def remove_vercel_api_prefix(request: Request, call_next):
    """Vercel invokes ``api/index.py`` while retaining ``/api`` in ASGI path."""
    path = request.scope.get("path", "")
    if path == "/api":
        request.scope["path"] = "/"
    elif path.startswith("/api/"):
        request.scope["path"] = path[4:]
    return await call_next(request)

YOUCAN_AUTH_URL = "https://seller-area.youcan.shop/admin/oauth/authorize"
YOUCAN_TOKEN_URL = "https://api.youcan.shop/oauth/token"
YOUCAN_API_BASE = "https://api.youcan.shop"
_PROCESSED_INSTAGRAM_MESSAGE_IDS: set[str] = set()
_MAX_DEDUPLICATION_IDS = 2_000
_VOICE_SINGLEFLIGHT: dict[str, asyncio.Lock] = {}
_CONVERSATION_LOCKS: dict[str, asyncio.Lock] = {}
_MAX_VOICE_BYTES = 20 * 1024 * 1024


def env(name: str) -> str:
    """Read a usable runtime secret, never treating Vercel redaction as a value.

    ``vercel env pull`` intentionally serializes sensitive values as
    ``[SENSITIVE]``.  That marker must never pass a configuration check: doing
    so would make health checks green and defer the real error until OAuth,
    Pinecone, or Meta was already handling a request.
    """
    value = os.getenv(name, "").strip()
    return "" if value in {"[SENSITIVE]", "<SENSITIVE>", "undefined", "null"} else value


def require_env(*names: str) -> None:
    missing = [name for name in names if not env(name)]
    if missing:
        raise HTTPException(503, f"Missing required server configuration: {', '.join(missing)}")


def public_base(request: Request) -> str:
    return env("PUBLIC_BASE_URL") or str(request.base_url).rstrip("/")


def youcan_redirect_uri(request: Request) -> str:
    return env("YOUCAN_REDIRECT_URI") or f"{public_base(request)}/api/youcan/callback"


def sign_state(payload: dict[str, Any]) -> str:
    require_env("AGENT_CORE_STATE_SECRET")
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(env("AGENT_CORE_STATE_SECRET").encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def read_state(state: str) -> dict[str, Any]:
    try:
        body, signature = state.rsplit(".", 1)
        expected = hmac.new(env("AGENT_CORE_STATE_SECRET").encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        if int(payload["exp"]) < int(time.time()):
            raise ValueError("expired")
        if not payload.get("merchant_account_id"):
            raise ValueError("merchant")
        return payload
    except Exception as exc:
        raise HTTPException(400, "Invalid, expired, or replayed YouCan state") from exc


async def fetch_youcan_snapshot(access_token: str) -> YouCanSnapshot:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers, timeout=25) as client:
        me = await client.get(f"{YOUCAN_API_BASE}/me")
        if me.status_code != 200:
            raise HTTPException(502, f"YouCan store lookup failed (HTTP {me.status_code})")
        raw_store = me.json()
        store = raw_store.get("data", raw_store.get("store", raw_store)) if isinstance(raw_store, dict) else {}
        store_id = str(store.get("id") or "")
        if not store_id:
            raise HTTPException(502, "YouCan did not return a store id")

        products: list[YouCanProduct] = []
        page = 1
        while page <= 50:
            response = await client.get(
                f"{YOUCAN_API_BASE}/products?include=variants,categories&per_page=100&page={page}"
            )
            if response.status_code != 200:
                raise HTTPException(502, f"YouCan product import failed (HTTP {response.status_code})")
            payload = response.json()
            batch = payload.get("data", payload) if isinstance(payload, dict) else []
            if not isinstance(batch, list) or not batch:
                break
            for product in batch:
                if not isinstance(product, dict):
                    continue
                variants_raw = product.get("variants") or []
                if isinstance(variants_raw, dict):
                    variants_raw = variants_raw.get("data", [])
                variants = tuple(
                    str(item.get("name") or item.get("title") or item.get("sku") or "").strip()
                    for item in variants_raw if isinstance(item, dict)
                )
                products.append(YouCanProduct(
                    id=str(product.get("id") or product.get("slug") or secrets.token_hex(8)),
                    title=str(product.get("name") or product.get("title") or "Unnamed product"),
                    description=str(product.get("description") or product.get("short_description") or ""),
                    price=str(product.get("price") or product.get("sale_price") or product.get("regular_price") or "unknown"),
                    stock=str(product.get("stock") or product.get("inventory") or product.get("quantity") or "unknown"),
                    variants=tuple(value for value in variants if value),
                ))
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            pagination = meta.get("pagination", meta) if isinstance(meta, dict) else {}
            if not pagination or int(pagination.get("current_page", page)) >= int(pagination.get("total_pages", page)):
                break
            page += 1

        policies: list[str] = []
        shipping = await client.get(f"{YOUCAN_API_BASE}/shipping-zones?include=rates")
        if shipping.status_code == 200:
            zones = shipping.json()
            zones = zones.get("data", zones) if isinstance(zones, dict) else []
            for zone in zones if isinstance(zones, list) else []:
                if isinstance(zone, dict):
                    policies.append(f"Livraison: {json.dumps(zone, ensure_ascii=False)[:1200]}")

        address = ", ".join(str(store.get(key) or "").strip() for key in ("address", "address1", "city", "country") if store.get(key))
        return YouCanSnapshot(
            store_id=store_id,
            store_name=str(store.get("name") or store.get("title") or f"YouCan {store_id}"),
            address=address or "Adresse non fournie par YouCan",
            policies=tuple(policies),
            products=tuple(products),
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    runtime_required = [
        "OPENROUTER_API_KEY", "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME", "META_APP_SECRET", "META_AGENT_CORE_VERIFY_TOKEN",
    ]
    if env("INSTAGRAM_APP_ID") and env("INSTAGRAM_APP_ID") != env("META_APP_ID"):
        runtime_required.append("INSTAGRAM_APP_SECRET")
    missing = [name for name in runtime_required if not env(name)]
    youcan_oauth_missing = [name for name in ("YOUCAN_CLIENT_ID", "YOUCAN_CLIENT_SECRET") if not env(name)]
    instagram_rag: dict[str, Any] = {"bound": False, "vectors": 0, "ready": False}
    business_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    if business_id and not any(name in missing for name in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME")):
        try:
            knowledge_store = PineconeKnowledgeStore()
            binding = knowledge_store.channel_binding(
                channel=Channel.INSTAGRAM.value,
                channel_account_id=business_id,
            )
            if binding:
                vector_count = knowledge_store.namespace_vector_count(store_id=binding["store_id"])
                instagram_rag = {
                    "bound": True,
                    "store_id": binding["store_id"],
                    "vectors": vector_count,
                    "ready": vector_count > 0,
                }
        except Exception:
            logger.exception("Instagram RAG readiness check failed")
            instagram_rag["check_failed"] = True
    return {
        "ok": not missing and instagram_rag["ready"],
        "service": "scaliffy-agent-core",
        "missing": missing,
        "youcan_oauth_missing": youcan_oauth_missing,
        "instagram_rag": instagram_rag,
    }


@app.get("/youcan/connect")
async def start_youcan_connect(
    request: Request,
    merchant_account_id: str = Query(min_length=3, max_length=180),
    instagram_business_account_id: str = Query(default="", max_length=180),
):
    require_env("YOUCAN_CLIENT_ID", "YOUCAN_CLIENT_SECRET", "AGENT_CORE_STATE_SECRET")
    redirect_uri = youcan_redirect_uri(request)
    instagram_account_id = instagram_business_account_id.strip() or env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    state = sign_state({
        "merchant_account_id": merchant_account_id,
        "instagram_business_account_id": instagram_account_id,
        "nonce": secrets.token_urlsafe(18),
        "exp": int(time.time()) + 600,
    })
    auth_url = (
        f"{YOUCAN_AUTH_URL}?client_id={quote(env('YOUCAN_CLIENT_ID'), safe='')}"
        f"&redirect_uri={quote(redirect_uri, safe=':/')}"
        "&response_type=code&scope[]=*"
        f"&state={quote(state, safe='')}"
    )
    return RedirectResponse(auth_url, status_code=302)


@app.get("/youcan/callback")
async def youcan_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    payload = read_state(state)
    if error:
        raise HTTPException(400, f"YouCan authorization denied: {error}")
    if not code:
        raise HTTPException(400, "YouCan did not return an authorization code")
    redirect_uri = youcan_redirect_uri(request)
    async with httpx.AsyncClient(timeout=20) as client:
        token = await client.post(YOUCAN_TOKEN_URL, data={
            "grant_type": "authorization_code", "client_id": env("YOUCAN_CLIENT_ID"),
            "client_secret": env("YOUCAN_CLIENT_SECRET"), "redirect_uri": redirect_uri, "code": code,
        }, headers={"Accept": "application/json"})
    if token.status_code != 200:
        logger.warning("YouCan token exchange failed status=%s", token.status_code)
        raise HTTPException(502, f"YouCan token exchange failed (HTTP {token.status_code})")
    access_token = str(token.json().get("access_token") or "")
    if not access_token:
        raise HTTPException(502, "YouCan token exchange returned no access token")
    try:
        snapshot = await fetch_youcan_snapshot(access_token)
        knowledge_store = PineconeKnowledgeStore()
        documents = YouCanIndexer(knowledge_store).index_snapshot(snapshot)
        verified_documents = knowledge_store.verify_documents(documents)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("YouCan RAG indexing failed for merchant=%s", payload["merchant_account_id"])
        raise HTTPException(502, "YouCan RAG indexing verification failed") from exc
    instagram_account_id = str(payload.get("instagram_business_account_id") or "").strip()
    if instagram_account_id:
        try:
            knowledge_store.bind_channel(
                channel=Channel.INSTAGRAM.value,
                channel_account_id=instagram_account_id,
                merchant_account_id=str(payload["merchant_account_id"]),
                store_id=snapshot.store_id,
                store_name=snapshot.store_name,
            )
            _instagram_store_context_for.cache_clear()
            _channel_store_context_for.cache_clear()
        except Exception as exc:
            logger.exception("Instagram channel binding failed for store=%s", snapshot.store_id)
            raise HTTPException(502, "Instagram binding could not be saved") from exc
    logger.info(
        "YouCan RAG ready store=%s vectors=%s instagram_binding=%s",
        snapshot.store_id,
        verified_documents,
        bool(instagram_account_id),
    )
    return HTMLResponse(
        "<h1>YouCan connected and indexed</h1>"
        f"<p>Store: {snapshot.store_name}</p><p>Store ID: {snapshot.store_id}</p>"
        f"<p>RAG namespace: agent-core:{snapshot.store_id}</p><p>Chunks indexed and verified: {verified_documents}</p>"
        f"<p>Merchant account: {payload['merchant_account_id']}</p>"
        f"<p>Instagram binding: {'saved' if instagram_account_id else 'not requested'}</p>"
    )


class AgentMessageBody(BaseModel):
    merchant_account_id: str = Field(min_length=3, max_length=180)
    store_id: str = Field(min_length=1, max_length=180)
    store_name: str = Field(default="", max_length=180)
    channel: Channel = Channel.TEST
    message_id: str = Field(min_length=1, max_length=180)
    customer_id: str = Field(min_length=1, max_length=180)
    text: str = Field(default="", max_length=4_000)
    history: list[ConversationTurn] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    human_takeover: bool = False
    source: str = Field(default="text", max_length=32)
    spoken_language: str = Field(default="", max_length=64)
    surface: str = Field(default=ConversationSurface.DIRECT_MESSAGE.value, max_length=48)
    comment_id: str = Field(default="", max_length=180)
    reply_context: dict[str, Any] = Field(default_factory=dict)
    merchant_runtime_context: str = Field(default="", max_length=6_000)
    catalogue_context: dict[str, str] = Field(default_factory=dict)
    active_order: dict[str, str] = Field(default_factory=dict)
    known_customer: dict[str, str] = Field(default_factory=dict)
    store_brain: dict[str, Any] = Field(default_factory=dict)
    media_context: dict[str, Any] = Field(default_factory=dict)


class ScaliffyStoreImportBody(BaseModel):
    store_id: str = Field(min_length=3, max_length=180)
    merchant_account_id: str = Field(min_length=3, max_length=180)
    instagram_business_account_id: str = Field(min_length=3, max_length=180)


class ChannelBindingBody(BaseModel):
    channel: Channel
    channel_account_id: str = Field(min_length=3, max_length=180)
    merchant_account_id: str = Field(min_length=3, max_length=180)
    store_id: str = Field(min_length=1, max_length=180)
    store_name: str = Field(default="Store", max_length=180)


class StoreProductBody(BaseModel):
    id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=8_000)
    price: str = Field(default="", max_length=180)
    stock: str = Field(default="", max_length=180)
    variants: list[str] = Field(default_factory=list, max_length=80)
    image: str = Field(default="", max_length=3_000)
    images: list[str] = Field(default_factory=list, max_length=8)


class StoreSyncBody(BaseModel):
    """Generic, channel-neutral store snapshot accepted from Scaliffy.

    This is an adapter contract only. It refreshes a data namespace used by
    the one shared Core and never constructs a tenant-specific agent.
    """

    merchant_account_id: str = Field(min_length=3, max_length=180)
    store_id: str = Field(min_length=1, max_length=180)
    store_name: str = Field(default="Store", max_length=500)
    address: str = Field(default="", max_length=2_000)
    policies: list[str] = Field(default_factory=list, max_length=120)
    products: list[StoreProductBody] = Field(default_factory=list, max_length=2_000)
    merchant_knowledge: list[str] = Field(default_factory=list, max_length=300)
    channel_bindings: list[ChannelBindingBody] = Field(default_factory=list, max_length=20)


def _clean_catalog_text(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _scaliffy_snapshot(store: dict[str, Any], expected_store_id: str) -> YouCanSnapshot:
    if str(store.get("id") or "") != expected_store_id:
        raise HTTPException(502, "Scaliffy returned a different store")
    knowledge = store.get("knowledge_base") or {}
    if not isinstance(knowledge, dict):
        raise HTTPException(502, "Scaliffy store knowledge is invalid")
    raw_products = knowledge.get("products") or []
    if not isinstance(raw_products, list) or not raw_products:
        raise HTTPException(409, "Scaliffy store has no imported products")

    store_data = knowledge.get("store") or {}
    if not isinstance(store_data, dict):
        store_data = {}
    currency = str(store_data.get("currency") or store.get("currency") or "MAD")
    products: list[YouCanProduct] = []
    for raw_product in raw_products:
        if not isinstance(raw_product, dict):
            continue
        variants_raw = raw_product.get("variants") or []
        if isinstance(variants_raw, dict):
            variants_raw = variants_raw.get("data", [])
        variants: list[str] = []
        variant_stock = 0
        for variant in variants_raw if isinstance(variants_raw, list) else []:
            if not isinstance(variant, dict):
                continue
            label = str(variant.get("name") or variant.get("title") or variant.get("sku") or "").strip()
            if not label:
                options = variant.get("options") or variant.get("variations") or variant.get("values") or []
                if options:
                    label = _clean_catalog_text(json.dumps(options, ensure_ascii=False))[:300]
            if label:
                variants.append(label)
            try:
                variant_stock += int(variant.get("inventory") or variant.get("quantity") or 0)
            except (TypeError, ValueError):
                pass
        raw_stock = raw_product.get("inventory")
        stock = str(raw_stock if raw_stock is not None else variant_stock)
        raw_price = raw_product.get("price")
        price = f"{raw_price} {currency}" if raw_price not in (None, "") else "unknown"
        products.append(YouCanProduct(
            id=str(raw_product.get("id") or raw_product.get("slug") or secrets.token_hex(8)),
            title=str(raw_product.get("name") or raw_product.get("title") or "Unnamed product"),
            description=_clean_catalog_text(raw_product.get("description")),
            price=price,
            stock=stock,
            variants=tuple(variants),
            images=tuple(
                str(value.get("url") or value.get("src") or "")[:3_000]
                if isinstance(value, dict) else str(value)[:3_000]
                for value in (
                    raw_product.get("images")
                    if isinstance(raw_product.get("images"), list)
                    else [raw_product.get("image") or raw_product.get("image_url")]
                )[:8]
                if str(
                    (value.get("url") or value.get("src") or "")
                    if isinstance(value, dict) else value or ""
                ).startswith(("http://", "https://"))
            ),
        ))
    if not products:
        raise HTTPException(409, "Scaliffy store has no usable products")

    policies: list[str] = []
    for label, key in (("Livraison", "shipping"), ("Paiement", "payment_methods")):
        values = knowledge.get(key) or []
        for value in values if isinstance(values, list) else [values]:
            policies.append(f"{label}: {_clean_catalog_text(json.dumps(value, ensure_ascii=False))[:1500]}")
    address = str(store_data.get("address") or knowledge.get("address") or "Adresse non fournie")
    return YouCanSnapshot(
        store_id=expected_store_id,
        store_name=str(store.get("name") or store_data.get("name") or expected_store_id),
        address=address,
        policies=tuple(policies),
        products=tuple(products),
    )


@app.post("/admin/import/scaliffy-store")
async def import_scaliffy_store(request: Request, body: ScaliffyStoreImportBody) -> dict[str, Any]:
    expected = env("AGENT_CORE_ADMIN_TOKEN")
    authorization = request.headers.get("Authorization", "")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
    if not expected or not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(403, "Admin authorization failed")

    api_base = env("SCALIFFY_API_BASE") or "https://scaliffy.com/api"
    url = f"{api_base.rstrip('/')}/stores/{quote(body.store_id, safe='')}"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, headers={"Accept": "application/json"})
    if response.status_code != 200:
        raise HTTPException(502, f"Scaliffy store import failed (HTTP {response.status_code})")
    raw_store = response.json()
    if not isinstance(raw_store, dict):
        raise HTTPException(502, "Scaliffy returned an invalid store payload")

    snapshot = _scaliffy_snapshot(raw_store, body.store_id)
    knowledge_store = PineconeKnowledgeStore()
    try:
        documents = YouCanIndexer(knowledge_store).index_snapshot(snapshot)
        verified_documents = knowledge_store.verify_documents(documents)
        knowledge_store.bind_channel(
            channel=Channel.INSTAGRAM.value,
            channel_account_id=body.instagram_business_account_id,
            merchant_account_id=body.merchant_account_id,
            store_id=snapshot.store_id,
            store_name=snapshot.store_name,
        )
        _instagram_store_context_for.cache_clear()
        _channel_store_context_for.cache_clear()
        vector_count = knowledge_store.namespace_vector_count(store_id=snapshot.store_id)
    except Exception as exc:
        logger.exception("Scaliffy store import failed for store=%s", body.store_id)
        raise HTTPException(502, "Scaliffy store RAG indexing failed") from exc
    if vector_count < verified_documents:
        raise HTTPException(502, "Scaliffy store RAG count verification failed")
    logger.info(
        "Scaliffy store imported store=%s vectors=%s instagram_binding=true",
        body.store_id,
        vector_count,
    )
    return {
        "ok": True,
        "store_id": body.store_id,
        "namespace": f"agent-core:{body.store_id}",
        "chunks_verified": verified_documents,
        "vectors": vector_count,
        "instagram_bound": True,
    }


def _require_admin(request: Request) -> None:
    expected = env("AGENT_CORE_ADMIN_TOKEN")
    authorization = request.headers.get("Authorization", "")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
    if not expected or not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(403, "Admin authorization failed")


@app.post("/admin/channels/bind")
async def bind_customer_channel(request: Request, body: ChannelBindingBody) -> dict[str, Any]:
    """Bind any supported customer channel to an already indexed tenant RAG."""
    _require_admin(request)
    if body.channel not in {Channel.INSTAGRAM, Channel.MESSENGER, Channel.WHATSAPP}:
        raise HTTPException(400, "Only customer messaging channels can be bound")
    knowledge = PineconeKnowledgeStore()
    vector_count = knowledge.namespace_vector_count(store_id=body.store_id)
    if vector_count <= 0:
        raise HTTPException(409, "The selected store RAG is not indexed")
    knowledge.bind_channel(
        channel=body.channel.value,
        channel_account_id=body.channel_account_id,
        merchant_account_id=body.merchant_account_id,
        store_id=body.store_id,
        store_name=body.store_name,
    )
    _instagram_store_context_for.cache_clear()
    _channel_store_context_for.cache_clear()
    return {
        "ok": True,
        "channel": body.channel.value,
        "channel_account_id": body.channel_account_id,
        "store_id": body.store_id,
        "vectors": vector_count,
    }


@app.post("/admin/stores/sync")
async def sync_scaliffy_store(request: Request, body: StoreSyncBody) -> dict[str, Any]:
    """Refresh a tenant RAG and bind its active customer channels.

    Scaliffy owns commerce integrations and supplies a normalized snapshot.
    This endpoint only indexes data for the shared Core; it never creates a
    merchant-specific agent, prompt implementation, or channel brain.
    """
    _require_admin(request)
    snapshot = YouCanSnapshot(
        store_id=body.store_id,
        store_name=body.store_name,
        address=body.address or "Adresse non fournie",
        policies=tuple(str(item).strip()[:1_500] for item in body.policies if str(item).strip()),
        products=tuple(
            YouCanProduct(
                id=item.id,
                title=item.title,
                description=item.description,
                price=item.price or "non fourni",
                stock=item.stock or "non fourni",
                variants=tuple(str(value).strip()[:300] for value in item.variants if str(value).strip()),
                images=tuple(
                    str(value).strip()[:3_000]
                    for value in ([item.image] + item.images)
                    if str(value).strip().startswith(("http://", "https://"))
                )[:8],
            )
            for item in body.products
        ),
    )
    knowledge = PineconeKnowledgeStore()
    try:
        documents = YouCanIndexer(knowledge).documents_for(snapshot)
        documents.extend(
            KnowledgeDocument(
                id=f"store:{body.store_id}:merchant:{index}",
                store_id=body.store_id,
                source="scaliffy:merchant-knowledge",
                text=str(item).strip()[:8_000],
                metadata={"kind": "merchant_knowledge"},
            )
            for index, item in enumerate(body.merchant_knowledge)
            if str(item).strip()
        )
        knowledge.replace_store_documents(documents)
        verified_documents = knowledge.verify_documents(documents)
        for binding in body.channel_bindings:
            if binding.merchant_account_id != body.merchant_account_id:
                raise HTTPException(400, "Channel binding merchant does not match store merchant")
            if binding.store_id != body.store_id:
                raise HTTPException(400, "Channel binding store does not match snapshot store")
            knowledge.bind_channel(
                channel=binding.channel.value,
                channel_account_id=binding.channel_account_id,
                merchant_account_id=body.merchant_account_id,
                store_id=body.store_id,
                store_name=body.store_name,
            )
        _instagram_store_context_for.cache_clear()
        _channel_store_context_for.cache_clear()
        vectors = knowledge.namespace_vector_count(store_id=body.store_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scaliffy generic store sync failed store=%s", body.store_id)
        raise HTTPException(502, "Scaliffy store RAG sync failed") from exc
    if vectors < verified_documents:
        raise HTTPException(502, "Scaliffy store RAG count verification failed")
    return {
        "ok": True,
        "store_id": body.store_id,
        "namespace": f"agent-core:{body.store_id}",
        "chunks_verified": verified_documents,
        "vectors": vectors,
        "bound_channels": len(body.channel_bindings),
    }


def _meta_error(payload: Any) -> dict[str, str]:
    error = payload.get("error") if isinstance(payload, dict) else {}
    error = error if isinstance(error, dict) else {}
    return {
        "code": str(error.get("code") or ""),
        "type": str(error.get("type") or ""),
        "message": str(error.get("message") or "")[:300],
    }


@app.post("/admin/meta/register-webhook")
async def register_meta_webhook(request: Request) -> dict[str, Any]:
    """Register and then read back both required Instagram subscriptions."""
    _require_admin(request)
    require_env(
        "META_APP_ID", "META_APP_SECRET", "META_AGENT_CORE_VERIFY_TOKEN",
        "INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID",
    )
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    callback_url = f"{public_base(request)}/api/webhooks/meta"
    app_id = env("META_APP_ID")
    instagram_app_id = env("INSTAGRAM_APP_ID") or app_id
    app_access_token = f"{app_id}|{env('META_APP_SECRET')}"
    app_subscription_url = f"https://graph.facebook.com/{graph_version}/{app_id}/subscriptions"
    instagram_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    account_subscription_url = (
        f"https://graph.instagram.com/{graph_version}/{quote(instagram_id, safe='')}/subscribed_apps"
    )

    async with httpx.AsyncClient(timeout=25) as client:
        app_write = await client.post(app_subscription_url, data={
            "object": "instagram",
            "callback_url": callback_url,
            "verify_token": env("META_AGENT_CORE_VERIFY_TOKEN"),
            "fields": json.dumps(["messages", "messaging_postbacks", "comments", "live_comments"]),
            "include_values": "true",
            "access_token": app_access_token,
        })
        app_write_body = app_write.json() if app_write.content else {}
        app_read = await client.get(app_subscription_url, params={"access_token": app_access_token})
        app_read_body = app_read.json() if app_read.content else {}

        account_write = await client.post(account_subscription_url, params={
            "subscribed_fields": "messages,comments,live_comments",
            "access_token": env("INSTAGRAM_ACCESS_TOKEN"),
        })
        account_write_body = account_write.json() if account_write.content else {}
        account_read = await client.get(account_subscription_url, params={
            "access_token": env("INSTAGRAM_ACCESS_TOKEN"),
        })
        account_read_body = account_read.json() if account_read.content else {}

    subscriptions = app_read_body.get("data") if isinstance(app_read_body, dict) else []
    instagram_subscription = next((
        value for value in subscriptions or []
        if isinstance(value, dict) and value.get("object") == "instagram"
    ), None)
    callback_verified = bool(
        instagram_subscription
        and instagram_subscription.get("callback_url") == callback_url
        and instagram_subscription.get("active")
    )
    account_entries = account_read_body.get("data") if isinstance(account_read_body, dict) else []
    account_fields = sorted({
        str(field)
        for entry in account_entries or [] if isinstance(entry, dict)
        for field in entry.get("subscribed_fields", []) or []
    })
    configured_account_entry = next((
        entry for entry in account_entries or []
        if isinstance(entry, dict) and str(entry.get("id") or "") == instagram_app_id
    ), None)
    configured_account_fields = sorted(
        str(field) for field in (configured_account_entry or {}).get("subscribed_fields", []) or []
    )
    subscribed_apps = [
        {
            "id": str(entry.get("id") or ""),
            "name": str(entry.get("name") or "")[:120],
            "fields": sorted(str(field) for field in entry.get("subscribed_fields", []) or []),
        }
        for entry in account_entries or []
        if isinstance(entry, dict)
    ]
    # Seeing `messages` on another app is not proof that this app owns the
    # Instagram delivery. That mismatch produces valid webhooks signed with a
    # different app secret, which this callback must reject with HTTP 403.
    account_verified = "messages" in configured_account_fields
    result = {
        "ok": callback_verified and account_verified,
        "callback_url": callback_url,
        "app_subscription": {
            "write_status": app_write.status_code,
            "read_status": app_read.status_code,
            "callback_verified": callback_verified,
            "error": _meta_error(app_write_body),
        },
        "account_subscription": {
            "write_status": account_write.status_code,
            "read_status": account_read.status_code,
            "messages_verified": account_verified,
            "fields": account_fields,
            "instagram_app_id": instagram_app_id,
            "configured_app_found": bool(configured_account_entry),
            "configured_app_fields": configured_account_fields,
            "subscribed_apps": subscribed_apps,
            "error": _meta_error(account_write_body),
        },
    }
    logger.info(
        "Meta Instagram webhook registration callback_verified=%s account_verified=%s",
        callback_verified,
        account_verified,
    )
    return result


@app.get("/admin/meta/diagnostics")
async def meta_diagnostics(request: Request) -> dict[str, Any]:
    """Prove that the configured Instagram token belongs to the signing app."""
    _require_admin(request)
    require_env("META_APP_ID", "META_APP_SECRET", "INSTAGRAM_ACCESS_TOKEN")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    configured_app_id = env("INSTAGRAM_APP_ID") or env("META_APP_ID")
    configured_app_secret = env("INSTAGRAM_APP_SECRET") or (
        env("META_APP_SECRET") if configured_app_id == env("META_APP_ID") else ""
    )
    if not configured_app_secret:
        return {
            "ok": False,
            "configured_app_id": configured_app_id,
            "instagram_app_secret_configured": False,
            "token_app_matches_configured_app": False,
        }
    app_access_token = f"{configured_app_id}|{configured_app_secret}"
    async with httpx.AsyncClient(timeout=15) as client:
        debug_response = await client.get(
            f"https://graph.facebook.com/{graph_version}/debug_token",
            params={
                "input_token": env("INSTAGRAM_ACCESS_TOKEN"),
                "access_token": app_access_token,
            },
        )
    payload = debug_response.json() if debug_response.content else {}
    debug_data = payload.get("data") if isinstance(payload, dict) else {}
    debug_data = debug_data if isinstance(debug_data, dict) else {}
    token_app_id = str(debug_data.get("app_id") or "")
    return {
        "ok": debug_response.status_code == 200 and bool(debug_data.get("is_valid")) and token_app_id == configured_app_id,
        "debug_status": debug_response.status_code,
        "token_valid": bool(debug_data.get("is_valid")),
        "token_type": str(debug_data.get("type") or ""),
        "token_app_matches_configured_app": bool(token_app_id and token_app_id == configured_app_id),
        "configured_app_id": configured_app_id,
        "instagram_app_secret_configured": bool(env("INSTAGRAM_APP_SECRET")),
        "token_app_id": token_app_id,
        "scopes": sorted(str(value) for value in debug_data.get("scopes", []) or []),
        "error": _meta_error(payload),
    }


@app.post("/agent/reply")
async def agent_reply(body: AgentMessageBody, background_tasks: BackgroundTasks) -> dict[str, Any]:
    started_at = time.perf_counter()
    source = "instagram_voice" if body.source == "instagram_voice" else "text"
    try:
        surface = ConversationSurface(body.surface)
    except ValueError as exc:
        raise HTTPException(422, "Unsupported conversation surface") from exc
    model = OpenRouterLunaModel()
    agent = AgentCore(knowledge_store=PineconeKnowledgeStore(), model=model)
    result = agent.reply(
        store=StoreContext(body.merchant_account_id, body.store_id, body.store_name, body.channel, human_takeover=body.human_takeover),
        message=IncomingMessage(
            body.message_id, body.text, body.customer_id,
            tuple(body.attachments), tuple(body.history),
            source=source,
            spoken_language=body.spoken_language,
            channel=body.channel,
            surface=surface,
            comment_id=body.comment_id,
            store_name=body.store_name,
            reply_context=body.reply_context,
            merchant_runtime_context=body.merchant_runtime_context,
            catalogue_context=body.catalogue_context,
            active_order=body.active_order,
            known_customer=body.known_customer,
            store_brain=body.store_brain,
            media_context=body.media_context,
        ),
        defer_persistence=True,
    )
    if agent.deferred_persistence is not None:
        # Customer memory is durable but not part of the response critical
        # path. Sending first avoids holding every DM behind Pinecone upserts.
        background_tasks.add_task(agent.deferred_persistence)
    trace = {
        "event": "MERCHANT_RETRIEVAL_TRACE",
        "channel": body.channel,
        "merchant_id": body.merchant_account_id,
        "store_id": body.store_id,
        "retrieval_triggered": bool(result and result.rag_called),
        "namespace": result.retrieval_namespace if result else f"agent-core:{body.store_id}",
        "filter": result.retrieval_filter if result else {"store_id": body.store_id},
        "sources_queried": list(result.retrieval_sources) if result else [],
        "knowledge_ids": list(result.knowledge_ids) if result else [],
        "chunks_retrieved": list(result.retrieved_chunks) if result else [],
        "trace_id": result.trace_id if result else "",
        "reply_length": len(result.text) if result else 0,
        "reply_reason": result.reason if result else "no_reply",
        "raw_reply_length": len(result.raw_model_reply) if result else 0,
        "raw_reply_sha256": hashlib.sha256(
            (result.raw_model_reply if result else "").encode("utf-8")
        ).hexdigest() if result else "",
        "final_reply_sha256": hashlib.sha256(
            (result.text if result else "").encode("utf-8")
        ).hexdigest() if result else "",
        "provider": result.model_provider if result else "",
        "requested_model": result.requested_model if result else "",
        "resolved_model": result.resolved_model if result else "",
        "temperature": result.temperature if result else 0.0,
        "max_output_tokens": result.max_output_tokens if result else 0,
        "reasoning_effort": result.reasoning_effort if result else "",
        "response_format": result.response_format if result else "",
        "effective_prompt_sha256": result.effective_prompt_sha256 if result else "",
        "store_brain_version": result.store_brain_version if result else "",
        "store_brain_tokens": result.store_brain_tokens if result else 0,
        "shipping_provenance": dict(result.shipping_provenance) if result else {},
        "contact_action": str(result.contact_action) if result else "none",
        "order_action": str(result.order_action) if result else "none",
        "llm_cached_input_tokens": result.llm_cached_input_tokens if result else 0,
        "llm_ttft_ms": result.llm_ttft_ms if result else None,
    }
    logger.info("%s", json.dumps(trace, ensure_ascii=True, sort_keys=True))
    if (
        result is not None and not result.text.strip()
        and result.order_action not in {"start_order", "start_new_order", "resend_order_form"}
    ):
        # This should be unreachable because AgentCore owns the non-empty
        # invariant. Keep an explicit error here so future regressions are
        # visible before they reach a channel adapter.
        logger.error(
            "AGENT_CORE_EMPTY_REPLY_INVARIANT trace=%s store=%s channel=%s message=%s",
            result.trace_id, body.store_id, body.channel, body.message_id,
        )
    return {
        "ok": True,
        "reply": asdict(result) if result else None,
        "metrics": {
            "latency_total_ms": round((time.perf_counter() - started_at) * 1000),
            "luna_call_count": result.luna_call_count if result else 0,
            "embedding_call_count": result.embedding_call_count if result else 0,
            "rag_called": bool(result and result.rag_called),
            "llm_input_tokens": result.llm_input_tokens if result else 0,
            "llm_cached_input_tokens": result.llm_cached_input_tokens if result else 0,
            "store_brain_tokens": result.store_brain_tokens if result else 0,
            "llm_ttft_ms": result.llm_ttft_ms if result else None,
            "llm_output_tokens": result.llm_output_tokens if result else 0,
            "llm_latency_ms": result.llm_latency_ms if result else 0,
            "retrieval_latency_ms": result.retrieval_latency_ms if result else 0,
            "memory_called": bool(result and result.memory_called),
            "memory_updates_count": result.memory_updates_count if result else 0,
        },
    }


@lru_cache(maxsize=256)
def _instagram_store_context_for(business_id: str) -> StoreContext:
    """Cache an already verified channel binding inside one warm instance.

    The first request still proves the Pinecone registry mapping. Following DMs
    for that professional account avoid an otherwise redundant network fetch.
    The cache is cleared whenever this service writes a new binding.
    """
    binding = PineconeKnowledgeStore().channel_binding(
        channel=Channel.INSTAGRAM.value,
        channel_account_id=business_id,
    )
    if not binding:
        raise RuntimeError("Instagram channel is not bound to a verified Agent Core RAG")
    return StoreContext(
        merchant_account_id=binding["merchant_account_id"],
        store_id=binding["store_id"],
        store_name=binding["store_name"],
        channel=Channel.INSTAGRAM,
    )


def instagram_store_context() -> StoreContext:
    """Resolve a webhook tenant only through the durable verified binding."""
    business_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    if not business_id:
        raise RuntimeError("INSTAGRAM_BUSINESS_ACCOUNT_ID is not configured")
    return _instagram_store_context_for(business_id)


def instagram_identity_aliases() -> set[str]:
    """Return the two IDs Meta may legitimately use for the same IG account.

    Instagram Login uses a professional-account id for the Send API, while
    webhook delivery can identify that same account with its scoped-user id.
    Those are aliases of one merchant channel, not two customers.  We only
    accept the explicit IDs configured for this deployment; an arbitrary
    signed event from another account must still have its own durable binding.
    """
    return {
        value
        for value in (
            env("INSTAGRAM_BUSINESS_ACCOUNT_ID"),
            env("INSTAGRAM_SCOPED_USER_ID"),
        )
        if value
    }


def inbound_instagram_store_context(channel_account_id: str) -> StoreContext:
    """Resolve a webhook account without confusing Meta's two ID namespaces."""
    try:
        # Preferred path for every tenant: lookup the exact account id from
        # the event in the durable binding registry.
        return _instagram_store_context_for(channel_account_id)
    except RuntimeError:
        # The configured Instagram Login account is a verified alias pair.
        # Route its scoped webhook id to the binding held under its canonical
        # professional-account id. Do not fall back for any unrelated id.
        canonical_business_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        if (
            canonical_business_id
            and channel_account_id in instagram_identity_aliases()
        ):
            return _instagram_store_context_for(canonical_business_id)
        raise


@lru_cache(maxsize=1024)
def _channel_store_context_for(channel: str, channel_account_id: str) -> StoreContext:
    binding = PineconeKnowledgeStore().channel_binding(
        channel=channel,
        channel_account_id=channel_account_id,
    )
    if not binding:
        raise RuntimeError(f"{channel} channel is not bound to a verified Agent Core RAG")
    return StoreContext(
        merchant_account_id=binding["merchant_account_id"],
        store_id=binding["store_id"],
        store_name=binding["store_name"],
        channel=Channel(channel),
    )


def message_store_context(message: IncomingMessage) -> StoreContext:
    if message.channel is Channel.INSTAGRAM:
        account_id = message.channel_account_id or env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        return inbound_instagram_store_context(account_id)
    if message.channel in {Channel.MESSENGER, Channel.WHATSAPP} and message.channel_account_id:
        return _channel_store_context_for(message.channel.value, message.channel_account_id)
    raise RuntimeError("Inbound channel account cannot be resolved")


def instagram_identity_ids() -> set[str]:
    """Only the professional account id is our sender id.

    An Instagram-scoped user id identifies the customer inside this business'
    conversation. Treating that id as the business silently discarded every DM
    from the test customer before it reached the agent.
    """
    business_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    return {business_id} if business_id else set()


def parse_instagram_webhook(payload: dict[str, Any]) -> list[IncomingMessage]:
    """Extract inbound customer messages only; never react to our own echo."""
    if payload.get("object") != "instagram":
        return []
    own_ids = instagram_identity_ids()
    messages: list[IncomingMessage] = []
    for entry in payload.get("entry", []):
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id") or "")
        events: list[tuple[dict[str, Any], str]] = []
        comment_values: list[dict[str, Any]] = []
        for container_name in ("messaging", "standby"):
            container = entry.get(container_name) or []
            events.extend((value, entry_id) for value in container if isinstance(value, dict))
        if entry.get("field") in {"comments", "live_comments"}:
            value = entry.get("value") or {}
            if isinstance(value, list):
                comment_values.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                comment_values.append(value)
        for change in entry.get("changes", []) or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            if change.get("field") in {"comments", "live_comments"}:
                if isinstance(value, list):
                    comment_values.extend(item for item in value if isinstance(item, dict))
                elif isinstance(value, dict):
                    comment_values.append(value)
                continue
            if change.get("field") not in {"messages", "messaging"}:
                continue
            if not isinstance(value, dict):
                continue
            nested = value.get("messaging") or value.get("messages") or []
            if isinstance(nested, list):
                events.extend((item, entry_id) for item in nested if isinstance(item, dict))
            if isinstance(value.get("message"), dict):
                events.append((value, entry_id))

        for event, event_entry_id in events:
            if not isinstance(event, dict):
                continue
            message = event.get("message")
            sender = event.get("sender")
            recipient = event.get("recipient")
            if not isinstance(message, dict) or message.get("is_echo"):
                continue
            if not isinstance(sender, dict):
                continue
            sender_id = str(sender.get("id") or "")
            recipient_id = str(recipient.get("id") or "") if isinstance(recipient, dict) else ""
            message_id = str(message.get("mid") or "")
            # Instagram Login does not consistently set ``is_echo``. In the
            # webhook contract, entry.id is the professional account for this
            # batch: an event whose sender equals entry.id is our outbound echo.
            expected_recipient_ids = own_ids | ({event_entry_id} if event_entry_id else set())
            if (
                not sender_id
                or not message_id
                or sender_id in own_ids
                or (event_entry_id and sender_id == event_entry_id)
                # Some Instagram Login echoes omit ``is_echo`` and expose a
                # sender id that differs from the professional account id.
                # Direction is still unambiguous: a real inbound DM targets
                # this professional account; an outbound echo targets the
                # customer and must never reach Luna or the send API.
                or (
                    recipient_id
                    and expected_recipient_ids
                    and recipient_id not in expected_recipient_ids
                )
            ):
                continue
            attachments: list[Attachment] = []
            for item in message.get("attachments", []) or []:
                if not isinstance(item, dict):
                    continue
                attachment_payload = item.get("payload") or {}
                if not isinstance(attachment_payload, dict):
                    attachment_payload = {}
                url = str(
                    attachment_payload.get("url")
                    or attachment_payload.get("image_url")
                    or attachment_payload.get("video_url")
                    or ""
                )
                if url:
                    attachments.append(Attachment(
                        url=url,
                        mime_type=str(item.get("type") or "application/octet-stream"),
                        media_id=str(
                            item.get("id")
                            or attachment_payload.get("id")
                            or attachment_payload.get("attachment_id")
                            or ""
                        ),
                    ))
            messages.append(IncomingMessage(
                message_id=message_id,
                text=str(message.get("text") or "").strip(),
                customer_id=sender_id,
                attachments=tuple(attachments),
                channel=Channel.INSTAGRAM,
                channel_account_id=(recipient_id or event_entry_id or env("INSTAGRAM_BUSINESS_ACCOUNT_ID")),
                received_at=str(event.get("timestamp") or ""),
            ))
        for comment in comment_values:
            comment_id = str(comment.get("id") or "")
            author = comment.get("from") or {}
            author = author if isinstance(author, dict) else {}
            author_id = str(author.get("id") or comment.get("from_id") or "")
            author_username = str(author.get("username") or "")
            media = comment.get("media") or {}
            media = media if isinstance(media, dict) else {}
            media_id = str(media.get("id") or "")
            text = str(comment.get("text") or "").strip()
            # Public comments must never be folded into a customer's private
            # DM memory, even when Meta provides the same Instagram-scoped id.
            # We do however retain a tiny, isolated state bucket per commenter
            # and publication. It lets the agent answer once in public then
            # move a continued thread to PV, without ever reading DM history.
            commenter = author_id or author_username or comment_id
            thread_material = ":".join((entry_id, media_id or "unknown-media", commenter))
            thread_hash = hashlib.sha256(thread_material.encode("utf-8")).hexdigest()[:32]
            customer_id = f"instagram-comment-thread:{thread_hash}"
            if (
                not comment_id
                or not text
                # Meta marks a comment emitted by the professional account
                # with its self-scoped id. Never let our public reply become
                # a new inbound customer event.
                or bool(comment.get("self_ig_scoped_id"))
                or bool(author.get("self_ig_scoped_id"))
                or author_id in own_ids
                or (entry_id and author_id == entry_id)
            ):
                continue
            messages.append(IncomingMessage(
                message_id=comment_id,
                text=text,
                customer_id=customer_id,
                channel=Channel.INSTAGRAM,
                channel_account_id=(entry_id or env("INSTAGRAM_BUSINESS_ACCOUNT_ID")),
                received_at=str(entry.get("time") or ""),
                surface=ConversationSurface.INSTAGRAM_COMMENT,
                comment_id=comment_id,
            ))
    return messages


def parse_messenger_webhook(payload: dict[str, Any]) -> list[IncomingMessage]:
    """Extract customer messages from Messenger Page events with echo rejection."""
    if payload.get("object") != "page":
        return []
    messages: list[IncomingMessage] = []
    for entry in payload.get("entry", []) or []:
        if not isinstance(entry, dict):
            continue
        page_id = str(entry.get("id") or "")
        for container_name in ("messaging", "standby"):
            for event in entry.get(container_name, []) or []:
                if not isinstance(event, dict):
                    continue
                raw = event.get("message") or {}
                sender = event.get("sender") or {}
                recipient = event.get("recipient") or {}
                if not isinstance(raw, dict) or raw.get("is_echo") or not isinstance(sender, dict):
                    continue
                sender_id = str(sender.get("id") or "")
                recipient_id = str(recipient.get("id") or "") if isinstance(recipient, dict) else ""
                message_id = str(raw.get("mid") or "")
                if not sender_id or not message_id or sender_id == page_id or (recipient_id and recipient_id != page_id):
                    continue
                attachments: list[Attachment] = []
                for item in raw.get("attachments", []) or []:
                    if not isinstance(item, dict):
                        continue
                    attachment_payload = item.get("payload") or {}
                    url = str(attachment_payload.get("url") or "") if isinstance(attachment_payload, dict) else ""
                    if url:
                        attachments.append(Attachment(url=url, mime_type=str(item.get("type") or "application/octet-stream")))
                messages.append(IncomingMessage(
                    message_id=message_id,
                    text=str(raw.get("text") or "").strip(),
                    customer_id=sender_id,
                    attachments=tuple(attachments),
                    channel=Channel.MESSENGER,
                    channel_account_id=page_id,
                    received_at=str(event.get("timestamp") or ""),
                ))
    return messages


def parse_whatsapp_webhook(payload: dict[str, Any]) -> list[IncomingMessage]:
    """Extract WhatsApp Cloud API messages without sharing state across senders."""
    if payload.get("object") != "whatsapp_business_account":
        return []
    messages: list[IncomingMessage] = []
    for entry in payload.get("entry", []) or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes", []) or []:
            if not isinstance(change, dict) or change.get("field") != "messages":
                continue
            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue
            metadata = value.get("metadata") or {}
            phone_number_id = str(metadata.get("phone_number_id") or "") if isinstance(metadata, dict) else ""
            for raw in value.get("messages", []) or []:
                if not isinstance(raw, dict):
                    continue
                message_id = str(raw.get("id") or "")
                customer_id = str(raw.get("from") or "")
                message_type = str(raw.get("type") or "")
                content = raw.get(message_type) or {}
                text = ""
                if message_type == "text" and isinstance(content, dict):
                    text = str(content.get("body") or "")
                elif message_type in {"button", "interactive"} and isinstance(content, dict):
                    reply = content.get("button_reply") or content.get("list_reply") or content
                    text = str(reply.get("title") or reply.get("text") or "") if isinstance(reply, dict) else ""
                attachments: tuple[Attachment, ...] = ()
                if message_type in {"audio", "image", "video", "document"} and isinstance(content, dict):
                    attachments = (Attachment(
                        url="",
                        mime_type=str(content.get("mime_type") or message_type),
                        media_id=str(content.get("id") or ""),
                    ),)
                if message_id and customer_id and phone_number_id:
                    messages.append(IncomingMessage(
                        message_id=message_id,
                        text=text.strip(),
                        customer_id=customer_id,
                        attachments=attachments,
                        channel=Channel.WHATSAPP,
                        channel_account_id=phone_number_id,
                        received_at=str(raw.get("timestamp") or ""),
                    ))
    return messages


def parse_meta_messages(payload: dict[str, Any]) -> list[IncomingMessage]:
    parsed = [
        *parse_instagram_webhook(payload),
        *parse_messenger_webhook(payload),
        *parse_whatsapp_webhook(payload),
    ]
    # Meta can include the same DM in more than one container in a single
    # signed delivery. Dispatching both creates duplicate replies if one of
    # those representations uses an alias that is not a tenant binding.
    unique: dict[tuple[Channel, str], IncomingMessage] = {}
    for message in parsed:
        unique.setdefault((message.channel, message.message_id), message)
    return list(unique.values())


def _voice_attachment(message: IncomingMessage) -> Attachment | None:
    for attachment in message.attachments:
        media_type = attachment.mime_type.lower().split(";", 1)[0].strip()
        if media_type == "voice" or media_type == "audio" or media_type.startswith("audio/"):
            return attachment
    return None


def _audio_format(*, content_type: str, url: str, payload: bytes) -> str:
    media_type = content_type.lower().split(";", 1)[0].strip()
    formats = {
        "audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3",
        "audio/mp3": "mp3", "audio/flac": "flac", "audio/x-flac": "flac",
        "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/ogg": "ogg",
        "audio/webm": "webm", "audio/aac": "aac",
    }
    if media_type in formats:
        return formats[media_type]
    suffix = url.split("?", 1)[0].rsplit(".", 1)[-1].lower() if "." in url.split("?", 1)[0] else ""
    if suffix in {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}:
        return suffix
    if payload.startswith(b"OggS"):
        return "ogg"
    if payload.startswith(b"RIFF"):
        return "wav"
    if payload.startswith(b"fLaC"):
        return "flac"
    if payload.startswith(b"ID3") or payload[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "mp3"
    if len(payload) > 12 and payload[4:8] == b"ftyp":
        return "m4a"
    if payload.startswith(b"\x1aE\xdf\xa3"):
        return "webm"
    raise RuntimeError("Instagram voice format is unsupported")


async def transcribe_instagram_voice(
    *, store: StoreContext, message: IncomingMessage, attachment: Attachment,
) -> VoiceTranscription:
    """Download once and transcribe once, with durable retry idempotency."""
    require_env("OPENROUTER_API_KEY", "INSTAGRAM_ACCESS_TOKEN")
    knowledge = PineconeKnowledgeStore()
    lock_key = hashlib.sha256(f"{store.store_id}:{message.message_id}".encode()).hexdigest()
    lock = _VOICE_SINGLEFLIGHT.setdefault(lock_key, asyncio.Lock())
    async with lock:
        cached = await asyncio.to_thread(
            knowledge.voice_transcription,
            store_id=store.store_id,
            customer_id=message.customer_id,
            message_id=message.message_id,
        )
        if cached:
            return cached

        download_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=httpx.Timeout(12, connect=5), follow_redirects=True) as client:
            response = await client.get(
                attachment.url,
                headers={"Authorization": f"Bearer {env('INSTAGRAM_ACCESS_TOKEN')}"},
            )
        if response.status_code != 200:
            raise RuntimeError(f"Instagram voice download failed (HTTP {response.status_code})")
        announced_size = int(response.headers.get("content-length") or 0)
        if announced_size > _MAX_VOICE_BYTES or len(response.content) > _MAX_VOICE_BYTES:
            raise RuntimeError("Instagram voice exceeds the bounded transcription size")
        audio = bytes(response.content)
        if not audio:
            raise RuntimeError("Instagram voice download was empty")
        download_ms = round((time.perf_counter() - download_started) * 1000)
        audio_format = _audio_format(
            content_type=response.headers.get("content-type", ""),
            url=attachment.url,
            payload=audio,
        )

        transcription_started = time.perf_counter()
        stt_url = f"{(env('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1').rstrip('/')}/audio/transcriptions"
        async with httpx.AsyncClient(timeout=httpx.Timeout(25, connect=5)) as client:
            stt_response = await client.post(
                stt_url,
                headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}", "Content-Type": "application/json"},
                json={
                    "model": "openai/gpt-transcribe",
                    "input_audio": {
                        "data": base64.b64encode(audio).decode("ascii"),
                        "format": audio_format,
                    },
                    "temperature": 0,
                    # `json` is supported on the base64 STT route by every
                    # OpenRouter transcription provider. `verbose_json` is
                    # rejected with HTTP 400 by providers that do not expose
                    # word/segment metadata for GPT Transcribe.
                    "response_format": "json",
                },
            )
        if stt_response.status_code != 200:
            detail = stt_response.text.replace("\n", " ").strip()[:500]
            logger.warning(
                "Instagram voice transcription rejected status=%s detail=%s",
                stt_response.status_code,
                detail,
            )
            raise RuntimeError(f"Instagram voice transcription failed (HTTP {stt_response.status_code})")
        payload = stt_response.json() if stt_response.content else {}
        transcript = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        if not transcript:
            raise RuntimeError("Instagram voice transcription was empty")
        usage = payload.get("usage") if isinstance(payload, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        result = VoiceTranscription(
            transcript_original=transcript,
            spoken_language=str(payload.get("language") or ""),
            duration_seconds=float(payload.get("duration") or usage.get("seconds") or 0.0),
            model="openai/gpt-transcribe",
            provider_cost=float(usage.get("cost") or 0.0),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            media_download_ms=download_ms,
            transcription_ms=round((time.perf_counter() - transcription_started) * 1000),
        )
        await asyncio.to_thread(
            knowledge.cache_voice_transcription,
            store_id=store.store_id,
            customer_id=message.customer_id,
            message_id=message.message_id,
            media_id=attachment.media_id,
            transcription=result,
        )
        return result


def instagram_webhook_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Safe structural diagnostics: never log ids, message text or attachments."""
    entries = [entry for entry in payload.get("entry", []) or [] if isinstance(entry, dict)]
    return {
        "object": str(payload.get("object") or ""),
        "entries": len(entries),
        "messaging": sum(len(entry.get("messaging") or []) for entry in entries),
        "standby": sum(len(entry.get("standby") or []) for entry in entries),
        "change_fields": sorted({
            str(change.get("field") or "")
            for entry in entries
            for change in entry.get("changes", []) or []
            if isinstance(change, dict)
        }),
    }


async def send_instagram_text(*, recipient_id: str, text: str) -> str:
    """Send a customer-ready reply through Instagram API with Instagram Login."""
    require_env("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    url = (
        f"https://graph.instagram.com/{graph_version}/"
        f"{quote(env('INSTAGRAM_BUSINESS_ACCOUNT_ID'), safe='')}/messages"
    )
    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {env('INSTAGRAM_ACCESS_TOKEN')}"},
            json={"recipient": {"id": recipient_id}, "message": {"text": text}},
        )
    if response.status_code not in (200, 201):
        body = response.json() if response.content else {}
        error = _meta_error(body)
        # Do not return Meta details to the webhook caller; the protected log
        # keeps only the provider's bounded error code/type/message.
        raise RuntimeError(
            f"Instagram send failed (HTTP {response.status_code}, code={error['code']}, "
            f"type={error['type']}, message={error['message']})"
        )
    body = response.json()
    return str(body.get("message_id") or "")


async def send_instagram_comment_reply(*, comment_id: str, text: str) -> str:
    """Post a public reply under the exact Instagram comment that triggered it."""
    require_env("INSTAGRAM_ACCESS_TOKEN")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    url = f"https://graph.instagram.com/{graph_version}/{quote(comment_id, safe='')}/replies"
    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {env('INSTAGRAM_ACCESS_TOKEN')}"},
            json={"message": text},
        )
    if response.status_code not in (200, 201):
        body = response.json() if response.content else {}
        error = _meta_error(body)
        raise RuntimeError(
            f"Instagram comment reply failed (HTTP {response.status_code}, code={error['code']}, "
            f"type={error['type']}, message={error['message']})"
        )
    body = response.json() if response.content else {}
    return str(body.get("id") or "")


async def send_instagram_comment_private_reply(
    *, instagram_account_id: str, comment_id: str, text: str,
) -> str:
    """Send Meta's one allowed private reply for an Instagram comment."""
    require_env("INSTAGRAM_ACCESS_TOKEN")
    account_id = instagram_account_id or env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("Instagram business account id is not configured")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    url = (
        f"https://graph.instagram.com/{graph_version}/"
        f"{quote(account_id, safe='')}/messages"
    )
    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {env('INSTAGRAM_ACCESS_TOKEN')}"},
            json={
                "recipient": {"comment_id": comment_id},
                "message": {"text": text},
            },
        )
    if response.status_code not in (200, 201):
        body = response.json() if response.content else {}
        error = _meta_error(body)
        raise RuntimeError(
            f"Instagram private comment reply failed (HTTP {response.status_code}, code={error['code']}, "
            f"type={error['type']}, message={error['message']})"
        )
    body = response.json() if response.content else {}
    return str(body.get("message_id") or "")


_COMMENT_DM_ACKNOWLEDGEMENTS: dict[ReplyScript, tuple[str, ...]] = {
    ReplyScript.LATIN_DARIJA: (
        "Jwabtk f DM.",
        "Ra sift lik ljawab f DM.",
        "Ljawab siftou lik f DM.",
    ),
    ReplyScript.ARABIC_DARIJA: (
        "جاوبتك فالـDM.",
        "راه صيفطت ليك الجواب فالـDM.",
        "الجواب صيفطتو ليك فالـDM.",
    ),
}


def instagram_comment_dm_acknowledgement(message: IncomingMessage) -> str:
    """Acknowledge the already-sent private answer, without a canned PV pitch."""
    script = detect_reply_script(message.text)
    variants = _COMMENT_DM_ACKNOWLEDGEMENTS[script]
    digest = hashlib.blake2s(message.message_id.encode("utf-8"), digest_size=2).digest()
    return variants[int.from_bytes(digest, "big") % len(variants)]


async def send_messenger_text(*, page_id: str, recipient_id: str, text: str) -> str:
    token = env("MESSENGER_ACCESS_TOKEN") or env("FB_PAGE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Messenger access token is not configured")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.post(
            f"https://graph.facebook.com/{graph_version}/{quote(page_id, safe='')}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"recipient": {"id": recipient_id}, "message": {"text": text}},
        )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Messenger send failed (HTTP {response.status_code})")
    payload = response.json() if response.content else {}
    return str(payload.get("message_id") or "")


async def send_whatsapp_text(*, phone_number_id: str, recipient_id: str, text: str) -> str:
    token = env("WHATSAPP_ACCESS_TOKEN") or env("META_SYSTEM_USER_TOKEN")
    if not token:
        raise RuntimeError("WhatsApp access token is not configured")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.post(
            f"https://graph.facebook.com/{graph_version}/{quote(phone_number_id, safe='')}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messaging_product": "whatsapp",
                "to": recipient_id,
                "type": "text",
                "text": {"body": text},
            },
        )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"WhatsApp send failed (HTTP {response.status_code})")
    payload = response.json() if response.content else {}
    messages = payload.get("messages") if isinstance(payload, dict) else []
    return str(messages[0].get("id") or "") if messages else ""


async def send_channel_text(*, message: IncomingMessage, text: str) -> str:
    if message.channel is Channel.INSTAGRAM:
        if message.surface == ConversationSurface.INSTAGRAM_COMMENT:
            if not message.comment_id:
                raise RuntimeError("Instagram comment reply has no comment id")
            return await send_instagram_comment_reply(comment_id=message.comment_id, text=text)
        return await send_instagram_text(recipient_id=message.customer_id, text=text)
    if message.channel is Channel.MESSENGER:
        return await send_messenger_text(
            page_id=message.channel_account_id, recipient_id=message.customer_id, text=text,
        )
    if message.channel is Channel.WHATSAPP:
        return await send_whatsapp_text(
            phone_number_id=message.channel_account_id, recipient_id=message.customer_id, text=text,
        )
    raise RuntimeError("Unsupported outbound channel")


async def deliver_instagram_comment_answer(
    *, knowledge_store: PineconeKnowledgeStore, store: StoreContext,
    message: IncomingMessage, private_answer: str,
) -> int:
    """Deliver the answer privately, then acknowledge it under the comment.

    Meta permits only one initial private reply per comment. Mark it durable as
    sent immediately after that call so a later public-ack failure can never
    cause a second private-reply attempt on a webhook retry.
    """
    if not message.comment_id:
        raise RuntimeError("Instagram private comment reply has no comment id")
    await send_instagram_comment_private_reply(
        instagram_account_id=message.channel_account_id,
        comment_id=message.comment_id,
        text=private_answer,
    )
    await asyncio.to_thread(
        knowledge_store.mark_customer_message_sent,
        store=store,
        message=message,
    )
    try:
        await send_channel_text(
            message=message,
            text=instagram_comment_dm_acknowledgement(message),
        )
    except Exception:
        # The customer already has the actual answer. Do not retry the private
        # reply merely because the optional public acknowledgement failed.
        logger.exception("Instagram comment acknowledgement failed after private answer")
    return 1


async def channel_conversation_history(
    *, store: StoreContext, message: IncomingMessage,
) -> tuple[ConversationTurn, ...]:
    if message.surface == ConversationSurface.INSTAGRAM_COMMENT:
        # Public comments are isolated from private DMs and customer memory.
        return ()
    if message.channel is Channel.INSTAGRAM:
        return await instagram_conversation_history(
            customer_id=message.customer_id,
            current_message_id=message.message_id,
            store_id=store.store_id,
        )
    return await asyncio.to_thread(
        PineconeKnowledgeStore().recent_customer_history,
        store=store,
        customer_id=message.customer_id,
        limit=16,
    )


async def instagram_conversation_history(
    *, customer_id: str, current_message_id: str, store_id: str = "", limit: int = 40
) -> tuple[ConversationTurn, ...]:
    """Load the real customer thread so Luna never reasons from one DM alone.

    Instagram conversations are the durable source of truth here. A failed
    history read must not silence the current reply, so callers receive an
    empty history while the bounded operational error remains in server logs.
    """
    require_env("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    business_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    url = f"https://graph.instagram.com/{graph_version}/{quote(business_id, safe='')}/conversations"
    params = {
        "platform": "instagram",
        "user_id": customer_id,
        "fields": "id,updated_time,messages.limit(50){id,created_time,from,to,message}",
        "access_token": env("INSTAGRAM_ACCESS_TOKEN"),
    }
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(url, params=params)
        payload = response.json() if response.content else {}
        if response.status_code != 200:
            error = _meta_error(payload)
            logger.warning(
                "Instagram history unavailable status=%s code=%s",
                response.status_code,
                error["code"],
            )
            return ()
    except Exception:
        logger.exception("Instagram history read failed")
        return ()

    ordered_raw: list[tuple[str, str, str, str]] = []
    conversations = payload.get("data") if isinstance(payload, dict) else []
    for conversation in conversations or []:
        if not isinstance(conversation, dict):
            continue
        container = conversation.get("messages") or {}
        raw_messages = container.get("data") if isinstance(container, dict) else []
        for raw in raw_messages or []:
            if not isinstance(raw, dict):
                continue
            message_id = str(raw.get("id") or "")
            text = str(raw.get("message") or "").strip()
            sender = raw.get("from") or {}
            sender_id = str(sender.get("id") or "") if isinstance(sender, dict) else ""
            if message_id == current_message_id:
                continue
            if sender_id == business_id:
                role = "assistant"
            elif sender_id == customer_id:
                role = "customer"
            else:
                continue
            ordered_raw.append((
                str(raw.get("created_time") or ""),
                message_id,
                role,
                text,
            ))

    # Instagram message history exposes voice notes without text. Recover the
    # original source transcript from the tenant-scoped durable STT cache so a
    # later turn has the same conversational continuity as written messages.
    empty_customer_ids = [
        message_id for _, message_id, role, text in ordered_raw
        if role == "customer" and not text
    ]
    voice_texts: dict[str, str] = {}
    if store_id and empty_customer_ids:
        try:
            voice_texts = await asyncio.to_thread(
                PineconeKnowledgeStore().voice_transcriptions_for_messages,
                store_id=store_id,
                customer_id=customer_id,
                message_ids=empty_customer_ids,
            )
        except Exception:
            logger.exception("Instagram voice history cache read failed")

    ordered: list[tuple[str, str, ConversationTurn]] = []
    for created_at, message_id, role, text in ordered_raw:
        final_text = text or voice_texts.get(message_id, "")
        if final_text:
            source = "instagram_voice" if role == "customer" and not text and message_id in voice_texts else "text"
            ordered.append((created_at, message_id, ConversationTurn(role, final_text, source=source)))

    # Meta returns recent messages in reverse order. Timestamp + id gives a
    # deterministic order even when two messages share the same second.
    ordered.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ordered[-max(1, min(limit, 50)):])


@app.post("/admin/instagram/replay-latest")
async def replay_latest_instagram_dm(request: Request) -> dict[str, Any]:
    """Recover the latest unanswered inbound DM after a webhook configuration fix."""
    _require_admin(request)
    require_env("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID")
    graph_version = env("META_GRAPH_VERSION") or "v25.0"
    business_id = env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    url = f"https://graph.instagram.com/{graph_version}/{quote(business_id, safe='')}/conversations"
    params = {
        "platform": "instagram",
        "fields": "id,updated_time,messages.limit(20){id,created_time,from,to,message}",
        "limit": "50",
        "access_token": env("INSTAGRAM_ACCESS_TOKEN"),
    }
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.get(url, params=params)
    payload = response.json() if response.content else {}
    if response.status_code != 200:
        error = _meta_error(payload)
        logger.warning("Instagram conversation recovery failed status=%s code=%s", response.status_code, error["code"])
        raise HTTPException(502, f"Instagram conversation recovery failed (HTTP {response.status_code})")

    candidates: list[tuple[str, dict[str, Any]]] = []
    conversations = payload.get("data") if isinstance(payload, dict) else []
    for conversation in conversations or []:
        if not isinstance(conversation, dict):
            continue
        message_container = conversation.get("messages") or {}
        raw_messages = message_container.get("data") if isinstance(message_container, dict) else []
        for raw_message in raw_messages or []:
            if not isinstance(raw_message, dict):
                continue
            sender = raw_message.get("from") or {}
            sender_id = str(sender.get("id") or "") if isinstance(sender, dict) else ""
            message_id = str(raw_message.get("id") or "")
            text = str(raw_message.get("message") or "").strip()
            if not sender_id or sender_id == business_id or not message_id or not text:
                continue
            candidates.append((str(raw_message.get("created_time") or ""), {
                "message_id": message_id,
                "customer_id": sender_id,
                "text": text,
            }))
    if not candidates:
        return {"ok": False, "reason": "no_inbound_text_message", "conversations": len(conversations or [])}

    _, latest = max(candidates, key=lambda candidate: candidate[0])
    context = instagram_store_context()
    agent = AgentCore(knowledge_store=PineconeKnowledgeStore(), model=OpenRouterLunaModel())
    history = await instagram_conversation_history(
        customer_id=latest["customer_id"],
        current_message_id=latest["message_id"],
        store_id=context.store_id,
    )
    reply = agent.reply(store=context, message=IncomingMessage(**latest, history=history))
    if not reply:
        return {"ok": False, "reason": "agent_suppressed"}
    sent_message_id = await send_instagram_text(recipient_id=latest["customer_id"], text=reply.text)
    logger.info("Instagram latest DM replay delivered trace=%s", reply.trace_id)
    return {
        "ok": True,
        "used_rag": reply.used_rag,
        "reason": reply.reason,
        "reply": reply.text,
        "sent": bool(sent_message_id),
    }


@app.get("/webhooks/meta")
async def verify_meta_webhook(
    mode: str = Query(alias="hub.mode"),
    verify_token: str = Query(alias="hub.verify_token"),
    challenge: str = Query(alias="hub.challenge"),
):
    expected = env("META_AGENT_CORE_VERIFY_TOKEN")
    if mode != "subscribe" or not expected or not hmac.compare_digest(verify_token, expected):
        raise HTTPException(403, "Webhook verification failed")
    return PlainTextResponse(challenge)


async def _dispatch_channel_message(message: IncomingMessage) -> int:
    """Process one DM; different customers run concurrently, one thread stays ordered."""
    context_started_at = time.perf_counter()
    try:
        context = message_store_context(message)
    except RuntimeError:
        account_fingerprint = hashlib.sha256(message.channel_account_id.encode()).hexdigest()[:12]
        logger.warning(
            "Ignoring unbound channel event channel=%s account_fingerprint=%s event=%s",
            message.channel.value,
            account_fingerprint,
            message.message_id,
        )
        # A 2xx response is deliberate: a retry cannot acquire a missing
        # tenant binding and would otherwise duplicate any sibling event in
        # the same Meta delivery.
        return 0
    context_ms = round((time.perf_counter() - context_started_at) * 1000)
    lock_key = f"{context.merchant_account_id}:{message.channel.value}:{message.customer_id}"
    conversation_lock = _CONVERSATION_LOCKS.setdefault(lock_key, asyncio.Lock())
    async with conversation_lock:
        knowledge_store = PineconeKnowledgeStore()
        durable_result = await asyncio.to_thread(
            knowledge_store.customer_message_result,
            store=context,
            message=message,
        )
        if durable_result:
            cached_reply, already_sent = durable_result
            if already_sent:
                return 0
            if message.surface == ConversationSurface.INSTAGRAM_COMMENT:
                return await deliver_instagram_comment_answer(
                    knowledge_store=knowledge_store,
                    store=context,
                    message=message,
                    private_answer=cached_reply,
                )
            await send_channel_text(message=message, text=cached_reply)
            await asyncio.to_thread(
                knowledge_store.mark_customer_message_sent,
                store=context,
                message=message,
            )
            return 1
        deduplication_id = f"{context.merchant_account_id}:{message.channel.value}:{message.message_id}"
        if deduplication_id in _PROCESSED_INSTAGRAM_MESSAGE_IDS:
            return 0
        _PROCESSED_INSTAGRAM_MESSAGE_IDS.add(deduplication_id)
        if len(_PROCESSED_INSTAGRAM_MESSAGE_IDS) > _MAX_DEDUPLICATION_IDS:
            _PROCESSED_INSTAGRAM_MESSAGE_IDS.clear()
            _PROCESSED_INSTAGRAM_MESSAGE_IDS.add(deduplication_id)

        started_at = time.perf_counter()
        history_ms = agent_ms = send_ms = 0
        try:
            voice = _voice_attachment(message) if message.channel is Channel.INSTAGRAM else None
            phase_started_at = time.perf_counter()
            history_task = asyncio.create_task(channel_conversation_history(
                store=context,
                message=message,
            ))
            transcription_task = (
                asyncio.create_task(transcribe_instagram_voice(
                    store=context, message=message, attachment=voice,
                ))
                if voice else None
            )
            if transcription_task:
                try:
                    history, transcription = await asyncio.gather(history_task, transcription_task)
                except Exception:
                    logger.exception("Instagram voice could not be transcribed")
                    history = await history_task
                    recent_customer_text = next(
                        (turn.text for turn in reversed(history) if turn.role == "customer" and turn.text.strip()),
                        "",
                    )
                    fallback_script = detect_reply_script(recent_customer_text)
                    fallback = (
                        "ما قدرتش نسمع الفوكال مزيان. عافاك عاود صيفطو ولا كتب ليا الرسالة."
                        if fallback_script is ReplyScript.ARABIC_DARIJA
                        else "Ma 9dertch nsme3 lvocal mzyan. 3afak 3awd sifto ola kteb lia message."
                    )
                    await send_channel_text(message=message, text=fallback)
                    return 1
                if transcription.completed:
                    logger.info("Instagram voice retry already completed cache_hit=true")
                    return 0
                source_text = transcription.transcript_original
                source = "instagram_voice"
                spoken_language = transcription.spoken_language
            else:
                history = await history_task
                transcription = None
                source_text = message.text
                source = "text"
                spoken_language = ""
            history_ms = round((time.perf_counter() - phase_started_at) * 1000)
            enriched_message = IncomingMessage(
                message_id=message.message_id,
                text=source_text,
                customer_id=message.customer_id,
                attachments=message.attachments,
                history=history,
                source=source,
                spoken_language=spoken_language,
                channel=message.channel,
                channel_account_id=message.channel_account_id,
                received_at=message.received_at,
                surface=message.surface,
                comment_id=message.comment_id,
            )
            agent = AgentCore(knowledge_store=knowledge_store, model=OpenRouterLunaModel())
            phase_started_at = time.perf_counter()
            reply = agent.reply(store=context, message=enriched_message)
            agent_ms = round((time.perf_counter() - phase_started_at) * 1000)
            if not reply:
                return 0
            if message.surface == ConversationSurface.INSTAGRAM_COMMENT:
                delivered = await deliver_instagram_comment_answer(
                    knowledge_store=knowledge_store,
                    store=context,
                    message=message,
                    private_answer=reply.text,
                )
                total_ms = round((time.perf_counter() - started_at) * 1000)
                logger.info(
                    "Instagram comment answer delivered privately trace=%s total_ms=%s context_ms=%s history_ms=%s agent_ms=%s luna_calls=%s embedding_calls=%s rag_called=%s used_rag=%s",
                    reply.trace_id, total_ms, context_ms, history_ms, agent_ms,
                    reply.luna_call_count, reply.embedding_call_count,
                    reply.rag_called, reply.used_rag,
                )
                return delivered
            phase_started_at = time.perf_counter()
            await send_channel_text(message=message, text=reply.text)
            send_ms = round((time.perf_counter() - phase_started_at) * 1000)
            await asyncio.to_thread(
                knowledge_store.mark_customer_message_sent,
                store=context,
                message=message,
            )
            if transcription:
                await asyncio.to_thread(
                    PineconeKnowledgeStore().mark_voice_completed,
                    store_id=context.store_id,
                    customer_id=message.customer_id,
                    message_id=message.message_id,
                )
            total_ms = round((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Instagram Agent Core reply delivered trace=%s total_ms=%s context_ms=%s history_ms=%s agent_ms=%s send_ms=%s voice=%s stt_cache_hit=%s download_ms=%s transcription_ms=%s luna_calls=%s embedding_calls=%s rag_called=%s used_rag=%s memory_called=%s memory_updates=%s llm_input_tokens=%s llm_output_tokens=%s",
                reply.trace_id, total_ms, context_ms, history_ms, agent_ms, send_ms,
                bool(transcription),
                bool(transcription and transcription.cache_hit),
                transcription.media_download_ms if transcription else 0,
                transcription.transcription_ms if transcription else 0,
                reply.luna_call_count, reply.embedding_call_count, reply.rag_called,
                reply.used_rag, reply.memory_called, reply.memory_updates_count,
                reply.llm_input_tokens, reply.llm_output_tokens,
            )
            return 1
        except Exception:
            total_ms = round((time.perf_counter() - started_at) * 1000)
            logger.exception(
                "Instagram webhook dispatch failed for event=%s total_ms=%s context_ms=%s history_ms=%s agent_ms=%s send_ms=%s",
                message.message_id, total_ms, context_ms, history_ms, agent_ms, send_ms,
            )
            return 0


@app.post("/webhooks/meta")
async def receive_meta_webhook(request: Request) -> dict[str, Any]:
    # Meta signs the exact raw request body with the App Secret. The callback
    # therefore rejects spoofed events before any future channel dispatcher sees
    # message data.
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    signing_secrets = tuple(dict.fromkeys(
        value for value in (
            env("INSTAGRAM_APP_SECRET"), env("META_APP_SECRET"), env("FB_APP_SECRET")
        ) if value
    ))
    signature_valid = bool(signature) and any(
        hmac.compare_digest(
            signature,
            "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest(),
        )
        for secret in signing_secrets
    )
    if not signature_valid:
        logger.warning(
            "Meta webhook signature rejected signature_present=%s sha256_format=%s body_bytes=%s configured_signing_secrets=%s",
            bool(signature),
            signature.startswith("sha256="),
            len(raw_body),
            len(signing_secrets),
        )
        raise HTTPException(403, "Invalid Meta webhook signature")

    # Scaliffy owns the durable per-merchant channel bindings and encrypted
    # outbound tokens.  When the shared Core is configured as the Meta callback
    # (the legacy/transition setup), it must relay the *verified raw event* to
    # that channel adapter instead of trying to resolve a global Core binding.
    #
    # Forward the original signature too: the SaaS adapter verifies the same
    # Meta-signed bytes before it maps recipient -> merchant store.  A failed
    # relay deliberately returns 503 so Meta retries; acknowledging it here
    # would silently drop the customer's message.
    scaliffy_webhook_url = env("SCALIFFY_SAAS_META_WEBHOOK_URL").strip()
    if scaliffy_webhook_url:
        forward_headers = {"Content-Type": "application/json"}
        if signature:
            forward_headers["X-Hub-Signature-256"] = signature
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                forwarded = await client.post(
                    scaliffy_webhook_url,
                    content=raw_body,
                    headers=forward_headers,
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "Meta webhook relay unavailable target_configured=true error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(503, "Scaliffy channel relay unavailable") from exc
        if not 200 <= forwarded.status_code < 300:
            logger.warning(
                "Meta webhook relay rejected target_configured=true status=%s",
                forwarded.status_code,
            )
            raise HTTPException(503, "Scaliffy channel relay rejected event")
        logger.info("Meta webhook relayed to Scaliffy channel adapter")
        return {"received": True, "forwarded": True}

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid JSON webhook body") from exc

    messages = parse_meta_messages(payload)
    logger.info(
        "Meta channel webhook accepted shape=%s inbound_messages=%s",
        instagram_webhook_shape(payload),
        len(messages),
    )
    delivered = sum(await asyncio.gather(*(
        _dispatch_channel_message(message) for message in messages
    ))) if messages else 0
    logger.info("Meta Instagram webhook completed delivered=%s", delivered)
    return {"received": True, "delivered": delivered}
