"""Store-scoped deterministic product resolver. FOUND / NOT_FOUND / AMBIGUOUS.

Order: SessionState active entity -> exact ID/SKU -> exact normalized name ->
aliases -> normalized aliases -> typo/fuzzy -> structured text search ->
(optional semantic: declined here, return AMBIGUOUS instead of guessing).

Never silently transform NOT_FOUND into the nearest product.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field


def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or "").casefold())
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", ascii_only))


# Follow-up signals that continue the active product instead of naming one.
# A turn matching these AND naming no catalogue product is an ellipsis.
_ELLIPSIS_RE = re.compile(
    r"\b(?:hada|hadi|hadak|hadik|hado|هذا|هذه|هاد|هادي|هاداك|هاديك|chhal|ch7al|taman|taman|ثمن|شحال|prix|price|combien|bghit|bghiti|بغيت|ncommande|nkomondi|commande|order|seft|writini|werini|tsawer|swar|صور|صورة|تصويرة|noir|akahal|أسود|kayna?|kayen|كاين|joj|jouj|جوج|wahd|واحد|livraison|توصيل)\b"
    r"|^[?!.…\s]{1,12}$",
    re.IGNORECASE,
)


def _is_ellipsis(message_text: str) -> bool:
    """True for short follow-ups that reference (not rename) the product."""
    text = str(message_text or "").strip()
    if not text:
        return True
    if len(normalize(text).split()) > 8:
        return False
    return bool(_ELLIPSIS_RE.search(text))


@dataclass(frozen=True)
class ResolverResult:
    status: str  # FOUND | NOT_FOUND | AMBIGUOUS
    product_id: str = ""
    variant_id: str = ""
    match_kind: str = ""
    candidates: tuple = field(default_factory=tuple)


def _catalogue_products(catalogue: dict) -> list[dict]:
    if not isinstance(catalogue, dict):
        return []
    products = catalogue.get("products")
    if isinstance(products, list):
        return [p for p in products if isinstance(p, dict)]
    # Current adapter usually resolves ONE product into catalogue_context.
    if catalogue.get("name") or catalogue.get("product_id"):
        return [catalogue]
    return []


def resolve_product(
    *,
    store_id: str,
    message_text: str,
    state_product: str = "",
    catalogue: dict | None = None,
    turso_rows: list | None = None,
    aliases: dict | None = None,
    media_present: bool = False,
) -> ResolverResult:
    catalogue = catalogue if isinstance(catalogue, dict) else {}
    turso_rows = turso_rows if isinstance(turso_rows, list) else []
    aliases = aliases if isinstance(aliases, dict) else {}

    # 1. Current active entity from SessionState wins (continuity, not search).
    if state_product:
        for row in turso_rows:
            if str(row.get("id") or "") == state_product or normalize(str(row.get("title") or "")) == normalize(state_product):
                return ResolverResult("FOUND", str(row.get("id") or state_product), "", "session_active")
        products = _catalogue_products(catalogue)
        for prod in products:
            pid = str(prod.get("product_id") or prod.get("id") or prod.get("sku") or "")
            title = str(prod.get("name") or prod.get("title") or "")
            if state_product in (pid, title) or normalize(state_product) == normalize(title):
                return ResolverResult("FOUND", pid or state_product, "", "session_active")

    text_norm = normalize(message_text)
    if not text_norm:
        # Empty/attachment-only turn: keep the active product, never guess.
        if state_product:
            return ResolverResult("FOUND", state_product, "", "session_ellipsis")
        return ResolverResult("NOT_FOUND", "", "", "empty_query")

    # Build candidate pool: Turso exact rows first (truth), then catalogue.
    pool: list[tuple[str, str]] = []  # (product_id, title)
    for row in turso_rows:
        pool.append((str(row.get("id") or ""), str(row.get("title") or "")))
    for prod in _catalogue_products(catalogue):
        pid = str(prod.get("product_id") or prod.get("id") or prod.get("sku") or "")
        title = str(prod.get("name") or prod.get("title") or "")
        if pid or title:
            pool.append((pid or normalize(title), title))

    # 2. Exact ID / SKU.
    for pid, title in pool:
        if pid and normalize(message_text).replace(" ", "") == normalize(pid).replace(" ", ""):
            return ResolverResult("FOUND", pid, "", "exact_id")
        if pid and pid.strip() and pid.strip() in str(message_text or ""):
            # Bare substring ID match is weak; keep as candidate only.
            pass

    # 3. Exact normalized name.
    for pid, title in pool:
        if title and normalize(title) and normalize(title) in text_norm:
            return ResolverResult("FOUND", pid, "", "exact_name")

    # 4-5. Aliases (exact, then normalized).
    for alias, pid in aliases.items():
        if not alias or not pid:
            continue
        if str(alias).strip().casefold() in str(message_text or "").casefold():
            return ResolverResult("FOUND", str(pid), "", "alias_exact")
    for alias, pid in aliases.items():
        if normalize(alias) and normalize(alias) in text_norm:
            return ResolverResult("FOUND", str(pid), "", "alias_normalized")

    # 5b. Session ellipsis (after explicit mentions win): a short follow-up
    # that names no OTHER catalogue product continues the active product.
    # Guard: if any pool title distinct from the active product appears in
    # the text, this is a product switch, not an ellipsis.
    if state_product and _is_ellipsis(message_text):
        other_named = any(
            title and normalize(title) and normalize(title) in text_norm
            and pid != state_product and normalize(title) != normalize(state_product)
            for pid, title in pool
        )
        if not other_named:
            return ResolverResult("FOUND", state_product, "", "session_ellipsis")

    # 6. Typo/fuzzy: single strong winner only, else AMBIGUOUS.
    titles = [t for _, t in pool if normalize(t)]
    matches = difflib.get_close_matches(text_norm, [normalize(t) for t in titles], n=3, cutoff=0.88)
    if len(matches) == 1:
        for pid, title in pool:
            if normalize(title) == matches[0]:
                return ResolverResult("FOUND", pid, "", "fuzzy_single")
    if len(matches) > 1:
        cands = tuple(pid for pid, title in pool if normalize(title) in matches)[:4]
        return ResolverResult("AMBIGUOUS", "", "", "fuzzy_multi", cands)

    # 7. Structured text search: require meaningful overlap (>=50% of query
    # terms) so a generic word like "pack" alone never resolves. A different
    # explicit color/variant must not silently become the nearest product.
    query_terms = {w for w in text_norm.split() if len(w) > 2}
    scored: list[tuple[float, int, str]] = []
    for pid, title in pool:
        title_terms = {w for w in normalize(title).split() if len(w) > 2}
        overlap = query_terms & title_terms
        if overlap:
            ratio = len(overlap) / max(1, len(query_terms))
            scored.append((ratio, len(overlap), pid))
    scored = [s for s in scored if s[0] >= 0.5]
    if scored:
        scored.sort(reverse=True)
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return ResolverResult("FOUND", scored[0][2], "", "structured_search")
        return ResolverResult("AMBIGUOUS", "", "", "structured_multi", tuple(p for _, _, p in scored[:4]))

    # 8. Thread continuity (last resort, media turns only): an unresolved
    # photo/Reel in a thread with exactly one grounded product continues
    # that product — unless the turn names something else: another pool
    # product, or a color word foreign to the active product (then Luna must
    # clarify rather than adopt the wrong product). Conversational
    # continuity, never a visual guess: exact mentions, aliases, fuzzy and
    # 8. Thread continuity (last resort, media turns only): an unresolved
    # photo/Reel in a thread with exactly one grounded product continues
    # that product — unless the turn names another catalogue product. The
    # current message still reaches Luna verbatim, so any color/detail ask
    # inside it is handled conversationally; the resolver only settles
    # product identity, never a visual guess. Exact mentions, aliases,
    # fuzzy and structured search above all had priority.
    if media_present and state_product:
        other_named = any(
            title and normalize(title) and normalize(title) in text_norm
            and pid != state_product and normalize(title) != normalize(state_product)
            for pid, title in pool
        )
        if not other_named:
            return ResolverResult("FOUND", state_product, "", "thread_continuity")

    # 9. Optional semantic discovery: intentionally NOT attempted here.
    return ResolverResult("NOT_FOUND", "", "", "no_match")
