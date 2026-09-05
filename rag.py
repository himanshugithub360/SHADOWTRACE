"""
rag.py
======
Phase 5 — AI Assistant: Retrieval-Augmented Generation over the chat.

Responsibilities
-----------------
- Index the (sidebar-filtered) chat into a vector store (vector_store.py).
- Retrieve the most relevant chunks for a free-text query.
- Provide the "Semantic Search" feature directly (retrieval only, no LLM).
- Provide the retrieval half of "Ask Questions" / "Chat with History"
  (ai_helper.py supplies the LLM half).

Caching
--------
Building an index re-embeds every message, which costs API calls/time.
`get_or_build_index` is cached with `st.cache_resource`, keyed off a
cheap fingerprint of the DataFrame (shape + a hash of the message
column), so switching sidebar filters rebuilds the index only when the
underlying data actually changed.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd
import streamlit as st

import config
import nav
import vector_store


def _fingerprint(df: pd.DataFrame) -> str:
    """Cheap, stable fingerprint of a DataFrame's content, used as the
    cache key for index building (two calls with equal fingerprints are
    treated as "the same chat selection").

    Phase 6: this is now a FALLBACK only. Callers that already have a
    fingerprint on hand (see app.py's `get_ai_fingerprint()`, built from
    the chat's file fingerprint + selected user + active sidebar
    filters) should pass it in explicitly — it's O(1) to build instead
    of O(rows) to hash, and it's the fingerprint app.py wants persisted
    to disk. This function only runs when no explicit fingerprint is
    supplied, so older/direct callers keep working unchanged.
    """
    if df.empty:
        return "empty"
    payload = pd.util.hash_pandas_object(df[["user", "message", "date"]], index=False).values
    return hashlib.md5(payload.tobytes()).hexdigest()


@st.cache_resource(show_spinner="Preparing your conversation for search...")
def get_or_build_index(_df: pd.DataFrame, fingerprint: str, backend: str):
    """Build (or fetch from Streamlit's resource cache) the vector index
    for this exact chat selection. This is the ONLY place embeddings
    actually get computed — it only ever runs when something calls
    `retrieve()` / `build_context_block()` (Ask & Chat on question
    submit, or Semantic Search on search), never just from opening the
    AI Assistant tab.

    Args:
        _df: The chat DataFrame to index (leading underscore tells
            st.cache_resource not to try to hash it directly — we pass
            `fingerprint` in ourselves instead).
        fingerprint: The real cache key — chat + filters + selected user
            (see app.py's `get_ai_fingerprint()`), or `_fingerprint(_df)`
            as a fallback. Unchanged across reruns unless the underlying
            selection actually changed, so asking a different QUESTION
            about the same selection never triggers a rebuild.
        backend: "faiss" or "chroma".

    Returns:
        A LangChain VectorStore, or None if there was nothing to index.
    """
    with nav.time_block("rag_indexing"):
        return vector_store.build_index(_df, backend=backend, fingerprint=fingerprint)


def index_chat(df: pd.DataFrame, backend: Optional[str] = None, fingerprint: Optional[str] = None):
    """Convenience wrapper: fingerprint + cached build in one call."""
    backend = backend or config.VECTOR_STORE_BACKEND
    fp = fingerprint or _fingerprint(df)
    return get_or_build_index(df, fp, backend)


@st.cache_data(show_spinner=False)
def _cached_similarity_search(_store, fingerprint: str, backend: str, query: str, k: int) -> list[dict]:
    """The actual similarity-search call, cached by (fingerprint,
    backend, query, k). `_store` is underscore-prefixed so
    `st.cache_data` doesn't try to hash the VectorStore object itself —
    the real cache key is the four plain arguments after it. Re-running
    the exact same query against the exact same chat selection (e.g. the
    user re-opens a previous answer, or two prompt.py templates both
    want the same context) is served from cache instead of hitting the
    embeddings API again.

    `vector_store.EmbeddingRequestError` (raised when this index's
    pinned embedding provider has become unavailable — see
    `vector_store.FallbackEmbeddings`) is deliberately let through
    rather than caught by the generic fallback below: it means the
    INDEX itself is stale, not that this particular search mode isn't
    supported, so plain `similarity_search` would just fail identically.
    `retrieve()` catches it and rebuilds the index fresh.
    """
    try:
        results = _store.similarity_search_with_score(query, k=k)
        return [
            {"text": doc.page_content, "score": round(float(score), 4), "metadata": doc.metadata or {}}
            for doc, score in results
        ]
    except vector_store.EmbeddingRequestError:
        raise
    except Exception:
        # Some backends/embeddings don't support scored search — fall
        # back to plain similarity search rather than failing the tab.
        docs = _store.similarity_search(query, k=k)
        return [{"text": doc.page_content, "score": None, "metadata": doc.metadata or {}} for doc in docs]


def retrieve(
    df: pd.DataFrame,
    query: str,
    k: Optional[int] = None,
    fingerprint: Optional[str] = None,
    backend: Optional[str] = None,
) -> list[dict]:
    """Semantic search: return the top-k chunks most relevant to `query`.

    Args:
        df: Chat DataFrame to search within (already sidebar-filtered).
        query: Free-text search query.
        k: Number of results (defaults to config.RAG_TOP_K).
        fingerprint: Stable cache key for this chat+filters+user
            selection. Pass this through from the caller when available
            (see app.py) instead of letting it be recomputed from `df`.
        backend: "faiss" or "chroma". Defaults to config.VECTOR_STORE_BACKEND.

    Returns:
        List of {"text": str, "score": float | None, "metadata": dict}
        dicts, best first. `metadata` (Fix 3) holds whatever
        `vector_store.chunk_documents` recorded for that chunk —
        `users`, `start_date`, `end_date`, `message_start_index`,
        `message_end_index`, `message_count` — or `{}` for an index
        built before this metadata existed. Empty list if there's
        nothing indexable or no query given.

    Note on embedding fallback: if the index's pinned embedding provider
    (see vector_store.FallbackEmbeddings) becomes unavailable between
    when the index was built and when this query runs (e.g. quota ran
    out, key revoked), the stale index is discarded and rebuilt fresh
    with whatever embedding provider is currently available — never
    silently mixed with the old provider's vectors. This costs one
    re-embed of the whole chat selection, but only ever happens on a
    genuine provider failure, not on every query.
    """
    query = (query or "").strip()
    if not query:
        return []

    backend = (backend or config.VECTOR_STORE_BACKEND).lower()
    fp = fingerprint or _fingerprint(df)

    try:
        store = index_chat(df, backend=backend, fingerprint=fp)
    except vector_store.EmbeddingRequestError:
        # Building the index itself failed (e.g. a malformed/invalid
        # request even the local fallback couldn't handle). Never let a
        # raw exception reach the UI — an empty result just shows the
        # existing "No matching conversations found" empty state.
        return []
    if store is None:
        return []

    k = k or config.RAG_TOP_K
    try:
        return _cached_similarity_search(store, fp, backend, query, k)
    except vector_store.EmbeddingRequestError:
        vector_store.discard_persisted_index(fp, backend)
        get_or_build_index.clear()  # forces a fresh build next call — see docstring above
        try:
            store = index_chat(df, backend=backend, fingerprint=fp)
        except vector_store.EmbeddingRequestError:
            return []  # every embedding provider (including local) is unavailable right now
        if store is None:
            return []
        try:
            return _cached_similarity_search(store, fp, backend, query, k)
        except vector_store.EmbeddingRequestError:
            return []


def _format_hit(hit: dict) -> str:
    """One retrieved chunk as prompt-ready text. Prefixes the chunk's
    text with its metadata (participants + date range, when the index
    was built with Fix 3's message-aware chunking) so the LLM can
    ground "who/when" claims instead of only having the raw text."""
    meta = hit.get("metadata") or {}
    users = meta.get("users")
    start_date, end_date = meta.get("start_date"), meta.get("end_date")
    if users and start_date:
        span = start_date if start_date == end_date else f"{start_date} to {end_date}"
        header = f"[{', '.join(users)} | {span}]\n"
    else:
        header = ""
    return f"{header}{hit['text']}"


def build_context_block(
    df: pd.DataFrame, query: str, k: Optional[int] = None, fingerprint: Optional[str] = None,
) -> str:
    """Retrieve chunks for `query` and format them as a single block of
    text ready to drop into an LLM prompt (see prompt.QA_PROMPT)."""
    hits = retrieve(df, query, k=k, fingerprint=fingerprint)
    if not hits:
        return "(No relevant chat excerpts found for this query.)"
    return "\n---\n".join(_format_hit(h) for h in hits)
