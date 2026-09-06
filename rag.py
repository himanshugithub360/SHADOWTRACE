
from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd
import streamlit as st

import config
import nav
import vector_store


def _fingerprint(df: pd.DataFrame) -> str:

    if df.empty:
        return "empty"
    payload = pd.util.hash_pandas_object(df[["user", "message", "date"]], index=False).values
    return hashlib.md5(payload.tobytes()).hexdigest()


@st.cache_resource(show_spinner="Preparing your conversation for search...")
def get_or_build_index(_df: pd.DataFrame, fingerprint: str, backend: str):

    with nav.time_block("rag_indexing"):
        return vector_store.build_index(_df, backend=backend, fingerprint=fingerprint)


def index_chat(df: pd.DataFrame, backend: Optional[str] = None, fingerprint: Optional[str] = None):
    """Convenience wrapper: fingerprint + cached build in one call."""
    backend = backend or config.VECTOR_STORE_BACKEND
    fp = fingerprint or _fingerprint(df)
    return get_or_build_index(df, fp, backend)


@st.cache_data(show_spinner=False)
def _cached_similarity_search(_store, fingerprint: str, backend: str, query: str, k: int) -> list[dict]:
    
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
