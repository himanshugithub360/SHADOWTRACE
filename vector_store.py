"""
vector_store.py
================
Phase 5 — AI Assistant: turns the (filtered) chat DataFrame into
embedded, searchable documents and manages the FAISS / ChromaDB index
used for RAG (Retrieval-Augmented Generation).

Design principle
-----------------
Just like Phase 4's optional-dependency pattern (Detoxify, BERTopic,
KeyBERT), embeddings degrade gracefully instead of crashing the app —
and, since Phase 21, that fallback is a REAL runtime safety net, not
just a construction-time one:

    Gemini embeddings (GOOGLE_API_KEY set)
        -> OpenRouter embeddings (OPENROUTER_API_KEY set)
        -> local, dependency-light hashing embeddings (no key needed)

A provider can fail either when its client is first built (missing
key, bad import) OR later, mid-request (quota exhausted, auth revoked,
model renamed, a network blip) — this module now handles both. See
`_embed_texts_with_fallback` and `FallbackEmbeddings` below.

The local fallback is deliberately simple (a hashed bag-of-words
vector) — it's a stand-in so Semantic Search / RAG still *work* with
zero API cost/config for local development and demos, not a
replacement for real embeddings. `EMBEDDINGS_BACKEND` / `EMBEDDINGS_MODEL`
tell app.py/nav.py's (debug-only) status which mode last actually
succeeded.

CRITICAL invariant — never mix embedding spaces
------------------------------------------------
A single FAISS/Chroma index must never contain vectors produced by two
different embedding backends/models: their vector spaces are not
comparable, so "similarity" scores across them would be meaningless.
Every code path below either (a) embeds an ENTIRE batch with one
provider and discards it completely on failure before trying the next
provider (never merges partial results from two providers), or (b)
detects that a persisted index's pinned provider is no longer usable
and discards + rebuilds the WHOLE index with the current provider,
rather than silently querying it with a different provider's vectors.

This module deliberately does NOT import or reuse anything from
ai_helper.py's AI *generation* fallback chain (Gemini -> Groq ->
OpenRouter -> Mistral, config.PROVIDER_PRIORITY /
config.PROVIDER_QUOTA_COOLDOWN_SECONDS). Embedding fallback has its own
priority list (config.EMBEDDING_PROVIDER_PRIORITY), its own cooldown
window (config.EMBEDDING_PROVIDER_COOLDOWN_SECONDS), and its own
in-memory bookkeeping (`_embedding_cooldowns` / `_embedding_disabled`
below) — a generation provider going down (or a user pinning
AI_PROVIDER to one provider) has zero effect on which embedding
provider Semantic Search / RAG use, and vice versa.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config

# ---------------------------------------------------------------------------
# Embedding backend status — set only by an actual successful embedding
# call (never by client construction alone), read by anything that wants
# to know "what really answered last" (debug panel only; never shown to
# normal users — see the no-developer-UI rule).
# ---------------------------------------------------------------------------
EMBEDDINGS_BACKEND: str = "none"
EMBEDDINGS_MODEL: str = ""

# Per-process (not per-session) bookkeeping, deliberately mirroring the
# shape of ai_helper.py's generation-side cooldown dict but kept as an
# independent copy — see module docstring for why the two must never
# share state.
_embedding_cooldowns: dict[str, float] = {}  # {provider: unix_timestamp_until}
_embedding_disabled: set[str] = set()  # providers permanently off for this process (auth failure)


class EmbeddingRequestError(Exception):
    """Raised when a piece of text could not be embedded by ANY eligible
    provider (or when a provider's response is unusable for a reason
    that must not be silently retried against every other provider).
    Always carries an already user-safe message — never a raw provider
    error — so callers can show it directly."""


class LocalHashingEmbeddings(Embeddings):
    """Dependency-light, API-key-free embedding fallback.

    Hashes each word into one of `dim` buckets (a classic "hashing
    trick" bag-of-words vector), L2-normalized. This captures crude
    lexical overlap — good enough for "find messages that mention
    similar words" but NOT true semantic similarity. Used only when
    neither GOOGLE_API_KEY nor OPENROUTER_API_KEY is configured, so
    Semantic Search / RAG still return *something* instead of an error.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._word_re = re.compile(r"[a-zA-Z']+")

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in self._word_re.findall((text or "").lower()):
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ---------------------------------------------------------------------------
# Error classification — mirrors ai_helper._classify_error's categories
# and ordering (a deliberate, independent copy — see module docstring).
# ---------------------------------------------------------------------------
_QUOTA_MARKERS = ("resource_exhausted", "429", "quota", "rate limit", "rate_limit")
_AUTH_MARKERS = (
    "401", "403", "api key", "apikey", "unauthorized", "permission",
    "invalid_argument: api key", "authentication",
)
_MODEL_NOT_FOUND_MARKERS = ("404", "model not found", "does not exist", "no such model", "not_found")
_NETWORK_MARKERS = ("connection", "timed out", "timeout", "network", "unreachable", "dns", "ssl")
_TRANSIENT_MARKERS = (
    "503", "502", "500", "temporarily unavailable", "overloaded",
    "internal error", "internal_error", "server error", "try again",
)
_INVALID_REQUEST_MARKERS = ("400", "invalid_argument", "invalid request", "unsupported", "bad request")


def _classify_embedding_error(exc: Exception) -> str:
    """Return one of: "quota", "auth", "model_not_found", "network",
    "transient", "invalid_request", "unknown". Order matters — e.g. a
    404 is checked against MODEL_NOT_FOUND before the more generic
    INVALID_REQUEST bucket."""
    msg = str(exc).lower()
    if any(m in msg for m in _QUOTA_MARKERS):
        return "quota"
    if any(m in msg for m in _AUTH_MARKERS):
        return "auth"
    if any(m in msg for m in _MODEL_NOT_FOUND_MARKERS):
        return "model_not_found"
    if any(m in msg for m in _NETWORK_MARKERS):
        return "network"
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return "transient"
    if any(m in msg for m in _INVALID_REQUEST_MARKERS):
        return "invalid_request"
    return "unknown"


def _is_cooling_down(provider: str) -> bool:
    until = _embedding_cooldowns.get(provider)
    return bool(until and time.time() < until)


def _start_cooldown(provider: str) -> None:
    _embedding_cooldowns[provider] = time.time() + config.EMBEDDING_PROVIDER_COOLDOWN_SECONDS


def _disable_provider(provider: str) -> None:
    """Auth failures disable a provider for the rest of THIS process —
    a bad/revoked key won't fix itself mid-session, so retrying it on
    every subsequent chat/query would just waste a round trip and
    reproduce the same failure. (Restarting the app re-enables it, in
    case the key was fixed in `.env` and the process restarted.)"""
    _embedding_disabled.add(provider)


def _provider_available(provider: str) -> bool:
    if provider == "local":
        return True  # pure-Python, no key/network — never disabled or cooling down
    return provider not in _embedding_disabled and not _is_cooling_down(provider)


# ---------------------------------------------------------------------------
# Per-provider client construction. These build a LangChain Embeddings
# client object only — they never themselves make an embedding call, so
# they're cheap/safe to invoke speculatively (e.g. to pre-pin a loaded
# index's provider) without spending API quota.
# ---------------------------------------------------------------------------
def _build_gemini_embeddings() -> Optional[Embeddings]:
    if not config.GOOGLE_API_KEY:
        return None
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            # Bounds worst-case wait the same way config.AI_REQUEST_TIMEOUT_SECONDS
            # bounds AI generation calls (ai_helper.py) — without it, a
            # hung Gemini embedding call could block indefinitely before
            # this module's fallback logic ever gets a chance to react.
            request_options={"timeout": config.AI_REQUEST_TIMEOUT_SECONDS},
        )
    except Exception:
        return None


def _build_openrouter_embeddings() -> Optional[Embeddings]:
    # OpenRouter provides an OpenAI-compatible embeddings endpoint
    # (https://openrouter.ai/api/v1/embeddings), so this reuses
    # OpenAIEmbeddings pointed at the OpenRouter base URL.
    if not config.OPENROUTER_API_KEY:
        return None
    try:
        from langchain_openai import OpenAIEmbeddings

        default_headers = {}
        if config.OPENROUTER_SITE_URL:
            default_headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
        if config.OPENROUTER_SITE_NAME:
            default_headers["X-Title"] = config.OPENROUTER_SITE_NAME

        return OpenAIEmbeddings(
            model=config.OPENROUTER_EMBEDDING_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers=default_headers or None,
            timeout=config.AI_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception:
        return None


def _build_local_embeddings() -> Embeddings:
    return LocalHashingEmbeddings()


_PROVIDER_BUILDERS = {
    "gemini": _build_gemini_embeddings,
    "openrouter": _build_openrouter_embeddings,
    "local": _build_local_embeddings,
}

_PROVIDER_MODEL_NAMES: dict[str, str] = {
    "gemini": config.GEMINI_EMBEDDING_MODEL,
    "openrouter": config.OPENROUTER_EMBEDDING_MODEL,
    "local": "local-hashing-v1",
}


@st.cache_resource(show_spinner=False)
def _client_for(provider: str) -> Optional[Embeddings]:
    """Cached, per-provider client construction (once per process per
    provider name) — never performs an embedding call itself."""
    builder = _PROVIDER_BUILDERS.get(provider)
    return builder() if builder else None


def get_embeddings() -> Embeddings:
    """Best-effort single client for callers that just need *a* working
    embeddings object without needing the full runtime fallback/pinning
    guarantee (e.g. Chroma's "does this collection already have
    vectors?" probe, which never itself calls embed_query/embed_documents).
    Skips providers currently cooling down or disabled. Real runtime
    fallback-on-failure for actual embedding calls happens in
    `_embed_texts_with_fallback` / `FallbackEmbeddings` below — this
    function intentionally does not update EMBEDDINGS_BACKEND, since
    constructing a client is not the same as it having succeeded."""
    for provider in config.EMBEDDING_PROVIDER_PRIORITY:
        if not _provider_available(provider):
            continue
        client = _client_for(provider)
        if client is not None:
            return client
    return LocalHashingEmbeddings()


def _embed_texts_with_fallback(texts: list[str]) -> tuple[list[list[float]], str, str]:
    """Embed a whole batch of texts, trying providers in
    config.EMBEDDING_PROVIDER_PRIORITY order. Never mixes vectors from
    different providers: if a provider fails partway through a batch,
    everything it produced is discarded and the NEXT provider embeds the
    ENTIRE batch again from scratch — the caller only ever sees vectors
    from exactly one provider.

    Returns:
        (vectors, backend_name, model_name) from whichever provider
        succeeded. "local" always succeeds (pure Python, no
        network/key), so this only raises if something is very wrong.

    Raises:
        EmbeddingRequestError: the request itself was invalid (kind ==
            "invalid_request") — retrying the same malformed input
            against every other provider would not help, so this fails
            fast instead of burning through the whole chain.
    """
    last_exc: Optional[Exception] = None

    for provider in config.EMBEDDING_PROVIDER_PRIORITY:
        if not _provider_available(provider):
            continue

        client = _client_for(provider)
        if client is None:
            continue

        attempts = 1 + (config.EMBEDDING_TRANSIENT_RETRY_ATTEMPTS if provider != "local" else 0)
        for attempt in range(attempts):
            try:
                vectors = client.embed_documents(texts)
                return vectors, provider, _PROVIDER_MODEL_NAMES.get(provider, "")
            except Exception as exc:  # noqa: BLE001 -- classify, maybe fall through
                last_exc = exc
                kind = _classify_embedding_error(exc)

                if kind == "quota":
                    _start_cooldown(provider)
                    break  # this provider is exhausted -- move on, don't retry it
                if kind == "auth":
                    _disable_provider(provider)
                    break
                if kind == "model_not_found":
                    break  # retrying the same misconfigured model is pointless
                if kind == "invalid_request":
                    # Not a provider-availability problem -- the request
                    # itself is malformed. Don't blindly resend the same
                    # bad input to every other provider.
                    raise EmbeddingRequestError(
                        "We couldn't prepare part of this conversation for search. "
                        "Please try again."
                    ) from exc
                if kind in ("network", "transient") and attempt < attempts - 1:
                    continue  # one small retry on the SAME provider
                break  # retries exhausted (or "unknown") -- fall through to next provider

    # "local" is always in the chain and always available/succeeds, so
    # reaching here means something unexpected happened even in the
    # local fallback -- surface a clear, user-safe message rather than
    # letting a raw exception escape.
    raise EmbeddingRequestError(
        "We couldn't prepare this conversation for search right now. Please try again in a moment."
    ) from last_exc


class FallbackEmbeddings(Embeddings):
    """LangChain `Embeddings` wrapper providing runtime multi-provider
    fallback, safe to hand to FAISS/Chroma as their stored embedding
    function.

    CRITICAL invariant: once this wrapper's first successful
    `embed_documents()` call has picked a provider (index build time),
    it PINS itself to that exact provider/model for every subsequent
    `embed_query()` call — it will NEVER silently switch providers for
    queries against an already-built index, because that would mix two
    incompatible vector spaces. If the pinned provider later fails at
    query time, `embed_query()` raises `EmbeddingRequestError` instead;
    callers (see rag.py) catch that, discard the now-stale index, and
    rebuild it fresh with whatever provider is currently best.
    """

    def __init__(self, pinned_backend: Optional[str] = None, pinned_model: Optional[str] = None):
        self.backend: Optional[str] = pinned_backend
        self.model: Optional[str] = pinned_model
        self._pinned_client: Optional[Embeddings] = (
            _client_for(pinned_backend) if pinned_backend else None
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors, backend, model = _embed_texts_with_fallback(texts)
        self.backend = backend
        self.model = model
        self._pinned_client = _client_for(backend)
        global EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL
        EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL = backend, model
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self._pinned_client is None:
            # No index has been built through this instance yet (e.g. a
            # freshly-loaded wrapper with no prior pin) -- resolve a
            # provider now and pin to it, same as embed_documents would.
            vectors, backend, model = _embed_texts_with_fallback([text])
            self.backend = backend
            self.model = model
            self._pinned_client = _client_for(backend)
            return vectors[0]

        try:
            return self._pinned_client.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            kind = _classify_embedding_error(exc)
            if kind == "quota" and self.backend:
                _start_cooldown(self.backend)
            elif kind == "auth" and self.backend:
                _disable_provider(self.backend)
            raise EmbeddingRequestError(
                "Your conversation search needs to be refreshed. Please try your search again."
            ) from exc


# ---------------------------------------------------------------------------
# DataFrame -> Documents
# ---------------------------------------------------------------------------
def messages_to_documents(df: pd.DataFrame) -> list[Document]:
    """Convert chat rows into one Document per message, with metadata
    the retriever/UI can use to show source/date/user.

    Args:
        df: Chat DataFrame (already filtered to whatever scope should
            be indexed — e.g. the sidebar-filtered selection).

    Returns:
        A list of `langchain_core.documents.Document`, skipping empty
        rows. Each page_content is "User: message" so the embedded text
        keeps speaker context. `metadata["message_index"]` is this
        document's position among the *kept* (non-empty, non-media)
        messages for this selection — used by `chunk_documents` below
        to record which message range each chunk came from.
    """
    docs: list[Document] = []
    for i, (_, row) in enumerate(df.iterrows()):
        message = str(row.get("message", "")).strip()
        if not message or message == config.MEDIA_PLACEHOLDER:
            continue
        user = row.get("user", "unknown")
        date = row.get("date")
        date_str = date.strftime("%Y-%m-%d %H:%M") if pd.notna(date) else "unknown-date"
        docs.append(Document(
            page_content=f"[{date_str}] {user}: {message}",
            metadata={"user": str(user), "date": date_str, "message_index": i},
        ))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Group individual message-Documents into windows of consecutive
    messages sized for embedding (single messages are often too
    short/context-free on their own).

    Fix 3 (RAG metadata preservation): the previous implementation
    joined every message into one giant string and ran a character
    splitter over it, which meant every chunk's metadata was empty —
    there was no way to tell which users/dates/messages a retrieved
    chunk actually came from. This version accumulates whole messages
    (never splitting a message's own text) up to `config.RAG_CHUNK_SIZE`
    characters, so each resulting chunk can carry real metadata:
    the participant(s) in it, its date range, and the message index
    range it spans — enough for retrieval results to be explainable.

    `config.RAG_CHUNK_OVERLAP` (originally a character-overlap knob for
    the old splitter) is interpreted here as "carry the last message of
    the previous chunk into the next one" whenever it's non-zero — a
    message-aware analogue of the same idea, not a literal character
    count, since chunk boundaries are now message boundaries.
    """
    if not docs:
        return []

    carry_last_message = config.RAG_CHUNK_OVERLAP > 0
    chunks: list[Document] = []
    current: list[Document] = []
    current_len = 0

    def _flush() -> None:
        if not current:
            return
        text = "\n".join(d.page_content for d in current)
        users = sorted({str(d.metadata.get("user", "unknown")) for d in current})
        dates = [
            d.metadata.get("date") for d in current
            if d.metadata.get("date") and d.metadata.get("date") != "unknown-date"
        ]
        chunks.append(Document(
            page_content=text,
            metadata={
                "users": users,
                "start_date": dates[0] if dates else None,
                "end_date": dates[-1] if dates else None,
                "message_start_index": current[0].metadata.get("message_index"),
                "message_end_index": current[-1].metadata.get("message_index"),
                "message_count": len(current),
            },
        ))

    for doc in docs:
        doc_len = len(doc.page_content) + 1  # +1 for the joining newline
        if current and current_len + doc_len > config.RAG_CHUNK_SIZE:
            _flush()
            current = current[-1:] if carry_last_message else []
            current_len = sum(len(d.page_content) + 1 for d in current)
        current.append(doc)
        current_len += doc_len

    _flush()
    return chunks


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------
# Phase 6: both backends now accept a `fingerprint` (the same
# chat+filters+user fingerprint rag.py already builds) and, when
# `config.RAG_PERSIST_INDEX` is on, persist to / load from disk under
# that fingerprint. This means:
#   - Same fingerprint, same Streamlit session -> served from
#     `st.cache_resource` (rag.get_or_build_index); this code never runs.
#   - Same fingerprint, NEW session / after a process restart -> loaded
#     from disk here instead of re-embedding every message.
#   - Different fingerprint (new upload, changed filters/user) -> a
#     fresh index is built and saved under its own fingerprint; chats
#     never share or collide with each other's persisted index.
#
# Phase 21: every persisted index is now paired with a small sidecar
# metadata file recording which embedding backend/model/dimension built
# it. This is what lets a reload safely PIN the query-time embedding
# function to the exact same provider the index's vectors came from
# (see FallbackEmbeddings) instead of guessing — and what lets a stale
# index (built by a provider that's since been disabled/exhausted) be
# detected and discarded rather than silently queried with mismatched
# vectors.
def _faiss_index_dir(fingerprint: str) -> Path:
    return config.VECTOR_STORE_DIR / "faiss" / fingerprint


def _meta_path(index_dir: Path) -> Path:
    return index_dir / "shadowtrace_embedding_meta.json"


def _read_meta(index_dir: Path) -> Optional[dict]:
    try:
        return json.loads(_meta_path(index_dir).read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(index_dir: Path, backend: str, model: str, dimension: int) -> None:
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        _meta_path(index_dir).write_text(
            json.dumps({
                "embedding_backend": backend,
                "embedding_model": model,
                "embedding_dimension": dimension,
            }),
            encoding="utf-8",
        )
    except Exception:
        pass  # metadata is best-effort; a missing/unreadable file just forces a safe rebuild later


def discard_persisted_index(fingerprint: str, backend: str = "faiss") -> None:
    """Delete a persisted index (and its metadata) for `fingerprint` so
    the NEXT build starts completely fresh instead of reloading a stale,
    now-incompatible index (e.g. its pinned provider ran out of quota or
    had its key revoked). Safe to call even if nothing is persisted."""
    try:
        if backend == "chroma":
            return  # Chroma persistence is a shared directory keyed by collection name, not a single deletable folder per fingerprint
        shutil.rmtree(_faiss_index_dir(fingerprint), ignore_errors=True)
    except Exception:
        pass


def build_faiss_index(docs: list[Document], fingerprint: Optional[str] = None):
    """Build (or load a persisted) FAISS index from documents."""
    from langchain_community.vectorstores import FAISS

    global EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL

    if fingerprint and config.RAG_PERSIST_INDEX:
        index_dir = _faiss_index_dir(fingerprint)
        meta = _read_meta(index_dir) if index_dir.exists() else None
        if meta and meta.get("embedding_backend"):
            try:
                pinned = FallbackEmbeddings(meta["embedding_backend"], meta.get("embedding_model"))
                store = FAISS.load_local(
                    str(index_dir), pinned, allow_dangerous_deserialization=True,
                )
                EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL = pinned.backend, pinned.model
                return store
            except Exception:
                pass  # corrupted/incompatible on-disk index — fall through and rebuild

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectors, backend, model = _embed_texts_with_fallback(texts)
    EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL = backend, model

    pinned = FallbackEmbeddings(backend, model)
    store = FAISS.from_embeddings(list(zip(texts, vectors)), embedding=pinned, metadatas=metadatas)

    if fingerprint and config.RAG_PERSIST_INDEX:
        try:
            index_dir = _faiss_index_dir(fingerprint)
            index_dir.mkdir(parents=True, exist_ok=True)
            store.save_local(str(index_dir))
            _write_meta(index_dir, backend, model, len(vectors[0]) if vectors else 0)
        except Exception:
            pass  # persistence is a nice-to-have; never fail the request over it

    return store


def _chroma_persist_dir() -> Path:
    return config.VECTOR_STORE_DIR / "chroma"


def build_chroma_index(docs: list[Document], fingerprint: Optional[str] = None,
                        persist_directory: Optional[Path] = None):
    """Build (or load a persisted) ChromaDB index from documents.

    The collection name is scoped by `fingerprint` so two different
    chats (or the same chat under different filters/user) never read or
    write each other's vectors in the shared persist directory — the
    original code used one fixed collection name for every chat, which
    would have silently mixed embeddings across uploads.
    """
    from langchain_chroma import Chroma

    global EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL

    collection = f"{config.CHROMA_COLLECTION_NAME}_{fingerprint}" if fingerprint else config.CHROMA_COLLECTION_NAME
    persist_dir_path = Path(persist_directory or _chroma_persist_dir())
    persist_dir_path.mkdir(parents=True, exist_ok=True)
    persist_dir = str(persist_dir_path)
    meta = _read_meta(persist_dir_path / collection)

    if fingerprint and config.RAG_PERSIST_INDEX and meta and meta.get("embedding_backend"):
        try:
            pinned = FallbackEmbeddings(meta["embedding_backend"], meta.get("embedding_model"))
            existing = Chroma(
                collection_name=collection,
                persist_directory=persist_dir,
                embedding_function=pinned,
            )
            if len(existing.get()["ids"]) > 0:
                EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL = pinned.backend, pinned.model
                return existing  # already embedded for this fingerprint — reuse, don't re-embed
        except Exception:
            pass  # fall through and build fresh below

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata or {} for d in docs]
    vectors, backend, model = _embed_texts_with_fallback(texts)
    EMBEDDINGS_BACKEND, EMBEDDINGS_MODEL = backend, model
    pinned = FallbackEmbeddings(backend, model)

    store = Chroma.from_texts(
        texts, pinned, metadatas=metadatas, collection_name=collection, persist_directory=persist_dir,
    )

    if fingerprint and config.RAG_PERSIST_INDEX:
        _write_meta(persist_dir_path / collection, backend, model, len(vectors[0]) if vectors else 0)

    return store


def build_index(df: pd.DataFrame, backend: Optional[str] = None, fingerprint: Optional[str] = None):
    """Build a vector index for `df` using the configured backend.

    Args:
        df: Chat DataFrame to index.
        backend: "faiss" or "chroma". Defaults to config.VECTOR_STORE_BACKEND.
        fingerprint: Stable cache/persistence key (chat + filters +
            selected user). See rag.py. When omitted, the index is
            still built correctly but skips disk persistence, since
            there's no stable key to save/load it under.

    Returns:
        A LangChain VectorStore (empty-safe: returns None if `df` has
        no indexable text).

    Raises:
        EmbeddingRequestError: no configured embedding provider (Gemini,
            OpenRouter, or the always-available local fallback) could
            embed this chat's text. Always a user-safe message — see
            `_embed_texts_with_fallback`.
    """
    docs = chunk_documents(messages_to_documents(df))
    if not docs:
        return None

    backend = (backend or config.VECTOR_STORE_BACKEND).lower()
    if backend == "chroma":
        return build_chroma_index(docs, fingerprint=fingerprint)
    return build_faiss_index(docs, fingerprint=fingerprint)
