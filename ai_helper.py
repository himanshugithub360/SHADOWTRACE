from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

import pandas as pd
import streamlit as st

import analytics
import config
import helper
import nav
import nlp_helper
import prompt
import rag
import sentiment
import toxicity
import utils
from langchain_core.prompts import ChatPromptTemplate


for _noisy_logger_name in ("google_genai.models", "google.genai.models", "google_genai", "google.genai"):
    logging.getLogger(_noisy_logger_name).setLevel(logging.ERROR)

AI_SETUP_MESSAGE = (
    "\U0001f511 No AI provider is configured yet. Add at least one of "
    "`GOOGLE_API_KEY` (Gemini), `GROQ_API_KEY` (Groq), `OPENROUTER_API_KEY` "
    "(OpenRouter), or `MISTRAL_API_KEY` (Mistral) to your `.env` file and "
    "restart the app -- see `.env.example` and the README's *AI Assistant "
    "Setup* section."
)

QUOTA_EXHAUSTED_MESSAGE = (
    "\u23f3 The AI provider's quota is currently exhausted. Your existing "
    "cached AI results are still available -- reopening the same "
    "summary/question/report will show them instantly with no new request. "
    "Please try again after your quota resets (free-tier quotas typically "
    "reset daily)."
)

ALL_PROVIDERS_UNAVAILABLE_MESSAGE = (
    "\u26a0\ufe0f AI services are currently unavailable. Please try again "
    "later. Your locally calculated chat analytics are still available."
)

_PROVIDER_LABELS: dict[str, str] = {
    "gemini": "Gemini", "groq": "Groq", "openrouter": "OpenRouter", "mistral": "Mistral",
}

AI_BACKEND: str = "none"  # last provider that actually answered; read by app.py
LAST_PROVIDER_STATUS: str = ""  # last human status line (fallback banner); read by app.py


_provider_cooldowns: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Provider selection -- config.py is the single source of truth for which
# providers exist, their priority order, and their API keys/models.
# ---------------------------------------------------------------------------
def _model_supports_temperature(model_name: str) -> bool:
    """False for Gemini model families that use fixed sampling defaults
    and just log a warning if `temperature` is passed (see
    config.GEMINI_FIXED_SAMPLING_PREFIXES). Keeping the prefix list in
    config.py (not here) keeps model behavior config-driven per Phase 11
    -- this function is just the lookup."""
    return not model_name.startswith(config.GEMINI_FIXED_SAMPLING_PREFIXES)


def _gemini_thinking_kwargs(model_name: str) -> dict:
    """Thinking-config kwargs for ChatGoogleGenerativeAI, safe across
    whatever version of langchain-google-genai is actually installed
    (Fix 1 — Gemini thinking configuration).

    Only Gemini's thinking-capable model families (see
    config.GEMINI_THINKING_MODEL_PREFIXES) spend part of their output
    token budget on internal reasoning before writing the visible
    answer; for every other model this returns {} and nothing changes.

    Rather than hardcoding a keyword argument name that may not exist
    in the installed client version (which would raise a TypeError at
    construction and silently take the whole Gemini provider down),
    this introspects the actual installed constructor and only sends a
    thinking-budget kwarg if it recognizes one already accepted by
    ChatGoogleGenerativeAI.__init__. If none is recognized, it sends
    nothing — meaning thinking configuration is honestly unsupported by
    this installed version rather than guessed at.
    """
    if not model_name.startswith(config.GEMINI_THINKING_MODEL_PREFIXES):
        return {}
    try:
        import inspect
        from langchain_google_genai import ChatGoogleGenerativeAI
        params = inspect.signature(ChatGoogleGenerativeAI.__init__).parameters
    except Exception:
        return {}
    # Known parameter names used across different langchain-google-genai
    # releases for this setting -- checked in order, first match wins.
    for name in ("thinking_budget", "thinking_budget_tokens"):
        if name in params:
            return {name: config.GEMINI_THINKING_BUDGET}
    return {}


def _timeout_kwargs(client_cls, provider: str) -> dict:
    """Per-request network timeout kwargs for any LangChain chat client
    class, safe across SDK versions -- fixes AI_REQUEST_TIMEOUT_SECONDS
    being defined in config.py but never actually applied anywhere.

    Without this, a slow/unresponsive provider blocks on that SDK's own
    default timeout (often several minutes) before the multi-provider
    fallback chain (see _provider_chain) can even try the next one --
    with up to `len(config.PROVIDER_PRIORITY)` providers to fall through
    on a bad day, those minutes stack up fast, which is the difference
    between a slow response and one that appears to hang for several
    minutes. Bounding every client to AI_REQUEST_TIMEOUT_SECONDS caps
    the worst case at roughly that value times the number of configured
    providers (plus one retry's worth, for the truncation-retry path in
    _cached_ai_request) instead of being effectively unbounded.

    Same compatibility-safe pattern as `_gemini_thinking_kwargs`:
    introspects the actual installed constructor rather than guessing a
    kwarg name, so an SDK version that renamed/dropped the parameter
    degrades to "no explicit timeout sent" instead of a TypeError that
    would take the whole provider down.
    """
    kwargs: dict = {}
    try:
        import inspect
        params = inspect.signature(client_cls.__init__).parameters
        for name in ("timeout", "request_timeout"):
            if name in params:
                kwargs = {name: config.AI_REQUEST_TIMEOUT_SECONDS}
                break
    except Exception:
        pass
    # Debug-only (PERFORMANCE_DEBUG): confirms the timeout is actually
    # being sent (and with what value) rather than silently falling back
    # to the SDK's own default -- this only fires the first time this
    # provider+budget client is built per process (st.cache_resource),
    # not on every request, same as the thinking-budget counter above.
    nav.set_perf_count(
        f"{provider}_timeout_kwarg",
        next(iter(kwargs.values())) if kwargs else "not sent (SDK unsupported)",
    )
    return kwargs


def _provider_chain() -> list[str]:
    """The ordered list of providers to attempt for THIS request.

    - AI_PROVIDER="auto" (or any unrecognized value) -> every provider in
      config.PROVIDER_PRIORITY that actually has an API key configured,
      in priority order (Gemini -> Groq -> OpenRouter -> Mistral).
      Providers with no key are skipped entirely -- never attempted,
      never counted against the 4-attempt cap.
    - AI_PROVIDER="gemini"/"groq"/"openrouter"/"mistral" -> exactly that
      one provider, whether or not its key is present (a missing key
      there produces a clear configuration message -- see Phase 17 --
      instead of silently substituting another provider).
    """
    mode = config.AI_PROVIDER
    if mode in config.PROVIDER_PRIORITY:
        return [mode]
    return [p for p in config.PROVIDER_PRIORITY if config.PROVIDER_API_KEYS.get(p)]


def configured_providers_status() -> dict[str, bool]:
    """{"gemini": True, "groq": False, ...} -- which providers have an
    API key configured, in priority order. Side-effect-free (just reads
    config strings); safe for app.py to call on every render."""
    return {p: bool(config.PROVIDER_API_KEYS.get(p)) for p in config.PROVIDER_PRIORITY}


def _is_cooling_down(provider: str) -> bool:
    until = _provider_cooldowns.get(provider)
    return bool(until and time.time() < until)


def _start_cooldown(provider: str) -> None:
    _provider_cooldowns[provider] = time.time() + config.PROVIDER_QUOTA_COOLDOWN_SECONDS


@st.cache_resource(show_spinner=False)
def _build_provider_client(provider: str, max_output_tokens: Optional[int] = None):
    """Construct (once per process, per provider + token budget) the
    LangChain chat client for one named provider. Returns None if that
    provider has no API key configured, or if constructing the client
    itself fails (missing optional dependency, malformed key, etc.) --
    never raises. Constructing a client makes no network call by
    itself, so this never consumes any provider's quota.

    Args:
        max_output_tokens: Overrides config.LLM_MAX_OUTPUT_TOKENS for
            this client. Used by the truncation-retry path (Fix 1) to
            build a second, larger-budget client for one bounded retry
            without touching the normal-budget client every other call
            uses. `st.cache_resource` naturally gives each distinct
            (provider, max_output_tokens) pair its own cached client,
            so this never rebuilds the everyday client.
    """
    api_key = config.PROVIDER_API_KEYS.get(provider, "")
    if not api_key:
        return None

    budget = max_output_tokens or config.LLM_MAX_OUTPUT_TOKENS

    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs = dict(
                model=config.GEMINI_CHAT_MODEL,
                google_api_key=api_key,
                max_output_tokens=budget,
            )
            # Only pass temperature to models that actually use it -- some
            # newer Gemini models ("fixed sampling defaults") just warn and
            # ignore it, so we skip sending it at all for those instead of
            # silencing/hiding the warning. This changes NOTHING about the
            # model's actual sampling behavior either way.
            if _model_supports_temperature(config.GEMINI_CHAT_MODEL):
                kwargs["temperature"] = config.LLM_TEMPERATURE
            thinking_kwargs = _gemini_thinking_kwargs(config.GEMINI_CHAT_MODEL)
            kwargs.update(thinking_kwargs)
            kwargs.update(_timeout_kwargs(ChatGoogleGenerativeAI, "gemini"))
            # Debug-only (PERFORMANCE_DEBUG): record whether a thinking-
            # budget kwarg was actually sent this request, and what value,
            # so a latency regression on a thinking-capable Gemini model
            # can be diagnosed from the sidebar instead of guessing.
            nav.set_perf_count(
                "gemini_thinking_kwarg",
                next(iter(thinking_kwargs.values())) if thinking_kwargs else "not sent (model/SDK unsupported)",
            )
            return ChatGoogleGenerativeAI(**kwargs)

        if provider == "openrouter":
            # OpenRouter exposes an OpenAI-compatible /chat/completions API
            # in front of many vendors' models, so the existing OpenAI
            # LangChain client works unmodified -- just point `base_url`
            # at OpenRouter and use an OpenRouter API key/model id
            # ("vendor/model", e.g. "openai/gpt-4o-mini"). No new SDK
            # needed beyond the langchain-openai already required for
            # embeddings (see vector_store.py).
            from langchain_openai import ChatOpenAI
            default_headers = {}
            if config.OPENROUTER_SITE_URL:
                default_headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
            if config.OPENROUTER_SITE_NAME:
                default_headers["X-Title"] = config.OPENROUTER_SITE_NAME
            return ChatOpenAI(
                model=config.OPENROUTER_CHAT_MODEL,
                api_key=api_key,
                base_url=config.OPENROUTER_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=budget,
                default_headers=default_headers or None,
                **_timeout_kwargs(ChatOpenAI, "openrouter"),
            )

        if provider == "groq":
            # langchain-groq exposes an OpenAI-compatible ChatGroq class --
            # smallest possible integration, per Phase 12.
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=config.GROQ_CHAT_MODEL,
                api_key=api_key,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=budget,
                **_timeout_kwargs(ChatGroq, "groq"),
            )

        if provider == "mistral":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(
                model=config.MISTRAL_CHAT_MODEL,
                api_key=api_key,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=budget,
                **_timeout_kwargs(ChatMistralAI, "mistral"),
            )
    except Exception:
        return None

    return None


def get_llm():
    """Backward-compatible accessor: returns a client for the FIRST
    provider in the current chain (whatever `AI_PROVIDER` resolves to),
    or None if none is configured. Kept for any external/test code that
    still calls `get_llm()` directly -- the actual multi-provider
    fallback logic lives in `_cached_ai_request`, not here."""
    chain = _provider_chain()
    for provider in chain:
        client = _build_provider_client(provider)
        if client is not None:
            return client
    return None


def is_ai_available() -> bool:
    """True if at least one usable provider client can be built. NOTE:
    this actually INSTANTIATES a LangChain provider client the first
    time it runs per provider (cached per-provider after that) -- so
    this should only be called right before an actual LLM invocation,
    never just to decide whether to show a banner. Use
    `is_ai_configured()` for that."""
    return get_llm() is not None


def is_ai_configured() -> bool:
    """Cheap, side-effect-free check for whether ANY provider's API key
    is present.

    Unlike `is_ai_available()`, this NEVER builds a LangChain provider
    client, an embeddings model, or a RAG index -- it just reads
    already-loaded config strings. Safe to call every time the AI
    Assistant tab renders (see app.py's `render_ai`) without violating
    "opening the tab must not build anything"."""
    return any(config.PROVIDER_API_KEYS.values())


# ---------------------------------------------------------------------------
# Response normalization (Phase 8) -- always returns `str`, never raises,
# never calls .strip() until a value is confirmed to be a string.
# ---------------------------------------------------------------------------
def _normalize_llm_content(content) -> str:
    """Collapse any shape LangChain/Gemini might hand back for
    `result.content` into a plain string:
        1. str                              -> returned as-is (stripped)
        2. list[str]                        -> joined
        3. list[{"type": "text", "text": ...}] (and other dict blocks)
                                              -> "text" values joined
        4. anything else (None, LangChain message objects, ...)
                                              -> str(content), stripped

    This is the single choke point that prevents
    `'list' object has no attribute 'strip'` -- .strip() is only ever
    called on something already confirmed to be `str`.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
            else:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    parts.append(block_text)
        combined = "\n".join(p for p in parts if p).strip()
        if combined:
            return combined
        # A list with no extractable text still needs *some* string back.
        return str(content).strip()

    if content is None:
        return ""

    return str(content).strip()


# ---------------------------------------------------------------------------
# Error classification (Phases 8, 9, 18) -- decides fallback eligibility
# and produces a safe, user-facing message. Never leaks an API key: we
# only ever look at / echo back `str(exc)`, and every provider's own
# exception text does not include the key itself (it's sent as a header,
# not echoed in errors), but we still avoid ever printing the raw
# provider payload verbatim.
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


class AIRequestError(Exception):
    """Non-retryable, already-user-safe error message. Raised by
    `_cached_ai_request` for cases (like "no provider configured", or
    "every configured provider failed") that should never be treated as
    a cacheable *answer* -- `st.cache_data` does not cache exceptions,
    so the next explicit button click always gets a genuine fresh
    attempt instead of replaying a stale failure."""


def _classify_error(exc: Exception) -> str:
    """Return one of: "quota", "auth", "model_not_found", "network",
    "transient", "invalid_request", "unknown". Order matters below --
    e.g. a 404 is checked against MODEL_NOT_FOUND before the more
    generic INVALID_REQUEST bucket."""
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


# Phase 8 fallback policy: which error kinds are eligible to move on to
# the NEXT provider in auto mode, vs. stop immediately.
_FALLBACK_ELIGIBLE_KINDS = {"quota", "auth", "model_not_found", "network", "transient"}


def _error_to_message(exc: Exception, attempted: Optional[list[str]] = None) -> str:
    kind = _classify_error(exc)
    if kind == "quota":
        return QUOTA_EXHAUSTED_MESSAGE
    if kind == "auth":
        return (
            "\U0001f511 AI request failed: invalid or missing API credentials for "
            "the configured provider. Check the relevant API key in your `.env` file."
        )
    if kind == "model_not_found":
        return "\u26a0\ufe0f AI request failed: the configured model could not be found. Check your model name in `.env`."
    if kind == "network":
        return "\u26a0\ufe0f AI request failed: a network error occurred while contacting the AI provider."
    if kind == "invalid_request":
        return "\u26a0\ufe0f AI request failed: the request was invalid for this provider's API."
    if attempted and len(attempted) > 1:
        return ALL_PROVIDERS_UNAVAILABLE_MESSAGE
    return "\u26a0\ufe0f AI request failed. Please try again in a moment."


# ---------------------------------------------------------------------------
# Truncation detection + bounded retry (Fix 1).
# ---------------------------------------------------------------------------
# Provider-reported "why did generation stop" values that mean the model
# was cut off by the token budget rather than finishing naturally. Every
# provider here speaks an OpenAI-compatible or Gemini-shaped
# response_metadata, so this small set covers Gemini ("MAX_TOKENS"),
# OpenAI-compatible providers (OpenRouter/Groq: "length"), and Mistral
# ("length"). Values are compared lowercased.
_TRUNCATION_FINISH_REASONS = {"max_tokens", "length"}

# Endings that are legitimate ways for a real, complete answer to stop
# without terminal punctuation -- used only by the conservative fallback
# heuristic below, when no provider metadata is available at all.
_SAFE_TRAILING_CHARS = ".!?\"'\u201d\u2019)`]}*"


def _get_finish_reason(result) -> Optional[str]:
    """Best-effort extraction of the provider's stop reason from a
    LangChain AIMessage's `response_metadata`. Returns None (never
    raises) if the field isn't present -- callers fall back to the
    conservative heuristic in that case rather than assuming truncation."""
    meta = getattr(result, "response_metadata", None)
    if not isinstance(meta, dict):
        return None
    reason = meta.get("finish_reason") or meta.get("finishReason")
    if reason is None:
        # Gemini's raw shape occasionally surfaces it nested under a
        # "candidates" list instead of the top-level key LangChain
        # usually normalizes it to.
        candidates = meta.get("candidates")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                reason = first.get("finish_reason") or first.get("finishReason")
    return str(reason).lower() if reason else None


def _heuristic_looks_truncated(text: str) -> bool:
    """Conservative, evidence-only truncation guess for when no provider
    metadata is available at all (Fix 1, requirement 2). Deliberately
    biased toward "not truncated" -- short answers, bullet/numbered
    lines, headings, code fences, and URLs are all common, complete ways
    for a real response to end without terminal punctuation, so none of
    them should trigger a false-positive retry."""
    stripped = (text or "").rstrip()
    if len(stripped) < 40:
        return False  # too short to safely judge either way

    if stripped[-1] in _SAFE_TRAILING_CHARS:
        return False
    if stripped.endswith(("```", ":", "-", "*", ")")):
        return False

    last_line = stripped.splitlines()[-1].strip()
    if re.match(r"^(#{1,6}\s|[-*\u2022]\s|\d+[.)]\s)", last_line):
        return False  # heading / bullet / numbered-list line
    if re.search(r"https?://\S+$", stripped):
        return False  # ends on a bare URL

    # Cut off mid-word: ends on a letter/digit with no sentence-ending
    # punctuation anywhere nearby is the strongest signal available
    # without real metadata.
    return stripped[-1].isalnum()


def _looks_truncated(text: str, finish_reason: Optional[str]) -> bool:
    """True if this response should be treated as truncated.

    Prefers real provider metadata (`finish_reason`) whenever it's
    available -- e.g. Gemini's "MAX_TOKENS" or the OpenAI-compatible
    "length" -- and only falls back to the conservative text heuristic
    when no such metadata was returned at all."""
    if finish_reason is not None:
        return finish_reason in _TRUNCATION_FINISH_REASONS
    return _heuristic_looks_truncated(text)


def _status_message(provider: str, chain: list[str]) -> str:
    """Human status line for the UI (Phase 16) -- plain success line if
    the FIRST provider in the chain answered, an explicit fallback
    banner otherwise. Never includes API keys or raw exception text."""
    label = _PROVIDER_LABELS.get(provider, provider.title())
    if chain and provider == chain[0]:
        return f"\U0001f916 Response generated by {label}."
    return f"\u26a0\ufe0f Primary provider unavailable -- response generated by {label} (fallback)."


@st.cache_data(show_spinner=False)
def _cached_ai_request(_template, feature: str, prompt_version: str, **kwargs) -> dict:
    chain = _provider_chain()
    if not chain:
        mode = config.AI_PROVIDER
        if mode in config.PROVIDER_PRIORITY:
            env_var = config.PROVIDER_ENV_VAR_NAMES[mode]
            raise AIRequestError(
                f"\U0001f511 AI_PROVIDER is set to \"{mode}\" but `{env_var}` is not configured."
            )
        raise AIRequestError(AI_SETUP_MESSAGE)

    is_auto = config.AI_PROVIDER not in config.PROVIDER_PRIORITY
    last_exc: Optional[Exception] = None
    unknown_fallback_used = False

    for provider in chain:
        if is_auto and _is_cooling_down(provider):
            continue  # Phase 26/27: recently quota-exhausted -- skip without spending a request

        client = _build_provider_client(provider)
        if client is None:
            continue  # no key / client construction failed -- try next configured provider

        try:
            result = (_template | client).invoke(kwargs)
            text = _normalize_llm_content(getattr(result, "content", result))
            if not text:
                raise AIRequestError(f"{_PROVIDER_LABELS.get(provider, provider)} returned an empty response.")

          
            finish_reason = _get_finish_reason(result)
            truncated = _looks_truncated(text, finish_reason)
         
            nav.set_perf_count("ai_truncation_flagged", f"{truncated} (finish_reason={finish_reason})")
            if truncated and config.LLM_MAX_OUTPUT_TOKENS < config.LLM_MAX_OUTPUT_TOKENS_RETRY_CAP:
                retry_budget = min(config.LLM_MAX_OUTPUT_TOKENS * 2, config.LLM_MAX_OUTPUT_TOKENS_RETRY_CAP)
                nav.set_perf_count("ai_retry_fired", True)
                try:
                    retry_client = _build_provider_client(provider, max_output_tokens=retry_budget)
                    if retry_client is not None:
                        retry_result = (_template | retry_client).invoke(kwargs)
                        retry_text = _normalize_llm_content(getattr(retry_result, "content", retry_result))
                        retry_finish = _get_finish_reason(retry_result)
                        retry_still_truncated = _looks_truncated(retry_text, retry_finish)
                        # Use the retry only if it's a real improvement --
                        # complete where the original wasn't, or at least
                        # longer. Never discard a usable original answer
                        # just because the retry itself came back thin.
                        if retry_text and (not retry_still_truncated or len(retry_text) > len(text)):
                            text = retry_text
                except Exception:
                    pass  # retry failed -- keep the original (possibly truncated) response, never crash
            else:
                nav.set_perf_count("ai_retry_fired", False)

            return {"text": text, "provider": provider}
        except AIRequestError:
            raise  # empty-response case above -- already a final, user-safe message
        except Exception as exc:  # noqa: BLE001 -- classify, maybe fall through
            last_exc = exc
            kind = _classify_error(exc)

            if kind == "quota":
                _start_cooldown(provider)  # Phase 26: skip this provider for a short while

            if not is_auto:
                break  # explicit single-provider mode: NEVER silently switch providers (Phase 17)

            if kind == "invalid_request":
                break  # Phase 8: don't blindly resend a bad request to every other provider

            if kind not in _FALLBACK_ELIGIBLE_KINDS:  # "unknown" -- fall back at most once
                if unknown_fallback_used:
                    break
                unknown_fallback_used = True

            continue  # quota / auth / model_not_found / network / transient (or first unknown) -> try next

    if last_exc is not None:
        raise AIRequestError(_error_to_message(last_exc, attempted=chain))
    raise AIRequestError(AI_SETUP_MESSAGE)


def _run(template, feature: str = "generic", **kwargs) -> str:
    """Invoke `template` through the cached, quota-aware, multi-provider
    router.

    Args:
        template: The ChatPromptTemplate to run.
        feature: Short cache-namespacing key (e.g. "daily_summary",
            "ask_question", "recommendations") so two features can
            never collide in the cache even with coincidentally
            identical kwargs.
        **kwargs: Prompt template variables -- these fully determine
            the cache key alongside `feature`/prompt version.

    Returns:
        The model's text response, OR a friendly, non-crashing message
        for setup/quota/auth/invalid/transient/all-providers-down
        failures (Phases 9, 17, 18, 33). Never raises.
    """
    global AI_BACKEND, LAST_PROVIDER_STATUS

    try:
        with nav.time_block("ai_request"):
            result = _cached_ai_request(template, feature, config.AI_PROMPT_VERSION, **kwargs)
    except AIRequestError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 -- must never crash the Streamlit app
        return _error_to_message(exc, attempted=_provider_chain())

    AI_BACKEND = result["provider"]
    LAST_PROVIDER_STATUS = _status_message(result["provider"], _provider_chain())
   
    nav.set_perf_count("ai_provider_used", result["provider"])
    return result["text"]


def _is_error_message(text: str) -> bool:
    """True if `_run()` returned a setup/quota/auth/invalid/transient
    failure message rather than a real answer -- used to decide whether
    a corrective retry is even worth attempting (never retry a request
    that already failed outright)."""
    return isinstance(text, str) and text.startswith(("\U0001f511", "\u23f3", "\u26a0\ufe0f"))


def _count_numbered_items(text: str) -> int:
    """Count numbered markdown items such as 1., 2., 3., etc."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"(?m)^\s*(?:[-*]\s*)?(\d+)[.)]\s+", text))


_RETRY_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", prompt.SYSTEM_PERSONA),
    ("human", """Generate EXACTLY 5 actionable recommendations from
the DATA below.

The previous response did not contain exactly 5 recommendations.

STRICT RULES:
- Return exactly 5 items.
- Number them exactly 1, 2, 3, 4, 5.
- Do not stop early.
- Do not provide an introduction.
- Do not provide a conclusion.
- Each item must be different.
- Every item must be supported by the DATA.
- Do not invent facts or statistics.

Required format:

1. **Title** -- explanation.
2. **Title** -- explanation.
3. **Title** -- explanation.
4. **Title** -- explanation.
5. **Title** -- explanation.

DATA:
{stats_block}
"""),
])


def _retry_recommendations(stats_block: str) -> str:
    """Ask the LLM once more when the first recommendation response
    doesn't contain exactly five numbered recommendations. Uses its own
    "recommendations_retry" feature name so it gets its own cache entry
    (never collides with, and never re-triggers, the original attempt).
    Only ever called at most once -- see `ai_recommendations`, which
    also refuses to call this at all if the FIRST attempt already
    failed outright (quota/auth/invalid), per Phase 10."""
    return _run(_RETRY_TEMPLATE, feature="recommendations_retry", stats_block=stats_block)


# ---------------------------------------------------------------------------
# Shared: turning DataFrame -> compact stats / message blocks for prompts
# ---------------------------------------------------------------------------
def _messages_block(df: pd.DataFrame, selected_user: str = "Overall", limit: Optional[int] = None) -> str:
    """Recent real (non-media) messages, formatted one per line, capped
    so prompts stay a bounded size/cost regardless of chat length."""
    limit = limit or config.AI_MAX_RAW_MESSAGES_IN_PROMPT
    real = nlp_helper.clean_messages(df, selected_user).dropna(subset=['date']).sort_values('date')
    if real.empty:
        return "(no messages in this selection)"
    real = real.tail(limit)
    lines = [
        f"[{row['date'].strftime('%Y-%m-%d %H:%M')}] {row['user']}: {row['message']}"
        for _, row in real.iterrows()
    ]
    return "\n".join(lines)


# Fix 4 — feature-aware AI context. Previously every AI feature that
# called `_stats_block` triggered the exact same full statistics bundle
# below (KPIs, sentiment, response times, conversation starters, sleep
# schedule, engagement, language, toxicity, ...), even for a feature that
# only needed a fraction of it -- e.g. asking "what was the trip about?"
# computed sleep-schedule estimation and weekend-vs-weekday activity for
# no reason. None of the underlying helper.py/analytics.py/sentiment.py
# functions changed here (and each is still individually cached), but
# each named scope below now only calls the subset of them a given
# feature actually uses, so the FIRST AI request for a lightweight
# feature (Ask & Chat, a summary, recommendations) no longer pays for
# every heavy metric in the app.
_SNAPSHOT_SCOPES: dict[str, set[str]] = {
    # Insights and full Reports intentionally look across everything --
    # that breadth is the point of those two features.
    "full": {
        "headline", "kpis", "busiest_users", "response_time", "starters", "sleep",
        "weekend_vs_weekday", "sentiment", "emotion", "engagement", "toxicity", "language",
    },
    # Summary: high-level context only -- KPIs, activity, participants, sentiment.
    "summary": {"headline", "kpis", "busiest_users", "sentiment"},
    # Recommendations: activity, engagement, sentiment, interaction balance.
    "recommendations": {"headline", "kpis", "response_time", "sentiment", "engagement"},
    # Ask & Chat: RAG (built separately in ask_question) does the heavy
    # lifting; stats here are just minimal grounding, not a full snapshot.
    "minimal": {"headline", "kpis"},
    # People Analysis (personality read): the selected person's own
    # activity/timing/tone -- not group-wide metrics like busiest_users.
    "personality": {"headline", "kpis", "response_time", "sentiment", "emotion"},
}


def _stats_snapshot(selected_user: str, df: pd.DataFrame, scope: str = "full") -> dict:
    """Gather already-computed statistics (from helper.py / analytics.py
    / sentiment.py / toxicity.py) into one dict, limited to what `scope`
    actually needs (see `_SNAPSHOT_SCOPES`). Every sub-call is wrapped so
    one feature's failure (e.g. too little data) never blocks the rest
    of the snapshot."""
    wanted = _SNAPSHOT_SCOPES.get(scope, _SNAPSHOT_SCOPES["full"])
    out: dict = {}

    def safe(key, fn):
        if key not in wanted:
            return
        try:
            out[key] = fn()
        except Exception:
            out[key] = None

    safe("headline", lambda: helper.fetch_stats(selected_user, df))
    safe("kpis", lambda: utils.compute_all_kpis(df))
    if selected_user == "Overall":
        safe("busiest_users", lambda: helper.most_busy_users(df)[0].to_dict())
    safe("response_time", lambda: analytics.response_time_stats(selected_user, df))
    safe("starters", lambda: analytics.conversation_starters(df).to_dict())
    safe("sleep", lambda: analytics.estimate_sleep_schedule(selected_user, df))
    safe("weekend_vs_weekday", lambda: analytics.weekend_vs_weekday(selected_user, df).to_dict("records"))
    safe("sentiment", lambda: sentiment.sentiment_distribution(selected_user, df).to_dict("records"))
    safe("emotion", lambda: sentiment.emotion_distribution(selected_user, df).to_dict("records"))
    if selected_user == "Overall":
        safe("engagement", lambda: analytics.engagement_scores(df).to_dict("records"))
  
    if "toxicity" in wanted and ((not toxicity.DETOXIFY_AVAILABLE) or toxicity.detoxify_ready()):
        safe("toxicity", lambda: toxicity.toxicity_summary(selected_user, df).to_dict("records"))
    safe("language", lambda: nlp_helper.language_distribution(selected_user, df).to_dict())
    return out


def _format_snapshot(snapshot: dict) -> str:
    """Render `_stats_snapshot`'s dict as compact markdown-ish bullet
    text for a prompt (skips any section that failed/was empty)."""
    lines = []

    headline = snapshot.get("headline")
    if headline:
        n_msg, n_words, n_media, n_links = headline
        lines.append(f"- Messages: {n_msg}, Words: {n_words}, Media: {n_media}, Links: {n_links}")

    kpis = snapshot.get("kpis")
    if kpis:
        lines.append(
            f"- Avg msgs/day: {kpis.get('avg_messages_per_day')}, "
            f"Avg words/msg: {kpis.get('avg_words_per_message')}, "
            f"Most active hour: {kpis.get('most_active_hour')}, "
            f"Most active day: {kpis.get('most_active_day')}"
        )

    busiest = snapshot.get("busiest_users")
    if busiest:
        top = ", ".join(f"{u}: {c}" for u, c in list(busiest.items())[:5])
        lines.append(f"- Most active users (message count): {top}")

    resp = snapshot.get("response_time")
    if resp and resp.get("reply_count"):
        lines.append(
            f"- Avg reply time: {resp['avg_minutes']} min (median {resp['median_minutes']} min). "
            f"Fastest replier: {resp['fastest_user']} ({resp['fastest_minutes']} min). "
            f"Slowest replier: {resp['slowest_user']} ({resp['slowest_minutes']} min)."
        )

    starters = snapshot.get("starters")
    if starters:
        top_starters = ", ".join(f"{u}: {c}" for u, c in list(starters.items())[:5])
        lines.append(f"- Conversation starters: {top_starters}")

    sleep = snapshot.get("sleep")
    if sleep and sleep.get("sleep_start") is not None:
        lines.append(f"- Estimated quiet/sleep window: {sleep['sleep_start']:02d}:00 to {sleep['wake_up']:02d}:00")

    weekend = snapshot.get("weekend_vs_weekday")
    if weekend:
        lines.append(f"- Weekend vs weekday activity: {weekend}")

    sent = snapshot.get("sentiment")
    if sent:
        lines.append(f"- Sentiment distribution: {sent}")

    emo = snapshot.get("emotion")
    if emo:
        lines.append(f"- Emotion distribution: {emo}")

    eng = snapshot.get("engagement")
    if eng:
        top_eng = ", ".join(f"{r['user']}: {r['engagement_score']}" for r in eng[:5])
        lines.append(f"- Engagement scores (0-100): {top_eng}")

    tox = snapshot.get("toxicity")
    if tox:
        lines.append(f"- Toxicity labels: {tox}")

    lang = snapshot.get("language")
    if lang:
        lines.append(f"- Language mix: {lang}")

    return "\n".join(lines) if lines else "(not enough data for detailed stats in this selection)"


def _stats_block(selected_user: str, df: pd.DataFrame, scope: str = "full") -> str:
    # Gated timer (config.PERFORMANCE_DEBUG only) -- see nav.time_block.
    # Lets the performance debug panel show how much of an AI feature's
    # total latency is "prep our own numbers" vs. the actual LLM request.
    with nav.time_block("ai_stats_snapshot"):
        snapshot = _stats_snapshot(selected_user, df, scope=scope)
    return _format_snapshot(snapshot)


# ---------------------------------------------------------------------------
# 1-4. Summaries (Chat / Daily / Weekly / Monthly)
# ---------------------------------------------------------------------------
def _summarize_window(selected_user: str, df: pd.DataFrame, window_df: pd.DataFrame,
                       period_label: str, scope_label: str, n_paragraphs: int = 2) -> str:
    if window_df.empty:
        return f"No messages found for this {period_label.lower()}."
    return _run(
        prompt.SUMMARY_PROMPT,
        feature=f"summary_{period_label.lower().replace(' ', '_')}",
        period_label=period_label,
        scope_label=scope_label,
        stats_block=_stats_block(selected_user, window_df, scope="summary"),
        messages_block=_messages_block(window_df, selected_user),
        n_paragraphs=n_paragraphs,
    )


def ai_chat_summary(selected_user: str, df: pd.DataFrame, n_paragraphs: int = 3) -> str:
    """Summarize the entire current selection (whatever the sidebar
    filters currently narrow the chat down to)."""
    scope = "All time (current filters)" if selected_user == "Overall" else f"User: {selected_user}"
    return _summarize_window(selected_user, df, df, "Overall chat", scope, n_paragraphs)


def ai_daily_summary(selected_user: str, df: pd.DataFrame) -> str:
    """Summarize the most recent calendar day present in `df`."""
    real = df.dropna(subset=['date'])
    if real.empty:
        return "No messages available for a daily summary."
    latest_date = real['only_date'].max()
    window = real[real['only_date'] == latest_date]
    return _summarize_window(
        selected_user, df, window, "Daily", f"Date: {latest_date}", n_paragraphs=1,
    )


def ai_weekly_summary(selected_user: str, df: pd.DataFrame) -> str:
    """Summarize the most recent 7-day window present in `df`."""
    real = df.dropna(subset=['date'])
    if real.empty:
        return "No messages available for a weekly summary."
    latest = real['date'].max()
    window = real[real['date'] > latest - pd.Timedelta(days=7)]
    return _summarize_window(
        selected_user, df, window, "Weekly", f"Last 7 days ending {latest.date()}", n_paragraphs=2,
    )


def ai_monthly_summary(selected_user: str, df: pd.DataFrame) -> str:
    """Summarize the most recent 30-day window present in `df`."""
    real = df.dropna(subset=['date'])
    if real.empty:
        return "No messages available for a monthly summary."
    latest = real['date'].max()
    window = real[real['date'] > latest - pd.Timedelta(days=30)]
    return _summarize_window(
        selected_user, df, window, "Monthly", f"Last 30 days ending {latest.date()}", n_paragraphs=3,
    )


def _det_message_counts(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    stats = helper.fetch_stats(selected_user, df)
    if not stats:
        return None
    n_msg, n_words, n_media, n_links = stats
    scope = "Overall, this selection has" if selected_user == "Overall" else f"{selected_user} sent"
    return (
        f"{scope} **{n_msg}** messages, **{n_words}** words, {n_media} media items, "
        f"and {n_links} links in the current selection."
    )


def _det_most_active_user(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    if selected_user != "Overall":
        return (
            f"You've filtered to a single participant (**{selected_user}**), so there's no one "
            "to compare against here -- switch to \"Overall\" in the sidebar to see who's most active."
        )
    busiest = helper.most_busy_users(df)[0]
    if busiest is None or busiest.empty:
        return None
    top_user, top_count = busiest.index[0], int(busiest.iloc[0])
    return f"**{top_user}** is the most active participant, with **{top_count}** messages in the current selection."


def _det_fastest_replier(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    resp = analytics.response_time_stats(selected_user, df)
    if not resp or not resp.get("reply_count"):
        return None
    return (
        f"**{resp['fastest_user']}** replies fastest, averaging **{resp['fastest_minutes']} min**. "
        f"The overall average reply time in this selection is {resp['avg_minutes']} min "
        f"(median {resp['median_minutes']} min)."
    )


def _det_slowest_replier(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    resp = analytics.response_time_stats(selected_user, df)
    if not resp or not resp.get("reply_count"):
        return None
    return f"**{resp['slowest_user']}** is the slowest to reply, averaging **{resp['slowest_minutes']} min**."


def _det_most_active_hour(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    hour = utils.kpi_most_active_hour(df)
    if hour == "N/A":
        return None
    return f"The most active hour in this selection is **{hour}**."


def _det_most_active_day(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    day = utils.kpi_most_active_day(df)
    if day == "N/A":
        return None
    return f"The most active day of the week in this selection is **{day}**."


def _det_avg_messages_per_day(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    return f"On average, this selection has **{utils.kpi_avg_messages_per_day(df)}** messages per active day."


def _det_avg_words_per_message(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    return f"On average, each message in this selection is **{utils.kpi_avg_words_per_message(df)}** words long."


def _det_longest_message(selected_user: str, df: pd.DataFrame) -> Optional[str]:
    return f"The longest message in this selection is **{utils.kpi_longest_message(df)}** words long."


# (patterns, handler) pairs, checked in order -- first match wins.
_DETERMINISTIC_HANDLERS: list[tuple[list[str], Callable[[str, pd.DataFrame], Optional[str]]]] = [
    ([r"who.*(talks?|messages?|chats?).*most", r"most active (user|person|participant)"], _det_most_active_user),
    ([r"who.*(repl(y|ies)|responds?).*fastest", r"fastest repl(y|ier)"], _det_fastest_replier),
    ([r"who.*(repl(y|ies)|responds?).*(slow|longest)", r"slowest repl(y|ier)"], _det_slowest_replier),
    ([r"most active hour", r"busiest hour", r"what (time|hour).*most active"], _det_most_active_hour),
    ([r"most active day", r"busiest day", r"what day.*most active"], _det_most_active_day),
    ([r"average.*messages?.*per day", r"avg.*messages?.*per day"], _det_avg_messages_per_day),
    ([r"average.*words?.*per message", r"avg.*words?.*per message"], _det_avg_words_per_message),
    ([r"longest message"], _det_longest_message),
    ([r"how many messages", r"total messages", r"message count", r"number of messages"], _det_message_counts),
]


def try_deterministic_answer(question: str, selected_user: str, df: pd.DataFrame) -> Optional[str]:
    """Return a Python-computed answer for `question` if it matches a
    known deterministic pattern, else None (meaning: fall through to the
    LLM/RAG path). Never raises -- a handler error just means "no match"."""
    q = (question or "").strip().lower()
    if not q:
        return None
    for patterns, handler in _DETERMINISTIC_HANDLERS:
        if any(re.search(p, q) for p in patterns):
            try:
                answer = handler(selected_user, df)
            except Exception:
                answer = None
            if answer:
                return answer
    return None


# ---------------------------------------------------------------------------
# 5 & 8. Ask Questions / Chat with History (RAG + computed stats)
# ---------------------------------------------------------------------------
def ask_question(
    question: str,
    selected_user: str,
    df: pd.DataFrame,
    chat_history: Optional[list[tuple[str, str]]] = None,
    fingerprint: Optional[str] = None,
) -> str:
    """Answer a free-text question about the chat.

    Phase 6: tries the deterministic Python router FIRST -- questions
    like "Who talks the most?" are answered directly from
    helper.py/analytics.py/utils.py with zero LLM calls and zero RAG
    indexing. Only questions that don't match a known pattern fall
    through to the grounded LLM flow, which combines two sources:
    - Exact computed stats (who talks most, reply speed, activity hours, etc.)
    - RAG-retrieved chat excerpts (for "what did we say about X" questions),
      built/loaded lazily HERE, on submit -- never before.

    Args:
        question: The user's question, e.g. "Who replies fastest?".
        selected_user: "Overall" or a specific participant (sidebar scope).
        df: The (sidebar-filtered) chat DataFrame.
        chat_history: Prior (question, answer) turns in this session, most
            recent last. Trimmed to `config.AI_MEMORY_TURNS` automatically.
        fingerprint: Stable chat+filters+user cache key for the RAG index
            (see app.py's `get_ai_fingerprint()`). Reusing this instead of
            re-deriving it from `df` means asking a different question
            about the same selection never triggers a re-embed.

    Returns:
        The model's answer, or a setup/error message.
    """
    question = (question or "").strip()
    if not question:
        return "Ask me something about this chat -- e.g. \"Who talks the most?\" or \"When am I most active?\""

    if config.AI_DETERMINISTIC_ROUTING:
        deterministic = try_deterministic_answer(question, selected_user, df)
        if deterministic is not None:
            return deterministic

    history = (chat_history or [])[-config.AI_MEMORY_TURNS:]
    history_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in history) or "(no prior turns)"

    return _run(
        prompt.QA_PROMPT,
        feature="ask_question",
        stats_block=_stats_block(selected_user, df, scope="minimal"),
        context_block=rag.build_context_block(df, question, fingerprint=fingerprint),
        history_block=history_block,
        question=question,
    )


# ---------------------------------------------------------------------------
# 6. AI Insights
# ---------------------------------------------------------------------------
def generate_insights(selected_user: str, df: pd.DataFrame) -> str:
    return _run(prompt.INSIGHTS_PROMPT, feature="insights", stats_block=_stats_block(selected_user, df))


# ---------------------------------------------------------------------------
# 9. Semantic Search (retrieval only — no LLM call, so it works even
# with the local hashing-embeddings fallback and costs nothing). The
# vector index is built lazily inside rag.retrieve() the first time a
# query is actually submitted — never just from opening this sub-tab.
# ---------------------------------------------------------------------------
def semantic_search(df: pd.DataFrame, query: str, k: int = 10, fingerprint: Optional[str] = None) -> list[dict]:
    return rag.retrieve(df, query, k=k, fingerprint=fingerprint)


# ---------------------------------------------------------------------------
# 10. AI Recommendations
# ---------------------------------------------------------------------------
def ai_recommendations(selected_user: str, df: pd.DataFrame) -> str:
    """Generate exactly five actionable AI recommendations.

    The first LLM call is validated. If it doesn't contain exactly five
    numbered recommendations, at most ONE corrective retry is performed
    (see config.AI_MAX_RETRY, default 1) -- and only if the first
    response was an actual (if malformed) answer, never if it was
    already a quota/auth/invalid/setup failure. Retrying a failed
    request would just burn a second request on a quota that's already
    known to be a problem.
    """
    stats_block = _stats_block(selected_user, df, scope="recommendations")

    result = _run(prompt.RECOMMENDATIONS_PROMPT, feature="recommendations", stats_block=stats_block)

    # First response is already valid -- most common case, 1 request total.
    if _count_numbered_items(result) == 5:
        return result

    # The first call itself failed (quota/auth/invalid/setup) -- do NOT
    # retry; that would spend a second request for no benefit.
    if _is_error_message(result):
        return result

    # Retry disabled via config -- return whatever we got.
    if config.AI_MAX_RETRY < 1:
        return result

    # Corrective retry: exactly one, and only reached when the first
    # call succeeded but the LLM didn't follow the "exactly 5" format.
    return _retry_recommendations(stats_block)


# ---------------------------------------------------------------------------
# 11. Smart Conversation Highlights
# ---------------------------------------------------------------------------
def conversation_highlights(selected_user: str, df: pd.DataFrame) -> str:
    return _run(prompt.HIGHLIGHTS_PROMPT, feature="highlights", messages_block=_messages_block(df, selected_user))


# ---------------------------------------------------------------------------
# 12. AI-generated Reports
# ---------------------------------------------------------------------------
def ai_report(selected_user: str, df: pd.DataFrame) -> str:
    return _run(prompt.REPORT_PROMPT, feature="report", stats_block=_stats_block(selected_user, df))


# ---------------------------------------------------------------------------
# 13. AI Personality Analysis (per-user; requires a specific participant)
# ---------------------------------------------------------------------------
def personality_analysis(user: str, df: pd.DataFrame) -> str:
    if user == "Overall":
        return "Pick a specific participant (not \"Overall\") to run a personality read."
    return _run(
        prompt.PERSONALITY_PROMPT,
        feature="personality",
        user=user,
        stats_block=_stats_block(user, df, scope="personality"),
        messages_block=_messages_block(df, user, limit=150),
    )


# ---------------------------------------------------------------------------
# 14. AI Friendship / Relationship Analysis (between two participants)
# ---------------------------------------------------------------------------
def friendship_analysis(df: pd.DataFrame, user_a: str, user_b: str) -> str:
    if user_a == user_b:
        return "Pick two different participants to analyze their dynamic."

    pair_df = df[df['user'].isin([user_a, user_b])]
    stats_lines = []
    try:
        comparison = analytics.compare_users(df, user_a, user_b)
        stats_lines.append(f"- Side-by-side comparison: {comparison.to_dict('records')}")
    except Exception:
        pass
    try:
        resp = analytics.response_time_stats("Overall", pair_df)
        if resp.get("reply_count"):
            stats_lines.append(
                f"- Reply timing between them: avg {resp['avg_minutes']} min, "
                f"fastest {resp['fastest_user']} ({resp['fastest_minutes']} min)"
            )
    except Exception:
        pass
    try:
        sent = sentiment.sentiment_by_user(pair_df)
        stats_lines.append(f"- Sentiment per person (in their exchanges): {sent.to_dict('records')}")
    except Exception:
        pass

    stats_block = "\n".join(stats_lines) or "(not enough shared data between these two)"
    return _run(
        prompt.FRIENDSHIP_PROMPT, feature="friendship", user_a=user_a, user_b=user_b, stats_block=stats_block,
    )


# ---------------------------------------------------------------------------
# 15. AI Conversation Quality Score
# ---------------------------------------------------------------------------
def compute_quality_score(selected_user: str, df: pd.DataFrame) -> dict:
    """Compute a deterministic 0-100 quality score from existing metrics
    (Python-computed, NOT the LLM) so the number is stable and
    reproducible. The LLM (see `conversation_quality_score`) only
    narrates/explains this already-final number.

    Signals blended (equal-ish weighting, each normalized to 0-1):
    - Positive sentiment ratio
    - Inverse average reply time (faster = better)
    - Low toxicity rate
    - Day-to-day consistency (active days / span of days)
    - Balance of participation (Overall only; 1.0 for a single user)
    """
    signals: dict[str, float] = {}

    try:
        sent = sentiment.sentiment_distribution(selected_user, df)
        total = sent['count'].sum() if not sent.empty else 0
        positive = sent.loc[sent['sentiment'] == 'Positive', 'count'].sum() if not sent.empty else 0
        signals['positive_ratio'] = float(positive / total) if total else 0.5
    except Exception:
        signals['positive_ratio'] = 0.5

    try:
        resp = analytics.response_time_stats(selected_user, df)
        if resp.get('reply_count'):
            # 0 min -> 1.0, 60+ min -> ~0.0
            signals['response_speed'] = max(0.0, 1.0 - min(resp['avg_minutes'], 60) / 60)
        else:
            signals['response_speed'] = 0.5
    except Exception:
        signals['response_speed'] = 0.5

    try:
        tox = toxicity.toxicity_summary(selected_user, df)
        total = tox['count'].sum() if not tox.empty else 0
        clean = tox.loc[tox['label'] == 'Clean', 'count'].sum() if not tox.empty else total
        signals['clean_ratio'] = float(clean / total) if total else 1.0
    except Exception:
        signals['clean_ratio'] = 1.0

    try:
        real = df.dropna(subset=['date'])
        active_days = real['only_date'].nunique()
        span_days = max(1, (real['only_date'].max() - real['only_date'].min()).days + 1)
        signals['consistency'] = min(1.0, active_days / span_days)
    except Exception:
        signals['consistency'] = 0.5

    try:
        if selected_user == "Overall":
            counts = df[df['user'] != config.GROUP_NOTIFICATION_USER]['user'].value_counts()
            if len(counts) >= 2:
                share = counts / counts.sum()
                # 1.0 = perfectly balanced, lower = one person dominates
                signals['balance'] = float(1 - (share.max() - 1 / len(share)))
            else:
                signals['balance'] = 1.0
        else:
            signals['balance'] = 1.0
    except Exception:
        signals['balance'] = 1.0

    weights = {
        'positive_ratio': 0.30, 'response_speed': 0.25,
        'clean_ratio': 0.20, 'consistency': 0.15, 'balance': 0.10,
    }
    score = sum(signals[k] * w for k, w in weights.items()) * 100
    return {"score": round(score, 1), "signals": {k: round(v, 3) for k, v in signals.items()}}


def conversation_quality_score(selected_user: str, df: pd.DataFrame) -> dict:
    """Full feature: computes the deterministic score, then asks the LLM
    to narrate/justify it (never to change it).

    Returns:
        dict with keys: score (float), signals (dict), narrative (str).
    """
    result = compute_quality_score(selected_user, df)
    stats_block = "\n".join(f"- {k}: {v}" for k, v in result["signals"].items())
    narrative = _run(
        prompt.QUALITY_SCORE_PROMPT, feature="quality_narrative", stats_block=stats_block, score=result["score"],
    )
    result["narrative"] = narrative
    return result
