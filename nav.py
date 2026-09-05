"""
nav.py
======
Part 4 — True lazy section loading.

Why this file exists
---------------------
`st.tabs()` looks like it gives you separate pages, but it doesn't:
Streamlit executes the body of EVERY `with tab:` block on EVERY rerun,
regardless of which tab is visually active. With 27 tabs — several of
them running NLP models, topic modeling, toxicity scoring, or touching
the AI/RAG stack — that means switching to "Overview" was silently
re-running Sentiment, Topics, Toxicity, and everything else, every
single time.

This module replaces that with manual, conditional navigation:
exactly one top-level section (and, inside it, exactly one
sub-section) is tracked in `st.session_state`. app.py branches on the
returned string with plain `if/elif` — never `st.tabs()` — so only the
`render_*()` function matching the current selection ever executes.

Nothing in this module touches pandas, NLP, or AI — it's pure
Streamlit UI plumbing, safe to import from app.py only.
"""
from __future__ import annotations

import base64
import html
import time
from typing import Sequence

import streamlit as st

import config


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    """Base64-embed the gradient-diamond brand mark (same asset used for
    the browser favicon in config.PAGE_ICON) so the sidebar wordmark can
    inline it without Streamlit needing to serve it as a static file.
    Cached: reads the PNG at most once per process; returns "" (silently
    omitted by the caller) if the asset is ever missing."""
    try:
        data = config.LOGO_PATH.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _brand_logo_html(size: int = 26) -> str:
    uri = _logo_data_uri()
    if not uri:
        return ""
    return f'<img class="st-brand-logo" src="{uri}" width="{size}" height="{size}" alt="" />'

TOP_SECTIONS: list[str] = [
    "Overview",
    "Activity",
    "People",
    "Conversations",
    "Insights",
    "AI Assistant",
    "Tools",
]

# Small, single-glyph accents (not the primary visual identity — text
# carries that). Used only for lightweight visual polish in the nav bar.
SECTION_ICONS: dict[str, str] = {
    "Overview": "◆",
    "Activity": "◷",
    "People": "◉",
    "Conversations": "◈",
    "Insights": "◎",
    "AI Assistant": "✦",
    "Tools": "◇",
}


def _has_segmented_control() -> bool:
    """`st.segmented_control` only exists on newer Streamlit versions.
    Degrade to a horizontal radio (same single-choice semantics) rather
    than hard-requiring a specific version."""
    return hasattr(st, "segmented_control")


def render_sidebar_brand() -> None:
    """Small wordmark shown at the very top of the sidebar, above the
    navigation rail — see reference layout: brand stays visible no
    matter which section is active, the way a real product's sidebar
    header does. Pure markup; no state, no computation."""
    st.sidebar.markdown(
        f"""
        <div class="st-sidebar-brand">
            <div class="st-sidebar-brand-word st-brand-row">{_brand_logo_html(26)}SHADOW<span>TRACE</span></div>
            <div class="st-sidebar-brand-tag">Every conversation leaves a trace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_nav(key: str = "nav_section") -> str:
    """Render the primary section switcher as a horizontal pill navbar
    at the top of the main content area (Overview / Activity / People /
    Conversations / Insights / AI Assistant / Tools) — NOT in the
    sidebar, which is reserved for the current-analysis summary and
    filters.

    Returns the single selected section label. Because this is a plain
    single-select widget (not `st.tabs()`), the caller can safely do
    `if selected == "...": render_x()` and know that ONLY that branch
    runs this rerun — switching sections never executes the other six
    `render_*()` functions.
    """
    if key not in st.session_state:
        st.session_state[key] = TOP_SECTIONS[0]

    def _labelled(section: str) -> str:
        icon = SECTION_ICONS.get(section, "")
        return f"{icon}  {section}" if icon else section

    with st.container(key="top_navbar"):
        if _has_segmented_control():
            choice = st.segmented_control(
                "Navigate",
                TOP_SECTIONS,
                key=f"{key}_widget",
                default=st.session_state[key],
                required=True,
                format_func=_labelled,
                label_visibility="collapsed",
                width="stretch",
            )
        else:
            choice = st.radio(
                "Navigate",
                TOP_SECTIONS,
                key=f"{key}_widget",
                index=TOP_SECTIONS.index(st.session_state[key]),
                format_func=_labelled,
                label_visibility="collapsed",
                horizontal=True,
            )

    st.session_state[key] = choice
    return choice


def render_sidebar_analysis_summary(filename: str, message_count: int, participant_count: int) -> bool:
    """"Current analysis" card at the top of the dashboard sidebar's
    filter area (see reference design): which chat is active, how big
    it is, and an "Upload New Chat" escape hatch back to the landing
    page. Purely presentational — the counts are computed by the
    caller from data that's already loaded; this function does not
    touch the DataFrame, caching, or fingerprints itself.

    Returns True on the rerun "Upload New Chat" is clicked. app.py
    decides what resetting session_state actually means (see
    `_reset_to_landing` there) — this module only reports the click.
    """
    safe_name = html.escape(filename)
    with st.sidebar:
        st.markdown(
            f"""
            <div class="st-current-analysis">
                <div class="st-current-analysis-label">Current analysis</div>
                <div class="st-current-analysis-file">📄 {safe_name}</div>
                <div class="st-current-analysis-meta">
                    {message_count:,} messages · {participant_count} participant(s)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        clicked = st.button("⬆ Upload New Chat", key="_upload_new_chat", width='stretch')
    return clicked


def sub_nav(label: str, options: Sequence[str], key: str) -> str:
    """Render a secondary, in-section switcher with the same
    exactly-one-branch-executes contract as `top_nav`.

    Args:
        label: Widget label shown above the control.
        options: The sub-section choices for this section.
        key: Unique session_state key for this control (one per
            section, e.g. "activity_sub", "insights_sub").
    """
    options = list(options)
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = options[0]

    # Wrapped in a keyed container so CSS (.st-key-subnav_*) can give
    # every in-section tab strip a consistent "underline tab" look,
    # visually distinct from the pill-shaped main navbar above it.
    with st.container(key=f"subnav_{key}"):
        if _has_segmented_control():
            choice = st.segmented_control(label, options, default=st.session_state[key], key=f"{key}_widget")
            choice = choice or st.session_state[key]
        else:
            choice = st.radio(
                label, options, horizontal=True, key=f"{key}_widget",
                index=options.index(st.session_state[key]),
            )

    st.session_state[key] = choice
    return choice


# ---------------------------------------------------------------------------
# Part 8 — Lightweight performance debug mode.
# ---------------------------------------------------------------------------
# `time_block` is a lean, opt-in stopwatch: a complete no-op (no
# time.perf_counter() call, no session_state write) whenever
# config.PERFORMANCE_DEBUG is False, so it costs nothing for normal
# users. When enabled, wrapping a block with `with nav.time_block("x"):`
# records how long that block took the LAST time it ran, overwriting the
# previous value for that label -- same "persists until the next time
# this runs" spirit as `log_exec` below, not accumulated/summed and not
# reset every rerun (so a label from an earlier rerun stays visible
# until that stage runs again, instead of flickering to zero).
def _perf_enabled() -> bool:
    return bool(getattr(config, "PERFORMANCE_DEBUG", False))


class time_block:
    """Coarse-grained timer for the performance debug panel. See module
    note above. Usage:

        with nav.time_block("preprocessing"):
            df = data_layer.load_and_preprocess(raw_data, fingerprint)
    """
    __slots__ = ("label", "_start")

    def __init__(self, label: str):
        self.label = label
        self._start = None

    def __enter__(self):
        if _perf_enabled():
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            timings = st.session_state.setdefault("_perf_timings", {})
            timings[self.label] = elapsed
        return False  # never swallow exceptions


def set_perf_count(label: str, value) -> None:
    """Record a non-timing debug metric (e.g. messages processed). Also
    a no-op unless config.PERFORMANCE_DEBUG is True."""
    if _perf_enabled():
        counts = st.session_state.setdefault("_perf_counts", {})
        counts[label] = value


# ---------------------------------------------------------------------------
# Temporary execution-proof debug logging.
# ---------------------------------------------------------------------------
# This is scaffolding for PROVING lazy loading works, not a permanent
# feature. Safe to delete once verified. It deliberately does nothing
# expensive — just a timestamp + function name, capped history — so it
# can stay on during Part 4 review without affecting performance.
def log_exec(name: str) -> None:
    """Call this as the FIRST line of every render_*() function. Each
    call appends one line to a visible log — if a section's name never
    appears in the log, its render function never ran."""
    log = st.session_state.setdefault("_exec_log", [])
    log.append(f"{time.strftime('%H:%M:%S')} — {name}()")
    st.session_state["_exec_log"] = log[-30:]  # keep it short


def render_debug_panel() -> None:
    """Sidebar panels for lazy-loading verification and performance
    timing. Both gated behind `config.PERFORMANCE_DEBUG` -- neither is
    meant for normal end users; this is a developer-only tool. The
    lazy-execution log was originally always-on scaffolding used to
    prove `nav.py`'s section-switching never runs more than the active
    section/sub-section per rerun (see the log_exec() docstring above);
    it's gated the same way as the performance timers now so a single
    setting controls the whole debug surface instead of two.
    """
    if not _perf_enabled():
        return

    with st.sidebar.expander("🐛 Debug: lazy execution log", expanded=False):
        log = st.session_state.get("_exec_log", [])
        if not log:
            st.caption("No sections rendered yet.")
        else:
            st.caption("Newest first — only the active section/sub-section should add a line per rerun.")
            for line in reversed(log):
                st.text(line)
        if st.button("Clear log", key="_clear_exec_log", width='stretch'):
            st.session_state["_exec_log"] = []
            st.rerun()

    with st.sidebar.expander("🐛 Performance Debug", expanded=False):
        timings = st.session_state.get("_perf_timings", {})
        counts = st.session_state.get("_perf_counts", {})
        if not timings and not counts:
            st.caption("No timings recorded yet — interact with the app once.")
        else:
            st.caption("From the most recent rerun that executed each stage.")
            row_labels = [
                ("startup", "Startup"),
                ("preprocessing", "Preprocessing"),
                ("filtering", "Filtering"),
                ("section", "Section"),
                ("model_loading", "Model loading"),
                ("rag_indexing", "RAG indexing"),
                ("ai_stats_snapshot", "AI stats prep"),
                ("ai_request", "AI request (LLM)"),
            ]
            for key, label in row_labels:
                if key in timings:
                    st.text(f"{label}: {timings[key]:.2f} s")
                elif key in ("model_loading", "rag_indexing", "ai_stats_snapshot", "ai_request"):
                    st.text(f"{label}: not executed")
            if "messages_processed" in counts:
                st.text(f"Messages processed: {counts['messages_processed']:,}")
            if "ai_provider_used" in counts:
                st.text(f"AI provider that answered: {counts['ai_provider_used']}")
            for _provider in ("gemini", "openrouter", "groq", "mistral"):
                _key = f"{_provider}_timeout_kwarg"
                if _key in counts:
                    st.text(f"{_provider.capitalize()} request timeout sent: {counts[_key]}")
            if "gemini_thinking_kwarg" in counts:
                st.text(f"Gemini thinking budget sent: {counts['gemini_thinking_kwarg']}")
            if "ai_truncation_flagged" in counts:
                st.text(f"Last AI response flagged truncated: {counts['ai_truncation_flagged']}")
            if "ai_retry_fired" in counts:
                st.text(f"Last AI request retried: {counts['ai_retry_fired']}")
            # Deferred import: nav.py is otherwise pure Streamlit UI
            # plumbing (see module docstring) — this one line is
            # debug-only and gated behind config.PERFORMANCE_DEBUG,
            # so normal users never trigger the import or see a
            # provider name (see the no-developer-UI rule).
            import vector_store
            if vector_store.EMBEDDINGS_BACKEND != "none":
                st.text(
                    f"Embedding backend last used: {vector_store.EMBEDDINGS_BACKEND} "
                    f"({vector_store.EMBEDDINGS_MODEL})"
                )
        if st.button("Clear timings", key="_clear_perf_timings", width='stretch'):
            st.session_state["_perf_timings"] = {}
            st.session_state["_perf_counts"] = {}
            st.rerun()
