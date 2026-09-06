
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import pandas as pd
import streamlit as st

import preprocessor
import utils

# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def compute_fingerprint(raw_bytes: bytes) -> str:
    """Stable, cheap hash of the uploaded file's raw bytes.

    Computed once per upload (in app.py, right after `getvalue()`) and
    threaded through every cached function below instead of letting
    Streamlit hash the full decoded text on every cache lookup.
    """
    return hashlib.md5(raw_bytes).hexdigest()



_DECODE_FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-16", "cp1252")


def _looks_like_utf16(raw_bytes: bytes) -> bool:
    """Heuristic for whether `raw_bytes` is plausibly UTF-16, so the
    UTF-16 fallback attempt is only made when there's real evidence for
    it. This matters because single-byte legacy encodings (cp1252,
    latin-1) never raise UnicodeDecodeError on ANY byte sequence -- and
    neither, in practice, does UTF-16 on most even-length inputs -- so a
    plain "try encodings in order, catch the exception" strategy can't
    tell them apart by success/failure alone. A genuine UTF-16 file
    either starts with a byte-order mark, or (lacking one) is mostly
    ASCII text encoded as UTF-16, which means roughly every other byte
    is 0x00 -- a legacy single-byte export essentially never looks like
    that.
    """
    if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return True
    sample = raw_bytes[:4000]
    if not sample:
        return False
    return (sample.count(0) / len(sample)) > 0.3


def decode_uploaded_chat(raw_bytes: bytes) -> tuple[str, bool]:
    """Decode raw uploaded chat bytes into text, without ever raising.

    Args:
        raw_bytes: The exact bytes read from the uploaded .txt file.

    Returns:
        (text, used_fallback) where `used_fallback` is True only when
        plain UTF-8 (with or without a leading BOM) could not decode the
        file and a less-common fallback encoding — or, as a last resort,
        lossy replacement — had to be used instead. Callers can use this
        flag to show a brief, non-alarming notice rather than silently
        producing possibly-garbled text.

    Encoding order:
        1. utf-8-sig — plain UTF-8 (the overwhelming majority of exports,
           including Hindi/Devanagari, emoji, and mixed-language chats)
           AND UTF-8-with-BOM in one attempt, since utf-8-sig only strips
           a BOM when present and otherwise behaves exactly like utf-8.
        2. utf-16 — only attempted when `_looks_like_utf16` finds real
           evidence for it (BOM, or a high null-byte ratio); some export
           tools (mostly older/Android backup utilities) save as UTF-16,
           but since UTF-16 rarely raises on arbitrary bytes, attempting
           it unconditionally would silently shadow the cp1252 fallback
           below with garbled (but "successfully decoded") text.
        3. cp1252 — a common Windows-locale fallback for older exports.
           Like UTF-16, this never raises on arbitrary bytes, so it's
           deliberately last among the "real" attempts.
        4. utf-8 with errors="replace" — final, never-fails fallback so
           an unrecognized/binary-like file still produces *something*
           parseable instead of crashing the upload.
    """
    if not raw_bytes:
        return "", False

    try:
        return raw_bytes.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        pass

    if _looks_like_utf16(raw_bytes):
        try:
            return raw_bytes.decode("utf-16"), True
        except UnicodeDecodeError:
            pass

    try:
        return raw_bytes.decode("cp1252"), True
    except UnicodeDecodeError:
        pass

    return raw_bytes.decode("utf-8", errors="replace"), True


# ---------------------------------------------------------------------------
# Dtype optimization
# ---------------------------------------------------------------------------
# NOTE: 'month' is intentionally NOT converted to category. helper.py's
# monthly_timeline() and analytics.py's emoji_timeline_monthly() /
# emoji_timeline_by_user() do `df['month'] + " " + df['year'].astype(str)`,
# and pandas does not support `+` on a Categorical Series. Converting it
# would silently break those (out-of-scope-for-this-part) modules, so
# 'month' stays a plain string/object column exactly as preprocessor.py
# produces it.
_CATEGORY_COLS = ["user", "day_name", "period"]

_SMALL_INT_DTYPES = {
    "year": "Int16",
    "month_num": "Int8",
    "day": "Int8",
    "hour": "Int8",
    "minute": "Int8",
    "week_num": "Int16",
}


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast the cheap, universally-reused columns produced by
    preprocessor.preprocess(). Runs exactly once, right after parsing.

    - user / day_name / period -> category (repeated string values,
      used constantly in `==`, `.isin()`, `groupby()`, `value_counts()`,
      `pivot_table()` — all category-safe).
    - year / month_num / day / hour / minute / week_num -> small
      nullable integer dtypes (values are tiny and bounded; nullable
      because unparseable rows leave NaN/NaT in the date-derived
      columns).
    - 'message', 'date', 'only_date', 'date_str' are left untouched:
      every NLP module (sentiment.py, topics.py, toxicity.py,
      nlp_helper.py) relies on `.str` / `.apply()` over 'message' as a
      plain string column, and 'date'/'only_date' are already
      appropriately typed by preprocessor.py.
    """
    if df.empty:
        return df

    for col in _CATEGORY_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col, dtype in _SMALL_INT_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


# ---------------------------------------------------------------------------
# Cached preprocessing — the ONE place the raw .txt gets parsed.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading your conversation...")
def load_and_preprocess(_raw_text: str, fingerprint: str) -> pd.DataFrame:
    """Parse + dtype-optimize the chat exactly once per unique file.

    Args:
        _raw_text: Full decoded chat text. Leading underscore tells
            `st.cache_data` to skip hashing it.
        fingerprint: `compute_fingerprint(raw_bytes)` for this file —
            the actual cache key.

    Returns:
        The preprocessed, dtype-optimized DataFrame. Cached: calling
        this again with the same fingerprint returns the same object
        without re-parsing or re-optimizing.
    """
    df = preprocessor.preprocess(_raw_text)
    return _optimize_dtypes(df)


# ---------------------------------------------------------------------------
# Cached filtering layer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_filtered_view(
    _df: pd.DataFrame,
    fingerprint: str,
    selected_user: str,
    years: Sequence = (),
    months: Sequence = (),
    weeks: Sequence = (),
    days: Sequence = (),
    date_range: Optional[tuple] = None,
    hour_range: tuple = (0, 23),
) -> pd.DataFrame:
    """The single place `selected_user` + every sidebar filter gets
    applied to the base chat DataFrame.

    Every tab in app.py should read from the DataFrame this returns
    instead of doing its own `df[df["user"] == selected_user]` — that
    keeps a chat with N tabs from repeating the same boolean mask N
    times per rerun.

    Args:
        _df: The full, preprocessed chat DataFrame (unhashed — see
            module docstring).
        fingerprint: The current chat's fingerprint. Included so this
            cache can NEVER return a filtered slice of a *different*
            chat than the one currently loaded, even if `_df` itself
            isn't part of the cache key.
        selected_user: "Overall" or a specific participant.
        years, months, weeks, days: Hashable sequences (pass tuples,
            ideally pre-sorted so equivalent selections in a different
            order still hit the same cache entry).
        date_range: (start_date, end_date) tuple, or None.
        hour_range: (start_hour, end_hour) tuple.

    Returns:
        The filtered DataFrame for this exact combination of
        fingerprint + user + filters.
    """
    base = _df if selected_user == "Overall" else _df[_df["user"] == selected_user]
    return utils.apply_filters(
        base,
        years=years,
        months=months,
        weeks=weeks,
        days=days,
        date_range=date_range,
        hour_range=hour_range,
    )
