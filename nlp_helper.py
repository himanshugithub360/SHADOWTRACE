"""
nlp_helper.py
=============
Phase 4 — NLP: shared utilities used by sentiment.py, topics.py, and
toxicity.py — text cleaning, language detection, readability scoring,
and word similarity.

Design principle
-----------------
Every optional third-party library (`langdetect`, `textstat`) is
imported inside a try/except. If a package listed in requirements.txt
somehow isn't installed, the affected function degrades to a lighter,
dependency-free fallback instead of crashing the whole app — every
public function in this file always returns *something* usable.

Follows the same `(selected_user, df)` convention as helper.py /
analytics.py so it plugs into app.py the same way.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

# ---------------------------------------------------------------------------
# Part 5 — shared cache-key helper.
# ---------------------------------------------------------------------------
# sentiment.py, topics.py, toxicity.py, and this module all run per-message
# NLP work (VADER, emotion/intent lexicons, toxicity, TF-IDF/YAKE/KeyBERT,
# language detection...) keyed off `(selected_user, df)` — the same
# convention as helper.py. `filtered_df` itself is already produced by
# data_layer.get_filtered_view(), which is cached per
# (fingerprint, selected_user, filters), so hashing the *whole* DataFrame
# again here on every call would be redundant and, for `st.cache_data`,
# expensive (pandas hashing is O(n)).
#
# Instead every NLP-result cache in this project keys off `content_key()`:
# a cheap hash of just the 'message' column's values. That's sufficient to
# guarantee correctness (Requirement in Part 5 doc: "a new uploaded chat
# must never receive results from an old chat") because the message text
# is exactly the input every one of these functions actually consumes —
# any change in fingerprint, filters, or selected_user necessarily changes
# which rows/messages are present, hence the hash.
def content_key(df: pd.DataFrame, columns: tuple[str, ...] = ("user", "message")) -> str:
    """Cheap, deterministic cache key for a DataFrame's relevant columns.

    Args:
        df: A (possibly already user/filter-narrowed) chat DataFrame.
        columns: Which columns the downstream computation actually
            depends on. Defaults to ('user', 'message') since almost
            every caller here re-filters by `selected_user` internally
            (via `clean_messages`) — including 'user' in the key means
            two DataFrames that happen to share identical message text
            but different senders per row can never collide. Pass a
            narrower/wider tuple (e.g. add "date") for functions that
            group by another column too.

    Returns:
        A short string that changes whenever those columns' values (or
        row order) change, and stays identical across reruns for an
        unchanged selection — safe to pass as an explicit
        `st.cache_data` key alongside the DataFrame itself (passed as
        an underscore-prefixed arg so Streamlit doesn't hash it too).
    """
    if df.empty:
        return "empty"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "empty"
    return str(pd.util.hash_pandas_object(df[cols], index=True).sum())

try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0  # deterministic results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False


_WORD_RE = re.compile(r"[A-Za-z']+")
_HINGLISH_STOPWORDS: Optional[set[str]] = None


def _load_hinglish_stopwords() -> set[str]:
    with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return set(f.read().split())


def load_stopwords_combined() -> set[str]:
    """Hinglish stop-word list (reused from the word cloud feature) unioned
    with scikit-learn's standard English stop-word list. Shared across
    every Phase 4 TF-IDF-based function so results stay consistent."""
    global _HINGLISH_STOPWORDS
    if _HINGLISH_STOPWORDS is None:
        _HINGLISH_STOPWORDS = _load_hinglish_stopwords()
    return _HINGLISH_STOPWORDS | set(ENGLISH_STOP_WORDS)


def clean_messages(
    df: pd.DataFrame,
    selected_user: str,
    drop_media: bool = True,
    drop_notifications: bool = True,
) -> pd.DataFrame:
    """Filter a chat DataFrame down to real text messages for one user
    (or everyone). Every Phase 4 module starts from this.

    Args:
        df: The (already filtered-by-sidebar) chat DataFrame.
        selected_user: "Overall" or a specific participant's name.
        drop_media: Drop `<Media omitted>` placeholder rows.
        drop_notifications: Drop WhatsApp's own system messages.

    Returns:
        The filtered DataFrame.
    """
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    if drop_notifications:
        df = df[df['user'] != config.GROUP_NOTIFICATION_USER]
    if drop_media:
        df = df[df['message'] != config.MEDIA_PLACEHOLDER]
    return df


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """Classify one message as English / Hindi / Hinglish / Mixed / Unknown.

    WhatsApp chats are usually romanized Hindi ("Hinglish"), which a
    script-based language-ID model can't recognize as Hindi at all. So
    this combines two signals:
      - Devanagari script presence -> Hindi (native script).
      - `langdetect`'s verdict for English, plus a Hinglish stop-word
        ratio (romanized function words like "hai", "nahi", "kya", ...)
        as a heuristic for romanized Hindi when no Devanagari is present.
    """
    global _HINGLISH_STOPWORDS
    if _HINGLISH_STOPWORDS is None:
        _HINGLISH_STOPWORDS = _load_hinglish_stopwords()

    text = (text or "").strip()
    if not text:
        return "Unknown"

    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    words = _WORD_RE.findall(text.lower())
    if not words:
        return "Hindi" if has_devanagari else "Unknown"

    hinglish_hits = sum(1 for w in words if w in _HINGLISH_STOPWORDS and w.isascii())
    hinglish_ratio = hinglish_hits / len(words)

    detected_english = False
    if LANGDETECT_AVAILABLE:
        try:
            detected_english = detect(text) == 'en'
        except LangDetectException:
            detected_english = False
    else:
        detected_english = hinglish_ratio < 0.15

    if has_devanagari:
        return "Mixed" if detected_english else "Hindi"
    if hinglish_ratio >= 0.35:
        return "Hinglish"
    if hinglish_ratio >= 0.12:
        return "Mixed"
    return "English"


@st.cache_data(show_spinner=False)
def _language_distribution_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.Series:
    real = clean_messages(_df, selected_user)
    if real.empty:
        return pd.Series(dtype=int)
    # WhatsApp chats are full of exact-duplicate short messages ("ok",
    # "haha", a lone emoji, forwarded texts, ...). Classify each *unique*
    # message text once and reuse the result for every repeat instead of
    # re-running detect_language()/langdetect on the same string over and
    # over -- same output, far fewer langdetect calls on a typical chat.
    unique_messages = real['message'].unique()
    lang_by_text = {msg: detect_language(msg) for msg in unique_messages}
    return real['message'].map(lang_by_text).value_counts()


def language_distribution(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Message counts per detected language bucket.

    Cached per (message content, selected_user): language detection is
    a per-message loop (langdetect or the Devanagari/Hinglish
    heuristic), so a rerun that doesn't change the selection reuses the
    prior result instead of re-classifying every message.
    """
    return _language_distribution_cached(df, content_key(df), selected_user)


# ---------------------------------------------------------------------------
# Readability Score
# ---------------------------------------------------------------------------
def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def readability_score(text: str) -> float:
    """Flesch Reading Ease score (0-100; higher = easier to read).
    Uses `textstat` if installed, otherwise a manual implementation of
    the same formula with a naive vowel-group syllable counter."""
    text = (text or "").strip()
    if not text:
        return 0.0

    if TEXTSTAT_AVAILABLE:
        try:
            return round(float(textstat.flesch_reading_ease(text)), 2)
        except Exception:
            pass  # fall through to the manual formula below

    sentences = max(1, len(re.split(r'[.!?]+', text)))
    words = _WORD_RE.findall(text) or text.split()
    n_words = max(1, len(words))
    syllables = sum(_count_syllables(w) for w in words) or n_words
    score = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (syllables / n_words)
    return round(score, 2)


@st.cache_data(show_spinner=False)
def _readability_by_user_cached(_df: pd.DataFrame, key: str) -> pd.DataFrame:
    real = clean_messages(_df, "Overall")
    if real.empty:
        return pd.DataFrame(columns=['user', 'avg_readability'])
    scored = real.assign(score=real['message'].apply(readability_score))
    out = scored.groupby('user')['score'].mean().round(2).reset_index()
    out.columns = ['user', 'avg_readability']
    return out.sort_values('avg_readability', ascending=False)


def readability_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Average Flesch Reading Ease score per user (group chats).

    Cached per message content — avoids re-scoring every message on
    every rerun that doesn't change the underlying selection.
    """
    return _readability_by_user_cached(df, content_key(df))


# ---------------------------------------------------------------------------
# Word Similarity
# ---------------------------------------------------------------------------
# Both functions below fit a fresh TF-IDF matrix as their "embedding".
# word_similarity() and most_similar_words() are typically queried
# several times in a row for the same selection (different word pairs /
# target words picked from the same dropdown), so the expensive part —
# fitting the vectorizer over the corpus — is cached per
# (content, selected_user, min_df); scoring a specific word/pair against
# an already-fit matrix is cheap and stays uncached (it's just an index
# lookup + a single cosine_similarity call).
@st.cache_data(show_spinner=False)
def _tfidf_matrix_cached(_df: pd.DataFrame, key: str, selected_user: str, min_df: int):
    real = clean_messages(_df, selected_user)
    if real.empty:
        return None
    corpus = real['message'].str.lower().tolist()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=min_df)
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return None
    return matrix, vectorizer.vocabulary_


def word_similarity(df: pd.DataFrame, selected_user: str, word_a: str, word_b: str) -> Optional[float]:
    """Cosine similarity between two words, based on the TF-IDF signature
    each word has across every message in the chat -- a lightweight
    distributional-similarity stand-in that needs no pretrained word
    vectors (works entirely from this chat's own text).

    Returns:
        A similarity score in [0, 1], or None if either word never
        appears in the selection.
    """
    fitted = _tfidf_matrix_cached(df, content_key(df), selected_user, min_df=1)
    if fitted is None:
        return None
    matrix, vocab = fitted

    wa, wb = word_a.lower().strip(), word_b.lower().strip()
    if wa not in vocab or wb not in vocab:
        return None

    col_a = matrix[:, vocab[wa]].toarray().ravel()
    col_b = matrix[:, vocab[wb]].toarray().ravel()
    sim = cosine_similarity(col_a.reshape(1, -1), col_b.reshape(1, -1))[0][0]
    return round(float(sim), 4)


def most_similar_words(df: pd.DataFrame, selected_user: str, word: str, top_n: int = 10) -> pd.DataFrame:
    """Rank every other word in the corpus by similarity to `word`,
    using each word's TF-IDF vector across messages (word x message
    matrix) as its distributional signature."""
    fitted = _tfidf_matrix_cached(df, content_key(df), selected_user, min_df=2)
    if fitted is None:
        return pd.DataFrame(columns=['word', 'similarity'])
    matrix, vocab = fitted

    target = word.lower().strip()
    if target not in vocab:
        return pd.DataFrame(columns=['word', 'similarity'])

    word_vectors = matrix.T  # rows become word-vectors across messages
    target_vec = word_vectors[vocab[target]]
    sims = cosine_similarity(target_vec, word_vectors).ravel()
    inv_vocab = {i: w for w, i in vocab.items()}

    ranked = sorted(
        ((inv_vocab[i], round(float(s), 4)) for i, s in enumerate(sims) if inv_vocab[i] != target),
        key=lambda t: t[1], reverse=True,
    )[:top_n]
    return pd.DataFrame(ranked, columns=['word', 'similarity'])
