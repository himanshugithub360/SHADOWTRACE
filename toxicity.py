"""
toxicity.py
===========
Phase 4 — NLP: toxicity/abuse/hate-speech detection and spam detection.

Toxicity detection uses a small, generic keyword heuristic (see
`config.PROFANITY_WORDS`) — deliberately mild and non-exhaustive, a
stand-in signal rather than moderation-grade coverage. This module
previously supported an optional Detoxify (torch/transformers) backend
for higher-quality scoring; that path has been removed to keep the app
lightweight and fast to start (no heavy model download/load). See
MIGRATION.md if you want to reintroduce a transformer-based backend.
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

import config
import nlp_helper

# Kept as a constant (rather than removed outright) so any external code
# still checking this flag degrades gracefully instead of erroring.
DETOXIFY_AVAILABLE = False


def detoxify_ready() -> bool:
    """Always False now that Detoxify support has been removed. Kept for
    API compatibility with callers like ai_helper.py that ask "has the
    heavy model already been paid for this session?" — the answer is
    now unconditionally no, since there's no heavy model at all."""
    return False


_TOXICITY_KEYS = ("toxicity", "obscene", "insult", "threat", "identity_attack")


# ---------------------------------------------------------------------------
# Toxicity Detection
# ---------------------------------------------------------------------------
def score_toxicity_batch(texts: list[str]) -> list[dict]:
    """Score many messages in one pass using the fast keyword heuristic.

    Returns:
        A list of sub-score dicts, same length and order as `texts`.
    """
    if not texts:
        return []
    cleaned = [(t or "").strip() for t in texts]
    return [_heuristic_toxicity(t) if t else {k: 0.0 for k in _TOXICITY_KEYS} for t in cleaned]


def score_toxicity(text: str) -> dict:
    """Sub-scores in [0, 1]: toxicity, obscene, insult, threat,
    identity_attack, for a single message. Uses Detoxify when
    installed. Kept for API compatibility / one-off lookups — anything
    scoring more than one message (toxicity_table/summary/by_user)
    should call `score_toxicity_batch()` directly instead."""
    return score_toxicity_batch([text])[0]


def _heuristic_toxicity(text: str) -> dict:
    text_l = text.lower()
    words = re.findall(r"[a-zA-Z']+", text_l)
    if not words:
        return {k: 0.0 for k in _TOXICITY_KEYS}

    hits = sum(1 for w in words if w in config.PROFANITY_WORDS)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    exclaim_boost = min(text.count('!'), 3) * 0.02
    caps_boost = 0.1 if (caps_ratio > 0.6 and len(text) > 8) else 0.0

    toxicity = min(1.0, round(hits / len(words) + caps_boost + exclaim_boost, 4))
    return {"toxicity": toxicity, "obscene": toxicity, "insult": toxicity, "threat": 0.0, "identity_attack": 0.0}


def classify_toxicity(scores: dict) -> str:
    """Map sub-scores to a single label: Clean / Offensive / Abusive / Hate Speech."""
    if scores["identity_attack"] >= config.TOXICITY_THRESHOLD:
        return "Hate Speech"
    if scores["threat"] >= config.TOXICITY_THRESHOLD or scores["insult"] >= config.TOXICITY_THRESHOLD:
        return "Abusive"
    if scores["toxicity"] >= config.TOXICITY_THRESHOLD * 0.6:
        return "Offensive"
    return "Clean"


@st.cache_data(show_spinner="Reviewing message tone...")
def _toxicity_scored_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    """`clean_messages` output with every sub-score + the final label
    attached, computed via ONE batched `score_toxicity_batch()` call.
    toxicity_table(), toxicity_summary(), and toxicity_by_user() all
    read from this instead of each re-scoring the same messages."""
    real = nlp_helper.clean_messages(_df, selected_user)
    if real.empty:
        empty_cols = {k: pd.Series(dtype=float) for k in _TOXICITY_KEYS}
        return real.assign(**empty_cols, label=pd.Series(dtype=str))

    scores = score_toxicity_batch(real['message'].tolist())
    out = real.copy()
    for k in _TOXICITY_KEYS:
        out[k] = [s[k] for s in scores]
    out['label'] = [classify_toxicity(s) for s in scores]
    return out


def _toxicity_scored(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    return _toxicity_scored_cached(df, nlp_helper.content_key(df), selected_user)


def toxicity_table(selected_user: str, df: pd.DataFrame, only_flagged: bool = True) -> pd.DataFrame:
    """Per-message toxicity scores + label. `only_flagged=True` returns
    just the non-Clean messages, for a moderation review table."""
    cols = ['date', 'user', 'message', 'label', 'toxicity']
    scored = _toxicity_scored(selected_user, df)
    if scored.empty:
        return pd.DataFrame(columns=cols)

    out = scored[cols]
    if only_flagged:
        out = out[out['label'] != "Clean"]
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out.sort_values('toxicity', ascending=False)


def toxicity_summary(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Count of messages per toxicity label."""
    scored = _toxicity_scored(selected_user, df)
    if scored.empty:
        return pd.DataFrame(columns=['label', 'count', 'percent'])
    counts = scored['label'].value_counts().reset_index()
    counts.columns = ['label', 'count']
    counts['percent'] = round(counts['count'] / counts['count'].sum() * 100, 2)
    return counts


def toxicity_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each user's messages flagged as non-Clean (group chats)."""
    scored = _toxicity_scored("Overall", df)
    if scored.empty:
        return pd.DataFrame(columns=['user', 'flagged_pct'])
    out = scored.groupby('user')['label'].apply(lambda s: round((s != 'Clean').mean() * 100, 2)).reset_index()
    out.columns = ['user', 'flagged_pct']
    return out.sort_values('flagged_pct', ascending=False)


# ---------------------------------------------------------------------------
# Spam Detection — heuristic scoring. There's no labeled spam dataset for
# a personal WhatsApp export, so this uses interpretable signals instead
# of a trained classifier: link floods, repeated characters, ALL-CAPS
# shouting, promotional keywords, and exact-duplicate messages resent
# several times (a classic forward/spam pattern).
# ---------------------------------------------------------------------------
def spam_score(text: str, link_count: int = 0) -> float:
    text = text or ""
    if not text.strip() or text == config.MEDIA_PLACEHOLDER:
        return 0.0

    score = 0.0
    if link_count >= 2:
        score += 0.4
    if re.search(r'(.)\1{4,}', text):  # e.g. "heyyyyyy", "!!!!!"
        score += 0.2

    words = text.split()
    if words:
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        if caps_words / len(words) > 0.5 and len(words) > 3:
            score += 0.2

    text_words = set(text.lower().split())
    if config.SPAM_KEYWORDS & text_words:
        score += 0.3
    if len(text) > 300 and link_count >= 1:
        score += 0.2

    return round(min(1.0, score), 2)


def is_spam(text: str, link_count: int = 0) -> bool:
    return spam_score(text, link_count) >= config.SPAM_SCORE_THRESHOLD


@st.cache_data(show_spinner=False)
def _spam_table_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    from urlextract import URLExtract
    extractor = URLExtract()

    real = nlp_helper.clean_messages(_df, selected_user, drop_media=False)
    if real.empty:
        return pd.DataFrame(columns=['date', 'user', 'message', 'spam_score'])

    dup_counts = real['message'].value_counts()
    rows = []
    for _, r in real.iterrows():
        links = len(extractor.find_urls(r['message']))
        score = spam_score(r['message'], links)
        if dup_counts.get(r['message'], 0) >= config.SPAM_DUPLICATE_THRESHOLD:
            score = min(1.0, score + 0.3)
        if score >= config.SPAM_SCORE_THRESHOLD:
            rows.append({'date': r['date'], 'user': r['user'], 'message': r['message'], 'spam_score': score})

    if not rows:
        return pd.DataFrame(columns=['date', 'user', 'message', 'spam_score'])
    return pd.DataFrame(rows).sort_values('spam_score', ascending=False)


def spam_table(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Flag likely-spam messages: link floods, shouting, and near-duplicate
    messages (the same text repeated 3+ times counts as a spam/forward
    signal on its own, on top of its individual spam_score).

    Cached per (message content incl. media placeholders, selected_user)
    — spam_summary() below calls this too, so a rerun computes the link
    counts / duplicate scan once, not twice.
    """
    key = nlp_helper.content_key(df)
    return _spam_table_cached(df, key, selected_user)


def spam_summary(selected_user: str, df: pd.DataFrame) -> dict:
    """Headline numbers for the Spam Detection panel."""
    real = nlp_helper.clean_messages(df, selected_user, drop_media=False)
    total = len(real) if not real.empty else 0
    flagged = len(spam_table(selected_user, df))
    return {
        "total_messages": total,
        "flagged_messages": flagged,
        "spam_rate_pct": round(flagged / total * 100, 2) if total else 0.0,
    }
