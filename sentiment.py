"""
sentiment.py
============
Phase 4 — NLP: sentiment analysis (VADER), emotion detection, and
per-user / per-month "most important words" (TF-IDF weighted).

Sentiment engine
-----------------
Uses the self-contained `vaderSentiment` package (bundles its own
lexicon, so no NLTK corpus download is required at runtime). VADER is
rule-based and tuned for short, informal, emoji/slang-heavy text —
a good fit for WhatsApp messages — and returns a "compound" score in
[-1, 1] combining positive/negative/neutral intensity.

If `vaderSentiment` isn't installed, every function here still runs and
returns a neutral (0.0 / "Neutral") result instead of crashing —
`VADER_AVAILABLE` tells app.py whether to show a "using placeholder"
notice.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import nlp_helper

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _analyzer = None
    VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------
def score_message(text: str) -> float:
    """VADER compound sentiment score for one message, in [-1, 1]."""
    if not VADER_AVAILABLE or not text:
        return 0.0
    return _analyzer.polarity_scores(text)['compound']


def _label(compound: float) -> str:
    if compound >= config.SENTIMENT_POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= config.SENTIMENT_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


@st.cache_data(show_spinner=False)
def _scored_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    real = nlp_helper.clean_messages(_df, selected_user)
    if real.empty:
        return real.assign(compound=pd.Series(dtype=float), sentiment=pd.Series(dtype=str))
    out = real.copy()
    out['compound'] = out['message'].apply(score_message)
    out['sentiment'] = out['compound'].apply(_label)
    return out


def _scored(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Real text messages for `selected_user`, with `compound` and
    `sentiment` columns attached.

    Cached per (message content, selected_user) — Part 5: this is the
    one place VADER actually scores messages. sentiment_distribution(),
    sentiment_timeline(), and sentiment_by_user() all call this, so
    caching it here (instead of in each of those) means a chat's
    messages get scored once per selection, not three times.
    """
    return _scored_cached(df, nlp_helper.content_key(df), selected_user)


def sentiment_distribution(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Positive / Neutral / Negative message counts + percentages."""
    scored = _scored(selected_user, df)
    if scored.empty:
        return pd.DataFrame(columns=['sentiment', 'count', 'percent'])
    counts = scored['sentiment'].value_counts()
    out = counts.reindex(['Positive', 'Neutral', 'Negative']).fillna(0).astype(int).reset_index()
    out.columns = ['sentiment', 'count']
    total = out['count'].sum()
    out['percent'] = round(out['count'] / total * 100, 2) if total else 0.0
    return out


def sentiment_timeline(
    selected_user: str,
    df: pd.DataFrame,
    freq: str = "M"
) -> pd.DataFrame:

    """Average sentiment compound score resampled to a chosen cadence

    (D=daily, W=weekly, M=monthly), plus message count per bucket."""

    scored = _scored(selected_user, df).dropna(subset=['date'])

    if scored.empty:
        return pd.DataFrame(columns=['period', 'avg_compound', 'messages'])

    # Pandas 2.2+ uses "ME" for month-end instead of "M"
    if freq == "M":
        freq = "ME"

    ts = scored.set_index(pd.to_datetime(scored['date']))

    out = ts.resample(freq).agg(
        avg_compound=('compound', 'mean'),
        messages=('compound', 'size')
    )

    out = out.dropna().reset_index().rename(columns={'date': 'period'})
    out['avg_compound'] = out['avg_compound'].round(3)

    return out


def sentiment_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment distribution + avg compound score per user (group chats)."""
    scored = _scored("Overall", df)
    if scored.empty:
        return pd.DataFrame(columns=['user', 'positive_pct', 'neutral_pct', 'negative_pct', 'avg_compound'])
    rows = []
    for user, grp in scored.groupby('user'):
        counts = grp['sentiment'].value_counts()
        total = len(grp)
        rows.append({
            'user': user,
            'positive_pct': round(counts.get('Positive', 0) / total * 100, 2),
            'neutral_pct': round(counts.get('Neutral', 0) / total * 100, 2),
            'negative_pct': round(counts.get('Negative', 0) / total * 100, 2),
            'avg_compound': round(grp['compound'].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values('avg_compound', ascending=False)


# ---------------------------------------------------------------------------
# Emotion Detection — lightweight lexicon + emoji cue classifier
# (Happy / Sad / Angry / Excited), with a VADER-polarity fallback when no
# cue matches so every message still gets a label instead of "Unknown".
# ---------------------------------------------------------------------------
def detect_emotion(text: str) -> str:
    text_l = (text or "").lower()
    if not text_l.strip():
        return "Neutral"

    scores = {emotion: 0 for emotion in config.EMOTION_LEXICON}
    for emotion, cues in config.EMOTION_LEXICON.items():
        scores[emotion] += sum(1 for cue in cues if cue in text_l)

    for ch in text:
        for emotion, emojis in config.EMOTION_EMOJI.items():
            if ch in emojis:
                scores[emotion] = scores.get(emotion, 0) + 1

    best_emotion = max(scores, key=scores.get)
    if scores[best_emotion] > 0:
        return best_emotion

    # No lexicon/emoji cue matched — fall back to a coarse VADER-based guess.
    compound = score_message(text)
    if compound >= 0.5:
        return "Excited"
    if compound > 0.05:
        return "Happy"
    if compound <= -0.4:
        return "Angry"
    if compound < -0.05:
        return "Sad"
    return "Neutral"


@st.cache_data(show_spinner=False)
def _emotion_scored(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    """`clean_messages` output with an `emotion` column attached,
    cached per (message content, selected_user) so emotion_distribution()
    and emotion_by_user() (called with different `selected_user` scopes)
    each classify their own selection's messages only once per rerun."""
    real = nlp_helper.clean_messages(_df, selected_user)
    if real.empty:
        return real.assign(emotion=pd.Series(dtype=str))
    return real.assign(emotion=real['message'].apply(detect_emotion))


def emotion_distribution(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    scored = _emotion_scored(df, nlp_helper.content_key(df), selected_user)
    if scored.empty:
        return pd.DataFrame(columns=['emotion', 'count', 'percent'])
    counts = scored['emotion'].value_counts().reset_index()
    counts.columns = ['emotion', 'count']
    counts['percent'] = round(counts['count'] / counts['count'].sum() * 100, 2)
    return counts


def emotion_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Crosstab of user x emotion message counts (group chats)."""
    scored = _emotion_scored(df, nlp_helper.content_key(df), "Overall")
    if scored.empty:
        return pd.DataFrame()
    return pd.crosstab(scored['user'], scored['emotion'])


# ---------------------------------------------------------------------------
# Most Important Words — by user / by month (TF-IDF, so a word distinctive
# to one user/month outranks words everybody uses equally often).
# ---------------------------------------------------------------------------
def _tfidf_top_terms(documents: list[str], labels: list[str], top_n: int = 10) -> pd.DataFrame:
    stopwords = list(nlp_helper.load_stopwords_combined())
    vectorizer = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return pd.DataFrame(columns=['group', 'word', 'score'])
    terms = vectorizer.get_feature_names_out()

    rows = []
    for i, label in enumerate(labels):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        for idx in top_idx:
            if row[idx] > 0:
                rows.append({'group': label, 'word': terms[idx], 'score': round(float(row[idx]), 4)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _important_words_by_user_cached(_df: pd.DataFrame, key: str, top_n: int) -> pd.DataFrame:
    real = nlp_helper.clean_messages(_df, "Overall")
    if real.empty:
        return pd.DataFrame(columns=['group', 'word', 'score'])
    grouped = real.groupby('user')['message'].apply(lambda s: " ".join(s)).reset_index()
    if len(grouped) < 2:
        return pd.DataFrame(columns=['group', 'word', 'score'])
    return _tfidf_top_terms(grouped['message'].tolist(), grouped['user'].tolist(), top_n)


def important_words_by_user(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Most distinctive words per user: each user's combined messages is
    treated as one TF-IDF "document", so words everyone uses score low
    and words characteristic of one person score high."""
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _important_words_by_user_cached(df, key, top_n)


@st.cache_data(show_spinner=False)
def _important_words_by_month_cached(_df: pd.DataFrame, key: str, selected_user: str, top_n: int) -> pd.DataFrame:
    real = nlp_helper.clean_messages(_df, selected_user).dropna(subset=['date'])
    if real.empty:
        return pd.DataFrame(columns=['group', 'word', 'score'])
    real = real.assign(period=real['date'].dt.strftime('%Y-%m'))
    grouped = real.groupby('period')['message'].apply(lambda s: " ".join(s)).reset_index()
    if len(grouped) < 2:
        return pd.DataFrame(columns=['group', 'word', 'score'])
    return _tfidf_top_terms(grouped['message'].tolist(), grouped['period'].tolist(), top_n)


def important_words_by_month(selected_user: str, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Most distinctive words per calendar month, for the selected user."""
    key = nlp_helper.content_key(df, columns=("user", "date", "message"))
    return _important_words_by_month_cached(df, key, selected_user, top_n)
