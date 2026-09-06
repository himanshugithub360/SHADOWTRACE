
from __future__ import annotations

import importlib.util
import re
from collections import Counter

import pandas as pd
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config
import nav
import nlp_helper

BERTOPIC_AVAILABLE = importlib.util.find_spec("bertopic") is not None

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

KEYBERT_AVAILABLE = importlib.util.find_spec("keybert") is not None


@st.cache_resource(show_spinner="Preparing keyword extraction...")
def _get_keybert_model():
    """Lazily import and instantiate (once per session) the KeyBERT
    model. Only runs the first time KeyBERT-backed keyword extraction is
    actually used — never at module import / app startup."""
    from keybert import KeyBERT
    return KeyBERT()


def _corpus(selected_user: str, df: pd.DataFrame, min_words: int = 3) -> list[str]:
    real = nlp_helper.clean_messages(df, selected_user)
    if real.empty:
        return []
    return [m for m in real['message'].astype(str) if len(m.split()) >= min_words]


# ---------------------------------------------------------------------------
# Topic Modeling
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Discovering conversation themes...")
def _topic_model_cached(
    _df: pd.DataFrame, key: str, selected_user: str, n_topics: int, top_n_words: int,
) -> pd.DataFrame:
    docs = _corpus(selected_user, _df)
    if len(docs) < max(10, n_topics * 2):
        return pd.DataFrame(columns=['topic', 'size', 'top_words'])

    if BERTOPIC_AVAILABLE:
        try:
            from bertopic import BERTopic  # lazy: only imported when actually used
            model = BERTopic(nr_topics=n_topics, verbose=False)
            model.fit_transform(docs)
            info = model.get_topic_info()
            rows = []
            for _, row in info.iterrows():
                if row['Topic'] == -1:
                    continue  # BERTopic's outlier bucket, not a real topic
                words = [w for w, _ in model.get_topic(row['Topic'])[:top_n_words]]
                rows.append({'topic': int(row['Topic']), 'size': int(row['Count']), 'top_words': ", ".join(words)})
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            pass  # fall through to the LDA fallback below

    return _lda_topics(docs, n_topics, top_n_words)


def topic_model(selected_user: str, df: pd.DataFrame, n_topics: int = 5, top_n_words: int = 8) -> pd.DataFrame:
    """Discover the top discussion topics in this selection.

    Tries BERTopic first (transformer embeddings + HDBSCAN clustering —
    generally better topic separation on short, informal chat text).
    Falls back to scikit-learn's LDA (classic bag-of-words topic model)
    if BERTopic isn't installed, or if it raises for any reason (e.g.
    too little text to cluster).

    Cached per (message content, selected_user, n_topics, top_n_words):
    BERTopic in particular is expensive to fit, and neither it nor LDA
    needs re-fitting on a rerun that doesn't change the selection or
    the topic-count/word-count controls.

    Returns:
        DataFrame with columns: topic, size, top_words.
    """
    key = nlp_helper.content_key(df)
    return _topic_model_cached(df, key, selected_user, n_topics, top_n_words)


def _lda_topics(docs: list[str], n_topics: int, top_n_words: int) -> pd.DataFrame:
    stopwords = list(nlp_helper.load_stopwords_combined())
    vectorizer = CountVectorizer(
        stop_words=stopwords, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", max_df=0.9, min_df=2,
    )
    try:
        dtm = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=['topic', 'size', 'top_words'])
    if dtm.shape[1] == 0:
        return pd.DataFrame(columns=['topic', 'size', 'top_words'])

    n_topics = max(1, min(n_topics, dtm.shape[0]))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    doc_topic = lda.fit_transform(dtm)
    terms = vectorizer.get_feature_names_out()
    assignments = doc_topic.argmax(axis=1)
    sizes = Counter(assignments)

    rows = []
    for topic_idx, component in enumerate(lda.components_):
        top_idx = component.argsort()[::-1][:top_n_words]
        words = [terms[i] for i in top_idx]
        rows.append({'topic': topic_idx, 'size': int(sizes.get(topic_idx, 0)), 'top_words': ", ".join(words)})
    return pd.DataFrame(rows).sort_values('size', ascending=False)


# ---------------------------------------------------------------------------
# Keyword Extraction — TF-IDF (always available) / YAKE / KeyBERT.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _keywords_tfidf_cached(_df: pd.DataFrame, key: str, selected_user: str, top_n: int) -> pd.DataFrame:
    docs = _corpus(selected_user, _df, min_words=1)
    if not docs:
        return pd.DataFrame(columns=['keyword', 'score'])
    stopwords = list(nlp_helper.load_stopwords_combined())
    vectorizer = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=['keyword', 'score'])
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda t: t[1], reverse=True)[:top_n]
    return pd.DataFrame(ranked, columns=['keyword', 'score']).round({'score': 3})


def keywords_tfidf(selected_user: str, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """TF-IDF keyword ranking: needs only scikit-learn, so this is the
    baseline every other method falls back to. Cached per (message
    content, selected_user, top_n) — also reused as the fallback body
    for keywords_yake()/keywords_keybert() when their package isn't
    installed, so switching methods without changing the selection
    doesn't re-run TF-IDF from scratch."""
    key = nlp_helper.content_key(df)
    return _keywords_tfidf_cached(df, key, selected_user, top_n)


@st.cache_data(show_spinner=False)
def _keywords_yake_cached(_df: pd.DataFrame, key: str, selected_user: str, top_n: int) -> pd.DataFrame:
    docs = _corpus(selected_user, _df, min_words=1)
    if not docs:
        return pd.DataFrame(columns=['keyword', 'score'])
    text = " ".join(docs)
    extractor = yake.KeywordExtractor(lan="en", n=2, top=top_n)
    pairs = extractor.extract_keywords(text)
    out = pd.DataFrame(pairs, columns=['keyword', 'raw_score'])
    # YAKE: lower raw_score = more relevant. Invert to a "higher = better" scale for display.
    out['score'] = round(1 / (1 + out['raw_score']), 4)
    return out[['keyword', 'score']].sort_values('score', ascending=False)


def keywords_yake(selected_user: str, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """YAKE keyword extraction — statistical and unsupervised (no
    training/model download needed). Falls back to TF-IDF if `yake`
    isn't installed. Cached per (message content, selected_user, top_n)."""
    if not YAKE_AVAILABLE:
        return keywords_tfidf(selected_user, df, top_n)
    docs_key = nlp_helper.content_key(df)
    return _keywords_yake_cached(df, docs_key, selected_user, top_n)


@st.cache_data(show_spinner="Extracting keywords...")
def _keywords_keybert_cached(_df: pd.DataFrame, key: str, selected_user: str, top_n: int) -> pd.DataFrame:
    docs = _corpus(selected_user, _df, min_words=1)
    if not docs:
        return pd.DataFrame(columns=['keyword', 'score'])
    text = " ".join(docs)[:20000]  # cap input size so this stays responsive in a UI
    try:
        with nav.time_block("model_loading"):
            model = _get_keybert_model()
        pairs = model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n,
        )
    except Exception:
        return keywords_tfidf(selected_user, _df, top_n)
    return pd.DataFrame(pairs, columns=['keyword', 'score']).round({'score': 4})


def keywords_keybert(selected_user: str, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """KeyBERT keyword extraction — embedding-based, so it captures
    phrases that are semantically central rather than just frequent.
    Falls back to TF-IDF if `keybert` (and its `sentence-transformers`
    dependency) isn't installed. `_get_keybert_model()` (imported/built
    once per session via `st.cache_resource`) is never even called here
    unless KeyBERT is both installed AND this specific function is
    invoked — selecting TF-IDF or YAKE never touches this path.

    Cached per (message content, selected_user, top_n): KeyBERT's
    embedding inference is the most expensive keyword method, so a
    rerun that doesn't change the selection or top_n reuses the prior
    extraction instead of re-embedding the corpus.
    """
    if not KEYBERT_AVAILABLE:
        return keywords_tfidf(selected_user, df, top_n)
    key = nlp_helper.content_key(df)
    return _keywords_keybert_cached(df, key, selected_user, top_n)


# ---------------------------------------------------------------------------
# Text Summarization — extractive, frequency/TF-IDF based ("TextRank-lite"):
# scores each sentence by its summed TF-IDF weight, keeps the top-scoring
# sentences, and re-orders them back to their original position. Runs
# fully offline with no model download, which matters for a report that
# should regenerate on every rerun.
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def summarize_text(text: str, n_sentences: int = 5) -> str:
    """Extractive summary of arbitrary text down to `n_sentences`."""
    sentences = _split_sentences(text)
    if not sentences:
        return "Not enough text to summarize."
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    stopwords = list(nlp_helper.load_stopwords_combined())
    vectorizer = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:n_sentences])

    scores = matrix.sum(axis=1).A1
    top_idx = sorted(scores.argsort()[::-1][:n_sentences])  # restore original order
    return " ".join(sentences[i] for i in top_idx)


def summarize_chat(selected_user: str, df: pd.DataFrame, n_sentences: int = 6) -> str:
    """Summarize whatever messages are already in `df` — the caller
    (app.py) is expected to have applied any date window first."""
    real = nlp_helper.clean_messages(df, selected_user)
    if real.empty:
        return "No messages in this selection to summarize."
    text = ". ".join(real['message'].astype(str))
    return summarize_text(text, n_sentences)


def summarize_period(selected_user: str, df: pd.DataFrame, period: str = "W", n_sentences: int = 6) -> str:
    """Summarize the most recent 7 days ('W') or 30 days ('M') relative
    to the latest message date in the (already filtered) DataFrame."""
    real = nlp_helper.clean_messages(df, selected_user).dropna(subset=['date'])
    if real.empty:
        return "No messages to summarize."
    days = 7 if period == "W" else 30
    cutoff = real['date'].max() - pd.Timedelta(days=days)
    window = real[real['date'] >= cutoff]
    if window.empty:
        return f"No messages in the last {days} days."
    text = ". ".join(window['message'].astype(str))
    return summarize_text(text, n_sentences)


# ---------------------------------------------------------------------------
# Intent Classification — keyword-driven classifier over a fixed label
# set (Study / Work / Travel / Shopping / Movies / Sports). A lightweight,
# explainable alternative to training a text classifier, reasonable for
# short chat messages where a handful of trigger words reliably signal
# the topic. See config.INTENT_KEYWORDS.
# ---------------------------------------------------------------------------
def classify_intent(text: str) -> str:
    text_l = (text or "").lower()
    if not text_l.strip():
        return "Other"
    scores = {intent: sum(1 for kw in kws if kw in text_l) for intent, kws in config.INTENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def intent_distribution(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    real = nlp_helper.clean_messages(df, selected_user)
    if real.empty:
        return pd.DataFrame(columns=['intent', 'count', 'percent'])
    intents = real['message'].apply(classify_intent)
    counts = intents.value_counts().reset_index()
    counts.columns = ['intent', 'count']
    counts['percent'] = round(counts['count'] / counts['count'].sum() * 100, 2)
    return counts


def intent_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Crosstab of user x intent message counts (group chats)."""
    real = nlp_helper.clean_messages(df, "Overall")
    if real.empty:
        return pd.DataFrame()
    real = real.assign(intent=real['message'].apply(classify_intent))
    return pd.crosstab(real['user'], real['intent'])
