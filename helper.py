
from __future__ import annotations

from collections import Counter
from typing import Tuple

import emoji
import pandas as pd
import streamlit as st
from urlextract import URLExtract
from wordcloud import WordCloud

import config
import nlp_helper

extract = URLExtract()


def _load_stopwords() -> set[str]:
    """Load the Hinglish stop-word list used to clean text for the
    word cloud and most-common-words chart.

    Returns:
        A set of lowercase stop words for fast membership testing.
    """
    with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return set(f.read().split())


# ---------------------------------------------------------------------------
# Headline stats
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _fetch_stats_cached(_df: pd.DataFrame, key: str, selected_user: str) -> Tuple[int, int, int, int]:
    df = _df
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == config.MEDIA_PLACEHOLDER].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def fetch_stats(selected_user: str, df: pd.DataFrame) -> Tuple[int, int, int, int]:
    """Compute the four headline stats: messages, words, media, links.

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The (optionally pre-filtered) chat DataFrame.

    Returns:
        A tuple of (num_messages, num_words, num_media_messages, num_links).

    Cached per (message content, selected_user) -- see module docstring.
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _fetch_stats_cached(df, key, selected_user)


@st.cache_data(show_spinner=False)
def _most_busy_users_cached(_df: pd.DataFrame, key: str) -> Tuple[pd.Series, pd.DataFrame]:
  
    _df = _df[_df["user"] != config.GROUP_NOTIFICATION_USER]
    x = _df["user"].value_counts().head()
    percent_df = round((_df['user'].value_counts() / _df.shape[0]) * 100, 2).reset_index()
    percent_df.rename(columns={'user': 'name', 'count': 'percent'}, inplace=True)
    return x, percent_df


def most_busy_users(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Find the most active participants in a group chat.

    Args:
        df: The full (Overall) chat DataFrame.

    Returns:
        A tuple of:
        - Series of the top-5 users' message counts (for a bar chart).
        - DataFrame of every user's percentage share of total messages.

    Cached per message content -- only the 'user' column is consumed,
    so this only re-runs when the actual set of rows/senders changes.
    """
    key = nlp_helper.content_key(df, columns=("user",))
    return _most_busy_users_cached(df, key)


# ---------------------------------------------------------------------------
# Word cloud / word frequency / emoji
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building your word cloud...")
def _create_wordcloud_cached(_df: pd.DataFrame, key: str, selected_user: str):
    stop_words = _load_stopwords()
    df = _df

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df["user"] != config.GROUP_NOTIFICATION_USER]
    temp = temp[temp["message"] != config.MEDIA_PLACEHOLDER]

    def remove_stop_words(message: str) -> str:
        return " ".join(word for word in message.lower().split() if word not in stop_words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    cleaned = temp["message"].apply(remove_stop_words)
    df_wc = wc.generate(cleaned.str.cat(sep=" ") or " ")

    return df_wc.to_array()


def create_wordcloud(selected_user: str, df: pd.DataFrame):
    """Generate a word cloud image for the selected user's messages.

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A `wordcloud.WordCloud`-rendered numpy array, suitable for
        `st.image()`. (No matplotlib required.)

    Cached per (message content, selected_user): `WordCloud.generate()`
    is the single most expensive per-rerun call in the app on a large
    chat, and it was previously re-run on *every* Streamlit rerun,
    including ones triggered by widgets in unrelated tabs.
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _create_wordcloud_cached(df, key, selected_user)


@st.cache_data(show_spinner=False)
def _most_common_words_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    stop_words = _load_stopwords()
    df = _df

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df["user"] != config.GROUP_NOTIFICATION_USER]
    temp = temp[temp["message"] != config.MEDIA_PLACEHOLDER]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))


def most_common_words(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Find the 20 most frequently used words (stop words excluded).

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A two-column DataFrame: word (0) and count (1).

    Cached per (message content, selected_user) -- see module docstring.
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _most_common_words_cached(df, key, selected_user)


@st.cache_data(show_spinner=False)
def _emoji_helper_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    df = _df
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df["message"]:
        emojis.extend(c for c in message if c in emoji.EMOJI_DATA)

    counts = Counter(emojis)
    return pd.DataFrame(counts.most_common(len(counts)))


def emoji_helper(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Count emoji usage frequency.

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A two-column DataFrame: emoji (0) and count (1), sorted descending.

    Cached per (message content, selected_user): the character-by-
    character emoji scan over every message is O(total characters),
    the same cost class as the word cloud. `Counter(emojis)` is also
    now built once instead of twice (the original built it, then
    rebuilt it again inside `len(Counter(emojis))`).
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _emoji_helper_cached(df, key, selected_user)


# ---------------------------------------------------------------------------
# Timelines / activity maps
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _monthly_timeline_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    df = _df
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + " " + timeline['year'].astype(str)

    return timeline


def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Message count grouped by calendar month (e.g. 'March 2024').

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A DataFrame with columns year, month_num, month, message, time.
    """
    key = nlp_helper.content_key(df, columns=("user", "year", "month_num", "month"))
    return _monthly_timeline_cached(df, key, selected_user)


@st.cache_data(show_spinner=False)
def _daily_timeline_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    df = _df
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Message count grouped by calendar date.

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A DataFrame with columns only_date, message.
    """
    key = nlp_helper.content_key(df, columns=("user", "only_date"))
    return _daily_timeline_cached(df, key, selected_user)


def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Message count grouped by day-of-week (Monday-Sunday).

    Not cached: a single `.value_counts()` over an already-`category`-
    typed column (see data_layer._optimize_dtypes) is effectively O(1)
    relative to the loop-based functions above, so a cache lookup would
    cost about as much as just computing it.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Message count grouped by month name (January-December).

    Not cached -- see `week_activity_map`.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df["month"].value_counts()


def hourly_activity(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Message count grouped by hour-of-day (0-23).

    New in this phase - powers the interactive "Hourly Analysis" chart.
    Not cached -- see `week_activity_map`.

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A Series indexed 0-23 with message counts, sorted by hour.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['hour'].value_counts().sort_index()


def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Build a day-of-week x hour-period pivot table for the heatmap chart.

    Args:
        selected_user: "Overall" or a specific participant's name.
        df: The chat DataFrame.

    Returns:
        A pivot table (day_name x period) of message counts, 0-filled.

    Not cached -- a single `pivot_table` over two `category`-typed
    columns is already cheap; see `week_activity_map`.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(
        index='day_name', columns='period', values='message', aggfunc='count'
    ).fillna(0)
