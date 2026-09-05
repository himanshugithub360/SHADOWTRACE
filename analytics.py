"""
analytics.py
============
Advanced analytics for the WhatsApp Chat Analyzer — Phase 3.

This module is purely additive: nothing in helper.py, charts.py, or
preprocessor.py imports it, so its presence can never break Phase 1/2
behavior. app.py is the only file that calls into it, through a new
set of tabs appended after the original seven.

Calling convention
-------------------
Most functions here follow the same `(selected_user, df)` pattern used
throughout helper.py, so they drop straight into the existing sidebar
filtering flow in app.py. A few need two usernames (User Comparison) or
operate chat-wide by nature (Engagement Score, Emoji Timeline by User)
and take `df` alone instead.

Sections (matches the Phase 3 feature list)
--------------------------------------------
 1. Response Time Analysis
 2. Conversation Starters
 3. Average Message Length
 4. Longest Message
 5. Weekend vs Weekday
 6. Peak Activity
 7. User Comparison
 8. Emoji Timeline
 9. Most Used Phrases (n-grams)
10. Shared Domains
11. Calendar Heatmap
12. Sleep Schedule Estimation
13. Weekly Report
14. Message Frequency
15. Engagement Score

Part 7 — caching
------------------
Every function that scans every message character-by-character (emoji
counting), tokenizes every message (n-grams), regex-matches every
message (shared domains), or sorts/diffs the full timeline
(`_reply_pairs`, `_mark_conversation_starts`) is now cached with
`@st.cache_data`, keyed on a cheap content hash via
`nlp_helper.content_key` — the same convention already used by
sentiment.py/topics.py/toxicity.py.

This matters for two reasons specific to this module:

1. `filtered_df` is cached by data_layer, but Streamlit reruns this
   entire module's calls on *every* widget interaction anywhere in the
   app, not just when a Phase 3 filter changes.
2. Several public functions here call a shared private helper
   (`_reply_pairs`, `_mark_conversation_starts`) independently, so
   without caching, a single rerun could redundantly re-sort/re-diff
   the *same* DataFrame 2–4 times (once per caller) even before
   counting the cross-tab reruns from #1.

Functions that only do a `groupby`/`value_counts`/`pivot_table` over
already-small or already-`category`-typed columns (`peak_hours`,
`quiet_hours`, `calendar_heatmap_data`, `estimate_sleep_schedule`,
`message_frequency`) are left uncached: a cache lookup would cost
about as much as just recomputing them, and they don't scale with
message *text* size the way the loop-based functions do.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import emoji
import pandas as pd
import streamlit as st
from urlextract import URLExtract

import config
import nlp_helper

extract = URLExtract()


# ---------------------------------------------------------------------------
# Shared internals — small building blocks reused by several sections below.
# ---------------------------------------------------------------------------
def _real_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Drop WhatsApp's own 'group_notification' rows (joins/leaves/etc.)."""
    return df[df['user'] != config.GROUP_NOTIFICATION_USER]


def _non_media(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `<Media omitted>` placeholder rows, keeping only real text."""
    return df[df['message'] != config.MEDIA_PLACEHOLDER]


def _filter_user(df: pd.DataFrame, selected_user: str) -> pd.DataFrame:
    """Narrow to one participant unless `selected_user == 'Overall'`."""
    if selected_user != 'Overall':
        return df[df['user'] == selected_user]
    return df


def _count_emojis(text: str) -> int:
    """Count emoji characters in a single message string."""
    return sum(1 for ch in text if ch in emoji.EMOJI_DATA)


def _count_links(text: str) -> int:
    """Count URLs in a single message string."""
    return len(extract.find_urls(text))


def _load_stopwords() -> set[str]:
    """Load the same Hinglish stop-word list helper.py uses, so phrase
    extraction and the word cloud/common-words features agree on what
    counts as "noise"."""
    with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return set(f.read().split())


def hourly_activity_full(df: pd.DataFrame) -> pd.Series:
    """Message count for every hour 0-23, 0-filled.

    Unlike `helper.hourly_activity`, which only returns hours that
    appear at least once (via `value_counts()`), this always returns
    24 entries — needed for anything that scans *every* hour, like
    Peak Activity and Sleep Schedule.
    """
    counts = df['hour'].value_counts()
    return counts.reindex(range(24), fill_value=0).sort_index()


# ---------------------------------------------------------------------------
# 1. Response Time Analysis
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _reply_pairs_cached(_df: pd.DataFrame, key: str) -> pd.DataFrame:
    real = _real_messages(_df).sort_values('date').reset_index(drop=True)
    if len(real) < 2:
        return pd.DataFrame(columns=['date', 'replier', 'replied_to', 'reply_minutes'])

    prev_user = real['user'].shift(1)
    prev_date = real['date'].shift(1)
    gap_minutes = (real['date'] - prev_date).dt.total_seconds() / 60.0

    is_reply = (
        (real['user'] != prev_user)
        & gap_minutes.notna()
        & (gap_minutes >= 0)
        & (gap_minutes <= config.REPLY_GAP_CAP_MINUTES)
    )

    out = pd.DataFrame({
        'date': real.loc[is_reply, 'date'],
        'replier': real.loc[is_reply, 'user'],
        'replied_to': prev_user[is_reply],
        'reply_minutes': gap_minutes[is_reply],
    })
    return out.reset_index(drop=True)


def _reply_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per "reply": a message sent by a different user than
    the one immediately before it, within `config.REPLY_GAP_CAP_MINUTES`
    of it. Longer gaps are treated as a new conversation rather than a
    slow reply, so an overnight silence doesn't blow out the average
    (see Sleep Schedule, which uses long silences deliberately instead).

    Args:
        df: Chat DataFrame — any subset (already user/date filtered is
            fine, since this only looks at consecutive rows).

    Returns:
        DataFrame with columns: date (the reply's timestamp), replier
        (who sent the reply), replied_to (who they replied to),
        reply_minutes (gap in minutes, always >= 0).

    Cached per (user, date) content: `response_time_stats`,
    `response_time_by_user`, `response_time_histogram_data`, and
    `response_time_trend` each call this independently, so caching it
    here means the sort + diff over every message runs once per unique
    `df`, not once per caller.
    """
    key = nlp_helper.content_key(df, columns=("user", "date"))
    return _reply_pairs_cached(df, key)


def response_time_stats(selected_user: str, df: pd.DataFrame) -> dict:
    """Headline response-time numbers for the KPI cards.

    Args:
        selected_user: "Overall" or a specific participant. When a
            specific user is chosen, only *their* replies are counted
            (so "fastest"/"slowest" collapse to that one user).
        df: Chat DataFrame.

    Returns:
        dict with keys: avg_minutes, median_minutes, fastest_user,
        fastest_minutes, slowest_user, slowest_minutes, reply_count.
        Falls back to "N/A" / 0 when there's not enough data.
    """
    pairs = _reply_pairs(df)
    if selected_user != 'Overall':
        pairs = pairs[pairs['replier'] == selected_user]

    if pairs.empty:
        return {
            'avg_minutes': 0.0, 'median_minutes': 0.0,
            'fastest_user': 'N/A', 'fastest_minutes': 0.0,
            'slowest_user': 'N/A', 'slowest_minutes': 0.0,
            'reply_count': 0,
        }

    by_user = pairs.groupby('replier')['reply_minutes'].mean().sort_values()

    return {
        'avg_minutes': round(float(pairs['reply_minutes'].mean()), 1),
        'median_minutes': round(float(pairs['reply_minutes'].median()), 1),
        'fastest_user': str(by_user.index[0]),
        'fastest_minutes': round(float(by_user.iloc[0]), 1),
        'slowest_user': str(by_user.index[-1]),
        'slowest_minutes': round(float(by_user.iloc[-1]), 1),
        'reply_count': int(len(pairs)),
    }


def response_time_by_user(df: pd.DataFrame) -> pd.Series:
    """Average reply time (minutes) per user, fastest first."""
    pairs = _reply_pairs(df)
    if pairs.empty:
        return pd.Series(dtype=float)
    return pairs.groupby('replier')['reply_minutes'].mean().round(1).sort_values()


def response_time_histogram_data(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Raw reply-time values (minutes), ready to hand to a histogram."""
    pairs = _reply_pairs(df)
    if selected_user != 'Overall':
        pairs = pairs[pairs['replier'] == selected_user]
    return pairs['reply_minutes']


def response_time_trend(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Daily average reply time — charts how response speed drifts over
    the course of the conversation."""
    pairs = _reply_pairs(df)
    if selected_user != 'Overall':
        pairs = pairs[pairs['replier'] == selected_user]
    if pairs.empty:
        return pd.DataFrame(columns=['only_date', 'reply_minutes'])

    trend = pairs.copy()
    trend['only_date'] = trend['date'].dt.date
    return trend.groupby('only_date')['reply_minutes'].mean().round(1).reset_index()


# ---------------------------------------------------------------------------
# 2. Conversation Starters
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _mark_conversation_starts_cached(_df: pd.DataFrame, key: str) -> pd.DataFrame:
    real = _real_messages(_df).sort_values('date').reset_index(drop=True)
    if real.empty:
        return real.assign(is_starter=pd.Series(dtype=bool))

    gap_minutes = real['date'].diff().dt.total_seconds() / 60.0
    real = real.copy()
    real['is_starter'] = gap_minutes.isna() | (gap_minutes > config.CONVERSATION_GAP_MINUTES)
    return real


def _mark_conversation_starts(df: pd.DataFrame) -> pd.DataFrame:
    """Flag every message that begins a new "conversation": the very
    first message, or one that follows a silence longer than
    `config.CONVERSATION_GAP_MINUTES`.

    Cached per (user, date) content: `conversation_starters` and
    `conversation_starters_by_period` each call this independently, so
    caching means the sort + diff runs once per unique `df`.
    """
    key = nlp_helper.content_key(df, columns=("user", "date"))
    return _mark_conversation_starts_cached(df, key)


def conversation_starters(df: pd.DataFrame) -> pd.Series:
    """Count of conversations started by each user, most first."""
    marked = _mark_conversation_starts(df)
    starters = marked[marked['is_starter']]
    if starters.empty:
        return pd.Series(dtype=int)
    return starters['user'].value_counts()


def conversation_starters_by_period(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """Conversations started per user, bucketed by day/week/month.

    Args:
        df: Chat DataFrame.
        freq: 'D' (daily), 'W' (weekly) or 'M' (monthly).

    Returns:
        A pivoted DataFrame — period (as the index) x user — of
        conversations started, 0-filled and sorted chronologically.
    """
    marked = _mark_conversation_starts(df)
    starters = marked[marked['is_starter']].copy()
    if starters.empty:
        return pd.DataFrame()

    starters['period'] = pd.to_datetime(starters['date']).dt.to_period(freq).dt.to_timestamp()
    pivot = starters.pivot_table(
        index='period', columns='user', values='date', aggfunc='count'
    ).fillna(0).astype(int)
    return pivot.sort_index()


# ---------------------------------------------------------------------------
# 3. Average Message Length
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _message_length_stats_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    text_df = _non_media(_real_messages(_filter_user(_df, selected_user))).copy()
    if text_df.empty:
        return pd.DataFrame(columns=['user', 'avg_words', 'avg_characters', 'avg_emoji', 'avg_links'])

    text_df['word_count'] = text_df['message'].str.split().str.len()
    text_df['char_count'] = text_df['message'].str.len()
    text_df['emoji_count'] = text_df['message'].apply(_count_emojis)
    text_df['link_count'] = text_df['message'].apply(_count_links)

    summary = text_df.groupby('user').agg(
        avg_words=('word_count', 'mean'),
        avg_characters=('char_count', 'mean'),
        avg_emoji=('emoji_count', 'mean'),
        avg_links=('link_count', 'mean'),
    ).round(2).reset_index()

    return summary.sort_values('avg_words', ascending=False)


def message_length_stats(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Average words / characters / emoji / links per message, per user.

    Media placeholder messages are excluded since they carry no real
    text content. When `selected_user` isn't "Overall", the result has
    a single row for that user.

    Cached per (message content, selected_user): the emoji/link counts
    are per-message `.apply()` scans, the same cost class as the word
    cloud in helper.py.
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _message_length_stats_cached(df, key, selected_user)


# ---------------------------------------------------------------------------
# 4. Longest Message
# ---------------------------------------------------------------------------
def longest_message_details(selected_user: str, df: pd.DataFrame) -> dict:
    """Details of the single longest (by word count) text message.

    Returns:
        dict with keys: message, sender, date, word_count, char_count.
        `message` is truncated to 500 characters for safe display.
    """
    text_df = _non_media(_real_messages(_filter_user(df, selected_user))).copy()
    if text_df.empty:
        return {'message': '', 'sender': 'N/A', 'date': None, 'word_count': 0, 'char_count': 0}

    text_df['word_count'] = text_df['message'].str.split().str.len()
    row = text_df.loc[text_df['word_count'].idxmax()]

    return {
        'message': row['message'][:5000],
        'sender': row['user'],
        'date': row['date'],
        'word_count': int(row['word_count']),
        'char_count': int(len(row['message'])),
    }


# ---------------------------------------------------------------------------
# 5. Weekend vs Weekday
# ---------------------------------------------------------------------------
_WEEKEND_DAYS = {'Saturday', 'Sunday'}


@st.cache_data(show_spinner=False)
def _weekend_vs_weekday_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    scoped = _real_messages(_filter_user(_df, selected_user)).copy()
    if scoped.empty:
        return pd.DataFrame(
            columns=['period', 'messages', 'words', 'media', 'links', 'emoji', 'active_days', 'avg_messages_per_day']
        )

    scoped['bucket'] = scoped['day_name'].apply(lambda d: 'Weekend' if d in _WEEKEND_DAYS else 'Weekday')
    scoped['word_count'] = scoped['message'].str.split().str.len()
    scoped['is_media'] = scoped['message'] == config.MEDIA_PLACEHOLDER
    scoped['link_count'] = scoped['message'].apply(_count_links)
    scoped['emoji_count'] = scoped['message'].apply(_count_emojis)

    grouped = scoped.groupby('bucket').agg(
        messages=('message', 'count'),
        words=('word_count', 'sum'),
        media=('is_media', 'sum'),
        links=('link_count', 'sum'),
        emoji=('emoji_count', 'sum'),
        active_days=('only_date', 'nunique'),
    ).reset_index().rename(columns={'bucket': 'period'})

    grouped['avg_messages_per_day'] = (grouped['messages'] / grouped['active_days'].replace(0, 1)).round(2)
    return grouped


def weekend_vs_weekday(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compare activity on weekends vs weekdays: messages, words, media,
    links, and emoji — each as a total, plus a per-active-day average.

    Cached per (message content, selected_user) — see
    `message_length_stats`.
    """
    key = nlp_helper.content_key(df, columns=("user", "message", "day_name", "only_date"))
    return _weekend_vs_weekday_cached(df, key, selected_user)


# ---------------------------------------------------------------------------
# 6. Peak Activity
# ---------------------------------------------------------------------------
def peak_hours(selected_user: str, df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """The `top_n` busiest hours-of-day (0-23), most active first."""
    scoped = _filter_user(df, selected_user)
    return scoped['hour'].value_counts().sort_values(ascending=False).head(top_n)


def quiet_hours(selected_user: str, df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """The `top_n` quietest hours-of-day (0-23), least active first.

    Uses `hourly_activity_full` (0-filled) rather than `value_counts()`
    so genuinely silent hours are counted as 0, not simply omitted.
    """
    scoped = _filter_user(df, selected_user)
    full = hourly_activity_full(scoped)
    return full.sort_values().head(top_n)


# ---------------------------------------------------------------------------
# 7. User Comparison
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _compare_users_cached(_df: pd.DataFrame, key: str, user1: str, user2: str) -> pd.DataFrame:
    reply_series = response_time_by_user(_df)

    def _stats_for(user: str) -> dict:
        scoped = _real_messages(_df[_df['user'] == user])
        text = _non_media(scoped)
        word_counts = text['message'].str.split().str.len() if not text.empty else pd.Series(dtype=float)
        return {
            'Messages': int(scoped.shape[0]),
            'Words': int(word_counts.sum()) if not text.empty else 0,
            'Media Shared': int((scoped['message'] == config.MEDIA_PLACEHOLDER).sum()),
            'Links Shared': int(text['message'].apply(_count_links).sum()) if not text.empty else 0,
            'Emoji Used': int(text['message'].apply(_count_emojis).sum()) if not text.empty else 0,
            'Avg Words / Message': round(float(word_counts.mean()), 2) if not text.empty else 0.0,
            'Avg Reply Time (min)': float(reply_series.get(user, 0.0)),
        }

    stats1, stats2 = _stats_for(user1), _stats_for(user2)
    return pd.DataFrame({user1: stats1, user2: stats2})


def compare_users(df: pd.DataFrame, user1: str, user2: str) -> pd.DataFrame:
    """Side-by-side KPI comparison of two participants.

    Args:
        df: Chat DataFrame (e.g. the sidebar-filtered dataset).
        user1: First participant's name.
        user2: Second participant's name.

    Returns:
        A DataFrame with one row per metric and one column per user,
        covering messages, words, media, links, emoji, avg words per
        message, and average reply time (minutes).

    Cached per (message content, user1, user2): the emoji/link counts
    are per-message `.apply()` scans over each user's messages.
    """
    key = nlp_helper.content_key(df, columns=("user", "message", "date"))
    return _compare_users_cached(df, key, user1, user2)


# ---------------------------------------------------------------------------
# 8. Emoji Timeline
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _emoji_timeline_monthly_cached(_df: pd.DataFrame, key: str, selected_user: str) -> pd.DataFrame:
    scoped = _filter_user(_df, selected_user).copy()
    if scoped.empty:
        return pd.DataFrame(columns=['time', 'emoji_count'])

    scoped['emoji_count'] = scoped['message'].apply(_count_emojis)
    monthly = scoped.groupby(['year', 'month_num', 'month'])['emoji_count'].sum().reset_index()
    monthly = monthly.sort_values(['year', 'month_num'])
    monthly['time'] = monthly['month'] + " " + monthly['year'].astype(str)
    return monthly[['time', 'emoji_count']]


def emoji_timeline_monthly(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Total emoji count per calendar month (e.g. 'March 2024').

    Cached per (message content, selected_user) — the emoji scan is
    per-message and character-by-character.
    """
    key = nlp_helper.content_key(df, columns=("user", "message", "year", "month_num", "month"))
    return _emoji_timeline_monthly_cached(df, key, selected_user)


@st.cache_data(show_spinner=False)
def _emoji_timeline_by_user_cached(_df: pd.DataFrame, key: str, top_n_users: int) -> pd.DataFrame:
    scoped = _real_messages(_df).copy()
    if scoped.empty:
        return pd.DataFrame()

    scoped['emoji_count'] = scoped['message'].apply(_count_emojis)
    top_users = (
        scoped.groupby('user')['emoji_count'].sum().sort_values(ascending=False).head(top_n_users).index
    )
    scoped = scoped[scoped['user'].isin(top_users)].copy()
    if scoped.empty:
        return pd.DataFrame()

    scoped['time'] = scoped['month'] + " " + scoped['year'].astype(str)
    scoped['sort_key'] = pd.to_datetime(
        scoped['year'].astype(str) + '-' + scoped['month_num'].astype(str) + '-01'
    )

    pivot = scoped.pivot_table(
        index=['sort_key', 'time'], columns='user', values='emoji_count', aggfunc='sum'
    ).fillna(0)
    pivot = pivot.reset_index().sort_values('sort_key').drop(columns='sort_key').set_index('time')
    return pivot


def emoji_timeline_by_user(df: pd.DataFrame, top_n_users: int = 6) -> pd.DataFrame:
    """Monthly emoji count per user, limited to the `top_n_users` who use
    the most emoji overall (keeps the chart from getting too crowded).

    Returns:
        A pivoted DataFrame: time (month) x user, emoji counts, 0-filled.

    Cached per (message content, top_n_users) — see
    `emoji_timeline_monthly`.
    """
    key = nlp_helper.content_key(df, columns=("user", "message", "year", "month_num", "month"))
    return _emoji_timeline_by_user_cached(df, key, top_n_users)


# ---------------------------------------------------------------------------
# 9. Most Used Phrases (n-grams)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _top_ngrams_cached(_df: pd.DataFrame, key: str, selected_user: str, n: int, top_n: int) -> pd.DataFrame:
    stop_words = _load_stopwords()
    text_df = _non_media(_real_messages(_filter_user(_df, selected_user)))

    phrase_counter: Counter = Counter()
    for message in text_df['message']:
        tokens = [w for w in message.lower().split() if w not in stop_words and w.isalnum()]
        for i in range(len(tokens) - n + 1):
            phrase_counter[" ".join(tokens[i:i + n])] += 1

    most_common = phrase_counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=['phrase', 'count'])


def top_ngrams(selected_user: str, df: pd.DataFrame, n: int = 2, top_n: int = 20) -> pd.DataFrame:
    """Most frequent word n-grams (n=2 for bigrams, n=3 for trigrams).

    Stop words are stripped before phrases are built and media/system
    messages are dropped, matching `helper.most_common_words`'s cleaning
    approach so the two features stay consistent with each other.

    Cached per (message content, selected_user, n, top_n): tokenizing
    every message to build n-grams is one of the more expensive loops
    in this module.
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _top_ngrams_cached(df, key, selected_user, n, top_n)


# ---------------------------------------------------------------------------
# 10. Shared Domains
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _shared_domains_cached(_df: pd.DataFrame, key: str, selected_user: str, top_n: int) -> pd.DataFrame:
    text_df = _non_media(_real_messages(_filter_user(_df, selected_user)))

    domain_counter: Counter = Counter()
    for message in text_df['message']:
        for url in extract.find_urls(message):
            host = re.sub(r'^https?://', '', url).split('/')[0].lower()
            host = re.sub(r'^www\.', '', host)
            label = config.DOMAIN_LABELS.get(host, host)
            domain_counter[label] += 1

    most_common = domain_counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=['domain', 'count'])


def shared_domains(selected_user: str, df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most frequently shared link domains (GitHub, LinkedIn, YouTube, ...).

    Recognized hosts are relabeled to a friendly name via
    `config.DOMAIN_LABELS` (e.g. "youtu.be" -> "YouTube"); anything else
    is still counted, under its bare hostname (e.g. "example.com").

    Cached per (message content, selected_user, top_n): URL-extraction
    runs a regex pass over every message.
    """
    key = nlp_helper.content_key(df, columns=("user", "message"))
    return _shared_domains_cached(df, key, selected_user, top_n)


# ---------------------------------------------------------------------------
# 11. Calendar Heatmap
# ---------------------------------------------------------------------------
def calendar_heatmap_data(selected_user: str, df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """Daily message counts for a GitHub-style calendar heatmap.

    Args:
        selected_user: "Overall" or a specific participant.
        df: Chat DataFrame.
        year: Calendar year to show. Defaults to the most recent year
            present in the (filtered) data.

    Returns:
        DataFrame with one row per day of that year — only_date, count,
        week_of_year, day_of_week (0=Monday) — including days with 0
        messages, so the calendar renders without gaps.
    """
    scoped = _filter_user(df, selected_user)
    if scoped.empty or scoped['year'].dropna().empty:
        return pd.DataFrame(columns=['only_date', 'count', 'week_of_year', 'day_of_week'])

    if year is None:
        year = int(scoped['year'].max())

    daily_counts = scoped[scoped['year'] == year].groupby('only_date').size()

    full_year = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
    calendar_df = pd.DataFrame({'only_date': full_year.date})
    calendar_df['count'] = calendar_df['only_date'].map(daily_counts).fillna(0).astype(int)
    calendar_df['week_of_year'] = pd.to_datetime(calendar_df['only_date']).dt.isocalendar().week.astype(int)
    calendar_df['day_of_week'] = pd.to_datetime(calendar_df['only_date']).dt.weekday  # 0=Monday

    return calendar_df


# ---------------------------------------------------------------------------
# 12. Sleep Schedule Estimation
# ---------------------------------------------------------------------------
def estimate_sleep_schedule(selected_user: str, df: pd.DataFrame) -> dict:
    """Rough estimate of sleeping hours, based on the quietest contiguous
    block of hours in the user's 24-hour activity profile.

    This is a heuristic, not a measurement: it assumes someone is least
    likely to be messaging while asleep and looks for the longest run
    of `config.SLEEP_WINDOW_HOURS` consecutive hours (allowed to wrap
    past midnight) with the fewest total messages.

    Returns:
        dict with keys: sleep_start (hour 0-23 or None), wake_up (hour
        0-23 or None), window_hours, quiet_message_count.
    """
    scoped = _filter_user(df, selected_user)
    hourly = hourly_activity_full(scoped)
    window = config.SLEEP_WINDOW_HOURS

    if hourly.sum() == 0:
        return {'sleep_start': None, 'wake_up': None, 'window_hours': window, 'quiet_message_count': 0}

    best_start, best_count = 0, float('inf')
    for start in range(24):
        hours = [(start + i) % 24 for i in range(window)]
        count = hourly.loc[hours].sum()
        if count < best_count:
            best_start, best_count = start, count

    wake_up = (best_start + window) % 24
    return {
        'sleep_start': best_start,
        'wake_up': wake_up,
        'window_hours': window,
        'quiet_message_count': int(best_count),
    }


# ---------------------------------------------------------------------------
# 13. Weekly Report
# ---------------------------------------------------------------------------
def generate_weekly_report(selected_user: str, df: pd.DataFrame) -> str:
    """A short, human-readable analytics summary for the most recent
    7-day window present in `df` (or the whole range, if shorter).

    Pulls from several of the functions above, so it doubles as a
    "highlights" digest rather than a raw KPI dump.
    """
    scoped = _filter_user(df, selected_user)
    if scoped.empty:
        return "No messages available to summarize."

    latest_date = scoped['date'].max()
    window_start = latest_date - pd.Timedelta(days=7)
    week_df = scoped[scoped['date'] > window_start]
    if week_df.empty:
        week_df = scoped

    text_week = _non_media(_real_messages(week_df))
    total_messages = int(week_df.shape[0])
    total_words = int(text_week['message'].str.split().str.len().sum()) if not text_week.empty else 0

    busiest_day = week_df['day_name'].value_counts().idxmax() if not week_df.empty else "N/A"
    busiest_hour = week_df['hour'].value_counts().idxmax() if not week_df.empty else "N/A"

    starters = conversation_starters(week_df)
    top_starter = starters.index[0] if not starters.empty else "N/A"

    resp_stats = response_time_stats('Overall', week_df)

    lines = [
        f"**Weekly Report** ({window_start.date()} to {latest_date.date()})",
        f"- Total messages: **{total_messages:,}**, total words: **{total_words:,}**",
        f"- Busiest day: **{busiest_day}**, busiest hour: **{busiest_hour}:00**",
        f"- Most conversations started by: **{top_starter}**",
    ]
    if resp_stats['reply_count'] > 0:
        lines.append(
            f"- Average reply time: **{resp_stats['avg_minutes']} min** "
            f"(fastest: {resp_stats['fastest_user']} at {resp_stats['fastest_minutes']} min)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 14. Message Frequency
# ---------------------------------------------------------------------------
def message_frequency(selected_user: str, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """Message counts resampled to a given cadence.

    Args:
        selected_user: "Overall" or a specific participant.
        df: Chat DataFrame.
        freq: 'D' (daily), 'W' (weekly) or 'M' (monthly).

    Returns:
        DataFrame with columns: period (Timestamp), messages (count).
    """
    scoped = _filter_user(df, selected_user).copy()
    if scoped.empty:
        return pd.DataFrame(columns=['period', 'messages'])

    scoped['period'] = pd.to_datetime(scoped['date']).dt.to_period(freq).dt.to_timestamp()
    result = scoped.groupby('period').size().reset_index(name='messages')
    return result.sort_values('period')


# ---------------------------------------------------------------------------
# 15. Engagement Score
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _engagement_scores_cached(_df: pd.DataFrame, key: str) -> pd.DataFrame:
    real = _real_messages(_df)
    if real.empty:
        return pd.DataFrame(columns=['user', 'engagement_score'])

    text = _non_media(real).copy()
    text['word_count'] = text['message'].str.split().str.len()
    text['emoji_count'] = text['message'].apply(_count_emojis)
    text['link_count'] = text['message'].apply(_count_links)

    total_active_days = real['only_date'].nunique() or 1

    per_user = pd.DataFrame({
        'messages': real.groupby('user').size(),
        'media': real[real['message'] == config.MEDIA_PLACEHOLDER].groupby('user').size(),
        'words': text.groupby('user')['word_count'].sum(),
        'emoji': text.groupby('user')['emoji_count'].sum(),
        'links': text.groupby('user')['link_count'].sum(),
        'active_days': real.groupby('user')['only_date'].nunique(),
    }).fillna(0)

    reply_speed = response_time_by_user(_df)
    # Faster replies -> higher score, so invert the raw minutes. Users
    # with no timed replies get the worst (not a free-pass) score.
    if not reply_speed.empty:
        max_wait = reply_speed.max()
        per_user['response_speed'] = per_user.index.map(lambda u: max_wait - reply_speed.get(u, max_wait))
    else:
        per_user['response_speed'] = 0.0

    per_user['consistency'] = per_user['active_days'] / total_active_days

    def _normalize(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / span

    weights = config.ENGAGEMENT_WEIGHTS
    score = sum(_normalize(per_user[col]) * weight for col, weight in weights.items())

    result = pd.DataFrame({
        'user': score.index,
        'engagement_score': (score * 100).round(1).values,
    }).sort_values('engagement_score', ascending=False).reset_index(drop=True)

    return result


def engagement_scores(df: pd.DataFrame) -> pd.DataFrame:
    """A composite 0-100 engagement score per user, blending several
    signals so no single metric (e.g. raw message count) dominates.

    Components (weights configured in `config.ENGAGEMENT_WEIGHTS`):
        - messages: share of total message volume
        - words: share of total words written
        - media: share of media shared
        - emoji: share of emoji used
        - links: share of links shared
        - response_speed: how quickly they reply (faster = higher score)
        - consistency: distinct active days as a share of the chat's
          total active days

    Each component is min-max normalized across users to a 0-1 range,
    weighted, summed, and rescaled to 0-100. This is a *relative*
    ranking within this specific chat, not an absolute or universal
    engagement metric.

    Cached per (user, message, date) content: this is the most
    expensive single function in the module — two `.apply()` scans
    over every real message, plus a call into `response_time_by_user`
    (itself now cached via `_reply_pairs`).
    """
    key = nlp_helper.content_key(df, columns=("user", "message", "date", "only_date"))
    return _engagement_scores_cached(df, key)
