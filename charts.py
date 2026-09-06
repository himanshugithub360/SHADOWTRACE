
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:

    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Segoe UI, Helvetica Neue, sans-serif", size=13, color="#e5e5ec"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#1b1b26", bordercolor="rgba(124,92,255,0.4)", font_size=13, font_color="#f5f5f7"),
        legend=dict(font=dict(color="#c8c8d4")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


def monthly_timeline_chart(timeline_df: pd.DataFrame) -> go.Figure:
    """Line chart of message volume per calendar month."""
    fig = px.line(
        timeline_df, x='time', y='message', markers=True,
        color_discrete_sequence=[config.PRIMARY_COLOR],
        labels={'time': 'Month', 'message': 'Messages'},
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Monthly Timeline")


def daily_timeline_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Line chart of message volume per calendar date."""
    fig = px.line(
        daily_df, x='only_date', y='message',
        color_discrete_sequence=[config.SECONDARY_COLOR],
        labels={'only_date': 'Date', 'message': 'Messages'},
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Daily Timeline")


def week_activity_chart(busy_day: pd.Series) -> go.Figure:
    """Bar chart of message volume per day-of-week."""
    ordered = busy_day.reindex([d for d in config.DAY_ORDER if d in busy_day.index])
    fig = px.bar(
        x=ordered.index, y=ordered.values,
        color=ordered.index, color_discrete_sequence=config.COLOR_SEQUENCE,
        labels={'x': 'Day', 'y': 'Messages'},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Most Busy Day")


def month_activity_chart(busy_month: pd.Series) -> go.Figure:
    """Bar chart of message volume per month."""
    ordered = busy_month.reindex([m for m in config.MONTH_ORDER if m in busy_month.index])
    fig = px.bar(
        x=ordered.index, y=ordered.values,
        color=ordered.index, color_discrete_sequence=config.COLOR_SEQUENCE,
        labels={'x': 'Month', 'y': 'Messages'},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Most Busy Month")


def hourly_activity_chart(hourly: pd.Series) -> go.Figure:
    """Bar chart of message volume per hour-of-day (0-23)."""
    fig = px.bar(
        x=hourly.index, y=hourly.values,
        color_discrete_sequence=[config.ACCENT_COLOR],
        labels={'x': 'Hour of Day', 'y': 'Messages'},
    )
    fig.update_xaxes(dtick=1)
    return _apply_theme(fig, "Hourly Analysis")


def activity_heatmap_chart(user_heatmap: pd.DataFrame) -> go.Figure:
    """Heatmap of message volume by day-of-week x hour-period."""
    ordered_index = [d for d in config.DAY_ORDER if d in user_heatmap.index]
    heatmap_data = user_heatmap.reindex(ordered_index)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, "#1b1b26"], [1, config.SECONDARY_COLOR]],
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Messages: %{z}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Weekly Activity Heatmap")


def most_busy_users_chart(x: pd.Series) -> go.Figure:
    """Bar chart of the top-5 most active group members."""
    fig = px.bar(
        x=x.index, y=x.values,
        color=x.index, color_discrete_sequence=config.COLOR_SEQUENCE,
        labels={'x': 'User', 'y': 'Messages'},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Most Busy Users")


def most_common_words_chart(most_common_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the 20 most frequently used words."""
    # Reverse so the most common word appears at the top of the chart.
    df_sorted = most_common_df.iloc[::-1]
    fig = px.bar(
        df_sorted, x=1, y=0, orientation='h',
        color_discrete_sequence=[config.PRIMARY_COLOR],
        labels={'0': 'Word', '1': 'Count'},
    )
    return _apply_theme(fig, "Most Common Words")


def emoji_pie_chart(emoji_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Pie chart of the most-used emoji."""
    top = emoji_df.head(top_n)
    fig = px.pie(
        top, values=1, names=0, hole=0.35,
        color_discrete_sequence=config.COLOR_SEQUENCE,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return _apply_theme(fig, "Emoji Distribution")


def ranking_bar_chart(series: pd.Series, title: str, value_label: str) -> go.Figure:
    """Generic horizontal ranking bar chart for a Series indexed by user
    (or any other category) with a single numeric value per entry.
    Reused by several features below instead of writing a near-identical
    one-off chart function for every simple "rank X by Y" view.
    """
    ordered = series.sort_values()
    fig = px.bar(
        x=ordered.values, y=ordered.index.astype(str), orientation='h',
        color=ordered.index.astype(str), color_discrete_sequence=config.COLOR_SEQUENCE,
        labels={'x': value_label, 'y': ''},
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, title)


def response_time_histogram_chart(reply_minutes: pd.Series) -> go.Figure:
    """Histogram of individual reply times (minutes)."""
    fig = px.histogram(
        x=reply_minutes, nbins=30,
        color_discrete_sequence=[config.PRIMARY_COLOR],
        labels={'x': 'Reply Time (minutes)'},
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, "Response Time Distribution")


def response_time_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Line chart of average daily reply time, to see if replies are
    getting faster or slower over the course of the chat."""
    fig = px.line(
        trend_df, x='only_date', y='reply_minutes', markers=True,
        color_discrete_sequence=[config.SECONDARY_COLOR],
        labels={'only_date': 'Date', 'reply_minutes': 'Avg Reply Time (min)'},
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Response Time Trend")


def conversation_starters_chart(starters: pd.Series) -> go.Figure:
    """Bar chart of how many conversations each user started."""
    return ranking_bar_chart(starters, "Conversation Starters", "Conversations Started")


def conversation_starters_trend_chart(pivot_df: pd.DataFrame) -> go.Figure:
    """Multi-line chart of conversations started per user, over time."""
    fig = go.Figure()
    for user in pivot_df.columns:
        fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[user], mode='lines+markers', name=user))
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Conversation Starters Over Time")


def message_length_chart(length_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing avg words/characters/emoji/links per
    message across users."""
    melted = length_df.melt(
        id_vars='user', value_vars=['avg_words', 'avg_characters', 'avg_emoji', 'avg_links'],
        var_name='metric', value_name='value',
    )
    fig = px.bar(
        melted, x='user', y='value', color='metric', barmode='group',
        color_discrete_sequence=config.COLOR_SEQUENCE,
        labels={'user': 'User', 'value': 'Average per Message'},
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Average Message Length")


def weekend_weekday_chart(weekend_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of messages/words/media/links/emoji, Weekend
    vs Weekday."""
    melted = weekend_df.melt(
        id_vars='period', value_vars=['messages', 'words', 'media', 'links', 'emoji'],
        var_name='metric', value_name='value',
    )
    fig = px.bar(
        melted, x='metric', y='value', color='period', barmode='group',
        color_discrete_sequence=[config.PRIMARY_COLOR, config.ACCENT_COLOR],
        labels={'metric': 'Metric', 'value': 'Total'},
    )
    return _apply_theme(fig, "Weekend vs Weekday")


def peak_activity_chart(hourly_full: pd.Series, top_hours: pd.Series, quiet_hours_s: pd.Series) -> go.Figure:
    """24-hour bar chart with the busiest hours highlighted in the
    primary color and the quietest hours greyed out."""
    colors = []
    for hour in hourly_full.index:
        if hour in top_hours.index:
            colors.append(config.PRIMARY_COLOR)
        elif hour in quiet_hours_s.index:
            colors.append("#4b5563")
        else:
            colors.append(config.ACCENT_COLOR)

    fig = go.Figure(go.Bar(x=hourly_full.index, y=hourly_full.values, marker_color=colors))
    fig.update_xaxes(dtick=1, title="Hour of Day")
    fig.update_yaxes(title="Messages")
    return _apply_theme(fig, "Peak vs Quiet Hours")


def user_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing two users across KPIs.

    Each metric row is min-max normalized across the two users (0-1)
    purely for the chart's shared radial scale — the underlying real
    numbers belong in the table shown alongside this chart, not read
    off this axis.
    """
    normalized = comparison_df.copy()
    for metric in normalized.index:
        row = normalized.loc[metric]
        span = row.max() - row.min()
        normalized.loc[metric] = 0.5 if span == 0 else (row - row.min()) / span

    metrics = normalized.index.tolist()
    fig = go.Figure()
    for user in normalized.columns:
        values = normalized[user].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=metrics + metrics[:1],
            fill='toself', name=user,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return _apply_theme(fig, "User Comparison")


def emoji_timeline_chart(timeline_df: pd.DataFrame) -> go.Figure:
    """Line chart of total emoji usage per month."""
    fig = px.line(
        timeline_df, x='time', y='emoji_count', markers=True,
        color_discrete_sequence=[config.ACCENT_COLOR],
        labels={'time': 'Month', 'emoji_count': 'Emoji Used'},
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Emoji Usage Over Time")


def emoji_timeline_by_user_chart(pivot_df: pd.DataFrame) -> go.Figure:
    """Multi-line chart of monthly emoji usage, one line per top user."""
    fig = go.Figure()
    for user in pivot_df.columns:
        fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[user], mode='lines+markers', name=user))
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Emoji Usage by User")


def ngram_chart(ngram_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the most frequent bigrams/trigrams."""
    df_sorted = ngram_df.iloc[::-1]
    fig = px.bar(
        df_sorted, x='count', y='phrase', orientation='h',
        color_discrete_sequence=[config.SECONDARY_COLOR],
    )
    return _apply_theme(fig, "Most Used Phrases")


def domains_chart(domains_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the most frequently shared link domains."""
    fig = px.bar(
        domains_df.iloc[::-1], x='count', y='domain', orientation='h',
        color_discrete_sequence=[config.PRIMARY_COLOR],
    )
    return _apply_theme(fig, "Shared Domains")


def calendar_heatmap_chart(calendar_df: pd.DataFrame, year: int) -> go.Figure:
    """GitHub-style calendar heatmap: week-of-year on the x-axis,
    day-of-week on the y-axis, message count as color intensity."""
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(
        x=calendar_df['week_of_year'], y=calendar_df['day_of_week'],
        z=calendar_df['count'],
        colorscale=[[0, "#1b1b26"], [1, config.PRIMARY_COLOR]],
        hovertemplate="Date: %{customdata}<br>Messages: %{z}<extra></extra>",
        customdata=calendar_df['only_date'].astype(str),
        showscale=True,
    ))
    fig.update_yaxes(tickmode='array', tickvals=list(range(7)), ticktext=day_labels, autorange='reversed')
    fig.update_xaxes(title="Week of Year")
    return _apply_theme(fig, f"Activity Calendar — {year}")


def sleep_schedule_chart(hourly_full: pd.Series, sleep_start: int, wake_up: int) -> go.Figure:
    """24-hour bar chart with the estimated sleep window greyed out."""
    colors = []
    for hour in hourly_full.index:
        in_sleep_window = (
            (sleep_start <= hour < wake_up) if sleep_start < wake_up
            else (hour >= sleep_start or hour < wake_up)
        )
        colors.append("#4b5563" if in_sleep_window else config.ACCENT_COLOR)

    fig = go.Figure(go.Bar(x=hourly_full.index, y=hourly_full.values, marker_color=colors))
    fig.update_xaxes(dtick=1, title="Hour of Day")
    fig.update_yaxes(title="Messages")
    return _apply_theme(fig, f"Estimated Sleep Window: {sleep_start:02d}:00 – {wake_up:02d}:00")


def message_frequency_chart(freq_df: pd.DataFrame, freq_label: str) -> go.Figure:
    """Line chart of message counts resampled to a chosen cadence."""
    fig = px.line(
        freq_df, x='period', y='messages', markers=True,
        color_discrete_sequence=[config.PRIMARY_COLOR],
        labels={'period': freq_label, 'messages': 'Messages'},
    )
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, f"Message Frequency ({freq_label})")


def engagement_score_chart(scores_df: pd.DataFrame) -> go.Figure:
    """Horizontal ranking bar chart of each user's engagement score."""
    series = scores_df.set_index('user')['engagement_score']
    return ranking_bar_chart(series, "Engagement Score", "Score (0-100)")



def distribution_pie_chart(dist_df: pd.DataFrame, names_col: str, values_col: str, title: str) -> go.Figure:
    """Generic pie chart for any label/count distribution — reused by
    Sentiment, Emotion, Intent, Toxicity, and Language breakdowns instead
    of writing a near-identical pie chart for each one."""
    fig = px.pie(
        dist_df, values=values_col, names=names_col, hole=0.35,
        color_discrete_sequence=config.COLOR_SEQUENCE,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return _apply_theme(fig, title)


def sentiment_timeline_chart(timeline_df: pd.DataFrame) -> go.Figure:
    """Line chart of average sentiment compound score over time, with a
    zero reference line separating net-positive from net-negative periods."""
    fig = px.line(
        timeline_df, x='period', y='avg_compound', markers=True,
        color_discrete_sequence=[config.SECONDARY_COLOR],
        labels={'period': 'Period', 'avg_compound': 'Avg Sentiment (-1 to 1)'},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_xaxes(tickangle=-45)
    return _apply_theme(fig, "Sentiment Over Time")


def grouped_stack_chart(cross_df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    """Stacked bar chart from a crosstab DataFrame (index = category,
    columns = sub-category) — reused by Emotion-by-User and Intent-by-User."""
    fig = go.Figure()
    for col in cross_df.columns:
        fig.add_trace(go.Bar(name=str(col), x=cross_df.index.astype(str), y=cross_df[col]))
    fig.update_layout(barmode='stack')
    fig.update_xaxes(tickangle=-45, title=cross_df.index.name or "")
    fig.update_yaxes(title=y_label)
    return _apply_theme(fig, title)


def keyword_bar_chart(keyword_df: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal bar chart for a (keyword, score) DataFrame — shared by
    TF-IDF, YAKE, and KeyBERT keyword extraction results."""
    df_sorted = keyword_df.sort_values('score').tail(20)
    fig = px.bar(
        df_sorted, x='score', y='keyword', orientation='h',
        color_discrete_sequence=[config.PRIMARY_COLOR],
    )
    return _apply_theme(fig, title)


def topic_size_chart(topics_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of topic sizes (message count per topic),
    with each topic's top words available on hover."""
    fig = px.bar(
        topics_df, x='size', y=topics_df['topic'].astype(str), orientation='h',
        color=topics_df['topic'].astype(str), color_discrete_sequence=config.COLOR_SEQUENCE,
        hover_data={'top_words': True},
        labels={'x': 'Messages', 'y': 'Topic'},
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, "Discussion Topics")
