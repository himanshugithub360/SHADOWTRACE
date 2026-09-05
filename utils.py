"""
utils.py
========
Generic, reusable helpers that don't belong to a single "topic" like
chat statistics (helper.py) or charts (charts.py). This includes:

- Sidebar filter functions (year, month, week, day, date range, hour)
- KPI / statistics-card calculations
- Search (keyword, user, date, regex)
- Export helpers (CSV, Excel, JSON)
- A small Streamlit setup helper (custom CSS injection)

Keeping these separate from helper.py avoids helper.py becoming a giant
"do everything" file as the project grows.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Optional
from xml.sax.saxutils import escape as _xml_escape

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config


# ---------------------------------------------------------------------------
# Streamlit setup helpers
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject the shared dashboard CSS (see config.CUSTOM_CSS).

    Call this once near the top of app.py, after `st.set_page_config`.
    """
    st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def get_filter_options(df: pd.DataFrame) -> dict:
    """Collect the distinct values available for each sidebar filter.

    Args:
        df: The full, preprocessed chat DataFrame.

    Returns:
        A dict of sorted option lists keyed by filter name, ready to be
        passed straight into Streamlit multiselect/select widgets.
    """
    real_messages = df[df['user'] != config.GROUP_NOTIFICATION_USER]

    return {
        "years": sorted(df['year'].dropna().unique().tolist()),
        "months": [m for m in config.MONTH_ORDER if m in df['month'].unique()],
        "weeks": sorted(df['week_num'].dropna().unique().tolist()),
        "days": [d for d in config.DAY_ORDER if d in df['day_name'].unique()],
        "users": sorted(real_messages['user'].unique().tolist()),
        "min_date": df['only_date'].min(),
        "max_date": df['only_date'].max(),
    }


def apply_filters(
    df: pd.DataFrame,
    years: Optional[list] = None,
    months: Optional[list] = None,
    weeks: Optional[list] = None,
    days: Optional[list] = None,
    date_range: Optional[tuple] = None,
    hour_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """Apply the "Advanced Filters" sidebar selections to a DataFrame.

    Every argument is optional; an empty/None value means "no filtering
    on that dimension". This lets app.py call this function once with
    whatever the user picked, instead of chaining a dozen if-statements.

    Args:
        df: DataFrame to filter (already narrowed to the selected user,
            if applicable).
        years: List of years to keep.
        months: List of month names to keep.
        weeks: List of ISO week numbers to keep.
        days: List of day names to keep.
        date_range: (start_date, end_date) tuple (inclusive).
        hour_range: (start_hour, end_hour) tuple (inclusive, 0-23).

    Returns:
        The filtered DataFrame. Note: this is NOT guaranteed to be a
        copy when no filter narrows the data (years/months/weeks/days/
        date_range/hour_range are all empty) — in that case the input
        `df` is returned as-is, since boolean-mask indexing below
        already produces a fresh DataFrame the moment any filter is
        applied, making an unconditional upfront `.copy()` wasted work.
        Callers should treat the return value as read-only, same as
        they already must with the original `df`.
    """
    filtered = df

    if years:
        filtered = filtered[filtered['year'].isin(years)]
    if months:
        filtered = filtered[filtered['month'].isin(months)]
    if weeks:
        filtered = filtered[filtered['week_num'].isin(weeks)]
    if days:
        filtered = filtered[filtered['day_name'].isin(days)]
    if date_range and len(date_range) == 2 and all(date_range):
        start, end = date_range
        filtered = filtered[
            (filtered['only_date'] >= start) & (filtered['only_date'] <= end)
        ]
    if hour_range:
        start_hour, end_hour = hour_range
        filtered = filtered[
            (filtered['hour'] >= start_hour) & (filtered['hour'] <= end_hour)
        ]

    return filtered


# ---------------------------------------------------------------------------
# KPI / statistics-card calculations
# ---------------------------------------------------------------------------
def kpi_today_messages(df: pd.DataFrame) -> int:
    """Count messages sent "today" relative to the most recent date in the chat.

    We use the chat's own most recent date (rather than the real-world
    "today") since chat exports are historical and rarely include the
    actual current day.
    """
    if df.empty:
        return 0
    latest_date = df['only_date'].max()
    return int((df['only_date'] == latest_date).sum())


def kpi_avg_messages_per_day(df: pd.DataFrame) -> float:
    """Average number of messages sent per active day."""
    if df.empty:
        return 0.0
    per_day = df.groupby('only_date').size()
    return round(float(per_day.mean()), 2)


def kpi_avg_words_per_message(df: pd.DataFrame) -> float:
    """Average number of words per message (excludes media placeholders)."""
    real = df[df['message'] != config.MEDIA_PLACEHOLDER]
    if real.empty:
        return 0.0
    word_counts = real['message'].str.split().str.len()
    return round(float(word_counts.mean()), 2)


def kpi_longest_message(df: pd.DataFrame) -> int:
    """Word count of the single longest message in the (filtered) chat."""
    real = df[df['message'] != config.MEDIA_PLACEHOLDER]
    if real.empty:
        return 0
    return int(real['message'].str.split().str.len().max())


def kpi_most_active_hour(df: pd.DataFrame) -> str:
    """The hour bucket (e.g. '21-22') with the most messages."""
    if df.empty or df['period'].dropna().empty:
        return "N/A"
    return str(df['period'].value_counts().idxmax())


def kpi_most_active_day(df: pd.DataFrame) -> str:
    """The day-of-week with the most messages."""
    if df.empty:
        return "N/A"
    return str(df['day_name'].value_counts().idxmax())


def compute_all_kpis(df: pd.DataFrame) -> dict:
    """Bundle every KPI card value into a single dict for easy unpacking."""
    return {
        "today_messages": kpi_today_messages(df),
        "avg_messages_per_day": kpi_avg_messages_per_day(df),
        "avg_words_per_message": kpi_avg_words_per_message(df),
        "longest_message": kpi_longest_message(df),
        "most_active_hour": kpi_most_active_hour(df),
        "most_active_day": kpi_most_active_day(df),
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_messages(
    df: pd.DataFrame,
    keyword: str = "",
    user: Optional[str] = None,
    date: Optional[object] = None,
    use_regex: bool = False,
) -> pd.DataFrame:
    """Search chat messages by keyword, user, and/or date.

    Args:
        df: The DataFrame to search within.
        keyword: Text (or regex pattern, if use_regex=True) to look for
            inside the 'message' column. Empty string matches everything.
        user: If given, only messages from this user are searched.
        date: If given (a date object), only messages on this date are
            searched.
        use_regex: Treat `keyword` as a regular expression instead of a
            plain substring.

    Returns:
        A DataFrame of matching rows (a subset of the original columns
        useful for display: date, user, message).

    Raises:
        re.error: If use_regex=True and `keyword` is not a valid pattern.
            app.py should catch this and show a friendly error message.
    """
    # No upfront .copy(): each boolean-mask filter below already returns
    # a fresh DataFrame the moment it's applied (same reasoning as
    # apply_filters() above), and the final `.sort_values()` at the
    # bottom of this function guarantees a new object is returned even
    # when no filter narrows the input at all. An unconditional .copy()
    # here would duplicate the entire (potentially 100k+ row) DataFrame
    # on every Search-tab keystroke/widget change, even before any
    # filter runs.
    result = df

    if user and user != "Overall":
        result = result[result['user'] == user]

    if date is not None:
        result = result[result['only_date'] == date]

    if keyword:
        result = result[
            result['message'].str.contains(keyword, case=False, na=False, regex=use_regex)
        ]

    return result[['date', 'user', 'message']].sort_values('date')


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes, ready for st.download_button."""
    return df.to_csv(index=False).encode('utf-8')


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to an in-memory .xlsx file's bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='chat_data')
    return buffer.getvalue()


def to_json_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to JSON bytes (records orientation)."""
    # default=str handles Timestamp/date objects that json can't serialize.
    records = json.loads(df.to_json(orient='records', date_format='iso'))
    return json.dumps(records, indent=2, default=str).encode('utf-8')


# ---------------------------------------------------------------------------
# PDF export (replaces the old JSON export button in app.py)
# ---------------------------------------------------------------------------
# A PDF is a paginated, laid-out document, not a bulk data format like CSV/
# JSON/Excel: rendering one Paragraph flowable per cell for 100,000+ rows
# would mean building hundreds of thousands of Platypus objects and a
# multi-thousand-page file, which is slow to generate, slow to open, and not
# actually useful to a reader. CSV/Excel remain the complete-dataset export
# path (unchanged); PDF is capped at a documented row count and says so
# in the document itself, rather than silently truncating.
_PDF_MAX_ROWS = 5000

# --- Unicode font support (Hindi / Devanagari + WhatsApp emoji) -----------
# ReportLab's built-in (Type 1 / "base14") fonts -- Helvetica, used
# previously -- only support the Latin-1 code page, so WhatsApp exports
# containing Devanagari or emoji were silently flattened to '?'. Instead,
# we discover and register real Unicode TTF fonts (once per process, not
# per cell) and render each message with up to three fonts depending on
# which script each character run belongs to:
#   - _PDF_FONT_REGULAR / _PDF_FONT_BOLD : general Unicode text (Latin,
#     numbers, punctuation, most symbols). Prefers a bundled font under
#     assets/fonts/ next to this file, then common Linux/Windows/macOS
#     DejaVu Sans install locations.
#   - _PDF_FONT_DEVANAGARI : Hindi/Devanagari text (Noto Sans Devanagari /
#     Lohit Devanagari on Linux, Nirmala UI / Mangal on Windows, etc.).
#   - _PDF_FONT_EMOJI : monochrome emoji glyphs (Noto Emoji / Symbola /
#     Segoe UI Symbol). ReportLab's TTFont support only embeds classic
#     outline glyphs, not colored bitmap emoji, so emoji render in
#     black-and-white when this font is available.
# If a given script's font isn't found anywhere on the deployment machine,
# that script's characters degrade to '?' (documented fallback, same as
# the old behavior) instead of emitting unrenderable glyphs -- PDF
# generation always succeeds, it just loses coverage for that script in
# that specific environment.
_PDF_FONTS_REGISTERED = False
_PDF_FONT_REGULAR = "Helvetica"
_PDF_FONT_BOLD = "Helvetica-Bold"
_PDF_FONT_DEVANAGARI: Optional[str] = None
_PDF_FONT_EMOJI: Optional[str] = None


def _first_readable(paths: list) -> Optional[str]:
    """Return the first path in `paths` that exists as a readable file."""
    for path in paths:
        try:
            if path and os.path.isfile(path):
                return path
        except OSError:
            continue
    return None


def _try_register_font(font_name: str, path: Optional[str]) -> Optional[str]:
    """Register a TTF with ReportLab; return its name on success, else None
    (never raises -- a corrupt/unsupported font file must not crash PDF
    export, it should just fall through to the next candidate)."""
    if not path:
        return None
    try:
        pdfmetrics.registerFont(TTFont(font_name, path))
        return font_name
    except Exception:
        return None


def _register_pdf_fonts() -> None:
    """Discover and register Unicode-capable fonts, once per process.

    Safe font discovery order for each font role:
        1. Application/project-bundled font (assets/fonts/ next to this
           file), so a deployment can guarantee full Hindi + emoji
           coverage by simply dropping font files there.
        2. Common Linux font locations (Debian/Ubuntu `fonts-dejavu` /
           `fonts-noto` / `fonts-lohit-deva` package paths).
        3. Common Windows font locations (DejaVu if installed, else
           Windows' own Nirmala UI/Mangal for Devanagari).
        4. Common macOS font locations.
        5. Safe fallback: ReportLab's base14 Helvetica (Latin-1 only).

    Registering a TTFont is real file I/O + parsing, so this is guarded
    by `_PDF_FONTS_REGISTERED` and only ever runs once per process,
    however many rows/cells a PDF export renders.
    """
    global _PDF_FONTS_REGISTERED, _PDF_FONT_REGULAR, _PDF_FONT_BOLD
    global _PDF_FONT_DEVANAGARI, _PDF_FONT_EMOJI

    if _PDF_FONTS_REGISTERED:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, "assets", "fonts")

    regular_path = _first_readable([
        os.path.join(bundled, "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ])
    bold_path = _first_readable([
        os.path.join(bundled, "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        "/Library/Fonts/DejaVuSans-Bold.ttf",
    ])
    devanagari_path = _first_readable([
        os.path.join(bundled, "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        "/usr/share/fonts/lohit-devanagari/Lohit-Devanagari.ttf",
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\Mangal.ttf",
        "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
    ])
    emoji_path = _first_readable([
        os.path.join(bundled, "NotoEmoji-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
        "/usr/share/fonts/noto-emoji/NotoEmoji-Regular.ttf",
        "/usr/share/fonts/truetype/symbola/Symbola.ttf",
        "/usr/share/fonts/truetype/ancient-scripts/Symbola.ttf",
        "/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf",  # Debian/Ubuntu fonts-symbola package
        r"C:\Windows\Fonts\seguisym.ttf",
        "/Library/Fonts/NotoEmoji-Regular.ttf",
    ])

    registered_regular = _try_register_font("PDFUnicode", regular_path)
    if registered_regular:
        _PDF_FONT_REGULAR = registered_regular

    registered_bold = _try_register_font("PDFUnicode-Bold", bold_path)
    if registered_bold:
        _PDF_FONT_BOLD = registered_bold
    elif _PDF_FONT_REGULAR != "Helvetica":
        # Found a Unicode regular face but no matching bold face -- reuse
        # the regular face for header text rather than silently dropping
        # back to Helvetica-Bold, which would lose Devanagari/symbol
        # coverage in the (colored) header row.
        _PDF_FONT_BOLD = _PDF_FONT_REGULAR

    _PDF_FONT_DEVANAGARI = _try_register_font("PDFDevanagari", devanagari_path)
    _PDF_FONT_EMOJI = _try_register_font("PDFEmoji", emoji_path)

    _PDF_FONTS_REGISTERED = True


_PDF_COLUMNS = [
    ("date", "Date", 70),
    ("user", "User", 80),
    ("message", "Message", None),  # None = takes remaining width
    ("year", "Year", 40),
    ("month", "Month", 60),
    ("day_name", "Day", 60),
    ("hour", "Hour", 36),
]

# Character-run classification used by `_pdf_safe_text` to pick a font per
# script. \uFE0F (variation selector-16) and \u200D (zero-width joiner) are
# included in the emoji class so multi-codepoint emoji sequences (e.g.
# "❤️" = U+2764 U+FE0F) stay in one run instead of being split.
_DEVANAGARI_RANGE = "\u0900-\u097F"
_EMOJI_RANGE = (
    "\u2190-\u21FF"       # Arrows
    "\u2300-\u23FF"       # Misc Technical (e.g. ⌚ ⏰)
    "\u2600-\u27BF"       # Misc Symbols + Dingbats (e.g. ❤ ☀ ✔ ✨)
    "\u2B00-\u2BFF"       # Misc Symbols and Arrows (e.g. ⭐ ➡)
    "\uFE0F"              # variation selector-16 (emoji presentation)
    "\u200D"              # zero-width joiner (emoji sequences)
    "\U0001F000-\U0001FFFF"  # emoticons, pictographs, flags, symbols, etc.
)
_SCRIPT_RUN_RE = re.compile(
    r"(?P<devanagari>[" + _DEVANAGARI_RANGE + r"]+)"
    r"|(?P<emoji>[" + _EMOJI_RANGE + r"]+)"
    r"|(?P<default>[^" + _DEVANAGARI_RANGE + _EMOJI_RANGE + r"]+)"
)


def _pdf_safe_text(value: object) -> str:
    """Render any cell value as PDF-safe, XML-escaped ReportLab mini-markup.

    Handles:
        - Special characters: escaped via `xml.sax.saxutils.escape` so
          '&', '<', '>' in a message can never be misread as ReportLab's
          Paragraph mini-markup.
        - Devanagari (Hindi) and emoji runs: wrapped in
          `<font face="...">` spans pointing at the registered
          `_PDF_FONT_DEVANAGARI` / `_PDF_FONT_EMOJI` fonts, so those
          scripts render with real glyphs instead of '?' wherever a
          matching Unicode font was found on the deployment machine
          (see `_register_pdf_fonts`). If no such font is available,
          that run degrades to '?' per character -- the same documented,
          crash-free fallback as before, now scoped to just the
          unsupported script instead of every non-Latin-1 character.
        - Everything else (Latin, numbers, punctuation, most symbols):
          left as plain text in the cell's base font
          (`_PDF_FONT_REGULAR`); if that font itself is still the
          Helvetica fallback (no Unicode TTF found at all), it's
          additionally latin-1-encoded same as the original behavior.
        - Newlines inside a message: converted to '<br/>' (after
          escaping) so a multi-line message wraps onto multiple visual
          lines within its cell instead of being squashed onto one.
    """
    text = "" if value is None else str(value)
    if not text:
        return ""

    _register_pdf_fonts()

    parts = []
    for match in _SCRIPT_RUN_RE.finditer(text):
        script = match.lastgroup
        run = match.group()

        if script == "devanagari" and _PDF_FONT_DEVANAGARI:
            parts.append(f'<font face="{_PDF_FONT_DEVANAGARI}">{_xml_escape(run)}</font>')
        elif script == "emoji" and _PDF_FONT_EMOJI:
            parts.append(f'<font face="{_PDF_FONT_EMOJI}">{_xml_escape(run)}</font>')
        elif script in ("devanagari", "emoji"):
            # No dedicated Unicode font found for this script on this
            # machine -- degrade to '?' (documented fallback) instead of
            # emitting glyphs the active font can't render.
            visible_len = len(run.replace("\uFE0F", "").replace("\u200D", ""))
            parts.append("?" * visible_len)
        else:
            if _PDF_FONT_REGULAR == "Helvetica":
                run = run.encode("latin-1", "replace").decode("latin-1")
            parts.append(_xml_escape(run))

    safe = "".join(parts)
    return safe.replace("\r\n", "<br/>").replace("\n", "<br/>")


def _pdf_page_footer(canvas, doc) -> None:
    """Draw 'Page N' bottom-right on every page (header row is handled
    separately by the table's own `repeatRows`, not this footer)."""
    _register_pdf_fonts()
    canvas.saveState()
    canvas.setFont(_PDF_FONT_REGULAR, 8)
    canvas.drawRightString(doc.pagesize[0] - 24, 16, f"Page {doc.page}")
    canvas.restoreState()


def to_pdf_bytes(df: pd.DataFrame, max_rows: int = _PDF_MAX_ROWS) -> bytes:
    """Serialize a DataFrame to a paginated PDF table, ready for
    `st.download_button`.

    Args:
        df: The export DataFrame -- same shape used by `to_csv_bytes` /
            `to_excel_bytes` / the old `to_json_bytes` (the filtered
            selection narrowed to the export columns in app.py).
        max_rows: Safety cap on rows actually rendered into the PDF
            table (see `_PDF_MAX_ROWS`). CSV/Excel are unaffected and
            still export every filtered row.

    Returns:
        PDF file bytes. Always returns a valid (openable) PDF, even for
        an empty DataFrame, a single row, very long messages/usernames,
        or messages containing special/non-Latin-1 characters.

    Large datasets:
        If `df` has more than `max_rows` rows, only the first
        `max_rows` are rendered and a note is printed at the top of the
        document stating how many messages were included out of the
        total, and pointing to the CSV/Excel export for the complete
        dataset -- the truncation is documented in the file, not silent.

    Pagination / layout:
        Built with ReportLab's Platypus `SimpleDocTemplate` + a single
        `Table` flowable. Platypus automatically paginates a `Table`
        that doesn't fit on one page, and `repeatRows=1` makes the
        header row reprint at the top of every continuation page.
        Long messages/usernames are wrapped (not overflowed) by placing
        each cell's text in a `Paragraph` instead of a plain string, so
        long content grows the row's height rather than the page's
        width.
    """
    _register_pdf_fonts()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28,
        title="WhatsApp Chat Export",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PDFTitle", parent=styles["Title"], fontName=_PDF_FONT_BOLD,
    )
    normal_style = ParagraphStyle(
        "PDFNormal", parent=styles["Normal"], fontName=_PDF_FONT_REGULAR,
    )
    header_style = ParagraphStyle(
        "PDFHeader", parent=styles["BodyText"],
        fontName=_PDF_FONT_BOLD, fontSize=9, leading=11,
        textColor=colors.white,
    )
    cell_style = ParagraphStyle(
        "PDFCell", parent=styles["BodyText"],
        fontName=_PDF_FONT_REGULAR, fontSize=8, leading=10,
    )

    story = [
        Paragraph("WhatsApp Chat Export", title_style),
        Paragraph(f"{len(df):,} messages matched the current filters.", normal_style),
        Spacer(1, 10),
    ]

    if df.empty:
        story.append(Paragraph("No messages match the current filters.", normal_style))
        doc.build(story)
        return buffer.getvalue()

    total_rows = len(df)
    table_df = df.head(max_rows) if total_rows > max_rows else df

    if total_rows > max_rows:
        story.append(Paragraph(
            f"Showing the first {max_rows:,} of {total_rows:,} matching messages. "
            "Use the CSV or Excel export above for the complete dataset.",
            normal_style,
        ))
        story.append(Spacer(1, 8))

    col_keys = [key for key, _label, _width in _PDF_COLUMNS if key in table_df.columns]
    col_widths_raw = [w for key, _label, w in _PDF_COLUMNS if key in table_df.columns]
    fixed_width = sum(w for w in col_widths_raw if w is not None)
    flexible_count = sum(1 for w in col_widths_raw if w is None)
    remaining = max(doc.width - fixed_width, 60 * max(flexible_count, 1))
    col_widths = [
        w if w is not None else remaining / flexible_count
        for w in col_widths_raw
    ]

    header_row = [
        Paragraph(_pdf_safe_text(label), header_style)
        for key, label, _width in _PDF_COLUMNS if key in table_df.columns
    ]
    data_rows = []
    for row in table_df[col_keys].itertuples(index=False, name=None):
        data_rows.append([Paragraph(_pdf_safe_text(value), cell_style) for value in row])

    table = Table([header_row] + data_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#075E54")),  # WhatsApp-green header
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    doc.build(story, onFirstPage=_pdf_page_footer, onLaterPages=_pdf_page_footer)
    return buffer.getvalue()
