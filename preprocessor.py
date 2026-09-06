
import re
import pandas as pd

# Regex that matches a WhatsApp date/time stamp, e.g. "12/08/24, 9:41 pm - "
_DATE_PATTERN = r'\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s*[ap]m\s-\s'

# Unicode "invisible" direction-control characters that some phones insert
# around numbers/names in exported chats. We strip them so text is clean.
_INVISIBLE_CHARS_PATTERN = r'[\u2066-\u2069]'


def preprocess(data: str, verbose: bool = False) -> pd.DataFrame:
    """Parse a raw WhatsApp chat export into a tidy DataFrame.

    Args:
        data: The full text content of the exported WhatsApp .txt file.
        verbose: If True, prints basic parsing diagnostics (message/date
            counts). Defaults to False to keep the app's console output
            clean.

    Returns:
        A DataFrame with one row per message and columns:
        user, message, date, date_str, year, month_num, month, only_date,
        day, day_name, week_num, hour, minute, period.
    """
    messages = re.split(_DATE_PATTERN, data, flags=re.IGNORECASE)[1:]
    messages = [
        re.sub(_INVISIBLE_CHARS_PATTERN, '', msg).replace('@', ' ').strip()
        for msg in messages
    ]

    dates = re.findall(_DATE_PATTERN, data, flags=re.IGNORECASE)
    # Some exports use a non-breaking space (U+202F) between time and am/pm.
    dates = [d.replace('\u202f', ' ') for d in dates]

    if verbose:
        print(f"Messages parsed: {len(messages)}")
        print(f"Dates parsed: {len(dates)}")

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert the raw date strings into real datetime objects.
    df['message_date'] = pd.to_datetime(
        df['message_date'],
        format='%d/%m/%y, %I:%M %p - ',
        errors='coerce',
    )

    if verbose:
        print(f"Unparseable dates (NaT): {df['message_date'].isna().sum()}")

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Split "Author: message text" into separate columns. Messages without
    # a "User: " prefix are WhatsApp's own system messages (e.g. "X joined").
    users, messages_clean = [], []
    for msg in df['user_message']:
        if ': ' in msg:
            user, message = msg.split(': ', 1)
            users.append(user)
            messages_clean.append(message)
        else:
            users.append('group_notification')
            messages_clean.append(msg)

    df['user'] = users
    df['message'] = messages_clean
    df.drop(columns=['user_message'], inplace=True)

    # Derived date/time columns used throughout the dashboard.
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['only_date'] = df['date'].dt.date
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['week_num'] = df['date'].dt.isocalendar().week.astype('Int64')
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')

    # Human-friendly hour bucket, e.g. "09-10", used by the heatmap.
    df['period'] = df['hour'].apply(
        lambda hour: f"{hour:02d}-{(hour + 1) % 24:02d}" if pd.notna(hour) else None
    )

    if verbose:
        print(f"Final DataFrame shape: {df.shape}")

    return df
