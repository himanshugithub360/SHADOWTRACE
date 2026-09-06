
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent

STOPWORDS_PATH: Path = BASE_DIR / "stop_hinglish.txt"

# ---------------------------------------------------------------------------
# Page settings
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "SHADOWTRACE"

LOGO_PATH: Path = BASE_DIR / "assets" / "logo.png"
PAGE_ICON = str(LOGO_PATH) if LOGO_PATH.exists() else "◆"
LAYOUT: str = "wide"


PRIMARY_COLOR: str = "#7c5cff"     # Electric violet — primary brand accent
SECONDARY_COLOR: str = "#22d3ee"   # Cyan — secondary accent
ACCENT_COLOR: str = "#a78bfa"      # Soft violet — tertiary accent
BACKGROUND_COLOR: str = "#0a0a0f"  # App background (near-black)

# A color sequence Plotly Express cycles through for multi-series charts.
COLOR_SEQUENCE: list[str] = [
    "#7c5cff",
    "#22d3ee",
    "#a78bfa",
    "#34d399",
    "#fbbf24",
    "#f472b6",
    "#60a5fa",
    "#f87171",
]


PLOTLY_TEMPLATE: str = "plotly_dark"

# ---------------------------------------------------------------------------
# Ordering helpers - used to sort categorical axes (days/months) so charts
# read left-to-right in calendar order instead of alphabetical order.
# ---------------------------------------------------------------------------
DAY_ORDER: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

MONTH_ORDER: list[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------------------------------------------------------------------------
# Misc constants
# ---------------------------------------------------------------------------
MEDIA_PLACEHOLDER: str = "<Media omitted>"
GROUP_NOTIFICATION_USER: str = "group_notification"


REPLY_GAP_CAP_MINUTES: int = 180

# A silence longer than this (minutes) marks the next message as the
# start of a new "conversation" for Conversation Starters.
CONVERSATION_GAP_MINUTES: int = 30

# Width (in hours) of the contiguous quiet window Sleep Schedule looks
# for when estimating sleep/wake hours.
SLEEP_WINDOW_HOURS: int = 6

# Number of top phrases returned by default for Most Used Phrases.
NGRAM_TOP_N: int = 20

# Friendly display names for commonly shared link domains (Shared
# Domains feature). Any domain not listed here is still counted, under
# its bare hostname.
DOMAIN_LABELS: dict[str, str] = {
    "github.com": "GitHub",
    "linkedin.com": "LinkedIn",
    "instagram.com": "Instagram",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "drive.google.com": "Google Drive",
    "docs.google.com": "Google Docs",
    "maps.google.com": "Google Maps",
    "goo.gl": "Google (shortened)",
    "twitter.com": "Twitter / X",
    "x.com": "Twitter / X",
    "facebook.com": "Facebook",
    "fb.watch": "Facebook",
    "whatsapp.com": "WhatsApp",
    "wa.me": "WhatsApp",
    "reddit.com": "Reddit",
    "amazon.com": "Amazon",
    "spotify.com": "Spotify",
    "netflix.com": "Netflix",
    "tiktok.com": "TikTok",
    "pinterest.com": "Pinterest",
}

# Relative weights for each component of the Engagement Score (must be
# non-negative; don't need to sum to 1 — the final score is rescaled to
# 0-100 regardless). Tweak these to change what "engaged" means for
# your chat.
ENGAGEMENT_WEIGHTS: dict[str, float] = {
    "messages": 0.30,
    "words": 0.15,
    "media": 0.10,
    "emoji": 0.10,
    "links": 0.05,
    "response_speed": 0.15,
    "consistency": 0.15,
}


SENTIMENT_POSITIVE_THRESHOLD: float = 0.05
SENTIMENT_NEGATIVE_THRESHOLD: float = -0.05

EMOTION_LEXICON: dict[str, list[str]] = {
    "Happy": [
        "haha", "hehe", "lol", "lmao", "great", "awesome", "nice", "good news",
        "love it", "amazing", "yay", "glad", "happy", "khushi", "badhiya", "mast", "love", "love u", "khushi", "khush", "khush hu",
        "khush hoon", "bahut khush",
        "bohot khush", "bohut khush",
        "khushiyan",
        "badhiya", "badiya",
        "bahut badhiya", "bohot badhiya",
        "mast", "mast hai", "ekdum mast",
        "full mast",
        "maza", "maza aaya", "maza aa gaya",
        "mazza aaya", "mazza aa gaya",
        "sahi", "sahi hai",
        "kya baat hai",
        "wah", "wah wah",
        "waah", "waah waah",
        "arre wah", "are wah",
        "kamaal", "shandaar",
        "zabardast", "jhakaas",
    ],
    "Excited": [
        "can't wait", "cant wait", "so excited", "omg", "let's go", "lets go",
        "finally", "woohoo", "yesss", "super excited", "pumped",
        "excited", "so excited", "very excited",
        "super excited", "really excited",
        "extremely excited", "im excited",
        "i'm excited",

        # Anticipation
        "can't wait", "cant wait",
        "cannot wait", "waiting eagerly",
        "looking forward",
        "really looking forward",
        "so looking forward",
        "eagerly waiting",

        # Celebration / hype
        "omg", "omgg", "omggg",
        "let's go", "lets go",
        "yeahhh", "yeaahhh",
        "yes", "yess", "yesss", "yessss",
        "woo", "wooo", "woohoo",
        "wohoo", "wohooo",
        "yay", "yayyy",
        "finally", "finallyyy",
        "we did it", "made it",

        # Energy / enthusiasm
        "pumped", "so pumped",
        "hyped", "so hyped",
        "hype", "lets do this",
        "let's do this",
        "bring it on",
        "this is gonna be awesome",
        "this will be amazing",
        "so ready", "readyyy",

        # Internet slang
        "yaaas", "yaaass",
        "slay", "fire", "lit",
        "lets goo", "letss gooo",
        "w", "big w",
        "goated",

        # Hinglish / Hindi
        "bahut excited", "bohot excited",
        "intezaar nahi ho raha",
        "wait nahi ho raha",
        "wait nhi ho raha",
        "bahut maza aayega",
        "maza aane wala hai",
        "kya baat", "mast hoga",
        "full excited",
        "josh", "full josh",

        # Emojis
        "🤩", "🥳", "🎉", "🔥",
        "🚀", "🙌", "💃", "🕺",
        "😆", "😱",
    ],
    "Sad": [
        "sad", "miss you", "miss u", "upset", "cry", "crying", "sorry to hear",
        "dukhi", "udaas", "heartbroken", "disappointed",

        # Direct sadness
        "sad", "so sad", "very sad",
        "really sad", "feeling sad",
        "feeling low", "feeling down",
        "down", "depressed",
        "unhappy", "miserable",
        "heartbroken", "broken",

        # Missing someone
        "miss you", "miss u",
        "miss ya", "missing you",
        "missing u", "i miss you",
        "i miss u", "miss him",
        "miss her", "miss them",
        "miss those days",

        # Crying
        "cry", "crying", "cried",
        "tears", "in tears",
        "want to cry", "feel like crying",
        "can't stop crying",
        "cant stop crying",
        "😭", "😢", "🥺",

        # Disappointment
        "disappointed", "so disappointed",
        "disappointment",
        "let down", "feeling let down",
        "expected better",
        "didn't expect this",
        "didnt expect this",

        # Emotional pain
        "hurt", "hurts", "hurt me",
        "feeling hurt", "pain",
        "painful", "it hurts",
        "feeling broken",
        "lost", "feeling lost",
        "lonely", "alone",
        "feeling alone",
        "empty", "feeling empty",

        # Regret / sympathy
        "sorry to hear",
        "sorry about that",
        "feel sorry",
        "unfortunate",
        "thats sad", "that's sad",
        "so unfortunate",
        "what a shame",

        # Hinglish / Hindi
        "dukhi", "udaas",
        "bahut udaas", "bohot udaas",
        "dil toot gaya",
        "dil dukha",
        "dil dukh gaya",
        "dil dukhta hai",
        "bura laga",
        "bahut bura laga",
        "bohot bura laga",
        "mann udaas hai",
        "man udaas hai",
        "akela", "akeli",
        "akela feel kar raha",
        "yaad aa rahi hai",
        "bahut yaad aa raha",
        "rona aa raha hai",
        "ro raha hu",
        "ro rahi hu",
        "aankhon mein aansu",
        "aansu",

        # Emojis
        "😢", "😭", "🥺",
        "😞", "😔", "😟",
        "💔", "😿",
    ],
    "Angry": [
        "angry", "furious", "annoyed", "irritated", "gussa", "pagal kar diya",
        "so frustrating", "hate this", "fed up", "not fair",

        "angry", "so angry", "very angry",
        "really angry", "furious",
        "rage", "raging", "mad",
        "so mad", "pissed",
        "pissed off", "annoyed",
        "irritated", "frustrated",
        "so frustrated", "fed up",

        # Complaints / frustration
        "hate this", "hate it",
        "i hate this", "i hate it",
        "this sucks", "sucks",
        "worst", "so bad",
        "not fair", "unfair",
        "enough is enough",
        "had enough",
        "sick of this",
        "tired of this",
        "can't take this anymore",
        "cant take this anymore",

        # Annoyance
        "stop it", "just stop",
        "leave me alone",
        "dont disturb me",
        "don't disturb me",
        "so annoying",
        "annoying", "irritating",
        "what the hell",
        "wtf", "wth",
        "seriously",

        # Conflict
        "how dare you",
        "are you serious",
        "you ruined it",
        "you ruined everything",
        "why would you do that",
        "what is wrong with you",

        # Hinglish / Hindi
        "gussa", "gusse mein",
        "bahut gussa", "bohot gussa",
        "gussa aa raha hai",
        "gussa aa gaya",
        "mujhe gussa aa raha hai",
        "pagal", "pagal kar diya",
        "dimag kharab",
        "dimaag kharab",
        "dimag mat kharab kar",
        "dimaag mat kharab kar",
        "irritate mat kar",
        "irritating hai",
        "pak gaya", "pak gayi",
        "tang aa gaya",
        "tang aa gayi",
        "bas karo", "chup karo",
        "bakwas", "faltu",
        "kya bakwas hai",
        "bekaar", "ghatiya",
        "bilkul pasand nahi",
        "nafrat hai",
        "chidh", "chidh raha hai",

        # Emojis
        "😡", "😠", "🤬",
        "😤", "👿",
    ],
}

# Emoji cues per emotion (checked per-character, so combine with the
# lexicon above rather than replacing it).
EMOTION_EMOJI: dict[str, set[str]] = {
    "Happy": {
        # Smiling / happiness
        "😀", "😁", "😄", "😃", "😆",
        "😊", "☺️", "🙂", "🙃",

        # Laughter / amusement
        "😂", "🤣", "😅",

        # Affection / positive feelings
        "🥰", "😍", "😘", "😚",
        "😇", "🤗",

        # Positive gestures
        "👍", "👏", "🫶",
        "❤️", "💖", "💕", "💗",
    },

    "Excited": {
        # Excitement / celebration
        "🤩", "🥳", "🎉", "🎊",
        "🙌", "🔥", "✨",

        # High energy
        "🚀", "💥", "💫",
        "⚡", "🎯",

        # Enthusiasm
        "🤪", "💃", "🕺",

        # Achievement / success
        "🏆", "🥇", "🎁",
        "🎈", "🎂",
    },

    "Sad": {
        # Crying
        "😢", "😭", "🥹",

        # Sad expressions
        "😞", "😔", "😟",
        "🙁", "☹️", "😕",

        # Emotional pain
        "🥺", "💔",

        # Loneliness / disappointment
        "😿", "😩",
    },

    "Angry": {
        # Direct anger
        "😠", "😡", "🤬",
        "👿", "💢",

        # Frustration
        "😤", "😒", "🙄",
        "😑", "😫",

        # Aggressive gestures
        "👎", "🖕",
    },
}

# Intent Classification: keyword cues (checked as substrings) per intent
# label. A message matching none of these is labeled "Other".
INTENT_KEYWORDS: dict[str, list[str]] = {
    "Study": [
        # General study
        "study", "studying", "study session", "study plan",
        "padhai", "padhna", "padh raha", "padh rahi",
        "padhne", "revision", "revise", "revising",

        # Academic activities
        "exam", "exams", "test", "quiz",
        "assignment", "assignments",
        "homework", "hw",
        "project submission", "submission",
        "practical", "lab", "lab work",
        "viva", "presentation",

        # Classes
        "lecture", "lectures",
        "class", "class today",
        "online class", "offline class",
        "attend class", "attendance",
        "teacher", "professor", "faculty",

        # Study material
        "notes", "study notes",
        "syllabus", "chapter", "topic",
        "unit", "module", "subject",
        "book", "textbook",
        "reference", "pdf",
        "question paper", "previous year",

        # Educational institutions
        "college", "university",
        "school", "campus",
        "semester", "academic",

        # Exam-related Hinglish
        "exam hai", "paper hai",
        "paper dena", "paper ki tayari",
        "exam preparation",
        "padhai karni hai",
        "padhna hai",
        "revision karni hai",
        "assignment karna hai",
    ],

    "Work": [
        # General work
        "work", "working", "office",
        "office work", "work from home",
        "wfh", "remote work",

        # Meetings
        "meeting", "meet", "call",
        "conference call",
        "team meeting", "client meeting",
        "standup", "daily standup",

        # Tasks and projects
        "task", "tasks", "workload",
        "project", "projects",
        "report", "documentation",
        "deliverable", "deliverables",
        "requirement", "requirements",

        # Deadlines
        "deadline", "due date",
        "submission deadline",
        "urgent", "priority",

        # Workplace
        "boss", "manager",
        "team lead", "colleague",
        "coworker", "employee",
        "client", "stakeholder",

        # Career
        "job", "career",
        "interview", "job interview",
        "offer letter",
        "joining", "internship",
        "intern", "promotion",
        "resign", "resignation",
        "notice period",

        # Salary
        "salary", "paycheck",
        "pay", "bonus",
        "increment",

        # Presentations
        "presentation", "ppt",
        "demo", "review",

        # Hinglish
        "office jana hai",
        "office jaana hai",
        "kaam", "kaam hai",
        "kaam kar raha",
        "kaam kar rahi",
        "meeting hai",
        "deadline hai",
        "client call",
        "boss ne bola",
        "task complete",
        "kaam khatam",
    ],

    "Travel": [
        # General travel
        "travel", "travelling", "traveling",
        "trip", "tour", "vacation",
        "holiday", "holiday trip",
        "journey", "tourism",

        # Flights
        "flight", "flights",
        "flight ticket",
        "boarding pass",
        "departure", "arrival",
        "layover",

        # Booking
        "ticket", "tickets",
        "booking", "booked",
        "reservation",
        "cancel booking",

        # Accommodation
        "hotel", "hostel",
        "resort", "stay",
        "accommodation",
        "check in", "check out",

        # Documents
        "passport", "visa",
        "travel insurance",
        "immigration",

        # Transportation
        "airport", "train",
        "railway station",
        "bus", "cab",
        "taxi", "metro",
        "road trip",
        "driving",

        # Planning
        "itinerary", "travel plan",
        "destination",
        "places to visit",
        "sightseeing",
        "backpacking",

        # Hinglish
        "ghumne", "ghoomne",
        "ghumne jana",
        "ghoomne jana",
        "trip pe", "trip par",
        "vacation pe",
        "flight hai",
        "train hai",
        "ticket book",
        "hotel book",
        "travel plan",
    ],

    "Shopping": [
        # General shopping
        "shopping", "shop",
        "buy", "buying",
        "purchase", "purchasing",
        "kharidna", "kharidne",
        "kharid raha", "kharid rahi",

        # Online shopping
        "order", "ordered",
        "place order",
        "online order",
        "cart", "add to cart",
        "checkout",

        # Pricing
        "price", "cost",
        "expensive", "cheap",
        "affordable",
        "budget",
        "price drop",

        # Offers
        "discount", "offer",
        "sale", "deal",
        "coupon", "promo code",
        "cashback",
        "buy one get one",
        "bogo",

        # Delivery
        "delivery", "delivered",
        "out for delivery",
        "shipment", "shipping",
        "tracking",

        # Payment
        "cod", "cash on delivery",
        "payment", "pay online",

        # Returns
        "return", "refund",
        "exchange",
        "return policy",
        "replacement",

        # Platforms
        "amazon", "flipkart",
        "myntra", "meesho",
        "ajio",

        # Hinglish
        "kharidna hai",
        "order karna hai",
        "order kiya",
        "sale chal rahi",
        "discount mila",
        "kitne ka hai",
        "price kya hai",
        "mehenga", "sasta",
    ],

    "Movies": [
        # General
        "movie", "movies",
        "film", "films",
        "cinema",

        # Watching
        "watch", "watching",
        "watch movie",
        "movie night",
        "watch party",

        # Theatres
        "theatre", "theater",
        "multiplex",
        "cinema hall",
        "movie hall",
        "showtime",
        "show time",

        # Streaming
        "netflix",
        "prime video",
        "amazon prime",
        "hotstar",
        "disney hotstar",
        "jiohotstar",
        "zee5", "sonyliv",

        # Content
        "trailer", "teaser",
        "release", "movie release",
        "review", "movie review",
        "spoiler",

        # Series
        "series", "tv series",
        "web series",
        "webseries",
        "episode", "episodes",
        "season", "next season",

        # OTT
        "ott", "streaming",

        # Hinglish
        "movie dekhne",
        "film dekhne",
        "picture dekhne",
        "movie dekhi",
        "film dekhi",
        "picture",
        "theatre jana",
        "movie night",
    ],

    "Sports": [
        # General
        "sports", "sport",
        "game", "match",

        # Cricket
        "cricket", "ipl",
        "odi", "test match",
        "t20", "wicket",
        "batting", "bowling",
        "runs", "six", "four",
        "century",

        # Football
        "football", "soccer",
        "goal", "penalty",
        "fifa", "champions league",

        # Competitions
        "tournament",
        "league", "final",
        "semi final", "semifinal",
        "playoff",
        "world cup",

        # Scores
        "score", "scorecard",
        "won", "lost",
        "team won",
        "team lost",
        "points",

        # Fitness
        "gym", "workout",
        "exercise", "training",
        "running", "jogging",
        "fitness",
        "cardio",
        "weightlifting",
        "lift", "lifting",

        # Hinglish
        "match dekh raha",
        "match dekh rahi",
        "match hai",
        "cricket khelne",
        "football khelne",
        "gym jana",
        "workout karna",
        "team jeet gayi",
        "match jeet gaye",
        "khelne jana",
    ],
}

# ---------------------------------------------------------------------------
# Toxicity Detection
# ---------------------------------------------------------------------------

# Toxicity score cutoff (0-1) used by toxicity.py to classify a message.
TOXICITY_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Toxicity heuristic word list
# ---------------------------------------------------------------------------
# Lightweight keyword/phrase-based toxicity signal.
#
# Single words and phrases are intentionally separated because phrases
# provide stronger context.
#
# Example:
#   "shut the door" -> not toxic
#   "shut up"       -> potentially toxic
#
# No slurs are included. This is not a moderation-grade classifier.
# ---------------------------------------------------------------------------


TOXIC_WORDS: set[str] = {
    # Intelligence-related insults
    "stupid",
    "idiot",
    "idiotic",
    "dumb",
    "moron",
    "fool",
    "foolish",
    "brainless",
    "clueless",

    # Personal insults
    "loser",
    "pathetic",
    "worthless",
    "useless",
    "jerk",
    "creep",
    "weirdo",

    # Strong negative descriptors
    "disgusting",
    "awful",
    "terrible",
    "horrible",
    "ridiculous",

    # Dismissive language
    "trash",
    "garbage",
    "nonsense",
    "rubbish",

    # Mild profanity
    "crap",
    "bullshit",

    # Hinglish / Hindi (Roman)
    "pagal",
    "bewakoof",
    "bakwas",
    "faltu",
    "ghatiya",
    "nikamma",
    "bekaar",
    "chutiya",
    "chutiye",
    "gadha",
    "gadhi",
    "nalayak",
    "kamina",
    "kamine",
    "harami",
    "haramkhor",
    "saala",
    "saale",
    "saali",
    "kutta",
    "kutiya",
    "kutte",
    "ullu",
    "bakchod",
    "chapri",
    "jhantu",
    "nalla",
    "dhakkan",
    "budhu",
    "gawar",
    "ganwar",
    "lafanga",
    "luchcha",
    "badmaash",
    "besharam",
    "aad", "aand", "bahenchod", "behenchod", "bhenchod", "bhenchodd",
"b.c.", "bc", "bakchod", "bakchodd", "bakchodi", "bevda", "bewda",
"bevdey", "bewday", "bevakoof", "bevkoof", "bevkuf", "bewakoof",
"bewkoof", "bewkuf", "bhadua", "bhaduaa", "bhadva", "bhadvaa",
"bhadwa", "bhadwaa", "bhosada", "bhosda", "bhosdaa", "bhosdike",
"bhonsdike", "bsdk", "b.s.d.k", "bhosdiki", "bhosdiwala",
"bhosdiwale", "bhosadchodal", "bhosadchod", "bhosadchodal",
"bhosadchod", "babbey", "bube", "bubey", "bur", "burr",
"buurr", "buur", "charsi", "chooche", "choochi", "chuchi", "chhod",
"chod", "chodd", "chudne", "chudney", "chudwa", "chudwaa",
"chudwane", "chudwaane", "choot", "chut", "chute", "chutia",
"chutiya", "chutiye", "chuttad", "chutad", "dalaal", "dalal",
"dalle", "dalley", "fattu", "gadha", "gadhe", "gadhalund", "gaand",
"gand", "gandu", "gandfat", "gandfut", "gandiya", "gandiye", "goo",
"gu", "gote", "gotey", "gotte", "hag", "haggu", "hagne", "hagney",
"harami", "haramjada", "haraamjaada", "haramzyada", "haraamzyaada",
"haraamjaade", "haraamzaade", "haraamkhor", "haramkhor", "jhat",
"jhaat", "jhaatu", "jhatu", "kutta", "kutte", "kuttey", "kutia",
"kutiya", "kuttiya", "kutti", "landi", "landy", "laude", "laudey",
"laura", "lora", "lauda", "ling", "loda", "lode", "lund", "launda",
"lounde", "laundey", "laundi", "loundi", "laundiya", "loundiya",
"lulli", "maar", "maro", "marunga", "madarchod", "madarchodd",
"madarchood", "madarchoot", "madarchut", "motherfucker", "m.c.", "mc", "mamme",
"mammey", "moot", "mut", "mootne", "mutne", "mooth", "muth", "nunni", "paaji", "paji", "pesaab", "pesab", "peshaab", "peshab",
"pilla", "pillay", "pille", "pilley", "pisaab", "pisab", "pkmkb",
"porkistan", "raand", "rand", "randi", "randy", "suar", "tatte",
"tatti", "tatty", "ullu"
}


TOXIC_PHRASES: set[str] = {
    # Direct hostility
    "shut up",
    "just shut up",
    "hate you",
    "hate u",
    "i hate you",
    "i hate u",

    # Dismissive / hostile phrases
    "get lost",
    "go away",
    "leave me alone",

    # Personal attacks
    "you are stupid",
    "you're stupid",
    "you are dumb",
    "you're dumb",
    "you are an idiot",
    "you're an idiot",
    "such an idiot",
    "what an idiot",

    # Negative personal attacks
    "you are useless",
    "you're useless",
    "you are worthless",
    "you're worthless",
    "you are pathetic",
    "you're pathetic",

    # Strong negative statements
    "you are trash",
    "you're trash",
    "absolute trash",
    "complete nonsense",

    # Hinglish / Hindi phrases
    "pagal hai kya",
    "pagal ho kya",
    "dimag kharab hai",
    "dimaag kharab hai",
    "dimag mat kharab kar",
    "dimaag mat kharab kar",
    "kya bakwas hai",
    "bakwas band kar",
    "chup kar",
    "chup ho ja",

      # -------------------------------------------------------------------
    # Abusive expressions based on the expanded word list
    # -------------------------------------------------------------------
    "gand mara",
    "gaand mara",
    "gand mar",
    "gaand mar",

    "mar ja",
    "ja mar",
    "mar jao",
    "jaake mar",
    "jaa ke mar",

    "marunga tujhe",
    "maar dunga",
    "maar dungi",
    "maarunga tujhe",
    "maarungi tujhe",

    "chutiya hai kya",
    "chutiya hai tu",
    "chutiya ho kya",
    "chutiya aadmi",
    "chutiya insaan",

    "gandu hai kya",
    "gandu hai tu",
    "gandu aadmi",

    "harami hai tu",
    "harami aadmi",
    "haramzada hai tu",
    "haraamzada hai tu",

    "kamina hai tu",
    "kamine ho tum",
    "kamina aadmi",

    "kutte ki aulaad",
    "kutta hai tu",
    "kutta saala",

    "saale kutte",
    "saala kutta",
    "saale harami",

    "madarchod hai tu",
    "madarchod saala",
    "madarchod aadmi",

    "bahenchod hai tu",
    "behenchod hai tu",
    "bhenchod hai tu",

    "bhosdike hai tu",
    "bhosdiwale hai tu",
    "bhosdiwale aadmi",

    "gand mein dum",
    "gaand mein dum",

    "gand fat gayi",
    "gaand fat gayi",
    "gand phat gayi",
    "gaand phat gayi",

    # -------------------------------------------------------------------
    # Variations commonly seen in chats
    # -------------------------------------------------------------------
    "bc kya hai",
    "mc kya hai",
    "bsdk hai tu",

    "abe chutiye",
    "abe chutiya",
    "oye chutiye",
    "oye chutiya",

    "abe gadhe",
    "oye gadhe",

    "abe saale",
    "oye saale",

    "abe kutte",
    "oye kutte",

    "saale haramkhor",
    "harami saale",
}

# Spam Detection thresholds.
SPAM_SCORE_THRESHOLD: float = 0.5
SPAM_DUPLICATE_THRESHOLD: int = 3  # same exact message resent >= N times
SPAM_KEYWORDS: set[str] = {
    "subscribe", "winner", "claim", "free", "prize", "lottery", "urgent",
    "limited offer", "click here", "act now", "congratulations", "cashback",
}

# ---------------------------------------------------------------------------
# Custom CSS injected once at app startup (see utils.inject_custom_css).
# Designed to look reasonable in BOTH Streamlit light and dark themes by
# relying on Streamlit's own CSS variables instead of hard-coded colors.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 5 — AI Assistant constants (see ai_helper.py, rag.py,
# vector_store.py, prompt.py). Nothing above this section was changed.
# ---------------------------------------------------------------------------

# NOTE: OpenAI was replaced by OpenRouter as a GENERATION provider. This
# is generation-only — it does not touch embeddings.
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "auto").strip().lower()

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
# Embeddings are independent from AI_PROVIDER generation routing.
#
# Priority: Gemini -> OpenRouter -> Local Hashing Embeddings (see
# EMBEDDING_PROVIDER_PRIORITY below and vector_store.py). OpenRouter
# exposes an OpenAI-compatible embeddings API, so it's accessed through
# langchain_openai.OpenAIEmbeddings pointed at OPENROUTER_BASE_URL —
# there is no separate direct-OpenAI embeddings path in this app (an
# `OPENAI_API_KEY` variable existed in an earlier revision before that
# fallback slot was migrated to OpenRouter; it is not read anywhere in
# the current codebase, so it's intentionally not defined here).

GEMINI_EMBEDDING_MODEL: str = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)

OPENROUTER_EMBEDDING_MODEL: str = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL",
    "openai/text-embedding-3-small",
)

# Ordered embedding provider fallback chain (vector_store.py). Kept as its
# own list — deliberately NOT config.PROVIDER_PRIORITY — because embedding
# fallback must never be coupled to AI generation routing/cooldowns: a
# generation provider going down (or being explicitly pinned via
# AI_PROVIDER) must have zero effect on which embedding provider Semantic
# Search / RAG use, and vice versa. "local" (LocalHashingEmbeddings) is
# always last and never fails — it's a pure-Python fallback with no
# network/key dependency, so the chain always has a working end.
EMBEDDING_PROVIDER_PRIORITY: tuple[str, ...] = ("gemini", "openrouter", "local")

# In-session cooldown (seconds) applied to an embedding provider right
# after it fails with a quota/rate-limit error, so the next Semantic
# Search / AI question in the SAME session skips straight past a
# provider already known to be exhausted instead of spending a request
# to rediscover that. Independent of PROVIDER_QUOTA_COOLDOWN_SECONDS
# (generation-only). Longer than the generation cooldown by default
# because embedding quotas tend to be smaller/slower to reset and a
# whole chat re-index is a much more expensive retry than one chat
# message.
EMBEDDING_PROVIDER_COOLDOWN_SECONDS: int = int(
    os.getenv("EMBEDDING_PROVIDER_COOLDOWN_SECONDS", "600")
)

# How many total attempts a single embedding provider gets for one batch
# when the failure looks network/transient (not quota/auth/model-not-
# found) before giving up on that provider and moving to the next one.
# "1" extra retry = 2 attempts total, matching the spec's "at most one
# small retry then fallback" — never hammer a struggling provider.
EMBEDDING_TRANSIENT_RETRY_ATTEMPTS: int = int(
    os.getenv("EMBEDDING_TRANSIENT_RETRY_ATTEMPTS", "1")
)

GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Fallback providers (generation only — embeddings never come from
# Groq/OpenRouter-as-chat/Mistral; they stay on the Gemini/OpenRouter/
# local chain in vector_store.py, independent of this list).
GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

# OpenRouter is an OpenAI-compatible HTTP API that fronts many models
# from many labs — routed through langchain_openai.ChatOpenAI pointed at
# OPENROUTER_BASE_URL (see ai_helper._build_provider_client). Model IDs
# are namespaced "vendor/model" (default below is OpenAI's gpt-4o-mini
# served *through* OpenRouter, keeping cost/behavior close to the old
# direct-OpenAI default it replaces).
OPENROUTER_CHAT_MODEL: str = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Optional attribution headers OpenRouter uses for its public leaderboards
# (https://openrouter.ai/docs) — harmless to omit, never required.
OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "").strip()

MISTRAL_CHAT_MODEL: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
# Bumped from 1024: this app's longer features (AI Recommendations'
# exactly-5-items format, AI Insights' 5-8 bullets, AI-generated
# Reports' 6 sections) routinely need more than 1024 tokens once a
# "thinking" model's reasoning tokens are also drawn from this same
# budget (see GEMINI_THINKING_BUDGET below) -- 1024 was tight even
# without that, and was the main cause of responses stopping mid-word
# with no error. ai_helper._cached_ai_request also retries once with
# double this budget if a response still comes back looking cut off.
LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
# Hard ceiling for the one-time "response looked truncated" retry in
# ai_helper._cached_ai_request, regardless of how large
# LLM_MAX_OUTPUT_TOKENS is configured -- keeps a single retry bounded
# in cost/latency instead of scaling unboundedly with the base setting.
LLM_MAX_OUTPUT_TOKENS_RETRY_CAP: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS_RETRY_CAP", "4096"))

# Per-request network timeout (seconds) applied to every provider client
# (Gemini/Groq/OpenRouter/Mistral). Without this, a slow/unresponsive
# provider blocks on its SDK's own (much longer) default timeout before
# ai_helper's multi-provider fallback ever gets a chance to try the next
# one -- this is the main reason "AI Recommendations"/"AI Insights" can
# feel like they hang. Bounding each attempt keeps the worst case (every
# configured provider timing out) to roughly
# AI_REQUEST_TIMEOUT_SECONDS * number_of_configured_providers instead of
# unbounded.
AI_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "20"))

# Some newer Gemini model families ("fixed sampling defaults") ignore an
# explicit `temperature` and log a warning instead of erroring — the
# gemini-3.x line does this as of this writing. Rather than hardcode
# that check in ai_helper.py, the prefixes live here (single source of
# truth for model behavior, same as the model names themselves).
# get_llm() only passes `temperature=` to models that DON'T start with
# one of these prefixes, which silences the warning without changing
# any model's actual behavior (a model that ignores the parameter
# behaves identically whether we send it or not).
GEMINI_FIXED_SAMPLING_PREFIXES: tuple[str, ...] = ("gemini-3",)

# Gemini's "thinking" model families (2.5+/3.x) spend part of their
# response token budget on internal reasoning ("thought tokens") before
# writing any visible answer -- and that reasoning draws from the SAME
# max_output_tokens budget as the final text. Left unbounded, this was
# the actual root cause of "the AI generates half the answer and then
# gets stuck": the model's thinking consumed most/all of the token
# budget, and the API stops the response (finish_reason=MAX_TOKENS)
# mid-sentence with no error surfaced. Setting a fixed, small thinking
# budget keeps that reasoning bounded so the bulk of
# LLM_MAX_OUTPUT_TOKENS goes to the actual answer -- this app's
# features (grounded summaries/insights/recommendations from
# already-computed stats) don't need deep chain-of-thought, so 0
# (thinking disabled) is the default; raise it via the env var if a
# future Gemini model benefits from some reasoning budget.
GEMINI_THINKING_MODEL_PREFIXES: tuple[str, ...] = ("gemini-2.5", "gemini-3")
GEMINI_THINKING_BUDGET: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

# ---------------------------------------------------------------------------
# Central provider registry — the single source of truth for the 4-way
# AI router in ai_helper.py. PROVIDER_PRIORITY is also the fixed fallback
# order for AI_PROVIDER=auto: Gemini (primary) -> Groq -> OpenRouter ->
# Mistral. (OpenRouter replaces the old direct-OpenAI generation slot;
# embeddings are unaffected — see the embedding-configuration note above.)
# ---------------------------------------------------------------------------
PROVIDER_PRIORITY: tuple[str, ...] = ("gemini", "groq", "openrouter", "mistral")

PROVIDER_API_KEYS: dict[str, str] = {
    "gemini": GOOGLE_API_KEY,
    "groq": GROQ_API_KEY,
    "openrouter": OPENROUTER_API_KEY,
    "mistral": MISTRAL_API_KEY,
}

PROVIDER_ENV_VAR_NAMES: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

PROVIDER_CHAT_MODELS: dict[str, str] = {
    "gemini": GEMINI_CHAT_MODEL,
    "groq": GROQ_CHAT_MODEL,
    "openrouter": OPENROUTER_CHAT_MODEL,
    "mistral": MISTRAL_CHAT_MODEL,
}

# Maximum providers ever attempted for a single request (Gemini, OpenAI,
# Groq, Mistral — one pass through PROVIDER_PRIORITY, never more).
AI_MAX_PROVIDER_ATTEMPTS: int = len(PROVIDER_PRIORITY)

# Short-lived, in-session-only cooldown (seconds) applied to a provider
# immediately after it returns a quota (429/RESOURCE_EXHAUSTED) failure,
# so a burst of requests later in the SAME session skips straight past a
# provider already known to be exhausted instead of spending a request
# to rediscover that. This is set only at the moment a real request
# fails and read only at the moment another real request is routed —
# never a background health check (see Phase 27/28: periodic pings that
# call all providers just to check availability are explicitly NOT
# implemented anywhere in this app).
PROVIDER_QUOTA_COOLDOWN_SECONDS: int = int(os.getenv("PROVIDER_QUOTA_COOLDOWN_SECONDS", "120"))

# Vector store backend for RAG: "faiss" (fast, file-based, default) or
# "chroma" (persistent client, nicer for incremental/large chats).
VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "faiss").strip().lower()
VECTOR_STORE_DIR: Path = BASE_DIR / ".vector_store"
CHROMA_COLLECTION_NAME: str = "whatsapp_chat"

# How chat messages are grouped into documents before embedding. Single
# messages are usually too short/context-free to embed well, so
# consecutive messages from the (already filtered) chat are windowed
# into overlapping chunks instead.
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))       # characters
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))  # characters
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))                    # chunks retrieved per question

# Chat-with-history: how many previous Q&A turns are replayed back to
# the LLM as conversational memory (older turns are dropped, not lost —
# they stay visible in the UI transcript).
AI_MEMORY_TURNS: int = int(os.getenv("AI_MEMORY_TURNS", "6"))

# Safety cap on how many raw messages are fed directly into a single
# prompt (e.g. Daily Summary). Longer windows are pre-aggregated by
# analytics.py instead of dumped in raw, to control token usage/cost.
# Phase 6: lowered the default from 400 -> 150. Summaries/highlights lean
# on `_stats_block` (already-computed numbers) for the bulk of grounding;
# raw messages are only there to add color/quotes, so they don't need to
# be the full window. Override via .env if you want more.
AI_MAX_RAW_MESSAGES_IN_PROMPT: int = int(os.getenv("AI_MAX_RAW_MESSAGES_IN_PROMPT", "150"))

# ---------------------------------------------------------------------------
# Phase 6 — AI + RAG Optimization
# ---------------------------------------------------------------------------
# Route obviously-deterministic questions ("Who talks the most?", "What's
# the most active hour?") straight to the existing Python analytics
# functions instead of calling the LLM at all. Zero tokens, zero API
# latency, and the number can never be "confidently wrong" the way an LLM
# restating a stat sometimes is. Questions that don't match a known
# pattern still fall through to the normal stats+RAG+LLM flow untouched.
AI_DETERMINISTIC_ROUTING: bool = os.getenv("AI_DETERMINISTIC_ROUTING", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

# Persist the RAG vector index to disk (under VECTOR_STORE_DIR), keyed by
# the same fingerprint used for the in-memory st.cache_resource index.
# This means the index survives a Streamlit process restart / a
# `st.cache_resource.clear()` — re-opening the same chat (same fingerprint)
# loads the saved index instead of re-embedding every message again.
RAG_PERSIST_INDEX: bool = os.getenv("RAG_PERSIST_INDEX", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

# ---------------------------------------------------------------------------
# Phase 7 (quota hardening) — Gemini request minimization
# ---------------------------------------------------------------------------
# Bump this whenever a prompt template in prompt.py changes meaning
# (wording that would change the answer). It's folded into every AI
# response's cache key (see ai_helper._run), so old cached answers from
# a stale prompt are never served after a prompt edit — but simply
# restarting the app / rerunning Streamlit does NOT bump it, so caches
# still survive normal usage.
AI_PROMPT_VERSION: str = os.getenv("AI_PROMPT_VERSION", "v1")

# Maximum number of corrective retries for AI Recommendations (the only
# feature with a validate-then-maybe-retry step: "exactly 5 numbered
# recommendations"). Free-tier Gemini quotas are small (the reported
# limit was 20 requests), so this defaults to 1 and is NEVER used when
# the first attempt already failed outright (auth/quota/invalid-request
# errors are never retried — see ai_helper._classify_error). Set to 0 in
# .env to disable the retry entirely and always accept whatever the
# first response contains.
AI_MAX_RETRY: int = int(os.getenv("AI_MAX_RETRY", "1"))

# ---------------------------------------------------------------------------
# Part 8 — Performance debug mode
# ---------------------------------------------------------------------------
# When enabled, nav.py's sidebar debug panel gains a lightweight
# "Performance Debug" section showing timing for startup, preprocessing,
# filtering, the currently active section, model loading, and RAG
# indexing (see nav.time_block), plus the number of messages processed.
# Off by default so normal users never see it. Uses time.perf_counter()
# around a handful of coarse-grained blocks only -- no per-line or
# per-chart-call instrumentation.
PERFORMANCE_DEBUG: bool = os.getenv("PERFORMANCE_DEBUG", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

CUSTOM_CSS: str = """
<style>
/* Metric / KPI cards */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 1rem 1rem 0.6rem 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

div[data-testid="stMetric"] label {
    font-weight: 600;
}

/* Section headers */
h1, h2, h3 {
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.15);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
}

/* Dataframe rounded corners */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
"""
