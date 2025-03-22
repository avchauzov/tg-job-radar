import os
import re
from re import Pattern
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SERVER_URL: str = os.getenv("SERVER_URL", "")

# InfluxDB configuration
INFLUXDB: dict[str, str] = {
    "TOKEN": os.getenv("INFLUXDB_TOKEN", ""),
    "ORG": os.getenv("INFLUXDB_ORG", ""),
    "BUCKET": os.getenv("INFLUXDB_BUCKET", ""),
}

# Group related environment variables using dictionaries for better organization
DATABASE: dict[str, str] = {
    "HOST": os.getenv("DB_HOST", ""),
    "NAME": os.getenv("DB_NAME", ""),
    "DB_TELEFLOW_NAME": os.getenv("DB_TELEFLOW_NAME", ""),
    "USER": os.getenv("DB_USER", ""),
    "PASSWORD": os.getenv("DB_PASSWORD", ""),
}

EMAIL: dict[str, Any] = {
    "SENDER": os.getenv("SENDER_EMAIL", ""),
    "RECIPIENT": os.getenv("RECIPIENT_EMAIL", ""),
    "GMAIL_APP_PASSWORD": os.getenv("GMAIL_APP_PASSWORD", ""),
}

TELEGRAM: dict[str, str] = {
    "API_ID": os.getenv("TG_API_ID", "0"),
    "API_HASH": os.getenv("TG_API_HASH", ""),
}

CV_DOC_ID: str = os.getenv("CV_DOC_ID", "")
LLM_BASE_MODEL = "claude-3-5-haiku-latest"

# Add type hints and default values for critical configurations
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


# URL regex patterns with type hints
URL_EXTRACT_PATTERN: str = r"https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])"
URL_REMOVAL_PATTERN: Pattern = re.compile(r"(https?://\S+|www\.\S+)")

PROBLEM_CHARS = {
    "\n",
    "\r",
    "\t",
    '"',
    "'",
    "«",
    "»",
    "„",
    "‟",
    "—",
    "-",
    "―",
    "•",
    "…",
    "\u200b",
    "\u200e",
    "\u200f",
    "\ufeff",
    "″",
    "‴",
    "⁗",
    "≪",
    "≫",
    "❛",
    "❜",
    "❝",
    "❞",
}

# Constants with type hints
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
TEXT_SIMILARITY_THRESHOLD: float = 0.90
MATCH_SCORE_THRESHOLD: int = 85  # Value between 0-100

DATA_BATCH_SIZE = 32
NUMBER_OF_BATCHES = 32
MAX_RETRY_ATTEMPTS = 3

RAW_DATA__TG_POSTS = "raw_data.tg_posts"
STAGING_DATA__POSTS = "staging_data.posts"
PROD_DATA__JOBS = "prod_data.jobs"

# Validate critical configurations
missing_db_fields = [k for k, v in DATABASE.items() if not v]
if missing_db_fields:
    raise ValueError(
        f"Missing database configuration values: {', '.join(missing_db_fields)}"
    )

missing_email_fields = [
    k
    for k, v in EMAIL.items()
    if k in ["SENDER", "RECIPIENT", "GMAIL_APP_PASSWORD"] and not v
]
if missing_email_fields:
    raise ValueError(
        f"Missing email configuration values: {', '.join(missing_email_fields)}"
    )

missing_telegram_fields = [k for k, v in TELEGRAM.items() if not v]
if missing_telegram_fields:
    raise ValueError(
        f"Missing Telegram configuration values: {', '.join(missing_telegram_fields)}"
    )

if not CV_DOC_ID:
    raise ValueError("CV_DOC_ID environment variable is required")

if not (0 <= MATCH_SCORE_THRESHOLD <= 100):
    raise ValueError("MATCH_SCORE_THRESHOLD must be between 0 and 100")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

# Validate Telegram API_ID is numeric
try:
    int(TELEGRAM["API_ID"])
except Exception as error:
    raise Exception(f"TG_API_ID must be a numeric value: {error}") from error
