import os
import re
from typing import Any, Dict, Pattern

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Group related environment variables using dictionaries for better organization
DATABASE: Dict[str, str] = {
    "HOST": os.getenv("DB_HOST", ""),
    "NAME": os.getenv("DB_NAME", ""),
    "USER": os.getenv("DB_USER", ""),
    "PASSWORD": os.getenv("DB_PASSWORD", ""),
}

EMAIL: Dict[str, Any] = {
    "SENDER": os.getenv("SENDER_EMAIL", ""),
    "RECIPIENT": os.getenv("RECIPIENT_EMAIL", ""),
    "GMAIL_APP_PASSWORD": os.getenv("GMAIL_APP_PASSWORD", ""),
}

TELEGRAM: Dict[str, str] = {
    "API_ID": os.getenv("TG_API_ID", "0"),
    "API_HASH": os.getenv("TG_API_HASH", ""),
}

CV_DOC_ID: str = os.getenv("CV_DOC_ID", "")
LLM_BASE_MODEL = "gpt-4o-mini"

# Add type hints and default values for critical configurations
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


# URL regex patterns with type hints
URL_EXTRACT_PATTERN: str = r"https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])"
URL_REMOVAL_PATTERN: Pattern = re.compile(r"(https?://\S+|www\.\S+)")

# Constants with type hints
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
TEXT_SIMILARITY_THRESHOLD: float = 0.95
MATCH_SCORE_THRESHOLD: int = 70  # Value between 0-100

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

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Validate Telegram API_ID is numeric
try:
    int(TELEGRAM["API_ID"])
except ValueError:
    raise ValueError("TG_API_ID must be a numeric value")
