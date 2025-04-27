import os
import re
from re import Pattern
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CURRENT_SERVER_URL: str = os.getenv("CURRENT_SERVER_URL", "")

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

# Instance configuration
EC2_MANAGER: dict[str, str] = {
    "EC2_MANAGER_URL": os.getenv("EC2_MANAGER_URL", "http://localhost:8000"),
    "LLM_INSTANCE_ID": os.getenv("LLM_INSTANCE_ID", ""),
    "LLM_INSTANCE_REGION": os.getenv("LLM_INSTANCE_REGION", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}

CV_DOC_ID: str = os.getenv("CV_DOC_ID", "")


# Custom model configuration with sensible defaults
CUSTOM_MODEL_ENABLED: bool = True
LLM_INSTANCE_URL: str = os.getenv("LLM_INSTANCE_URL", "")


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
MATCH_SCORE_THRESHOLD: float = 2.0  # Value between 1.0-3.0

LOOKBACK_DAYS = 30

DATA_BATCH_SIZE = 8
NUMBER_OF_BATCHES = 1
MAX_RETRY_ATTEMPTS = 3

GDOCS_TIMEOUT_SECONDS = 8
MIN_CV_LENGTH = 128
CV_COMPRESSION_RATIO = 2  # Ratio for CV summarization
JOB_POST_COMPRESSION_RATIO = 2  # Ratio for job post rewriting
MAX_CONTEXT_TOKENS = 8096 * 0.75  # Maximum tokens for input (leaving room for prompts)

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

if not (1.0 <= MATCH_SCORE_THRESHOLD <= 3.0):
    raise ValueError("MATCH_SCORE_THRESHOLD must be between 1.0 and 3.0")

# Validate Telegram API_ID is numeric
try:
    int(TELEGRAM["API_ID"])
except Exception as error:
    raise Exception(f"TG_API_ID must be a numeric value: {error}") from error
