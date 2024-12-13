import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configurations
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

TG_API_ID = os.getenv("TG_API_ID")
TG_API_HASH = os.getenv("TG_API_HASH")

# API Keys and Authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_MODEL = "gpt-4o-mini"

# Email Configuration
EMAIL_NOTIFICATION_CHUNK_SIZE = 32
EMAIL_NOTIFICATION_CHUNK_MULTIPLIER = 10

# Database Table Names
RAW_DATA__TG_POSTS = "raw_data.tg_posts"
STAGING_DATA__POSTS = "staging_data.posts"
PROD_DATA__JOBS = "prod_data.jobs"

# Constants
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Regex patterns
URL_EXTRACT_PATTERN = r"https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])"
URL_REMOVAL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
