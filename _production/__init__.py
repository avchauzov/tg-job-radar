import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database credentials
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_NAME = os.getenv('POSTGRES_NAME')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASS = os.getenv('POSTGRES_PASS')

# API credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')

# Email settings
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')

# Database table names
RAW_DATA__TG_POSTS = 'raw_data.tg_posts'
STAGING_DATA__POSTS = 'staging_data.posts'
PROD_DATA__JOBS = 'prod_data.jobs'

# Constants
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LLM_BASE_MODEL = 'gpt-4o-mini'
EMAIL_NOTIFICATION_CHUNK_SIZE = 16

# Regex patterns
URL_EXTRACT_PATTERN = r'https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])'
URL_REMOVAL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')

# TODO: add tests