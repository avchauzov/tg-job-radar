import os
import re

from dotenv import load_dotenv


load_dotenv()

POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_NAME = os.getenv('POSTGRES_NAME')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASS = os.getenv('POSTGRES_PASS')

PHONE_NUMBER = os.getenv('PHONE_NUMBER')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')

RAW_DATA__TG_POSTS = 'raw_data.tg_posts'
RAW_DATA__TG_POSTS__CONFLICT = ['id']

STAGING_DATA__POSTS = 'staging_data.posts'
STAGING_DATA__POSTS__CONFLICT = ['id']
RAW_TO_STAGING__WHERE = f'id not in (select id from {STAGING_DATA__POSTS})'

PROD_DATA__JOBS = 'prod_data.jobs'
STAGING_TO_PROD__WHERE = f"""is_job_post is True and 
is_single_job_post is True and
id not in (select id from {PROD_DATA__JOBS})"""

PROD_DATA__JOBS__SELECT = 'id, channel, post_structured, post_link'
PROD_DATA__JOBS__WHERE = 'notificated = false'
PROD_DATA__JOBS__ORDER_BY = 'date asc, channel'

PROD_DATA__JOBS__UPDATE_COLUMN = 'notificated'
PROD_DATA__JOBS__CONDITION_COLUMN = 'id'

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LLM_BASE_MODEL = 'gpt-4o-mini'

EMAIL_NOTIFICATION_CHUNK_SIZE = 16

URL_EXTRACT_PATTERN = r'https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])'
URL_REMOVAL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')
