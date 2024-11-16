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

RAW_DATA__TG_POSTS__NAME = 'raw_data.tg_posts'
RAW_DATA__TG_POSTS__CONFLICT = ['id']

STAGING_DATA__JOBS__NAME = 'staging_data.posts'
RAW_DATA_TO_STAGING_DATA__WHERE_CONDITION = f'id not in (select id from {STAGING_DATA__JOBS__NAME})'

PROD_DATA__JOBS__NAME = 'prod_data.jobs'
STAGING_DATA_TO_PROD_DATA__WHERE_CONDITION = f'id not in (select id from {PROD_DATA__JOBS__NAME})'

PROD_DATA__JOBS__SELECT_CONDITION = 'id, channel, post, post_link'
PROD_DATA__JOBS__WHERE_CONDITION = 'notificated = false'
PROD_DATA__JOBS__ORDER_BY_CONDITION = 'date asc, channel, post'

PROD_DATA__UPDATE_COLUMN__NAME = 'notificated'
PROD_DATA__CONDITION_COLUMN__NAME = 'id'

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LLM_BASE_MODEL = 'gpt-4o-mini'

EMAIL_NOTIFICATION_CHUNK_SIZE = 16

URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)')
