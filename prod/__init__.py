import os

from dotenv import load_dotenv


load_dotenv()

POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_NAME = os.getenv('POSTGRES_NAME')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASS = os.getenv('POSTGRES_PASS')

PHONE_NUMBER = os.getenv('PHONE_NUMBER')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

RAW_DATA__TG_POSTS__NAME = 'raw_data.tg_posts'
RAW_DATA__TG_POSTS__CONFLICT = ['id', 'channel']

PROD_DATA__JOBS__NAME = 'prod_data.jobs'
PROD_DATA__JOBS__CONFLICT = []

RAW_DATA_TO_PROD_DATA_SELECT_CONDITION = '*'
RAW_DATA_TO_PROD_DATA_WHERE_CONDITION = f'(id, channel) not in (select id, channel from {PROD_DATA__JOBS__NAME})'

PROD_DATA__JOBS__SELECT_CONDITION = 'channel, post, post_link'
PROD_DATA__JOBS__WHERE_CONDITION = 'notificated = false'
PROD_DATA__JOBS__ORDER_BY_CONDITION = 'date asc, channel, post'

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LLM_BASE_MODEL = 'gpt-4o-mini'

SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
