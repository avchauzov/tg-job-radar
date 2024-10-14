import os

from dotenv import load_dotenv


load_dotenv()

POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_NAME = os.getenv('POSTGRES_NAME')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASS = os.getenv('POSTGRES_PASS')

PHONE_NUMBER = os.getenv('PHONE_NUMBER')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

RAW_DATA__TG_POSTS_NAME = 'raw_data.tg_posts'
RAW_DATA__TG_POSTS_CONFLICT = ['id', 'channel']
RAW_DATA__TG_POSTS_COLUMNS = ['id', 'channel', 'post', 'date', 'created_at', 'post_link', 'links']

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
