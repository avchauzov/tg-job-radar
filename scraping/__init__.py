import os

from dotenv import load_dotenv


load_dotenv()

TG_API_ID = os.getenv('TG_API_ID')
TG_API_HASH = os.getenv('TG_API_HASH')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')

SESSION_NAME = 'main'

KEYWORDS = ['ai', 'ml', 'data scientist']
KEYWORDS_LENGTH = {key: key.count(' ') + 1 for key in KEYWORDS}
MIN_NGRAM_LENGTH = min(list(KEYWORDS_LENGTH.values()))
MAX_NGRAM_LENGTH = max(list(KEYWORDS_LENGTH.values()))
