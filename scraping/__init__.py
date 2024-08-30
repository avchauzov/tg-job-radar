import os

from dotenv import load_dotenv

from utils.functions import load_json_into_dict, validate_keywords


load_dotenv()

TG_API_ID = os.getenv('TG_API_ID')
TG_API_HASH = os.getenv('TG_API_HASH')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')

SESSION_NAME = 'main'

KEYWORDS = load_json_into_dict('./keywords.json').get('keywords', [])
KEYWORDS = validate_keywords(KEYWORDS)

if len(KEYWORDS) == 0:
	raise "Keywords can't be loaded!"

KEYWORDS_LENGTH = {key: key.count(' ') + 1 for key in KEYWORDS}
MIN_NGRAM_LENGTH = min(list(KEYWORDS_LENGTH.values()))
MAX_NGRAM_LENGTH = max(list(KEYWORDS_LENGTH.values()))
