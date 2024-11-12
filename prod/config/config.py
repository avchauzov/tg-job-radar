import json
import logging
import os
import sys

import openai
from telethon.sessions import StringSession
from telethon.sync import TelegramClient

from prod import OPENAI_API_KEY, RAW_DATA__TG_POSTS__NAME
from prod.utils.functions_common import get_correct_path, setup_logging
from prod.utils.functions_sql import get_table_columns
from prod.utils.functions_tg_api import create_session_string


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)

if OPENAI_API_KEY:
	OPENAI_CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY)

else:
	logging.error('OPENAI_API_KEY is not set.')
	OPENAI_CLIENT = None

config_path = get_correct_path('config/config.json')
try:
	with open(config_path) as file:
		CONFIG = json.load(file)

except (FileNotFoundError, json.JSONDecodeError) as error:
	logging.error(f'Error loading config: {error}')
	raise FileNotFoundError(f'Error loading config: {error}')

SOURCE_CHANNELS = CONFIG.get('source_channels', [])
PREFILTERING_WORDS = CONFIG.get('prefiltering_words', [])
DATA_COLLECTION_BATCH_SIZE = CONFIG.get('data_collection_batch_size', 32)

TG_STRING_SESSION = CONFIG.get('tg_string_session')
logging.debug(f'TG_STRING_SESSION: {TG_STRING_SESSION}')

TG_API_ID = os.getenv('TG_API_ID')
TG_API_HASH = os.getenv('TG_API_HASH')
if not TG_API_ID or not TG_API_HASH:
	logging.error("Environment variables 'TG_API_ID' and 'TG_API_HASH' must be set.")
	raise ValueError('Missing Telegram API credentials.')

if not TG_STRING_SESSION:
	with TelegramClient(StringSession(), TG_API_ID, TG_API_HASH) as client:
		create_session_string(client, CONFIG, config_path)
		sys.exit(0)

TG_CLIENT = TelegramClient(StringSession(TG_STRING_SESSION), TG_API_ID, TG_API_HASH)

RAW_DATA__TG_POSTS__COLUMNS = get_table_columns(RAW_DATA__TG_POSTS__NAME, to_exclude=[])

"""OPENAI_API_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


def load_json_into_dict(file_path):
	with open(file_path, 'r') as file:
		return json.load(file)


def replace_non_alpha_with_space(text):
	return re.sub(r'[^a-zA-Z]', ' ', text)


def remove_consecutive_spaces(text):
	return re.sub(r'\s+', ' ', text).strip()


def validate_keywords(keywords):
	return [remove_consecutive_spaces(replace_non_alpha_with_space(value)).strip().lower() for value in keywords]


KEYWORDS = load_json_into_dict('./keywords.json').get('keywords', [])
KEYWORDS = validate_keywords(KEYWORDS)

if len(KEYWORDS) == 0:
	raise "Keywords can't be loaded!"

KEYWORDS_LENGTH = {key: key.count(' ') + 1 for key in KEYWORDS}
MIN_NGRAM_LENGTH = min(list(KEYWORDS_LENGTH.values()))
MAX_NGRAM_LENGTH = max(list(KEYWORDS_LENGTH.values()))"""
