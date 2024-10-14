import json
import os

from telethon import TelegramClient

from prod.lib.utils import get_correct_path


try:
	with open(get_correct_path('config/config.json')) as file:
		CONFIG = json.load(file)

except FileNotFoundError as error:
	raise FileNotFoundError(f'Error loading config: {error}')

SOURCE_CHANNELS = CONFIG.get('source_channels', [])
PREFILTERING_WORDS = CONFIG.get('prefiltering_words', [])

TG_CLIENT = TelegramClient(os.getenv('TG_SESSION_NAME'), os.getenv('TG_API_ID'), os.getenv('TG_API_HASH'))

'''OPENAI_API_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


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
MAX_NGRAM_LENGTH = max(list(KEYWORDS_LENGTH.values()))'''
