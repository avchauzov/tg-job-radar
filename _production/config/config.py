import json
import logging
import os
import sys

from _production.config.config_hidden import MATCH_SCORE_THRESHOLD
from _production.utils.functions_sql import generate_db_mappings
import openai
from telethon.sessions import StringSession
from telethon.sync import TelegramClient

from _production import OPENAI_API_KEY, PROD_DATA__JOBS, STAGING_DATA__POSTS
from _production.utils.functions_common import get_correct_path, setup_logging
from _production.utils.functions_tg_api import create_session_string


file_name = __file__[:-3]
setup_logging(file_name)

if OPENAI_API_KEY:
    OPENAI_CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY)

else:
    logging.error("OPENAI_API_KEY is not set.")
    OPENAI_CLIENT = None

config_path = get_correct_path("config/config.json")
try:
    with open(config_path) as file:
        CONFIG = json.load(file)

except (FileNotFoundError, json.JSONDecodeError) as error:
    logging.error(f"Error loading config: {error}")
    raise FileNotFoundError(f"Error loading config: {error}")

SOURCE_CHANNELS = CONFIG.get("source_channels", [])
PREFILTERING_WORDS = CONFIG.get("prefiltering_words", [])
DATA_COLLECTION_BATCH_SIZE = CONFIG.get("data_collection_batch_size", 32)

TG_STRING_SESSION = CONFIG.get("tg_string_session")
logging.debug(f"TG_STRING_SESSION: {TG_STRING_SESSION}")

TG_API_ID = os.getenv("TG_API_ID")
TG_API_HASH = os.getenv("TG_API_HASH")
if not TG_API_ID or not TG_API_HASH:
    logging.error("Environment variables 'TG_API_ID' and 'TG_API_HASH' must be set.")
    raise ValueError("Missing Telegram API credentials.")

if not TG_STRING_SESSION:
    with TelegramClient(StringSession(), TG_API_ID, TG_API_HASH) as client:
        create_session_string(client, CONFIG, config_path)
        sys.exit(0)

TG_CLIENT = TelegramClient(StringSession(TG_STRING_SESSION), TG_API_ID, TG_API_HASH)

LLM_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Generate DB mappings and column definitions
try:
    DB_MAPPINGS = generate_db_mappings()

    # Extract column definitions
    RAW_DATA__TG_POSTS__COLUMNS = list(DB_MAPPINGS["schemas"]["raw"].keys())
    STAGING_DATA__POSTS__COLUMNS = list(DB_MAPPINGS["schemas"]["staging"].keys())
    PROD_DATA__JOBS__COLUMNS = [
        col
        for col in list(DB_MAPPINGS["schemas"]["prod"].keys())
        if col != "notificated"
    ]

    # Extract mappings for data movement
    RAW_TO_STAGING_MAPPING = DB_MAPPINGS["mappings"]["raw_to_staging"]
    STAGING_TO_PROD_MAPPING = DB_MAPPINGS["mappings"]["staging_to_prod"]

    # SQL query components
    RAW_TO_STAGING__SELECT = ", ".join(RAW_TO_STAGING_MAPPING["source_columns"])
    STAGING_TO_PROD__SELECT = ", ".join(STAGING_TO_PROD_MAPPING["source_columns"])

    RAW_TO_STAGING__WHERE = f"id not in (select id from {STAGING_DATA__POSTS})"
    STAGING_TO_PROD__WHERE = f"""
        post_structured != '{{}}' and 
        id not in (select id from {PROD_DATA__JOBS})
    """

except Exception as error:
    logging.error(f"Failed to generate DB mappings: {error}")
    raise
