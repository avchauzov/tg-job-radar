import json
import logging
import os
import sys

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
    logging.debug("OpenAI API key validation failed", exc_info=True)
    raise ValueError(
        "OpenAI API key is not set. "
        "Please set the OPENAI_API_KEY environment variable."
    )

config_path = get_correct_path("config/config.json")
try:
    with open(config_path) as file:
        CONFIG = json.load(file)

except (FileNotFoundError, json.JSONDecodeError) as error:
    logging.debug(f"Configuration loading failed. Details: {str(error)}", exc_info=True)

    raise FileNotFoundError(
        f"Failed to load configuration. "
        f"Path: {config_path}, "
        f"Error type: {error.__class__.__name__}, "
        f"Original error: {str(error)}"
    ) from error

SOURCE_CHANNELS = CONFIG.get("source_channels", [])
DESIRED_KEYWORDS = CONFIG.get("prefiltering_words", [])
DATA_BATCH_SIZE = CONFIG.get("data_batch_size", 32)

MATCHING_CONFIG = CONFIG.get("matching_config", {})
CV_DOC_ID = MATCHING_CONFIG.get("cv_doc_id", "google_doc_id")
MATCH_SCORE_THRESHOLD = MATCHING_CONFIG.get("match_score_threshold", 70)

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
        post_structured IS NOT NULL 
        AND post_structured != '{{}}'::jsonb 
        AND id NOT IN (SELECT id FROM {PROD_DATA__JOBS})
    """


except Exception as error:
    logging.error(f"Failed to generate DB mappings: {error}")
    raise
