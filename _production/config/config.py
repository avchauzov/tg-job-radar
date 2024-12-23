import json
import logging
import sys
from functools import lru_cache

import openai
from telethon.sessions import StringSession
from telethon.sync import TelegramClient

from _production import (
    OPENAI_API_KEY,
    TELEGRAM,
)
from _production.utils.common import get_correct_path, setup_logging
from _production.utils.tg import create_session_string

setup_logging(__file__[:-3])

# OpenAI client initialization
try:
    if OPENAI_API_KEY:
        OPENAI_CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY)
        openai._utils._logs.logger.setLevel(logging.WARNING)
        openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

    else:
        logging.error("OpenAI API key not set")
        raise ValueError(
            "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
        )

except Exception:
    logging.error("Failed to initialize OpenAI client", exc_info=True)
    raise


@lru_cache(maxsize=1)
def load_config():
    """
    Load configuration from both environment variables and config file.
    Uses caching to prevent repeated file reads.
    """
    config_path = get_correct_path("config/config.json")

    try:
        with open(config_path) as file:
            file_config = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}, using defaults")
        file_config = {}

    return file_config, config_path


CONFIG, CONFIG_PATH = load_config()
SOURCE_CHANNELS = CONFIG.get("source_channels", [])
DESIRED_KEYWORDS = CONFIG.get("prefiltering_words", [])

TG_STRING_SESSION = CONFIG.get("tg_string_session")
logging.debug(f"TG_STRING_SESSION: {TG_STRING_SESSION}")


if not TELEGRAM["API_ID"] or not TELEGRAM["API_HASH"]:
    logging.error("Telegram API credentials not set")
    raise ValueError(
        "Missing Telegram API credentials. Both TG_API_ID and TG_API_HASH must be set."
    )

if not TG_STRING_SESSION:
    with TelegramClient(
        StringSession(), int(TELEGRAM["API_ID"]), TELEGRAM["API_HASH"]
    ) as client:
        create_session_string(client, CONFIG, CONFIG_PATH)
        sys.exit(0)

TG_CLIENT = TelegramClient(
    StringSession(TG_STRING_SESSION), int(TELEGRAM["API_ID"]), TELEGRAM["API_HASH"]
)
