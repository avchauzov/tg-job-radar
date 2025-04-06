"""Configuration module for loading and managing application settings.

Handles configuration from environment variables, JSON files, and database sources,
providing centralized access to application settings including API credentials,
channel sources, and operational parameters.
"""

import json
import logging
import sys
from functools import lru_cache

from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from telethon.sessions import StringSession
from telethon.sync import TelegramClient

from _production import (
    CUSTOM_MODEL_BASE_URL,
    DATABASE,
    INFLUXDB,
    SERVER_URL,
    TELEGRAM,
)
from _production.utils.common import get_correct_path, setup_logging
from _production.utils.sql import fetch_from_db
from _production.utils.tg import create_session_string

setup_logging(__file__[:-3])


# Initialize InfluxDB client
try:
    if all(INFLUXDB.values()):
        INFLUXDB_CLIENT = InfluxDBClient(
            url=SERVER_URL + ":8086",
            token=INFLUXDB["TOKEN"],
            org=INFLUXDB["ORG"],
        )
        INFLUXDB_WRITE_API = INFLUXDB_CLIENT.write_api(write_options=SYNCHRONOUS)
    else:
        logging.warning(
            "InfluxDB configuration is incomplete. Metrics will not be stored."
        )
        INFLUXDB_CLIENT = None
        INFLUXDB_WRITE_API = None
except Exception as error:
    logging.error(f"Failed to initialize InfluxDB client: {error}")
    INFLUXDB_CLIENT = None
    INFLUXDB_WRITE_API = None

# No Anthropic client initialization, only using custom model
ANTHROPIC_CLIENT = None
logging.info("Using custom model server for LLM operations")

# Custom model server configuration
CUSTOM_MODEL_CONFIG = {
    "BASE_URL": CUSTOM_MODEL_BASE_URL,  # URL for the local model server
    "TIMEOUT": 300,  # Request timeout in seconds (increased from 120 to 300)
    "DEFAULT_TEMPERATURE": 0.0,  # Default temperature for deterministic outputs
    "DEFAULT_MAX_TOKENS": 1024,  # Default maximum tokens to generate
}


@lru_cache(maxsize=1)
def load_config():
    """Load configuration from both environment variables and config file.

    Uses caching to prevent repeated file reads.
    """
    config_path = get_correct_path("config/config.json")

    try:
        with open(config_path) as file:
            file_config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in config file: {error}") from error
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}, using defaults")
        file_config = {}

    # Fetch channels from database
    try:
        _, db_channels = fetch_from_db(
            "raw_data.current_channels",
            select_condition="username",
            where_condition="channel_group = 'jobs' and username is not null",
            database=DATABASE["DB_TELEFLOW_NAME"],
        )
        # Convert tuple results to list of channel names
        db_channel_list = [channel[0] for channel in db_channels]

        # Set source_channels directly from database
        file_config["source_channels"] = db_channel_list
        logging.info(f"Added {len(db_channel_list)} channels from database")
    except Exception as error:
        logging.error(f"Failed to fetch channels from database: {error}")
        # Initialize with empty list if database fetch fails
        file_config["source_channels"] = []

    return file_config, config_path


CONFIG, CONFIG_PATH = load_config()
SOURCE_CHANNELS = CONFIG.get("source_channels", [])
DESIRED_KEYWORDS = CONFIG.get("prefiltering_tokens", [])

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
