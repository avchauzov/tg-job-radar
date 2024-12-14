import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv


def setup_logging(file_name):
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Configure file handler
    handler = RotatingFileHandler(
        f"{file_name}.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setLevel(logging.INFO)

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Configure module logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Add handler if not already present
    if not logger.hasHandlers():
        logger.addHandler(handler)

    # Suppress common noisy loggers if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_correct_path(file_name):
    try:
        load_dotenv()
        base_dir = os.getenv("CONFIG_BASE_DIR")
        local_dir = os.getenv("CONFIG_BASE_LOCAL_DIR")
        logging.info(f"Starting to resolve file path for: {file_name}")

        base_path = Path(base_dir).resolve() if base_dir else None
        local_path = Path(local_dir).resolve() if local_dir else None

        for path in [base_path, local_path, Path(__file__).resolve().parent]:
            if path and path.exists():
                file_path = path / file_name
                logging.info(f"File path resolved to: {file_path}")
                return file_path

        logging.error(f"File not found at any checked path for {file_name}")
        raise FileNotFoundError(f"Could not find file: {file_name}")

    except Exception:
        logging.error(f"Error resolving path for {file_name}", exc_info=True)
        raise


def generate_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()
