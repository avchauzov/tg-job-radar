import datetime
import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv


def setup_logging(file_name: str) -> logging.Logger:
    """Configure logging with file rotation and standard formatting.

    Args:
        file_name (str): Base name for the log file (will be appended with .log)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure root logger and file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Use simple log filename without timestamp
    log_path = f"{file_name}.log"

    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
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

    # Add console handler for development visibility
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_correct_path(file_name: str) -> Path:
    """Locate a file by searching through configured directories.

    Args:
        file_name (str): Name of the file to find

    Returns:
        Path: Resolved path to the file

    Raises:
        FileNotFoundError: If file cannot be found in any configured location
    """
    load_dotenv()
    search_paths = [
        os.getenv("CONFIG_PATH"),
        os.getenv("CONFIG_LOCAL_PATH"),
    ]

    # Filter None values
    valid_paths = [p for p in search_paths if p]

    for path in valid_paths:
        file_path = Path(path)
        if file_path.exists():
            logging.info(f"File path resolved to: {file_path}")
            return file_path

    logging.error(f"File not found at any checked path for {file_name}")
    raise FileNotFoundError(f"Could not find file: {file_name}")


def generate_hash(input_string: str) -> str:
    """Generate SHA-256 hash of input string.

    Args:
        input_string (str): String to hash

    Returns:
        str: Hexadecimal representation of hash
    """
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()


def process_date(date: Union[datetime.datetime, Any]) -> Optional[datetime.datetime]:
    """Convert date to UTC datetime without timezone info."""
    try:
        if isinstance(date, datetime.datetime) and date.tzinfo is not None:
            return date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return date if isinstance(date, datetime.datetime) else None
    except Exception as e:
        logging.error(f"Error processing date {date}: {str(e)}")
        return None
