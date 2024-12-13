import logging
import re

from _production.config.config import DESIRED_KEYWORDS


def contains_keywords(text):
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)}")

        text_cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]+|\s+", " ", text).lower().strip()

        # Using word boundaries for exact word matching
        return any(
            re.search(r"\b" + re.escape(keyword) + r"\b", text_cleaned)
            for keyword in DESIRED_KEYWORDS
        )

    except re.error as regex_error:
        logging.error(
            f"Regex error in contains_keywords: {regex_error}\nInput text: {text[:100]}..."
        )
        raise

    except Exception as error:
        logging.error(
            f"Unexpected error in contains_keywords: {error}\nInput text: {text[:100]}..."
        )
        raise
