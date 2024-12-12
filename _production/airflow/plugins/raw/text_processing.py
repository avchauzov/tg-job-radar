import logging
import re

from _production.config.config import DESIRED_KEYWORDS


def contains_keywords(text):
    try:
        text_cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]+|\s+", " ", text).lower().strip()

        # Using word boundaries for exact word matching
        return any(
            re.search(r"\b" + re.escape(keyword) + r"\b", text_cleaned)
            for keyword in DESIRED_KEYWORDS
        )

    except Exception as error:
        logging.error(f"Error in contains_job_keywords: {error}\n{text}")
        return False
