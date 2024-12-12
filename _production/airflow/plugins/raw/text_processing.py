import logging
import re

from _production.config.config import DESIRED_KEYWORDS


def contains_keywords(text):
    try:
        text_cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]+|\s+", " ", text).lower().strip()
        result = any(
            keyword in text_cleaned.split() for keyword in DESIRED_KEYWORDS
        )  # we have a lot of matches on 'engineer', maybe we can improve?

        # logging.debug(f"Processed text: {text_cleaned[: 128]}...")
        return result

    except Exception as error:
        logging.error(f"Error in contains_job_keywords: {error}\n{text}")
        return False
