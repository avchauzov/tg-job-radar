import logging
import re
from difflib import SequenceMatcher
from typing import List, Pattern

from _production import TEXT_SIMILARITY_THRESHOLD


def contains_keywords(
    text: str, keywords: List[str], patterns: List[Pattern[str] | None] | None = None
) -> bool:
    """Check if any keywords appear in the given text.

    Args:
        text: The input text to search for keywords
        keywords: List of keywords to search for
        patterns: Pre-compiled regex patterns (optional)

    Returns:
        bool: True if any keyword is found
    """
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)}")

        if patterns is None:
            patterns = [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]

        text_cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]+|\s+", " ", text).lower().strip()

        return any(pattern and pattern.search(text_cleaned) for pattern in patterns)

    except re.error as regex_error:
        logging.error(f"Regex error in contains_keywords: {regex_error}")
        raise
    except Exception as error:
        logging.error(f"Unexpected error in contains_keywords: {error}")
        raise


def clean_job_description(text):
    try:
        if not isinstance(text, str):
            logging.warning(
                f"Expected a string, but got {type(text)}\n{text}\nReturning input as string."
            )
            return str(text)

        text_cleaned = text.replace("\n", " ")
        text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()

        return text_cleaned

    except Exception as error:
        logging.warning(f"Error cleaning job description: {error}\n{text}")
        return text


def text_similarity(text1, text2):
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def is_duplicate_post(
    new_post, existing_posts, similarity_threshold=TEXT_SIMILARITY_THRESHOLD
):
    """Check if post is similar to existing posts."""
    for existing_post in existing_posts:
        similarity_score = text_similarity(new_post, existing_post)
        if similarity_score >= similarity_threshold:
            return True, existing_post, similarity_score
    return False, None, 0.0
