import logging
import re
from difflib import SequenceMatcher

from _production import PROBLEM_CHARS, TEXT_SIMILARITY_THRESHOLD


def contains_keywords(text: str, keywords: list[str] = []) -> bool:
    """Check if any keywords appear in the given text.

    Args:
        text (str): The input text to search for keywords
        keywords (List[str], optional): List of keywords to search for. Defaults to empty list.

    Returns:
        bool: True if any keyword is found in the text, False otherwise

    Raises:
        TypeError: If text is not a string
        re.error: If there's an error in regex pattern
    """
    try:
        text_cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]+|\s+", " ", text).lower().strip()

        return any(keyword in text_cleaned for keyword in keywords)

    except re.error as regex_error:
        logging.error(f"Regex error in contains_keywords: {regex_error}")
        raise
    except Exception as error:
        logging.error(f"Unexpected error in contains_keywords: {error}")
        raise


def clean_text(text):
    try:
        if not isinstance(text, str):
            logging.warning(
                f"Expected a string, but got {type(text)}\n{text}\nReturning input as string."
            )
            return str(text)

        for char in PROBLEM_CHARS:
            text = text.replace(char, " ")

        text = re.sub(r"\s+", " ", text).strip()

        return text

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
