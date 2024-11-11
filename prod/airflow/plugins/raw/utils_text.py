import logging
import re

from prod.config.config import PREFILTERING_WORDS
from prod.utils.functions_common import normalize_url


URL_PATTERN = r'https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])'


def contains_job_keywords(text):
	try:
		text_cleaned = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]+|\s+', ' ', text).lower().strip()
		result = any(keyword in text_cleaned.split() for keyword in PREFILTERING_WORDS)
		
		logging.debug(f'Processed text: {text_cleaned[: 128]}...')
		return result
	
	except Exception as error:
		logging.error(f'Error in contains_job_keywords: {error}')
		return False


def extract_urls(text):
	try:
		urls = list(set(re.findall(URL_PATTERN, text)))
		urls = [normalize_url(url) for url in urls]
		
		logging.debug(f'Extracted URLs: {urls}')
		return sorted(list(set(urls)))
	
	except Exception as error:
		logging.error(f'Error in extract_urls: {error}')
		return []
