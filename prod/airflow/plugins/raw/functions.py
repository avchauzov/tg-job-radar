import sys

from prod.utils.functions_common import normalize_url


sys.path.insert(0, '/home/job_search')

import logging
import re

from prod.config.config import PREFILTERING_WORDS


def contains_job_keywords(text):
	try:
		text_cleaned = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text).lower()
		text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
		
		return any(keyword in text_cleaned.split() for keyword in PREFILTERING_WORDS)
	
	except Exception as error:
		logging.error(f'Error in contains_job_keywords: {error}')
		return False


def extract_urls(text):
	try:
		url_pattern = r'https?://[^\s()]+(?:\([\w\d]+\)|[^\s,()])*(?<![.,?!])'
		urls = list(set(re.findall(url_pattern, text)))
		urls = [normalize_url(url) for url in urls]
		
		return sorted(list(set(urls)))
	
	except Exception as error:
		logging.error(f'Error in extract_links: {error}')
		return []
