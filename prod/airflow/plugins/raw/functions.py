import re

from prod.config.config import PREFILTERING_WORDS


def contains_job_keywords(text):
	text_cleaned = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text).lower()
	text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
	
	return any(keyword in text_cleaned.split() for keyword in PREFILTERING_WORDS)


def extract_links(text):
	url_pattern = r'(https?://[^\s]+)'
	email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
	telegram_pattern = r'@[\w_]+'
	
	links = re.findall(url_pattern, text)
	emails = re.findall(email_pattern, text)
	telegram_channels = re.findall(telegram_pattern, text)
	
	all_links = links + emails + telegram_channels
	
	return sorted(set(all_links))
