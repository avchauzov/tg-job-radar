import logging
import re


def clean_job_description(text):
	try:
		text_cleaned = text.replace('\n', ' ')
		text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
		
		return text_cleaned
	
	except Exception as error:
		logging.error(f'Error cleaning job description: {error}')
		return text
