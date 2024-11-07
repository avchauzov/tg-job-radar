import logging
import re


def clean_job_description(text):
	try:
		
		if not isinstance(text, str):
			logging.warning(f'Expected a string, but got {type(text)}. Returning input as-is.')
			return text
		
		text_cleaned = text.replace('\n', ' ')
		text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
		
		return text_cleaned
	
	except Exception as error:
		logging.warning(f'Error cleaning job description: {error}')
		return text
