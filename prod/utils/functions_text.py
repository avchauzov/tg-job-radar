import re


def clean_job_description(text):
	text_cleaned = text.replace('\n', ' ')
	text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
	
	return text_cleaned
