import logging
import re
from urllib.parse import urlparse, urlunparse


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

# TODO: revise logging format
# TODO: more 'raise' than skip
# TODO: airflow: notifications to email

def normalize_url(url):
	try:
		logging.info(f'Normalizing URL: {url}')
		parsed_url = urlparse(url)
		
		normalized_path = parsed_url.path.lstrip('/').rstrip('/')
		normalized_url = urlunparse(
				parsed_url._replace(
						scheme=parsed_url.scheme.lower(),
						netloc=parsed_url.netloc.lower(),
						path=normalized_path
						)
				)
		
		return normalized_url
	
	except ValueError as error:
		logging.warning(f'Invalid URL provided for normalization: {url} | Error: {error}')
		return None
