import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv


def setup_logging(file_name):
	handler = RotatingFileHandler(f'{file_name}.log', maxBytes=10 * 1024 * 1024, backupCount=3)
	handler.setLevel(logging.INFO)
	
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	
	if not logger.hasHandlers():
		logger.addHandler(handler)


def get_channel_link_header(entity):
	if hasattr(entity, 'username') and entity.username:
		return f'https://t.me/{entity.username}/'
	
	elif hasattr(entity, 'id'):
		return f'https://t.me/c/{entity.id}/'
	
	logging.warning("Entity lacks 'username' and 'id' attributes")
	return None


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


def get_correct_path(file_name):
	load_dotenv()
	base_dir = os.getenv('CONFIG_BASE_DIR')
	local_dir = os.getenv('CONFIG_BASE_LOCAL_DIR')
	logging.info(f'Starting to resolve file path for: {file_name}')
	
	base_path = Path(base_dir).resolve() if base_dir else None
	local_path = Path(local_dir).resolve() if local_dir else None
	
	for path in [base_path, local_path, Path(__file__).resolve().parent]:
		if path and path.exists():
			file_path = path / file_name
			logging.info(f'File path resolved to: {file_path}')
			return file_path
	
	logging.error(f'File not found at any checked path for {file_name}')
	raise FileNotFoundError(f'File not found for {file_name}')


def generate_hash(input_string):
	hash_object = hashlib.sha256(input_string.encode())
	return hash_object.hexdigest()


'''def get_html(url):
	try:
		response = requests.get(url)
		
		if response.status_code == 200:
			return response.text
		
		else:
			return f'Failed to retrieve content. Status code: {response.status_code}'
	
	except requests.exceptions.RequestException as error:
		return f'An error occurred: {str(error)}'


def call_openai_api(content):
	class JobDescription(BaseModel):
		title: str
		company_name: str
		location: str
		remote_status: str
		visa_requirements: str
		relocation_support: str
		description: str
	
	try:
		response = OPENAI_API_CLIENT.beta.chat.completions.parse(
				model='gpt-4o',
				messages=[
						{
								"role": "user", 'content': f"From the following job posting, extract the key details:\n"
								                           "- Job Title\n"
								                           "- Company Name\n"
								                           "- Location\n"
								                           "- Remote Status (fully remote, hybrid, or on-site; any location restrictions)\n"
								                           "- Visa Requirements (visa sponsorship or local applicants only)\n"
								                           "- Relocation Support (is relocation offered or restricted)\n"
								                           "- Job Description\n\n"
								                           f"Job Posting:\n{content}"
								}
						],
				temperature=0.0,
				response_format=JobDescription
				)
		
		return response.choices[0].message.parsed
	
	except openai.OpenAIError as error:
		return f'OpenAI API error: {str(error)}'
	
	except Exception as error:
		return f'An error occurred: {str(error)}'


def estimate_match(content):
	class FitEstimation(BaseModel):
		location: str
		remote_status: str
		visa_requirements: str
	
	try:
		response = OPENAI_API_CLIENT.beta.chat.completions.parse(
				model='gpt-4o',
				messages=[
						{
								"role": "user", 'content': f"From the following job posting, extract the key details:\n"
								                           "- Job Title\n"
								                           "- Company Name\n"
								                           "- Location\n"
								                           "- Remote Status (fully remote, hybrid, or on-site; any location restrictions)\n"
								                           "- Visa Requirements (visa sponsorship or local applicants only)\n"
								                           "- Relocation Support (is relocation offered or restricted)\n"
								                           "- Job Description\n\n"
								                           f"Job Posting:\n{content}"
								}
						],
				temperature=0.0,
				response_format=JobDescription
				)
		
		return response.choices[0].message.parsed
	
	except openai.OpenAIError as error:
		return f'OpenAI API error: {str(error)}'
	
	except Exception as error:
		return f'An error occurred: {str(error)}'''
