import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse, urlunparse


def setup_logging(file_name):
	handler = RotatingFileHandler(f'{file_name}.log', maxBytes=10 * 1024 * 1024, backupCount=3)
	handler.setLevel(logging.INFO)
	
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	
	logging.getLogger().addHandler(handler)
	logging.getLogger().setLevel(logging.INFO)


def get_channel_link_header(entity):
	if hasattr(entity, 'username') and entity.username:
		link_header = f'https://t.me/{entity.username}/'
	
	else:
		link_header = f'https://t.me/c/{entity.id}/'
	
	return link_header


def normalize_url(url):
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
