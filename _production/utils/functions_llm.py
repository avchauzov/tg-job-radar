import logging
import time

from pydantic import BaseModel

from _production import LLM_BASE_MODEL
from _production.config.config import OPENAI_CLIENT


class JobURLResult(BaseModel):
	url: list[str]
	is_direct_job_description: list[int]


def filter_job_urls(description, url_list, max_retries=3, sleep_time=10):
	for attempt in range(max_retries):
		try:
			response = OPENAI_CLIENT.beta.chat.completions.parse(
					model=LLM_BASE_MODEL,
					messages=[
							{
									'role'   : 'system',
									'content': (
											'Task:\n'
											'For each URL in the provided list, determine if it likely leads to a direct job description (1) '
											'or not (0). A direct job description URL should contain specific job information and an option '
											'to apply directly. Exclude URLs leading to general company pages, career listing pages, or '
											'"other jobs" pages. Return 1 for direct job descriptions and 0 otherwise. '
											'Ensure the output order matches the input URL order.')
									},
							{'role': 'user', 'content': f'Job post: {description}\n\nURLs:\n{url_list}'}
							], temperature=0.0,
					response_format=JobURLResult
					)
			
			filtered_urls = [url for url, mask in zip(url_list, response.choices[0].message.parsed.is_direct_job_description) if mask]
			return filtered_urls
		
		except Exception as error:
			if 'Too Many Requests' in str(error):
				logging.warning(f'Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})')
				time.sleep(sleep_time)
			
			else:
				logging.error(f'Error filtering job URLs: {error}')
				return url_list
	
	logging.error(f'Failed to filter URLs after {max_retries} attempts')
	return url_list


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	
	job_description = 'We\'re hiring a Software Engineer to join our team. Apply now!'
	test_urls = [
			'https://example.com/job/software-engineer-123',
			'https://example.com/careers',
			'https://example.com/company/about',
			'https://example.com/jobs/engineering/software-engineer'
			]
	
	filtered_urls = filter_job_urls(job_description, test_urls)
	print(filtered_urls)
