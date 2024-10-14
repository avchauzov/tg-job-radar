import asyncio  # Add this import

import html2text

from dev.scraping.functions import clean_text, iterate_lines
from prod.airflow.plugins.raw.functions import contains_job_keywords, extract_links
from prod.config.config import SOURCE_CHANNELS, TG_CLIENT
from prod.utils.functions_common import call_openai_api, get_html


async def main():
	await TG_CLIENT.start()
	
	for channel in SOURCE_CHANNELS:
		async for message in TG_CLIENT.iter_messages(  # TODO: filtering by date
				entity=channel,
				reverse=True
				):
			job_description = message.text
			
			if not contains_job_keywords(job_description):
				continue
			
			links = extract_links(job_description)
			
			# requests and scraperapi
			
			links = ['https://code-leap-ag.jobs.personio.de/job/901378?display=en&language=en&pid=50ee7896-f935-4fcf-8849-929d84268b78&it=JrTuaMJn8Trz-e855vVfhw&_ghcid=71e67ea4-f328-4404-ad0c-95e7b0e31743#apply']
			
			for link in links:
				html_content = get_html(link)  # selenium
				content = html2text.html2text(html_content)
				
				response = call_openai_api(content)
				
				my_meta = {
						"location"         : "Currently based in Indonesia, planning to move to Vietnam in the next few months. Open to jobs in Southeast Asia or fully remote roles (from anywhere), with a willingness to relocate globally.",
						"remote_status"    : "Prefer fully remote positions but open to hybrid or onsite roles if relocation assistance is provided.",
						"visa_requirements": "Russian citizen with no work visas for other countries."
						}
				
				# relocation
				
				print(f'{response.location}\n{response.remote_status}\n{response.visa_requirements}')
				
				'''for location, remote_status, visa_requirements in [[response.location, 'location',
				                                                    response.remote_status, 'remote_status',
				                                                    response.visa_requirements, 'visa_requirements']]:'''
			
			# filtering
			# relocation support and visa
			
			cleaned_text = clean_text(job_description)
			lines = iterate_lines(cleaned_text)
			
			'''for length in range(MIN_NGRAM_LENGTH, MAX_NGRAM_LENGTH + 1):
				keywords_subset = [key for key, value in KEYWORDS_LENGTH.items() if value == length]
				
				for line in lines:
					if ngram_search(line, keywords_subset, length):
						print(message.text)
						print()
						
						break  # remove URLs'''


async def run():
	async with client:
		await main()


if __name__ == '__main__':
	asyncio.run(run())

# single word search
