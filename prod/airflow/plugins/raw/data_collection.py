import asyncio
import datetime
import logging
import os

from prod import DATE_FORMAT, RAW_DATA__TG_POSTS_COLUMNS, RAW_DATA__TG_POSTS_CONFLICT, RAW_DATA__TG_POSTS_NAME
from prod.airflow.plugins.raw.functions import contains_job_keywords, extract_links
from prod.config.config import SOURCE_CHANNELS, TG_CLIENT
from prod.utils.functions_common import get_channel_link_header, setup_logging
from prod.utils.functions_sql import batch_insert_to_db, fetch_from_db
from prod.utils.functions_text import clean_job_description


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


async def scrape_tg_function():
	await TG_CLIENT.start()
	
	logging.info('Started scraping process.')
	
	for channel in SOURCE_CHANNELS:
		logging.info(f'Starting to scrape channel: {channel}')
		
		_, last_date = fetch_from_db(RAW_DATA__TG_POSTS_NAME, 'max(date) as date', order_condition='date desc')
		last_date = last_date[0][0]
		
		if not last_date:
			last_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
		
		try:
			entity = await TG_CLIENT.get_entity(channel)
			link_header = await get_channel_link_header(entity)
			
			async for message in TG_CLIENT.iter_messages(
					entity=channel,
					reverse=True,
					offset_date=last_date
					):
				job_description, date = message.text, message.date
				
				if not job_description or not date:
					logging.warning(f'Skipping message with missing job description or date in channel: {channel}')
					continue
				
				message_link = f'{link_header}{message.id}'
				job_description_cleaned = clean_job_description(job_description)
				
				if not contains_job_keywords(job_description_cleaned):
					logging.info(f'Skipping message (no keywords) in channel: {channel}. Content: {job_description_cleaned[:256]}')
					continue
				
				links = extract_links(job_description_cleaned)
				logging.info(f'Extracted links: {links} from message ID: {message.id} in channel: {channel}')
				
				result = {
						'id'        : int(message.id),
						'channel'   : channel,
						'post'      : job_description,
						'date'      : date,
						'created_at': datetime.datetime.utcnow().strftime(DATE_FORMAT),
						'post_link' : message_link,
						'links'     : links
						}
				
				logging.info(f'Inserting message ID: {message.id} from channel: {channel} into database.')
				batch_insert_to_db(RAW_DATA__TG_POSTS_NAME, RAW_DATA__TG_POSTS_COLUMNS, RAW_DATA__TG_POSTS_CONFLICT, [result])
		
		except Exception as e:
			logging.error(f'Error occurred while scraping channel: {channel}. Error: {error}')
	
	logging.info('Scraping process completed.')


async def scrape_tg():
	async with TG_CLIENT:
		await scrape_tg_function()


if __name__ == '__main__':
	asyncio.run(scrape_tg())
