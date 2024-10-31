import sys


sys.path.insert(0, '/home/job_search')

import datetime
import logging
import os

from prod import RAW_DATA__TG_POSTS_COLUMNS, RAW_DATA__TG_POSTS_CONFLICT, RAW_DATA__TG_POSTS_NAME
from prod.airflow.plugins.raw.functions import contains_job_keywords, extract_urls
from prod.config.config import DATA_COLLECTION_BATCH_SIZE, SOURCE_CHANNELS, TG_CLIENT
from prod.utils.functions_common import get_channel_link_header, setup_logging
from prod.utils.functions_sql import batch_insert_to_db, fetch_from_db
from prod.utils.functions_text import clean_job_description


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def scrape_tg(tg_client):
	with TG_CLIENT as tg_client:
		logging.info('Started scraping process.')
		
		_, last_date = fetch_from_db(RAW_DATA__TG_POSTS_NAME, 'channel, max(date) as date', group_by_condition='channel', order_condition='date desc')
		last_date_dict = dict(last_date)
		
		for channel in SOURCE_CHANNELS:
			logging.info(f'Starting to scrape channel: {channel}')
			
			last_date = last_date_dict.get(channel)
			if not last_date:
				last_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
				logging.info(f'No previous records found. Starting from 30 days ago: {last_date}')
			
			try:
				entity = tg_client.get_entity(channel)
				link_header = get_channel_link_header(entity)
				
				results, results_count = [], 0
				for message in tg_client.iter_messages(
						entity=channel,
						reverse=True,
						offset_date=last_date
						):
					job_description, date = message.text, message.date
					
					if not job_description or not date:
						logging.info(f'Skipping message with missing job description or date in channel: {channel}')
						continue
					
					if isinstance(date, datetime.datetime) and date.tzinfo is not None:
						date = date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
					
					message_link = f'{link_header}{message.id}'
					job_description_cleaned = clean_job_description(job_description)
					
					if not contains_job_keywords(job_description_cleaned):
						logging.info(f'Skipping message (no keywords) in channel: {channel}. Content: {job_description_cleaned[:256]}')
						continue
					
					urls = extract_urls(job_description_cleaned)
					logging.info(f'Extracted urls: {urls} from message ID: {message.id} in channel: {channel}')
					
					result = {
							'id'        : int(message.id),
							'channel'   : channel,
							'post'      : job_description,
							'date'      : date if isinstance(date, datetime.datetime) else None,
							'created_at': datetime.datetime.utcnow(),
							'post_link' : message_link,
							'urls'      : urls,
							}
					
					results.append(result)
					
					if len(results) == DATA_COLLECTION_BATCH_SIZE:
						batch_insert_to_db(RAW_DATA__TG_POSTS_NAME, RAW_DATA__TG_POSTS_COLUMNS, RAW_DATA__TG_POSTS_CONFLICT, results)
						logging.info(f'Inserting batch of {len(results)} messages into database.')
						
						results_count += len(results)
						results = []
				
				if results:
					batch_insert_to_db(RAW_DATA__TG_POSTS_NAME, RAW_DATA__TG_POSTS_COLUMNS, RAW_DATA__TG_POSTS_CONFLICT, results)
					logging.info(f'Inserting batch of {len(results)} messages into database.')
					
					results_count += len(results)
				
				logging.info(f'Added {results_count} posts!')
			
			except Exception as error:
				logging.error(f'Error occurred while scraping channel: {channel}. Error: {error}')
		
		logging.info('Scraping process completed.')


if __name__ == '__main__':
	with TG_CLIENT as tg_client:
		scrape_tg(tg_client)
