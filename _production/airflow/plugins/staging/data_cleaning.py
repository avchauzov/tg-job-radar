import sys


sys.path.insert(0, '/home/job_search')

import logging
import pandas as pd

from _production import RAW_DATA__TG_POSTS, RAW_TO_STAGING__WHERE, STAGING_DATA__POSTS
from _production.utils.functions_llm import job_description_detection
from _production.config.config import RAW_TO_STAGING__SELECT, STAGING_DATA__POSTS__COLUMNS
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import batch_insert_to_db, fetch_from_db


file_name = __file__[: -3]
setup_logging(file_name)


def clean_and_move_data():
	try:
		columns, data = fetch_from_db(
				RAW_DATA__TG_POSTS,
				select_condition=RAW_TO_STAGING__SELECT,
				where_condition=RAW_TO_STAGING__WHERE
				)
		
		df = pd.DataFrame(data, columns=columns)
		df['job_post'] = df['post'].apply(lambda post: job_description_detection(post))
		
		batch_insert_to_db(STAGING_DATA__POSTS, STAGING_DATA__POSTS__COLUMNS, [], df.to_dict(orient='records'))
		logging.info('Data successfully moved to staging.')
	
	except Exception as error:
		logging.error(f'Failed to move data: {error}')


if __name__ == '__main__':
	clean_and_move_data()
