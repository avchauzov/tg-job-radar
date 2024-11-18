import sys

import pandas as pd

from _production.utils.functions_llm import job_description_detection


sys.path.insert(0, '/home/job_search')

import logging
import os

from _production import (
	RAW_DATA__TG_POSTS__NAME,
	)

from _production.config.config import RAW_DATA_TO_STAGING_DATA__SELECT_CONDITION
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import fetch_from_db


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def clean_and_move_data():
	try:
		columns, data = fetch_from_db(
			RAW_DATA__TG_POSTS__NAME,
			select_condition=RAW_DATA_TO_STAGING_DATA__SELECT_CONDITION,
			# where_condition=RAW_DATA_TO_STAGING_DATA__WHERE_CONDITION
			)
		
		df = pd.DataFrame(data, columns=columns)[: 2]
		df['job_post'] = df['post'].apply(lambda post: job_description_detection(post))
	
	# TODO: finish
	# batch_insert_to_db(RAW_DATA__TG_POSTS__NAME, RAW_DATA__TG_POSTS__COLUMNS, STAGING_DATA__POSTS__CONFLICT, results)
	
	except Exception as error:
		logging.error(f'Failed to move data: {error}')


if __name__ == '__main__':
	clean_and_move_data()
