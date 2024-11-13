import sys


sys.path.insert(0, '/home/job_search')

import logging
import os

from _production import (
	RAW_DATA__TG_POSTS__NAME,
	RAW_DATA_TO_STAGING_DATA__WHERE_CONDITION, STAGING_DATA__JOBS__NAME,
	)

from _production.config.config import RAW_DATA_TO_STAGING_DATA__SELECT_CONDITION
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import move_data_with_condition


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def clean_and_move_data():
	try:
		move_data_with_condition(RAW_DATA__TG_POSTS__NAME, STAGING_DATA__JOBS__NAME, select_condition=RAW_DATA_TO_STAGING_DATA__SELECT_CONDITION, where_condition=RAW_DATA_TO_STAGING_DATA__WHERE_CONDITION)
	
	except Exception as error:
		logging.error(f'Failed to move data: {error}')


if __name__ == '__main__':
	clean_and_move_data()
