import logging
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from _production import PROD_DATA__JOBS__NAME, PROD_DATA__JOBS__ORDER_BY_CONDITION, PROD_DATA__JOBS__SELECT_CONDITION, PROD_DATA__JOBS__WHERE_CONDITION, RAW_DATA__TG_POSTS__NAME, RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, RAW_DATA_TO_PROD_DATA_WHERE_CONDITION, RECIPIENT_EMAIL, SENDER_EMAIL
from _production.airflow.plugins.production.helpers import format_email_content, send_email
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import fetch_from_db, move_data_with_condition


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def notify_me():
	try:
		move_data_with_condition(RAW_DATA__TG_POSTS__NAME, PROD_DATA__JOBS__NAME, select_condition=RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, where_condition=RAW_DATA_TO_PROD_DATA_WHERE_CONDITION)
		columns, new_posts = fetch_from_db(PROD_DATA__JOBS__NAME, select_condition=PROD_DATA__JOBS__SELECT_CONDITION, where_condition=PROD_DATA__JOBS__WHERE_CONDITION, order_by_condition=PROD_DATA__JOBS__ORDER_BY_CONDITION)
		
		if not new_posts:
			logging.info('No new posts found to send.')
			return
		
		df = pd.DataFrame(new_posts, columns=columns)
		email_content = format_email_content(df)
		
		message = MIMEMultipart('alternative')
		message['Subject'] = 'TG Job Notifications'
		message['From'] = SENDER_EMAIL
		message['To'] = RECIPIENT_EMAIL
		message.attach(MIMEText(email_content, 'html'))
		
		send_email(message)
		logging.info('Email sent successfully!')
	
	except Exception as error:
		logging.error(f'Failed to complete notification process: {error}')


if __name__ == '__main__':
	notify_me()
