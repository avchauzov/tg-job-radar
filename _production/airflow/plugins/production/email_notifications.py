import logging
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from _production import (
	EMAIL_NOTIFICATION_CHUNK_SIZE, PROD_DATA__CONDITION_COLUMN__NAME, PROD_DATA__JOBS__NAME, PROD_DATA__JOBS__ORDER_BY_CONDITION, PROD_DATA__JOBS__SELECT_CONDITION, PROD_DATA__JOBS__WHERE_CONDITION, PROD_DATA__UPDATE_COLUMN__NAME, RECIPIENT_EMAIL, SENDER_EMAIL, STAGING_DATA__JOBS__NAME,
	STAGING_DATA_TO_PROD_DATA__WHERE_CONDITION,
	)
from _production.airflow.plugins.production.helpers import format_email_content, send_email
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import batch_update_to_db, fetch_from_db, move_data_with_condition


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def fetch_new_posts():
	columns, new_posts = fetch_from_db(
			PROD_DATA__JOBS__NAME,
			select_condition=PROD_DATA__JOBS__SELECT_CONDITION,
			where_condition=PROD_DATA__JOBS__WHERE_CONDITION,
			order_by_condition=PROD_DATA__JOBS__ORDER_BY_CONDITION
			)
	
	if not new_posts:
		logging.info('No new posts found to send.')
		return None, None
	
	df = pd.DataFrame(new_posts, columns=columns)
	logging.info(f'Fetched {len(df)} new posts.')
	
	return df


def send_notifications(df):
	chunks = [df.iloc[i:i + EMAIL_NOTIFICATION_CHUNK_SIZE] for i in range(0, len(df), EMAIL_NOTIFICATION_CHUNK_SIZE)]
	total_chunks = len(chunks)
	
	successfull_ids = []
	for index, chunk in enumerate(chunks, start=1):
		email_content = format_email_content(chunk)
		message = MIMEMultipart('alternative')
		message['Subject'] = f'Andrew: Job Notifications ({index}/{total_chunks})'
		message['From'] = SENDER_EMAIL
		message['To'] = RECIPIENT_EMAIL
		message.attach(MIMEText(email_content, 'html'))
		
		if send_email(message):
			logging.info(f'Email {index}/{total_chunks} sent successfully!')
			successfull_ids.extend(chunk['id'].values)
	
	return successfull_ids


def update_notifications(successfull_ids):
	update_data = [{'id': id, 'notificated': True} for id in successfull_ids]
	batch_update_to_db(
			table_name=PROD_DATA__JOBS__NAME,
			update_columns=[PROD_DATA__UPDATE_COLUMN__NAME],
			condition_column=PROD_DATA__CONDITION_COLUMN__NAME,
			data=update_data
			)
	
	logging.info(f'Updated {len(update_data)} rows in the database.')


def notify_me():
	try:
		move_data_with_condition(
				STAGING_DATA__JOBS__NAME,
				PROD_DATA__JOBS__NAME,
				select_condition='*',
				where_condition=STAGING_DATA_TO_PROD_DATA__WHERE_CONDITION
				)
		
		df = fetch_new_posts()
		if df is not None:
			successfull_ids = send_notifications(df)
			
			if successfull_ids:
				update_notifications(successfull_ids)
	
	except Exception as error:
		logging.error(f'Failed to complete notification process: {error}', exc_info=True)


if __name__ == '__main__':
	notify_me()
