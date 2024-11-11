import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from prod import GMAIL_APP_PASSWORD, PROD_DATA__JOBS__NAME, PROD_DATA__JOBS__ORDER_BY_CONDITION, PROD_DATA__JOBS__SELECT_CONDITION, PROD_DATA__JOBS__WHERE_CONDITION, RAW_DATA__TG_POSTS__NAME, RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, RAW_DATA_TO_PROD_DATA_WHERE_CONDITION, RECIPIENT_EMAIL, SENDER_EMAIL
from prod.utils.functions_common import setup_logging
from prod.utils.functions_sql import fetch_from_db, move_data_with_condition


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


def format_email_content(df):
	email_content = ''
	index = 0
	
	for _, row in df.iterrows():
		channel = row['channel']
		post = row['post'].replace('\n', '<br>')
		post_link = row['post_link']
		
		email_content += f"<div style='margin-bottom: 10px;'>{post}<br><a href='{post_link}'>Link</a></div>"
		
		if index == len(df) - 1 or df.iloc[index + 1]['channel'] != channel:
			email_content += "</div><hr style='border: 1px solid #ccc; width: 80%; margin: 20px auto;'>"
		
		index += 1
		if index > 10:
			break
	
	return email_content


def send_email(message):
	try:
		with smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'), int(os.getenv('SMTP_PORT', 587))) as server:
			server.starttls()
			server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
			server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())
	
	except smtplib.SMTPException as smtp_error:
		logging.error(f'SMTP error occurred: {smtp_error}')
	
	except Exception as error:
		logging.error(f'Failed to send email: {error}')


if __name__ == '__main__':
	notify_me()
