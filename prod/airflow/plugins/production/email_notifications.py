import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from prod import GMAIL_APP_PASSWORD, PROD_DATA__JOBS__NAME, PROD_DATA__JOBS__ORDER_BY_CONDITION, PROD_DATA__JOBS__SELECT_CONDITION, PROD_DATA__JOBS__WHERE_CONDITION, RAW_DATA__TG_POSTS__NAME, RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, RAW_DATA_TO_PROD_DATA_WHERE_CONDITION, RECIPIENT_EMAIL, SENDER_EMAIL
from prod.utils.functions_sql import fetch_from_db, move_data_with_condition


sys.path.insert(0, '/home/job_search')

import os

from prod.utils.functions_common import setup_logging


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def notify_me():
	move_data_with_condition(RAW_DATA__TG_POSTS__NAME, PROD_DATA__JOBS__NAME, select_condition=RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, where_condition=RAW_DATA_TO_PROD_DATA_WHERE_CONDITION)
	columns, new_posts = fetch_from_db(PROD_DATA__JOBS__NAME, select_condition=PROD_DATA__JOBS__SELECT_CONDITION, where_condition=PROD_DATA__JOBS__WHERE_CONDITION, order_by_condition=PROD_DATA__JOBS__ORDER_BY_CONDITION)
	
	df = pd.DataFrame(new_posts, columns=columns)
	
	message = MIMEMultipart('alternative')
	message['Subject'] = 'TG Job Notifications'
	message['From'] = SENDER_EMAIL
	message['To'] = RECIPIENT_EMAIL
	
	index, email_content = 0, ''
	for row in df.iterrows():
		channel = row[1]['channel']
		post = row[1]['post'].replace('\n', '<br>')
		post_link = row[1]['post_link']
		
		email_content += f"<div style='margin-bottom: 10px;'>{post}<br><a href='{post_link}'>Link</a></div>"
		if row[0] == len(df) - 1 or df.iloc[row[0] + 1]['channel'] != channel:
			email_content += "</div><hr style='border: 1px solid #ccc; width: 80%; margin: 20px auto;'>"
		
		index += 1
		if index > 10:
			break
	
	message.attach(MIMEText(email_content, 'html'))
	
	try:
		with smtplib.SMTP('smtp.gmail.com', 587) as server:
			server.starttls()
			server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
			server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())
			print('Email sent successfully!')
	
	except Exception as error:
		print(f'Failed to send email: {error}')


if __name__ == '__main__':
	notify_me()
