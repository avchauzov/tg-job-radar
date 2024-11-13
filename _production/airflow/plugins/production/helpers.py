import logging
import os
import smtplib

from _production import GMAIL_APP_PASSWORD, RECIPIENT_EMAIL, SENDER_EMAIL


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
