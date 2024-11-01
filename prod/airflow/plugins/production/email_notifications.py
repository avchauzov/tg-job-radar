import sys

from prod import PROD_DATA__JOBS__NAME, PROD_DATA__JOBS__ORDER_BY_CONDITION, PROD_DATA__JOBS__SELECT_CONDITION, PROD_DATA__JOBS__WHERE_CONDITION, RAW_DATA__TG_POSTS__NAME, RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, RAW_DATA_TO_PROD_DATA_WHERE_CONDITION
from prod.utils.functions_sql import fetch_from_db, move_data_with_condition


sys.path.insert(0, '/home/job_search')

import os

from prod.utils.functions_common import setup_logging


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def notify_me():
	move_data_with_condition(RAW_DATA__TG_POSTS__NAME, PROD_DATA__JOBS__NAME, select_condition=RAW_DATA_TO_PROD_DATA_SELECT_CONDITION, where_condition=RAW_DATA_TO_PROD_DATA_WHERE_CONDITION)
	columns, new_posts = fetch_from_db(PROD_DATA__JOBS__NAME, select_condition=PROD_DATA__JOBS__SELECT_CONDITION, where_condition=PROD_DATA__JOBS__WHERE_CONDITION, order_by_condition=PROD_DATA__JOBS__ORDER_BY_CONDITION)
	
	print(123)
	
	# add to prod_data.jobs all rows from raw_data.tg_posts that do not exist in prod_data.jobs (matching by id)
	# while doing this, new posts are added with mask 'notificated' = False
	# then, iterated through every post with 'notificated' = False, wrap this post (columns post and post_link)
	# and send it to my email
	# aggregate several posts into one message using group by channel
	# after that, put 'notificated' to True
	
	'''
	def send_email(posts_by_channel):
	    sender_email = "your_email@example.com"
	    receiver_email = "recipient@example.com"
	    smtp_server = "smtp.example.com"
	    smtp_port = 587
	    smtp_user = "your_email@example.com"
	    smtp_password = "your_password"
	    
	    message = MIMEMultipart("alternative")
	    message["Subject"] = "New Job Notifications"
	    message["From"] = sender_email
	    message["To"] = receiver_email
	
	    # Create the email content
	    email_content = ""
	    for channel, posts in posts_by_channel.items():
	        email_content += f"<h3>Channel: {channel}</h3><ul>"
	        for post in posts:
	            email_content += f"<li>{post['post']} - <a href='{post['post_link']}'>Link</a></li>"
	        email_content += "</ul><br>"
	
	    message.attach(MIMEText(email_content, "html"))
	
	    with smtplib.SMTP(smtp_server, smtp_port) as server:
	        server.starttls()
	        server.login(smtp_user, smtp_password)
	        server.sendmail(sender_email, receiver_email, message.as_string())
	
	def update_notifications():
	    # Mark notificated = True for sent notifications
	    query = "UPDATE prod_data.jobs SET notificated = True WHERE notificated = False"
	    with engine.connect() as conn:
	        conn.execute(query)
	
	def notify_me():
	    fetch_new_posts()
	    
	    unsent_posts = get_unsent_notifications()
	    if not unsent_posts.empty:
	        # Group posts by channel
	        posts_by_channel = unsent_posts.groupby('channel')[['post', 'post_link']].apply(lambda x: x.to_dict('records')).to_dict()
	        
	        # Send email
	        send_email(posts_by_channel)
	        
	        # Mark notifications as sent
	        update_notifications()
	
	if __name__ == '__main__':
	    notify_me()
		'''
	
	pass


if __name__ == '__main__':
	notify_me()
