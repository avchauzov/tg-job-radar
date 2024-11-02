import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_via_gmail(sender_email, app_password, recipient_email, subject, body):
	# Set up the email content
	msg = MIMEMultipart()
	msg['From'] = sender_email
	msg['To'] = recipient_email
	msg['Subject'] = subject
	msg.attach(MIMEText(body, 'plain'))
	
	try:
		# Connect to Gmail's SMTP server
		with smtplib.SMTP('smtp.gmail.com', 587) as server:
			server.starttls()  # Start TLS for security
			server.login(sender_email, app_password)  # Log in with app password
			server.sendmail(sender_email, recipient_email, msg.as_string())
			print("Email sent successfully!")
	except Exception as e:
		print(f"Failed to send email: {e}")


# Example usage
sender_email = ""
app_password = ""
recipient_email = ""
subject = "Test Email from Python"
body = "Hello! This is a test email sent directly through Gmail's SMTP server."

send_email_via_gmail(sender_email, app_password, recipient_email, subject, body)
