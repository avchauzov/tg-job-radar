"""Utilities for email formatting and sending."""

import json
import logging
import os
import smtplib
from typing import Any, Dict

from _production import EMAIL

JsonDict = Dict[str, Any]


def format_email_content(df):
    """Format job posts into HTML email content using structured JSON data"""

    def get_formatted_fields(job_data):
        """Get formatted fields in the correct order"""
        # Define field order and their display names
        field_order = [
            ("job_title", "Position"),
            ("seniority_level", "Level"),
            ("company_name", "Company"),
            ("location", "Location"),
            ("remote_status", "Work Mode"),
            ("salary_range", "Salary"),
            ("relocation_support", "Relocation Support"),
            ("visa_sponsorship", "Visa Sponsorship"),
            ("description", "Description"),
        ]

        formatted_fields = []
        job_dict = json.loads(job_data) if isinstance(job_data, str) else job_data

        for field, display_name in field_order:
            value = job_dict.get(field)
            if value is not None:  # Only append if there's a value
                # Convert boolean to "Yes"
                value = "Yes" if isinstance(value, bool) and value else value
                formatted_fields.append(f"<strong>{display_name}:</strong> {value}")

        return formatted_fields

    html_parts = []
    for _, row in df.iterrows():
        formatted_fields = get_formatted_fields(row["post_structured"])
        if formatted_fields:
            job_html = f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
                {'<br>'.join(formatted_fields)}
                <br><br>
                <a href="{row['post_link']}" style="color: #0066cc;">View Original Post</a>
            </div>
            """
            html_parts.append(job_html)

    if not html_parts:
        return "<p>No new job posts to display.</p>"

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        {''.join(html_parts)}
    </body>
    </html>
    """


def send_email(message):
    try:
        with smtplib.SMTP(
            os.getenv("SMTP_SERVER", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", 587))
        ) as server:
            try:
                server.starttls()
                server.login(EMAIL["SENDER"], EMAIL["GMAIL_APP_PASSWORD"])
                server.sendmail(
                    EMAIL["SENDER"], EMAIL["RECIPIENT"], message.as_string()
                )
                return True
            except smtplib.SMTPAuthenticationError as auth_error:
                logging.error(f"Authentication failed: {auth_error}")
                return False
            except smtplib.SMTPException as smtp_error:
                logging.error(f"SMTP error occurred: {smtp_error}")
                return False

    except (ValueError, OSError) as connection_error:
        logging.error(f"Connection setup failed: {connection_error}")
        return False

    except Exception as error:
        logging.error(f"Unexpected error while sending email: {error}")
        return False
