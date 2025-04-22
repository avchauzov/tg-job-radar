"""Email handling utilities for job notifications."""

import logging
import os
import smtplib
from typing import Any

from _production import EMAIL

JsonDict = dict[str, Any]


def format_email_content(df):
    """Format job posting data into HTML email content.

    Args:
        df: DataFrame containing job posting information.

    Returns:
        str: Formatted HTML content for email.
    """

    def format_job_fields(job_dict):
        """Format individual job posting fields into HTML.

        Args:
            job_dict: Dictionary containing job posting data.

        Returns:
            list: Formatted HTML strings for each field.
        """
        # Define field order and their display names
        field_order = [
            ("job_title", "Position"),
            ("location", "Location"),
            ("skills", "Skills"),
            ("seniority_level", "Level"),
            ("salary_range", "Salary"),
            ("company_name", "Company"),
            ("description", "Description"),
            ("full_description", "Full Description"),
        ]

        formatted_fields = []

        # First try all fields except description and full_description
        for field, display_name in field_order[
            :-2
        ]:  # Exclude description and full_description
            value = job_dict.get(field)
            if value is not None:  # Only append if there's a value
                formatted_fields.append(f"<strong>{display_name}:</strong> {value}")

        # If no fields were added, try description or full_description
        if not formatted_fields:
            description = job_dict.get("description")
            full_description = job_dict.get("full_description")

            if description is not None:
                formatted_fields.append(f"<strong>Description:</strong> {description}")
            elif full_description is not None:
                formatted_fields.append(
                    f"<strong>Description:</strong> {full_description}"
                )

        if not formatted_fields:
            for key, value in job_dict.items():
                if key not in [item[0] for item in field_order]:
                    raise ValueError(f"Invalid field: {key}")

        return formatted_fields

    html_parts = []
    for _, row in df.iterrows():
        formatted_fields = format_job_fields(row["post_structured"])
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
        return None

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        {''.join(html_parts)}
    </body>
    </html>
    """


def send_email(message):
    """Send an email using SMTP with configured credentials.

    Args:
        message: Email message object to send.

    Returns:
        bool: True if email was sent successfully, False otherwise.

    Raises:
        SMTPAuthenticationError: If authentication with the SMTP server fails.
    """
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
