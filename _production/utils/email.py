"""Utilities for email formatting and sending."""

import json
import logging
import os
import smtplib
from typing import Any, Dict

from _production import EMAIL

JsonDict = Dict[str, Any]


def format_email_content(df) -> str:
    """Format job posts into HTML email content."""
    if df.empty:
        return "No jobs to display"

    FIELD_ORDER = [
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

    def get_formatted_fields(job_data: JsonDict) -> list[str]:
        job_dict = json.loads(job_data) if isinstance(job_data, str) else job_data
        return [
            f"<strong>{display_name}:</strong> {str(job_dict[field])}"
            for field, display_name in FIELD_ORDER
            if job_dict.get(field) is not None
        ]

    formatted_jobs = []
    for _, row in df.iterrows():
        fields = get_formatted_fields(row)
        formatted_jobs.append("<br>".join(fields))

    return "<hr><br>".join(formatted_jobs)


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
