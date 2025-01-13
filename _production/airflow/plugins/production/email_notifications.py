"""
Email Notification Module

Handles the process of:
1. Moving data from staging to production
2. Fetching unnotified posts
3. Sending email notifications in chunks
4. Updating notification status
"""

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from _production import DATA_BATCH_SIZE, EMAIL, PROD_DATA__JOBS, STAGING_DATA__POSTS
from _production.config.config_db import STAGING_TO_PROD__SELECT, STAGING_TO_PROD__WHERE
from _production.utils.common import setup_logging
from _production.utils.email import format_email_content, send_email
from _production.utils.sql import (
    batch_update_to_db,
    fetch_from_db,
    move_data_with_condition,
)

# Setup logging
setup_logging(__file__[:-3])


def fetch_new_posts() -> Optional[pd.DataFrame]:
    """
    Fetch new unnotified posts from the production database.

    Returns:
        pd.DataFrame | None: DataFrame with new posts or None if no posts found
    Raises:
        Exception: If database fetch fails
    """
    try:
        columns, new_posts = fetch_from_db(
            PROD_DATA__JOBS,
            select_condition="DISTINCT ON (post_structured) *",
            where_condition="notificated = FALSE",
        )

        if not new_posts:
            logging.info("No new posts found to send.")
            return None

        df = pd.DataFrame(new_posts, columns=columns)
        logging.info(f"Fetched {len(df)} new posts.")
        return df

    except Exception as error:
        logging.error("Failed to fetch new posts from database", exc_info=True)
        raise Exception(f"Database fetch failed: {str(error)}") from error


def create_email_message(
    chunk: pd.DataFrame, index: int, total_chunks: int
) -> MIMEMultipart:
    """
    Create an email message for a chunk of posts.

    Args:
        chunk: DataFrame containing a subset of posts
        index: Current chunk number
        total_chunks: Total number of chunks

    Returns:
        MIMEMultipart: Formatted email message
    """
    message = MIMEMultipart("alternative")
    message["Subject"] = f"Andrew: Job Notifications ({index}/{total_chunks})"
    message["From"] = EMAIL["SENDER"]
    message["To"] = EMAIL["RECIPIENT"]
    message.attach(MIMEText(format_email_content(chunk), "html"))
    return message


def handle_retry_failure(retry_state):
    logging.error(
        f"Failed to process chunk after {retry_state.attempt_number} attempts"
    )
    return []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry_error_callback=handle_retry_failure,
)
def process_chunk(chunk: pd.DataFrame, index: int, total_chunks: int) -> List[int]:
    """
    Process a single chunk of posts with retry logic.

    Args:
        chunk: DataFrame containing a subset of posts
        index: Current chunk number
        total_chunks: Total number of chunks

    Returns:
        List[int]: List of successfully processed post IDs
    """
    message = create_email_message(chunk, index, total_chunks)

    if send_email(message):
        logging.info(f"Email {index}/{total_chunks} sent successfully!")
        return chunk["id"].tolist()

    raise Exception(f"Failed to send email chunk {index}/{total_chunks}")


def send_notifications(df: pd.DataFrame) -> List[int]:
    """
    Send email notifications in chunks.

    Args:
        df: DataFrame containing posts to notify about

    Returns:
        List[int]: List of successfully processed post IDs
    """
    try:
        chunks = [
            df.iloc[i : i + DATA_BATCH_SIZE] for i in range(0, len(df), DATA_BATCH_SIZE)
        ]
        total_chunks = len(chunks)
        successful_ids: List[int] = []

        for index, chunk in enumerate(chunks, start=1):
            try:
                chunk_ids = process_chunk(chunk, index, total_chunks)
                successful_ids.extend(chunk_ids)
            except Exception:
                logging.error(
                    f"Failed to process chunk {index}/{total_chunks} after all retries",
                    exc_info=True,
                )
                continue

        return successful_ids

    except Exception as error:
        logging.error("Failed to process notification chunks", exc_info=True)
        raise Exception(f"Notification processing failed: {str(error)}") from error


def update_notifications(successful_ids: List[int]) -> None:
    """
    Update notification status for successfully sent emails.

    Args:
        successful_ids: List of post IDs to mark as notified
    """
    try:
        if not successful_ids:
            logging.info("No successful notifications to update")
            return

        update_data = [{"id": _id, "notificated": True} for _id in successful_ids]
        batch_update_to_db(
            table_name=PROD_DATA__JOBS,
            update_columns=["notificated"],
            condition_column="id",
            data=update_data,
        )

        logging.info(f"Updated {len(update_data)} rows in the database.")

    except Exception as error:
        logging.error("Failed to update notification status", exc_info=True)
        raise Exception(f"Database update failed: {str(error)}") from error


def notify_me() -> None:
    """
    Main function to handle the complete notification process.

    1. Moves data from staging to production
    2. Fetches new posts
    3. Sends notifications
    4. Updates notification status
    """
    try:
        # Move data to production
        try:
            move_data_with_condition(
                STAGING_DATA__POSTS,
                PROD_DATA__JOBS,
                select_condition=STAGING_TO_PROD__SELECT,
                where_condition=STAGING_TO_PROD__WHERE,
                json_columns=["post_structured"],
            )
        except Exception as move_error:
            if "no results to fetch" in str(move_error):
                logging.info("No new data to move from staging to production")
            else:
                raise  # Re-raise if it's a different error

        # Fetch and process new posts
        df = fetch_new_posts()
        if df is not None:
            successful_ids = send_notifications(df)
            if successful_ids:
                update_notifications(successful_ids)

    except Exception as error:
        logging.error("Notification process failed", exc_info=True)
        raise Exception(f"Notification process failed: {str(error)}") from error


if __name__ == "__main__":
    notify_me()
