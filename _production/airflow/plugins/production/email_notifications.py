import sys

sys.path.insert(0, "/home/job_search")

import logging
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from _production import (
    EMAIL_NOTIFICATION_CHUNK_SIZE,
    PROD_DATA__JOBS,
    RECIPIENT_EMAIL,
    SENDER_EMAIL,
    STAGING_DATA__POSTS,
)

from _production.config.config import STAGING_TO_PROD__SELECT, STAGING_TO_PROD__WHERE
from _production.airflow.plugins.production.helpers import (
    format_email_content,
    send_email,
)
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import (
    batch_update_to_db,
    fetch_from_db,
    move_data_with_condition,
)

# Setup logging
file_name = __file__[:-3]
setup_logging(file_name)


def fetch_new_posts():
    """Fetch new unnotified posts from the production database."""
    try:
        columns, new_posts = fetch_from_db(
            PROD_DATA__JOBS, select_condition="*", where_condition="notificated = FALSE"
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


def send_notifications(df):
    try:
        chunks = [
            df.iloc[i : i + EMAIL_NOTIFICATION_CHUNK_SIZE]
            for i in range(0, len(df), EMAIL_NOTIFICATION_CHUNK_SIZE)
        ]
        total_chunks = len(chunks)

        successfull_ids = []
        for index, chunk in enumerate(chunks, start=1):
            try:
                email_content = format_email_content(chunk)
                message = MIMEMultipart("alternative")
                message["Subject"] = (
                    f"Andrew: Job Notifications ({index}/{total_chunks})"
                )
                message["From"] = SENDER_EMAIL
                message["To"] = RECIPIENT_EMAIL
                message.attach(MIMEText(email_content, "html"))

                if send_email(message):
                    logging.info(f"Email {index}/{total_chunks} sent successfully!")
                    successfull_ids.extend(chunk["id"].values)
                else:
                    logging.warning(
                        f"Failed to send email chunk {index}/{total_chunks}"
                    )

            except Exception:
                logging.error(
                    f"Failed to process chunk {index}/{total_chunks}", exc_info=True
                )
                # Continue with next chunk instead of failing completely
                continue

        return successfull_ids

    except Exception as error:
        logging.error("Failed to process notification chunks", exc_info=True)
        raise Exception(f"Notification processing failed: {str(error)}") from error


def update_notifications(successfull_ids):
    """Update notification status for successfully sent emails."""
    try:
        if not successfull_ids:
            logging.info("No successful notifications to update")
            return

        update_data = [{"id": _id, "notificated": True} for _id in successfull_ids]

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


def notify_me():
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
            logging.error("Failed to move data to production", exc_info=True)
            raise Exception(f"Data movement failed: {str(move_error)}") from move_error

        # Fetch and process new posts
        df = fetch_new_posts()
        if df is not None:
            successfull_ids = send_notifications(df)
            if successfull_ids:
                update_notifications(successfull_ids)

    except Exception as error:
        logging.error("Notification process failed", exc_info=True)
        raise Exception(f"Notification process failed: {str(error)}") from error


if __name__ == "__main__":
    notify_me()
