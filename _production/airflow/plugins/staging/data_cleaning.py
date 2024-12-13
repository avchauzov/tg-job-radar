import sys

sys.path.insert(0, "/home/job_search")

import datetime
import logging
import pandas as pd
import json
import requests

from _production.config.config import (
    DATA_BATCH_SIZE,
    RAW_TO_STAGING__WHERE,
    STAGING_DATA__POSTS__COLUMNS,
    CV_DOC_ID,
    MATCH_SCORE_THRESHOLD,
)
from _production import (
    EMAIL_NOTIFICATION_CHUNK_SIZE,
    EMAIL_NOTIFICATION_CHUNK_MULTIPLIER,
    RAW_DATA__TG_POSTS,
    STAGING_DATA__POSTS,
)
from _production.utils.functions_llm import (
    job_post_detection,
    single_job_post_detection,
    job_post_parsing,
    match_cv_with_job,
)
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import batch_insert_to_db, fetch_from_db


file_name = __file__[:-3]
setup_logging(file_name)


def fetch_cv_content(doc_id):
    try:
        response = requests.get(
            f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        )
        response.raise_for_status()
        return response.text

    except requests.HTTPError as error:
        # Specific handling for HTTP errors
        logging.debug(
            f"CV fetch failed with HTTP error {error.response.status_code}",
            exc_info=True,
        )
        raise
    except requests.ConnectionError as _:
        # Network connection errors
        logging.debug("CV fetch failed due to connection error", exc_info=True)
        raise
    except requests.RequestException as _:
        # Catch-all for other request-related errors
        logging.debug("CV fetch failed", exc_info=True)
        raise


def clean_and_move_data():
    try:
        # Download CV once before processing
        cv_content = fetch_cv_content(CV_DOC_ID)
        if cv_content is None:
            raise ValueError("Failed to fetch CV content")

        try:
            columns, data = fetch_from_db(
                RAW_DATA__TG_POSTS,
                select_condition="*",
                where_condition=RAW_TO_STAGING__WHERE,
                random_limit=EMAIL_NOTIFICATION_CHUNK_SIZE
                * EMAIL_NOTIFICATION_CHUNK_MULTIPLIER,
            )
            df = pd.DataFrame(data, columns=columns)
        except Exception as db_error:
            logging.debug(
                "Failed to fetch or create DataFrame from database", exc_info=True
            )
            raise Exception(f"Database operation failed: {str(db_error)}") from db_error

        # Process in batches
        for i in range(0, len(df), DATA_BATCH_SIZE):
            batch_df = df[i : i + DATA_BATCH_SIZE]

            try:
                try:
                    batch_df["is_job_post"] = batch_df["post"].apply(
                        lambda post: job_post_detection(post)
                    )
                    # Filter out rows where is_job_post is None
                    batch_df = batch_df[batch_df["is_job_post"].notna()]

                    batch_df["is_single_job_post"] = batch_df.apply(
                        lambda row: False
                        if not row["is_job_post"]
                        else single_job_post_detection(row["post"]),
                        axis=1,
                    )
                    # Filter out rows where is_single_job_post is None
                    batch_df = batch_df[batch_df["is_single_job_post"].notna()]
                except Exception as detection_error:
                    raise Exception(
                        f"Job post detection failed: {str(detection_error)}"
                    ) from detection_error

                try:
                    batch_df["score"] = batch_df.apply(
                        lambda row: 0
                        if not row["is_single_job_post"]
                        else match_cv_with_job(cv_content, row["post"]),
                        axis=1,
                    )
                    # Filter out rows where score is None
                    batch_df = batch_df[batch_df["score"].notna()]

                except Exception as matching_error:
                    raise Exception(
                        f"CV matching failed: {str(matching_error)}"
                    ) from matching_error

                # Only process rows that have valid scores and are single job posts
                valid_posts_mask = batch_df["is_single_job_post"] & (
                    batch_df["score"] >= MATCH_SCORE_THRESHOLD
                )

                batch_df["post_structured"] = batch_df.apply(
                    lambda row: json.dumps(job_post_parsing(row["post"]))
                    if row["is_single_job_post"]
                    and row["score"] >= MATCH_SCORE_THRESHOLD
                    else json.dumps({}),
                    axis=1,
                )

                batch_df["created_at"] = datetime.datetime.now(datetime.UTC)

                records = batch_df.to_dict(orient="records")
                batch_insert_to_db(
                    STAGING_DATA__POSTS, STAGING_DATA__POSTS__COLUMNS, ["id"], records
                )
                logging.info(
                    f"Processed and loaded batch {i // DATA_BATCH_SIZE + 1}, size: {len(records)}"
                )

            except Exception as batch_error:
                logging.debug(
                    f"Batch processing failed. Details: {str(batch_error)}",
                    exc_info=True,
                )

                raise Exception(
                    f"Failed to process batch {i // DATA_BATCH_SIZE + 1}. "
                    f"Batch size: {len(df)}, "
                    f"Records processed: {len(records)}. "
                    f"Original error: {str(batch_error)}"
                ) from batch_error

        logging.info("Successfully processed all data batches")

    except Exception as error:
        logging.debug("Data pipeline failed", exc_info=True)

        raise Exception(
            f"Data pipeline failed. "
            f"Total records: {len(df) if 'df' in locals() else 'unknown'}, "
            f"Completed batches: {i//DATA_BATCH_SIZE if 'i' in locals() else 0}. "
            f"Original error: {str(error)}"
        ) from error


if __name__ == "__main__":
    clean_and_move_data()
