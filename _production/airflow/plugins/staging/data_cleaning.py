import sys

sys.path.insert(0, "/home/job_search")


import logging
import pandas as pd
import json

from _production.config.config import (
    RAW_TO_STAGING__WHERE,
    STAGING_DATA__POSTS__COLUMNS,
)
from _production.config.config_hidden import CV_DOC_ID, MATCH_SCORE_THRESHOLD

from _production import RAW_DATA__TG_POSTS, STAGING_DATA__POSTS
from _production.utils.functions_llm import (
    job_post_detection,
    single_job_post_detection,
    reformat_post,
    match_cv_with_job,
)
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import batch_insert_to_db, fetch_from_db


file_name = __file__[:-3]
setup_logging(file_name)


def get_cv_content():
    """Download CV content from Google Docs public link"""
    try:
        import requests

        # Convert document ID to export link
        export_link = (
            f"https://docs.google.com/document/d/{CV_DOC_ID}/export?format=txt"
        )

        response = requests.get(export_link)
        response.raise_for_status()  # Raise exception for bad status codes

        return response.text

    except Exception as error:
        logging.error(f"Failed to fetch CV: {error}")
        return None


def clean_and_move_data():
    try:
        # Download CV once before processing
        cv_content = get_cv_content()
        if cv_content is None:
            raise Exception("Failed to fetch CV content")

        columns, data = fetch_from_db(
            RAW_DATA__TG_POSTS,
            select_condition="*",
            where_condition=RAW_TO_STAGING__WHERE,
        )

        df = pd.DataFrame(data, columns=columns)  # .sample(n=16)
        df["is_job_post"] = df["post"].apply(lambda post: job_post_detection(post))
        df["is_single_job_post"] = df.apply(
            lambda row: False
            if not row["is_job_post"]
            else single_job_post_detection(row["post"]),
            axis=1,
        )

        # Calculate match score for single job posts
        df["score"] = df.apply(
            lambda row: match_cv_with_job(cv_content, row["post"])
            if row["is_single_job_post"]
            else 0,
            axis=1,
        )

        # Only process posts that meet the score threshold
        df["post_structured"] = df.apply(
            lambda row: json.dumps(reformat_post(row["post"]))
            if row["is_single_job_post"] and row["score"] >= MATCH_SCORE_THRESHOLD
            else json.dumps({}),
            axis=1,
        )

        records = df.to_dict(orient="records")
        batch_insert_to_db(
            STAGING_DATA__POSTS, STAGING_DATA__POSTS__COLUMNS, ["id"], records
        )
        logging.info("Data successfully moved to staging.")

    except Exception as error:
        logging.error(f"Failed to move data: {error}")


if __name__ == "__main__":
    clean_and_move_data()
