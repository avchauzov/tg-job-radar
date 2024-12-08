import sys

sys.path.insert(0, "/home/job_search")


import logging
import pandas as pd
import json

from _production.config.config import (
    RAW_TO_STAGING__WHERE,
    STAGING_DATA__POSTS__COLUMNS,
)

from _production import RAW_DATA__TG_POSTS, STAGING_DATA__POSTS
from _production.utils.functions_llm import (
    job_post_detection,
    single_job_post_detection,
    reformat_post,
)
from _production.utils.functions_common import setup_logging
from _production.utils.functions_sql import batch_insert_to_db, fetch_from_db


file_name = __file__[:-3]
setup_logging(file_name)


def clean_and_move_data():
    try:
        columns, data = fetch_from_db(
            RAW_DATA__TG_POSTS,
            select_condition="*",  # revise
            where_condition=RAW_TO_STAGING__WHERE,
        )

        df = pd.DataFrame(data, columns=columns)  # .sample(n=128)
        df["is_job_post"] = df["post"].apply(lambda post: job_post_detection(post))
        df["is_single_job_post"] = df.apply(
            lambda row: False
            if not row["is_job_post"]
            else single_job_post_detection(row["post"]),
            axis=1,
        )

        # Only process single job posts, return empty JSON for others
        df["post_structured"] = df.apply(
            lambda row: json.dumps(reformat_post(row["post"]))
            if row["is_single_job_post"]
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
