"""
Data cleaning module for processing job posts and matching them against CV content.

This module handles the ETL process from raw data to staging, including:
- CV content validation and retrieval
- Job post detection and classification
- CV-to-job matching with scoring
- Structured data parsing and storage
"""

import contextlib
import datetime
import json
import logging

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from _production import (
    CV_DOC_ID,
    DATA_BATCH_SIZE,
    MATCH_SCORE_THRESHOLD,
    MAX_RETRY_ATTEMPTS,
    NUMBER_OF_BATCHES,
    RAW_DATA__TG_POSTS,
    STAGING_DATA__POSTS,
)
from _production.config.config_db import (
    RAW_TO_STAGING__WHERE,
    STAGING_DATA__POSTS__COLUMNS,
)
from _production.utils.agents import enhanced_cv_matching
from _production.utils.common import setup_logging
from _production.utils.llm import (
    job_post_detection,
    job_post_parsing,
    match_cv_with_job,
    single_job_post_detection,
)
from _production.utils.sql import batch_insert_to_db, fetch_from_db

setup_logging(__file__[:-3])

GDOCS_TIMEOUT_SECONDS = 8
MIN_CV_LENGTH = 128


def validate_cv_content(cv_content: str) -> bool:
    """
    Validate CV content meets minimum requirements.

    Args:
        cv_content: String content of the CV
    Returns:
        bool: True if valid, False otherwise
    """
    if not cv_content or len(cv_content.strip()) < MIN_CV_LENGTH:
        logging.error(
            f"CV content too short or empty. Length: {len(cv_content) if cv_content else 0}"
        )
        return False
    return True


@retry(
    stop=stop_after_attempt(DATA_BATCH_SIZE),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry_error_callback=lambda retry_state: logging.error(
        f"All retry attempts failed for CV fetch after {retry_state.attempt_number} attempts"
    ),
)
def fetch_cv_content(doc_id: str) -> str | None:
    """
    Fetch CV content from Google Docs with retry logic and timeout.

    Args:
        doc_id: Google Doc ID
    Returns:
        Optional[str]: CV content if successful, None if all retries fail
    """
    try:
        response = requests.get(
            f"https://docs.google.com/document/d/{doc_id}/export?format=txt",
            timeout=GDOCS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.text

        if not validate_cv_content(content):
            raise ValueError("CV content validation failed")

        return content
    except requests.RequestException as error:
        logging.debug(f"CV fetch failed: {error!s}", exc_info=True)
        raise


def process_batch(batch_df: pd.DataFrame, cv_content: str) -> pd.DataFrame | None:
    """
    Process a batch of posts against CV content for job matching.

    Args:
        batch_df: DataFrame containing posts to be processed
        cv_content: String content of the CV to match against

    Returns:
        DataFrame with processed job posts or None if processing fails
    """
    try:
        # Initialize columns with appropriate default values
        batch_df.loc[:, "is_job_post"] = False
        batch_df.loc[:, "is_single_job_post"] = False
        batch_df.loc[:, "score"] = 0.0  # Changed from 0 to 0.0 to ensure float dtype
        batch_df.loc[:, "post_structured"] = "{}"

        # Job post detection
        job_post_mask = batch_df["post"].apply(job_post_detection)
        batch_df.loc[job_post_mask, "is_job_post"] = True

        # Single job post detection - only for confirmed job posts
        single_post_mask = (batch_df["is_job_post"]) & (
            batch_df[batch_df["is_job_post"]]["post"].apply(single_job_post_detection)
        )
        batch_df.loc[single_post_mask, "is_single_job_post"] = True

        # First pass: Quick scoring with optimized function
        score_mask_simple = batch_df["is_single_job_post"] & batch_df["is_job_post"]

        # Safely handle score assignment with proper error handling
        def safe_match_score(post: str) -> float:
            try:
                score = match_cv_with_job(cv_content, post)
                return float(score) if score is not None else 0.0
            except Exception as error:
                logging.warning(f"Error in match_cv_with_job: {error}")
                return 0.0

        batch_df.loc[score_mask_simple, "score"] = batch_df.loc[
            score_mask_simple, "post"
        ].apply(safe_match_score)

        # Second pass: Detailed scoring for promising candidates
        score_mask_advanced = (
            batch_df["is_single_job_post"] & batch_df["is_job_post"]
        ) & (batch_df["score"] >= MATCH_SCORE_THRESHOLD)

        # Safely handle enhanced CV matching with proper error handling
        def safe_enhanced_matching(post: str) -> float:
            try:
                score = enhanced_cv_matching(cv_content, post)
                return float(score) if score is not None else 0.0
            except Exception as error:
                logging.warning(f"Error in enhanced_cv_matching: {error}")
                return 0.0

        batch_df.loc[score_mask_advanced, "score"] = batch_df.loc[
            score_mask_advanced, "post"
        ].apply(safe_enhanced_matching)

        # Post parsing - only for posts meeting all criteria
        parsing_mask = (
            batch_df["is_single_job_post"]
            & batch_df["is_job_post"]
            & (batch_df["score"] >= MATCH_SCORE_THRESHOLD)
        )

        # Safely handle job post parsing with proper error handling
        def safe_job_parsing(post: str) -> str:
            try:
                parsed_data = job_post_parsing(post)
                if parsed_data:
                    return json.dumps(parsed_data)
                return "{}"
            except Exception as error:
                logging.warning(f"Error in job_post_parsing: {error}")
                return "{}"

        batch_df.loc[parsing_mask, "post_structured"] = batch_df.loc[
            parsing_mask, "post"
        ].apply(safe_job_parsing)

        # Set timestamp
        batch_df.loc[:, "created_at"] = pd.Timestamp(
            datetime.datetime.now(datetime.UTC)
        ).tz_localize(None)

        # Return only rows with valid structured posts
        return batch_df[batch_df["post_structured"] != "{}"]

    except Exception as error:
        logging.error(f"Error processing batch: {error!s}", exc_info=True)
        return None


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def process_batch_with_retry(
    batch_df: pd.DataFrame, cv_content: str
) -> pd.DataFrame | None:
    """
    Process batch with retry logic.

    Args:
        batch_df: DataFrame containing batch data
        cv_content: CV content string
    Returns:
        Optional[pd.DataFrame]: Processed DataFrame or None if processing fails
    """
    try:
        return process_batch(batch_df, cv_content)
    except Exception as error:
        logging.error(f"Batch processing failed: {error!s}", exc_info=True)
        raise


def clean_and_move_data():
    """Main function to clean and move data from raw to staging."""
    try:
        # Set multiprocessing start method to 'spawn' to avoid fork-related issues
        import multiprocessing

        if hasattr(multiprocessing, "set_start_method"):
            with contextlib.suppress(RuntimeError):
                multiprocessing.set_start_method("spawn", force=True)

        # Fetch and validate CV content
        cv_content = fetch_cv_content(CV_DOC_ID)

        # Fetch raw data
        columns, data = fetch_from_db(
            RAW_DATA__TG_POSTS,
            select_condition="*",
            where_condition=RAW_TO_STAGING__WHERE,
            random_limit=DATA_BATCH_SIZE * NUMBER_OF_BATCHES,
        )
        df = pd.DataFrame(data, columns=columns)

        if df.empty:
            logging.info("No data to process: DataFrame is empty")
            return

        # Process batches with retry logic
        for batch_num, i in enumerate(range(0, len(df), DATA_BATCH_SIZE), 1):
            try:
                batch_df = df[i : i + DATA_BATCH_SIZE].copy()

                # Ensure proper column dtypes before processing
                if "score" in batch_df.columns:
                    batch_df["score"] = batch_df["score"].astype(float)

                processed_df = process_batch_with_retry(batch_df, str(cv_content))

                if processed_df is not None and not processed_df.empty:
                    # Ensure all numeric columns are properly converted to the right type
                    if "score" in processed_df.columns:
                        processed_df["score"] = processed_df["score"].astype(float)

                    records = processed_df.to_dict(orient="records")
                    batch_insert_to_db(
                        STAGING_DATA__POSTS,
                        STAGING_DATA__POSTS__COLUMNS,
                        ["id"],
                        [{str(k): v for k, v in record.items()} for record in records],
                    )
                    logging.info(
                        f"Processed and loaded batch {batch_num}, size: {len(records)}"
                    )
                else:
                    logging.info(f"No valid records in batch {batch_num}")

            except Exception as batch_error:
                logging.error(
                    f"Batch {batch_num} processing failed: {batch_error!s}",
                    exc_info=True,
                )
                continue

        logging.info("Successfully processed all data batches")

    except Exception as error:
        logging.error("Data pipeline failed", exc_info=True)
        raise Exception(f"Data pipeline failed: {error!s}") from error


if __name__ == "__main__":
    clean_and_move_data()
