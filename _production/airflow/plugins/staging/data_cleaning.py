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
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from _production import (
    CV_DOC_ID,
    DATA_BATCH_SIZE,
    GDOCS_TIMEOUT_SECONDS,
    MATCH_SCORE_THRESHOLD,
    MIN_CV_LENGTH,
    NUMBER_OF_BATCHES,
    RAW_DATA__TG_POSTS,
    STAGING_DATA__POSTS,
)
from _production.config.config_db import (
    RAW_TO_STAGING__WHERE,
    STAGING_DATA__POSTS__COLUMNS,
)
from _production.utils.common import setup_logging
from _production.utils.influxdb import store_metrics
from _production.utils.llm import (
    job_post_detection,
    job_post_parsing,
    match_cv_with_job,
    single_job_post_detection,
)
from _production.utils.sql import batch_insert_to_db, fetch_from_db

setup_logging(__file__[:-3])


@dataclass
class CleaningStats:
    """Statistics for the data cleaning process."""

    total_posts: int = 0
    job_posts: int = 0
    single_job_posts: int = 0
    posts_above_threshold: int = 0
    posts_below_threshold: int = 0
    posts_with_structured_data: int = 0
    posts_without_structured_data: int = 0
    scores: list[float] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a dictionary for metrics storage."""
        if not self.scores:
            return {
                "total_posts": self.total_posts,
                "job_posts_rate": 0.0,
                "single_job_posts_rate": 0.0,
                "above_threshold_rate": 0.0,
                "below_threshold_rate": 0.0,
                "structured_data_rate": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "execution_time_ms": self.execution_time_ms,
            }

        # Calculate rates
        job_posts_rate = (
            (self.job_posts / self.total_posts * 100) if self.total_posts > 0 else 0.0
        )
        single_job_posts_rate = (
            (self.single_job_posts / self.job_posts * 100)
            if self.job_posts > 0
            else 0.0
        )
        above_threshold_rate = (
            (self.posts_above_threshold / self.single_job_posts * 100)
            if self.single_job_posts > 0
            else 0.0
        )
        below_threshold_rate = (
            (self.posts_below_threshold / self.single_job_posts * 100)
            if self.single_job_posts > 0
            else 0.0
        )
        structured_data_rate = (
            (self.posts_with_structured_data / self.posts_above_threshold * 100)
            if self.posts_above_threshold > 0
            else 0.0
        )

        # Calculate score statistics
        score_mean = sum(self.scores) / len(self.scores)
        score_std = (
            sum((x - score_mean) ** 2 for x in self.scores) / len(self.scores)
        ) ** 0.5

        return {
            "total_posts": int(self.total_posts),
            "job_posts_rate": float(job_posts_rate),
            "single_job_posts_rate": float(single_job_posts_rate),
            "above_threshold_rate": float(above_threshold_rate),
            "below_threshold_rate": float(below_threshold_rate),
            "structured_data_rate": float(structured_data_rate),
            "score_mean": float(score_mean),
            "score_std": float(score_std),
            "execution_time_ms": float(self.execution_time_ms),
        }


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


def process_batch(
    batch_df: pd.DataFrame,
    cv_content: str,
    batch_num: int,
    total_batches: int,
    stats: CleaningStats,
) -> pd.DataFrame | None:
    """
    Process a batch of posts against CV content for job matching.

    Args:
        batch_df: DataFrame containing posts to be processed
        cv_content: String content of the CV to match against
        batch_num: Current batch number (1-based)
        total_batches: Total number of batches to process
        stats: Statistics object for tracking metrics

    Returns:
        DataFrame with processed job posts or None if processing fails
    """
    try:
        initial_count = len(batch_df)
        stats.total_posts += initial_count
        logging.info("=" * 80)
        logging.info(
            f"🔄 BATCH {batch_num}/{total_batches} PROCESSING: Starting with {initial_count} posts"
        )
        logging.info("=" * 80)

        # Initialize columns with appropriate default values
        batch_df.loc[:, "is_job_post"] = None
        batch_df.loc[:, "is_single_job_post"] = None
        batch_df.loc[:, "score"] = None
        batch_df.loc[:, "post_structured"] = "{}"
        batch_df.loc[:, "parsing_error"] = False
        logging.info("✓ Initialized default values for all columns")

        # Job post detection
        logging.info("-" * 40)
        logging.info("STEP 1: Job Post Detection")
        logging.info("-" * 40)
        job_post_mask = batch_df["post"].apply(job_post_detection)
        batch_df.loc[job_post_mask, "is_job_post"] = True
        batch_df.loc[~job_post_mask, "is_job_post"] = False
        job_posts_count = job_post_mask.sum()
        stats.job_posts += job_posts_count

        # Calculate state statistics
        true_count = batch_df["is_job_post"].sum()
        false_count = (batch_df["is_job_post"] == False).sum()
        none_count = batch_df["is_job_post"].isna().sum()

        logging.info(
            f"📊 Results:\n"
            f"  - Total posts: {initial_count}\n"
            f"  - Job posts found: {job_posts_count}\n"
            f"  - Success rate: {(job_posts_count/initial_count)*100:.1f}%\n"
            f"  - State distribution:\n"
            f"    • True: {true_count} ({true_count/initial_count*100:.1f}%)\n"
            f"    • False: {false_count} ({false_count/initial_count*100:.1f}%)\n"
            f"    • None: {none_count} ({none_count/initial_count*100:.1f}%)"
        )

        # Single job post detection
        logging.info("\n" + "-" * 40)
        logging.info("STEP 2: Single Job Post Detection")
        logging.info("-" * 40)
        single_post_mask = batch_df["is_job_post"] & (
            batch_df[batch_df["is_job_post"]]["post"].apply(single_job_post_detection)
        )
        batch_df.loc[single_post_mask, "is_single_job_post"] = True
        batch_df.loc[
            batch_df["is_job_post"] & ~single_post_mask, "is_single_job_post"
        ] = False
        single_posts_count = single_post_mask.sum()
        stats.single_job_posts += single_posts_count

        # Calculate state statistics
        true_count = batch_df["is_single_job_post"].sum()
        false_count = (batch_df["is_single_job_post"] == False).sum()
        none_count = batch_df["is_single_job_post"].isna().sum()

        logging.info(
            f"📊 Results:\n"
            f"  - Job posts analyzed: {job_posts_count}\n"
            f"  - Single posts found: {single_posts_count}\n"
            f"  - Success rate: {(single_posts_count/job_posts_count)*100:.1f}% of job posts\n"
            f"  - State distribution:\n"
            f"    • True: {true_count} ({true_count/initial_count*100:.1f}%)\n"
            f"    • False: {false_count} ({false_count/initial_count*100:.1f}%)\n"
            f"    • None: {none_count} ({none_count/initial_count*100:.1f}%)"
        )

        # First pass scoring
        logging.info("\n" + "-" * 40)
        logging.info("STEP 3: Initial CV Matching Score")
        logging.info("-" * 40)
        # Create a mask that only includes True values, excluding None
        score_mask_simple = batch_df["is_single_job_post"].fillna(False)
        posts_to_score = score_mask_simple.sum()
        logging.info(f"⚡ Quick scoring {posts_to_score} eligible posts")

        def safe_match_score(post: str) -> float | None:
            try:
                score = match_cv_with_job(cv_content, post)
                return float(score) if score is not None else None
            except Exception as error:
                logging.warning(f"⚠️ Error in match_cv_with_job: {error}")
                return None

        # Apply scoring only to posts where is_single_job_post is True
        batch_df.loc[score_mask_simple, "score"] = batch_df.loc[
            score_mask_simple, "post"
        ].apply(safe_match_score)

        # Track scores for statistics
        valid_scores = batch_df.loc[score_mask_simple, "score"].dropna()
        stats.scores.extend(valid_scores.tolist())

        # Score threshold analysis
        above_threshold = batch_df[batch_df["score"] >= MATCH_SCORE_THRESHOLD]
        threshold_count = len(above_threshold)
        stats.posts_above_threshold += threshold_count
        stats.posts_below_threshold += len(
            batch_df[score_mask_simple & (batch_df["score"] < MATCH_SCORE_THRESHOLD)]
        )

        # Calculate score statistics
        valid_scores = batch_df["score"].notna().sum()
        null_scores = batch_df["score"].isna().sum()

        logging.info(
            f"📊 Results:\n"
            f"  - Posts scored: {posts_to_score}\n"
            f"  - Above threshold ({MATCH_SCORE_THRESHOLD}): {threshold_count}\n"
            f"  - Success rate: {(threshold_count/posts_to_score)*100:.1f}% passed threshold\n"
            f"  - Score distribution:\n"
            f"    • Valid scores: {valid_scores} ({valid_scores/initial_count*100:.1f}%)\n"
            f"    • NULL scores: {null_scores} ({null_scores/initial_count*100:.1f}%)"
        )

        # Final parsing
        logging.info("\n" + "-" * 40)
        logging.info("STEP 5: Structured Data Parsing")
        logging.info("-" * 40)
        parsing_mask = batch_df["is_single_job_post"] & (
            batch_df["score"] >= MATCH_SCORE_THRESHOLD
        )
        posts_to_parse = parsing_mask.sum()

        # Calculate parsing statistics
        parsing_errors = batch_df["parsing_error"].sum()
        valid_parsing = (batch_df["post_structured"] != "{}").sum()
        stats.posts_with_structured_data += valid_parsing
        stats.posts_without_structured_data += posts_to_parse - valid_parsing

        logging.info(
            f"📝 Parsing {posts_to_parse} posts:\n"
            f"  - Met score threshold: {(batch_df['score'] >= MATCH_SCORE_THRESHOLD).sum()}\n"
            f"  - With NULL scores: {null_scores}\n"
            f"  - With parsing errors: {parsing_errors}\n"
            f"  - Parsing results:\n"
            f"    • Valid parsing: {valid_parsing} ({valid_parsing/initial_count*100:.1f}%)\n"
            f"    • Empty results: {posts_to_parse - valid_parsing} ({((posts_to_parse - valid_parsing)/initial_count)*100:.1f}%)"
        )

        def safe_job_parsing(post: str) -> str:
            try:
                parsed_data = job_post_parsing(post)
                if not parsed_data:
                    logging.info("⚠️ job_post_parsing returned empty result")
                    return "{}"

                # Handle case where LLM returns action-based format
                if isinstance(parsed_data, dict) and "action" in parsed_data:
                    logging.info(
                        f"⚠️ LLM returned action format instead of job post data: {json.dumps(parsed_data)[:200]}..."
                    )
                    return "{}"

                # Validate the parsed data structure
                if not isinstance(parsed_data, dict):
                    logging.info(f"⚠️ Unexpected parsed_data type: {type(parsed_data)}")
                    return "{}"

                try:
                    # Attempt to serialize to ensure valid JSON
                    json_str = json.dumps(parsed_data)
                    return json_str
                except (TypeError, ValueError) as json_error:
                    logging.info(
                        f"⚠️ Failed to serialize parsed data: {json_error}. Data: {parsed_data}"
                    )
                    return "{}"

            except Exception as error:
                logging.info(
                    f"⚠️ Error in job_post_parsing: {error}\nPost preview: {post[:200]}...",
                    exc_info=True,
                )
                return "{}"

        batch_df.loc[parsing_mask, "post_structured"] = batch_df.loc[
            parsing_mask, "post"
        ].apply(safe_job_parsing)

        # Set timestamp and prepare final results
        batch_df.loc[:, "created_at"] = pd.Timestamp(
            datetime.datetime.now(datetime.UTC)
        ).tz_localize(None)

        # Final results with strict filtering
        final_df = batch_df[
            (batch_df["is_job_post"])  # Only True job posts
            & (
                (batch_df["score"] >= MATCH_SCORE_THRESHOLD)  # Score above threshold
                | (batch_df["score"].isna())  # Or score is None
            )
        ]
        final_count = len(final_df)

        logging.info("\n" + "=" * 80)
        logging.info("🏁 FINAL RESULTS")
        logging.info("=" * 80)
        logging.info(
            f"Pipeline Summary:\n"
            f"  Initial posts: {initial_count}\n"
            f"  ↳ Job posts: {job_posts_count} ({(job_posts_count/initial_count)*100:.1f}%)\n"
            f"    ↳ Single posts: {single_posts_count} ({(single_posts_count/job_posts_count)*100:.1f}%)\n"
            f"      ↳ Above threshold: {threshold_count} ({(threshold_count/single_posts_count)*100:.1f}%)\n"
            f"        ↳ Final valid posts: {final_count} ({(final_count/initial_count)*100:.1f}% of initial)"
        )
        logging.info("=" * 80)

        return final_df

    except Exception as error:
        logging.error(f"❌ Error processing batch: {error!s}", exc_info=True)
        return None


def clean_and_move_data():
    """Main function to clean and move data from raw to staging."""
    try:
        # Set multiprocessing start method to 'spawn' to avoid fork-related issues
        import multiprocessing

        if hasattr(multiprocessing, "set_start_method"):
            with contextlib.suppress(RuntimeError):
                multiprocessing.set_start_method("spawn", force=True)

        # Initialize statistics
        stats = CleaningStats()
        start_time = time.time_ns()

        # Fetch and validate CV content
        cv_content = fetch_cv_content(CV_DOC_ID)

        # Fetch raw data
        columns, data = fetch_from_db(
            RAW_DATA__TG_POSTS,
            select_condition="*",
            where_condition=RAW_TO_STAGING__WHERE,
            order_by_condition="created_at DESC",
            limit=DATA_BATCH_SIZE * NUMBER_OF_BATCHES,
        )
        df = pd.DataFrame(data, columns=columns)

        if df.empty:
            logging.info("No data to process: DataFrame is empty")
            return

        total_batches = (len(df) + DATA_BATCH_SIZE - 1) // DATA_BATCH_SIZE
        logging.info(
            f"Starting processing of {len(df)} records in {total_batches} batches"
        )

        # Process batches
        for batch_num, i in enumerate(range(0, len(df), DATA_BATCH_SIZE), 1):
            try:
                batch_df = df[i : i + DATA_BATCH_SIZE].copy()

                # Ensure proper column dtypes before processing
                if "score" in batch_df.columns:
                    batch_df["score"] = batch_df["score"].astype(float)

                processed_df = process_batch(
                    batch_df, str(cv_content), batch_num, total_batches, stats
                )

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

        # Calculate execution time
        stats.execution_time_ms = float((time.time_ns() - start_time) / 1_000_000)

        # Store metrics in InfluxDB
        store_metrics(
            measurement="tg-job-radar__data_cleaning__stats",
            fields=stats.to_dict(),
            tags={
                "environment": "production",
                "script": "data_cleaning",
            },
        )

        logging.info("Successfully processed all data batches")

    except Exception as error:
        logging.error("Data pipeline failed", exc_info=True)
        raise Exception(f"Data pipeline failed: {error!s}") from error


if __name__ == "__main__":
    clean_and_move_data()
