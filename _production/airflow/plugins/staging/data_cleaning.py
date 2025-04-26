"""
Data cleaning module for processing job posts and matching them against CV content.

This module handles the ETL process from raw data to staging, including:
- Structured data parsing and storage
"""

import contextlib
import datetime
import json
import logging
import multiprocessing
import re
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
    count_tokens,
    job_post_detection,
    rewrite_job_post,
    single_job_post_detection,
    summarize_cv_content,
)
from _production.utils.sql import batch_insert_to_db, fetch_from_db
from _production.utils.text import extensive_clean_text

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
    parsing_errors: int = 0  # Track parsing errors
    valid_parsing: int = 0  # Track successful parsing
    empty_parsing: int = 0  # Track empty parsing results
    scores: list[float] = field(default_factory=list)
    execution_time_ms: float = 0.0
    server_errors: int = 0  # Track server errors

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
                "parsing_error_rate": 0.0,
                "valid_parsing_rate": 0.0,
                "empty_parsing_rate": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "execution_time_ms": self.execution_time_ms,
                "server_errors": self.server_errors,
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
        parsing_error_rate = (
            (self.parsing_errors / self.total_posts * 100)
            if self.total_posts > 0
            else 0.0
        )
        valid_parsing_rate = (
            (self.valid_parsing / self.total_posts * 100)
            if self.total_posts > 0
            else 0.0
        )
        empty_parsing_rate = (
            (self.empty_parsing / self.total_posts * 100)
            if self.total_posts > 0
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
            "parsing_error_rate": float(parsing_error_rate),
            "valid_parsing_rate": float(valid_parsing_rate),
            "empty_parsing_rate": float(empty_parsing_rate),
            "score_mean": float(score_mean),
            "score_std": float(score_std),
            "execution_time_ms": float(self.execution_time_ms),
            "server_errors": self.server_errors,
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


def safe_job_post_detection(post: str) -> bool | None:
    """
    Safely detect if a post contains a job posting with proper error handling and logging.

    Args:
        post: Text to analyze

    Returns:
        Optional[bool]: True if job posting detected, False if not, None if error
    """
    try:
        # Validate input
        if not post or not post.strip():
            logging.warning("Empty post received for job post detection")
            return None

        # Count tokens and log input
        token_count = count_tokens(post)
        logging.info(f"Input tokens: {token_count}")

        # Log first 256 chars for better context
        preview = post[:256] + "..." if len(post) > 256 else post
        logging.info(f"Input preview:\n{preview}")

        # Check if input is too long
        if token_count > 4000:  # GPT-4 context window is 8k, leave room for prompt
            logging.warning(
                f"Input too long ({token_count} tokens), truncating to 4000 tokens"
            )
            post = post[:4000]  # Approximate truncation

        # Call LLM
        logging.info("Requesting job post detection from LLM...")
        result = job_post_detection(post)

        # Log result
        if result is None:
            logging.error("LLM returned None for job post detection")
            return None

        logging.info(f"Job post detection result: {result}")

        # Add delay to prevent rate limiting
        time.sleep(1.5)

        return result

    except Exception as error:
        logging.error(f"Error in job post detection: {error!s}", exc_info=True)
        return None


def safe_single_job_detection(post: str) -> bool | None:
    """
    Safely detect if a post contains exactly one job posting with proper error handling and logging.

    Args:
        post: Text to analyze

    Returns:
        Optional[bool]: True if exactly one job posting detected, False if not, None if error
    """
    try:
        # Validate input
        if not post or not post.strip():
            logging.warning("Empty post received for single job detection")
            return None

        # Count tokens and log input
        token_count = count_tokens(post)
        logging.info(f"Input tokens: {token_count}")

        # Log first 256 chars for better context
        preview = post[:256] + "..." if len(post) > 256 else post
        logging.info(f"Input preview:\n{preview}")

        # Check if input is too long
        if token_count > 4000:  # GPT-4 context window is 8k, leave room for prompt
            logging.warning(
                f"Input too long ({token_count} tokens), truncating to 4000 tokens"
            )
            post = post[:4000]  # Approximate truncation

        # Call LLM
        logging.info("Requesting single job detection from LLM...")
        result = single_job_post_detection(post)

        # Log result
        if result is None:
            logging.error("LLM returned None for single job detection")
            return None

        logging.info(f"Single job detection result: {result}")

        # Add delay to prevent rate limiting
        time.sleep(1.5)

        return result

    except Exception as error:
        logging.error(f"Error in single job detection: {error!s}", exc_info=True)
        return None


def process_batch(
    batch_df: pd.DataFrame,
    cv_content: str,
    batch_num: int,
    total_batches: int,
    stats: CleaningStats,
) -> pd.DataFrame | None:
    """Process a batch of Telegram posts.

    Args:
        batch_df: DataFrame containing posts to process
        cv_content: Content of the CV document
        batch_num: Current batch number
        total_batches: Total number of batches
        stats: Statistics object

    Returns:
        DataFrame with processed job posts or None if processing fails
    """
    try:
        initial_count = len(batch_df)
        if initial_count == 0:
            logging.info(f"Batch {batch_num}/{total_batches} is empty, skipping...")
            return None

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

        # Step 1: Extensive cleaning
        logging.info("\nSTEP 1: Extensive Text Cleaning")
        logging.info("-" * 40)
        batch_df.loc[:, "clean_post"] = batch_df["post"].apply(extensive_clean_text)
        batch_df = batch_df.loc[batch_df["clean_post"].notna()]

        if batch_df.empty:
            logging.info("❌ No valid posts found after cleaning, skipping batch")
            return None

        logging.info(f"✓ Cleaned {len(batch_df)} posts out of {initial_count}")

        # Step 2: Job post detection
        logging.info("\nSTEP 2: Job Post Detection")
        logging.info("-" * 40)
        batch_df.loc[:, "is_job_post"] = batch_df["clean_post"].apply(
            safe_job_post_detection
        )
        job_posts_df = batch_df.loc[batch_df["is_job_post"]]

        if job_posts_df.empty:
            logging.info("❌ No job posts found, skipping batch")
            return None

        # Calculate job post detection statistics
        job_posts_count = len(job_posts_df)
        stats.job_posts += job_posts_count
        true_count = job_posts_df["is_job_post"].sum()
        false_count = (batch_df["is_job_post"] == False).sum()
        none_count = batch_df["is_job_post"].isna().sum()

        logging.info(
            f"📊 Job Post Detection Results:\n"
            f"  - Total posts: {initial_count}\n"
            f"  - Job posts found: {job_posts_count}\n"
            f"  - Success rate: {(job_posts_count/initial_count)*100:.1f}%\n"
            f"  - State distribution:\n"
            f"    • True: {true_count} ({true_count/initial_count*100:.1f}%)\n"
            f"    • False: {false_count} ({false_count/initial_count*100:.1f}%)\n"
            f"    • None: {none_count} ({none_count/initial_count*100:.1f}%)"
        )

        # Add a delay between steps
        logging.info("⏳ Pausing for 3 seconds between processing steps")
        time.sleep(3)

        # Step 3: Single job post detection
        logging.info("\nSTEP 3: Single Job Post Detection")
        logging.info("-" * 40)
        job_posts_df.loc[:, "is_single_job_post"] = job_posts_df["clean_post"].apply(
            safe_single_job_detection
        )
        single_job_posts_df = job_posts_df.loc[job_posts_df["is_single_job_post"]]

        if single_job_posts_df.empty:
            logging.info("❌ No single job posts found, skipping batch")
            return None

        # Calculate single job post detection statistics
        single_posts_count = len(single_job_posts_df)
        stats.single_job_posts += single_posts_count
        true_count = single_job_posts_df["is_single_job_post"].sum()
        false_count = (job_posts_df["is_single_job_post"] == False).sum()
        none_count = job_posts_df["is_single_job_post"].isna().sum()

        logging.info(
            f"📊 Single Job Post Detection Results:\n"
            f"  - Job posts analyzed: {job_posts_count}\n"
            f"  - Single posts found: {single_posts_count}\n"
            f"  - Success rate: {(single_posts_count/job_posts_count)*100:.1f}% of job posts\n"
            f"  - State distribution:\n"
            f"    • True: {true_count} ({true_count/job_posts_count*100:.1f}%)\n"
            f"    • False: {false_count} ({false_count/job_posts_count*100:.1f}%)\n"
            f"    • None: {none_count} ({none_count/job_posts_count*100:.1f}%)"
        )

        # Add a delay between steps
        logging.info("⏳ Pausing for 3 seconds between processing steps")
        time.sleep(3)

        # Step 4: Job Post Compression
        logging.info("\nSTEP 4: Job Post Compression")
        logging.info("-" * 40)
        single_job_posts_df.loc[:, "compressed_post"] = single_job_posts_df[
            "clean_post"
        ].apply(rewrite_job_post)
        compressed_posts_df = single_job_posts_df.loc[
            single_job_posts_df["compressed_post"].notna()
        ]

        if compressed_posts_df.empty:
            logging.info("❌ No posts could be compressed, skipping batch")
            return None

        # Calculate compression statistics
        compressed_count = len(compressed_posts_df)
        stats.posts_with_structured_data += compressed_count
        stats.posts_without_structured_data += (
            len(single_job_posts_df) - compressed_count
        )

        logging.info(
            f"📊 Compression Results:\n"
            f"  - Posts compressed: {compressed_count}\n"
            f"  - Success rate: {(compressed_count/single_posts_count)*100:.1f}%"
        )

        # Add a delay between steps
        logging.info("⏳ Pausing for 3 seconds between processing steps")
        time.sleep(3)

        # Step 5: Job Post Parsing
        logging.info("\nSTEP 5: Job Post Parsing")
        logging.info("-" * 40)

        def safe_job_parsing(post: str) -> str:
            """Clean and format job post text for structured storage.

            Args:
                post: Raw job post text

            Returns:
                JSON string containing cleaned post data
            """
            try:
                token_count = count_tokens(post)
                logging.info(f"Input: {post[:128]}... (tokens: {token_count})")
                logging.info("Request to job_post_parsing...")

                # Replace multiple spaces with single space
                cleaned = re.sub(r"\s+", " ", post)
                # Replace multiple newlines with single newline
                cleaned = re.sub(r"\n+", "\n", cleaned)
                # Cut to 1024 chars
                cleaned = cleaned[:1024]

                # Create structured data
                structured_data = {
                    "description": cleaned,
                    "full_description": post,  # Keep original for reference
                }

                # Convert to JSON string
                return json.dumps(structured_data)
            except Exception as error:
                logging.error(f"Error in job_post_parsing: {error!s}", exc_info=True)
                return json.dumps({"full_description": post})

        # Apply parsing to all compressed posts
        compressed_posts_df.loc[:, "post_structured"] = compressed_posts_df[
            "compressed_post"
        ].apply(safe_job_parsing)

        # Log parsing results
        parsed_count = len(compressed_posts_df)
        stats.posts_with_structured_data += parsed_count
        stats.posts_without_structured_data += len(compressed_posts_df) - parsed_count

        logging.info(
            f"📊 Parsing Results:\n"
            f"  - Posts parsed: {parsed_count}\n"
            f"  - Success rate: {(parsed_count/compressed_count)*100:.1f}%"
        )

        # Add a delay between steps
        logging.info("⏳ Pausing for 3 seconds between processing steps")
        time.sleep(3)

        ###

        parsed_posts_df = compressed_posts_df.loc[
            compressed_posts_df["post_structured"] != "{}"
        ]

        if parsed_posts_df.empty:
            logging.info("❌ No single job posts found, skipping batch")
            return None

        # Calculate parsing statistics
        parsing_errors = parsed_posts_df["parsing_error"].sum()
        valid_parsing = (compressed_posts_df["post_structured"] != "{}").sum()
        empty_parsing = initial_count - valid_parsing

        # Update stats
        stats.parsing_errors += parsing_errors
        stats.valid_parsing += valid_parsing
        stats.empty_parsing += empty_parsing

        logging.info(
            f"📝 Parsing results:\n"
            f"  - Total posts: {initial_count}\n"
            f"  - With parsing errors: {parsing_errors}\n"
            f"  - Parsing results:\n"
            f"    • Valid parsing: {valid_parsing} ({valid_parsing/initial_count*100:.1f}%)\n"
            f"    • Empty results: {empty_parsing} ({empty_parsing/initial_count*100:.1f}%)"
        )

        # Set timestamp and prepare final results
        parsed_posts_df.loc[:, "created_at"] = pd.Timestamp(
            datetime.datetime.now(datetime.UTC)
        ).tz_localize(None)

        # Final results
        final_df = parsed_posts_df.copy()
        final_count = len(final_df)

        if final_count == 0:
            logging.info("No valid posts found in this batch after processing")
            return None

        logging.info("\n" + "=" * 80)
        logging.info("🏁 FINAL RESULTS")
        logging.info("=" * 80)
        logging.info(
            f"Pipeline Summary:\n"
            f"  Initial posts: {initial_count}\n"
            f"  Final valid posts: {final_count} ({(final_count/initial_count)*100:.1f}% of initial)"
        )
        logging.info("=" * 80)

        return final_df

    except Exception as error:
        logging.error(f"❌ Error processing batch: {error!s}", exc_info=True)
        return None


def clean_and_move_data():
    """Main function to clean and move data from raw to staging."""
    try:
        if hasattr(multiprocessing, "set_start_method"):
            with contextlib.suppress(RuntimeError):
                multiprocessing.set_start_method("spawn", force=True)

        # Initialize statistics
        stats = CleaningStats()
        start_time = time.time_ns()

        # Fetch and validate CV content
        cv_content = fetch_cv_content(CV_DOC_ID)

        if not cv_content:
            raise ValueError("CV content is empty")

        cv_summary = summarize_cv_content(cv_content)

        if not cv_summary:
            raise ValueError("CV summary is empty")

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
            f"Starting processing of {len(df)} records in {total_batches} batches (batch size: {DATA_BATCH_SIZE})"
        )

        # Process batches
        for batch_num in range(1, total_batches + 1):
            try:
                start_idx = (batch_num - 1) * DATA_BATCH_SIZE
                end_idx = min(start_idx + DATA_BATCH_SIZE, len(df))
                batch_df = df.iloc[start_idx:end_idx].copy()

                # Log batch start with clear visual separation
                logging.info("\n" + "=" * 80)
                logging.info(f"🔄 STARTING BATCH {batch_num}/{total_batches}")
                logging.info("=" * 80)

                # Ensure proper column dtypes before processing
                if "score" in batch_df.columns:
                    batch_df["score"] = batch_df["score"].astype(float)

                processed_df = process_batch(
                    batch_df,
                    cv_summary,
                    batch_num,
                    total_batches,
                    stats,
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
                        f"✅ Successfully processed and loaded batch {batch_num}/{total_batches}, size: {len(records)}"
                    )
                else:
                    logging.info(
                        f"⚠️ No valid records in batch {batch_num}/{total_batches}"
                    )

                # Add a delay between batches to allow the model server to recover
                if batch_num % 10 == 0:
                    logging.info(
                        f"⏳ Taking a 30-second break after batch {batch_num} to let the model server recover"
                    )
                    time.sleep(30)
                else:
                    logging.info(
                        f"⏳ Small pause between batches ({batch_num}/{total_batches})"
                    )
                    time.sleep(3)

            except Exception as batch_error:
                logging.error(
                    f"❌ Batch {batch_num}/{total_batches} processing failed: {batch_error!s}",
                    exc_info=True,
                )
                # Add longer pause after error
                logging.info(
                    f"⏳ Taking a 60-second break after error in batch {batch_num}/{total_batches}"
                )
                time.sleep(60)
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

        logging.info("✅ Successfully processed all data batches")

    except Exception as error:
        logging.error("❌ Data pipeline failed", exc_info=True)
        raise Exception(f"Data pipeline failed: {error!s}") from error


if __name__ == "__main__":
    clean_and_move_data()
