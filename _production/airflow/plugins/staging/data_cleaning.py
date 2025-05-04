"""
Data cleaning module for processing job posts and matching them against CV content.

This module handles the ETL process from raw data to staging, including:
- Structured data parsing and storage
"""

import contextlib
import datetime
import functools
import json
import logging
import multiprocessing
import time
import traceback
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
    clean_job_post_values,
    job_post_detection,
    job_post_parsing,
    match_cv_with_job,
    rewrite_job_post,
    single_job_post_detection,
    summarize_cv_content,
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
    parsing_errors: int = 0  # Track parsing errors
    valid_parsing: int = 0  # Track successful parsing
    empty_parsing: int = 0  # Track empty parsing results
    scores: list[float] = field(default_factory=list)
    execution_time_ms: float = 0.0
    server_errors: int = 0  # Track server errors


def llm_safe_call(default_return=None, input_preview=256):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if args:
                    preview = str(args[0])[:input_preview] + (
                        "..." if len(str(args[0])) > input_preview else ""
                    )
                    logging.info(f"{func.__name__} input: {preview}")
                result = func(*args, **kwargs)
                logging.info(f"{func.__name__} output: {result}")
                return result if result is not None else default_return
            except Exception as error:
                logging.error(
                    f"Error in {func.__name__}: {error!s}\n{traceback.format_exc()}"
                )
                return default_return

        return wrapper

    return decorator


@retry(
    stop=stop_after_attempt(DATA_BATCH_SIZE),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry_error_callback=lambda retry_state: logging.error(
        f"All retry attempts failed for CV fetch after {retry_state.attempt_number} attempts"
    ),
)
@llm_safe_call(default_return=None)
def safe_fetch_cv_content(doc_id: str) -> str | None:
    """
    Fetch CV content from Google Docs with retry logic and timeout. Returns None if content is empty or too short.
    """
    try:
        response = requests.get(
            f"https://docs.google.com/document/d/{doc_id}/export?format=txt",
            timeout=GDOCS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.text
        if not content or len(content.strip()) < MIN_CV_LENGTH:
            logging.error(
                f"CV content too short or empty. Length: {len(content) if content else 0}"
            )
            return None
        return content
    except requests.RequestException as error:
        logging.debug(f"CV fetch failed: {error!s}", exc_info=True)
        return None


@llm_safe_call(default_return=None)
def safe_summarize_cv_content(cv_content: str) -> str | None:
    return summarize_cv_content(cv_content)


@llm_safe_call(default_return=None)
def safe_job_post_detection(post: str) -> bool | None:
    return job_post_detection(post)


@llm_safe_call(default_return=None)
def safe_single_job_detection(post: str) -> bool | None:
    return single_job_post_detection(post)


@llm_safe_call()
def safe_rewrite_job_post(post: str) -> str:
    result = rewrite_job_post(post)
    return result if result is not None else post


@llm_safe_call(default_return=None)
def safe_match_cv_with_job(cv_text: str, post: str) -> float | None:
    """
    Safely match CV with job post, logging input and output. (Demo stub)
    """
    return match_cv_with_job(cv_text, post)


@llm_safe_call(default_return=None)
def safe_job_parsing(post: str, compressed_post: str) -> str:
    """
    Safely parse a job post into structured data. Always returns a JSON string with at least {'description': compressed_post} on error.

    Args:
        post: The original job post text
        compressed_post: The compressed (rewritten) job post text
    Returns:
        str: JSON string of structured data, always with at least 'description'.
    """
    result = job_post_parsing(post, compressed_post)
    return json.dumps(result)


@llm_safe_call(default_return={})
def safe_clean_job_post_values(
    response: dict[str, Any], exclude_fields: list[str] | None = None
) -> dict[str, Any]:
    """
    Safely clean job post values, logging input and output. (Demo stub)
    """
    return clean_job_post_values(response, exclude_fields=exclude_fields)


def process_batch(
    batch_df: pd.DataFrame,
    cv_content: str,
    batch_num: int,
    total_batches: int,
    stats: CleaningStats,
) -> pd.DataFrame | None:
    """Process a batch of Telegram posts with full LLM pipeline and detailed logging."""
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

        # Step 1: Job post detection
        logging.info("\nSTEP 1: Job Post Detection")
        logging.info("-" * 40)
        batch_df.loc[:, "is_job_post"] = batch_df["post"].apply(safe_job_post_detection)
        job_posts_df = batch_df.loc[batch_df["is_job_post"]]
        job_posts_count = len(job_posts_df)
        stats.job_posts += job_posts_count
        true_count = batch_df["is_job_post"].sum()
        false_count = (batch_df["is_job_post"] == False).sum()
        none_count = batch_df["is_job_post"].isna().sum()
        logging.info(
            f"Job post detection input preview: {batch_df['post'].head(3).tolist()}"
        )
        logging.info(
            f"Job post detection output preview: {batch_df['is_job_post'].head(3).tolist()}"
        )
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
        if job_posts_df.empty:
            logging.info("❌ No job posts found, skipping batch")
            return None
        time.sleep(3)

        # Step 2: Single job post detection
        logging.info("\nSTEP 2: Single Job Post Detection")
        logging.info("-" * 40)
        job_posts_df.loc[:, "is_single_job_post"] = job_posts_df["post"].apply(
            safe_single_job_detection
        )
        single_job_posts_df = job_posts_df.loc[job_posts_df["is_single_job_post"]]
        single_posts_count = len(single_job_posts_df)
        stats.single_job_posts += single_posts_count
        true_count = job_posts_df["is_single_job_post"].sum()
        false_count = (job_posts_df["is_single_job_post"] == False).sum()
        none_count = job_posts_df["is_single_job_post"].isna().sum()
        logging.info(
            f"Single job detection input preview: {job_posts_df['post'].head(3).tolist()}"
        )
        logging.info(
            f"Single job detection output preview: {job_posts_df['is_single_job_post'].head(3).tolist()}"
        )
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
        if single_job_posts_df.empty:
            logging.info("❌ No single job posts found, skipping batch")
            return None
        time.sleep(3)

        # Step 3: Job Post Compression (rewrite)
        logging.info("\nSTEP 3: Job Post Compression (Rewrite)")
        logging.info("-" * 40)
        single_job_posts_df.loc[:, "compressed_post"] = single_job_posts_df[
            "post"
        ].apply(safe_rewrite_job_post)
        compressed_posts_df = single_job_posts_df.loc[
            single_job_posts_df["compressed_post"].notna()
        ]
        compressed_count = len(compressed_posts_df)
        rewritten_count = (
            single_job_posts_df["compressed_post"] != single_job_posts_df["post"]
        ).sum()
        fallback_count = (
            single_job_posts_df["compressed_post"] == single_job_posts_df["post"]
        ).sum()
        logging.info(
            f"Rewrite input preview: {single_job_posts_df['post'].head(3).tolist()}"
        )
        logging.info(
            f"Rewrite output preview: {single_job_posts_df['compressed_post'].head(3).tolist()}"
        )
        logging.info(
            f"📊 Job Post Compression Results:\n"
            f"  - Single job posts analyzed: {single_posts_count}\n"
            f"  - Compressed posts: {compressed_count}\n"
            f"  - Rewritten (LLM): {rewritten_count} ({(rewritten_count/single_posts_count)*100:.1f}%)\n"
            f"  - Fallback (original): {fallback_count} ({(fallback_count/single_posts_count)*100:.1f}%)"
        )
        if compressed_posts_df.empty:
            logging.info("❌ No posts could be compressed, skipping batch")
            return None
        time.sleep(3)

        # Step 4: CV Matching (score)
        logging.info("\nSTEP 4: CV Matching (Scoring)")
        logging.info("-" * 40)
        compressed_posts_df.loc[:, "score"] = compressed_posts_df[
            "compressed_post"
        ].apply(
            lambda compressed_post: safe_match_cv_with_job(cv_content, compressed_post)
        )
        filtered_df = compressed_posts_df.loc[
            (compressed_posts_df["score"] == 0)
            | (compressed_posts_df["score"] >= MATCH_SCORE_THRESHOLD)
        ]
        filtered_count = len(filtered_df)
        scored_count = len(compressed_posts_df)
        zero_count = (compressed_posts_df["score"] == 0).sum()
        nonzero_count = (compressed_posts_df["score"] > 0).sum()
        logging.info(
            f"Score input preview: {compressed_posts_df['compressed_post'].head(3).tolist()}"
        )
        logging.info(
            f"Score output preview: {compressed_posts_df['score'].head(3).tolist()}"
        )
        logging.info(
            f"📊 CV Matching Results:\n"
            f"  - Compressed posts analyzed: {compressed_count}\n"
            f"  - Scored posts: {scored_count}\n"
            f"  - Zero score (bad match): {zero_count} ({(zero_count/compressed_count)*100:.1f}%)\n"
            f"  - Positive score: {nonzero_count} ({(nonzero_count/compressed_count)*100:.1f}%)\n"
            f"  - Filtered posts after scoring: {filtered_count} / {scored_count}"
        )
        logging.info(f"Filtered posts after scoring: {filtered_count} / {scored_count}")
        if filtered_df.empty:
            logging.info("❌ No posts passed the score threshold, skipping batch")
            return None
        time.sleep(3)

        # Step 5: Job Post Parsing
        logging.info("\nSTEP 5: Job Post Parsing")
        logging.info("-" * 40)
        filtered_df.loc[:, "post_structured"] = filtered_df.apply(
            lambda row: safe_job_parsing(row["post"], row["compressed_post"]), axis=1
        )

        # Parse JSON and analyze structure
        def is_fallback_structured(s):
            try:
                d = json.loads(s)
                return set(d.keys()) == {"description"}
            except Exception:
                return True  # treat parse errors as fallback

        def is_full_structured(s):
            try:
                d = json.loads(s)
                return set(d.keys()) != {"description"}
            except Exception:
                return False

        fallback_count = (
            filtered_df["post_structured"].apply(is_fallback_structured).sum()
        )
        full_count = filtered_df["post_structured"].apply(is_full_structured).sum()
        total_count = len(filtered_df)
        logging.info(
            f"Parsing input preview: {filtered_df['compressed_post'].head(3).tolist()}"
        )
        logging.info(
            f"Parsing output preview: {filtered_df['post_structured'].head(3).tolist()}"
        )
        logging.info(
            f"📊 Job Post Parsing Results:\n"
            f"  - Compressed posts analyzed: {total_count}\n"
            f"  - Fully structured posts: {full_count} ({(full_count/total_count)*100:.1f}%)\n"
            f"  - Fallback (description-only): {fallback_count} ({(fallback_count/total_count)*100:.1f}%)\n"
        )
        time.sleep(3)

        # Step 6: Clean Job Post Values
        logging.info("\nSTEP 6: Clean Job Post Values")
        logging.info("-" * 40)
        filtered_df.loc[:, "post_structured_clean"] = filtered_df[
            "post_structured"
        ].apply(
            lambda s: safe_clean_job_post_values(
                json.loads(s), exclude_fields=["description"]
            )
        )
        cleaned_count = (
            filtered_df["post_structured_clean"].apply(lambda d: bool(d)).sum()
        )
        total_count = len(filtered_df)
        logging.info(
            f"Clean values input preview: {filtered_df['post_structured'].head(3).tolist()}"
        )
        logging.info(
            f"Clean values output preview: {filtered_df['post_structured_clean'].head(3).tolist()}"
        )
        logging.info(
            f"📊 Clean Job Post Values Results:\n"
            f"  - Parsed posts analyzed: {total_count}\n"
            f"  - Cleaned posts (all non-empty): {cleaned_count} ({(cleaned_count/total_count)*100:.1f}%)\n"
        )
        time.sleep(3)

        filtered_df = filtered_df.drop(columns=["post_structured"]).rename(
            columns={"post_structured_clean": "post_structured"}
        )

        # Serialize 'post_structured' column to JSON string for DB compatibility
        if "post_structured" in filtered_df.columns:
            filtered_df["post_structured"] = filtered_df["post_structured"].apply(
                json.dumps
            )

        # Set timestamp
        filtered_df.loc[:, "created_at"] = pd.Timestamp(
            datetime.datetime.now(datetime.UTC)
        ).tz_localize(None)

        logging.info("\n" + "=" * 80)
        logging.info("🏁 FINAL RESULTS")
        logging.info("=" * 80)
        logging.info(
            f"Pipeline Summary:\n"
            f"  Initial posts: {initial_count}\n"
            f"  Final valid posts: {len(filtered_df)} ({(len(filtered_df)/initial_count)*100:.1f}% of initial)"
        )
        logging.info("=" * 80)

        return filtered_df

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
        cv_content = safe_fetch_cv_content(CV_DOC_ID)
        if not cv_content:
            raise ValueError("CV content is empty")

        cv_summary = safe_summarize_cv_content(cv_content)
        if not cv_summary:
            logging.warning("CV summary is empty, falling back to original CV content")
            cv_summary = cv_content

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

        # Store final metrics in InfluxDB
        store_metrics(
            measurement="tg-job-radar__data_cleaning__stats",
            fields={
                "total_posts": float(stats.total_posts),
                "job_posts": float(stats.job_posts),
                "single_job_posts": float(stats.single_job_posts),
                "posts_above_threshold": float(stats.posts_above_threshold),
                "posts_below_threshold": float(stats.posts_below_threshold),
                "posts_with_structured_data": float(stats.posts_with_structured_data),
                "posts_without_structured_data": float(
                    stats.posts_without_structured_data
                ),
                "parsing_errors": float(stats.parsing_errors),
                "valid_parsing": float(stats.valid_parsing),
                "empty_parsing": float(stats.empty_parsing),
                "server_errors": float(stats.server_errors),
                "total_scores": float(len(stats.scores)),
                "job_posts_rate": float(
                    (stats.job_posts / stats.total_posts * 100)
                    if stats.total_posts > 0
                    else 0.0
                ),
                "single_job_posts_rate": float(
                    (stats.single_job_posts / stats.job_posts * 100)
                    if stats.job_posts > 0
                    else 0.0
                ),
                "above_threshold_rate": float(
                    (stats.posts_above_threshold / stats.single_job_posts * 100)
                    if stats.single_job_posts > 0
                    else 0.0
                ),
                "below_threshold_rate": float(
                    (stats.posts_below_threshold / stats.single_job_posts * 100)
                    if stats.single_job_posts > 0
                    else 0.0
                ),
                "structured_data_rate": float(
                    (
                        stats.posts_with_structured_data
                        / stats.posts_above_threshold
                        * 100
                    )
                    if stats.posts_above_threshold > 0
                    else 0.0
                ),
                "parsing_error_rate": float(
                    (stats.parsing_errors / stats.total_posts * 100)
                    if stats.total_posts > 0
                    else 0.0
                ),
                "valid_parsing_rate": float(
                    (stats.valid_parsing / stats.total_posts * 100)
                    if stats.total_posts > 0
                    else 0.0
                ),
                "empty_parsing_rate": float(
                    (stats.empty_parsing / stats.total_posts * 100)
                    if stats.total_posts > 0
                    else 0.0
                ),
                "score_mean": (
                    float(sum(stats.scores) / len(stats.scores))
                    if stats.scores
                    else 0.0
                ),
                "score_std": (
                    float(
                        (
                            sum(
                                (x - (sum(stats.scores) / len(stats.scores))) ** 2
                                for x in stats.scores
                            )
                            / len(stats.scores)
                        )
                        ** 0.5
                    )
                    if stats.scores
                    else 0.0
                ),
                "execution_time_ms": float(stats.execution_time_ms),
            },
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
