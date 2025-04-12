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
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from _production import (
    CV_DOC_ID,
    DATA_BATCH_SIZE,
    GDOCS_TIMEOUT_SECONDS,
    LLM_INSTANCE_URL,
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


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Error counting tokens: {e!s}")
        return len(text) // 4  # Fallback: rough estimate of 4 chars per token


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

        # Job post detection
        logging.info("-" * 40)
        logging.info("STEP 1: Job Post Detection")
        logging.info("-" * 40)

        # Function to safely process a single post
        def safe_job_post_detection(post):
            try:
                token_count = count_tokens(post)
                logging.info(f"Input: {post[:128]}... (tokens: {token_count})")
                logging.info("Request to job_post_detection...")
                result = job_post_detection(post)
                logging.info(f"Response: {result}")
                # Add a small delay after each LLM call to prevent overloading
                time.sleep(1.5)
                return result
            except Exception as error:
                logging.error(f"Error in job_post_detection: {error!s}", exc_info=True)
                return None

        # Process all posts at once
        batch_df.loc[:, "is_job_post"] = batch_df["post"].apply(safe_job_post_detection)

        job_post_mask = batch_df["is_job_post"].fillna(False)
        job_posts_count = job_post_mask.sum()
        stats.job_posts += job_posts_count

        if job_posts_count == 0:
            logging.info(
                "No job posts found in this batch, skipping further processing"
            )
            return None

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

        # Add a delay between steps
        logging.info("Pausing for 3 seconds between processing steps")
        time.sleep(3)

        # Single job post detection - only for posts identified as job posts
        logging.info("\n" + "-" * 40)
        logging.info("STEP 2: Single Job Post Detection")
        logging.info("-" * 40)

        # Safe version of single job post detection
        def safe_single_job_detection(post):
            try:
                token_count = count_tokens(post)
                logging.info(f"Input: {post[:128]}... (tokens: {token_count})")
                logging.info("Request to single_job_post_detection...")
                result = single_job_post_detection(post)
                logging.info(f"Response: {result}")
                # Add a small delay after each LLM call
                time.sleep(1.5)
                return result
            except Exception as e:
                logging.error(
                    f"Error in single_job_post_detection: {e!s}", exc_info=True
                )
                return None

        # Only process posts that were identified as job posts
        job_posts_df = batch_df[job_post_mask]

        if len(job_posts_df) > 0:
            # Process all job posts at once
            batch_df.loc[job_posts_df.index, "is_single_job_post"] = job_posts_df[
                "post"
            ].apply(safe_single_job_detection)

        single_post_mask = batch_df["is_single_job_post"].fillna(False)
        single_posts_count = single_post_mask.sum()
        stats.single_job_posts += single_posts_count

        if single_posts_count == 0:
            logging.info(
                "No single job posts found in this batch, skipping further processing"
            )
            return None

        # Calculate state statistics
        true_count = batch_df["is_single_job_post"].sum()
        false_count = (batch_df["is_single_job_post"] == False).sum()
        none_count = batch_df["is_single_job_post"].isna().sum()

        logging.info(
            f"📊 Results:\n"
            f"  - Job posts analyzed: {job_posts_count}\n"
            f"  - Single posts found: {single_posts_count}\n"
            f"  - Success rate: {(single_posts_count/max(job_posts_count, 1))*100:.1f}% of job posts\n"
            f"  - State distribution:\n"
            f"    • True: {true_count} ({true_count/initial_count*100:.1f}%)\n"
            f"    • False: {false_count} ({false_count/initial_count*100:.1f}%)\n"
            f"    • None: {none_count} ({none_count/initial_count*100:.1f}%)"
        )

        # Add a delay between steps
        logging.info("Pausing for 3 seconds between processing steps")
        time.sleep(3)

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
                cv_token_count = count_tokens(cv_content)
                post_token_count = count_tokens(post)
                total_tokens = cv_token_count + post_token_count

                logging.info(f"Input: {post[:128]}... (tokens: {post_token_count})")
                logging.info("Request to match_cv_with_job...")
                logging.info(
                    f"Query to structured_generate:\n"
                    f"  - CV tokens: {cv_token_count}\n"
                    f"  - Post tokens: {post_token_count}\n"
                    f"  - Total tokens: {total_tokens}\n"
                    f"  - Endpoint: {LLM_INSTANCE_URL}/structured_generate"
                )

                # Add retry logic for server errors
                max_retries = 3
                retry_delay = 5

                for attempt in range(max_retries):
                    try:
                        score = match_cv_with_job(cv_content, post)
                        logging.info(f"Response: {score}")
                        score_value = float(score) if score is not None else None
                        return score_value
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 500:
                            if attempt < max_retries - 1:
                                logging.warning(
                                    f"Server error (500) on attempt {attempt + 1}/{max_retries}. "
                                    f"Retrying in {retry_delay} seconds..."
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                logging.error(
                                    f"All retry attempts failed for match_cv_with_job. "
                                    f"Last error: {e!s}"
                                )
                                return None
                        else:
                            logging.error(f"HTTP error in match_cv_with_job: {e!s}")
                            return None
                    except Exception as e:
                        logging.error(f"Unexpected error in match_cv_with_job: {e!s}")
                        return None

                # Add a larger delay after CV matching (more intensive)
                time.sleep(3)
                return None

            except Exception as error:
                logging.error(f"Error in match_cv_with_job: {error!s}", exc_info=True)
                return None

        if posts_to_score > 0:
            # Process all posts at once for scoring
            posts_to_score_df = batch_df[score_mask_simple]
            batch_df.loc[posts_to_score_df.index, "score"] = posts_to_score_df[
                "post"
            ].apply(safe_match_score)

        # After all score calculations, convert decimal scores to percentage (0-100 scale)
        # This makes the comparison with threshold (85) work correctly
        batch_df["score"] = batch_df["score"] * 100

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

        if threshold_count == 0:
            logging.info(
                "No posts above threshold found in this batch, skipping further processing"
            )
            return None

        # Calculate score statistics
        valid_scores = batch_df["score"].notna().sum()
        null_scores = batch_df["score"].isna().sum()

        logging.info(
            f"📊 Results:\n"
            f"  - Posts scored: {posts_to_score}\n"
            f"  - Above threshold ({MATCH_SCORE_THRESHOLD}): {threshold_count}\n"
            f"  - Success rate: {(threshold_count/max(posts_to_score, 1))*100:.1f}% passed threshold\n"
            f"  - Score distribution:\n"
            f"    • Valid scores: {valid_scores} ({valid_scores/initial_count*100:.1f}%)\n"
            f"    • NULL scores: {null_scores} ({null_scores/initial_count*100:.1f}%)"
        )

        # Add a delay between steps
        logging.info("Pausing for 3 seconds between processing steps")
        time.sleep(3)

        # Final parsing
        logging.info("\n" + "-" * 40)
        logging.info("STEP 5: Structured Data Parsing")
        logging.info("-" * 40)
        parsing_mask = batch_df["is_single_job_post"] & (
            batch_df["score"] >= MATCH_SCORE_THRESHOLD
        )
        posts_to_parse = parsing_mask.sum()

        if posts_to_parse == 0:
            logging.info("No posts to parse in this batch, skipping parsing step")
            return None

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
                token_count = count_tokens(post)
                logging.info(f"Input: {post[:128]}... (tokens: {token_count})")
                logging.info("Request to job_post_parsing...")
                logging.info(
                    f"Query to structured_generate:\n"
                    f"  - Post tokens: {token_count}\n"
                    f"  - Endpoint: {LLM_INSTANCE_URL}/structured_generate"
                )

                # Add retry logic for server errors
                max_retries = 3
                retry_delay = 5  # seconds

                for attempt in range(max_retries):
                    try:
                        parsed_data = job_post_parsing(post)
                        logging.info(f"Response: {parsed_data}")

                        if not parsed_data:
                            return json.dumps({"full_description": post})

                        # Handle case where LLM returns action-based format
                        if isinstance(parsed_data, dict) and "action" in parsed_data:
                            return json.dumps({"full_description": post})

                        # Validate the parsed data structure
                        if not isinstance(parsed_data, dict):
                            return json.dumps({"full_description": post})

                        try:
                            # Attempt to serialize to ensure valid JSON
                            json_str = json.dumps(parsed_data)
                            return json_str
                        except (TypeError, ValueError):
                            return json.dumps({"full_description": post})

                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 500:
                            if attempt < max_retries - 1:
                                logging.warning(
                                    f"Server error (500) on attempt {attempt + 1}/{max_retries}. "
                                    f"Retrying in {retry_delay} seconds..."
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                logging.error(
                                    f"All retry attempts failed for job_post_parsing. "
                                    f"Last error: {e!s}"
                                )
                                return json.dumps({"full_description": post})
                        else:
                            logging.error(f"HTTP error in job_post_parsing: {e!s}")
                            return json.dumps({"full_description": post})
                    except Exception as e:
                        logging.error(f"Unexpected error in job_post_parsing: {e!s}")
                        return json.dumps({"full_description": post})

                # Add a larger delay after parsing (more intensive)
                time.sleep(3)
                return json.dumps({"full_description": post})

            except Exception as error:
                logging.error(f"Error in job_post_parsing: {error!s}", exc_info=True)
                return json.dumps({"full_description": post})

        if posts_to_parse > 0:
            # Process all posts at once for parsing
            posts_to_parse_df = batch_df[parsing_mask]
            batch_df.loc[posts_to_parse_df.index, "post_structured"] = (
                posts_to_parse_df["post"].apply(safe_job_parsing)
            )

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

        if final_count == 0:
            logging.info(
                "No valid posts found in this batch after all processing steps"
            )
            return None

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

        logging.info("Successfully processed all data batches")

    except Exception as error:
        logging.error("Data pipeline failed", exc_info=True)
        raise Exception(f"Data pipeline failed: {error!s}") from error


if __name__ == "__main__":
    clean_and_move_data()
