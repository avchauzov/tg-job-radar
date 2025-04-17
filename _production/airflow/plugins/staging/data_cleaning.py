"""
Data cleaning module for processing job posts and matching them against CV content.

This module handles the ETL process from raw data to staging, including:
- Structured data parsing and storage
"""

import contextlib
import datetime
import json
import logging
import time

import pandas as pd
import requests
import tiktoken

from _production import (
    DATA_BATCH_SIZE,
    LLM_INSTANCE_URL,
    NUMBER_OF_BATCHES,
    RAW_DATA__TG_POSTS,
    STAGING_DATA__POSTS,
)
from _production.config.config_db import (
    RAW_TO_STAGING__WHERE,
    STAGING_DATA__POSTS__COLUMNS,
)
from _production.utils.common import setup_logging
from _production.utils.llm import job_post_parsing
from _production.utils.sql import batch_insert_to_db, fetch_from_db
from _production.utils.text import extensive_clean_text

setup_logging(__file__[:-3])


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
    batch_num: int,
    total_batches: int,
) -> pd.DataFrame | None:
    """Process a batch of Telegram posts.

    Args:
        batch_df: DataFrame containing posts to process
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

        logging.info("=" * 80)
        logging.info(
            f"🔄 BATCH {batch_num}/{total_batches} PROCESSING: Starting with {initial_count} posts"
        )
        logging.info("=" * 80)

        # Initialize columns with appropriate default values
        batch_df.loc[:, "post_structured"] = "{}"
        batch_df.loc[:, "parsing_error"] = False
        logging.info("✓ Initialized default values for all columns")

        # Structured data parsing
        logging.info("\n" + "-" * 40)
        logging.info("STEP 1: Structured Data Parsing")
        logging.info("-" * 40)

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

                        if parsed_data is None:
                            parsed_data = {}

                        parsed_data["full_description"] = post

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

        # TODO: ADD
        batch_df.loc[:, "post"] = batch_df["post"].apply(extensive_clean_text)

        # Process all posts for parsing
        batch_df.loc[:, "post_structured"] = batch_df["post"].apply(safe_job_parsing)

        # Calculate parsing statistics
        parsing_errors = batch_df["parsing_error"].sum()
        valid_parsing = (batch_df["post_structured"] != "{}").sum()

        logging.info(
            f"📝 Parsing results:\n"
            f"  - Total posts: {initial_count}\n"
            f"  - With parsing errors: {parsing_errors}\n"
            f"  - Parsing results:\n"
            f"    • Valid parsing: {valid_parsing} ({valid_parsing/initial_count*100:.1f}%)\n"
            f"    • Empty results: {initial_count - valid_parsing} ({((initial_count - valid_parsing)/initial_count)*100:.1f}%)"
        )

        # Set timestamp and prepare final results
        batch_df.loc[:, "created_at"] = pd.Timestamp(
            datetime.datetime.now(datetime.UTC)
        ).tz_localize(None)

        # Final results
        final_df = batch_df.copy()
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
        # Set multiprocessing start method to 'spawn' to avoid fork-related issues
        import multiprocessing

        if hasattr(multiprocessing, "set_start_method"):
            with contextlib.suppress(RuntimeError):
                multiprocessing.set_start_method("spawn", force=True)

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

                processed_df = process_batch(batch_df, batch_num, total_batches)

                if processed_df is not None and not processed_df.empty:
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

        logging.info("Successfully processed all data batches")

    except Exception as error:
        logging.error("Data pipeline failed", exc_info=True)
        raise Exception(f"Data pipeline failed: {error!s}") from error


if __name__ == "__main__":
    clean_and_move_data()
