"""Module for collecting and processing raw job post data from Telegram channels."""

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from _production import DATA_BATCH_SIZE, LOOKBACK_DAYS, RAW_DATA__TG_POSTS
from _production.config.config import (
    DESIRED_KEYWORDS,
    SOURCE_CHANNELS,
    TG_CLIENT,
)
from _production.config.config_db import RAW_DATA__TG_POSTS__COLUMNS
from _production.utils.common import generate_hash, process_date, setup_logging
from _production.utils.influxdb import store_metrics
from _production.utils.sql import batch_insert_to_db, fetch_from_db
from _production.utils.text import (
    clean_job_description,
    contains_keywords,
    is_duplicate_post,
)
from _production.utils.tg import get_channel_link_header

setup_logging(__file__[:-3])


def process_batch(
    results: list[dict[str, Any]],
    table: str,
    columns: list[str],
    key_columns: list[str],
) -> int:
    """Process and insert a batch of results."""
    try:
        if not results:
            return 0
        batch_insert_to_db(table, columns, key_columns, results)
        logging.info(f"Inserting batch of {len(results)} messages into database.")
        return len(results)
    except Exception as error:
        logging.error(f"Error processing batch: {error!s}")
        raise


def get_entity_title(entity):
    """Get the title based on entity type."""
    if hasattr(entity, "title") and entity.title:  # For channels
        return entity.title
    elif hasattr(entity, "first_name"):  # For users
        full_name = entity.first_name or ""
        if hasattr(entity, "last_name") and entity.last_name:
            full_name += f" {entity.last_name}"
        return full_name.strip()
    return str(entity.id)  # Fallback to entity ID


@dataclass
class ScrapingStats:
    """Statistics for the scraping process."""

    total_messages_checked: int = 0
    messages_without_text: int = 0
    messages_without_date: int = 0
    messages_without_keywords: int = 0
    duplicate_messages: int = 0
    successful_jobs: int = 0
    post_dates: list[datetime.datetime] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a dictionary for logging."""
        if not self.post_dates:
            return {
                "total_messages_checked": self.total_messages_checked,
                "messages_without_text": self.messages_without_text,
                "messages_without_date": self.messages_without_date,
                "messages_without_keywords": self.messages_without_keywords,
                "duplicate_messages": self.duplicate_messages,
                "successful_jobs": self.successful_jobs,
                "post_dates": {"min": None, "max": None, "average": None},
            }

        # Calculate average date by converting to timestamps
        timestamps = [date.timestamp() for date in self.post_dates]
        avg_timestamp = sum(timestamps) / len(timestamps)
        avg_date = datetime.datetime.fromtimestamp(avg_timestamp, tz=datetime.UTC)

        return {
            "total_messages_checked": self.total_messages_checked,
            "messages_without_text": self.messages_without_text,
            "messages_without_date": self.messages_without_date,
            "messages_without_keywords": self.messages_without_keywords,
            "duplicate_messages": self.duplicate_messages,
            "successful_jobs": self.successful_jobs,
            "post_dates": {
                "min": min(self.post_dates).isoformat(),
                "max": max(self.post_dates).isoformat(),
                "average": avg_date.isoformat(),
            },
        }


def scrape_channel(tg_client, channel, last_date, stats: ScrapingStats):
    """Scrape a single channel and return the number of processed messages."""
    entity = tg_client.get_entity(channel)
    link_header = get_channel_link_header(entity)
    entity_title = get_entity_title(entity)

    if not last_date:
        last_date = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=LOOKBACK_DAYS
        )
        logging.info(
            f"No previous records found. Starting from {LOOKBACK_DAYS} days ago: {last_date}"
        )
    else:
        # Ensure last_date is timezone-aware
        if last_date.tzinfo is None:
            last_date = last_date.replace(tzinfo=datetime.UTC)

        # Ensure we don't fetch messages older than LOOKBACK_DAYS days even if last_date is older
        thirty_days_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=LOOKBACK_DAYS
        )
        if last_date < thirty_days_ago:
            last_date = thirty_days_ago
            logging.info(
                f"Last date is older than {LOOKBACK_DAYS} days. Adjusting to: {last_date}"
            )

    results, results_count = [], 0

    # Fetch recent posts using fetch_from_db directly
    recent_posts = fetch_from_db(
        RAW_DATA__TG_POSTS,
        select_condition="post",
        where_condition=f"date >= NOW() - INTERVAL '{LOOKBACK_DAYS} days'",
    )[1]

    logging.info(f"Checking messages in channel: {channel}")
    for message in tg_client.iter_messages(
        entity=channel, reverse=True, offset_date=last_date
    ):
        stats.total_messages_checked += 1

        if not message.text:
            stats.messages_without_text += 1
            continue

        if not message.date:
            stats.messages_without_date += 1
            continue

        job_description = message.text
        job_description_cleaned = clean_job_description(job_description)
        if not contains_keywords(
            text=job_description_cleaned, keywords=DESIRED_KEYWORDS
        ):
            stats.messages_without_keywords += 1
            continue

        # Check for duplicates
        is_duplicate, existing_post, similarity_score = is_duplicate_post(
            job_description_cleaned, recent_posts
        )
        if is_duplicate:
            stats.duplicate_messages += 1
            logging.info(
                f"Skipping duplicate message in channel: {channel}. "
                f"Similarity score: {similarity_score}.\nCurrent post: {job_description_cleaned}.\nExisting post: {existing_post}"
            )
            continue

        message_link = f"{link_header}{message.id}".lower().strip()
        result = {
            "id": generate_hash(message_link),
            "username": entity_title,
            "post": job_description_cleaned,
            "date": process_date(message.date),
            "created_at": datetime.datetime.now(datetime.UTC),
            "post_link": message_link,
        }
        results.append(result)
        stats.post_dates.append(message.date)

        if len(results) == DATA_BATCH_SIZE:
            results_count += process_batch(
                results,
                RAW_DATA__TG_POSTS,
                RAW_DATA__TG_POSTS__COLUMNS,
                ["id"],
            )
            results = []

    results_count += process_batch(
        results, RAW_DATA__TG_POSTS, RAW_DATA__TG_POSTS__COLUMNS, ["id"]
    )
    stats.successful_jobs += results_count

    logging.info(f"Added {results_count} posts!")
    return results_count


def scrape_tg():
    """Main function to scrape Telegram channels."""
    stats = ScrapingStats()
    start_time = time.time_ns()

    with TG_CLIENT as tg_client:
        logging.info("Started scraping process.")
        try:
            # Fetch last dates
            _, last_date = fetch_from_db(
                RAW_DATA__TG_POSTS,
                "username, max(date) as date",
                group_by_condition="username",
                order_by_condition="date desc",
            )
            last_date_dict = dict(last_date)

            for channel in sorted(SOURCE_CHANNELS):
                logging.info(f"Starting to scrape channel: {channel}")
                try:
                    posts_collected = scrape_channel(
                        tg_client, channel, last_date_dict.get(channel), stats
                    )
                except Exception as error:
                    raise Exception(
                        f"Failed to scrape channel {channel}. "
                        f"Posts collected: {posts_collected if 'posts_collected' in locals() else 0}. "
                        f"Error: {error!s}"
                    ) from error

        except Exception:
            logging.error("Scraping process failed", exc_info=True)
            raise

        finally:
            # Store metrics
            store_metrics(
                measurement="tg-job-radar__data_collection__scraping_stats",
                fields={
                    "total_messages_checked": stats.total_messages_checked,
                    "messages_without_text_rate": float(
                        (
                            stats.messages_without_text
                            / stats.total_messages_checked
                            * 100
                        )
                        if stats.total_messages_checked > 0
                        else 0.0
                    ),
                    "messages_without_date_rate": float(
                        (
                            stats.messages_without_date
                            / stats.total_messages_checked
                            * 100
                        )
                        if stats.total_messages_checked > 0
                        else 0.0
                    ),
                    "messages_without_keywords_rate": float(
                        (
                            stats.messages_without_keywords
                            / stats.total_messages_checked
                            * 100
                        )
                        if stats.total_messages_checked > 0
                        else 0.0
                    ),
                    "duplicate_messages_rate": float(
                        (stats.duplicate_messages / stats.total_messages_checked * 100)
                        if stats.total_messages_checked > 0
                        else 0.0
                    ),
                    "successful_jobs_rate": float(
                        (stats.successful_jobs / stats.total_messages_checked * 100)
                        if stats.total_messages_checked > 0
                        else 0.0
                    ),
                    "execution_time_ms": float(
                        (time.time_ns() - start_time) / 1_000_000
                    ),
                },
                tags={
                    "environment": "production",
                    "script": "data_collection",
                },
            )

            # Store post date statistics
            if stats.post_dates:
                store_metrics(
                    measurement="tg-job-radar__data_collection__post_dates",
                    fields={
                        "min_date": min(stats.post_dates).timestamp(),
                        "max_date": max(stats.post_dates).timestamp(),
                        "avg_date": datetime.datetime.fromtimestamp(
                            sum(date.timestamp() for date in stats.post_dates)
                            / len(stats.post_dates),
                            tz=datetime.UTC,
                        ).timestamp(),
                    },
                    tags={
                        "environment": "production",
                        "script": "data_collection",
                    },
                )

            logging.info("Scraping process completed. Metrics stored in InfluxDB.")


if __name__ == "__main__":
    scrape_tg()
