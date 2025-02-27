import datetime
import logging
from typing import Any, Dict, List

from _production import DATA_BATCH_SIZE, RAW_DATA__TG_POSTS
from _production.config.config import (
    DESIRED_KEYWORDS,
    SOURCE_CHANNELS,
    TG_CLIENT,
)
from _production.config.config_db import RAW_DATA__TG_POSTS__COLUMNS
from _production.utils.common import generate_hash, process_date, setup_logging
from _production.utils.sql import batch_insert_to_db, fetch_from_db
from _production.utils.text import (
    clean_job_description,
    contains_keywords,
    is_duplicate_post,
)
from _production.utils.tg import get_channel_link_header

setup_logging(__file__[:-3])


def process_batch(
    results: List[Dict[str, Any]],
    table: str,
    columns: List[str],
    key_columns: List[str],
) -> int:
    """Process and insert a batch of results."""
    try:
        if not results:
            return 0
        batch_insert_to_db(table, columns, key_columns, results)
        logging.info(f"Inserting batch of {len(results)} messages into database.")
        return len(results)
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
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


def scrape_channel(tg_client, channel, last_date):
    """Scrape a single channel and return the number of processed messages."""
    entity = tg_client.get_entity(channel)
    link_header = get_channel_link_header(entity)
    entity_title = get_entity_title(entity)

    if not last_date:
        last_date = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=30)
        logging.info(
            f"No previous records found. Starting from 30 days ago: {last_date}"
        )

    results, results_count = [], 0

    # Fetch recent posts using fetch_from_db directly
    recent_posts = fetch_from_db(
        RAW_DATA__TG_POSTS,
        select_condition="post",
        where_condition="date >= NOW() - INTERVAL '30 days'",
    )[1]

    for message in tg_client.iter_messages(
        entity=channel, reverse=True, offset_date=last_date
    ):
        if not message.text or not message.date:
            logging.info(
                f"Skipping message with missing job description or date in channel: {channel}"
            )
            continue

        job_description = message.text
        job_description_cleaned = clean_job_description(job_description)
        if not contains_keywords(
            text=job_description_cleaned, keywords=DESIRED_KEYWORDS
        ):
            logging.info(f"Skipping message (no keywords) in channel: {channel}")
            continue

        # Check for duplicates
        is_duplicate, existing_post, similarity_score = is_duplicate_post(
            job_description_cleaned, recent_posts
        )
        if is_duplicate:
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

    logging.info(f"Added {results_count} posts!")
    return results_count


def scrape_tg():
    """Main function to scrape Telegram channels."""
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

            for channel in SOURCE_CHANNELS:
                logging.info(f"Starting to scrape channel: {channel}")
                try:
                    posts_collected = scrape_channel(
                        tg_client, channel, last_date_dict.get(channel)
                    )
                except Exception as error:
                    raise Exception(
                        f"Failed to scrape channel {channel}. "
                        f"Posts collected: {posts_collected if 'posts_collected' in locals() else 0}. "
                        f"Error: {str(error)}"
                    ) from error

        except Exception:
            logging.error("Scraping process failed", exc_info=True)
            raise

        finally:
            logging.info("Scraping process completed.")


if __name__ == "__main__":
    scrape_tg()
