import datetime
import logging

from _production import DATA_BATCH_SIZE, RAW_DATA__TG_POSTS
from _production.config.config import (
    DESIRED_KEYWORDS,
    SOURCE_CHANNELS,
    TG_CLIENT,
)
from _production.config.config_db import RAW_DATA__TG_POSTS__COLUMNS
from _production.utils.common import generate_hash, setup_logging
from _production.utils.sql import batch_insert_to_db, fetch_from_db
from _production.utils.text import clean_job_description, contains_keywords
from _production.utils.tg import get_channel_link_header

setup_logging(__file__[:-3])


def process_date(date):
    """Convert date to UTC datetime without timezone info."""
    if isinstance(date, datetime.datetime) and date.tzinfo is not None:
        return date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return date if isinstance(date, datetime.datetime) else None


def process_batch(results, table, columns, key_columns):
    """Process and insert a batch of results."""
    if not results:
        return 0
    batch_insert_to_db(table, columns, key_columns, results)
    logging.info(f"Inserting batch of {len(results)} messages into database.")
    return len(results)


def scrape_channel(tg_client, channel, last_date):
    """Scrape a single channel and return the number of processed messages."""
    entity = tg_client.get_entity(channel)
    link_header = get_channel_link_header(entity)

    if not last_date:
        last_date = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=30)
        logging.info(
            f"No previous records found. Starting from 30 days ago: {last_date}"
        )

    results, results_count = [], 0

    for message in tg_client.iter_messages(
        entity=channel, reverse=True, offset_date=last_date
    ):
        if not message.text or not message.date:
            logging.info(
                f"Skipping message with missing job description or date in channel: {channel}"
            )
            continue

        job_description_cleaned = clean_job_description(message.text)
        if not contains_keywords(
            text=job_description_cleaned, keywords=DESIRED_KEYWORDS
        ):
            logging.info(f"Skipping message (no keywords) in channel: {channel}")
            continue

        message_link = f"{link_header}{message.id}".lower().strip()
        result = {
            "id": generate_hash(message_link),
            "channel": channel,
            "post": message.text,
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
                "channel, max(date) as date",
                group_by_condition="channel",
                order_by_condition="date desc",
            )
            last_date_dict = dict(last_date)

            for channel in SOURCE_CHANNELS:
                logging.info(f"Starting to scrape channel: {channel}")
                try:
                    posts_collected = scrape_channel(
                        tg_client, channel, last_date_dict.get(channel)
                    )
                except Exception as e:
                    raise Exception(
                        f"Failed to scrape channel {channel}. "
                        f"Posts collected: {posts_collected if 'posts_collected' in locals() else 0}. "
                        f"Error: {str(e)}"
                    ) from e

        except Exception:
            logging.error("Scraping process failed", exc_info=True)
            raise

        finally:
            logging.info("Scraping process completed.")


if __name__ == "__main__":
    scrape_tg()
