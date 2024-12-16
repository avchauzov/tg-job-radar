import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, "/home/job_search")
sys.path.insert(0, str(PROJECT_ROOT))


import datetime
import logging

from _production import RAW_DATA__TG_POSTS
from _production.config.config import (
    DATA_BATCH_SIZE,
    RAW_DATA__TG_POSTS__COLUMNS,
    SOURCE_CHANNELS,
    TG_CLIENT,
)
from _production.utils.functions_common import generate_hash, setup_logging
from _production.utils.functions_sql import batch_insert_to_db, fetch_from_db
from _production.utils.functions_text import clean_job_description
from _production.utils.functions_tg_api import get_channel_link_header
from _production.airflow.plugins.raw.text_processing import contains_keywords


file_name = __file__[:-3]
setup_logging(file_name)


def scrape_tg():
    with TG_CLIENT as tg_client:
        logging.info("Started scraping process.")
        try:
            # Fetch last dates
            try:
                _, last_date = fetch_from_db(
                    RAW_DATA__TG_POSTS,
                    "channel, max(date) as date",
                    group_by_condition="channel",
                    order_by_condition="date desc",
                )
                last_date_dict = dict(last_date)
            except Exception as db_error:
                logging.debug("Failed to fetch last dates from database", exc_info=True)
                raise Exception(f"Database query failed: {str(db_error)}") from db_error

            for channel in SOURCE_CHANNELS:
                logging.info(f"Starting to scrape channel: {channel}")
                try:
                    last_date = last_date_dict.get(channel)
                    if not last_date:
                        last_date = datetime.datetime.utcnow() - datetime.timedelta(
                            days=30
                        )
                        logging.info(
                            f"No previous records found. Starting from 30 days ago: {last_date}"
                        )

                    try:
                        entity = tg_client.get_entity(channel)
                    except Exception as entity_error:
                        logging.debug(
                            f"Failed to get entity for channel: {channel}",
                            exc_info=True,
                        )
                        raise Exception(
                            f"Failed to get Telegram entity: {str(entity_error)}"
                        ) from entity_error

                    link_header = get_channel_link_header(entity)

                    results, results_count = [], 0
                    for message in tg_client.iter_messages(
                        entity=channel, reverse=True, offset_date=last_date
                    ):
                        job_description, date = message.text, message.date

                        if not job_description or not date:
                            logging.info(
                                f"Skipping message with missing job description or date in channel: {channel}"
                            )
                            continue

                        if (
                            isinstance(date, datetime.datetime)
                            and date.tzinfo is not None
                        ):
                            date = date.astimezone(datetime.timezone.utc).replace(
                                tzinfo=None
                            )

                        message_link = f"{link_header}{message.id}".lower().strip()
                        job_description_cleaned = clean_job_description(job_description)

                        if not contains_keywords(job_description_cleaned):
                            logging.info(
                                f"Skipping message (no keywords) in channel: {channel}"
                            )
                            continue

                        result = {
                            "id": generate_hash(message_link),
                            "channel": channel,
                            "post": job_description,
                            "date": date
                            if isinstance(date, datetime.datetime)
                            else None,
                            "created_at": datetime.datetime.now(datetime.UTC),
                            "post_link": message_link,
                        }

                        results.append(result)

                        if len(results) == DATA_BATCH_SIZE:
                            batch_insert_to_db(
                                RAW_DATA__TG_POSTS,
                                RAW_DATA__TG_POSTS__COLUMNS,
                                ["id"],
                                results,
                            )
                            logging.info(
                                f"Inserting batch of {len(results)} messages into database."
                            )

                            results_count += len(results)
                            results = []

                    if results:
                        batch_insert_to_db(
                            RAW_DATA__TG_POSTS,
                            RAW_DATA__TG_POSTS__COLUMNS,
                            ["id"],
                            results,
                        )
                        logging.info(
                            f"Inserting batch of {len(results)} messages into database."
                        )

                        results_count += len(results)

                    logging.info(f"Added {results_count} posts!")

                except Exception as channel_error:
                    logging.debug(
                        f"Channel scraping failed. Details: {str(channel_error)}",
                        exc_info=True,
                    )
                    raise Exception(
                        f"Failed to scrape channel. "
                        f"Channel: {channel}, "
                        f"Posts collected: {results_count if 'results_count' in locals() else 0}, "
                        f"Last date: {last_date}, "
                        f"Original error: {str(channel_error)}"
                    ) from channel_error

        except Exception as error:
            logging.debug("Scraping process failed", exc_info=True)
            raise Exception(f"Scraping process failed: {str(error)}") from error

        finally:
            logging.info("Scraping process completed.")


if __name__ == "__main__":
    scrape_tg()
