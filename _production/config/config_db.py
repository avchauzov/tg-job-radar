import logging

from _production import (
    PROD_DATA__JOBS,
    RAW_DATA__TG_POSTS,
    STAGING_DATA__POSTS,
)
from _production.utils.common import setup_logging
from _production.utils.sql import get_table_schema

setup_logging(__file__[:-3])


def generate_db_mappings():
    """
    Generate database mappings for all relevant tables.
    Returns a dictionary containing table schemas and column mappings.
    """
    try:
        # Get schemas for all tables
        raw_schema = get_table_schema(RAW_DATA__TG_POSTS)
        staging_schema = get_table_schema(STAGING_DATA__POSTS)
        prod_schema = get_table_schema(PROD_DATA__JOBS)

        # Define column lists
        raw_columns = list(raw_schema.keys())
        staging_columns = list(staging_schema.keys())
        prod_columns = [col for col in list(prod_schema.keys()) if col != "notificated"]

        # Define mappings based on actual table structures
        raw_to_staging = {
            "source_columns": raw_columns,
            "target_columns": [col for col in raw_columns if col in staging_schema],
            "json_columns": [],
            "source_table": RAW_DATA__TG_POSTS,
            "target_table": STAGING_DATA__POSTS,
        }

        staging_to_prod = {
            "source_columns": [
                "id",
                "channel",
                "created_at",
                "post_link",
                "post_structured",
            ],
            "target_columns": [
                "id",
                "channel",
                "created_at",
                "post_link",
                "post_structured",
            ],
            "json_columns": ["post_structured"],
            "source_table": STAGING_DATA__POSTS,
            "target_table": PROD_DATA__JOBS,
        }

        return {
            "schemas": {
                "raw": raw_schema,
                "staging": staging_schema,
                "prod": prod_schema,
            },
            "mappings": {
                "raw_to_staging": raw_to_staging,
                "staging_to_prod": staging_to_prod,
            },
            "columns": {
                "raw": raw_columns,
                "staging": staging_columns,
                "prod": prod_columns,
            },
        }

    except Exception:
        logging.error("Error generating DB mappings", exc_info=True)
        raise


def validate_table_mappings(mappings):
    """Validate that all mapped columns exist in source and target tables"""
    for mapping_name, mapping in mappings.items():
        source_table = mapping["source_table"]
        target_table = mapping["target_table"]
        source_schema = get_table_schema(source_table)
        target_schema = get_table_schema(target_table)

        # Validate source columns exist
        for col in mapping["source_columns"]:
            if col not in source_schema:
                raise ValueError(f"Column {col} not found in {source_table}")

        # Validate target columns exist
        for col in mapping["target_columns"]:
            if col not in target_schema:
                raise ValueError(f"Column {col} not found in {target_table}")


# Generate DB Mappings
try:
    DB_MAPPINGS = generate_db_mappings()

    # Validate mappings
    validate_table_mappings(DB_MAPPINGS["mappings"])

    # Extract column definitions
    RAW_DATA__TG_POSTS__COLUMNS = DB_MAPPINGS["columns"]["raw"]
    STAGING_DATA__POSTS__COLUMNS = DB_MAPPINGS["columns"]["staging"]
    PROD_DATA__JOBS__COLUMNS = DB_MAPPINGS["columns"]["prod"]

    # Extract mappings
    RAW_TO_STAGING_MAPPING = DB_MAPPINGS["mappings"]["raw_to_staging"]
    STAGING_TO_PROD_MAPPING = DB_MAPPINGS["mappings"]["staging_to_prod"]

    # SQL query components
    RAW_TO_STAGING__SELECT = ", ".join(RAW_TO_STAGING_MAPPING["source_columns"])
    STAGING_TO_PROD__SELECT = ", ".join(STAGING_TO_PROD_MAPPING["source_columns"])

    RAW_TO_STAGING__WHERE = (
        f"id not in (select id from {RAW_TO_STAGING_MAPPING['target_table']})"
    )
    STAGING_TO_PROD__WHERE = """
        post_structured IS NOT NULL
        AND post_structured != '{}'::jsonb
        AND post_structured != 'null'::jsonb
        AND id NOT IN (SELECT id FROM prod_data.jobs)
    """

except Exception:
    logging.error("Failed to generate DB mappings", exc_info=True)
    raise
