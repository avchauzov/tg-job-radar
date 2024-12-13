import logging
import time
import json
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_batch

from _production import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_USER,
    PROD_DATA__JOBS,
    RAW_DATA__TG_POSTS,
    STAGING_DATA__POSTS,
)


@contextmanager
def establish_db_connection():
    connection = None
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=5432,
        )

        yield connection

    except Exception as error:
        logging.error(f"Error connecting to the database: {error}")

        if connection:
            connection.close()

        raise

    finally:
        if connection:
            connection.close()


def batch_insert_to_db(table_name, columns, conflict, data):
    try:
        if not data:
            logging.warning("No data provided for insertion.")
            return

        columns_str = ", ".join(
            f'"{column}"' if column.isdigit() else column for column in columns
        )
        conflict_str = ", ".join(
            f'"{column}"' if column.isdigit() else column for column in conflict
        )
        columns_types = ", ".join(["%s"] * len(columns))

        db_insert_query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({columns_types})
        """

        if conflict:
            db_insert_query += f" ON CONFLICT ({conflict_str}) DO NOTHING"

        db_insert_query += ";"

        if isinstance(data[0], dict):
            data_tuples = [
                tuple(detail.get(column) for column in columns) for detail in data
            ]

        else:
            data_tuples = data

        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                execute_batch(cursor, db_insert_query, data_tuples)

            connection.commit()

    except Exception:
        logging.error("Error inserting data", exc_info=True)
        if "connection" in locals() and connection:
            connection.rollback()
        raise


def batch_update_to_db(table_name, update_columns, condition_column, data):
    try:
        if not data:
            logging.warning("No data provided for updating.")
            return

        update_str = ", ".join(f"{col} = %s" for col in update_columns)
        condition_str = f"{condition_column} = %s"

        db_update_query = f"""
            UPDATE {table_name}
            SET {update_str}
            WHERE {condition_str};
        """

        if isinstance(data[0], dict):
            data_tuples = [
                tuple(row[col] for col in update_columns) + (row[condition_column],)
                for row in data
            ]
        else:
            data_tuples = data

        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                execute_batch(cursor, db_update_query, data_tuples)

            connection.commit()

    except Exception:
        logging.error("Error updating data", exc_info=True)
        if "connection" in locals() and connection:
            connection.rollback()
        raise


def execute_query(query: str) -> tuple[list, list]:
    """
    Execute a SQL query and return results.

    Args:
        query: SQL query to execute

    Returns:
        Tuple of (columns, data) where columns is a list of column names
        and data is a list of tuples containing the row values
    """
    try:
        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                return columns, data

    except Exception:
        logging.error("Query execution failed", exc_info=True)
        raise


def fetch_from_db(
    table: str,
    select_condition: str = "*",
    where_condition: str = None,
    group_by_condition: str = None,
    order_by_condition: str = None,
    random_limit: int = None,
) -> tuple[list, list]:
    """
    Fetch data from database table.

    Args:
        table: Table name
        select_condition: Columns to select
        where_condition: WHERE clause
        group_by_condition: GROUP BY clause
        order_by_condition: ORDER BY clause
        random_limit: Limit for random selection

    Returns:
        Tuple of (columns, data)
    """
    try:
        query = f"SELECT {select_condition} FROM {table}"

        if where_condition:
            query += f" WHERE {where_condition}"

        if group_by_condition:
            query += f" GROUP BY {group_by_condition}"

        if order_by_condition:
            query += f" ORDER BY {order_by_condition}"

        if random_limit:
            query += f" ORDER BY RANDOM() LIMIT {random_limit}"

        return execute_query(query)

    except Exception:
        logging.error("Database fetch failed", exc_info=True)
        raise


def get_table_columns(table_name, to_exclude=[]):
    try:
        table_schema, table_name = table_name.split(".")

        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                query = """
	            select column_name
	            from information_schema.columns
	            where table_schema = %s and table_name = %s
	            """

                cursor.execute(
                    query,
                    (
                        table_schema,
                        table_name,
                    ),
                )
                columns = [
                    row[0] for row in cursor.fetchall() if row[0] not in to_exclude
                ]
                return columns

    except Exception:
        logging.error(f"Error fetching columns for table {table_name}", exc_info=True)
        raise


def move_data_with_condition(
    source_table,
    target_table,
    select_condition="",
    where_condition="",
    json_columns=None,
):
    """
    Move data between tables with condition.

    Args:
            source_table (str): Source table name
            target_table (str): Target table name
            select_condition (str): SELECT clause
            where_condition (str): WHERE clause
            json_columns (list): List of column names that should be serialized as JSON
    """
    try:
        json_columns = json_columns or []  # Default to empty list if None
        select_query = (
            f"SELECT {select_condition} FROM {source_table} WHERE {where_condition};"
        )
        logging.info(f"Executing select query: {select_query}")

        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                start_time = time.time()
                cursor.execute(select_query)
                data_to_move = cursor.fetchall()
                fetch_time = time.time() - start_time
                logging.info(f"Data fetched in {fetch_time:.2f} seconds.")

                if not data_to_move:
                    logging.info("No data to move based on the condition.")
                    return

                column_names = [desc[0] for desc in cursor.description]

                # Serialize only specified columns
                serialized_data = []
                for row in data_to_move:
                    serialized_row = list(row)
                    for i, col_name in enumerate(column_names):
                        if col_name in json_columns:
                            value = row[i]
                            if value is not None:  # Only serialize non-None values
                                serialized_row[i] = json.dumps(value)
                    serialized_data.append(tuple(serialized_row))

                placeholders = ", ".join(["%s"] * len(column_names))
                insert_query = f'INSERT INTO {target_table} ({", ".join(column_names)}) VALUES ({placeholders})'

                logging.info(
                    f"Inserting {len(serialized_data)} records into {target_table}."
                )

                start_time = time.time()
                cursor.executemany(insert_query, serialized_data)
                insert_time = time.time() - start_time
                logging.info(f"Data inserted in {insert_time:.2f} seconds.")

                connection.commit()
                logging.info(
                    f"{len(serialized_data)} records successfully moved from {source_table} to {target_table}."
                )

    except Exception:
        logging.error(
            f"Error moving data from {source_table} to {target_table}", exc_info=True
        )
        raise


def get_table_schema(table_name):
    """
    Get table schema including column names, types, and constraints.
    Returns a dictionary of column definitions.
    """
    try:
        schema, table = table_name.split(".")

        query = """
		SELECT 
			column_name,
			data_type,
			is_nullable,
			column_default,
			character_maximum_length
		FROM information_schema.columns 
		WHERE table_schema = %s 
		AND table_name = %s
		ORDER BY ordinal_position;
		"""

        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (schema, table))
                columns = cursor.fetchall()

                schema_dict = {}
                for col in columns:
                    name, dtype, nullable, default, max_length = col

                    # Build the column type definition
                    type_def = dtype.upper()
                    if max_length:
                        type_def += f"({max_length})"
                    if not nullable == "YES":
                        type_def += " NOT NULL"
                    if default:
                        type_def += f" DEFAULT {default}"

                    schema_dict[name] = type_def

                # Get primary key information
                pk_query = """
				SELECT c.column_name
				FROM information_schema.table_constraints tc
				JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
				JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
					AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
				WHERE constraint_type = 'PRIMARY KEY'
					AND tc.table_schema = %s
					AND tc.table_name = %s;
				"""
                cursor.execute(pk_query, (schema, table))
                pk_columns = [row[0] for row in cursor.fetchall()]

                # Mark primary keys in schema
                for pk in pk_columns:
                    if pk in schema_dict:
                        schema_dict[pk] += " PRIMARY KEY"

                return schema_dict

    except Exception:
        logging.error(f"Error getting schema for {table_name}", exc_info=True)
        raise


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

        # Define mappings based on actual table structures
        raw_to_staging = {
            "source_columns": list(raw_schema.keys()),
            "target_columns": [
                col for col in raw_schema.keys() if col in staging_schema
            ],
            "json_columns": [],
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
