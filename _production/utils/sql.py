import csv
import io
import json
import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Any

import psycopg2
from psycopg2.extras import execute_batch
from psycopg2.pool import SimpleConnectionPool

from _production import DATABASE


# Add custom exception type
class DatabaseError(Exception):
    """Custom exception for database operations"""

    pass


# Add at the top of the file with other constants
POOL_MIN_CONNECTIONS = 1
POOL_MAX_CONNECTIONS = 8
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Initialize connection pool
connection_pool = SimpleConnectionPool(
    POOL_MIN_CONNECTIONS,
    POOL_MAX_CONNECTIONS,
    host=DATABASE["HOST"],
    database=DATABASE["NAME"],
    user=DATABASE["USER"],
    password=DATABASE["PASSWORD"],
    port=5432,
)


def with_retry(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator to add retry logic for database operations"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except DatabaseError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
            if last_error:
                raise last_error
            raise DatabaseError("Maximum retries exceeded")

        return wrapper

    return decorator


@contextmanager
def establish_db_connection():
    """Creates and manages a database connection with proper error handling"""
    connection = None
    try:
        connection = psycopg2.connect(
            host=DATABASE["HOST"],
            database=DATABASE["NAME"],
            user=DATABASE["USER"],
            password=DATABASE["PASSWORD"],
            port=5432,
        )
        yield connection
    except Exception as error:
        logging.error(f"Error connecting to the database: {error}")
        raise DatabaseError(str(error))
    finally:
        if connection:
            connection.close()


def batch_insert_to_db(
    table_name: str,
    columns: list[str],
    unique_columns: list[str],
    data: Sequence[dict[str, Any] | tuple[Any, ...]],
) -> None:
    """
    Batch insert data into database table.

    Args:
        table_name: Name of the target table
        columns: List of column names
        conflict: List of columns for conflict resolution
        data: List of dictionaries or tuples containing the data
    """
    if not data:
        logging.warning("No data provided for insertion.")
        return

    # Move query building outside of try block
    columns_str = ", ".join(
        f'"{column}"' if column.isdigit() else column for column in columns
    )
    conflict_str = ", ".join(
        f'"{column}"' if column.isdigit() else column for column in unique_columns
    )
    columns_types = ", ".join(["%s"] * len(columns))

    db_insert_query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({columns_types})
        {f"ON CONFLICT ({conflict_str}) DO NOTHING" if unique_columns else ""};
    """

    # Optimize data conversion
    data_tuples = (
        [
            tuple(
                row[col] if isinstance(row, dict) else row[columns.index(col)]
                for col in columns
            )
            for row in data
        ]
        if isinstance(data[0], (dict, tuple))
        else data
    )

    try:
        with establish_db_connection() as connection:
            with connection.cursor() as cursor:
                execute_batch(cursor, db_insert_query, data_tuples)
            connection.commit()
    except Exception as e:
        logging.error("Error inserting data", exc_info=True)
        raise DatabaseError(f"Insert operation failed: {str(e)}")


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
                if cursor.description is None:
                    return [], []
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                return columns, data

    except Exception:
        logging.error("Query execution failed", exc_info=True)
        raise


def fetch_from_db(
    table: str,
    select_condition: str = "*",
    where_condition: str | None = None,
    group_by_condition: str | None = None,
    order_by_condition: str | None = None,
    random_limit: int | None = None,
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


@with_retry()
def move_data_with_condition(
    source_table: str,
    target_table: str,
    select_condition: str = "*",
    where_condition: str = "",
    json_columns: list[str] | None = None,
    batch_size: int = 8096,
    use_copy: bool = False,
) -> int:
    """
    Move data between tables with condition, supporting both COPY and INSERT methods.

    Args:
        source_table (str): Source table name
        target_table (str): Target table name
        select_condition (str): SELECT clause
        where_condition (str): WHERE clause
        json_columns (list): List of column names that should be serialized as JSON
        batch_size (int): Number of records to process in each batch
        use_copy (bool): Whether to use COPY command for data transfer

    Returns:
        int: Number of records moved

    Raises:
        DatabaseError: If the operation fails
    """
    try:
        json_columns = json_columns or []
        select_query = f"""
            SELECT {select_condition}
            FROM {source_table}
            {f'WHERE {where_condition}' if where_condition else ''};
        """

        connection = connection_pool.getconn()
        try:
            with connection.cursor() as cursor:
                cursor.execute(select_query)

                # Check if there are any results
                if cursor.rowcount == 0:
                    logging.info(
                        f"No data found to move from {source_table} to {target_table}"
                    )
                    return 0

                column_names = [desc[0] for desc in cursor.description]

                if use_copy:
                    return _copy_data(cursor, target_table, column_names, json_columns)
                else:
                    return _batch_insert_data(
                        cursor, target_table, column_names, json_columns, batch_size
                    )

        finally:
            connection_pool.putconn(connection)

    except Exception as error:
        logging.error(
            f"Error moving data from {source_table} to {target_table}", exc_info=True
        )
        raise DatabaseError(f"Data movement failed: {str(error)}")


def _copy_data(
    cursor, target_table: str, column_names: list, json_columns: list
) -> int:
    """Helper function to move data using COPY command"""
    output = io.StringIO()
    writer = csv.writer(
        output, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    total_rows = 0
    while True:
        rows = cursor.fetchmany(8096)
        if not rows:
            break

        for row in rows:
            # Process JSON columns
            processed_row = list(row)
            for i, col_name in enumerate(column_names):
                if col_name in json_columns and processed_row[i] is not None:
                    processed_row[i] = json.dumps(processed_row[i])
            writer.writerow(processed_row)
            total_rows += 1

    output.seek(0)
    columns_str = ", ".join(f'"{col}"' for col in column_names)

    with cursor.copy(
        f"COPY {target_table} ({columns_str}) FROM STDIN WITH CSV DELIMITER E'\\t' QUOTE '\"'"
    ) as copy:
        copy.write(output.getvalue())

    return total_rows


def _batch_insert_data(
    cursor, target_table: str, column_names: list, json_columns: list, batch_size: int
) -> int:
    """Helper function to move data using batch INSERT"""
    # Check if there are any rows to process
    if cursor.rowcount == 0:
        return 0

    placeholders = ", ".join(["%s"] * len(column_names))
    columns_str = ", ".join(f'"{col}"' for col in column_names)
    insert_query = f"INSERT INTO {target_table} ({columns_str}) VALUES ({placeholders})"

    total_rows = 0
    while True:
        try:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            # Process JSON columns
            processed_rows = []
            for row in rows:
                processed_row = list(row)
                for i, col_name in enumerate(column_names):
                    if col_name in json_columns and processed_row[i] is not None:
                        processed_row[i] = json.dumps(processed_row[i])
                processed_rows.append(tuple(processed_row))

            execute_batch(cursor, insert_query, processed_rows)
            total_rows += len(rows)

        except psycopg2.ProgrammingError as e:
            if "no results to fetch" in str(e):
                break
            raise

    return total_rows


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
