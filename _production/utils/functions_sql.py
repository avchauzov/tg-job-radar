import logging
import time
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_batch

from _production import POSTGRES_HOST, POSTGRES_NAME, POSTGRES_PASS, POSTGRES_USER


@contextmanager
def establish_db_connection():
	connection = None
	try:
		connection = psycopg2.connect(
				host=POSTGRES_HOST,
				database=POSTGRES_NAME,
				user=POSTGRES_USER,
				password=POSTGRES_PASS,
				port=5432
				)
		
		yield connection
	
	except Exception as error:
		logging.error(f'Error connecting to the database: {error}')
		
		if connection:
			connection.close()
		
		raise
	
	finally:
		
		if connection:
			connection.close()


def batch_insert_to_db(table_name, columns, conflict, data):
	try:
		if not data:
			logging.warning('No data provided for insertion.')
			return
		
		columns_str = ', '.join(f'"{column}"' if column.isdigit() else column for column in columns)
		conflict_str = ', '.join(f'"{column}"' if column.isdigit() else column for column in conflict)
		columns_types = ', '.join(['%s'] * len(columns))
		
		db_insert_query = f'''
            INSERT INTO {table_name} ({columns_str})
            VALUES ({columns_types})
        '''
		
		if conflict:
			db_insert_query += f' ON CONFLICT ({conflict_str}) DO NOTHING'
		
		db_insert_query += ';'
		
		if isinstance(data[0], dict):
			data_tuples = [tuple(detail.get(column) for column in columns) for detail in data]
		
		else:
			data_tuples = data
		
		with establish_db_connection() as connection:
			with connection.cursor() as cursor:
				execute_batch(cursor, db_insert_query, data_tuples)
			
			connection.commit()
	
	except Exception as error:
		logging.error(f'Error inserting data: {error}')
		
		if 'connection' in locals() and connection:
			connection.rollback()


def batch_update_to_db(table_name, update_columns, condition_column, data):
	try:
		if not data:
			logging.warning('No data provided for updating.')
			return
		
		update_str = ', '.join(f'{col} = %s' for col in update_columns)
		condition_str = f'{condition_column} = %s'
		
		db_update_query = f'''
            UPDATE {table_name}
            SET {update_str}
            WHERE {condition_str};
        '''
		
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
	
	except Exception as error:
		logging.error(f'Error updating data: {error}')
		
		if 'connection' in locals() and connection:
			connection.rollback()


def fetch_from_db(table_name, select_condition='', where_condition='', group_by_condition='', order_by_condition=''):
	try:
		select_query = f'SELECT * FROM {table_name}'
		
		if select_condition:
			select_query = f'SELECT {select_condition} FROM {table_name}'
		
		if where_condition:
			select_query += f' WHERE {where_condition}'
		
		if group_by_condition:
			select_query += f' GROUP BY {group_by_condition}'
		
		if order_by_condition:
			select_query += f' ORDER BY {order_by_condition}'
		
		select_query += ';'
		
		logging.info(f'Executing query: {select_query}')
		with establish_db_connection() as connection:
			with connection.cursor() as cursor:
				cursor.execute(select_query)
				db_data = cursor.fetchall()
				
				column_names = [desc[0] for desc in cursor.description]
				
				if not db_data:
					logging.info('No data found for the query.')
				
				else:
					logging.info(f'Found {len(db_data)} records from {table_name}.')
		
		return column_names, db_data
	
	except Exception as error:
		logging.error(f'Error selecting data from {table_name}: {error}')
		raise


def get_table_columns(table_name, to_exclude=[]):
	try:
		table_schema, table_name = table_name.split('.')
		
		with establish_db_connection() as connection:
			with connection.cursor() as cursor:
				query = """
	            select column_name
	            from information_schema.columns
	            where table_schema = %s and table_name = %s
	            """
				
				cursor.execute(query, (table_schema, table_name,))
				columns = [row[0] for row in cursor.fetchall() if row[0] not in to_exclude]
				return columns
	
	except Exception as error:
		logging.error(f'Error fetching columns for table {table_name}: {error}')
		return []


def move_data_with_condition(source_table, target_table, select_condition='', where_condition=''):
	try:
		select_query = f'SELECT {select_condition} FROM {source_table} WHERE {where_condition};'
		logging.info(f'Executing select query: {select_query}')
		
		with establish_db_connection() as connection:
			with connection.cursor() as cursor:
				start_time = time.time()
				cursor.execute(select_query)
				data_to_move = cursor.fetchall()
				fetch_time = time.time() - start_time
				logging.info(f'Data fetched in {fetch_time:.2f} seconds.')
				
				if not data_to_move:
					logging.info('No data to move based on the condition.')
					return
				
				column_names = [desc[0] for desc in cursor.description]
				
				placeholders = ', '.join(['%s'] * len(column_names))
				insert_query = f'INSERT INTO {target_table} ({", ".join(column_names)}) VALUES ({placeholders})'
				
				logging.info(f'Inserting {len(data_to_move)} records into {target_table}.')
				
				start_time = time.time()
				cursor.executemany(insert_query, data_to_move)
				insert_time = time.time() - start_time
				logging.info(f'Data inserted in {insert_time:.2f} seconds.')
				
				connection.commit()
				logging.info(f'{len(data_to_move)} records successfully moved from {source_table} to {target_table}.')
	
	except Exception as error:
		logging.error(f'Error moving data from {source_table} to {target_table}: {error}')
		raise
