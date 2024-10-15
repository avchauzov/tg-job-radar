import logging

import asyncpg
import psycopg2
from psycopg2.extras import execute_batch

from .. import POSTGRES_HOST, POSTGRES_NAME, POSTGRES_PASS, POSTGRES_USER


def establish_db_connection():
	try:
		connection = psycopg2.connect(
				host=POSTGRES_HOST,
				database=POSTGRES_NAME,
				user=POSTGRES_USER,
				password=POSTGRES_PASS,
				port=5432
				)
		return connection
	
	except Exception as error:
		logging.error(f'Error connecting to the database: {error}')
		raise


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
		
		if connection:
			connection.rollback()


async def batch_insert_to_db_async(table_name, columns, conflict, data):
	try:
		if not data:
			logging.warning('No data provided for insertion.')
			return
		
		columns_str = ', '.join(f'"{column}"' if column.isdigit() else column for column in columns)
		conflict_str = ', '.join(f'"{column}"' if column.isdigit() else column for column in conflict)
		columns_types = ', '.join(['$' + str(i) for i in range(1, len(columns) + 1)])
		
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
		
		connection = await asyncpg.connect(
				user=POSTGRES_USER, password=POSTGRES_PASS,
				database=POSTGRES_NAME, host=POSTGRES_HOST
				)
		try:
			async with connection.transaction():
				await connection.executemany(db_insert_query, data_tuples)
			
			logging.info(f'Successfully inserted {len(data_tuples)} rows into {table_name}')
		
		finally:
			await connection.close()
	
	except Exception as error:
		logging.error(f'Error inserting data: {error}')


def fetch_from_db(table_name, select_condition='', where_condition='', order_condition=''):
	try:
		select_query = f'SELECT * FROM {table_name}'
		
		if select_condition:
			select_query = f'SELECT {select_condition} FROM {table_name}'
		
		if where_condition:
			select_query += f' WHERE {where_condition}'
		
		if order_condition:
			select_query += f' ORDER BY {order_condition}'
		
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
