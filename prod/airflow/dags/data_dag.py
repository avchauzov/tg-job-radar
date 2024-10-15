import sys


sys.path.insert(0, '/home/job_search')

from prod.airflow.plugins.raw.data_collection import scrape_tg
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator


default_args = {
		'owner'          : 'avchauzov',
		'depends_on_past': False,
		'start_date'     : datetime(2023, 1, 1),
		'retries'        : 3,
		'retry_delay'    : timedelta(minutes=5),
		}

with DAG(
		'data',
		default_args=default_args,
		schedule_interval=timedelta(days=1),
		catchup=False,
		max_active_runs=1
		) as dag:
	def scrape_tg_function(**kwargs):
		scrape_tg()
	
	
	scrape_tg_operator = PythonOperator(
			task_id='scrape_tg_function',
			python_callable=scrape_tg_function,
			provide_context=True
			)
