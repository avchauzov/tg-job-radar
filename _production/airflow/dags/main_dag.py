import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, "/home/job_search")
sys.path.insert(0, str(PROJECT_ROOT))

from _production.airflow.plugins.raw.data_collection import scrape_tg
from _production.airflow.plugins.staging.data_cleaning import clean_and_move_data
from _production.airflow.plugins.production.email_notifications import notify_me
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import logging

default_args = {
    "owner": "job_search",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email": ["avchauzov.dev@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
}


with DAG(
    "job_search",
    default_args=default_args,
    schedule=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
) as dag:

    def scrape_tg_function(**kwargs):
        try:
            scrape_tg()
        except Exception as error:
            logging.error(f"Error in scrape_tg_function: {str(error)}")
            raise AirflowException(f"Scraping failed: {str(error)}")

    def clean_and_move_data_function(**kwargs):
        try:
            clean_and_move_data()
        except Exception as error:
            logging.error(f"Error in clean_and_move_data_function: {str(error)}")
            raise AirflowException(f"Data cleaning failed: {str(error)}")

    def notify_me_function(**kwargs):
        try:
            notify_me()
        except Exception as error:
            logging.error(f"Error in notify_me_function: {str(error)}")
            raise AirflowException(f"Notification failed: {str(error)}")

    scrape_tg_operator = PythonOperator(
        task_id="scrape_tg_function", python_callable=scrape_tg_function
    )

    clean_and_move_data_operator = PythonOperator(
        task_id="clean_and_move_data_function",
        python_callable=clean_and_move_data_function,
    )

    notify_me_operator = PythonOperator(
        task_id="notify_me_function", python_callable=notify_me_function
    )

    scrape_tg_operator >> clean_and_move_data_operator >> notify_me_operator
