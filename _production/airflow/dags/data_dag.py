import sys


sys.path.insert(0, "/home/job_search")

from _production.airflow.plugins.raw.data_collection import scrape_tg
from _production.airflow.plugins.staging.data_cleaning import clean_and_move_data
from _production.airflow.plugins.production.email_notifications import notify_me
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator  # TODO: install and check actuality


default_args = {
    "owner": "job_search",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email": ["avchauzov.dev@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    # TODO: connect email to phone and email clients
}

with DAG(
    "data",
    default_args=default_args,
    schedule=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
) as dag:

    def scrape_tg_function(**kwargs):
        scrape_tg()

    def clean_and_move_data_function(**kwargs):
        clean_and_move_data()

    def notify_me_function(**kwargs):
        notify_me()

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
