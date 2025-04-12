"""
Telegram Job Radar DAG.

This DAG performs the following operations:
1. Starts the LLM instance if needed
2. Scrapes job postings from specified Telegram channels
3. Cleans and processes the collected data
4. Sends notification about the results
5. Stops the LLM instance

Schedule: Daily at 6 AM GMT
"""

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

project_path = "/home/tg-job-radar"
if project_path not in sys.path:
    sys.path.append(project_path)

from _production import EC2_MANAGER
from _production.airflow.plugins.instance.instance_manager import InstanceManager
from _production.airflow.plugins.production.email_notifications import notify_me
from _production.airflow.plugins.raw.data_collection import scrape_tg
from _production.airflow.plugins.staging.data_cleaning import clean_and_move_data

default_args = {
    "owner": "tg-job-radar",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=60),
    "email": ["avchauzov.dev@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
}

# Initialize instance manager
manager = InstanceManager(base_url=EC2_MANAGER["EC2_MANAGER_URL"])


def start_instance(**context):
    """Start the LLM instance."""
    status = manager.start_instance(
        instance_id=EC2_MANAGER["LLM_INSTANCE_ID"],
        region=EC2_MANAGER["LLM_INSTANCE_REGION"],
        aws_access_key_id=EC2_MANAGER["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=EC2_MANAGER["AWS_SECRET_ACCESS_KEY"],
    )
    if status.state == "error":
        raise Exception(f"Failed to start instance: {status.message}")
    return status


'''def stop_instance(**context):
    """Stop the LLM instance."""
    status = manager.stop_instance(
        instance_id=EC2_MANAGER["LLM_INSTANCE_ID"],
        region=EC2_MANAGER["LLM_INSTANCE_REGION"],
        aws_access_key_id=EC2_MANAGER["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=EC2_MANAGER["AWS_SECRET_ACCESS_KEY"],
    )
    if status.state == "error":
        raise Exception(f"Failed to stop instance: {status.message}")
    return status'''


with DAG(
    "tg-job-radar",
    default_args=default_args,
    schedule="0 6 * * *",  # Run at 6 AM GMT daily
    catchup=False,
    max_active_runs=1,
) as dag:

    def scrape_tg_function(**kwargs):
        scrape_tg()

    def clean_and_move_data_function(**kwargs):
        clean_and_move_data()

    def notify_me_function(**kwargs):
        notify_me()

    # Instance management tasks
    start_instance_operator = PythonOperator(
        task_id="start_instance",
        python_callable=start_instance,
        provide_context=True,
    )

    """stop_instance_operator = PythonOperator(
        task_id="stop_instance",
        python_callable=stop_instance,
        provide_context=True,
        trigger_rule="all_done",
    )"""

    # Main tasks
    scrape_tg_operator = PythonOperator(
        task_id="scrape_tg_function",
        python_callable=scrape_tg_function,
    )

    clean_and_move_data_operator = PythonOperator(
        task_id="clean_and_move_data_function",
        python_callable=clean_and_move_data_function,
    )

    notify_me_operator = PythonOperator(
        task_id="notify_me_function",
        python_callable=notify_me_function,
    )

    # Set up task dependencies
    (
        scrape_tg_operator
        >> start_instance_operator
        >> clean_and_move_data_operator
        >> notify_me_operator
        # >> stop_instance_operator
    )
