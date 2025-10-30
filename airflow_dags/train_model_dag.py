from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml_core.train import train_models

default_args = {
    'owner': 'mlops',
    'retries': 1,
}

with DAG(
    dag_id='train_models_dag',
    default_args=default_args,
    description='Автоматический DAG для обучения ML моделей',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training'],
    schedule="@once",
    is_paused_upon_creation=False
) as dag:

    train_task = PythonOperator(
        task_id='train_models_task',
        python_callable=train_models,
    )