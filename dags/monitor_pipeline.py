from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import os

# Constants
PIPELINE_PATH = "/opt/airflow/data/fitted_pipeline.pkl"
MLFLOW_TRACKING_URI = "file:/opt/airflow/mlruns"  # or "http://host.docker.internal:5000" if running server

# Task function
def log_pipeline_to_mlflow():
    import mlflow
    from mlflow import log_artifact

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Check if file exists
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline file not found at: {PIPELINE_PATH}")

    print("✅ Found fitted pipeline:", PIPELINE_PATH)
    mlflow.set_experiment("Sales_conversion_Airflow_docker")
    # Log the artifact
    with mlflow.start_run(run_name="airflow_pipeline_log"):
        print("📦 Logging artifact to MLflow...")
        mlflow.log_artifact(PIPELINE_PATH, artifact_path="preprocessing_pipeline")
        print("✅ Logged to MLflow.")

# DAG definition
with DAG(
    dag_id="log_pipeline_to_mlflow",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual run only
    catchup=False,
    tags=["mlflow", "pipeline"],
) as dag:

    log_pipeline = PythonOperator(
        task_id="log_pipeline_artifact",
        python_callable=log_pipeline_to_mlflow
    )

    log_pipeline
