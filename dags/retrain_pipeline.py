from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
import subprocess
import signal
import os
import time

def run_flask_app():
    """Run Flask indefinitely (never returns)"""
    subprocess.run(["python", "/opt/airflow/deployment/flask_app/app.py"])

from airflow.exceptions import AirflowFailException

import psutil

import sys  # Add this import at the top

from airflow.models import DagRun

def force_fail_dag(context):
    """Mark the entire DAG as failed if critical tasks fail."""
    dag_run = context['dag_run']
    dag_run.set_state('failed')  

'''def monitor_drift():
    """Drift detection with explicit success/failure states"""
    print("called the monitor_drift in the dag file")
    try:
        # Start monitoring process
        #monitor_proc = subprocess.Popen(
        #    ["python", "/opt/airflow/monitoring/monitoring.py"],
        #    stdout=subprocess.PIPE,
        #    stderr=subprocess.STDOUT,
        #    text=True,
        #    bufsize=1
        #)
        with subprocess.Popen(
            [sys.executable, "-u", "/opt/airflow/monitoring/monitoring.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as monitor_proc:
            for line in monitor_proc.stdout:
                print(line.strip())  # this will be shown in Airflow logs        
        print("Subprocess finished.")
        # Stream output and watch for drift
        flask_proc = None
        drift_detected = False
        
        for line in monitor_proc.stdout:
            print(line.strip())
            if "DRIFT_DETECTED" in line:  # Typo fixed from original
                print("🔄 Drift detected - initiating model update cycle...")
                drift_detected = True
                
                # Find and stop Flask process
                for proc in psutil.process_iter(['pid', 'cmdline']):
                    try:
                        if (proc.info['cmdline'] and 
                            'python' in proc.info['cmdline'][0] and 
                            '/opt/airflow/deployment/flask_app/app.py' in ' '.join(proc.info['cmdline'])):
                            flask_proc = proc
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if flask_proc:
                    print(f"🛑 Stopping Flask (PID: {flask_proc.pid})")
                    flask_proc.terminate()
                    try:
                        flask_proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        flask_proc.kill()
                
                # Stop monitoring process cleanly
                monitor_proc.terminate()
                monitor_proc.wait(timeout=5)
                break
        
        # Handle process completion
        if not drift_detected:
            monitor_proc.wait()
            if monitor_proc.returncode != 0:
                raise RuntimeError(f"Monitoring failed with exit code {monitor_proc.returncode}")
        
        # Explicit success state for drift detection
        if drift_detected:
            print("✅ Successfully detected drift - triggering pipeline refresh")
            return True  # Explicit success
            
    except Exception as e:
        print(f"⚠️ Critical monitoring failure: {e}")
        raise'''
def monitor_drift(): 
    """Drift detection with explicit success/failure states"""
    print("📡 Called the monitor_drift in the DAG file")
    try:
        # Start monitoring process
        monitor_proc = subprocess.Popen(
            [sys.executable, "-u", "/opt/airflow/monitoring/monitoring.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Read output and check for drift
        stdout, _ = monitor_proc.communicate()
        print(stdout)  # Log full output to Airflow logs

        drift_detected = "DRIFT_DETECTED" in stdout
        flask_proc = None

        if drift_detected:
            print("🔄 Drift detected - initiating model update cycle...")

            # Find and stop Flask process
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if (proc.info['cmdline'] and 
                        'python' in proc.info['cmdline'][0] and 
                        '/opt/airflow/deployment/flask_app/app.py' in ' '.join(proc.info['cmdline'])):
                        flask_proc = proc
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if flask_proc:
                print(f"🛑 Stopping Flask (PID: {flask_proc.pid})")
                flask_proc.terminate()
                try:
                    flask_proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    flask_proc.kill()

        # Check subprocess result
        if monitor_proc.returncode != 0 and not drift_detected:
            raise RuntimeError(f"Monitoring failed with exit code {monitor_proc.returncode}")

        # Return explicit success state
        if drift_detected:
            print("✅ Successfully detected drift - triggering pipeline refresh")
            return True

    except Exception as e:
        print(f"⚠️ Critical monitoring failure: {e}")
        raise

with DAG(
    "ml_pipeline_dag_updated",
    start_date=datetime(2025, 7, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        'retries': 0,  # Disable automatic retries
    }
) as dag:

    # Pipeline tasks
    ingest = BashOperator(
        task_id="data_ingestion",
        bash_command="python /opt/airflow/data_ingestion/from_postgres.py"
    )

    preprocess = BashOperator(
        task_id="preprocessing",
        bash_command="python /opt/airflow/preprocessing/pipelines.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/main.py",
    )

    predict = BashOperator(
        task_id="predict",
        bash_command="python /opt/airflow/ml/predict.py",
    )

    # Concurrent endpoints
    flask_app = PythonOperator(
        task_id="flask_app",
        python_callable=run_flask_app,
        # Critical settings for concurrency:
        do_xcom_push=False,  # No need to pass process info
        trigger_rule="all_success",
        retries=0
    )

    monitor_drift_task = PythonOperator(
        task_id="monitor_drift",
        python_callable=monitor_drift,
        trigger_rule="all_success",
        retries=0
    )
    restart_pipeline = TriggerDagRunOperator(
        task_id="restart_pipeline",
        trigger_dag_id="ml_pipeline_dag_updated",  # Same DAG
        wait_for_completion=False,
        trigger_rule="all_success",  # Only trigger if monitor_drift success
    )

    # Pipeline flow (key change - parallel execution)
    ingest >> preprocess >> train_model >> predict >> [flask_app, monitor_drift_task]

    monitor_drift_task >> restart_pipeline