import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Sales_conversion_Airflow_docker"
MODEL_NAME = "best_conversion_model_airflow"
METRIC_TO_OPTIMIZE = "f1_score"  # Can be f1_score, precision, recall etc.

def promote_best_model_to_production():
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns") # added extra for docker
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' does not exist.")
    
    experiment_id = experiment.experiment_id

    print(f"🔍 Searching for best run sorted by {METRIC_TO_OPTIMIZE}...")
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{METRIC_TO_OPTIMIZE} DESC"],
        max_results=1000
    )

    runs = runs[runs[f"metrics.{METRIC_TO_OPTIMIZE}"].notnull()]
    if runs.empty:
        print("❌ No valid runs found.")
        return

    best_run = runs.iloc[0]
    run_id = best_run.run_id
    best_score = best_run[f"metrics.{METRIC_TO_OPTIMIZE}"]

    artifact_path = None
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    for artifact in artifacts:
        if artifact.path.endswith("_model"):
            artifact_path = artifact.path
            break

    if not artifact_path:
        print("❌ No final model artifact found in the best run.")
        return

    print(f"✅ Best run: {run_id} | {METRIC_TO_OPTIMIZE} = {best_score:.4f}")
    print(f"📦 Artifact path: {artifact_path}")

    model_uri = f"runs:/{run_id}/{artifact_path}"
    print("📥 Registering model...")
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    import time
    while True:
        model_info = client.get_model_version(name=MODEL_NAME, version=model_version.version)
        if model_info.status == "READY":
            break
        time.sleep(1)

    # Get current production model version (if exists)
    print("🔎 Checking existing Production model...")
    current_production_versions = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
    
    if current_production_versions:
        current_version = current_production_versions[0]
        current_metrics = mlflow.get_run(current_version.run_id).data.metrics
        current_score = current_metrics.get(METRIC_TO_OPTIMIZE, None)

        print(f"📊 Current Production version: {current_version.version} | Score: {current_score}")
        print(f"📊 New Candidate version: {model_version.version} | Score: {best_score}")

        if current_score is not None and best_score < current_score:
            print("⚠️ New model is not better than current Production model. Skipping promotion.")
            return
        else:
            print("✅ New model is better. Proceeding with promotion.")
    else:
        print("ℹ️ No model in Production yet. Proceeding with first promotion.")

    # Promote the better model to Production
    print(f"🚀 Transitioning model version {model_version.version} to Production...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🎉 Model '{MODEL_NAME}' version {model_version.version} is now in PRODUCTION!")

# Uncomment to run
#promote_best_model_to_production()
