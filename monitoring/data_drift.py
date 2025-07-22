import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from sklearn.model_selection import train_test_split
import sys 
import os 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mlflow.set_experiment("Lead Data Drift Analysis")

# === Function to run and log drift analysis ===
def analyze_data_drift(final_df):
    # Step 1: Split features/target
    X = final_df.drop(columns=["Converted"])
    y = final_df["Converted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert y to DataFrame
    y_train_df = pd.DataFrame({"converted": y_train})
    y_test_df = pd.DataFrame({"converted": y_test})

    # Step 2: Run and log reports
    def log_data_drift_run(run_name, reference_df, current_df, report_file):
        report = Report(metrics=[
         DataQualityPreset(),
            DataDriftPreset(),
            TargetDriftPreset()
        ])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(report_file)
        report_dict = report.as_dict()

        with mlflow.start_run(run_name=run_name):
            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DataDriftTable":
                    for feature, result in metric["result"].get("drift_by_columns", {}).items():
                        mlflow.log_metric(f"{feature}_drift_score", result.get("drift_score"))
                        mlflow.log_param(f"{feature}_stat_test", result.get("stat_test_name"))
                        mlflow.log_param(f"{feature}_drift_detected", result.get("drift_detected"))
            mlflow.log_artifact(report_file, artifact_path="evidently_report")
            print(f"✅ Logged {run_name} → Report: {report_file}")

    # Run drift checks
    log_data_drift_run(
        run_name="X_train_vs_X_test",
        reference_df=X_train,
        current_df=X_test,
        report_file="report_X_train_vs_X_test.html"
    )

    log_data_drift_run(
        run_name="y_train_vs_y_test",
        reference_df=y_train_df,
        current_df=y_test_df,
        report_file="report_y_train_vs_y_test.html"
    )

    print("✅ Data drift analysis complete.")
df = pd.read_csv("data/processed/eda_cleaned.csv")
analyze_data_drift(df)