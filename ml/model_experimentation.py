# ml/model_experimentation.py

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
import joblib
import tempfile
from pathlib import Path



def run_classification_experiment(model_name, model, param_grid, X_train, y_train, X_test, y_test, pipeline=None):
           # added extra 
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    # Ensure MLflow directory exists
    #mlruns_dir = Path("file:///opt/airflow/mlruns")
    #mlruns_dir.mkdir(parents=True, exist_ok=True)
    
    #mlflow.set_tracking_uri(str(mlruns_dir))
    mlflow.set_experiment("Sales_conversion_Airflow_docker")

    with mlflow.start_run(run_name=model_name):
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Save model
        #mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name}_model")

        # Save the fitted pipeline
        if pipeline:
            # Define base artifact directory (shared volume)
            artifact_dir = "/opt/airflow/data/artifacts"
            os.makedirs(artifact_dir, exist_ok=True)  # Ensure the directory exists

            # Define full path for the artifact
            pipeline_filename = "fitted_pipeline.pkl"
            pipeline_path = os.path.join(artifact_dir, pipeline_filename)

            # Save the artifact (e.g., a preprocessing pipeline)
            joblib.dump(pipeline, pipeline_path)

            # Log the artifact using MLflow
            if os.path.exists(pipeline_path):
                mlflow.log_artifact(pipeline_path, artifact_path="preprocessing_pipeline")
                
                print(f"✅ Logged artifact: {pipeline_path}")
            else:
                print(f"❌ Could not find artifact at: {pipeline_path}")
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        best_model.fit(X_full, y_full)

        # Save full model again
        mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name}_final_model")

        print(f"\n✅ {model_name} Logged to MLflow:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

def evaluate_classifiers(X_train, X_test, y_train, y_test, pipeline=None):
    dt_params = {
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }

    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 3],
    }

    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5]
    }

    run_classification_experiment("DecisionTree", DecisionTreeClassifier(random_state=42), dt_params, X_train, y_train, X_test, y_test, pipeline)
    run_classification_experiment("RandomForest", RandomForestClassifier(random_state=42), rf_params, X_train, y_train, X_test, y_test, pipeline)
    run_classification_experiment("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_params, X_train, y_train, X_test, y_test, pipeline)
