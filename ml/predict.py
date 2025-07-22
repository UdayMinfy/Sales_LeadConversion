import mlflow
import pandas as pd
import joblib
import os
import shutil
import sys
import shap
import numpy as np
from mlflow.tracking import MlflowClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_NAME = "best_conversion_model_airflow"
ARTIFACT_SUBPATH_PIPELINE = "preprocessing_pipeline/fitted_pipeline.pkl"

def load_production_model_and_pipeline():
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")  # Added Extra for Docker 
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        raise ValueError("❌ No model found in 'Production' stage.")

    prod_model = prod_versions[0]
    run_id = prod_model.run_id
    artifact_uri = f"runs:/{run_id}/{ARTIFACT_SUBPATH_PIPELINE}"

    print(f"✅ Loading model from: {prod_model.source}")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

    local_temp_path = mlflow.artifacts.download_artifacts(artifact_uri)
    target_path = os.path.join(os.getcwd(), "downloaded_fitted_pipeline.pkl")
    shutil.copy(local_temp_path, target_path)

    pipeline = joblib.load(target_path)
    print(f"✅ Pipeline loaded from: {target_path}")
    return model, pipeline

def predict_sample(model, pipeline):
    raw_input = pd.DataFrame([{
        "Prospect ID": "7927b2df-8bba-4d29-b9a2-b6e0beafe620",
        "Lead Number": 660737,
        'Lead Origin': 'API',
        'Lead Source': 'Olark Chat',
        'Do Not Email': 'No',
        'Do Not Call': 'No',
        'TotalVisits': 0,
        'Total Time Spent on Website': 0,
        'Page Views Per Visit': 0,
        'Last Activity': 'Page Visited on Website',
        'Country': None,
        'Specialization': 'Select',
        'How did you hear about X Education': 'Select',
        'What is your current occupation': 'Unemployed',
        'What matters most to you in choosing a course': 'Better Career Prospects',
        'Search': 'No',
        'Magazine': 'No',
        'Newspaper Article': 'No',
        'X Education Forums': 'No',
        'Newspaper': 'No',
        'Digital Advertisement': 'No',
        'Through Recommendations': 'No',
        'Receive More Updates About Our Courses': 'No',
        'Tags': 'Interested in other courses',
        'Lead Quality': 'Low in Relevance',
        'Update me on Supply Chain Content': 'No',
        'Get updates on DM Content': 'No',
        'Lead Profile': 'Select',
        'City': 'Select',
        'Asymmetrique Activity Index': '02.Medium',
        'Asymmetrique Profile Index': '02.Medium',
        'Asymmetrique Activity Score': 15,
        'Asymmetrique Profile Score': 15,
        'I agree to pay the amount through cheque': 'No',
        'A free copy of Mastering The Interview': 'No',
        'Last Notable Activity': 'Modified'
    }])

    # Transform input
    processed_input = pipeline.transform(raw_input)

    # Get feature names after preprocessing
    try:
        feature_names = pipeline.get_feature_names_out()
    except:
        # Handle older sklearn versions
        from sklearn.compose import ColumnTransformer
        if isinstance(pipeline, ColumnTransformer):
            feature_names = pipeline.get_feature_names()
        else:
            feature_names = [f"f{i}" for i in range(processed_input.shape[1])]

    # Convert to DataFrame with named columns
    processed_df = pd.DataFrame(processed_input, columns=feature_names)

    # Prediction
    prediction = model.predict(processed_input)
    print(f"🧠 Prediction: {prediction[0]}")

    # Feature Importance from RandomForest
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n🎯 Top Features by RandomForest Importance:\n")
    print(importance_df.head(10))

    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(processed_df)
    feature_names = processed_df.columns
    input_values = processed_df.iloc[0].values
    shap_vals = shap_values.values[0].flatten()
    
    # Zip and create list of tuples: (feature, input_value, shap_value)
    shap_result = list(zip(feature_names, input_values, shap_vals))
    
    # Sort by absolute SHAP value in descending order
    shap_result_sorted = sorted(shap_result, key=lambda x: abs(x[2]), reverse=True)
    
    # Display top 10
    print("\n🔝 Top 10 SHAP Features:")
    for feat, val, shap_val in shap_result_sorted[:10]:
        print(f"{feat}: input={val} ➜ SHAP={shap_val:.4f}")
    #for feat, val, shap_val in zip(feature_names, processed_df.iloc[0], shap_values.values[0]):
        #print(f"{feat}: input={val} ➜ SHAP={shap_val[0]:.4f}")


if __name__ == "__main__":
    model, pipeline = load_production_model_and_pipeline()
    predict_sample(model, pipeline)
