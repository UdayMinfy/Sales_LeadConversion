from flask import Flask, render_template, request,  jsonify, send_file
import pandas as pd
import mlflow
import shap
import joblib
import os
import sys
import io
import numpy as np
from io import BytesIO
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = Flask(__name__)

dropdowns = {
    "Lead Origin": ["Landing Page Submission", "API", "Lead Add Form", "Lead Import", "Quick Add Form"],
    "Lead Source": ["Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference", "Welingak Website", "Referral Sites", "Facebook", "bing", "google", "Click2call", "Press_Release", "Social Media", "Live Chat", "youtubechannel", "testone", "Pay per Click Ads", "welearnblog_Home", "WeLearn", "blog", "NC_EDM"],
    "Do Not Email": ["No", "Yes"],
    "Do Not Call": ["No", "Yes"],
    "Last Notable Activity": ["Modified", "Email Opened", "SMS Sent", "Page Visited on Website", "Olark Chat Conversation", "Email Link Clicked", "Email Bounced", "Unsubscribed", "Unreachable", "Had a Phone Conversation", "Email Marked Spam", "Approached upfront", "Resubscribed to emails", "View in browser link Clicked", "Form Submitted on Website", "Email Received"],
    "Last Activity": ["Email Opened", "SMS Sent", "Olark Chat Conversation", "Page Visited on Website", "Converted to Lead", "Email Bounced", "Email Link Clicked", "Form Submitted on Website", "Unreachable", "Unsubscribed", "Had a Phone Conversation", "Approached upfront", "View in browser link Clicked", "Email Received", "Email Marked Spam", "Visited Booth in Tradeshow", "Resubscribed to emails"],
    "Country": ["India", "United States", "United Arab Emirates", "Singapore", "Saudi Arabia", "United Kingdom", "Australia", "Qatar", "Hong Kong", "Bahrain", "Oman", "France", "unknown", "South Africa", "Nigeria", "Germany", "Kuwait", "Canada", "Sweden", "China", "Asia/Pacific Region", "Uganda", "Bangladesh", "Italy", "Belgium", "Netherlands", "Ghana", "Philippines", "Russia", "Switzerland", "Vietnam", "Denmark", "Tanzania", "Liberia", "Malaysia", "Kenya", "Sri Lanka", "Indonesia"],
    "Specialization": ["Select", "Finance Management", "Human Resource Management", "Marketing Management", "Operations Management", "Business Administration", "IT Projects Management", "Supply Chain Management", "Banking, Investment And Insurance", "Travel and Tourism", "Media and Advertising", "International Business", "Healthcare Management", "Hospitality Management", "E-COMMERCE", "Retail Management", "Rural and Agribusiness", "E-Business", "Services Excellence"],
    "How did you hear about X Education": ["Select", "Online Search", "Word Of Mouth", "Student of SomeSchool", "Other", "Multiple Sources", "Advertisements", "Social Media", "Email", "SMS"],
    "What is your current occupation": ["Unemployed", "Working Professional", "Student", "Other", "Housewife", "Businessman"],
    "What matters most to you in choosing a course": ["Better Career Prospects", "Flexibility & Convenience", "Other"],
    "Search": ["No", "Yes"],
    "Magazine": ["No"],
    "Newspaper Article": ["No", "Yes"],
    "X Education Forums": ["No", "Yes"],
    "Newspaper": ["No", "Yes"],
    "Digital Advertisement": ["No", "Yes"],
    "Through Recommendations": ["No", "Yes"],
    "Receive More Updates About Our Courses": ["No"],
    "Tags": ["Will revert after reading the email", "Ringing", "Interested in other courses", "Already a student", "Closed by Horizzon", "switched off", "Busy", "Lost to EINS", "Not doing further education", "Interested  in full time MBA", "Graduation in progress", "invalid number", "Diploma holder (Not Eligible)", "wrong number given", "opp hangup", "number not provided", "in touch with EINS", "Lost to Others", "Still Thinking", "Want to take admission but has financial problems", "In confusion whether part time or DLP", "Interested in Next batch", "Lateral student", "Shall take in the next coming month", "University not recognized", "Recognition issue (DEC approval)"],
    "Lead Quality": ["Might be", "Not Sure", "High in Relevance", "Worst", "Low in Relevance"],
    "Update me on Supply Chain Content": ["No"],
    "Get updates on DM Content": ["No"],
    "Lead Profile": ["Select", "Potential Lead", "Other Leads", "Student of SomeSchool", "Lateral Student", "Dual Specialization Student"],
    "City": ["Mumbai", "Select", "Thane & Outskirts", "Other Cities", "Other Cities of Maharashtra", "Other Metro Cities", "Tier II Cities"],
    "Asymmetrique Activity Index": ["02.Medium", "01.High", "03.Low"],
    "Asymmetrique Profile Index": ["02.Medium", "01.High", "03.Low"],
    "I agree to pay the amount through cheque": ["No"],
    "A free copy of Mastering The Interview": ["No", "Yes"]
}

numeric_columns = [
        'TotalVisits',
        'Total Time Spent on Website',
        'Page Views Per Visit',
        'Asymmetrique Activity Score',
        'Asymmetrique Profile Score'
    ]


@app.route('/')
def form():

    return render_template("form.html", dropdowns=dropdowns, numeric_columns=numeric_columns)

MODEL_NAME = "best_conversion_model_airflow"
#MODEL_NAME = "best_conversion_model"
PIPELINE_PATH = "preprocessing_pipeline/fitted_pipeline.pkl"

# Load model and preprocessing pipeline
def load_model_and_pipeline():
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns") # Extra 
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        raise Exception("No production model found.")

    run_id = prod_versions[0].run_id
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

    artifact_uri = f"runs:/{run_id}/{PIPELINE_PATH}"
    local_pipeline_path = mlflow.artifacts.download_artifacts(artifact_uri)
    pipeline = joblib.load(local_pipeline_path)

    return model, pipeline

model, pipeline = load_model_and_pipeline()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])
        
        # Cast numeric fields
        #for col in numeric_columns:
        #    if col in input_df.columns:
        #        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Combine into full input
        raw_input = input_df.copy()

        # Transform input
        processed_input = pipeline.transform(raw_input)

        # Get feature names after preprocessing
        try:
            feature_names = pipeline.get_feature_names_out()
        except:
            from sklearn.compose import ColumnTransformer
            if isinstance(pipeline, ColumnTransformer):
                feature_names = pipeline.get_feature_names()
            else:
                feature_names = [f"f{i}" for i in range(processed_input.shape[1])]

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_input, columns=feature_names)

        # Predict
        prediction = model.predict(processed_input)
        result = int(prediction[0])
        print(f"🧠 Prediction: {result}")

        raw_input['Converted']=result
        # DATABASE_URI = 'postgresql+psycopg2://postgres:newpassword@localhost:5432/mydb5'  # for localmachine
        
        DATABASE_URI = 'postgresql+psycopg2://postgres:newpassword@host.docker.internal:5432/mydb5' # for docker
        engine = create_engine(DATABASE_URI)
        raw_input.to_sql('RealTimeLeadScoring', con=engine, if_exists='append', schema='public', index=False)

        # RandomForest Feature Importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\n🎯 Top Features by RandomForest Importance:")
        print(importance_df.head(10))

        # SHAP Explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(processed_df)
        shap_vals = shap_values.values[0].flatten()
        input_values = processed_df.iloc[0].values

        shap_result = list(zip(feature_names, input_values, shap_vals))
        shap_sorted = sorted(shap_result, key=lambda x: abs(x[2]), reverse=True)

        print("\n🔝 Top 10 SHAP Features:")
        for feat, val, shap_val in shap_sorted[:10]:
            print(f"{feat}: input={val} ➜ SHAP={shap_val:.4f}")
        # Convert RandomForest importance to readable list
        rf_top_features = [
            f"{row['Feature']}: {row['Importance']:.4f}"
            for _, row in importance_df.head(10).iterrows()
        ]

        # Convert SHAP values to readable strings
        shap_explanation = [
            f"{feat}: input={val} ➜ SHAP={shap_val:.4f}"
            for feat, val, shap_val in shap_sorted[:10]
        ]
        return render_template("form.html", prediction=result, top_features=rf_top_features, shap_explanations=shap_explanation, dropdowns=dropdowns,numeric_columns=numeric_columns )

    except Exception as e:
        return f"❌ Error: {str(e)}", 500

# Batch prediction route
@app.route("/batch_predict", methods=["GET","POST"])
def predict_batch_csv():
    file = request.files['file']
    if 'file' not in request.files:
        return jsonify({"error": "CSV file is required."}), 400

    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files allowed"}), 400

    try:
        df = pd.read_csv(file)

        results = []
        feature_names = pipeline.get_feature_names_out()  # optional fallback: use X.columns

        for idx, row in df.iterrows():
            # Transform one row
            row_df = pd.DataFrame([row])
            processed_df = pipeline.transform(row_df)
            feature_names = pipeline.get_feature_names_out()
    
            # Predict
            prediction = model.predict(processed_df)[0]
    
            # SHAP explanation
            explainer = shap.Explainer(model)
            shap_values = explainer(processed_df)
            shap_vals = shap_values.values[0].flatten()
            input_values = processed_df[0].toarray().flatten() if hasattr(processed_df[0], "toarray") else processed_df[0]
    
            shap_result = list(zip(feature_names, input_values, shap_vals))
            shap_sorted = sorted(shap_result, key=lambda x: abs(x[2]), reverse=True)
    
            # Format explanation
            explanation_text = "; ".join(
                f"{feat}={val:.2f} → {'↑' if sv >= 0 else '↓'} ${abs(sv):.2f}"
                for feat, val, sv in shap_sorted
            )
    
            row_result = row.to_dict()
            row_result["predicted_value"] = prediction
            row_result["shap_explanation"] = explanation_text
            results.append(row_result)
    
        # Merge results with original df
        results_df = pd.DataFrame(results)
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Save to CSV in memory
        csv_buffer = BytesIO()
        final_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Return as downloadable file
        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions_with_shap.csv'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)