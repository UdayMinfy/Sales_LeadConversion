# main.py

import pandas as pd
from ml.train import train_sales_classifier
from ml.bestmodel_to_production import promote_best_model_to_production

if __name__ == "__main__":
    df = pd.read_csv("/opt/airflow/data/processed/eda_cleaned.csv")
    train_sales_classifier(df, target_column="Converted")
    promote_best_model_to_production()