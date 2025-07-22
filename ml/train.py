from sklearn.model_selection import train_test_split
from ml.utils import get_pipeline_only
from ml.model_experimentation import evaluate_classifiers
from preprocessing.imputation import get_full_preprocessing_pipeline
import pandas as pd

def train_sales_classifier(df: pd.DataFrame, target_column="Converted"):
    df = df.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split first
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get pipeline and fit ONLY on X_train
    pipeline, num_cols, ord_cols, bin_cols, cat_cols = get_full_preprocessing_pipeline(X_train_raw)
    pipeline.fit(X_train_raw)

    # Transform the Xtrain and Xtest data 
    X_train=pipeline.transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    # Train + Log
    evaluate_classifiers(X_train, X_test, y_train, y_test, pipeline)
