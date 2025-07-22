# ml/utils.py

from preprocessing.imputation import get_full_preprocessing_pipeline

def get_pipeline_only(X):
    return get_full_preprocessing_pipeline(X)
