import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaningPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='converted', drop_threshold=0.8):
        self.target_column = target_column
        self.drop_threshold = drop_threshold

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, df):
        df = df.copy()

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        # Remove contradictory rows (Do Not Email & Call are 'yes' but converted is 1)
        contradiction_cols = ['do not email', 'do not call', self.target_column]
        if all(col in df.columns for col in contradiction_cols):
            contradiction_mask = (
                (df['do not email'].str.lower() == 'yes') &
                (df['do not call'].str.lower() == 'yes') &
                (df[self.target_column] == 1)
            )
            df = df[~contradiction_mask]

        # Drop columns with too many missing values
        missing_ratio = df.isna().mean()
        high_missing_cols = missing_ratio[missing_ratio > self.drop_threshold].index
        df.drop(columns=high_missing_cols, inplace=True)

        return df
