import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

# Enable pandas output from transformers
set_config(transform_output="pandas")


# Make sure this is defined at the top of your script
binary_cols = [
    'donotemail', 'donotcall', 'search', 'magazine', 'newspaperarticle',
    'xeducationforums', 'newspaper', 'digitaladvertisement',
    'throughrecommendations', 'receivemoreupdatesaboutourcourses',
    'updatemeonsupplychaincontent', 'getupdatesondmcontent',
    'iagreetopaytheamountthroughcheque', 'afreecopyofmasteringtheinterview'
]
# 1. First define all helper functions and classes

def safe_lower_strip(x):
    if pd.isna(x) or str(x).strip().lower() in ['nan', 'select', '']:
        return 'missing'
    return str(x).strip().lower()

def clean_categorical_values(df):
    return df.applymap(safe_lower_strip)

def clean_ordinal_values(x):
    return x.applymap(clean_ordinal_text)

def clean_ordinal_text(v):
    return str(v).strip().lower().replace('01.', '').replace('02.', '').replace('03.', '') if pd.notnull(v) else 'missing'

def fix_common_bad_values(x):
    return x.replace({'nan': 'missing', 'select': 'missing'}, regex=True)

def convert_to_dataframe_with_binary_cols(x):
    return pd.DataFrame(x, columns=binary_cols)

def normalize_column_names(df):
    return df.rename(columns=lambda c: c.lower().replace(' ', '').strip())

class NamedFunctionTransformer(FunctionTransformer):
    """FunctionTransformer that preserves feature names"""
    def get_feature_names_out(self, input_features=None):
        return input_features

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoding for binary columns that preserves DataFrame output"""
    def __init__(self):
        self.le_dict = {}
        self.columns = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.columns = X.columns.tolist()
        else:
            raise ValueError("CustomLabelEncoder requires DataFrame input with column names")
        
        for col in self.columns:
            le = LabelEncoder()
            self.le_dict[col] = le.fit(X[col].astype(str))
        return self

    def transform(self, X):
        if not hasattr(X, 'columns'):
            if self.columns is None:
                raise ValueError("Column names not available for numpy array input")
            X = pd.DataFrame(X, columns=self.columns)
            
        X_encoded = X.copy()
        for col in self.columns:
            le = self.le_dict[col]
            X_encoded[col] = le.transform(X_encoded[col].astype(str))
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)

def get_feature_names(pipeline, numeric_cols, ordinal_cols, binary_cols, categorical_cols):
    col_transformer = pipeline.named_steps['transform']

    ohe = col_transformer.named_transformers_['cat'].named_steps['onehot']
    onehot_feature_names = list(ohe.get_feature_names_out(categorical_cols))

    all_feature_names = numeric_cols + ordinal_cols + binary_cols + onehot_feature_names
    return all_feature_names


# 2. Now define the main pipeline function
def get_full_preprocessing_pipeline(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '')

    numeric_cols = [
        'totalvisits', 'totaltimespentonwebsite', 'pageviewspervisit',
        'asymmetriqueactivityscore', 'asymmetriqueprofilescore'
    ]

    ordinal_cols = ['asymmetriqueactivityindex', 'asymmetriqueprofileindex']
    ordinal_categories = [['missing','low', 'medium', 'high']] * 2

    binary_cols = [
        'donotemail', 'donotcall', 'search', 'magazine', 'newspaperarticle',
        'xeducationforums', 'newspaper', 'digitaladvertisement',
        'throughrecommendations', 'receivemoreupdatesaboutourcourses',
        'updatemeonsupplychaincontent', 'getupdatesondmcontent',
        'iagreetopaytheamountthroughcheque', 'afreecopyofmasteringtheinterview'
    ]

    id_columns = ['prospectid', 'leadnumber','converted']
    existing_ids = [col for col in id_columns if col in df.columns]
    if existing_ids:
        print(f"Dropping ID columns: {existing_ids}")
        df.drop(columns=existing_ids, inplace=True)



    categorical_cols = [
        'country', 'tags', 'leadsource', 'specialization', 'lastactivity',
        'lastnotableactivity', 'howdidyouhearaboutxeducation', 'city',
        'leadprofile', 'whatisyourcurrentoccupation', 'leadquality',
        'leadorigin', 'whatmattersmosttoyouinchoosingacourse'
    ]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', NamedFunctionTransformer(func=np.log1p, validate=False)),
        ('scaler', MinMaxScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('clean_text', NamedFunctionTransformer(clean_ordinal_values)),
        ('fix_common_bad_values', NamedFunctionTransformer(fix_common_bad_values)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories))
    ])

    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('to_df', NamedFunctionTransformer(convert_to_dataframe_with_binary_cols)),
        ('label_encoder', CustomLabelEncoder())
    ])

    categorical_pipeline = Pipeline([
        ('clean_values', NamedFunctionTransformer(clean_categorical_values)),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('ord', ordinal_pipeline, ordinal_cols),
            ('bin', binary_pipeline, binary_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    full_pipeline = Pipeline([
        ('clean_column_names', NamedFunctionTransformer(normalize_column_names)),
        ('transform', column_transformer)
    ])

    return full_pipeline, numeric_cols, ordinal_cols, binary_cols, categorical_cols
