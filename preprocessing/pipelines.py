from sklearn.pipeline import Pipeline
from clean_data import DataCleaningPipeline
from imputation import get_full_preprocessing_pipeline,get_feature_names
from sklearn.model_selection import train_test_split
import pandas as pd

def get_cleaning_pipeline():
    return Pipeline([
        ('cleaner', DataCleaningPipeline())
    ])

raw_df=pd.read_csv("/opt/airflow/data/raw/LeadScoring.csv")

cleaning_pipeline = get_cleaning_pipeline()
clean_df = cleaning_pipeline.fit_transform(raw_df)
print(clean_df.head())
print(clean_df.columns)
X=clean_df.drop(columns=['Converted'])
y=clean_df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# 1. Create pipeline and get column groupings
pipeline, num_cols, ord_cols, bin_cols, cat_cols = get_full_preprocessing_pipeline(X_train)

# 2. Fit pipeline externally
pipeline.fit(X_train)

# 3. Transform the data
Xtrain_transformed = pipeline.transform(X_train)  # returns NumPy array

# 4. Get feature names
feature_names = get_feature_names(pipeline, num_cols, ord_cols, bin_cols, cat_cols)
print(feature_names, len(feature_names))

Xtest_transformed= pipeline.transform(X_test)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# 6. Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 7. Train and evaluate each model
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(Xtrain_transformed, y_train)
    y_pred = model.predict(Xtest_transformed)
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
