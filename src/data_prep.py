# src/data_prep.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status"
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["id"])
    return df

def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ])
    return preprocessor

def prepare_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    joblib.dump(preprocessor, path)

def save_baseline_stats(X_train, path: str = "data/interim/baseline_stats.json"):
    # Compute means for each numeric feature
    stats = X_train[NUMERIC_FEATURES].mean().to_dict()
    # Ensure the interim folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write out as JSON
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved baseline stats to {path}")


def main(
    input_path: str = "data/raw/healthcare-dataset-stroke-data.csv",
    test_size: float = 0.2
):
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = prepare_data(df, test_size)
    save_baseline_stats(X_train)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    save_preprocessor(preprocessor)
    print(f"Preprocessor and baseline stats saved.")

if __name__ == "__main__":
    main()