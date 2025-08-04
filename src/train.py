# src/train.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
import mlflow.sklearn

from data_prep import load_data, prepare_data, build_preprocessor
from data_prep import save_baseline_stats


# ─── MLflow Tracking URI ───────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
mlflow.set_tracking_uri(tracking_uri)

def main(
    input_path: str = "data/raw/healthcare-dataset-stroke-data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
):
    # 1. Load & split data
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = prepare_data(df, test_size, random_state)
    save_baseline_stats(X_train)

    # 2. Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # 3. Define the full pipeline: preprocessing + classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # 4. Start MLflow run & log params
    mlflow.set_experiment("StrokePrediction")
    with mlflow.start_run():
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        }
        mlflow.log_params(params)

        # 5. Fit pipeline on training data
        full_pipeline.fit(X_train, y_train)

        # 6. Evaluate on test set
        preds   = full_pipeline.predict(X_test)
        probas  = full_pipeline.predict_proba(X_test)[:, 1]
        acc     = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probas)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        # 7. Log & register the full pipeline
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            registered_model_name="StrokeRF"
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        print(f"Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
