# src/train.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
import mlflow.sklearn
import joblib
from data_prep import load_data, prepare_data, build_preprocessor

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
    # 1. Load & preprocess
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = prepare_data(df, test_size, random_state)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 2. Start MLflow run
    mlflow.set_experiment("StrokePrediction")
    with mlflow.start_run():
        # Log hyperparameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        }
        mlflow.log_params(params)


        # 3. Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        model.fit(X_train_proc, y_train)

        # 4. Predict & evaluate
        preds = model.predict(X_test_proc)
        probas = model.predict_proba(X_test_proc)[:, 1]
        acc = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probas)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        # 5. Log preprocessing pipeline & model
        mlflow.sklearn.log_model(preprocessor, "preprocessor")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="StrokeRF"
        )
        run_info = mlflow.active_run().info
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
