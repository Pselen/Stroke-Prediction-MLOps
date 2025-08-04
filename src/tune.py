# src/tune.py

import os

import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from data_prep import build_preprocessor, load_data, prepare_data

# 1️⃣ Point MLflow at your tracking server (same as in train.py)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("StrokePrediction-Optuna")

# 2️⃣ Load & split once
df = load_data("data/raw/healthcare-dataset-stroke-data.csv")
X_train, X_test, y_train, y_test = prepare_data(df)
preprocessor = build_preprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)


def objective(trial):
    # 3️⃣ Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
    }

    # 4️⃣ Log each trial as a nested MLflow run
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train_proc, y_train)

        # Evaluate
        probs = model.predict_proba(X_test_proc)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        mlflow.log_metric("roc_auc", roc_auc)

        return roc_auc


if __name__ == "__main__":
    # 5️⃣ Create & run the Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 6️⃣ Print & save best hyperparameters
    best_params = study.best_trial.params
    print("Best ROC AUC:", study.best_value)
    print("Best hyperparameters:", best_params)

    # 7️⃣ (Optional) Retrain final model and register it
    #    You can reuse train.py or add code here:
    #
    # from train import train_and_register
    # train_and_register(**best_params)
