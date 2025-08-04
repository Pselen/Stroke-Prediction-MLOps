# at the top of src/monitor.py
from prefect import flow, task
from mlflow.tracking import MlflowClient
import os, json, pandas as pd, numpy as np
from datetime import datetime, timedelta
from time import perf_counter

# ─── Tasks & Flow (unchanged) ────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")

@task
def fetch_recent_data_metrics(hours: int = 24):
    MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    EXP_NAME    = "StrokePrediction-Monitoring"
    client      = MlflowClient(tracking_uri=MLFLOW_URI)
    exp         = client.get_experiment_by_name(EXP_NAME)
    cutoff_ms   = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
    runs        = client.search_runs([exp.experiment_id],
                                     filter_string=f"start_time >= {cutoff_ms}")
    records = []
    for run in runs:
        rec = {m: run.data.metrics[m] for m in run.data.metrics if m.startswith("data_mean_")}
        records.append(rec)
    return pd.DataFrame(records) if records else pd.DataFrame()

@task
def compute_drift(df_recent: pd.DataFrame):
    with open(os.path.join(BASE_DIR, "data/interim/baseline_stats.json")) as f:
        baseline = json.load(f)
    scores = {}
    for feat, base in baseline.items():
        if feat in df_recent.columns:
            recent_mean = df_recent[feat].mean()
            scores[feat] = abs(recent_mean - base) / base
    overall = float(np.mean(list(scores.values()))) if scores else 0.0
    return overall, scores

@task
def log_drift(overall: float, details: dict):
    import mlflow
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metric("drift_score", overall)
        for feat, score in details.items():
            mlflow.log_metric(f"drift_{feat}", score)

@flow(name="Daily-Drift-Check")
def daily_drift_check():
    # 1) fetch the recent metrics (runs in‐line)
    df_recent = fetch_recent_data_metrics()

    # 2) compute the drift
    overall, details = compute_drift(df_recent)

    # 3) log the drift back to MLflow
    log_drift(overall, details)

if __name__ == "__main__":
    # no in-code deploy block; we'll use `prefect deploy` from the CLI
    daily_drift_check()