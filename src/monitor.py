# src/monitor.py

import os
import json
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.metrics import mean_squared_error

# ── Config ─────────────────────────────────────────────────
MLFLOW_URI         = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MONITOR_EXP_NAME   = "StrokePrediction-Monitoring"
BASELINE_STATS_PATH= "data/interim/baseline_stats.json"  # created in Phase 2

client     = MlflowClient(MLFLOW_URI)
exp        = client.get_experiment_by_name(MONITOR_EXP_NAME)

@task
def fetch_recent_data_metrics(hours=24):
    """Retrieve all 'data_mean_*' metrics from the last `hours` hours."""
    cutoff_ms = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"start_time >= {cutoff_ms}"
    )
    records = []
    for run in runs:
        rec = {m: run.data.metrics[m] for m in run.data.metrics if m.startswith("data_mean_")}
        records.append(rec)
    if records:
        return pd.DataFrame(records)
    else:
        return pd.DataFrame()

@task
def compute_drift(df_recent: pd.DataFrame):
    """Compute average relative difference per feature vs. baseline."""
    with open(BASELINE_STATS_PATH) as f:
        baseline = json.load(f)
    scores = {}
    for feat, base_val in baseline.items():
        if feat in df_recent.columns:
            recent_mean = df_recent[feat].mean()
            scores[feat] = abs(recent_mean - base_val) / base_val
    overall_drift = float(np.mean(list(scores.values()))) if scores else 0.0
    return overall_drift, scores

@task
def log_drift(overall_drift: float, feature_drifts: dict):
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metric("drift_score", overall_drift)
        for feat, score in feature_drifts.items():
            mlflow.log_metric(f"drift_{feat}", score)

@flow(name="Daily-Drift-Check")
def daily_drift_check():
    df_recent         = fetch_recent_data_metrics()
    overall, details = compute_drift(df_recent)
    log_drift(overall, details)

if __name__ == "__main__":
    daily_drift_check()
