import json
from datetime import datetime

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.monitor import compute_drift, fetch_recent_data_metrics, log_drift


class DummyRun:
    def __init__(self, metrics):
        from types import SimpleNamespace

        self.data = SimpleNamespace(metrics=metrics)
        self.info = SimpleNamespace(
            start_time=int(datetime.utcnow().timestamp() * 1000)
        )


@pytest.fixture(autouse=True)
def patch_mlflow_client(tmp_path, monkeypatch):
    # Write baseline stats
    base_dir = tmp_path / "data" / "interim"
    base_dir.mkdir(parents=True)
    baseline = {"age": 50.0, "avg_glucose_level": 110.0, "bmi": 25.0}
    (base_dir / "baseline_stats.json").write_text(json.dumps(baseline))

    # Patch MlflowClient to return dummy experiment and runs
    dummy_runs = [DummyRun({"data_mean_age": 55.0})]

    def dummy_get_exp(name):
        return {"experiment_id": "exp1"}

    def dummy_search_runs(exp_ids, filter_string):
        return dummy_runs

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    monkeypatch.setattr(MlflowClient, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(MlflowClient, "get_experiment_by_name", dummy_get_exp)
    monkeypatch.setattr(MlflowClient, "search_runs", dummy_search_runs)
    # No-op for mlflow.log_metric
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kw: None)


def test_compute_drift(tmp_path, monkeypatch):
    # Given a df with mean age 55 and baseline 50 → drift 0.1
    df = pd.DataFrame([{"data_mean_age": 55.0}])
    overall, details = compute_drift(df)
    assert pytest.approx(overall) == 0.1
    assert pytest.approx(details["age"]) == 0.1


def test_fetch_and_log(tmp_path):
    # fetch_recent_data_metrics should return a DataFrame
    df = fetch_recent_data_metrics(hours=1)
    assert "data_mean_age" in df.columns

    # compute_drift uses that df without error
    overall, details = compute_drift(df)
    # log_drift should run without exception
    log_drift(overall, details)
