import pytest
from fastapi.testclient import TestClient


# Dummy pipeline for monkeypatching
class DummyPipeline:
    def predict(self, df):
        return [0.42] * len(df)

    def predict_proba(self, df):
        return [[0.58, 0.42] for _ in range(len(df))]


@pytest.fixture(autouse=True)
def patch_mlflow(monkeypatch):
    # Patch MlflowClient.get_latest_versions
    class DummyClient:
        def get_latest_versions(self, name, stages):
            from types import SimpleNamespace

            return [SimpleNamespace(run_id="dummy", version=5)]

    monkeypatch.setattr("src.app.MlflowClient", lambda *args, **kwargs: DummyClient())
    # Patch pyfunc.load_model to return our dummy pipeline
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: DummyPipeline())
    # Patch sklearn.load_model for compatibility
    monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: DummyPipeline())


from src.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_version"], int)


def test_predict():
    payload = {
        "age": 67.0,
        "gender": "Male",
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 27.3,
        "smoking_status": "formerly smoked",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    out = resp.json()
    assert out["prediction"] in (0, 1)
    assert pytest.approx(out["probability"], rel=1e-2) == 0.42
    assert out["latency_ms"] >= 0
