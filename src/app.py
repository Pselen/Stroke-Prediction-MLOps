import os
import time
from time import perf_counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# ─── Configuration ────────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME  = os.getenv("MODEL_NAME", "StrokeRF")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MONITOR_EXP = "StrokePrediction-Monitoring"

# ─── MLflow client & version check ───────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
if not versions:
    raise RuntimeError(f"No '{MODEL_STAGE}' versions of '{MODEL_NAME}' found")
mv = versions[0]

# We'll load the pipeline on first request
pipeline = None

# Switch to the monitoring experiment for telemetry logs
mlflow.set_experiment(MONITOR_EXP)

# ─── FastAPI setup ───────────────────────────────────────────
app = FastAPI()

class StrokeFeatures(BaseModel):
    age: float
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/health")
def health():
    # Quick check: service is up and we know which model version we'd serve
    return {"status": "ok", "model_version": mv.version}

@app.post("/predict")
def predict(features: StrokeFeatures):
    global pipeline

    # Lazy‐load the pipeline
    if pipeline is None:
        pipeline = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    # Build DataFrame
    df = pd.DataFrame([features.dict()])

    # Inference + Telemetry
    with mlflow.start_run(nested=True):
        # 1️⃣ Timestamp
        mlflow.log_metric("request_ts", time.time())

        # 2️⃣ Numeric feature means only
        numeric_means = df.select_dtypes(include="number").mean().to_dict()
        for col, val in numeric_means.items():
            mlflow.log_metric(f"data_mean_{col}", float(val))

        # 3️⃣ Predict & measure latency
        start = perf_counter()
        prob  = pipeline.predict(df).squeeze()
        latency_ms = (perf_counter() - start) * 1000
        mlflow.log_metric("inference_latency_ms", latency_ms)

        # 4️⃣ Predicted probability
        mlflow.log_metric("predicted_prob", float(prob))

    # Threshold for class
    pred = int(prob >= 0.5)
    return {
        "prediction": pred,
        "probability": float(prob),
        "latency_ms": latency_ms
    }