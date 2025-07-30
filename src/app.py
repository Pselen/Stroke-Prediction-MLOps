# src/app.py

import os
import time
from time import perf_counter
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Setup ─────────────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME  = os.getenv("MODEL_NAME", "StrokeRF")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_URI)
# Main model registry client
client = MlflowClient()

# Load the production model and preprocessor
mv           = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
preprocessor = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/preprocessor")
model        = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Set up a separate monitoring experiment
mlflow.set_experiment("StrokePrediction-Monitoring")

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
    return {"status": "ok", "model_version": mv.version}

@app.post("/predict")
def predict(features: StrokeFeatures):
    # 1. Prepare data
    data = features.dict()
    df   = pd.DataFrame([data])

    # 2. Preprocess
    X_proc = preprocessor.transform(df)

    # 3. Inference & telemetry
    with mlflow.start_run(nested=True):
        # Timestamp
        mlflow.log_metric("request_ts", time.time())

        # Data-quality stats (mean of each feature)
        for col, val in df.mean().items():
            mlflow.log_metric(f"data_mean_{col}", float(val))

        # Inference latency
        start = perf_counter()
        pred  = model.predict(X_proc)[0]
        prob  = model.predict_proba(X_proc)[0][1]
        latency_ms = (perf_counter() - start) * 1000
        mlflow.log_metric("inference_latency_ms", latency_ms)

        # Predicted probability
        mlflow.log_metric("predicted_prob", prob)

    return {"prediction": int(pred), "probability": float(prob)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
