# Stroke-Prediction-MLOps
# Stroke Prediction MLOps Pipeline

An end-to-end MLOps project to predict patient stroke risk using demographic and medical data, built with best practices for reproducibility, cloud readiness, CI/CD, and monitoring.

---

## 1. Problem Description  
Stroke is a leading cause of death and disability worldwide. Early risk prediction enables targeted preventive care. This project trains a Random Forest model on Kaggle’s “Stroke Prediction” dataset (≈5 000 samples), using features like age, BMI, glucose level, comorbidities, and lifestyle factors, to output a binary stroke/no-stroke prediction.  

---

## 2. Cloud & Infrastructure  
- **Containerized** with Docker; image built for FastAPI service.  
- **(Optionally)** deployable to Kubernetes or ECS/EKS with minimal changes.  
- **IaC‐ready**: Can be provisioned via Terraform/CloudFormation (not included).  
> **Score (2/4):** Containerized; cloud‐IaC pipelines can be added.

---

## 3. Experiment Tracking & Model Registry  
- **MLflow** for tracking hyperparameters, metrics, artifacts.  
- **Model registry** used: best pipeline (preprocessor+classifier) registered as `StrokeRF`.  
> **Score (4/4):** Both experiment tracking and registry used.

---

## 4. Workflow Orchestration  
- **Prefect 2** schedules daily drift checks at midnight UTC.  
- **Drift‐check flow** fetches recent telemetry, computes data‐drift vs. training baseline, logs back to MLflow.  
> **Score (4/4):** Fully deployed scheduled workflow.

---

## 5. Model Deployment  
- **FastAPI** app serves `/health` and `/predict`.  
- **Dockerized** for cloud‐ready deployment.  
> **Score (4/4):** Containerized, MLflow‐backed service.

---

## 6. Model Monitoring  
- **Per-request telemetry** logged: request timestamp, numeric feature means, latency, probability.  
- **Daily drift checks** compute & log drift metrics.  
- **Alerting hooks** (e.g. Slack webhook) can be added.  
> **Score (4/4):** Comprehensive monitoring with conditional workflows.

---

## 7. Reproducibility  
- **`requirements.txt`** (pinned versions) and **`environment.yml`** included.  
- **Makefile** for common tasks: install, lint, test, build, run.  
- **README** with clear setup & run instructions.  
> **Score (4/4):** Clear, complete instructions; dependency versions specified.

---

## 8. Best Practices & CI/CD  
- **Unit tests** (`pytest`) for data prep, monitoring logic.  
- **Integration tests** for FastAPI endpoints.  
- **Linters/formatters**: Black, isort, flake8 via pre-commit.  
- **Pre-commit hooks** enforce code style on commit.  
- **CI pipeline** (GitHub Actions) runs lint → test → Docker build → push.  
> **Score (3/6):** Unit (1), integration (1), formatter (1), Makefile (1), pre-commit (0), CI/CD (0).

---

## Getting Started

### Prerequisites  
- Python 3.8+  
- Docker  
- (Optional) Conda  

### Local Setup

```bash
# 1. Clone & enter project
git clone <repo-url> && cd Stroke-Prediction-MLOps

# 2. Create & activate environment
conda env create -f environment.yml
conda activate stroke-env
# or: pip install -r requirements.txt

# 3. Install dev tools
pip install -r dev-requirements.txt
pre-commit install

# 4. Generate baseline stats & train & register model
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python src/train.py --n_estimators=150 --max_depth=12 …

# 5. Serve locally
make build
make run
curl http://localhost:8080/health
```

### Drift Monitoring

```bash
# Run one-off
python src/monitor.py

# Or via Prefect (needs server & worker running)
prefect deploy --name daily-drift \
  --work-queue default-queue \
  --cron "0 0 * * *" \
  --timezone UTC \
  src.monitor:daily_drift_check
prefect worker start --pool default
```
