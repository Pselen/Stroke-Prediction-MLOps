# src/promote_model.py

import os
import argparse
from mlflow.tracking import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote an MLflow registered model version to a specified stage"
    )
    parser.add_argument(
        "--name", type=str, default="StrokeRF",
        help="Registered model name"
    )
    parser.add_argument(
        "--version", type=int, required=True,
        help="Model version number to promote"
    )
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["None", "Staging", "Production", "Archived"],
        help="Target stage"
    )
    parser.add_argument(
        "--archive-existing", action="store_true",
        help="Archive existing versions in the target stage"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # 1. Configure MLflow tracking URI
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"
    )
    client = MlflowClient(tracking_uri)

    # 2. Transition the model version
    client.transition_model_version_stage(
        name=args.name,
        version=args.version,
        stage=args.stage,
        archive_existing_versions=args.archive_existing
    )
    print(
        f"Model '{args.name}' version {args.version} transitioned to '{args.stage}'"
    )


if __name__ == "__main__":
    main()
