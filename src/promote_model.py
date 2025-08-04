# src/promote_model.py

import argparse
import os
import sys

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote an MLflow registered model version to a specified stage"
    )
    parser.add_argument(
        "--name", "-n", type=str, default="StrokeRF", help="Registered model name"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version", "-v", type=int, help="Specific model version number to promote"
    )
    group.add_argument(
        "--latest",
        "-l",
        action="store_true",
        help="Automatically pick the highest-available version to promote",
    )
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        default="Staging",
        choices=["Staging", "Production", "Archived"],
        help="Target stage (default: Staging)",
    )
    parser.add_argument(
        "--archive-existing",
        "-a",
        action="store_true",
        help="Archive existing versions in the target stage",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configure MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    client = MlflowClient(tracking_uri=tracking_uri)

    # 2. Fetch all versions for this model
    try:
        all_versions = client.get_latest_versions(args.name, stages=[])
    except MlflowException as e:
        print(
            f"❌ Failed to fetch versions for model '{args.name}': {e}", file=sys.stderr
        )
        sys.exit(1)

    if not all_versions:
        print(f"❌ No versions found for model '{args.name}'", file=sys.stderr)
        sys.exit(1)

    # 3. Determine which version to promote
    if args.latest:
        # pick the highest numeric version
        version = max(int(v.version) for v in all_versions)
    else:
        version = args.version
        if version not in [int(v.version) for v in all_versions]:
            print(
                f"❌ Specified version {version} not found for model '{args.name}'",
                file=sys.stderr,
            )
            sys.exit(1)

    # 4. Transition the version
    try:
        client.transition_model_version_stage(
            name=args.name,
            version=version,
            stage=args.stage,
            archive_existing_versions=args.archive_existing,
        )
    except MlflowException as e:
        print(f"❌ Failed promoting version {version}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"✅ Model '{args.name}' version {version} transitioned to '{args.stage}' "
        f"{'(archived others)' if args.archive_existing else ''}"
    )


if __name__ == "__main__":
    main()
