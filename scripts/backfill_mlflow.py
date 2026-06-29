"""One-time backfill: register existing SageMaker model packages in MLflow.

Reads all model packages from SageMaker Registry, registers each in MLflow
with its metadata, and sets the "champion" alias on the current Approved model.

Usage:
    PYTHONPATH=src python scripts/backfill_mlflow.py
"""
import os
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loan_rate_predictor import config
from loan_rate_predictor.registry import register_in_mlflow, _mlflow_client

MODEL_NAME = config.MODEL_PACKAGE_GROUP_NAME


def main():
    if not config.MLFLOW_TRACKING_ARN:
        print("MLFLOW_TRACKING_ARN not set. Set it in .env first.")
        sys.exit(1)

    session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    sm = session.client("sagemaker")

    # Get all model packages
    packages = []
    for status in ("Approved", "Rejected", "PendingManualApproval"):
        resp = sm.list_model_packages(
            ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus=status,
            SortBy="CreationTime",
            SortOrder="Ascending",
        )
        for pkg in resp.get("ModelPackageSummaryList", []):
            desc = sm.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
            packages.append({
                "arn": pkg["ModelPackageArn"],
                "version": desc.get("ModelPackageVersion"),
                "status": pkg["ModelApprovalStatus"],
                "meta": desc.get("CustomerMetadataProperties", {}),
            })

    packages.sort(key=lambda p: p["version"])
    print(f"Found {len(packages)} model packages in SageMaker Registry.\n")

    champion_mlflow_version = None

    for pkg in packages:
        meta = pkg["meta"]
        arn = pkg["arn"]

        metrics = {}
        for k in ("challenger_mae", "challenger_rmse"):
            v = meta.get(k)
            if v:
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass

        params = {
            "data_year": meta.get("trained_on", "?"),
            "objective": meta.get("objective", "?"),
            "group_split_key": meta.get("group_split_key", "?"),
        }
        for k in ("train_rows", "val_rows", "num_features"):
            v = meta.get(k)
            if v:
                params[k] = v

        mlflow_version = register_in_mlflow(metrics, params, arn)
        print(f"  v{pkg['version']} ({pkg['status']}) -> MLflow v{mlflow_version}")

        if pkg["status"] == "Approved":
            champion_mlflow_version = mlflow_version

    # Set champion alias
    if champion_mlflow_version:
        client = _mlflow_client()
        client.set_registered_model_alias(MODEL_NAME, "champion", champion_mlflow_version)
        print(f"\nMLflow alias 'champion' -> v{champion_mlflow_version}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
