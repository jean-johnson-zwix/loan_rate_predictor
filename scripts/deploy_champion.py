"""Promote the latest model and deploy it to the endpoint.

Flow:
  1. Find latest PendingManualApproval in SageMaker Registry
  2. Register it in MLflow (with metrics from SageMaker metadata)
  3. Set MLflow "champion" alias on the new version
  4. Update SageMaker approval status (backward compat)
  5. Update terraform.auto.tfvars with model_package_arn + trained_on
  6. Print next step (make tf-apply)

Usage:
    python scripts/deploy_champion.py [--dry-run]
"""
import argparse
import os
import re
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loan_rate_predictor import config
from loan_rate_predictor.registry import (
    get_latest_pending_sagemaker_arn,
    promote_champion,
    register_in_mlflow,
    resolve_champion,
)

TFVARS = os.path.join(os.path.dirname(__file__), "..", "infra", "terraform.auto.tfvars")


def _read_tfvars(path: str) -> str:
    with open(path) as f:
        return f.read()


def _set_var(content: str, key: str, value: str) -> str:
    pattern = rf'^{re.escape(key)}\s*=\s*"[^"]*"'
    replacement = f'{key}="{value}"'
    updated, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if n == 0:
        updated = content.rstrip("\n") + f'\n{replacement}\n'
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session = boto3.Session(profile_name=config.AWS_PROFILE)
    sm = session.client("sagemaker", region_name=config.AWS_REGION)

    # 1. Find latest pending model
    pending_arn = get_latest_pending_sagemaker_arn(sm)
    if not pending_arn:
        # No pending model; use current champion
        result = resolve_champion(sm)
        if result is None:
            print("No pending or approved model found.")
            sys.exit(1)
        champion_arn = result[0]
        print(f"No pending model. Current champion: {champion_arn}")
    else:
        champion_arn = pending_arn
        print(f"Found pending model: {champion_arn}")

    # 2. Read metadata from SageMaker
    desc = sm.describe_model_package(ModelPackageName=champion_arn)
    meta = desc.get("CustomerMetadataProperties", {})
    trained_on = meta.get("trained_on")
    if not trained_on:
        print(f"Model {champion_arn} has no 'trained_on' metadata. Tag it manually.")
        sys.exit(1)

    # 3. Register in MLflow + set alias
    mlflow_version = None
    if config.MLFLOW_TRACKING_ARN:
        metrics = {}
        params = {"data_year": trained_on}
        for k in ("challenger_mae", "challenger_rmse"):
            v = meta.get(k)
            if v:
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
        for k in ("train_rows", "val_rows", "num_features"):
            v = meta.get(k)
            if v:
                params[k] = v
        params["objective"] = meta.get("objective", config.XGBOOST_OBJECTIVE)
        params["group_split_key"] = meta.get("group_split_key", config.GROUP_SPLIT_KEY)

        mlflow_version = register_in_mlflow(metrics, params, champion_arn)

    # 4. Promote (MLflow alias + SageMaker status)
    promote_champion(sm, champion_arn, mlflow_version)

    # 5. Update tfvars
    content = _read_tfvars(TFVARS)
    updated = _set_var(content, "model_package_arn", champion_arn)
    updated = _set_var(updated, "trained_on", trained_on)

    print(f"\nChampion ARN : {champion_arn}")
    print(f"trained_on   : {trained_on}")
    if mlflow_version:
        print(f"MLflow ver   : {mlflow_version}")

    if args.dry_run:
        print("\n--- terraform.auto.tfvars (preview) ---")
        print(updated)
        return

    with open(TFVARS, "w") as f:
        f.write(updated)
    print(f"\nUpdated {TFVARS}")
    print("Run `make tf-apply` to deploy.")


if __name__ == "__main__":
    main()
