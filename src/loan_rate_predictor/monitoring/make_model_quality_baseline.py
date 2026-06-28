"""Create Model Monitor model-quality baseline from 2021 champion predictions"""
import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

from loan_rate_predictor import config
from loan_rate_predictor.training.prepare import load_data_year, split_group_aware, feature_columns


def _session():
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    return sagemaker.Session(boto_session=boto_session)


def _resolve_champion_artifact(sm_client) -> str:
    """Get the model artifact S3 URI from the latest approved model package."""
    packages = sm_client.list_model_packages(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not packages["ModelPackageSummaryList"]:
        raise RuntimeError("No approved model package found in registry")
    arn = packages["ModelPackageSummaryList"][0]["ModelPackageArn"]
    desc = sm_client.describe_model_package(ModelPackageName=arn)
    uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    print(f"Champion model: {arn}")
    print(f"Artifact: {uri}")
    return uri


def _download_s3(s3_client, uri: str, local_path: Path) -> None:
    parsed = urlparse(uri)
    s3_client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(local_path))


def _score_champion(model_tar: Path, X_val: np.ndarray, work_dir: Path) -> np.ndarray:
    """Extract champion model and score the val features."""
    # ponytail: inline xgboost import — only needed here, not a module dep
    import xgboost as xgb
    extract_dir = work_dir / "model"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar) as tar:
        tar.extractall(extract_dir)
    model = xgb.Booster()
    model.load_model(str(extract_dir / "xgboost-model"))
    return model.predict(xgb.DMatrix(X_val))


def _override_error_thresholds(s3_client, output_uri: str,
                                baseline_mae: float, baseline_rmse: float) -> None:
    """Download constraints.json, set MAE/RMSE thresholds to baseline × (1 + degradation), re-upload.

    Raises if either metric can't be found — a wrong-but-silent baseline is worse
    than a failed build.
    """
    parsed = urlparse(output_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/").rstrip("/") + "/constraints.json"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    s3_client.download_file(bucket, key, tmp_path)
    with open(tmp_path) as f:
        constraints = json.load(f)

    multiplier = 1 + config.MODEL_QUALITY_DEGRADATION_THRESHOLD
    targets = {
        "mae": baseline_mae * multiplier,
        "rmse": baseline_rmse * multiplier,
    }
    found = set()

    reg = constraints.get("regression_metrics", {})

    # Shape 1: list of dicts with "name" key
    if isinstance(reg, list):
        for metric in reg:
            name = metric.get("name", "")
            if name in targets:
                metric["threshold"] = targets[name]
                metric["comparison_operator"] = "GreaterThanThreshold"
                found.add(name)
    # Shape 2: dict keyed by metric name
    elif isinstance(reg, dict):
        for name, thresh in targets.items():
            if name in reg:
                reg[name]["threshold"] = thresh
                reg[name]["comparison_operator"] = "GreaterThanThreshold"
                found.add(name)

    missing = set(targets) - found
    if missing:
        raise RuntimeError(
            f"Could not find {missing} in constraints.json to override thresholds. "
            f"Inspect {key} and fix the parser for the actual schema shape."
        )

    with open(tmp_path, "w") as f:
        json.dump(constraints, f, indent=2)
    s3_client.upload_file(tmp_path, bucket, key)
    for name, thresh in targets.items():
        baseline = baseline_mae if name == "mae" else baseline_rmse
        print(f"Updated {name.upper()} threshold: {baseline:.4f} → {thresh:.4f} (×{multiplier})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", type=str, default=config.SAGEMAKER_ROLE_ARN)
    parser.add_argument("--wait", action="store_true", default=True)
    args = parser.parse_args()

    if not args.role_arn:
        raise ValueError("Set SAGEMAKER_ROLE_ARN env var or pass --role-arn")

    sm_session = _session()
    boto_session = sm_session.boto_session
    s3_client = boto_session.client("s3")
    sm_client = boto_session.client("sagemaker")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Download processed.csv, get 2021 val split
        local_csv = tmp_dir / "processed.csv"
        s3_client.download_file(
            config.S3_BUCKET,
            f"{config.S3_PROCESSED_PREFIX}/processed.csv",
            str(local_csv),
        )
        df = load_data_year(tmp_dir, config.TRAIN_YEAR)
        _, val_df = split_group_aware(df)
        features = feature_columns()
        X_val = val_df[features].values.astype(float)
        y_val = val_df[config.TARGET].values.astype(float)
        print(f"2021 val set: {len(val_df):,} rows")

        # 2. Download champion model
        artifact_uri = _resolve_champion_artifact(sm_client)
        model_tar = tmp_dir / "model.tar.gz"
        _download_s3(s3_client, artifact_uri, model_tar)

        # 3. Score
        preds = _score_champion(model_tar, X_val, tmp_dir)

        # 4. Sanity check
        mae = float(np.mean(np.abs(y_val - preds)))
        rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        print(f"Sanity check — MAE: {mae:.4f} (expect ~0.248), RMSE: {rmse:.4f} (expect ~0.339)")

        # 5. Write predictions + ground truth CSV
        baseline_df = pd.DataFrame({"prediction": preds, "ground_truth": y_val})
        baseline_path = tmp_dir / "model_quality_baseline.csv"
        baseline_df.to_csv(baseline_path, index=False)

        s3_key = f"{config.S3_BASELINE_PREFIX}/model_quality/input/model_quality_baseline.csv"
        s3_client.upload_file(str(baseline_path), config.S3_BUCKET, s3_key)
        baseline_uri = f"s3://{config.S3_BUCKET}/{s3_key}"
        print(f"Uploaded baseline dataset ({len(baseline_df):,} rows) → {baseline_uri}")

    # 6. Run suggest_baseline
    output_uri = f"s3://{config.S3_BUCKET}/{config.S3_BASELINE_PREFIX}/model_quality/output"

    monitor = ModelQualityMonitor(
        role=args.role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sm_session,
    )

    print("Starting model-quality baseline job...")
    print(f"  Input:  {baseline_uri}")
    print(f"  Output: {output_uri}")

    monitor.suggest_baseline(
        baseline_dataset=baseline_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=output_uri,
        problem_type="Regression",
        inference_attribute="prediction",
        ground_truth_attribute="ground_truth",
        wait=args.wait,
    )

    print(f"Baseline complete. Statistics and constraints at: {output_uri}")

    # 7. Override MAE/RMSE thresholds to baseline × (1 + degradation_threshold)
    s3_client_fresh = boto3.Session(
        profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION
    ).client("s3")
    _override_error_thresholds(s3_client_fresh, output_uri, mae, rmse)


if __name__ == "__main__":
    main()
