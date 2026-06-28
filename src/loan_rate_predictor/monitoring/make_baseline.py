"""Create Model Monitor data-quality baseline from 2021 training features.

This is the managed AWS data-quality baseline (per-feature statistics and
constraints).
"""
import argparse
import tempfile
from pathlib import Path

import boto3
import pandas as pd
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

from loan_rate_predictor import config


def _session():
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    return sagemaker.Session(boto_session=boto_session)


def _prepare_baseline_csv(s3_client, tmp_dir: Path) -> str:
    """Download processed.csv, filter to 2021, keep features + target with headers, upload to S3."""
    local = tmp_dir / "processed.csv"
    s3_client.download_file(config.S3_BUCKET, f"{config.S3_PROCESSED_PREFIX}/processed.csv", str(local))

    df = pd.read_csv(local, low_memory=False)
    df = df[df["activity_year"] == config.TRAIN_YEAR]

    cols = [config.TARGET] + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    cols = [c for c in cols if c in df.columns]
    baseline_df = df[cols]

    baseline_path = tmp_dir / "baseline_input.csv"
    baseline_df.to_csv(baseline_path, index=False)

    s3_key = f"{config.S3_BASELINE_PREFIX}/input/baseline_input.csv"
    s3_client.upload_file(str(baseline_path), config.S3_BUCKET, s3_key)
    print(f"Uploaded baseline input ({len(baseline_df):,} rows) → s3://{config.S3_BUCKET}/{s3_key}")
    return f"s3://{config.S3_BUCKET}/{s3_key}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", type=str, default=config.SAGEMAKER_ROLE_ARN)
    parser.add_argument("--wait", action="store_true", default=True)
    args = parser.parse_args()

    if not args.role_arn:
        raise ValueError("Set SAGEMAKER_ROLE_ARN env var or pass --role-arn")

    sm_session = _session()
    s3_client = sm_session.boto_session.client("s3")

    with tempfile.TemporaryDirectory() as tmp:
        baseline_uri = _prepare_baseline_csv(s3_client, Path(tmp))

    output_uri = f"s3://{config.S3_BUCKET}/{config.S3_BASELINE_PREFIX}/output"

    monitor = DefaultModelMonitor(
        role=args.role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sm_session,
    )

    print(f"Starting baseline job...")
    print(f"  Input:  {baseline_uri}")
    print(f"  Output: {output_uri}")

    monitor.suggest_baseline(
        baseline_dataset=baseline_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=output_uri,
        wait=args.wait,
    )

    print(f"Baseline complete. Statistics and constraints at: {output_uri}")


if __name__ == "__main__":
    main()
