"""Score a year with the frozen champion via SageMaker batch transform.

Prepares two S3 artifacts per year:
  - features-only CSV (no header) -> batch transform input -> predictions
  - features+target CSV (with headers) -> data-quality monitor input (matches baseline schema)

Column order/count assertion guards against schema mismatch masquerading as drift.

Usage:
    PYTHONPATH=src python scripts/run_batch_transform.py --year 2022
"""
import argparse
import tempfile
import time
from pathlib import Path

import boto3
import pandas as pd
import sagemaker
from sagemaker.transformer import Transformer

from loan_rate_predictor import config
from loan_rate_predictor.training.prepare import load_data_year, feature_columns


TRANSFORM_MODEL_NAME = "loan-rate-predictor-champion"


def _session():
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    return sagemaker.Session(boto_session=boto_session)


def _prepare_year_data(s3_client, year: int, tmp_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """Download processed.csv, filter to year, assert schema, upload scoring + monitor inputs."""
    local_csv = tmp_dir / "processed.csv"
    s3_client.download_file(
        config.S3_BUCKET,
        f"{config.S3_PROCESSED_PREFIX}/processed.csv",
        str(local_csv),
    )

    df = load_data_year(tmp_dir, year)
    features = feature_columns()
    expected_cols = len(features)
    actual_cols = len([c for c in features if c in df.columns])
    if actual_cols != expected_cols:
        raise ValueError(
            f"Schema mismatch: expected {expected_cols} features, "
            f"found {actual_cols} in year {year}. "
            f"Missing: {set(features) - set(df.columns)}"
        )
    print(f"Year {year}: {len(df):,} rows, {actual_cols} features — schema OK")

    # Stable sort before writing — pins row order for the ID-based join in join_predictions_labels
    df = df.sort_values(by=config.RECORD_ID).reset_index(drop=True)

    # Scoring input: features only, no header (XGBoost format)
    scoring_path = tmp_dir / "scoring_input.csv"
    df[features].to_csv(scoring_path, index=False, header=False)
    scoring_key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/input/scoring_input.csv"
    s3_client.upload_file(str(scoring_path), config.S3_BUCKET, scoring_key)
    scoring_uri = f"s3://{config.S3_BUCKET}/{scoring_key}"
    print(f"  Scoring input -> {scoring_uri}")

    # Row-ID sidecar: record_id in same row order as scoring_input.csv
    ids_path = tmp_dir / "scoring_ids.csv"
    df[[config.RECORD_ID]].to_csv(ids_path, index=False)
    ids_key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/input/scoring_ids.csv"
    s3_client.upload_file(str(ids_path), config.S3_BUCKET, ids_key)
    print(f"  Row IDs -> s3://{config.S3_BUCKET}/{ids_key}")

    # Monitor input: target + features with headers (matches data-quality baseline schema)
    monitor_path = tmp_dir / "monitor_input.csv"
    monitor_cols = [config.TARGET] + features
    df[monitor_cols].to_csv(monitor_path, index=False)
    monitor_key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitor/features.csv"
    s3_client.upload_file(str(monitor_path), config.S3_BUCKET, monitor_key)
    monitor_uri = f"s3://{config.S3_BUCKET}/{monitor_key}"
    print(f"  Monitor input -> {monitor_uri}")

    return df, scoring_uri, monitor_uri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    parser.add_argument("--wait", action="store_true", default=True)
    args = parser.parse_args()

    sm_session = _session()
    s3_client = sm_session.boto_session.client("s3")

    with tempfile.TemporaryDirectory() as tmp:
        df, scoring_uri, monitor_uri = _prepare_year_data(s3_client, args.year, Path(tmp))

    output_uri = f"s3://{config.S3_BUCKET}/{config.S3_PREDICTIONS_PREFIX}/{args.year}/output"
    job_name = f"loan-rate-predictor-score-{args.year}-{int(time.time())}"

    transformer = Transformer(
        model_name=TRANSFORM_MODEL_NAME,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=output_uri,
        sagemaker_session=sm_session,
    )

    print(f"Starting batch transform: {job_name}")
    print(f"  Input:  {scoring_uri}")
    print(f"  Output: {output_uri}")

    transformer.transform(
        data=scoring_uri,
        content_type="text/csv",
        split_type="Line",
        job_name=job_name,
        wait=args.wait,
        logs=args.wait,
    )

    print(f"Batch transform complete. Predictions at: {output_uri}")
    print(f"  Monitor features at: s3://{config.S3_BUCKET}/{config.S3_PREDICTIONS_PREFIX}/{args.year}/monitor/")


if __name__ == "__main__":
    main()
