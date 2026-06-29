"""Generate drift and model quality reports using Evidently.

Replaces SageMaker Model Monitor analyzer jobs with local Evidently reports.
Runs in seconds instead of 10-minute Processing jobs.

Two reports per vintage:
  A: Data drift — feature distribution shift (reference=2021, current=year)
  B: Model quality — regression metrics vs baseline thresholds

Outputs:
  - JSON report -> S3 (machine-readable, dashboard consumes this)
  - HTML report -> S3 (human-readable, viewable in browser)

Usage:
    PYTHONPATH=src python -m loan_rate_predictor.monitoring.drift_report --year 2022
"""
import argparse
import json
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, RegressionPreset

from loan_rate_predictor import config
from loan_rate_predictor.training.prepare import load_data_year, feature_columns

CW_NAMESPACE = "LoanRatePredictor/Monitoring"


def _session():
    return boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)


def _load_year_data(s3, year):
    """Download processed.csv and load a single year."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        s3.download_file(
            config.S3_BUCKET,
            f"{config.S3_PROCESSED_PREFIX}/processed.csv",
            str(tmp_path / "processed.csv"),
        )
        return load_data_year(tmp_path, year)


def _load_predictions(s3, year):
    """Download merged predictions+labels for a year."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/merged/merged.csv"
        local = tmp_path / "merged.csv"
        s3.download_file(config.S3_BUCKET, key, str(local))
        return pd.read_csv(local)


def run_data_drift(reference_df, current_df, year, s3):
    """Monitor A: feature distribution drift."""
    features = feature_columns()
    ref = reference_df[features].copy()
    cur = current_df[features].copy()

    report = Report([DataDriftPreset()], include_tests=True)
    snapshot = report.run(cur, ref)

    # Save HTML + JSON to S3
    prefix = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/data_quality"

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        json_path = Path(tmp) / "report.json"
        snapshot.save_html(str(html_path))
        snapshot.save_json(str(json_path))

        s3.upload_file(str(html_path), config.S3_BUCKET, f"{prefix}/report.html")
        s3.upload_file(str(json_path), config.S3_BUCKET, f"{prefix}/report.json")

    # Extract drift summary for CloudWatch + console
    report_dict = snapshot.dict()
    tests = report_dict.get("tests", [])
    failed_tests = [t for t in tests if t.get("status") == "FAIL"]

    print(f"[A] Data drift: {len(failed_tests)} failed tests out of {len(tests)}")
    for t in failed_tests:
        print(f"    {t.get('name', '?')}: {t.get('description', '')[:100]}")

    # Publish to CloudWatch
    session = _session()
    cw = session.client("cloudwatch")
    cw.put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[
            {"MetricName": "DataQualityViolations", "Dimensions": [{"Name": "Year", "Value": str(year)}],
             "Value": float(len(failed_tests)), "Unit": "Count"},
            {"MetricName": "DataQualityViolations", "Value": float(len(failed_tests)), "Unit": "Count"},
        ],
    )

    print(f"  HTML report -> s3://{config.S3_BUCKET}/{prefix}/report.html")
    print(f"  JSON report -> s3://{config.S3_BUCKET}/{prefix}/report.json")
    return len(failed_tests)


def run_model_quality(merged_df, year, s3):
    """Monitor B: regression quality metrics."""
    predictions = merged_df["prediction"].values
    ground_truth = merged_df["ground_truth"].values

    mae = float(np.mean(np.abs(ground_truth - predictions)))
    rmse = float(np.sqrt(np.mean((ground_truth - predictions) ** 2)))
    r2 = float(1 - np.sum((ground_truth - predictions) ** 2) / np.sum((ground_truth - ground_truth.mean()) ** 2))

    # Evidently regression report
    from evidently.core.datasets import Dataset, DataDefinition, Regression
    eval_df = pd.DataFrame({"target": ground_truth, "prediction": predictions})
    data_def = DataDefinition(regression=[Regression(target="target", prediction="prediction")])
    ds = Dataset.from_pandas(eval_df, data_definition=data_def)
    report = Report([RegressionPreset()], include_tests=True)
    snapshot = report.run(ds, ds)

    prefix = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/model_quality"

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        json_path = Path(tmp) / "report.json"
        snapshot.save_html(str(html_path))
        snapshot.save_json(str(json_path))
        s3.upload_file(str(html_path), config.S3_BUCKET, f"{prefix}/report.html")
        s3.upload_file(str(json_path), config.S3_BUCKET, f"{prefix}/report.json")

    # Check against baseline thresholds
    baseline_mae_threshold = 0.248 * (1 + config.MODEL_QUALITY_DEGRADATION_THRESHOLD)
    violations = 1 if mae > baseline_mae_threshold else 0

    # Publish to CloudWatch
    session = _session()
    cw = session.client("cloudwatch")
    cw.put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[
            {"MetricName": "ModelQualityViolations", "Dimensions": [{"Name": "Year", "Value": str(year)}],
             "Value": float(violations), "Unit": "Count"},
            {"MetricName": "ModelQualityViolations", "Value": float(violations), "Unit": "Count"},
        ],
    )

    breach = mae > baseline_mae_threshold
    print(f"[B] Model quality: MAE {mae:.4f}  RMSE {rmse:.4f}  R2 {r2:.4f}")
    print(f"    Threshold: {baseline_mae_threshold:.4f}  Breach: {breach}")
    print(f"  HTML report -> s3://{config.S3_BUCKET}/{prefix}/report.html")
    return violations, mae


def log_to_mlflow(year, champion_version, mae, dq_violations, mq_violations):
    """Log monitoring results to MLflow (no-op if not configured)."""
    try:
        from loan_rate_predictor.tracking import log_monitoring
        log_monitoring(year, champion_version, mae, dq_violations, mq_violations)
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


def _resolve_reference_year(s3, year, sm):
    """Get the training year of the champion that scored this year.

    Reads from meta.json (written by run_monitoring.py at monitoring time).
    Falls back to looking up the champion version in the registry.
    """
    # Try meta.json first (records which champion was active when monitoring ran)
    meta = _s3_json_safe(s3, f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/meta.json")
    if meta:
        arn = meta.get("champion_arn")
        if arn:
            try:
                desc = sm.describe_model_package(ModelPackageName=arn)
                trained_on = desc.get("CustomerMetadataProperties", {}).get("trained_on")
                if trained_on:
                    return int(trained_on)
            except Exception:
                pass

    # Fallback: current champion's training year
    from loan_rate_predictor.registry import resolve_champion
    result = resolve_champion(sm)
    if result:
        arn = result[0]
        desc = sm.describe_model_package(ModelPackageName=arn)
        trained_on = desc.get("CustomerMetadataProperties", {}).get("trained_on")
        if trained_on:
            return int(trained_on)
    return config.TRAIN_YEAR


def _s3_json_safe(s3, key):
    try:
        resp = s3.get_object(Bucket=config.S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    parser.add_argument("--monitor", choices=["data-drift", "model-quality", "both"], default="both")
    args = parser.parse_args()

    session = _session()
    s3 = session.client("s3")
    sm = session.client("sagemaker")

    ref_year = _resolve_reference_year(s3, args.year, sm)

    if args.monitor in ("data-drift", "both"):
        print(f"\nLoading reference ({ref_year}) and current ({args.year}) data...")
        ref_df = _load_year_data(s3, ref_year)
        cur_df = _load_year_data(s3, args.year)
        print(f"  Reference: {len(ref_df):,} rows, Current: {len(cur_df):,} rows")
        run_data_drift(ref_df, cur_df, args.year, s3)

    if args.monitor in ("model-quality", "both"):
        print(f"\nLoading predictions for {args.year}...")
        merged = _load_predictions(s3, args.year)
        print(f"  {len(merged):,} prediction-label pairs")
        run_model_quality(merged, args.year, s3)


if __name__ == "__main__":
    main()
