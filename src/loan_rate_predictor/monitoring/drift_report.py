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
    eval_df = pd.DataFrame({"target": ground_truth, "prediction": predictions})
    report = Report([RegressionPreset()], include_tests=True)
    snapshot = report.run(eval_df, eval_df)

    prefix = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/model_quality"

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        json_path = Path(tmp) / "report.json"
        snapshot.save_html(str(html_path))
        snapshot.save_json(str(json_path))

        # Also write a simple metrics JSON (backward compat with dashboard)
        metrics = {"mae": mae, "rmse": rmse, "r2": r2, "item_count": len(merged_df)}
        metrics_path = Path(tmp) / "statistics.json"
        metrics_path.write_text(json.dumps({
            "version": 0.0,
            "dataset": {"item_count": len(merged_df)},
            "regression_metrics": {
                "mae": {"value": mae},
                "rmse": {"value": rmse},
                "r2": {"value": r2},
            },
        }, indent=2))

        s3.upload_file(str(html_path), config.S3_BUCKET, f"{prefix}/report.html")
        s3.upload_file(str(json_path), config.S3_BUCKET, f"{prefix}/report.json")
        s3.upload_file(str(metrics_path), config.S3_BUCKET, f"{prefix}/statistics.json")

    # Check against baseline thresholds
    baseline_mae_threshold = 0.248 * (1 + config.MODEL_QUALITY_DEGRADATION_THRESHOLD)
    violations = 0
    if mae > baseline_mae_threshold:
        violations += 1

    # Write violations file (backward compat)
    viol_list = []
    if mae > baseline_mae_threshold:
        viol_list.append({"metric_name": "mae", "constraint_check_type": "GreaterThanThreshold",
                          "description": f"MAE {mae:.4f} exceeds threshold {baseline_mae_threshold:.4f}"})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"violations": viol_list}, f, indent=2)
        s3.upload_file(f.name, config.S3_BUCKET, f"{prefix}/constraint_violations.json")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    parser.add_argument("--monitor", choices=["data-drift", "model-quality", "both"], default="both")
    args = parser.parse_args()

    session = _session()
    s3 = session.client("s3")

    if args.monitor in ("data-drift", "both"):
        print(f"\nLoading reference (2021) and current ({args.year}) data...")
        ref_df = _load_year_data(s3, config.TRAIN_YEAR)
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
