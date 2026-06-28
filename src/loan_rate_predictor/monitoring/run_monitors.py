"""Run data-quality (A) and model-quality (B) monitors on-demand for a year.

Each monitor is a SageMaker Processing job using the analyzer container — the same
thing a monitoring schedule runs under the hood, without the cron wrapper.

Usage:
    PYTHONPATH=src python -m loan_rate_predictor.monitoring.run_monitors --year 2022
    PYTHONPATH=src python -m loan_rate_predictor.monitoring.run_monitors --year 2022 --monitor data-quality
    PYTHONPATH=src python -m loan_rate_predictor.monitoring.run_monitors --year 2022 --monitor model-quality
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

from loan_rate_predictor import config

CW_NAMESPACE = "LoanRatePredictor/Monitoring"


def _session():
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    return sagemaker.Session(boto_session=boto_session)


def _run_analyzer(
    job_name: str,
    image_uri: str,
    dataset_uri: str,
    baseline_constraints_uri: str,
    baseline_statistics_uri: str,
    output_uri: str,
    env: dict,
    role_arn: str,
    sm_session: sagemaker.Session,
    wait: bool = True,
) -> None:
    """Run the Model Monitor analyzer container as a one-shot Processing job."""
    processor = Processor(
        role=role_arn,
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        env=env,
        sagemaker_session=sm_session,
    )

    processor.run(
        inputs=[
            ProcessingInput(
                source=dataset_uri,
                destination="/opt/ml/processing/input/endpoint",
                input_name="endpoint",
            ),
            ProcessingInput(
                source=baseline_constraints_uri,
                destination="/opt/ml/processing/baseline/constraints",
                input_name="constraints",
            ),
            ProcessingInput(
                source=baseline_statistics_uri,
                destination="/opt/ml/processing/baseline/statistics",
                input_name="statistics",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=output_uri,
                output_name="result",
            ),
        ],
        job_name=job_name,
        wait=wait,
        logs=wait,
    )


def _count_violations(s3_client, output_uri: str) -> int:
    """Read constraint_violations.json from monitor output and return violation count."""
    parsed = urlparse(output_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    key = f"{prefix}/constraint_violations.json"
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        s3_client.download_file(bucket, key, tmp)
        with open(tmp) as f:
            data = json.load(f)
        violations = data.get("violations", [])
        return len(violations)
    except s3_client.exceptions.NoSuchKey:
        # No violations file = no violations
        return 0


def _publish_metric(cw_client, metric_name: str, year: int, value: float) -> None:
    """Publish a custom CloudWatch metric in two forms:
    - Dimensioned (Year=YYYY): for per-year observability in CloudWatch dashboards.
    - Undimensioned: for the Terraform alarm, which has no dimensions block and watches
      only the undimensioned form of the metric.
    CloudWatch treats dimensioned and undimensioned metrics as distinct — an alarm with
    no dimensions block only fires on the undimensioned publish.
    """
    cw_client.put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[
            {
                "MetricName": metric_name,
                "Dimensions": [{"Name": "Year", "Value": str(year)}],
                "Value": value,
                "Unit": "Count",
            },
            {
                "MetricName": metric_name,
                "Value": value,
                "Unit": "Count",
            },
        ],
    )
    print(f"  Published {CW_NAMESPACE}/{metric_name} = {value} (Year={year} + undimensioned)")


def _read_computed_mae(s3_client, output_uri: str) -> float | None:
    """Read statistics.json from model-quality monitor output and return computed MAE."""
    parsed = urlparse(output_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    key = f"{prefix}/statistics.json"
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        s3_client.download_file(bucket, key, tmp)
        with open(tmp) as f:
            stats = json.load(f)
        # Shape varies by SDK version — try common paths
        reg = stats.get("regression_metrics", {})
        if isinstance(reg, dict) and "mae" in reg:
            return reg["mae"].get("value")
        if isinstance(reg, list):
            for m in reg:
                if m.get("name") == "mae":
                    return m.get("value")
        return None
    except Exception:
        return None


def run_data_quality_monitor(year: int, role_arn: str, sm_session: sagemaker.Session,
                              wait: bool = True) -> str:
    """Monitor A: year features vs 2021 feature baseline."""
    image_uri = image_uris.retrieve("model-monitor", config.AWS_REGION)
    dataset_uri = (
        f"s3://{config.S3_BUCKET}/{config.S3_PREDICTIONS_PREFIX}/{year}/monitor/features.csv"
    )
    baseline_prefix = f"s3://{config.S3_BUCKET}/{config.S3_BASELINE_PREFIX}/output"
    output_uri = (
        f"s3://{config.S3_BUCKET}/{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/data_quality"
    )
    job_name = f"loan-rate-dq-{year}-{int(time.time())}"

    env = {
        "dataset_format": json.dumps({"csv": {"header": True}}),
        "dataset_source": "/opt/ml/processing/input/endpoint",
        "output_path": "/opt/ml/processing/output",
        "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
        "baseline_statistics": "/opt/ml/processing/baseline/statistics/statistics.json",
        # ponytail: publish_cloudwatch_metrics omitted — standalone Processing jobs (no schedule)
        # don't emit to the SageMaker schedule namespace. Violations are read from S3 and
        # published as custom metrics in LoanRatePredictor/Monitoring instead (_publish_metric below).
    }

    print(f"[A] Data-quality monitor for year {year}")
    print(f"  Dataset:  {dataset_uri}")
    print(f"  Baseline: {baseline_prefix}")
    print(f"  Output:   {output_uri}")

    _run_analyzer(
        job_name=job_name,
        image_uri=image_uri,
        dataset_uri=dataset_uri,
        baseline_constraints_uri=f"{baseline_prefix}/constraints.json",
        baseline_statistics_uri=f"{baseline_prefix}/statistics.json",
        output_uri=output_uri,
        env=env,
        role_arn=role_arn,
        sm_session=sm_session,
        wait=wait,
    )

    print(f"[A] Data-quality monitor complete → {output_uri}")

    if wait:
        s3 = sm_session.boto_session.client("s3")
        cw = sm_session.boto_session.client("cloudwatch")
        n = _count_violations(s3, output_uri)
        _publish_metric(cw, "DataQualityViolations", year, float(n))
        print(f"  {n} violation(s) found")

    return output_uri


def run_model_quality_monitor(year: int, role_arn: str, sm_session: sagemaker.Session,
                               wait: bool = True) -> str:
    """Monitor B: year predictions+labels vs 2021 performance baseline."""
    # Image URI for B (model-quality) differs from A (data-quality / model-monitor image).
    # Verify both by inspecting the processing jobs created by your two suggest_baseline() runs:
    #   aws sagemaker list-processing-jobs --name-contains "model-quality-baseline" --profile ...
    #   aws sagemaker describe-processing-job --processing-job-name <name> --profile ...
    #   → AppSpecification.ImageUri is the ground-truth URI for your region + SDK version.
    # ponytail: version="1.0" may need updating — copy the URI from your baseline job.
    image_uri = image_uris.retrieve("clarify", config.AWS_REGION, version="1.0")
    dataset_uri = (
        f"s3://{config.S3_BUCKET}/{config.S3_PREDICTIONS_PREFIX}/{year}/merged/merged.csv"
    )
    baseline_prefix = f"s3://{config.S3_BUCKET}/{config.S3_BASELINE_PREFIX}/model_quality/output"
    output_uri = (
        f"s3://{config.S3_BUCKET}/{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/model_quality"
    )
    job_name = f"loan-rate-mq-{year}-{int(time.time())}"

    env = {
        "dataset_format": json.dumps({"csv": {"header": True}}),
        "dataset_source": "/opt/ml/processing/input/endpoint",
        "output_path": "/opt/ml/processing/output",
        "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
        "baseline_statistics": "/opt/ml/processing/baseline/statistics/statistics.json",
        "analysis_type": "MODEL_QUALITY",
        "problem_type": "Regression",
        # Column names must match merged.csv headers and the B suggest_baseline() call exactly.
        "inference_attribute": "prediction",
        "ground_truth_attribute": "ground_truth",
        # ponytail: publish_cloudwatch_metrics omitted — see A env comment above.
    }

    print(f"[B] Model-quality monitor for year {year}")
    print(f"  Dataset:  {dataset_uri}")
    print(f"  Baseline: {baseline_prefix}")
    print(f"  Output:   {output_uri}")

    _run_analyzer(
        job_name=job_name,
        image_uri=image_uri,
        dataset_uri=dataset_uri,
        baseline_constraints_uri=f"{baseline_prefix}/constraints.json",
        baseline_statistics_uri=f"{baseline_prefix}/statistics.json",
        output_uri=output_uri,
        env=env,
        role_arn=role_arn,
        sm_session=sm_session,
        wait=wait,
    )

    print(f"[B] Model-quality monitor complete → {output_uri}")

    if wait:
        s3 = sm_session.boto_session.client("s3")
        cw = sm_session.boto_session.client("cloudwatch")
        n = _count_violations(s3, output_uri)
        _publish_metric(cw, "ModelQualityViolations", year, float(n))
        print(f"  {n} violation(s) found")

        mae = _read_computed_mae(s3, output_uri)
        if mae is not None:
            expected_baseline = 0.248
            threshold = expected_baseline * (1 + config.MODEL_QUALITY_DEGRADATION_THRESHOLD)
            print(f"  Computed MAE: {mae:.4f}  baseline: {expected_baseline:.3f}  "
                  f"threshold: {threshold:.3f}  breach: {mae > threshold}")
        else:
            print("  WARNING: could not read computed MAE from statistics.json")

    return output_uri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    parser.add_argument("--monitor", choices=["data-quality", "model-quality", "both"], default="both")
    parser.add_argument("--role-arn", type=str, default=config.SAGEMAKER_ROLE_ARN)
    parser.add_argument("--wait", action="store_true", default=True)
    args = parser.parse_args()

    if not args.role_arn:
        raise ValueError("Set SAGEMAKER_ROLE_ARN env var or pass --role-arn")

    sm_session = _session()

    if args.monitor in ("data-quality", "both"):
        run_data_quality_monitor(args.year, args.role_arn, sm_session, args.wait)

    if args.monitor in ("model-quality", "both"):
        run_model_quality_monitor(args.year, args.role_arn, sm_session, args.wait)


if __name__ == "__main__":
    main()
