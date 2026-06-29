"""Batch monitoring pipeline: score a year, join labels, run drift + quality reports.
  1. Batch transform (score frozen champion on year N)
  2. Join predictions with ground-truth labels
  3. Run Evidently drift + model quality reports

Usage:
    PYTHONPATH=src python scripts/run_monitoring.py --year 2022
"""
import argparse
import json
import subprocess
import sys

import boto3

from loan_rate_predictor import config
from loan_rate_predictor.registry import resolve_champion


def _run(args: list[str]) -> None:
    """Run a subprocess, forwarding stdout/stderr, failing on error."""
    print(f"\n{'=' * 60}")
    print(f"  {' '.join(args)}")
    print(f"{'=' * 60}\n")
    subprocess.run([sys.executable] + args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    args = parser.parse_args()

    year = str(args.year)

    # Resolve champion before scoring
    session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    sm = session.client("sagemaker")
    champion = resolve_champion(sm)
    if not champion:
        print("No approved champion in registry.")
        sys.exit(1)
    champion_arn, _ = champion
    desc = sm.describe_model_package(ModelPackageName=champion_arn)
    champion_version = desc.get("ModelPackageVersion")
    print(f"Champion: v{champion_version} ({champion_arn})")

    # 1. Score
    _run(["scripts/predict_with_champion.py", "--year", year])

    # 2. Join predictions with labels
    _run(["-m", "loan_rate_predictor.monitoring.join_predictions_labels", "--year", year])

    # 3. Run Evidently reports (data drift + model quality)
    _run(["-m", "loan_rate_predictor.monitoring.drift_report", "--year", year])

    # 4. Write monitoring metadata (which model produced these results)
    meta = {"year": int(year), "champion_version": champion_version, "champion_arn": champion_arn}
    meta_key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/meta.json"
    session.client("s3").put_object(
        Bucket=config.S3_BUCKET, Key=meta_key,
        Body=json.dumps(meta, indent=2), ContentType="application/json",
    )
    print(f"\nMonitoring complete for year {year} (champion v{champion_version}).")
    print("Check CloudWatch alarms and email for alerts.")


if __name__ == "__main__":
    main()
