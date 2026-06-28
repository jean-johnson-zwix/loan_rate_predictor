"""Batch process pipeline: score a year, join labels, run both monitors.
  1. Batch transform (score frozen champion on year N)
  2. Join predictions with ground-truth labels
  3. Run data-quality (A) and model-quality (B) monitors

Usage:
    PYTHONPATH=src python scripts/batch_process.py --year 2022
"""
import argparse
import subprocess
import sys

from loan_rate_predictor import config


def _run(args: list[str]) -> None:
    """Run a subprocess, forwarding stdout/stderr, failing on error."""
    print(f"\n{'=' * 60}")
    print(f"  {' '.join(args)}")
    print(f"{'=' * 60}\n")
    subprocess.run([sys.executable] + args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    parser.add_argument("--role-arn", type=str, default=config.SAGEMAKER_ROLE_ARN)
    args = parser.parse_args()

    year = str(args.year)
    role_args = ["--role-arn", args.role_arn] if args.role_arn else []

    # 1. Score
    _run(["scripts/predict_with_champion.py", "--year", year])

    # 2. Join predictions with labels
    _run(["-m", "loan_rate_predictor.monitoring.join_predictions_labels", "--year", year])

    # 3. Run both monitors
    _run(["-m", "loan_rate_predictor.monitoring.run_monitors", "--year", year] + role_args)

    print(f"\nMonitoring complete for year {year}.")
    print("Check CloudWatch alarms and email for alerts.")


if __name__ == "__main__":
    main()
