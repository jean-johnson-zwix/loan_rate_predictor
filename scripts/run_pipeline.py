"""Launch the training pipeline with resolved parameters.

Resolves the current champion ARN from the Model Registry (if one exists)
and starts the pipeline with the given data year.

Usage:
    PYTHONPATH=src python scripts/run_pipeline.py --data-year 2021
    PYTHONPATH=src python scripts/run_pipeline.py --data-year 2022
"""
import argparse
import os
import subprocess

import boto3
import sagemaker

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loan_rate_predictor import config
from loan_rate_predictor.registry import resolve_champion
from pipelines.training_pipeline import get_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-year", type=int, required=True, choices=config.YEARS)
    parser.add_argument("--role-arn", type=str, default=config.SAGEMAKER_ROLE_ARN)
    parser.add_argument("--upsert", action="store_true", default=True,
                        help="Create or update the pipeline definition before starting")
    args = parser.parse_args()

    if not args.role_arn:
        raise ValueError("Set SAGEMAKER_ROLE_ARN env var or pass --role-arn")

    os.environ.setdefault("AWS_DEFAULT_REGION", config.AWS_REGION)
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)
    sm_client = boto_session.client("sagemaker")

    # Upload source code to S3 (same pattern as run_processing_job.py)
    source_s3 = f"s3://{config.S3_BUCKET}/pipeline-source"
    subprocess.run(
        ["aws", "s3", "sync", "src", source_s3,
         "--profile", config.AWS_PROFILE, "--delete"],
        check=True,
    )

    pipeline = get_pipeline(args.role_arn, boto_session, source_s3)

    if args.upsert:
        pipeline.upsert(role_arn=args.role_arn)
        print(f"Pipeline '{pipeline.name}' upserted.")

    result = resolve_champion(sm_client)
    if result:
        champion_arn, champion_uri = result
        print(f"Champion: {champion_arn}")
        print(f"  Artifact: {champion_uri}")
    else:
        champion_uri = ""
        print("No champion found — bootstrap run.")

    execution = pipeline.start(
        parameters={
            "DataYear": str(args.data_year),
            "ChampionModelUri": champion_uri or "NONE",
        },
    )

    print(f"Pipeline execution started: {execution.arn}")
    print(f"  DataYear: {args.data_year}")
    print(f"  Champion: {champion_uri or '(bootstrap)'}")
    print(f"\nPipeline runs async. When it succeeds:")
    print(f"  make evaluate-retrain DATA_YEAR={args.data_year}")


if __name__ == "__main__":
    main()
