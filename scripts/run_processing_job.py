"""Submit the HMDA preprocessing + Feature Store ingest job to SageMaker Processing.

Usage:
    PYTHONPATH=src python scripts/run_processing_job.py
"""
import os
import subprocess

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

from loan_rate_predictor import config


def main() -> None:
    """Submit a SKLearnProcessor job that preprocesses HMDA data and ingests to Feature Store."""
    os.environ.setdefault("AWS_DEFAULT_REGION", config.AWS_REGION)

    boto_session = boto3.Session(
        profile_name=config.AWS_PROFILE,
        region_name=config.AWS_REGION,
    )
    sm_session = sagemaker.Session(boto_session=boto_session)

    # ponytail: aws s3 sync instead of S3Uploader.upload — the CLI normalises
    # Windows backslash paths to forward slashes in S3 keys, which S3Uploader does not.
    source_s3 = f"s3://{config.S3_BUCKET}/processing-source"
    subprocess.run(
        [
            "aws", "s3", "sync", "src", source_s3,
            "--profile", config.AWS_PROFILE,
            "--delete",
        ],
        check=True,
    )

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=config.SAGEMAKER_ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=sm_session,
    )

    processor.run(
        code="src/loan_rate_predictor/processing/preprocess.py",
        inputs=[
            ProcessingInput(
                source=source_s3,
                destination="/opt/ml/processing/input/source",
            ),
            ProcessingInput(
                source=f"s3://{config.S3_BUCKET}/{config.S3_RAW_PREFIX}/",
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{config.S3_BUCKET}/{config.S3_PROCESSED_PREFIX}/",
            )
        ],
        wait=True,
        logs=True,
    )

    print("Processing job complete.")


if __name__ == "__main__":
    main()
