"""SageMaker Processing entrypoint for the evaluate step.

Uses a separate container from training (SKLearnProcessor + xgboost install).
Champion model is downloaded via boto3 (not a ProcessingInput) to handle the
bootstrap case where no champion exists.
"""
import os
import subprocess
import sys

_SM_CODE = "/opt/ml/processing/input/source"
if os.path.isdir(_SM_CODE) and _SM_CODE not in sys.path:
    sys.path.insert(0, _SM_CODE)

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import boto3

from loan_rate_predictor.training.evaluate import evaluate


def _download_s3(uri: str, local_path: Path) -> None:
    """Download an S3 URI to a local path."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-model-uri", type=str, default="")
    args = parser.parse_args()

    val_path = Path("/opt/ml/processing/input/val/val.csv")
    train_path = Path("/opt/ml/processing/input/train/train.csv")
    challenger_tar = Path("/opt/ml/processing/input/model/model.tar.gz")
    output_dir = Path("/opt/ml/processing/output")
    work_dir = Path("/tmp/evaluate")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    champion_tar = None
    if args.champion_model_uri and args.champion_model_uri != "NONE":
        champion_tar = work_dir / "champion_model.tar.gz"
        print(f"Downloading champion model: {args.champion_model_uri}")
        _download_s3(args.champion_model_uri, champion_tar)
    else:
        print("No champion — bootstrap run")

    result = evaluate(val_path, train_path, challenger_tar, champion_tar, work_dir)

    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote evaluation.json → {output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
