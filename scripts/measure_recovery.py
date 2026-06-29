"""Measure recovery after retraining: score both champions on the held-out EVAL slice.

Uses the same split_group_aware (random_state=42) as the training pipeline,
so the EVAL slice is exactly the rows the new champion never trained on.

Both models are scored on identical rows — the recovery magnitude is an
honest before/after comparison, not detection-MAE vs training-MAE.

Writes recovery/{year}.json to S3.

Usage:
    PYTHONPATH=src python scripts/measure_recovery.py --year 2022
    PYTHONPATH=src python scripts/measure_recovery.py --year 2022 --frozen-arn <arn>
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loan_rate_predictor import config
from loan_rate_predictor.registry import resolve_champion
from loan_rate_predictor.training.prepare import (
    feature_columns,
    load_data_year,
    split_group_aware,
)


def _download_model_tar(s3_client, model_data_url: str, dest: Path) -> Path:
    """Download model.tar.gz from an S3 URI, return local path."""
    # s3://bucket/key/model.tar.gz → bucket, key
    parts = model_data_url.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    dest.mkdir(parents=True, exist_ok=True)
    local_path = dest / "model.tar.gz"
    s3_client.download_file(bucket, key, str(local_path))
    return local_path


def _load_and_score(model_tar: Path, X: np.ndarray, work_dir: Path) -> np.ndarray:
    """Extract XGBoost model from tar and score features."""
    import xgboost as xgb

    extract_dir = work_dir / "model"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar) as tar:
        tar.extractall(extract_dir)
    booster = xgb.Booster()
    booster.load_model(str(extract_dir / "xgboost-model"))
    return booster.predict(xgb.DMatrix(X))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    parser.add_argument(
        "--frozen-arn",
        type=str,
        default=None,
        help="ARN of the previous (frozen) champion. If omitted, reads the most recent Rejected package.",
    )
    args = parser.parse_args()

    os.environ.setdefault("AWS_DEFAULT_REGION", config.AWS_REGION)
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    sm = boto_session.client("sagemaker")
    s3 = boto_session.client("s3")

    # Resolve new champion (current Approved)
    result = resolve_champion(sm)
    if result is None:
        print("No approved champion in registry.")
        sys.exit(1)
    new_arn, new_artifact = result

    # Resolve frozen champion (the one that was just rejected)
    if args.frozen_arn:
        frozen_arn = args.frozen_arn
    else:
        resp = sm.list_model_packages(
            ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus="Rejected",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        rejected = resp.get("ModelPackageSummaryList", [])
        if not rejected:
            print("No rejected champion found — cannot measure recovery without a frozen baseline.")
            sys.exit(1)
        frozen_arn = rejected[0]["ModelPackageArn"]

    frozen_desc = sm.describe_model_package(ModelPackageName=frozen_arn)
    frozen_artifact = frozen_desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    print(f"Frozen champion : {frozen_arn}")
    print(f"  Artifact      : {frozen_artifact}")
    print(f"New champion    : {new_arn}")
    print(f"  Artifact      : {new_artifact}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download processed.csv and regenerate the exact EVAL slice
        s3.download_file(
            config.S3_BUCKET,
            f"{config.S3_PROCESSED_PREFIX}/processed.csv",
            str(tmp_path / "processed.csv"),
        )
        df = load_data_year(tmp_path, args.year)
        train_df, val_df = split_group_aware(df)
        print(f"\nYear {args.year}: {len(df):,} total, {len(val_df):,} EVAL rows "
              f"({len(train_df):,} train, held out)")

        features = feature_columns()
        X_eval = val_df[features].values.astype(float)
        y_eval = val_df[config.TARGET].values.astype(float)

        # Download and score both models on the EVAL slice
        frozen_tar = _download_model_tar(s3, frozen_artifact, tmp_path / "frozen")
        new_tar = _download_model_tar(s3, new_artifact, tmp_path / "new")

        frozen_preds = _load_and_score(frozen_tar, X_eval, tmp_path / "frozen")
        new_preds = _load_and_score(new_tar, X_eval, tmp_path / "new")

    frozen_eval_mae = _mae(y_eval, frozen_preds)
    new_eval_mae = _mae(y_eval, new_preds)
    recovery_magnitude = frozen_eval_mae - new_eval_mae

    print(f"\nRecovery measurement (EVAL slice, {len(y_eval):,} rows):")
    print(f"  Frozen champion MAE : {frozen_eval_mae:.4f}")
    print(f"  New champion MAE    : {new_eval_mae:.4f}")
    print(f"  Recovery magnitude  : {recovery_magnitude:.4f}")
    print(f"  Threshold (alarm)   : {config.MODEL_QUALITY_DEGRADATION_THRESHOLD}")

    if recovery_magnitude <= 0:
        print("\nWARNING: No recovery — new champion is not better on EVAL slice.")

    recovery = {
        "year": args.year,
        "eval_rows": len(y_eval),
        "frozen_champion_arn": frozen_arn,
        "new_champion_arn": new_arn,
        "frozen_eval_mae": round(frozen_eval_mae, 4),
        "new_eval_mae": round(new_eval_mae, 4),
        "recovery_magnitude": round(recovery_magnitude, 4),
        "threshold": config.MODEL_QUALITY_DEGRADATION_THRESHOLD,
    }

    # Tag both models with their eval MAE
    for arn, mae_val in [(frozen_arn, frozen_eval_mae), (new_arn, new_eval_mae)]:
        desc = sm.describe_model_package(ModelPackageName=arn)
        meta = desc.get("CustomerMetadataProperties", {})
        meta[f"eval_mae_{args.year}"] = str(round(mae_val, 4))
        sm.update_model_package(ModelPackageArn=arn, CustomerMetadataProperties=meta)
    print(f"Tagged both models with eval_mae_{args.year}")

    # Upload to S3
    recovery_key = f"recovery/{args.year}.json"
    s3_uri = f"s3://{config.S3_BUCKET}/{recovery_key}"
    boto_session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    boto_session.client("s3").put_object(
        Bucket=config.S3_BUCKET,
        Key=recovery_key,
        Body=json.dumps(recovery, indent=2),
        ContentType="application/json",
    )
    print(f"\nRecovery JSON -> {s3_uri}")

    # Log to MLflow
    try:
        from loan_rate_predictor.tracking import log_recovery
        log_recovery(args.year, frozen_eval_mae, new_eval_mae, recovery_magnitude, len(y_eval))
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
