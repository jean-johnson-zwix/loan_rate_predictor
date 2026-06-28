"""Join batch transform predictions with ground-truth labels for model-quality monitoring.

Batch transform preserves row order, so the join is positional: predictions[i]
corresponds to the i-th row of the year's processed data. Output is a two-column
CSV (prediction, ground_truth) that ModelQualityMonitor expects.

Usage:
    PYTHONPATH=src python -m loan_rate_predictor.monitoring.join_predictions_labels --year 2022
"""
import argparse
import tempfile
from pathlib import Path

import boto3
import pandas as pd

from loan_rate_predictor import config
from loan_rate_predictor.training.prepare import load_data_year


def _s3_client():
    return boto3.Session(
        profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION
    ).client("s3")


def join(year: int) -> None:
    s3 = _s3_client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Download row-ID sidecar (same row order as scoring_input.csv)
        ids_path = tmp_dir / "scoring_ids.csv"
        s3.download_file(
            config.S3_BUCKET,
            f"{config.S3_PREDICTIONS_PREFIX}/{year}/input/scoring_ids.csv",
            str(ids_path),
        )
        ids = pd.read_csv(ids_path)[config.RECORD_ID].values

        # 2. Download predictions (batch transform output, same row order as scoring_input.csv)
        pred_path = tmp_dir / "predictions.csv"
        s3.download_file(
            config.S3_BUCKET,
            f"{config.S3_PREDICTIONS_PREFIX}/{year}/output/scoring_input.csv.out",
            str(pred_path),
        )
        predictions = pd.read_csv(pred_path, header=None).iloc[:, 0].values

        if len(predictions) != len(ids):
            raise ValueError(
                f"Row count mismatch: {len(predictions)} predictions vs "
                f"{len(ids)} row IDs for year {year}"
            )

        # 3. Load ground-truth labels, keyed by record_id
        local_csv = tmp_dir / "processed.csv"
        s3.download_file(
            config.S3_BUCKET,
            f"{config.S3_PROCESSED_PREFIX}/processed.csv",
            str(local_csv),
        )
        df = load_data_year(tmp_dir, year)
        labels = df[[config.RECORD_ID, config.TARGET]].drop_duplicates(config.RECORD_ID)

        # 4. ID-based join — order-independent, correct by construction
        pred_df = pd.DataFrame({"prediction": predictions, config.RECORD_ID: ids})
        merged = pred_df.merge(labels, on=config.RECORD_ID, how="inner")
        merged = merged.rename(columns={config.TARGET: "ground_truth"})
        merged = merged[["prediction", "ground_truth"]]

        if len(merged) != len(ids):
            raise ValueError(
                f"Join lost rows: {len(ids)} predictions → {len(merged)} matched. "
                f"Check for duplicate or missing record_ids in year {year}."
            )

        # 5. Control check on 2021: full-data MAE (train+val) must stay under the
        # model-quality alarm threshold. This proves the pipeline runs clean and
        # monitors don't false-fire on in-distribution data. The stricter join-
        # integrity check (val-only MAE ≈ 0.248) lives in make_model_quality_baseline.py.
        if year == config.TRAIN_YEAR:
            mae = (merged["prediction"] - merged["ground_truth"]).abs().mean()
            threshold = 0.248 * (1 + config.MODEL_QUALITY_DEGRADATION_THRESHOLD)
            print(f"  Control check (2021 MAE): {mae:.4f}  threshold: {threshold:.3f}")
            if mae >= threshold:
                raise RuntimeError(
                    f"2021 full-data MAE {mae:.4f} >= threshold {threshold:.3f} — "
                    "expected no violation on in-distribution data. "
                    "Check record_id sidecar and predictions."
                )

        # 6. Write merged CSV
        merged_path = tmp_dir / "merged.csv"
        merged.to_csv(merged_path, index=False)
        merged_key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/merged/merged.csv"
        s3.upload_file(str(merged_path), config.S3_BUCKET, merged_key)
        print(f"Year {year}: {len(merged):,} rows joined")
        print(f"  → s3://{config.S3_BUCKET}/{merged_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=config.YEARS)
    args = parser.parse_args()
    join(args.year)


if __name__ == "__main__":
    main()
