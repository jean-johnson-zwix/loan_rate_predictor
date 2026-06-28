"""SageMaker Processing job entrypoint: filter, coerce types, engineer features, write output CSV, ingest to Feature Store."""
import os
import sys

_SM_CODE = "/opt/ml/processing/input/source"
if os.path.isdir(_SM_CODE) and _SM_CODE not in sys.path:
    sys.path.insert(0, _SM_CODE)

import hashlib
import json
from pathlib import Path

import pandas as pd

from loan_rate_predictor import config
from loan_rate_predictor.processing.features import engineer

_SM_INPUT = Path("/opt/ml/processing/input/data")
_SM_OUTPUT = Path("/opt/ml/processing/output")


def _dirs() -> tuple[Path, Path]:
    """Resolve input/output directories for SageMaker or local execution.

    Returns:
        Tuple of (input_dir, output_dir). Falls back to data/raw and
        data/processed when the SageMaker /opt/ml paths are absent.
    """
    input_dir = _SM_INPUT if _SM_INPUT.exists() else Path("data/raw")
    output_dir = _SM_OUTPUT if _SM_OUTPUT.exists() else Path("data/processed")
    return input_dir, output_dir


def load_raw(input_dir: Path) -> pd.DataFrame:
    """Load all HMDA vintage CSVs into one DataFrame with string dtypes.

    All columns are read as strings to preserve sentinel values for
    downstream coercion steps.

    Args:
        input_dir: Directory containing {year}.csv for each year in config.YEARS.

    Returns:
        Concatenated DataFrame across all vintages with original string dtypes.
    """
    frames = []
    for year in config.YEARS:
        path = input_dir / f"{year}.csv"
        df = pd.read_csv(path, dtype=str, low_memory=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only eligible loans per config.FILTERS.

    Keeps rows where action_taken=1, lien_status=1,
    business_or_commercial_purpose=2, reverse_mortgage=2,
    and open-end_line_of_credit=2.

    Args:
        df: Raw DataFrame with string-typed filter columns.

    Returns:
        Filtered DataFrame with reset index.
    """
    for col, allowed in config.FILTERS.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].isin(allowed)]
    return df.reset_index(drop=True)


def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel strings and coerce numeric columns to float64.

    Replaces config.SENTINELS with None before calling pd.to_numeric with
    errors='coerce'. Applies to NUMERIC_FEATURES and the TARGET column.

    Args:
        df: DataFrame with string-typed numeric columns.

    Returns:
        DataFrame with numeric columns as float64; missing/invalid values become NaN.
    """
    for col in config.NUMERIC_FEATURES + [config.TARGET]:
        if col not in df.columns:
            continue
        s = df[col].astype(str).replace(config.SENTINELS, None)
        df[col] = pd.to_numeric(s, errors="coerce")
    return df


def sanitize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel strings in categorical columns with None.

    Normalises config.SENTINELS and the literal string "nan" to NaN so
    that _label_encode treats them as missing (-1).

    Args:
        df: DataFrame after coerce_numerics.

    Returns:
        DataFrame with sentinel values replaced by None in CATEGORICAL_FEATURES.
    """
    for col in config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).replace(config.SENTINELS + ["nan"], None)
    return df


def make_record_id(df: pd.DataFrame) -> pd.Series:
    """Synthesise a 16-character surrogate record ID.

    Key: SHA-256(lei | census_tract | loan_amount | activity_year | row_index).
    Row index is included to guarantee uniqueness when loan attributes repeat.

    Args:
        df: DataFrame with lei, census_tract, loan_amount, activity_year columns.

    Returns:
        Series of 16-character lowercase hex strings.
    """
    key = (
        df.get("lei", pd.Series("", index=df.index)).astype(str)
        + "|" + df.get("census_tract", pd.Series("", index=df.index)).astype(str)
        + "|" + df.get("loan_amount", pd.Series("", index=df.index)).astype(str)
        + "|" + df.get("activity_year", pd.Series("", index=df.index)).astype(str)
        + "|" + df.index.astype(str)
    )
    return key.apply(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])


def _fs_safe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to satisfy Feature Store naming constraints ([a-zA-Z0-9_]).

    Replaces hyphens with underscores (e.g. derived_msa-md → derived_msa_md).

    Args:
        df: DataFrame with potentially hyphenated column names.

    Returns:
        DataFrame with sanitised column names (mutates df.columns in-place).
    """
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df


def main() -> None:
    """Run the full preprocessing pipeline.

    Loads raw HMDA CSVs, filters, coerces dtypes, engineers features,
    and writes processed.csv and categorical_encodings.json to the output
    directory.
    """
    input_dir, output_dir = _dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(input_dir)
    print(f"Loaded {len(df):,} rows from {input_dir}")

    df = apply_filters(df)
    print(f"After filters: {len(df):,} rows")

    df = coerce_numerics(df)
    df = sanitize_categoricals(df)

    df[config.RECORD_ID] = make_record_id(df)
    df["event_time"] = (
        pd.to_datetime(df["activity_year"].astype(str) + "-01-01")
        .astype("int64") // 10**9
    ).astype(float)

    df, encodings = engineer(df)

    keep = (
        [config.RECORD_ID, "event_time", config.TARGET]
        + config.NUMERIC_FEATURES
        + config.CATEGORICAL_FEATURES
        + list(config.GEO_KEYS)
        + config.DISPARITY_DIMENSIONS
        + ["activity_year"]
    )
    keep = [c for c in keep if c in df.columns]
    out = _fs_safe_columns(df[keep].copy())

    out.to_csv(output_dir / "processed.csv", index=False)
    with open(output_dir / "categorical_encodings.json", "w") as f:
        json.dump(encodings, f, indent=2)

    print(f"Wrote {len(out):,} rows → {output_dir}/processed.csv")
    print(f"Wrote categorical encodings → {output_dir}/categorical_encodings.json")


if __name__ == "__main__":
    main()
