"""Split processed.csv into XGBoost-format train/val CSVs with group-aware splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from loan_rate_predictor import config


def load_data_year(input_dir: Path, data_year: int) -> pd.DataFrame:
    """Load processed.csv and filter to a single data_year."""
    df = pd.read_csv(input_dir / "processed.csv", low_memory=False)
    df = df[df["activity_year"] == data_year].copy()
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    df = df.dropna(subset=[config.TARGET])
    return df


def feature_columns() -> list[str]:
    return config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def split_group_aware(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GroupShuffleSplit on config.GROUP_SPLIT_KEY. Returns (train, val)."""
    groups = df[config.GROUP_SPLIT_KEY].fillna("_missing_")
    gss = GroupShuffleSplit(n_splits=1, test_size=config.VAL_FRACTION, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=groups))

    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    overlap = train_groups & val_groups
    assert len(overlap) == 0, f"Group leakage: {len(overlap)} groups in both train and val"

    return df.iloc[train_idx], df.iloc[val_idx]


def assert_features_numeric(df: pd.DataFrame) -> None:
    """Assert all feature + target columns are numeric. Only checks model columns, not split/analysis keys."""
    cols = feature_columns() + [config.TARGET]
    cols = [c for c in cols if c in df.columns]
    non_numeric = df[cols].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")


def write_xgb_csv(df: pd.DataFrame, path: Path) -> None:
    """Write XGBoost-format CSV: target first column, features only, no header."""
    cols = [config.TARGET] + feature_columns()
    df[cols].to_csv(path, index=False, header=False)


def prepare(input_dir: Path, output_dir: Path, data_year: int) -> dict:
    """Full prepare step: load data_year, split, assert, write."""
    df = load_data_year(input_dir, data_year)
    print(f"Year {data_year}: {len(df):,} rows")

    assert_features_numeric(df)

    train_df, val_df = split_group_aware(df)
    n_train_groups = train_df[config.GROUP_SPLIT_KEY].nunique()
    n_val_groups = val_df[config.GROUP_SPLIT_KEY].nunique()
    print(f"  Train: {len(train_df):,} ({n_train_groups} groups)  "
          f"Val: {len(val_df):,} ({n_val_groups} groups)")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    write_xgb_csv(train_df, train_path)
    write_xgb_csv(val_df, val_path)

    return {
        "data_year": data_year,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_groups": n_train_groups,
        "val_groups": n_val_groups,
    }
