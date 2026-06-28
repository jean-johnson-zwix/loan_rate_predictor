"""Tests for XGBoost training data preparation."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loan_rate_predictor import config
from loan_rate_predictor.training.prepare import (
    assert_features_numeric,
    load_data_year,
    prepare,
    split_group_aware,
    write_xgb_csv,
)


def _processed_row(
    *,
    activity_year: int = 2021,
    lei: str = "lender-0",
    rate_spread: float | None = 1.0,
    offset: int = 0,
) -> dict:
    row = {
        "activity_year": activity_year,
        config.GROUP_SPLIT_KEY: lei,
        config.TARGET: rate_spread,
    }
    for i, col in enumerate(config.NUMERIC_FEATURES):
        row[col] = float(offset + i + 1)
    for i, col in enumerate(config.CATEGORICAL_FEATURES):
        row[col] = int((offset + i) % 4)
    return row


def test_load_data_year_filters_year_and_drops_missing_target(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    df = pd.DataFrame(
        [
            _processed_row(activity_year=2021, rate_spread=1.25),
            _processed_row(activity_year=2021, rate_spread=None),
            _processed_row(activity_year=2022, rate_spread=2.5),
        ]
    )
    df.to_csv(input_dir / "processed.csv", index=False)

    result = load_data_year(input_dir, 2021)

    assert len(result) == 1
    assert result[config.TARGET].iloc[0] == 1.25
    assert result["activity_year"].unique().tolist() == [2021]


def test_assert_features_numeric_raises_for_model_columns_only():
    df = pd.DataFrame(
        {
            config.TARGET: [1.0],
            config.NUMERIC_FEATURES[0]: ["not-a-number"],
            config.GROUP_SPLIT_KEY: ["lender-0"],
        }
    )

    with pytest.raises(ValueError, match=config.NUMERIC_FEATURES[0]):
        assert_features_numeric(df)


def test_assert_features_numeric_ignores_non_model_columns():
    df = pd.DataFrame(
        {
            config.TARGET: [1.0],
            config.NUMERIC_FEATURES[0]: [100000.0],
            config.GROUP_SPLIT_KEY: ["lender-0"],
        }
    )

    assert_features_numeric(df)


def test_split_group_aware_keeps_lenders_out_of_both_splits():
    df = pd.DataFrame(
        [
            _processed_row(lei=f"lender-{row_idx // 2}", offset=row_idx)
            for row_idx in range(24)
        ]
    )

    train_df, val_df = split_group_aware(df)

    train_groups = set(train_df[config.GROUP_SPLIT_KEY])
    val_groups = set(val_df[config.GROUP_SPLIT_KEY])
    assert len(train_df) + len(val_df) == len(df)
    assert train_groups.isdisjoint(val_groups)
    assert train_groups
    assert val_groups


def test_write_xgb_csv_writes_target_first_without_header(tmp_path):
    path = tmp_path / "train.csv"
    df = pd.DataFrame([_processed_row(rate_spread=2.5)])

    write_xgb_csv(df, path)

    cells = path.read_text().strip().split(",")
    assert cells[0] == "2.5"
    assert len(cells) == 1 + len(config.FEATURES)
    assert config.TARGET not in cells


def test_prepare_writes_train_and_val_csvs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    df = pd.DataFrame(
        [
            _processed_row(lei=f"lender-{row_idx // 2}", offset=row_idx)
            for row_idx in range(24)
        ]
        + [_processed_row(activity_year=2022, lei="future-lender", offset=99)]
    )
    df.to_csv(input_dir / "processed.csv", index=False)

    summary = prepare(input_dir, output_dir, 2021)

    train = pd.read_csv(output_dir / "train.csv", header=None)
    val = pd.read_csv(output_dir / "val.csv", header=None)
    assert summary["data_year"] == 2021
    assert summary["total_rows"] == 24
    assert summary["train_rows"] + summary["val_rows"] == 24
    assert train.shape[1] == 1 + len(config.FEATURES)
    assert val.shape[1] == 1 + len(config.FEATURES)
