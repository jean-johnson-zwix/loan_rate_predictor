"""Tests for the preprocessing pipeline (filters, coercion, record IDs, column sanitisation)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loan_rate_predictor import config
from loan_rate_predictor.processing.preprocess import (
    apply_filters,
    coerce_numerics,
    make_record_id,
    sanitize_categoricals,
    _fs_safe_columns,
)


def _raw_row(**overrides) -> dict:
    """Minimal valid raw row (all strings, passes all filters)."""
    base = {
        "action_taken": "1",
        "lien_status": "1",
        "business_or_commercial_purpose": "2",
        "reverse_mortgage": "2",
        "open-end_line_of_credit": "2",
        "loan_amount": "200000",
        "rate_spread": "1.5",
        "income": "75000",
        "activity_year": "2021",
        "lei": "ABC123",
        "census_tract": "04013000100",
    }
    base.update(overrides)
    return base


def test_apply_filters_keeps_valid():
    df = pd.DataFrame([_raw_row()])
    result = apply_filters(df)
    assert len(result) == 1


def test_apply_filters_drops_wrong_action():
    df = pd.DataFrame([_raw_row(action_taken="3")])
    result = apply_filters(df)
    assert len(result) == 0


def test_apply_filters_drops_commercial():
    df = pd.DataFrame([_raw_row(business_or_commercial_purpose="1")])
    result = apply_filters(df)
    assert len(result) == 0


def test_coerce_numerics_converts():
    df = pd.DataFrame([_raw_row()])
    result = coerce_numerics(df)
    assert pd.api.types.is_numeric_dtype(result["loan_amount"])
    assert result["rate_spread"].iloc[0] == 1.5


def test_coerce_numerics_replaces_sentinels():
    df = pd.DataFrame([_raw_row(rate_spread="Exempt")])
    result = coerce_numerics(df)
    assert pd.isna(result["rate_spread"].iloc[0])


def test_coerce_numerics_replaces_1111():
    df = pd.DataFrame([_raw_row(income="1111")])
    result = coerce_numerics(df)
    assert pd.isna(result["income"].iloc[0])


def test_sanitize_categoricals_replaces_sentinels():
    df = pd.DataFrame([{"loan_type": "Exempt", "loan_purpose": "NA"}])
    result = sanitize_categoricals(df)
    assert result["loan_type"].iloc[0] is None
    assert result["loan_purpose"].iloc[0] is None


def test_sanitize_categoricals_replaces_nan_string():
    df = pd.DataFrame([{"loan_type": "nan"}])
    result = sanitize_categoricals(df)
    assert result["loan_type"].iloc[0] is None


def test_make_record_id_length():
    df = pd.DataFrame([_raw_row()])
    ids = make_record_id(df)
    assert len(ids.iloc[0]) == 16


def test_make_record_id_unique_across_rows():
    df = pd.DataFrame([_raw_row(), _raw_row()])
    ids = make_record_id(df)
    assert ids.iloc[0] != ids.iloc[1]


def test_make_record_id_deterministic():
    df = pd.DataFrame([_raw_row()])
    assert make_record_id(df).iloc[0] == make_record_id(df).iloc[0]


def test_fs_safe_columns_replaces_hyphens():
    df = pd.DataFrame({"derived_msa-md": [1], "open-end_line_of_credit": [2]})
    result = _fs_safe_columns(df)
    assert "derived_msa_md" in result.columns
    assert "open_end_line_of_credit" in result.columns
    assert "derived_msa-md" not in result.columns
