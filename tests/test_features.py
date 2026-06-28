"""Tests for feature engineering (DTI ordinal encoding, label encoding)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loan_rate_predictor import config
from loan_rate_predictor.processing.features import _dti_encode, _label_encode, engineer


def test_dti_encode_known_buckets():
    s = pd.Series(["<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%"])
    result = _dti_encode(s)
    assert list(result) == [0, 1, 2, 17, 18]


def test_dti_encode_integer_buckets():
    s = pd.Series(["36", "37", "49"])
    result = _dti_encode(s)
    assert list(result) == [3, 4, 16]


def test_dti_encode_missing_returns_neg1():
    s = pd.Series(["Exempt", None, "unknown"])
    result = _dti_encode(s)
    assert list(result) == [-1, -1, -1]


def test_label_encode_sorted_order():
    s = pd.Series(["C", "A", "B", "A"])
    encoded, mapping = _label_encode(s)
    assert mapping == {"A": 0, "B": 1, "C": 2}
    assert list(encoded) == [2, 0, 1, 0]


def test_label_encode_missing_returns_neg1():
    s = pd.Series(["A", None, "B"])
    encoded, mapping = _label_encode(s)
    assert encoded.iloc[1] == -1


def test_engineer_encodes_all_categoricals():
    data = {col: ["1", "2", "1"] for col in config.CATEGORICAL_FEATURES}
    df = pd.DataFrame(data)
    result, encodings = engineer(df)
    for col in config.CATEGORICAL_FEATURES:
        assert result[col].dtype == int, f"{col} not int after engineer()"


def test_engineer_dti_uses_ordinal():
    df = pd.DataFrame({"debt_to_income_ratio": ["<20%", ">60%"]})
    result, encodings = engineer(df)
    assert list(result["debt_to_income_ratio"]) == [0, 18]
    assert "debt_to_income_ratio" not in encodings


def test_engineer_returns_encodings_for_non_dti():
    df = pd.DataFrame({"loan_type": ["1", "2", "3"]})
    result, encodings = engineer(df)
    assert "loan_type" in encodings
    assert len(encodings["loan_type"]) == 3
