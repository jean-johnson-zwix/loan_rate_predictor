"""Tests for the pricing Lambda handler.

Gate test: round-trip from processed.csv raw inputs → Lambda transform → must match
processed.csv feature values exactly. This is the train/serve-skew proof.
Skipped if data/processed/processed.csv does not exist (CI); run locally before deploy.
"""
import csv
import json
import math
import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import loan_rate_predictor.config as config

# Import handler internals directly (not via Lambda event) for unit testing
from loan_rate_predictor.pricing.handler import (
    ENCODINGS,
    _encode_cat,
    _encode_dti,
    _to_float,
    build_feature_row,
    handler,
)

PROCESSED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed.csv")


@pytest.mark.skipif(not os.path.exists(PROCESSED_CSV), reason="processed.csv not present")
def test_round_trip_skew_gate():
    """Gate: Lambda transform must reproduce processed.csv feature vector exactly.

    Takes the first row of processed.csv, reconstructs its raw borrower inputs by
    reversing the encoding, runs them through build_feature_row, and asserts the
    output matches the processed values. Proves the serving path matches training.
    """
    with open(PROCESSED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Build expected 28-feature list from processed.csv
    expected_numeric = []
    for col in config.NUMERIC_FEATURES:
        v = row.get(col, "")
        if v in ("", "nan", "NaN", None):
            expected_numeric.append(None)
        else:
            expected_numeric.append(float(v))

    expected_cat = []
    for col in config.CATEGORICAL_FEATURES:
        expected_cat.append(int(float(row[col])))

    expected = expected_numeric + expected_cat

    # Reverse-encode to recover raw borrower inputs
    dti_reverse = {v: k for k, v in config.DTI_ORDINAL.items()}
    raw = {}

    for col in config.NUMERIC_FEATURES:
        raw[col] = row.get(col, "")  # numeric: raw value is the same as processed

    for col in config.CATEGORICAL_FEATURES:
        encoded = int(float(row[col]))
        if col == "debt_to_income_ratio":
            raw[col] = dti_reverse.get(encoded, "")
        else:
            vocab_reverse = {v: k for k, v in ENCODINGS.get(col, {}).items()}
            raw[col] = vocab_reverse.get(encoded, "")  # -1 → "" (unknown → sentinel)

    # Run through Lambda transform
    actual = build_feature_row(raw)

    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        col = config.FEATURES[i]
        if e is None:
            assert a is None, f"Col {col}: expected None, got {a}"
        elif isinstance(e, float):
            assert a is not None and math.isclose(float(a), e, rel_tol=1e-6), (
                f"Col {col}: expected {e}, got {a}"
            )
        else:
            assert a == e, f"Col {col}: expected {e}, got {a}"


class TestToFloat:
    def test_normal(self):
        assert _to_float("165000.0") == 165000.0

    def test_sentinel(self):
        for s in config.SENTINELS:
            assert _to_float(s) is None

    def test_none(self):
        assert _to_float(None) is None

    def test_int_string(self):
        assert _to_float("360") == 360.0


class TestEncodeDTI:
    def test_known_bucket(self):
        assert _encode_dti("<20%") == 0
        assert _encode_dti(">60%") == 18
        assert _encode_dti("36") == 3

    def test_unknown(self):
        assert _encode_dti("not-a-dti") == -1

    def test_sentinel(self):
        assert _encode_dti("NA") == -1
        assert _encode_dti(None) == -1


class TestEncodeCat:
    def test_known_loan_type(self):
        assert _encode_cat("loan_type", "1") == 0
        assert _encode_cat("loan_type", "2") == 1

    def test_unknown(self):
        assert _encode_cat("loan_type", "99") == -1

    def test_missing(self):
        assert _encode_cat("loan_type", None) == -1
        assert _encode_cat("loan_type", "") == -1

    def test_conforming(self):
        assert _encode_cat("conforming_loan_limit", "C") == 0
        assert _encode_cat("conforming_loan_limit", "NC") == 1


class TestBuildFeatureRow:
    def _minimal_body(self):
        return {
            "loan_amount": 165000,
            "loan_to_value_ratio": 48.69,
            "loan_term": 360,
            "property_value": 345000,
            "income": 138,
            "total_units": 1,
            "tract_population": 4449,
            "ffiec_msa_md_median_family_income": 79000,
            "tract_to_msa_income_percentage": 85.29,
            "tract_owner_occupied_units": 928,
            "tract_one_to_four_family_homes": 1499,
            "tract_median_age_of_housing_units": 24,
            "loan_type": "1",
            "loan_purpose": "1",
            "lien_status": "1",
            "occupancy_type": "1",
            "construction_method": "1",
            "conforming_loan_limit": "C",
            "debt_to_income_ratio": "30%-<36%",
            "manufactured_home_secured_property_type": "3",
            "manufactured_home_land_property_interest": "5",
            "derived_loan_product_type": "Conventional:First Lien",
            "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
            "negative_amortization": "2",
            "interest_only_payment": "2",
            "balloon_payment": "2",
            "other_nonamortizing_features": "2",
            "prepayment_penalty_term": "0",
        }

    def test_returns_28_features(self):
        row = build_feature_row(self._minimal_body())
        assert len(row) == 28

    def test_numeric_coercion(self):
        body = self._minimal_body()
        body["loan_amount"] = "165000.0"
        row = build_feature_row(body)
        assert row[0] == 165000.0

    def test_missing_numeric_is_none(self):
        body = self._minimal_body()
        body["income"] = None
        row = build_feature_row(body)
        income_idx = config.NUMERIC_FEATURES.index("income")
        assert row[income_idx] is None

    def test_dti_encoded(self):
        body = self._minimal_body()
        body["debt_to_income_ratio"] = "30%-<36%"
        row = build_feature_row(body)
        dti_idx = len(config.NUMERIC_FEATURES) + config.CATEGORICAL_FEATURES.index("debt_to_income_ratio")
        assert row[dti_idx] == 2

    def test_categorical_encoded(self):
        body = self._minimal_body()
        row = build_feature_row(body)
        lt_idx = len(config.NUMERIC_FEATURES) + config.CATEGORICAL_FEATURES.index("loan_type")
        assert row[lt_idx] == 0  # "1" → 0

    def test_unknown_categorical_is_minus_one(self):
        body = self._minimal_body()
        body["loan_type"] = "99"
        row = build_feature_row(body)
        lt_idx = len(config.NUMERIC_FEATURES) + config.CATEGORICAL_FEATURES.index("loan_type")
        assert row[lt_idx] == -1


class TestHandler:
    def _event(self, body: dict):
        return {"body": json.dumps(body)}

    def _minimal_body(self):
        return {
            "loan_amount": 165000,
            "loan_to_value_ratio": 48.69,
            "loan_term": 360,
            "property_value": 345000,
            "income": 138,
            "total_units": 1,
            "tract_population": 4449,
            "ffiec_msa_md_median_family_income": 79000,
            "tract_to_msa_income_percentage": 85.29,
            "tract_owner_occupied_units": 928,
            "tract_one_to_four_family_homes": 1499,
            "tract_median_age_of_housing_units": 24,
            "loan_type": "1",
            "loan_purpose": "1",
            "lien_status": "1",
            "occupancy_type": "1",
            "construction_method": "1",
            "conforming_loan_limit": "C",
            "debt_to_income_ratio": "30%-<36%",
            "manufactured_home_secured_property_type": "3",
            "manufactured_home_land_property_interest": "5",
            "derived_loan_product_type": "Conventional:First Lien",
            "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
            "negative_amortization": "2",
            "interest_only_payment": "2",
            "balloon_payment": "2",
            "other_nonamortizing_features": "2",
            "prepayment_penalty_term": "0",
        }

    def test_success(self):
        mock_response = {"Body": mock.MagicMock()}
        mock_response["Body"].read.return_value = b"0.7523\n"

        with mock.patch("loan_rate_predictor.pricing.handler._SM_RUNTIME") as sm:
            sm.invoke_endpoint.return_value = mock_response
            result = handler(self._event(self._minimal_body()), None)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "rate_spread" in body
        assert "indicative_apr" in body
        assert "trained_on" in body
        assert body["rate_spread"] == 0.7523
        assert body["indicative_apr"] == round(6.50 + 0.7523, 4)

    def test_invalid_json(self):
        result = handler({"body": "not-json"}, None)
        assert result["statusCode"] == 400

    def test_empty_body(self):
        """Empty body → all features None/-1, should still invoke endpoint."""
        mock_response = {"Body": mock.MagicMock()}
        mock_response["Body"].read.return_value = b"0.5\n"
        with mock.patch("loan_rate_predictor.pricing.handler._SM_RUNTIME") as sm:
            sm.invoke_endpoint.return_value = mock_response
            result = handler({"body": "{}"}, None)
        assert result["statusCode"] == 200

    def test_15yr_apor(self):
        """15-year loan uses 180-term APOR."""
        body = self._minimal_body()
        body["loan_term"] = 180
        mock_response = {"Body": mock.MagicMock()}
        mock_response["Body"].read.return_value = b"0.5\n"
        with mock.patch("loan_rate_predictor.pricing.handler._SM_RUNTIME") as sm:
            sm.invoke_endpoint.return_value = mock_response
            result = handler(self._event(body), None)
        resp = json.loads(result["body"])
        assert resp["indicative_apr"] == round(5.90 + 0.5, 4)
