"""Lambda handler for POST /price — synchronous pricing endpoint.

Applies the same frozen transforms as the Stage-1 Processing job:
  - Numeric: sentinel → None (empty CSV field → NaN in XGBoost)
  - DTI ordinal: string bucket → int, unknown → -1
  - Other categoricals: label lookup from frozen categorical_encodings.json, unknown → -1

The feature vector must be byte-equivalent to what the batch Processing job produces
for the same borrower. Verified by the round-trip gate in tests/test_pricing.py.
"""
import json
import os

import boto3

try:
    from loan_rate_predictor import config  # package context (tests, local dev)
except ImportError:
    import config  # Lambda bundle: config.py is a sibling in the zip

def _find_encodings() -> str:
    """Find categorical_encodings.json — Lambda bundle (sibling) or dev/test fallback."""
    sibling = os.path.join(os.path.dirname(__file__), "categorical_encodings.json")
    if os.path.exists(sibling):
        return sibling
    # Walk up to project root for dev/test
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(here, "data", "processed", "categorical_encodings.json")
        if os.path.exists(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError("categorical_encodings.json not found (not in Lambda bundle or project root)")


with open(_find_encodings()) as _f:
    ENCODINGS = json.load(_f)

_SM_RUNTIME = boto3.client("sagemaker-runtime", region_name=config.AWS_REGION)
_TRAINED_ON = os.environ.get("TRAINED_ON", "2021")


def _to_float(val):
    """Coerce value to float; sentinels and None → None (serialized as empty CSV field)."""
    if val is None or str(val) in config.SENTINELS:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _encode_dti(val):
    """DTI ordinal lookup from config.DTI_ORDINAL; unknown/missing → -1."""
    if val is None or str(val) in config.SENTINELS:
        return -1
    return config.DTI_ORDINAL.get(str(val), -1)


def _encode_cat(col, val):
    """Label lookup from frozen encodings; unknown/missing → -1 (matches batch behavior)."""
    if val is None or str(val) in config.SENTINELS:
        return -1
    return ENCODINGS.get(col, {}).get(str(val), -1)


def build_feature_row(body: dict) -> list:
    """Transform raw borrower inputs → 28-element feature list in config.FEATURES order.

    Returns a list where numeric values are float|None and categoricals are int.
    None represents NaN (missing); serialized as empty string in CSV payload.
    """
    row = []
    for col in config.NUMERIC_FEATURES:
        row.append(_to_float(body.get(col)))
    for col in config.CATEGORICAL_FEATURES:
        if col == "debt_to_income_ratio":
            row.append(_encode_dti(body.get(col)))
        else:
            row.append(_encode_cat(col, body.get(col)))

    n = len(config.FEATURES)
    if len(row) != n:
        raise ValueError(f"Feature count mismatch: got {len(row)}, expected {n}")
    return row


def _row_to_csv(row: list) -> str:
    """Serialize feature row to CSV string (no header). None → empty field (XGBoost NaN)."""
    return ",".join("" if v is None else str(v) for v in row)


def handler(event, context):
    """API Gateway HTTP API v2 handler."""
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON body"})}

    try:
        row = build_feature_row(body)
    except (ValueError, Exception) as exc:
        return {"statusCode": 422, "body": json.dumps({"error": str(exc)})}

    response = _SM_RUNTIME.invoke_endpoint(
        EndpointName=config.ENDPOINT_NAME,
        ContentType="text/csv",
        Body=_row_to_csv(row).encode(),
    )
    rate_spread = float(response["Body"].read().decode().strip())

    loan_term = int(_to_float(body.get("loan_term")) or 360)
    apor = config.APOR_BY_TERM.get(loan_term, config.APOR_BY_TERM[360])

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "rate_spread": round(rate_spread, 4),
            "indicative_apr": round(apor + rate_spread, 4),
            "trained_on": _TRAINED_ON,
        }),
    }
