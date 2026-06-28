"""Feature engineering for the Processing job: DTI ordinal encoding and categorical label encoding."""
import json
import pandas as pd
from loan_rate_predictor import config


def _dti_encode(series: pd.Series) -> pd.Series:
    """Map DTI string buckets to ordinal integers per config.DTI_ORDINAL.

    Args:
        series: Raw debt_to_income_ratio column (e.g. "30%-<36%", ">60%").

    Returns:
        Integer-encoded series; sentinel/unknown values become -1.
    """
    return series.map(config.DTI_ORDINAL).fillna(-1).astype(int)


def _label_encode(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Assign a stable integer label to each unique non-null value.

    Labels are assigned in sorted order starting from 0. NaN and any
    value absent from the vocabulary map to -1.

    Args:
        series: Categorical series with sentinels already replaced by NaN.

    Returns:
        Tuple of (encoded integer series, {original_value: integer} mapping).
    """
    cats = sorted(str(v) for v in series.dropna().unique())
    mapping = {v: i for i, v in enumerate(cats)}
    encoded = series.map(lambda x: mapping.get(str(x), -1) if pd.notna(x) else -1)
    return encoded.astype(int), mapping


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode all categorical features in-place.

    Applies DTI ordinal encoding to debt_to_income_ratio and label-encodes
    all other categorical features. Encoding is fit-and-transform in one
    pass over the full dataset.

    Args:
        df: DataFrame after filtering and numeric coercion.

    Returns:
        Tuple of (mutated df, {column: {label: int}} encoding map).
        Save the encoding map as an artifact — it is required at serving time.
    """
    encodings: dict = {}
    for col in config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if col == "debt_to_income_ratio":
            df[col] = _dti_encode(df[col])
        else:
            df[col], encodings[col] = _label_encode(df[col])
    return df, encodings
