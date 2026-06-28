"""Evaluate challenger vs champion on the current data year's val set.

Scores both models live on the same val slice. Promotion metric is MAE
(matches downstream recovery KPI). Also reports RMSE.

Bootstrap case (no champion): compare challenger against dumb baselines only.
"""
import json
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from loan_rate_predictor import config


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def load_val(val_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XGBoost-format val CSV (target col 0, no header)."""
    df = pd.read_csv(val_path, header=None)
    return df.iloc[:, 1:].values, df.iloc[:, 0].values


def load_xgb_model(model_tar_path: Path, work_dir: Path):
    """Extract and load an XGBoost model from a SageMaker model.tar.gz."""
    import xgboost as xgb
    extract_dir = work_dir / "model"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(model_tar_path) as tar:
        tar.extractall(extract_dir)
    model = xgb.Booster()
    model.load_model(str(extract_dir / "xgboost-model"))
    return model


def score_model(model, X: np.ndarray) -> np.ndarray:
    """Run inference with an XGBoost Booster."""
    import xgboost as xgb
    dmat = xgb.DMatrix(X)
    return model.predict(dmat)


def mean_baseline_metrics(y_train_mean: float, y_val: np.ndarray) -> dict:
    pred = np.full_like(y_val, y_train_mean, dtype=float)
    return {"rmse": _rmse(y_val, pred), "mae": _mae(y_val, pred)}


def best_linear_baseline_metrics(X_val: np.ndarray, y_val: np.ndarray,
                                  X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Best single-feature OLS on training data, scored on val."""
    best = {"rmse": float("inf"), "mae": float("inf"), "feature_idx": -1}
    for j in range(X_train.shape[1]):
        x = X_train[:, j]
        mask = np.isfinite(x) & np.isfinite(y_train)
        if mask.sum() < 10:
            continue
        xm, ym = x[mask], y_train[mask]
        x_aug = np.column_stack([xm, np.ones(len(xm))])
        coefs, _, _, _ = np.linalg.lstsq(x_aug, ym, rcond=None)
        pred = X_val[:, j] * coefs[0] + coefs[1]
        pred = np.where(np.isfinite(pred), pred, y_train.mean())
        r = _rmse(y_val, pred)
        if r < best["rmse"]:
            best = {"rmse": r, "mae": _mae(y_val, pred), "feature_idx": int(j)}
    return best


def evaluate(val_path: Path, train_path: Path, challenger_tar: Path,
             champion_tar: Optional[Path], work_dir: Path) -> dict:
    """Full evaluation. Returns dict with 'promote' boolean for the ConditionStep."""
    X_val, y_val = load_val(val_path)
    X_train, y_train = load_val(train_path)

    challenger_model = load_xgb_model(challenger_tar, work_dir / "challenger")
    challenger_preds = score_model(challenger_model, X_val)
    challenger_metrics = {
        "rmse": _rmse(y_val, challenger_preds),
        "mae": _mae(y_val, challenger_preds),
    }

    mean_bl = mean_baseline_metrics(y_train.mean(), y_val)

    features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    linear_bl = best_linear_baseline_metrics(X_val, y_val, X_train, y_train)
    linear_feature = features[linear_bl["feature_idx"]] if linear_bl["feature_idx"] >= 0 else "none"

    result = {
        "challenger": challenger_metrics,
        "baselines": {
            "mean_predictor": mean_bl,
            "best_single_feature_linear": {
                "feature": linear_feature,
                "rmse": linear_bl["rmse"],
                "mae": linear_bl["mae"],
            },
        },
    }

    if champion_tar is not None:
        champion_model = load_xgb_model(champion_tar, work_dir / "champion")
        champion_preds = score_model(champion_model, X_val)
        champion_metrics = {
            "rmse": _rmse(y_val, champion_preds),
            "mae": _mae(y_val, champion_preds),
        }
        result["champion"] = champion_metrics
        promote = challenger_metrics["mae"] < champion_metrics["mae"]
        result["promote"] = promote
        print(f"Challenger MAE: {challenger_metrics['mae']:.4f}  "
              f"Champion MAE: {champion_metrics['mae']:.4f}  "
              f"Promote: {promote}")
    else:
        # Bootstrap: no champion, compare against baseline floor
        promote = (challenger_metrics["mae"] < mean_bl["mae"]
                   and challenger_metrics["mae"] < linear_bl["mae"])
        result["champion"] = None
        result["promote"] = promote
        print(f"Bootstrap — Challenger MAE: {challenger_metrics['mae']:.4f}  "
              f"Mean baseline MAE: {mean_bl['mae']:.4f}  "
              f"Linear baseline MAE: {linear_bl['mae']:.4f}  "
              f"Promote: {promote}")

    return result
