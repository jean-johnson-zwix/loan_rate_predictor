"""Tests for model evaluation metrics and promotion decisions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loan_rate_predictor.training import evaluate as evaluate_module


def _write_xgb_csv(path: Path, y: np.ndarray, X: np.ndarray) -> None:
    pd.DataFrame(np.column_stack([y, X])).to_csv(path, index=False, header=False)


def test_load_val_splits_target_and_features(tmp_path):
    path = tmp_path / "val.csv"
    _write_xgb_csv(
        path,
        np.array([1.0, 2.0]),
        np.array([[10.0, 11.0], [20.0, 21.0]]),
    )

    X, y = evaluate_module.load_val(path)

    assert X.tolist() == [[10.0, 11.0], [20.0, 21.0]]
    assert y.tolist() == [1.0, 2.0]


def test_mean_baseline_metrics_uses_training_mean():
    metrics = evaluate_module.mean_baseline_metrics(
        y_train_mean=2.0,
        y_val=np.array([1.0, 2.0, 4.0]),
    )

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))


def test_best_linear_baseline_selects_best_single_feature():
    train_x0 = np.arange(12, dtype=float)
    X_train = np.column_stack([train_x0, np.ones_like(train_x0)])
    y_train = 2 * train_x0 + 1
    val_x0 = np.array([12.0, 13.0, 14.0])
    X_val = np.column_stack([val_x0, np.ones_like(val_x0)])
    y_val = 2 * val_x0 + 1

    metrics = evaluate_module.best_linear_baseline_metrics(
        X_val,
        y_val,
        X_train,
        y_train,
    )

    assert metrics["feature_idx"] == 0
    assert metrics["mae"] == pytest.approx(0)
    assert metrics["rmse"] == pytest.approx(0)


def test_evaluate_bootstrap_promotes_challenger_that_beats_baselines(
    tmp_path,
    monkeypatch,
):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    y_val = np.array([1.0, 2.0, 3.0])
    _write_xgb_csv(train_path, np.zeros(12), np.zeros((12, 2)))
    _write_xgb_csv(val_path, y_val, np.zeros((3, 2)))

    monkeypatch.setattr(
        evaluate_module,
        "load_xgb_model",
        lambda model_tar_path, work_dir: object(),
    )
    monkeypatch.setattr(
        evaluate_module,
        "score_model",
        lambda model, X: y_val.copy(),
    )

    result = evaluate_module.evaluate(
        val_path,
        train_path,
        tmp_path / "challenger.tar.gz",
        champion_tar=None,
        work_dir=tmp_path,
    )

    assert result["promote"] is True
    assert result["champion"] is None
    assert result["challenger"]["mae"] == pytest.approx(0)


def test_evaluate_promotes_challenger_over_weaker_champion(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    y_val = np.array([1.0, 2.0, 3.0])
    _write_xgb_csv(train_path, np.zeros(12), np.zeros((12, 2)))
    _write_xgb_csv(val_path, y_val, np.zeros((3, 2)))

    class FakeModel:
        def __init__(self, kind: str) -> None:
            self.kind = kind

    def fake_load_model(model_tar_path: Path, work_dir: Path) -> FakeModel:
        kind = "champion" if "champion" in str(model_tar_path) else "challenger"
        return FakeModel(kind)

    def fake_score_model(model: FakeModel, X: np.ndarray) -> np.ndarray:
        if model.kind == "challenger":
            return y_val.copy()
        return np.zeros(len(X))

    monkeypatch.setattr(evaluate_module, "load_xgb_model", fake_load_model)
    monkeypatch.setattr(evaluate_module, "score_model", fake_score_model)

    result = evaluate_module.evaluate(
        val_path,
        train_path,
        tmp_path / "challenger.tar.gz",
        champion_tar=tmp_path / "champion.tar.gz",
        work_dir=tmp_path,
    )

    assert result["promote"] is True
    assert result["challenger"]["mae"] == pytest.approx(0)
    assert result["champion"]["mae"] == pytest.approx(2.0)
