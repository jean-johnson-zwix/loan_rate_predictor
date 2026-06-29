"""MLflow tracking integration for SageMaker managed MLflow.

Wraps mlflow calls so they're no-ops when MLFLOW_TRACKING_ARN is not set
(local dev, CI, etc). When set, logs to the SageMaker managed MLflow server.
"""
import os
from loan_rate_predictor import config

_initialized = False


def _ensure_init():
    global _initialized
    if _initialized:
        return
    _initialized = True
    arn = config.MLFLOW_TRACKING_ARN
    if not arn:
        return
    import mlflow
    mlflow.set_tracking_uri(arn)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def enabled():
    return bool(config.MLFLOW_TRACKING_ARN)


def log_training_run(data_year: int, metrics: dict, params: dict = None,
                     champion_metrics: dict = None, promoted: bool = False):
    """Log a training evaluation run to MLflow."""
    if not enabled():
        return
    _ensure_init()
    import mlflow

    with mlflow.start_run(run_name=f"train-{data_year}"):
        mlflow.set_tag("data_year", str(data_year))
        mlflow.set_tag("promoted", str(promoted))

        if params:
            mlflow.log_params(params)

        # Challenger metrics
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(f"challenger_{k}", v)

        # Champion metrics (if retrain, not bootstrap)
        if champion_metrics:
            for k, v in champion_metrics.items():
                if v is not None:
                    mlflow.log_metric(f"champion_{k}", v)


def log_recovery(year: int, frozen_mae: float, new_mae: float,
                 recovery_magnitude: float, eval_rows: int):
    """Log a recovery measurement to MLflow."""
    if not enabled():
        return
    _ensure_init()
    import mlflow

    with mlflow.start_run(run_name=f"recovery-{year}"):
        mlflow.set_tag("type", "recovery")
        mlflow.set_tag("year", str(year))
        mlflow.log_metric("frozen_eval_mae", frozen_mae)
        mlflow.log_metric("new_eval_mae", new_mae)
        mlflow.log_metric("recovery_magnitude", recovery_magnitude)
        mlflow.log_metric("eval_rows", eval_rows)


def log_monitoring(year: int, champion_version: int, mae: float,
                   dq_violations: int, mq_violations: int):
    """Log a monitoring run to MLflow."""
    if not enabled():
        return
    _ensure_init()
    import mlflow

    with mlflow.start_run(run_name=f"monitor-{year}"):
        mlflow.set_tag("type", "monitoring")
        mlflow.set_tag("year", str(year))
        mlflow.set_tag("champion_version", str(champion_version))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("dq_violations", dq_violations)
        mlflow.log_metric("mq_violations", mq_violations)
