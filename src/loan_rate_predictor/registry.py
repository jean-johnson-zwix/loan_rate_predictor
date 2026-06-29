"""Champion model resolution.

MLflow alias "champion" is the source of truth for which model is active.
SageMaker Model Registry stores artifacts for endpoint deployment.
The bridge: each MLflow model version carries a "sagemaker_arn" tag.

Fallback: if MLflow is not configured, falls back to SageMaker Approved status.
"""
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from loan_rate_predictor import config

MODEL_NAME = config.MODEL_PACKAGE_GROUP_NAME


def _mlflow_enabled():
    return bool(config.MLFLOW_TRACKING_ARN)


def _mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_ARN)
    return MlflowClient()


def resolve_champion(sm_client=None) -> tuple[str, str] | None:
    """Return (sagemaker_arn, artifact_uri) of the current champion.

    Reads the MLflow "champion" alias. Falls back to SageMaker Approved status
    if MLflow is not configured.
    """
    if _mlflow_enabled():
        try:
            client = _mlflow_client()
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            sagemaker_arn = mv.tags.get("sagemaker_arn")
            if not sagemaker_arn:
                print(f"MLflow champion (v{mv.version}) has no sagemaker_arn tag.")
                return None
            desc = sm_client.describe_model_package(ModelPackageName=sagemaker_arn)
            artifact_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
            return sagemaker_arn, artifact_uri
        except MlflowException:
            return None

    # Fallback: SageMaker Approved status
    return _resolve_from_sagemaker(sm_client)


def promote_champion(sm_client, new_sagemaker_arn: str, mlflow_version: str = None) -> None:
    """Set the "champion" alias on the new model version.

    If MLflow is configured, moves the alias. Also updates SageMaker approval
    status for backward compatibility (endpoint Terraform may depend on it).
    """
    if _mlflow_enabled() and mlflow_version:
        client = _mlflow_client()
        client.set_registered_model_alias(MODEL_NAME, "champion", mlflow_version)
        print(f"MLflow alias 'champion' -> v{mlflow_version}")

    # Also update SageMaker status for backward compat
    _promote_in_sagemaker(sm_client, new_sagemaker_arn)


def register_in_mlflow(metrics: dict, params: dict, sagemaker_arn: str) -> str | None:
    """Register a model version in MLflow, linked to its SageMaker ARN.

    Returns the MLflow version number, or None if MLflow is not configured.
    """
    if not _mlflow_enabled():
        return None

    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_ARN)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    client = MlflowClient()

    # Check if a version for this SageMaker ARN already exists
    try:
        existing = client.search_model_versions(f"name='{MODEL_NAME}'")
        for ev in existing:
            if (ev.tags or {}).get("sagemaker_arn") == sagemaker_arn:
                version = ev.version
                # Update metrics on the existing run
                if ev.run_id and metrics:
                    for k, v in metrics.items():
                        client.log_metric(ev.run_id, k, v)
                print(f"MLflow version v{version} already exists for {sagemaker_arn}, updated metrics")
                return str(version)
    except Exception:
        pass

    with mlflow.start_run(run_name=f"train-{params.get('data_year', '?')}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("sagemaker_arn", sagemaker_arn)

    # Create model version linked to the run
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass  # already exists

    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run.info.run_id}",
        run_id=run.info.run_id,
    )
    version = mv.version

    client.set_model_version_tag(MODEL_NAME, version, "sagemaker_arn", sagemaker_arn)
    client.set_model_version_tag(MODEL_NAME, version, "data_year", str(params.get("data_year", "")))

    print(f"Registered in MLflow: {MODEL_NAME} v{version} (sagemaker: {sagemaker_arn})")
    return str(version)


def get_latest_pending_sagemaker_arn(sm_client) -> str | None:
    """Get the latest PendingManualApproval model package ARN from SageMaker."""
    resp = sm_client.list_model_packages(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="PendingManualApproval",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    pending = resp.get("ModelPackageSummaryList", [])
    return pending[0]["ModelPackageArn"] if pending else None


# SageMaker fallback helpers

def _resolve_from_sagemaker(sm_client) -> tuple[str, str] | None:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    approved = response.get("ModelPackageSummaryList", [])
    if not approved:
        return None
    champion_arn = approved[0]["ModelPackageArn"]
    desc = sm_client.describe_model_package(ModelPackageName=champion_arn)
    artifact_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return champion_arn, artifact_uri


def _promote_in_sagemaker(sm_client, new_champion_arn: str) -> None:
    """Update SageMaker approval status (backward compat)."""
    current = _resolve_from_sagemaker(sm_client)
    if current:
        old_arn = current[0]
        if old_arn != new_champion_arn:
            sm_client.update_model_package(
                ModelPackageArn=old_arn, ModelApprovalStatus="Rejected")
            print(f"SageMaker: rejected {old_arn}")
    sm_client.update_model_package(
        ModelPackageArn=new_champion_arn, ModelApprovalStatus="Approved")
    print(f"SageMaker: approved {new_champion_arn}")
