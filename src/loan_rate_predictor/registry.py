"""Champion model resolution with single-champion invariant.

Write side (promote_champion): approves challenger + rejects incumbent = transition.
Read side (resolve_champion): asserts exactly one Approved, raises on corruption.
"""
from loan_rate_predictor import config


def resolve_champion(sm_client) -> tuple[str, str] | None:
    """Return (arn, artifact_uri) of the single champion, or None if no champion exists.

    Raises if multiple Approved packages exist — that's a promotion bug, not
    something to silently fix at read time.
    """
    response = sm_client.list_model_packages(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    approved = response.get("ModelPackageSummaryList", [])
    if not approved:
        return None

    if len(approved) > 1:
        arns = [p["ModelPackageArn"] for p in approved]
        raise RuntimeError(
            f"Multiple Approved model packages — promotion failed to reject incumbent. "
            f"Approved: {arns}. Fix manually, then investigate the promotion path."
        )

    champion_arn = approved[0]["ModelPackageArn"]
    desc = sm_client.describe_model_package(ModelPackageName=champion_arn)
    artifact_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return champion_arn, artifact_uri


def promote_champion(sm_client, new_champion_arn: str) -> None:
    """Approve new_champion_arn and reject the current champion (if any).

    This is the single write that maintains the one-champion invariant.
    """
    current = resolve_champion(sm_client)
    if current:
        old_arn = current[0]
        if old_arn == new_champion_arn:
            print(f"Already champion: {old_arn}")
            return
        sm_client.update_model_package(
            ModelPackageArn=old_arn,
            ModelApprovalStatus="Rejected",
        )
        print(f"Rejected incumbent: {old_arn}")

    sm_client.update_model_package(
        ModelPackageArn=new_champion_arn,
        ModelApprovalStatus="Approved",
    )
    print(f"Promoted champion: {new_champion_arn}")
