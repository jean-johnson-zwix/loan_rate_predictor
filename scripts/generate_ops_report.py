"""Generate ops dashboard data (frontend/ops-data.json) from Registry + S3 artifacts.

Data sources:
  1. Model Registry — all packages with metadata
  2. Model-quality baseline — thresholds the alarms fire on
  3. Monitoring outputs per year — violations + regression metrics
  4. Data-quality statistics — per-feature drift distances + mean shifts
  5. Recovery JSONs — before/after eval-slice MAE

Usage:
    PYTHONPATH=src python scripts/generate_ops_report.py
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loan_rate_predictor import config

OUTPUT = Path(__file__).resolve().parent.parent / "ops-dashboard" / "ops-data.json"
DRIFT_THRESHOLD = 0.1


def _s3_json(s3, key):
    try:
        resp = s3.get_object(Bucket=config.S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def _get_model_packages_from_mlflow():
    """Read model versions + metrics from MLflow registry."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_ARN)
    client = MlflowClient()
    model_name = config.MODEL_PACKAGE_GROUP_NAME

    # Get champion alias
    champion_version = None
    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
        champion_version = champ.version
    except Exception:
        pass

    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number ASC"])
    packages = []
    for mv in versions:
        tags = mv.tags or {}
        # Get metrics from the run
        metrics = {}
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                metrics = run.data.metrics
            except Exception:
                pass

        def _float(d, k):
            v = d.get(k) or metrics.get(k)
            if v is None: return None
            try: return float(v)
            except (ValueError, TypeError): return None

        def _int(d, k):
            v = d.get(k)
            if v is None: return None
            try: return int(float(v))
            except (ValueError, TypeError): return None

        is_champion = str(mv.version) == str(champion_version)
        run_params = {}
        if mv.run_id:
            try:
                run_params = client.get_run(mv.run_id).data.params
            except Exception:
                pass

        packages.append({
            "version": int(mv.version),
            "status": "Approved" if is_champion else "Rejected",
            "created": mv.creation_timestamp // 1000 if mv.creation_timestamp else None,
            "trained_on": tags.get("data_year") or run_params.get("data_year"),
            "objective": run_params.get("objective"),
            "group_split_key": run_params.get("group_split_key"),
            "challenger_mae": _float(metrics, "challenger_mae"),
            "challenger_rmse": _float(metrics, "challenger_rmse"),
            "train_rows": _int(run_params, "train_rows"),
            "val_rows": _int(run_params, "val_rows"),
            "num_features": _int(run_params, "num_features"),
        })

    packages.sort(key=lambda p: p["version"])
    return packages


def _get_model_packages_from_sagemaker(sm):
    """Fallback: read model packages from SageMaker Registry."""
    packages = []
    for status in ("Approved", "Rejected", "PendingManualApproval"):
        resp = sm.list_model_packages(
            ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus=status,
            SortBy="CreationTime",
            SortOrder="Ascending",
        )
        for pkg in resp.get("ModelPackageSummaryList", []):
            desc = sm.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
            meta = desc.get("CustomerMetadataProperties", {})
            def _float(k):
                v = meta.get(k)
                if v is None: return None
                try: return float(v)
                except ValueError: return None

            def _int(k):
                v = meta.get(k)
                if v is None: return None
                try: return int(float(v))
                except ValueError: return None

            packages.append({
                "version": desc.get("ModelPackageVersion"),
                "status": pkg["ModelApprovalStatus"],
                "created": pkg["CreationTime"].strftime("%Y-%m-%d %H:%M"),
                "trained_on": meta.get("trained_on"),
                "objective": meta.get("objective"),
                "group_split_key": meta.get("group_split_key"),
                "challenger_mae": _float("challenger_mae"),
                "challenger_rmse": _float("challenger_rmse"),
                "train_rows": _int("train_rows"),
                "val_rows": _int("val_rows"),
                "num_features": _int("num_features"),
            })
    packages.sort(key=lambda p: p["version"])
    return packages


def _get_model_packages(sm):
    """Read models from MLflow if configured, otherwise SageMaker."""
    if config.MLFLOW_TRACKING_ARN:
        try:
            return _get_model_packages_from_mlflow()
        except Exception as e:
            print(f"  MLflow read failed ({e}), falling back to SageMaker")
    return _get_model_packages_from_sagemaker(sm)


def _get_baseline(s3):
    stats = _s3_json(s3, f"{config.S3_BASELINE_PREFIX}/model_quality/output/statistics.json")
    constraints = _s3_json(s3, f"{config.S3_BASELINE_PREFIX}/model_quality/output/constraints.json")
    if not stats or not constraints:
        return None
    reg = stats.get("regression_metrics", {})
    thresholds = constraints.get("regression_constraints", {})
    return {
        "mae": reg.get("mae", {}).get("value"),
        "rmse": reg.get("rmse", {}).get("value"),
        "r2": reg.get("r2", {}).get("value"),
        "mae_threshold": thresholds.get("mae", {}).get("threshold"),
        "rmse_threshold": thresholds.get("rmse", {}).get("threshold"),
        "r2_threshold": thresholds.get("r2", {}).get("threshold"),
        "item_count": stats.get("dataset", {}).get("item_count"),
    }


def _get_monitoring(s3, year):
    prefix = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring"

    dq_viol = _s3_json(s3, f"{prefix}/data_quality/constraint_violations.json")
    if dq_viol is None:
        return None

    result = {"year": year}

    # Read monitoring metadata (which champion produced these results)
    meta = _s3_json(s3, f"{prefix}/meta.json")
    if meta:
        result["champion_version"] = meta.get("champion_version")

    violations = dq_viol.get("violations", [])
    result["dq_violations"] = [
        {
            "feature": v.get("feature_name", v.get("metric_name", "?")),
            "check": v.get("constraint_check_type", ""),
        }
        for v in violations
    ]

    mq_viol = _s3_json(s3, f"{prefix}/model_quality/constraint_violations.json")
    mq_list = mq_viol.get("violations", []) if mq_viol else []
    result["mq_violations"] = [{"metric": v.get("metric_name", "?")} for v in mq_list]

    mq_stats = _s3_json(s3, f"{prefix}/model_quality/statistics.json")
    if mq_stats:
        reg = mq_stats.get("regression_metrics", {})
        result["mae"] = reg.get("mae", {}).get("value")
        result["rmse"] = reg.get("rmse", {}).get("value")
        result["r2"] = reg.get("r2", {}).get("value")
        result["item_count"] = mq_stats.get("dataset", {}).get("item_count")

    return result


def _get_drift_features(s3, year):
    viol_data = _s3_json(s3, f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/data_quality/constraint_violations.json")
    if not viol_data:
        return []
    drifts = []
    for v in viol_data.get("violations", []):
        if v.get("constraint_check_type") == "baseline_drift_check":
            desc = v.get("description", "")
            distance = None
            if "distance:" in desc:
                try:
                    distance = float(desc.split("distance:")[1].split("exceeds")[0].strip())
                except (ValueError, IndexError):
                    pass
            drifts.append({"feature": v.get("feature_name", "?"), "distance": distance})
    drifts.sort(key=lambda d: d["distance"] or 0, reverse=True)
    return drifts


def _get_feature_means(s3, key):
    stats = _s3_json(s3, key)
    if not stats:
        return {}
    means = {}
    for f in stats.get("features", []):
        ns = f.get("numerical_statistics")
        if ns:
            means[f["name"]] = ns.get("mean")
    return means


def _get_recoveries(s3):
    recoveries = {}
    for year in config.YEARS:
        r = _s3_json(s3, f"recovery/{year}.json")
        if r:
            recoveries[str(year)] = r
    return recoveries


def main():
    os.environ.setdefault("AWS_DEFAULT_REGION", config.AWS_REGION)
    session = boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    print("Fetching model packages...")
    packages = _get_model_packages(sm)
    print(f"  {len(packages)} versions")

    print("Fetching baseline...")
    baseline = _get_baseline(s3)
    if baseline:
        print(f"  MAE threshold: {baseline['mae_threshold']:.3f}")

    print("Fetching monitoring results...")
    monitoring = {}
    for year in config.YEARS:
        m = _get_monitoring(s3, year)
        if m:
            monitoring[str(year)] = m
            print(f"  {year}: DQ={len(m['dq_violations'])} MQ={len(m['mq_violations'])} MAE={m.get('mae', '?')}")

    print("Fetching feature drift...")
    drift = {}
    for year in config.YEARS:
        d = _get_drift_features(s3, year)
        if d:
            drift[str(year)] = d
            print(f"  {year}: {len(d)} drifted features")

    print("Fetching feature means...")
    means_2021 = _get_feature_means(s3, f"{config.S3_BASELINE_PREFIX}/output/statistics.json")
    means_2022 = _get_feature_means(s3, f"{config.S3_PREDICTIONS_PREFIX}/2022/monitoring/data_quality/statistics.json")
    feature_means = {"2021": means_2021, "2022": means_2022}
    print(f"  2021: {len(means_2021)} features, 2022: {len(means_2022)} features")

    print("Fetching recovery artifacts...")
    recoveries = _get_recoveries(s3)
    print(f"  {len(recoveries)} recovery files")

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "models": packages,
        "baseline": baseline,
        "monitoring": monitoring,
        "drift": drift,
        "feature_means": feature_means,
        "recoveries": recoveries,
        "drift_threshold": DRIFT_THRESHOLD,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nData written to {OUTPUT}")


if __name__ == "__main__":
    main()
