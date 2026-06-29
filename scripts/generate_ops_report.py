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

OUTPUT = Path(__file__).resolve().parent.parent / "ui/ops-dashboard" / "ops-data.json"
DRIFT_THRESHOLD = 0.1


def _s3_json(s3, key):
    try:
        resp = s3.get_object(Bucket=config.S3_BUCKET, Key=key)
        body = resp["Body"].read()
        if not body or not body.strip():
            return None
        return json.loads(body)
    except (s3.exceptions.NoSuchKey, json.JSONDecodeError):
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

        sm_arn = tags.get("sagemaker_arn")
        packages.append({
            "version": int(mv.version),
            "_sagemaker_arn": sm_arn,  # internal, stripped before JSON output
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
    """Read monitoring results from Evidently report.json files."""
    prefix = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring"

    # Check if any report exists for this year
    dq_report = _s3_json(s3, f"{prefix}/data_quality/report.json")
    mq_report = _s3_json(s3, f"{prefix}/model_quality/report.json")
    if dq_report is None and mq_report is None:
        return None

    result = {"year": year}

    # Read monitoring metadata (which champion produced these results)
    meta = _s3_json(s3, f"{prefix}/meta.json")
    if meta:
        result["champion_version"] = meta.get("champion_version")

    # Data drift violations from Evidently report
    result["dq_violations"] = []
    if dq_report:
        for t in dq_report.get("tests", []):
            if t.get("status") == "FAIL":
                name = t.get("name", "")
                col = name.replace("Value Drift for column ", "") if "column" in name else name
                result["dq_violations"].append({
                    "feature": col,
                    "check": "drift",
                    "description": t.get("description", ""),
                })

    # Model quality metrics from Evidently report
    result["mq_violations"] = []
    if mq_report:
        for m in mq_report.get("metrics", []):
            name = m.get("metric_name", "")
            val = m.get("value")
            if "MAE" in name and isinstance(val, dict):
                result["mae"] = val.get("mean")
            elif "MAE" in name and isinstance(val, (int, float)):
                result["mae"] = val
            elif "RMSE" in name and isinstance(val, (int, float)):
                result["rmse"] = val
            elif "R2Score" in name and isinstance(val, (int, float)):
                result["r2"] = val

        # Check if MAE exceeds threshold
        baseline_mae_threshold = 0.248 * (1 + config.MODEL_QUALITY_DEGRADATION_THRESHOLD)
        if result.get("mae") and result["mae"] > baseline_mae_threshold:
            result["mq_violations"].append({"metric": "mae"})

    return result


def _get_drift_features(s3, year):
    """Extract per-feature drift scores from Evidently data_quality report.json."""
    report = _s3_json(s3, f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/data_quality/report.json")
    if not report:
        return []

    # Get drifted features from failed tests (only features that exceeded threshold)
    drifts = []
    for t in report.get("tests", []):
        if t.get("status") != "FAIL" or "column" not in t.get("name", ""):
            continue
        name = t.get("name", "")
        col = name.replace("Value Drift for column ", "") if "column" in name else name
        desc = t.get("description", "")
        distance = None
        if "Drift score is" in desc:
            try:
                distance = float(desc.split("Drift score is")[1].split(".")[0:2].__repr__())
            except (ValueError, IndexError):
                pass
        # Fallback: find the matching metric for the exact score
        if distance is None:
            for m in report.get("metrics", []):
                mname = m.get("metric_name", "")
                if f"column={col}" in mname and isinstance(m.get("value"), (int, float)):
                    distance = m["value"]
                    break
        drifts.append({"feature": col, "distance": distance})

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

    # Build SageMaker -> MLflow version lookup
    sm_to_mlflow = {}
    for m in packages:
        arn = m.get("_sagemaker_arn")
        if arn:
            sm_version = arn.split("/")[-1]
            sm_to_mlflow[int(sm_version)] = m["version"]
            sm_to_mlflow[arn] = m["version"]

    # Translate monitoring champion_version from SageMaker to MLflow
    for year, m in monitoring.items():
        sm_ver = m.get("champion_version")
        if sm_ver and sm_ver in sm_to_mlflow:
            m["champion_version"] = sm_to_mlflow[sm_ver]

    # Translate recovery ARNs to MLflow versions
    for year, r in recoveries.items():
        frozen_arn = r.get("frozen_champion_arn", "")
        new_arn = r.get("new_champion_arn", "")
        if frozen_arn in sm_to_mlflow:
            r["frozen_version"] = sm_to_mlflow[frozen_arn]
        elif year in monitoring and monitoring[year].get("champion_version"):
            r["frozen_version"] = monitoring[year]["champion_version"]
        if new_arn in sm_to_mlflow:
            r["new_version"] = sm_to_mlflow[new_arn]

    print("Downloading Evidently reports...")
    reports_dir = OUTPUT.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_links = {}
    for year in config.YEARS:
        year_reports = {}
        for report_type in ("data_quality", "model_quality"):
            key = f"{config.S3_PREDICTIONS_PREFIX}/{year}/monitoring/{report_type}/report.html"
            local_name = f"{year}_{report_type}.html"
            local_path = reports_dir / local_name
            try:
                s3.download_file(config.S3_BUCKET, key, str(local_path))
                year_reports[report_type] = f"reports/{local_name}"
                print(f"  {year}/{report_type} -> {local_path.name}")
            except s3.exceptions.ClientError:
                pass
        if year_reports:
            report_links[str(year)] = year_reports
    print(f"  {sum(len(v) for v in report_links.values())} reports downloaded")

    # Strip internal fields and sensitive ARNs before output
    clean_packages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in packages]
    for r in recoveries.values():
        r.pop("frozen_champion_arn", None)
        r.pop("new_champion_arn", None)

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "models": clean_packages,
        "baseline": baseline,
        "monitoring": monitoring,
        "drift": drift,
        "feature_means": feature_means,
        "recoveries": recoveries,
        "report_links": report_links,
        "drift_threshold": DRIFT_THRESHOLD,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nData written to {OUTPUT}")


if __name__ == "__main__":
    main()
