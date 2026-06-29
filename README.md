# loan_rate_predictor

AWS-managed MLOps project predicting mortgage `rate_spread` (APR - APOR) on Arizona HMDA data (2021-2024).

Two surfaces: a **synchronous pricing API** (borrower gets a rate estimate) and an **ops CLI** (engineer keeps the estimate accurate as the market moves).

**Pricing UI:** https://jean-johnson-zwix.github.io/loan_rate_predictor/ui/user-dashboard/

**Ops Dashboard:** https://jean-johnson-zwix.github.io/loan_rate_predictor/ui/ops-dashboard/

## Stack

| Layer | Choice |
|---|---|
| Model | SageMaker built-in XGBoost |
| Tuning | SageMaker AMT (Bayesian) |
| Registry | SageMaker Model Registry |
| Experiment tracking | SageMaker managed MLflow |
| Monitoring | Evidently (data drift + model quality) |
| Alerting | CloudWatch alarms + SNS |
| Infra | Terraform |

## Prerequisites

- Python 3.11+
- Terraform >= 1.6
- AWS CLI with profile `loan-rate-predictor-local-developer`
- `.env` file with `STORAGE_BUCKET_NAME`, `SAGEMAKER_ROLE_ARN`, and `MLFLOW_TRACKING_ARN`

## Quick start

```bash
make data                        # download AZ HMDA CSVs -> data/raw/
make upload-raw                  # sync data/raw/ -> S3
make run-preprocessing           # submit Processing job to SageMaker
make tf-init                     # bootstrap Terraform (once)
make tf-apply                    # apply infra (endpoint, alerts, MLflow server)
make run-pipeline DATA_YEAR=2021 # bootstrap training run (AMT -> evaluate -> register)
make invoke                      # send sample payload to serverless endpoint
```

## Monitoring

```bash
make monitor YEAR=2022           # score -> join -> Evidently drift + quality reports -> CloudWatch -> SNS
```

The frozen champion runs forward against new vintages. Evidently generates per-feature drift reports (HTML + JSON) and model quality metrics. When degradation exceeds thresholds, CloudWatch alarms fire with actionable SNS emails.

## Retraining

```bash
make retrain DATA_YEAR=2023          # start pipeline (async)
# wait for pipeline to succeed...
make evaluate-retrain DATA_YEAR=2023 # promote -> deploy -> measure recovery
```

The pipeline trains a challenger on the new year (with per-vintage winsorize bounds), evaluates against the frozen champion, and registers if the challenger wins. `evaluate-retrain` promotes, deploys to the endpoint, and measures recovery on the held-out eval slice.

## Ops Dashboard

```bash
make ops-report                  # generate ui/ops-dashboard/ops-data.json from AWS artifacts
```

Static dashboard (`ui/ops-dashboard/`) reads from a generated JSON file. Four zones: status, champion timeline, accuracy over time (eval slice), and per-year drill-down with Evidently report links. Deployed to GitHub Pages.

## Model metrics

| Champion | Trained On | Val MAE | Val RMSE | Recovery |
|----------|-----------|---------|----------|----------|
| v1 | HMDA 2021 | 0.248 | 0.339 | baseline |
| v2 | HMDA 2022 | 0.423 | — | +0.086 (17%) |
| v3 | HMDA 2023 | 0.546 | — | +0.090 (14%) |

Absolute MAE rises across vintages (target std grew 50%: 0.61 -> 0.90). Recovery is gap-closed vs frozen champion on held-out eval slice, not return to baseline. 28 features, Bayesian AMT, group-split on lender (`lei`).

## Pricing API

```bash
make package-lambda              # build dist/pricing_lambda.zip
make deploy-champion             # promote -> update tfvars -> tf-apply (Lambda + API Gateway)
```

`POST /price` takes a JSON loan payload, applies frozen transforms, invokes the serverless endpoint, and returns `{ rate_spread, indicative_apr, trained_on }`.

## Serving

| Workload | Mode |
|----------|------|
| Year scoring (ops) | Batch transform |
| Single prediction (pricing API) | Lambda -> serverless endpoint |
