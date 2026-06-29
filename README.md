# loan_rate_predictor

AWS-managed MLOps project predicting mortgage `rate_spread` (APR - APOR) on Arizona HMDA data (2021-2024).

Two surfaces: a **synchronous pricing API** (borrower gets a rate estimate) and an **ops CLI** (engineer keeps the estimate accurate as the market moves).

**Live UI:** https://jean-johnson-zwix.github.io/loan_rate_predictor/

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

Static dashboard (`ui/ops-dashboard/ops.html`) reads from a generated JSON file. Four zones: status, champion timeline, accuracy over time, per-vintage drill-down. Deployed to GitHub Pages.

## Model metrics (2021 champion)

| | RMSE | MAE |
|---|---|---|
| XGBoost (28 features, Bayesian AMT) | 0.339 | 0.248 |
| Mean predictor baseline | 0.537 | 0.389 |
| Best single-feature linear | 0.519 | 0.376 |

-36% MAE vs mean baseline. Val set group-split on lender (`lei`).

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
