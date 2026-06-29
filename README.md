# loan_rate_predictor

AWS-managed MLOps project predicting mortgage `rate_spread` (APR − APOR) on Arizona HMDA data (2021–2024). The managed-AWS counterpart to the [`six-eyes`](https://github.com/jeanj) open-source project.

Two surfaces: a **synchronous pricing API** (borrower gets a rate estimate) and an **ops CLI** (engineer keeps the estimate accurate as the market moves).

**Live UI:** https://jean-johnson-zwix.github.io/loan_rate_predictor/

## Stack

| Layer | Choice |
|---|---|
| Model | SageMaker built-in XGBoost |
| Tuning | SageMaker AMT (Bayesian) |
| Registry | SageMaker Model Registry |
| Monitoring | Model Monitor (on-trigger Processing jobs) |
| Infra | Terraform |

## Prerequisites

- Python 3.11+
- Terraform ≥ 1.6
- AWS CLI with profile `loan-rate-predictor-local-developer`
- `.env` file with `STORAGE_BUCKET_NAME` and `SAGEMAKER_ROLE_ARN`

## Quick start

```bash
make data                        # download AZ HMDA CSVs → data/raw/
make upload-raw                  # sync data/raw/ → S3
make run-preprocessing           # submit Processing job to SageMaker
make tf-init                     # bootstrap Terraform (once)
make tf-apply                    # apply infra
make run-pipeline DATA_YEAR=2021 # bootstrap training run (AMT → evaluate → register)
make make-baseline               # data-quality baseline
make make-model-quality-baseline # model-quality baseline (MAE/RMSE thresholds)
make invoke                      # send sample payload to serverless endpoint
```

## Monitoring

```bash
make predict YEAR=2022           # score a year with frozen champion
make monitor YEAR=2022           # full chain: score → join → monitors A+B → CloudWatch → SNS
```

The frozen 2021 champion runs forward against new vintages without retraining. When the monitor detects degradation, it fires a CloudWatch alarm with an actionable SNS email.

| Year | Data-quality | Model-quality | MAE |
|------|-------------|--------------|-----|
| 2021 (control) | 3 violations | 0 violations | 0.225 |
| 2022 (earned detection) | 5 violations | 4 violations | 0.414 |

2022 MAE 0.414 = 67% degradation vs baseline 0.248 → both alarms fired.

## Model metrics (2021 champion)

| | RMSE | MAE |
|---|---|---|
| XGBoost (28 features, Bayesian AMT) | 0.339 | 0.248 |
| Mean predictor baseline | 0.537 | 0.389 |
| Best single-feature linear | 0.519 | 0.376 |

−36% MAE vs mean baseline · −34% MAE vs best linear · val set group-split on lender (`lei`)

## Pricing API

```bash
make package-lambda              # build dist/pricing_lambda.zip
make deploy-champion             # resolve champion from registry → tf-apply (Lambda + API Gateway)
```

`POST /price` takes a JSON loan payload, applies frozen transforms (DTI ordinal + categorical label encoding), invokes the serverless endpoint, and returns `{ rate_spread, indicative_apr, trained_on }`.

The frontend (`frontend/index.html`) is a single-file form — dark editorial design, no build step, no dependencies.

## Serving

| Workload | Mode |
|----------|------|
| Year scoring (ops) | Batch transform |
| Single prediction (pricing API) | Lambda → serverless endpoint |
