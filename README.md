# loan_rate_predictor

AWS-managed MLOps project predicting mortgage `rate_spread` (APR − APOR) on Arizona HMDA data (2021–2024).

## Stack

| Layer | Choice |
|---|---|
| Model | SageMaker built-in XGBoost |
| Tuning | SageMaker AMT (Bayesian) |
| Registry | SageMaker Model Registry |
| Infra | Terraform |
| CI/CD | GitHub Actions → SageMaker |

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
make tf-apply                    # apply infra (Feature Group, Model Registry, etc.)
make run-pipeline DATA_YEAR=2021 # bootstrap training run (AMT → evaluate → register)
make make-baseline               # create Model Monitor data-quality baseline
make invoke                      # send sample payload to serverless endpoint
make test                        # run tests
```

## Serving

| Workload | Mode |
|----------|------|
| Run-forward vintage scoring | Batch transform |
| Demo single-quote prediction | Serverless endpoint |

Invoke via AWS CLI (`make invoke`) or Postman with AWS SigV4 auth.

## Model metrics (2021 champion)

| | RMSE | MAE |
|---|---|---|
| XGBoost (28 features, Bayesian AMT) | 0.339 | 0.248 |
| Mean predictor baseline | 0.537 | 0.389 |
| Best single-feature linear | 0.519 | 0.376 |

−36% MAE vs mean baseline · −34% MAE vs best linear · val set group-split on lender (`lei`)
