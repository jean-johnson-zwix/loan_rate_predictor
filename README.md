# loan_rate_predictor

AWS-managed MLOps project predicting mortgage `rate_spread` (APR − APOR) on Arizona HMDA data (2021–2024). The managed-AWS counterpart to the [`six-eyes`](https://github.com/your-org/six-eyes) OSS project — the contrast is the point.

## Stack

| Layer | Choice |
|---|---|
| Model | SageMaker built-in XGBoost |
| Tuning | SageMaker AMT (Bayesian/Hyperband) |
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
make data              # download AZ HMDA CSVs → data/raw/
make upload-raw        # sync data/raw/ → S3
make preprocess        # preprocess locally → data/processed/
make run-preprocessing # submit Processing job to SageMaker
make tf-init           # bootstrap Terraform (once)
make tf-plan           # preview infra changes
make tf-apply          # apply infra (Feature Group, etc.)
make test              # run tests
```
