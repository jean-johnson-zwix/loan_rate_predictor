.DELETE_ON_ERROR:
.SUFFIXES:

include .env
export

PYTHONPATH := src
PROJECT := loan-rate-predictor

.PHONY: all data upload-raw preprocess run-preprocessing \
        tf-init tf-plan tf-apply \
        run-pipeline retrain evaluate-retrain \
        predict monitor measure-recovery \
        deploy-champion package-lambda invoke \
        test ops-report help

## Default target
all: help

## Download AZ HMDA CSVs -> data/raw/
data:
	python scripts/download_hmda.py

## Sync data/raw/ -> S3
upload-raw:
	aws s3 sync data/raw/ "s3://$(STORAGE_BUCKET_NAME)/raw/" --profile "$(AWS_PROFILE)"

## Preprocess locally (set WINSORIZE_YEAR to override bounds)
preprocess:
	python src/loan_rate_predictor/processing/preprocess.py

## Submit Processing job to SageMaker
run-preprocessing:
	python scripts/run_processing_job.py

## Bootstrap Terraform (once)
tf-init:
	terraform -chdir=infra init

## Preview infra changes
tf-plan:
	terraform -chdir=infra plan

## Apply infra (endpoint, alerts, MLflow server)
tf-apply:
	terraform -chdir=infra apply

## Start training pipeline (async). Usage: make run-pipeline DATA_YEAR=2021
run-pipeline:
	python scripts/run_pipeline.py --data-year "$(DATA_YEAR)"

## Start retrain pipeline (async). Usage: make retrain DATA_YEAR=2022
retrain:
	python scripts/run_pipeline.py --data-year "$(DATA_YEAR)"

## After pipeline succeeds: promote + deploy + measure recovery. Usage: make evaluate-retrain DATA_YEAR=2022
evaluate-retrain:
	python scripts/deploy_champion.py
	terraform -chdir=infra apply
	python scripts/measure_recovery.py --year "$(DATA_YEAR)"

## Score a year with frozen champion. Usage: make predict YEAR=2022
predict:
	python scripts/predict_with_champion.py --year "$(YEAR)"

## Score -> join -> Evidently drift + quality reports -> CloudWatch -> SNS. Usage: make monitor YEAR=2022
monitor:
	python scripts/run_monitoring.py --year "$(YEAR)"

## Measure recovery on held-out eval slice. Usage: make measure-recovery YEAR=2022
measure-recovery:
	python scripts/measure_recovery.py --year "$(YEAR)"

## Promote latest pending model, update tfvars, deploy endpoint + Lambda
deploy-champion:
	python scripts/deploy_champion.py
	terraform -chdir=infra apply

## Build dist/pricing_lambda.zip
package-lambda:
	bash scripts/package_lambda.sh

## Send sample payload to serverless endpoint
invoke:
	aws sagemaker-runtime invoke-endpoint \
		--endpoint-name "$(PROJECT)-demo" \
		--content-type text/csv \
		--body fileb://data/sample_payload.csv \
		--profile "$(AWS_PROFILE)" \
		sagemaker_output.json && type sagemaker_output.json

## Run pytest
test:
	pytest tests/

## Generate ui/ops-dashboard/ops-data.json from Registry + S3 artifacts
ops-report:
	python scripts/generate_ops_report.py

## Show available targets
help:
	@echo "$(PROJECT) - MLOps pipeline for AZ HMDA rate_spread prediction"
	@echo ""
	@awk '/^## /{desc=substr($$0,4)} /^[a-zA-Z_-]+:/{if(desc){printf "  %-22s %s\n",$$1,desc; desc=""}}' $(firstword $(MAKEFILE_LIST))
