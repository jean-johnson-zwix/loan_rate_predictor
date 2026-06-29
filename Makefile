include .env
export

PYTHONPATH := src

.PHONY: data upload-raw preprocess tf-init tf-plan tf-apply run-preprocessing run-pipeline predict monitor test invoke package-lambda deploy-champion measure-recovery retrain evaluate-retrain ops-report

data:
	python scripts/download_hmda.py

upload-raw:
	aws s3 sync data/raw/ s3://$(STORAGE_BUCKET_NAME)/raw/ --profile $(AWS_PROFILE)

preprocess:
	python src/loan_rate_predictor/processing/preprocess.py

tf-init:
	terraform -chdir=infra init

tf-plan:
	terraform -chdir=infra plan

tf-apply:
	terraform -chdir=infra apply

run-preprocessing:
	python scripts/run_processing_job.py

run-pipeline:
	@echo "Usage: make run-pipeline DATA_YEAR=2021"
	python scripts/run_pipeline.py --data-year $(DATA_YEAR)

predict:
	@echo "Usage: make predict YEAR=2022"
	python scripts/predict_with_champion.py --year $(YEAR)

# score year with champion, join labels, run Evidently drift + quality reports
monitor:
	@echo "Usage: make monitor YEAR=2022"
	python scripts/run_monitoring.py --year $(YEAR)

test:
	pytest tests/

# promotes latest pending model, updates tfvars, deploys endpoint + Lambda
deploy-champion:
	python scripts/deploy_champion.py
	terraform -chdir=infra apply

package-lambda:
	bash scripts/package_lambda.sh

measure-recovery:
	@echo "Usage: make measure-recovery YEAR=2022"
	python scripts/measure_recovery.py --year $(YEAR)

# step 1: start retrain pipeline (async)
retrain:
	@echo "Usage: make retrain DATA_YEAR=2022"
	python scripts/run_pipeline.py --data-year $(DATA_YEAR)

# step 2: after pipeline succeeds — deploy new champion + measure recovery
evaluate-retrain:
	@echo "Usage: make evaluate-retrain DATA_YEAR=2022"
	python scripts/deploy_champion.py
	terraform -chdir=infra apply
	python scripts/measure_recovery.py --year $(DATA_YEAR)

# generates ops-dashboard/ops-data.json from Registry + S3 artifacts
ops-report:
	python scripts/generate_ops_report.py

invoke:
	aws sagemaker-runtime invoke-endpoint \
		--endpoint-name loan-rate-predictor-demo \
		--content-type text/csv \
		--body fileb://data/sample_payload.csv \
		--profile $(AWS_PROFILE) \
		sagemaker_output.json && type sagemaker_output.json
