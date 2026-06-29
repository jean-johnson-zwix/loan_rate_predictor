include .env
export

PYTHONPATH := src

.PHONY: data upload-raw preprocess tf-init tf-plan tf-apply run-preprocessing run-pipeline make-baseline make-model-quality-baseline predict monitor test invoke package-lambda deploy-champion

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

# requires: make tf-apply (Model Package Group must exist before pipeline registers a model)
run-pipeline:
	@echo "Usage: make run-pipeline DATA_YEAR=2021"
	python scripts/run_pipeline.py --data-year $(DATA_YEAR)

make-baseline:
	python -m loan_rate_predictor.monitoring.make_baseline

make-model-quality-baseline:
	python -m loan_rate_predictor.monitoring.make_model_quality_baseline

predict:
	@echo "Usage: make predict YEAR=2022"
	python scripts/predict_with_champion.py --year $(YEAR)

monitor:
	@echo "Usage: make monitor YEAR=2022"
	python scripts/run_monitoring.py --year $(YEAR)

test:
	pytest tests/

# promotes champion to pricing: reads trained_on tag from registry, updates tfvars, applies terraform
# run after make run-pipeline promotes a new champion
deploy-champion:
	python scripts/deploy_champion.py
	terraform -chdir=infra apply

# requires: data/processed/categorical_encodings.json (make preprocess or make run-preprocessing)
package-lambda:
	bash scripts/package_lambda.sh

# requires: endpoint deployed via make tf-apply
invoke:
	aws sagemaker-runtime invoke-endpoint \
		--endpoint-name loan-rate-predictor-demo \
		--content-type text/csv \
		--body fileb://data/sample_payload.csv \
		--profile $(AWS_PROFILE) \
		sagemaker_output.json && type sagemaker_output.json
