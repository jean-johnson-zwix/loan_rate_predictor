include .env
export

PYTHONPATH := src

.PHONY: data upload-raw preprocess tf-init tf-plan tf-apply run-preprocessing run-pipeline make-baseline test deploy invoke

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

test:
	pytest tests/

# requires: model_package_arn set in infra/terraform.auto.tfvars
deploy:
	terraform -chdir=infra apply

# requires: endpoint deployed via make deploy
invoke:
	aws sagemaker-runtime invoke-endpoint \
		--endpoint-name loan-rate-predictor-demo \
		--content-type text/csv \
		--body "$$(cat data/sample_payload.csv)" \
		--profile $(AWS_PROFILE) \
		/tmp/sagemaker_output.json && cat /tmp/sagemaker_output.json
