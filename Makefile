include .env
export

PYTHONPATH := src

.PHONY: data upload-raw preprocess tf-init tf-plan tf-apply run-preprocessing run-pipeline make-baseline test

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
