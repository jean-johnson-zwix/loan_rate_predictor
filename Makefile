include .env
export

PYTHONPATH := src

.PHONY: data upload-raw preprocess tf-init tf-plan tf-apply run-job test

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

test:
	pytest tests/
