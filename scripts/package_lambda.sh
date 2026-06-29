#!/bin/bash
# Package the pricing Lambda. No pip install — stdlib + boto3 (provided by Lambda runtime).
# Output: dist/pricing_lambda.zip
set -euo pipefail

DIST=dist
STAGE=$(mktemp -d)

cp src/loan_rate_predictor/pricing/handler.py "$STAGE/handler.py"
cp src/loan_rate_predictor/config.py "$STAGE/config.py"
cp data/processed/categorical_encodings.json "$STAGE/categorical_encodings.json"

mkdir -p "$DIST"
(cd "$STAGE" && zip -r - .) > "$DIST/pricing_lambda.zip"
rm -rf "$STAGE"

echo "Created $DIST/pricing_lambda.zip ($(du -sh $DIST/pricing_lambda.zip | cut -f1))"
