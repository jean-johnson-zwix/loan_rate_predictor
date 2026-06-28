resource "aws_sagemaker_model_package_group" "loan_rate_predictor" {
  model_package_group_name        = "loan-rate-predictor"
  model_package_group_description = "XGBoost models for rate_spread prediction, versioned by training vintage"

  tags = {
    Project = "loan-rate-predictor"
  }
}
