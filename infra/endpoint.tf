resource "aws_sagemaker_model" "champion" {
  name               = "loan-rate-predictor-champion"
  execution_role_arn = var.sagemaker_role_arn

  container {
    model_package_name = var.model_package_arn
  }
}

resource "aws_sagemaker_endpoint_configuration" "serverless" {
  name = "loan-rate-predictor-serverless"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.champion.name

    serverless_config {
      memory_size_in_mb = 2048
      max_concurrency   = 1
    }
  }
}

resource "aws_sagemaker_endpoint" "demo" {
  name                 = "loan-rate-predictor-demo"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.serverless.name
}
