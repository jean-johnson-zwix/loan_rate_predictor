resource "aws_sagemaker_mlflow_tracking_server" "main" {
  tracking_server_name = "loan-rate-predictor"
  role_arn             = var.sagemaker_role_arn
  artifact_store_uri   = "s3://${var.s3_bucket}/mlflow"
  tracking_server_size = "Small"

  tags = {
    Project = "loan-rate-predictor"
  }
}

output "mlflow_tracking_arn" {
  value = aws_sagemaker_mlflow_tracking_server.main.arn
}
