variable "s3_bucket" {
  type        = string
  description = "S3 bucket for Feature Store offline store and Terraform state"
}

variable "sagemaker_role_arn" {
  type        = string
  description = "IAM role ARN that SageMaker uses to access S3 and Feature Store"
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "model_package_arn" {
  type        = string
  description = "ARN of the approved model package to deploy (from Model Registry)"
}
