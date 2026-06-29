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

variable "alert_email" {
  type        = string
  description = "Email address for monitoring alert notifications"
}

variable "trained_on" {
  type        = string
  description = "Training year of the current champion (e.g. '2021'). Set alongside model_package_arn."
  default     = "2021"
}

variable "github_pages_origin" {
  type        = string
  description = "GitHub Pages hostname for CORS (e.g. 'your-username.github.io')"
}
