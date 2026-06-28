terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # ponytail: S3 backend, no DynamoDB lock — add lock table if concurrent applies become an issue
  backend "s3" {
    bucket  = "loan-rate-predictor-storage"
    key     = "terraform/loan-rate-predictor.tfstate"
    region  = "us-east-1"
    profile = "loan-rate-predictor-terraform"
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "loan-rate-predictor-terraform"
}
