locals {
  lambda_zip    = "${path.module}/../dist/pricing_lambda.zip"
  endpoint_arn  = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint/loan-rate-predictor-demo"
}

data "aws_caller_identity" "current" {}

resource "aws_iam_role" "pricing_lambda" {
  name = "loan-rate-predictor-pricing-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "pricing_lambda" {
  name = "loan-rate-predictor-pricing-lambda-policy"
  role = aws_iam_role.pricing_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sagemaker:InvokeEndpoint"
        Resource = local.endpoint_arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
    ]
  })
}

resource "aws_lambda_function" "pricing" {
  function_name = "loan-rate-predictor-pricing"
  role          = aws_iam_role.pricing_lambda.arn
  runtime       = "python3.12"
  handler       = "handler.handler"

  filename         = local.lambda_zip
  source_code_hash = filebase64sha256(local.lambda_zip)

  timeout     = 30
  memory_size = 256

  environment {
    variables = {
      MODEL_VINTAGE = var.trained_on
    }
  }
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pricing.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.pricing.execution_arn}/*/*"
}

resource "aws_apigatewayv2_api" "pricing" {
  name          = "loan-rate-predictor-pricing"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["https://${var.github_pages_origin}", "http://localhost:5500", "http://127.0.0.1:5500"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["Content-Type"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_integration" "pricing" {
  api_id                 = aws_apigatewayv2_api.pricing.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.pricing.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "price" {
  api_id    = aws_apigatewayv2_api.pricing.id
  route_key = "POST /price"
  target    = "integrations/${aws_apigatewayv2_integration.pricing.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.pricing.id
  name        = "$default"
  auto_deploy = true
}

output "pricing_api_url" {
  value       = aws_apigatewayv2_stage.default.invoke_url
  description = "Base URL for the pricing API. POST to {pricing_api_url}/price"
}
