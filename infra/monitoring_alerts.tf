resource "aws_sns_topic" "monitoring_alerts" {
  name = "loan-rate-predictor-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.monitoring_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "data_quality" {
  alarm_name        = "loan-rate-predictor-data-quality-drift"
  alarm_description = "INPUT DRIFT: feature distributions for a scored year shifted vs 2021 baseline. Check predictions/<year>/monitoring/data_quality/constraint_violations.json. Investigate before retraining."
  namespace         = "LoanRatePredictor/Monitoring"
  metric_name       = "DataQualityViolations"
  # No dimensions block — matches the undimensioned metric published by drift_report.py.
  # Dimensioned (per-year) metrics are published alongside for dashboards but cannot
  # drive an alarm without knowing the year at Terraform apply time.
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.monitoring_alerts.arn]
  ok_actions    = [aws_sns_topic.monitoring_alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "model_quality" {
  alarm_name        = "loan-rate-predictor-model-quality-degradation"
  alarm_description = "PERF DEGRADATION: frozen champion MAE exceeded 25pct threshold vs 2021 baseline (0.248). Check predictions/<year>/monitoring/model_quality/constraint_violations.json for computed MAE. Retrain: make run-pipeline DATA_YEAR=<year> (see alarm timestamp for year)."
  namespace         = "LoanRatePredictor/Monitoring"
  metric_name       = "ModelQualityViolations"
  # No dimensions block — see data_quality alarm comment above (per-year).
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.monitoring_alerts.arn]
  ok_actions    = [aws_sns_topic.monitoring_alerts.arn]
}
