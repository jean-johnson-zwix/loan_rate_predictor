locals {
  # Column names mirror preprocess.py output: hyphens replaced with underscores
  string_features = [
    "record_id",
    # geo keys (analysis / segmentation only)
    "state_code", "county_code", "census_tract", "derived_msa_md",
    # disparity dimensions (string)
    "derived_race", "derived_ethnicity", "derived_sex",
    # vintage
    "activity_year",
  ]

  fractional_features = [
    "event_time",   # Unix epoch seconds — designated event time
    "rate_spread",  # target
    # numeric features
    "loan_amount", "loan_to_value_ratio", "loan_term", "intro_rate_period",
    "property_value", "income", "total_units", "tract_population",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
    # disparity dimension (numeric)
    "tract_minority_population_percent",
  ]

  integral_features = [
    # categorical features — label-encoded to int by processing job
    "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "construction_method", "conforming_loan_limit", "debt_to_income_ratio",
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "derived_loan_product_type", "derived_dwelling_category",
    "negative_amortization", "interest_only_payment", "balloon_payment",
    "other_nonamortizing_features", "prepayment_penalty_term",
  ]
}

resource "aws_sagemaker_feature_group" "loan_rate_predictor" {
  feature_group_name             = "loan-rate-predictor"
  record_identifier_feature_name = "record_id"
  event_time_feature_name        = "event_time"
  role_arn                       = var.sagemaker_role_arn

  online_store_config {
    enable_online_store = true
  }

  offline_store_config {
    s3_storage_config {
      s3_uri = "s3://${var.s3_bucket}/feature-store"
    }
    disable_glue_table_creation = false
  }

  dynamic "feature_definition" {
    for_each = local.string_features
    content {
      feature_name = feature_definition.value
      feature_type = "String"
    }
  }

  dynamic "feature_definition" {
    for_each = local.fractional_features
    content {
      feature_name = feature_definition.value
      feature_type = "Fractional"
    }
  }

  dynamic "feature_definition" {
    for_each = local.integral_features
    content {
      feature_name = feature_definition.value
      feature_type = "Integral"
    }
  }
}
