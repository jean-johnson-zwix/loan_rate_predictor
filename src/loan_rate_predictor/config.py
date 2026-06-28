"""Single source of truth for loan_rate_predictor — features, exclusions, segments, thresholds."""
import os

TARGET = "rate_spread"

NUMERIC_FEATURES = [
    "loan_amount",
    "loan_to_value_ratio",
    "loan_term",
    "intro_rate_period",
    "property_value",
    "income",
    "total_units",
    "tract_population",
    "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

CATEGORICAL_FEATURES = [
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "construction_method",
    "conforming_loan_limit",
    "debt_to_income_ratio",  # → DTI_ORDINAL at preprocessing
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "derived_loan_product_type",
    "derived_dwelling_category",
    "negative_amortization",
    "interest_only_payment",
    "balloon_payment",
    "other_nonamortizing_features",
    "prepayment_penalty_term",  # ponytail: kept as feature (product term); move to LEAKAGE_FORBIDDEN if deemed priced
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Geography — segment keys only, never model inputs
GEO_KEYS = ["state_code", "county_code", "census_tract", "derived_msa-md"]

LEAKAGE_FORBIDDEN = [
    "interest_rate",
    "rate_spread",
    "total_loan_costs",
    "total_points_and_fees",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "hoepa_status",
]

PROTECTED_FORBIDDEN = [
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_race-1", "applicant_race-2", "applicant_race-3",
    "applicant_race-4", "applicant_race-5",
    "applicant_race_observed",
    "applicant_ethnicity-1", "applicant_ethnicity-2", "applicant_ethnicity-3",
    "applicant_ethnicity-4", "applicant_ethnicity-5",
    "applicant_ethnicity_observed",
    "applicant_sex",
    "applicant_sex_observed",
    "applicant_age",
    "applicant_age_above_62",
    "co-applicant_race-1", "co-applicant_race-2", "co-applicant_race-3",
    "co-applicant_race-4", "co-applicant_race-5",
    "co-applicant_race_observed",
    "co-applicant_ethnicity-1", "co-applicant_ethnicity-2", "co-applicant_ethnicity-3",
    "co-applicant_ethnicity-4", "co-applicant_ethnicity-5",
    "co-applicant_ethnicity_observed",
    "co-applicant_sex",
    "co-applicant_sex_observed",
    "co-applicant_age",
    "co-applicant_age_above_62",
    "tract_minority_population_percent",
]

DISPARITY_DIMENSIONS = [
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "tract_minority_population_percent",
]

FORBIDDEN = LEAKAGE_FORBIDDEN + PROTECTED_FORBIDDEN

OPERATIONAL_EXCLUDED = [
    "activity_year",
    "lei",
    "action_taken",
    "purchaser_type",
    "aus-1", "aus-2", "aus-3", "aus-4", "aus-5",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "applicant_credit_score_type",
    "co-applicant_credit_score_type",
    "preapproval",
    "submission_of_application",
    "initially_payable_to_institution",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "multifamily_affordable_units",
]

DTI_ORDINAL: dict[str, int] = {
    "<20%": 0,
    "20%-<30%": 1,
    "30%-<36%": 2,
    **{str(v): i + 3 for i, v in enumerate(range(36, 50))},  # "36"→3 … "49"→16
    "50%-60%": 17,
    ">60%": 18,
}
DTI_MISSING = ["", "NA", "Exempt", "exempt"]

STATE = "AZ"
YEARS = [2021, 2022, 2023, 2024]
TRAIN_YEAR = 2021

EVENT_TIME = "activity_year"
RECORD_ID = "record_id"

SENTINELS = ["", "NA", "Exempt", "exempt", "1111"]

FILTERS = {
    "action_taken": [1],
    "lien_status": [1],
    "business_or_commercial_purpose": [2],
    "reverse_mortgage": [2],
    "open-end_line_of_credit": [2],
}

SEGMENTS = {
    "loan_purpose": {
        "field": "loan_purpose",
        "groups": {
            "purchase": [1],
            "refinance": [31, 32],
            "other": [2, 4, 5],
        },
    },
    "loan_type": {
        "field": "loan_type",
        "groups": {
            "conventional": [1],
            "fha": [2],
            "va": [3],
            "rhs_fsa": [4],
        },
    },
    "tract": {
        "field": "county_code",
        "top_n": 8,
    },
}

PSI_SIGNIFICANT = 0.25
PERF_MATERIAL_RELATIVE = 0.25
SEGMENT_MATERIAL_RELATIVE = 0.40
SEGMENT_MIN_VOLUME_SHARE = 0.10
CALIBRATION_BIAS_SD_FRAC = 0.25

COMPLETENESS_GATE = {
    "min_rate_spread_fill": 0.60,
    "min_rows_per_year": 3000,
    "min_rows_per_segment": 300,
}

MONITORED_SIGNAL = "aggregate"  # set by 00_gate_distribution.ipynb

# AWS / S3
S3_BUCKET = os.getenv("STORAGE_BUCKET_NAME", "loan-rate-predictor-storage")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "loan-rate-predictor-local-developer")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
S3_RAW_PREFIX = "raw"
S3_PROCESSED_PREFIX = "processed"
FEATURE_GROUP_NAME = "loan-rate-predictor"

S3_TRAINING_PREFIX = "training"
S3_MODEL_PREFIX = "models"
S3_BASELINE_PREFIX = "baseline"
MODEL_PACKAGE_GROUP_NAME = "loan-rate-predictor"

# Percentile fractions — actual clip bounds computed from TRAIN_YEAR data at preprocessing time
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

XGBOOST_OBJECTIVE = "reg:squarederror"
EVAL_METRIC = "rmse"
PROMOTION_METRIC = "mae"
GROUP_SPLIT_KEY = "lei"
VAL_FRACTION = 0.20

AMT_MAX_JOBS = 20
AMT_MAX_PARALLEL = 4
AMT_STRATEGY = "Bayesian"

AMT_HYPERPARAMETER_RANGES = {
    "num_round": {"min": 50, "max": 500, "type": "Integer"},
    "max_depth": {"min": 3, "max": 10, "type": "Integer"},
    "eta": {"min": 0.01, "max": 0.3, "type": "Continuous"},
    "subsample": {"min": 0.5, "max": 1.0, "type": "Continuous"},
    "colsample_bytree": {"min": 0.5, "max": 1.0, "type": "Continuous"},
    "min_child_weight": {"min": 1, "max": 10, "type": "Integer"},
    "gamma": {"min": 0.0, "max": 5.0, "type": "Continuous"},
    "alpha": {"min": 0.0, "max": 5.0, "type": "Continuous"},
    "lambda": {"min": 0.0, "max": 5.0, "type": "Continuous"},
}
