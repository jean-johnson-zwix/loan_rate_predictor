"""SageMaker Pipeline DAG: prepare → AMT → evaluate → promote? → register.

DATA_YEAR-parameterized: same definition handles bootstrap (2021, no champion)
and retrains (2022+, challenger vs champion). The ConditionStep promotion gate
is the spine.
"""
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.parameter import ContinuousParameter, IntegerParameter
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TuningStep, CacheConfig

from loan_rate_predictor import config

XGBOOST_VERSION = "1.7-1"
SKLEARN_VERSION = "1.2-1"
INSTANCE_TYPE = "ml.m5.xlarge"
PIPELINE_NAME = "loan-rate-training"


def _build_pipeline(role_arn: str, boto_session,
                    source_s3_uri: str) -> Pipeline:
    """Construct the full training pipeline definition."""
    sm_session = PipelineSession(boto_session=boto_session)

    param_data_year = ParameterString(name="DataYear", default_value=str(config.TRAIN_YEAR))
    param_champion_uri = ParameterString(name="ChampionModelUri", default_value="NONE")

    # [1] Prepare: split processed.csv → train/val XGBoost CSVs
    prepare_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        role=role_arn,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=sm_session,
    )

    step_prepare = ProcessingStep(
        name="Prepare",
        processor=prepare_processor,
        code="scripts/prepare_entry.py",
        inputs=[
            ProcessingInput(
                source=source_s3_uri,
                destination="/opt/ml/processing/input/source",
            ),
            ProcessingInput(
                source=sagemaker.workflow.functions.Join(
                    on="/",
                    values=["s3:/", config.S3_BUCKET, config.S3_PROCESSED_PREFIX],
                ),
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output",
                destination=sagemaker.workflow.functions.Join(
                    on="/",
                    values=["s3:/", config.S3_BUCKET, config.S3_TRAINING_PREFIX, param_data_year],
                ),
            ),
        ],
        job_arguments=["--data-year", param_data_year],
    )

    # [2] Train + AMT: built-in XGBoost, Bayesian tuning
    image_uri = sagemaker.image_uris.retrieve("xgboost", config.AWS_REGION, version=XGBOOST_VERSION)

    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{config.S3_BUCKET}/{config.S3_MODEL_PREFIX}",
        sagemaker_session=sm_session,
    )
    estimator.set_hyperparameters(
        objective=config.XGBOOST_OBJECTIVE,
        eval_metric=config.EVAL_METRIC,
    )

    hp_ranges = {}
    for name, spec in config.AMT_HYPERPARAMETER_RANGES.items():
        if spec["type"] == "Integer":
            hp_ranges[name] = IntegerParameter(spec["min"], spec["max"])
        elif spec["type"] == "Continuous":
            hp_ranges[name] = ContinuousParameter(spec["min"], spec["max"])

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name=f"validation:{config.EVAL_METRIC}",
        hyperparameter_ranges=hp_ranges,
        objective_type="Minimize",
        strategy=config.AMT_STRATEGY,
        max_jobs=config.AMT_MAX_JOBS,
        max_parallel_jobs=config.AMT_MAX_PARALLEL,
    )

    train_uri = sagemaker.workflow.functions.Join(
        on="/",
        values=["s3:/", config.S3_BUCKET, config.S3_TRAINING_PREFIX, param_data_year, "train.csv"],
    )
    val_uri = sagemaker.workflow.functions.Join(
        on="/",
        values=["s3:/", config.S3_BUCKET, config.S3_TRAINING_PREFIX, param_data_year, "val.csv"],
    )

    tuner_args = tuner.fit(
        inputs={
            "train": TrainingInput(train_uri, content_type="text/csv"),
            "validation": TrainingInput(val_uri, content_type="text/csv"),
        },
    )

    _cache = CacheConfig(enable_caching=True, expire_after="PT24H")

    step_tuning = TuningStep(
        name="TrainAMT",
        step_args=tuner_args,
        cache_config=_cache,
        depends_on=[step_prepare],
    )

    # [3] Evaluate: score challenger (and champion if present) on val
    eval_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        role=role_arn,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=sm_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    challenger_model_uri = step_tuning.get_top_model_s3_uri(
        top_k=0,
        s3_bucket=config.S3_BUCKET,
        prefix=config.S3_MODEL_PREFIX,
    )

    step_evaluate = ProcessingStep(
        name="Evaluate",
        processor=eval_processor,
        code="scripts/evaluate_entry.py",
        inputs=[
            ProcessingInput(
                source=source_s3_uri,
                destination="/opt/ml/processing/input/source",
            ),
            ProcessingInput(
                input_name="train",
                source=train_uri,
                destination="/opt/ml/processing/input/train",
            ),
            ProcessingInput(
                input_name="val",
                source=val_uri,
                destination="/opt/ml/processing/input/val",
            ),
            ProcessingInput(
                input_name="model",
                source=challenger_model_uri,
                destination="/opt/ml/processing/input/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
            ),
        ],
        job_arguments=["--champion-model-uri", param_champion_uri],
        property_files=[evaluation_report],
    )

    # [4] Promote? — challenger MAE < champion MAE (or beats baselines for bootstrap)
    promote_condition = ConditionEquals(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="promote",
        ),
        right=True,
    )

    # [5] Register: best model → Model Registry
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=challenger_model_uri,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[INSTANCE_TYPE],
        transform_instances=[INSTANCE_TYPE],
        model_package_group_name=config.MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        customer_metadata_properties={
            "trained_on": param_data_year,
            "objective": config.XGBOOST_OBJECTIVE,
            "group_split_key": config.GROUP_SPLIT_KEY,
        },
    )

    step_condition = ConditionStep(
        name="PromoteCheck",
        conditions=[promote_condition],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[param_data_year, param_champion_uri],
        steps=[step_prepare, step_tuning, step_evaluate, step_condition],
        sagemaker_session=sm_session,
    )


def get_pipeline(role_arn: str, boto_session,
                 source_s3_uri: str) -> Pipeline:
    """Public entry point for pipeline construction."""
    return _build_pipeline(role_arn, boto_session, source_s3_uri)
