import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TuningStep,
    CacheConfig,
)
from sagemaker.workflow.step_collections import RegisterModel, CreateModelStep
from sagemaker import Model
from sagemaker.xgboost import XGBoostPredictor
from sagemaker.tuner import (
    ContinuousParameter,
    IntegerParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="banking-classification",  # Choose any name
    pipeline_name="tuning-step-pipeline",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
    base_job_prefix="sagemaker/DEMO-xgboost-banking",  # Choose any name
):
    """Gets a SageMaker ML Pipeline instance working with on BankingClassification data.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.t3.large"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    
    cache_config = CacheConfig(enable_caching=True, expire_after="1d")
    
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=os.path.join("s3://",default_bucket, base_job_prefix, 'data/bank-additional-full.csv'),  # Change this to point to the s3 location of your raw input data.
    )

    # Processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-banking-preprocess",  # choose any name
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    data_repo_prefix="{}/{}".format(base_job_prefix, "data")
    
    step_process = ProcessingStep(
        name="PreprocessBankingDataForHPO",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{default_bucket}/{data_repo_prefix}/{ExecutionVariables.PIPELINE_EXECUTION_ID}/PreprocessBankingDataForHPO",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=f"s3://{default_bucket}/{data_repo_prefix}/{ExecutionVariables.PIPELINE_EXECUTION_ID}/PreprocessBankingDataForHPO",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{default_bucket}/{data_repo_prefix}/{ExecutionVariables.PIPELINE_EXECUTION_ID}/PreprocessBankingDataForHPO",
            ),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"), 
        job_arguments=["--input-data", input_data],
    )

    # Training step for generating model artifacts
#     model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/BankingTrain"
    model_path = f"s3://{default_bucket}/{base_job_prefix}/BankingTopModel"
    
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/banking-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )        
    xgb_train.set_hyperparameters(
        eval_metric="logloss",
        objective="binary:logistic",  # Define the object metric for the training job
        num_round=50,
        eta=0.1,
        gamma=4,
        min_child_weight=3,
        subsample=0.7,
        silent=0,
        scale_pos_weight=7.7, # Based on imbalance_ratio calculation listed in the preprocess.py script
    )
    
    objective_metric_name = "validation:logloss"

    hyperparameter_ranges = {
        "alpha": ContinuousParameter(0.01, 10.0),  # , scaling_type="Logarithmic"
        "lambda": ContinuousParameter(0.01, 10.0),  # , scaling_type="Logarithmic"
        "max_depth": IntegerParameter(1, 10),
    }

    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=6,
        max_parallel_jobs=2,
        strategy="Bayesian",
        objective_type="Minimize",
        early_stopping_type = 'Auto'
    )

    step_tuning = TuningStep(
        name="HPTuning",
        tuner=tuner_log,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )
    
    model_bucket_key = f"{default_bucket}/{base_job_prefix}/BankingTopModel"
    best_model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key), # model_path
        sagemaker_session=sagemaker_session,
        role=role,
        predictor_cls=XGBoostPredictor,
    )

    step_create_first = CreateModelStep(
        name="BankingTopModel",
        model=best_model,
        inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m4.large"),
    )

# If we wanted to bypass tuning and jump straight to tuning we would use the code below
    #     step_train = TrainingStep(
#         name="HPTuning",
#         estimator=xgb_train,
#         inputs={
#             "train": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "train"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#             "validation": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "validation"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#         },
#     )

    # Processing step for evaluation    
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-tuning-step-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    step_eval = ProcessingStep(
        name="EvaluateTopModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),# model_path
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"), 
        property_files=[evaluation_report],
        cache_config=cache_config,
    )
    

    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )
    
    step_register_best = RegisterModel(
        name="RegisterBestBankingModel",
        estimator=xgb_train,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key), # best_model, model_path 
        content_types=["text/csv"],
        response_types=["text/csv"], # "application/json""text/csv"
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name= "EvaluateTopModel", # step_eval.name 
            property_file=evaluation_report,
            json_path="binary_classification_metrics.auc.value",
        ),
        right=0.7,
    )  
    step_cond = ConditionStep(
        name="CheckAUCBankingEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register_best],
        else_steps=[],
    )
    
    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name, #"tuning-step-pipeline",
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            input_data,
            model_approval_status,
        ],
        steps=[
            step_process,
            step_tuning,
            step_create_first,
            step_eval,
            step_cond,
        ],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline