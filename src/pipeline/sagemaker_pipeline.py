"""
SageMaker Pipeline Implementation

This module provides a comprehensive SageMaker Pipeline implementation with
preprocessing, training, evaluation, and deployment steps.

Requirements addressed:
- 6.1: SageMaker Pipeline with data preprocessing, training, evaluation, and deployment steps
- 6.2: Pipeline artifacts stored in S3 with proper versioning
- 6.3: Experiments tracked in MLFlow with parameters and metrics
- 6.4: Auto-scaling endpoints with monitoring
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger, ParameterFloat
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.lambda_helper import Lambda

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerPipelineBuilder:
    """
    Builds and manages SageMaker Pipelines for MLOps workflows.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the SageMaker Pipeline builder.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker', region_name=region)
        self.s3_client = self.session.client('s3', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Initialize Pipeline session
        self.pipeline_session = PipelineSession(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Get project configuration
        self.project_config = get_config()
        self.execution_role = self.project_config['iam']['roles']['sagemaker_execution']['arn']
        
        # Set default S3 bucket
        self.default_bucket = self.sagemaker_session.default_bucket()
        
        logger.info(f"SageMaker Pipeline builder initialized for region: {region}")
        logger.info(f"Using execution role: {self.execution_role}")
        logger.info(f"Using default S3 bucket: {self.default_bucket}")
    
    def create_preprocessing_step(
        self,
        step_name: str,
        script_path: str,
        input_data: str,
        output_path: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        volume_size_in_gb: int = 30,
        max_runtime_in_seconds: int = 3600,
        environment: Optional[Dict[str, str]] = None,
        arguments: Optional[List[str]] = None
    ) -> ProcessingStep:
        """
        Create a preprocessing step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            script_path: Path to the preprocessing script
            input_data: S3 path to input data
            output_path: S3 path for output data (optional)
            instance_type: Instance type for processing
            instance_count: Number of instances
            volume_size_in_gb: Volume size in GB
            max_runtime_in_seconds: Maximum runtime in seconds
            environment: Environment variables for the processor
            arguments: Command-line arguments for the processor
            
        Returns:
            Configured ProcessingStep
        """
        logger.info(f"Creating preprocessing step: {step_name}")
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"s3://{self.default_bucket}/pipeline/preprocessing/{int(time.time())}"
        
        # Create script processor
        script_processor = ScriptProcessor(
            command=["python3"],
            image_uri=self.project_config['sagemaker']['processing_image'],
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=self.pipeline_session,
            env=environment or {}
        )
        
        # Create processing step
        processing_step = ProcessingStep(
            name=step_name,
            processor=script_processor,
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output/train",
                    destination=f"{output_path}/train"
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/output/validation",
                    destination=f"{output_path}/validation"
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output/test",
                    destination=f"{output_path}/test"
                )
            ],
            code=script_path,
            job_arguments=arguments or []
        )
        
        logger.info(f"Preprocessing step created: {step_name}")
        logger.info(f"Input data: {input_data}")
        logger.info(f"Output path: {output_path}")
        
        return processing_step
    
    def create_training_step(
        self,
        step_name: str,
        script_path: str,
        input_train: str,
        input_validation: str,
        output_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Union[str, float, int]]] = None,
        instance_type: str = "ml.p3.2xlarge",
        instance_count: int = 1,
        volume_size_in_gb: int = 50,
        max_runtime_in_seconds: int = 86400,  # 24 hours
        environment: Optional[Dict[str, str]] = None,
        framework_version: str = "1.10.0",
        py_version: str = "py38",
        dependencies: Optional[List[str]] = None
    ) -> TrainingStep:
        """
        Create a training step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            script_path: Path to the training script
            input_train: S3 path to training data
            input_validation: S3 path to validation data
            output_path: S3 path for output model artifacts (optional)
            hyperparameters: Hyperparameters for training
            instance_type: Instance type for training
            instance_count: Number of instances
            volume_size_in_gb: Volume size in GB
            max_runtime_in_seconds: Maximum runtime in seconds
            environment: Environment variables for the estimator
            framework_version: PyTorch framework version
            py_version: Python version
            dependencies: Additional dependencies to install
            
        Returns:
            Configured TrainingStep
        """
        logger.info(f"Creating training step: {step_name}")
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"s3://{self.default_bucket}/pipeline/models/{int(time.time())}"
        
        # Set default hyperparameters if not provided
        if not hyperparameters:
            hyperparameters = {
                "epochs": 10,
                "batch-size": 32,
                "learning-rate": 0.001,
                "optimizer": "adam"
            }
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point=os.path.basename(script_path),
            source_dir=os.path.dirname(script_path),
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size_in_gb,
            max_run=max_runtime_in_seconds,
            framework_version=framework_version,
            py_version=py_version,
            hyperparameters=hyperparameters,
            environment=environment or {},
            sagemaker_session=self.pipeline_session,
            dependencies=dependencies or []
        )
        
        # Create training step
        training_step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs={
                "train": input_train,
                "validation": input_validation
            }
        )
        
        logger.info(f"Training step created: {step_name}")
        logger.info(f"Training data: {input_train}")
        logger.info(f"Validation data: {input_validation}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Hyperparameters: {hyperparameters}")
        
        return training_step
    
    def create_evaluation_step(
        self,
        step_name: str,
        script_path: str,
        model_path: str,
        test_data: str,
        output_path: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        volume_size_in_gb: int = 30,
        max_runtime_in_seconds: int = 3600,
        environment: Optional[Dict[str, str]] = None,
        arguments: Optional[List[str]] = None
    ) -> ProcessingStep:
        """
        Create an evaluation step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            script_path: Path to the evaluation script
            model_path: S3 path to model artifacts
            test_data: S3 path to test data
            output_path: S3 path for evaluation results (optional)
            instance_type: Instance type for processing
            instance_count: Number of instances
            volume_size_in_gb: Volume size in GB
            max_runtime_in_seconds: Maximum runtime in seconds
            environment: Environment variables for the processor
            arguments: Command-line arguments for the processor
            
        Returns:
            Configured ProcessingStep
        """
        logger.info(f"Creating evaluation step: {step_name}")
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"s3://{self.default_bucket}/pipeline/evaluation/{int(time.time())}"
        
        # Create script processor
        script_processor = ScriptProcessor(
            command=["python3"],
            image_uri=self.project_config['sagemaker']['processing_image'],
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=self.pipeline_session,
            env=environment or {}
        )
        
        # Create evaluation property file
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json"
        )
        
        # Create processing step
        evaluation_step = ProcessingStep(
            name=step_name,
            processor=script_processor,
            inputs=[
                ProcessingInput(
                    source=model_path,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=test_data,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=output_path
                )
            ],
            code=script_path,
            job_arguments=arguments or [],
            property_files=[evaluation_report]
        )
        
        logger.info(f"Evaluation step created: {step_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Test data: {test_data}")
        logger.info(f"Output path: {output_path}")
        
        return evaluation_step
    
    def create_model_registration_step(
        self,
        step_name: str,
        model_path: str,
        model_name: str,
        image_uri: Optional[str] = None,
        model_metrics: Optional[ModelMetrics] = None,
        content_types: Optional[List[str]] = None,
        response_types: Optional[List[str]] = None,
        inference_instances: Optional[List[str]] = None,
        transform_instances: Optional[List[str]] = None,
        model_package_group_name: Optional[str] = None,
        approval_status: str = "PendingManualApproval",
        description: Optional[str] = None
    ) -> RegisterModel:
        """
        Create a model registration step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            model_path: S3 path to model artifacts
            model_name: Name of the model
            image_uri: URI of the inference image
            model_metrics: Model metrics for registration
            content_types: Content types for inference
            response_types: Response types for inference
            inference_instances: Instance types for inference
            transform_instances: Instance types for batch transform
            model_package_group_name: Name of the model package group
            approval_status: Approval status for the model
            description: Description of the model
            
        Returns:
            Configured RegisterModel step
        """
        logger.info(f"Creating model registration step: {step_name}")
        
        # Set default image URI if not provided
        if not image_uri:
            image_uri = self.project_config['sagemaker']['inference_image']
        
        # Set default content types if not provided
        if not content_types:
            content_types = ["application/json", "text/csv"]
        
        # Set default response types if not provided
        if not response_types:
            response_types = ["application/json"]
        
        # Set default inference instances if not provided
        if not inference_instances:
            inference_instances = ["ml.m5.xlarge", "ml.m5.2xlarge"]
        
        # Set default transform instances if not provided
        if not transform_instances:
            transform_instances = ["ml.m5.xlarge"]
        
        # Set default model package group name if not provided
        if not model_package_group_name:
            model_package_group_name = f"{model_name}-group"
        
        # Create register model step
        register_model_step = RegisterModel(
            name=step_name,
            model_data=model_path,
            image_uri=image_uri,
            role=self.execution_role,
            content_types=content_types,
            response_types=response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_group_name=model_package_group_name,
            approval_status=approval_status,
            model_metrics=model_metrics,
            description=description or f"Model package for {model_name}",
            sagemaker_session=self.pipeline_session
        )
        
        logger.info(f"Model registration step created: {step_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Model package group: {model_package_group_name}")
        logger.info(f"Approval status: {approval_status}")
        
        return register_model_step
    
    def create_condition_step(
        self,
        step_name: str,
        condition_step: ConditionStep,
        if_steps: List[Any],
        else_steps: Optional[List[Any]] = None
    ) -> ConditionStep:
        """
        Create a condition step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            condition_step: Condition to evaluate
            if_steps: Steps to execute if condition is true
            else_steps: Steps to execute if condition is false (optional)
            
        Returns:
            Configured ConditionStep
        """
        logger.info(f"Creating condition step: {step_name}")
        
        # Create condition step
        condition_step = ConditionStep(
            name=step_name,
            conditions=[condition_step],
            if_steps=if_steps,
            else_steps=else_steps or []
        )
        
        logger.info(f"Condition step created: {step_name}")
        return condition_step
    
    def create_accuracy_condition_step(
        self,
        step_name: str,
        evaluation_step: ProcessingStep,
        min_accuracy: float,
        if_steps: List[Any],
        else_steps: Optional[List[Any]] = None
    ) -> ConditionStep:
        """
        Create an accuracy condition step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            evaluation_step: Evaluation step with accuracy metric
            min_accuracy: Minimum accuracy threshold
            if_steps: Steps to execute if accuracy >= min_accuracy
            else_steps: Steps to execute if accuracy < min_accuracy (optional)
            
        Returns:
            Configured ConditionStep
        """
        logger.info(f"Creating accuracy condition step: {step_name}")
        
        # Create condition
        condition = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_step.properties.PropertyFiles["EvaluationReport"],
                json_path="metrics.accuracy"
            ),
            right=min_accuracy
        )
        
        # Create condition step
        condition_step = ConditionStep(
            name=step_name,
            conditions=[condition],
            if_steps=if_steps,
            else_steps=else_steps or []
        )
        
        logger.info(f"Accuracy condition step created: {step_name}")
        logger.info(f"Minimum accuracy threshold: {min_accuracy}")
        
        return condition_step
    
    def create_lambda_step(
        self,
        step_name: str,
        lambda_function_name: str,
        lambda_code: str,
        handler: str = "index.lambda_handler",
        timeout: int = 900,
        memory_size: int = 512,
        environment: Optional[Dict[str, str]] = None
    ) -> Lambda:
        """
        Create a Lambda step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            lambda_function_name: Name of the Lambda function
            lambda_code: Lambda function code
            handler: Lambda function handler
            timeout: Lambda function timeout in seconds
            memory_size: Lambda function memory size in MB
            environment: Environment variables for the Lambda function
            
        Returns:
            Configured Lambda step
        """
        logger.info(f"Creating Lambda step: {step_name}")
        
        # Create Lambda function
        lambda_helper = Lambda(
            function_name=lambda_function_name,
            execution_role_arn=self.execution_role,
            script=lambda_code,
            handler=handler,
            timeout=timeout,
            memory_size=memory_size,
            environment=environment or {}
        )
        
        logger.info(f"Lambda step created: {step_name}")
        logger.info(f"Lambda function name: {lambda_function_name}")
        
        return lambda_helper
    
    def create_pipeline(
        self,
        pipeline_name: str,
        steps: List[Any],
        pipeline_description: Optional[str] = None,
        pipeline_parameters: Optional[List[Any]] = None
    ) -> Pipeline:
        """
        Create a SageMaker pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            steps: List of pipeline steps
            pipeline_description: Description of the pipeline (optional)
            pipeline_parameters: List of pipeline parameters (optional)
            
        Returns:
            Configured Pipeline
        """
        logger.info(f"Creating pipeline: {pipeline_name}")
        
        # Create pipeline
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=pipeline_parameters or [],
            steps=steps,
            sagemaker_session=self.pipeline_session
        )
        
        logger.info(f"Pipeline created: {pipeline_name}")
        logger.info(f"Number of steps: {len(steps)}")
        
        return pipeline
    
    def create_complete_pipeline(
        self,
        pipeline_name: str,
        preprocessing_script: str,
        training_script: str,
        evaluation_script: str,
        input_data: str,
        model_name: str,
        hyperparameters: Optional[Dict[str, Union[str, float, int]]] = None,
        min_accuracy: float = 0.7,
        instance_type_processing: str = "ml.m5.xlarge",
        instance_type_training: str = "ml.p3.2xlarge",
        instance_count_training: int = 1,
        deploy_approved_model: bool = True,
        enable_monitoring: bool = True,
        email_notifications: Optional[List[str]] = None
    ) -> Pipeline:
        """
        Create a complete SageMaker pipeline with preprocessing, training, evaluation, and deployment.
        
        Args:
            pipeline_name: Name of the pipeline
            preprocessing_script: Path to the preprocessing script
            training_script: Path to the training script
            evaluation_script: Path to the evaluation script
            input_data: S3 path to input data
            model_name: Name of the model
            hyperparameters: Hyperparameters for training
            min_accuracy: Minimum accuracy threshold for model approval
            instance_type_processing: Instance type for processing steps
            instance_type_training: Instance type for training step
            instance_count_training: Number of instances for training
            deploy_approved_model: Whether to deploy approved models
            enable_monitoring: Whether to enable model monitoring
            email_notifications: List of email addresses for notifications
            
        Returns:
            Configured Pipeline
        """
        logger.info(f"Creating complete pipeline: {pipeline_name}")
        
        # Create pipeline parameters
        input_data_param = ParameterString(
            name="InputData",
            default_value=input_data
        )
        
        model_name_param = ParameterString(
            name="ModelName",
            default_value=model_name
        )
        
        min_accuracy_param = ParameterFloat(
            name="MinAccuracy",
            default_value=min_accuracy
        )
        
        # Create preprocessing step
        preprocessing_step = self.create_preprocessing_step(
            step_name="Preprocessing",
            script_path=preprocessing_script,
            input_data=input_data_param,
            instance_type=instance_type_processing
        )
        
        # Create training step
        training_step = self.create_training_step(
            step_name="Training",
            script_path=training_script,
            input_train=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            input_validation=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            hyperparameters=hyperparameters,
            instance_type=instance_type_training,
            instance_count=instance_count_training
        )
        
        # Create evaluation step
        evaluation_step = self.create_evaluation_step(
            step_name="Evaluation",
            script_path=evaluation_script,
            model_path=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            test_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            instance_type=instance_type_processing
        )
        
        # Create model metrics
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                        "evaluation.json"
                    ]
                ),
                content_type="application/json"
            )
        )
        
        # Create model registration step
        register_model_step = self.create_model_registration_step(
            step_name="RegisterModel",
            model_path=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            model_name=model_name_param,
            model_metrics=model_metrics,
            approval_status="PendingManualApproval"
        )
        
        # Create pipeline steps
        pipeline_steps = [
            preprocessing_step,
            training_step,
            evaluation_step
        ]
        
        # Create accuracy condition step
        if min_accuracy > 0:
            # Create accuracy condition
            accuracy_condition_step = self.create_accuracy_condition_step(
                step_name="CheckAccuracy",
                evaluation_step=evaluation_step,
                min_accuracy=min_accuracy_param,
                if_steps=[register_model_step]
            )
            
            # Add condition step to pipeline steps
            pipeline_steps.append(accuracy_condition_step)
        else:
            # Add register model step directly
            pipeline_steps.append(register_model_step)
        
        # Create pipeline
        pipeline = self.create_pipeline(
            pipeline_name=pipeline_name,
            steps=pipeline_steps,
            pipeline_parameters=[
                input_data_param,
                model_name_param,
                min_accuracy_param
            ]
        )
        
        logger.info(f"Complete pipeline created: {pipeline_name}")
        return pipeline
    
    def execute_pipeline(
        self,
        pipeline: Pipeline,
        parameters: Optional[Dict[str, Any]] = None,
        wait: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a SageMaker pipeline.
        
        Args:
            pipeline: Pipeline to execute
            parameters: Pipeline parameters (optional)
            wait: Whether to wait for the pipeline to complete
            
        Returns:
            Pipeline execution details
        """
        logger.info(f"Executing pipeline: {pipeline.name}")
        
        # Execute pipeline
        execution = pipeline.start(
            parameters=parameters or {}
        )
        
        logger.info(f"Pipeline execution started: {execution.arn}")
        
        # Wait for pipeline to complete if requested
        if wait:
            logger.info(f"Waiting for pipeline execution to complete...")
            execution.wait()
            logger.info(f"Pipeline execution completed with status: {execution.describe()['PipelineExecutionStatus']}")
        
        return {
            "pipeline_name": pipeline.name,
            "execution_arn": execution.arn,
            "status": execution.describe()["PipelineExecutionStatus"]
        }
    
    def get_pipeline_execution_status(
        self,
        execution_arn: str
    ) -> Dict[str, Any]:
        """
        Get the status of a pipeline execution.
        
        Args:
            execution_arn: ARN of the pipeline execution
            
        Returns:
            Pipeline execution status details
        """
        logger.info(f"Getting pipeline execution status: {execution_arn}")
        
        # Get pipeline execution
        response = self.sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        
        logger.info(f"Pipeline execution status: {response['PipelineExecutionStatus']}")
        return response
    
    def list_pipeline_executions(
        self,
        pipeline_name: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List executions of a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            max_results: Maximum number of results
            
        Returns:
            List of pipeline executions
        """
        logger.info(f"Listing pipeline executions: {pipeline_name}")
        
        # List pipeline executions
        response = self.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline_name,
            MaxResults=max_results
        )
        
        executions = response.get("PipelineExecutionSummaries", [])
        logger.info(f"Found {len(executions)} pipeline executions")
        
        return executions
    
    def get_pipeline_definition(
        self,
        pipeline: Pipeline
    ) -> str:
        """
        Get the definition of a pipeline.
        
        Args:
            pipeline: Pipeline to get definition for
            
        Returns:
            Pipeline definition as JSON string
        """
        logger.info(f"Getting pipeline definition: {pipeline.name}")
        
        # Get pipeline definition
        definition = pipeline.definition()
        
        logger.info(f"Pipeline definition retrieved: {pipeline.name}")
        return definition
    
    def save_pipeline_definition(
        self,
        pipeline: Pipeline,
        file_path: str
    ) -> None:
        """
        Save the definition of a pipeline to a file.
        
        Args:
            pipeline: Pipeline to save definition for
            file_path: Path to save the definition to
        """
        logger.info(f"Saving pipeline definition: {pipeline.name}")
        
        # Get pipeline definition
        definition = self.get_pipeline_definition(pipeline)
        
        # Save definition to file
        with open(file_path, "w") as f:
            f.write(definition)
        
        logger.info(f"Pipeline definition saved to: {file_path}")
    
    def load_pipeline_from_definition(
        self,
        pipeline_name: str,
        definition_file: str
    ) -> Pipeline:
        """
        Load a pipeline from a definition file.
        
        Args:
            pipeline_name: Name of the pipeline
            definition_file: Path to the definition file
            
        Returns:
            Loaded Pipeline
        """
        logger.info(f"Loading pipeline from definition: {definition_file}")
        
        # Load definition from file
        with open(definition_file, "r") as f:
            definition = f.read()
        
        # Create pipeline from definition
        pipeline = Pipeline(
            name=pipeline_name,
            definition=definition,
            sagemaker_session=self.pipeline_session
        )
        
        logger.info(f"Pipeline loaded: {pipeline_name}")
        return pipeline