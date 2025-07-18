"""
SageMaker Pipeline Factory

This module provides a factory for creating SageMaker Pipelines with various components.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import Step
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger, ParameterFloat
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.steps import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

from src.pipeline.components.base import SessionManager
from src.pipeline.components.preprocessing import PreprocessingComponent
from src.pipeline.components.training import TrainingComponent
from src.pipeline.components.evaluation import EvaluationComponent

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating SageMaker Pipelines with various components.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the pipeline factory.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize session manager
        self.session_manager = SessionManager(aws_profile=aws_profile, region=region)
        
        # Get AWS sessions
        self.aws_session = self.session_manager.get_aws_session()
        self.sagemaker_session = self.session_manager.get_sagemaker_session()
        self.pipeline_session = self.session_manager.get_pipeline_session()
        
        # Get AWS clients
        self.sagemaker_client = self.session_manager.get_client('sagemaker')
        self.s3_client = self.session_manager.get_client('s3')
        
        # Get project configuration
        self.project_config = get_config()
        self.execution_role = self.project_config['iam']['roles']['sagemaker_execution']['arn']
        
        # Set default S3 bucket
        self.default_bucket = self.sagemaker_session.default_bucket()
        
        # Initialize components
        self.preprocessing_component = PreprocessingComponent(
            aws_profile=aws_profile,
            region=region,
            sagemaker_session=self.sagemaker_session,
            pipeline_session=self.pipeline_session,
            execution_role=self.execution_role
        )
        
        self.training_component = TrainingComponent(
            aws_profile=aws_profile,
            region=region,
            sagemaker_session=self.sagemaker_session,
            pipeline_session=self.pipeline_session,
            execution_role=self.execution_role
        )
        
        self.evaluation_component = EvaluationComponent(
            aws_profile=aws_profile,
            region=region,
            sagemaker_session=self.sagemaker_session,
            pipeline_session=self.pipeline_session,
            execution_role=self.execution_role
        )
        
        logger.info(f"Pipeline factory initialized for region: {region}")
        logger.info(f"Using execution role: {self.execution_role}")
        logger.info(f"Using default S3 bucket: {self.default_bucket}")
    
    def create_pipeline(self,
                       pipeline_name: str,
                       steps: List[Step],
                       pipeline_parameters: Optional[List[Any]] = None,
                       pipeline_description: Optional[str] = None) -> Pipeline:
        """
        Create a SageMaker pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            steps: List of pipeline steps
            pipeline_parameters: List of pipeline parameters (optional)
            pipeline_description: Description of the pipeline (optional)
            
        Returns:
            Configured Pipeline
        """
        logger.info(f"Creating pipeline: {pipeline_name}")
        
        # Create pipeline
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=pipeline_parameters or [],
            steps=steps,
            sagemaker_session=self.pipeline_session,
            description=pipeline_description
        )
        
        logger.info(f"Pipeline created: {pipeline_name}")
        logger.info(f"Number of steps: {len(steps)}")
        
        return pipeline
    
    def create_model_registration_step(self,
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
                                      description: Optional[str] = None) -> RegisterModel:
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
    
    def create_accuracy_condition_step(self,
                                      step_name: str,
                                      evaluation_step: Step,
                                      min_accuracy: float,
                                      if_steps: List[Step],
                                      else_steps: Optional[List[Step]] = None) -> ConditionStep:
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
    
    def create_complete_pipeline(self,
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
                               email_notifications: Optional[List[str]] = None) -> Pipeline:
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
        preprocessing_step = self.preprocessing_component.create_step(
            step_name="Preprocessing",
            script_path=preprocessing_script,
            input_data=input_data_param,
            instance_type=instance_type_processing,
            image_uri=self.project_config['sagemaker']['processing_image']
        )
        
        # Create training step
        training_step = self.training_component.create_step(
            step_name="Training",
            script_path=training_script,
            input_train=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            input_validation=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            hyperparameters=hyperparameters,
            instance_type=instance_type_training,
            instance_count=instance_count_training
        )
        
        # Create evaluation step
        evaluation_step = self.evaluation_component.create_step(
            step_name="Evaluation",
            script_path=evaluation_script,
            model_path=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            test_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            instance_type=instance_type_processing,
            image_uri=self.project_config['sagemaker']['processing_image']
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
            ],
            pipeline_description=f"End-to-end ML pipeline for {model_name}"
        )
        
        logger.info(f"Complete pipeline created: {pipeline_name}")
        return pipeline
    
    def execute_pipeline(self,
                        pipeline: Pipeline,
                        parameters: Optional[Dict[str, Any]] = None,
                        wait: bool = False) -> Dict[str, Any]:
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
    
    def get_pipeline_execution_status(self,
                                     execution_arn: str) -> Dict[str, Any]:
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
    
    def list_pipeline_executions(self,
                               pipeline_name: str,
                               max_results: int = 10) -> List[Dict[str, Any]]:
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
        
        return response.get('PipelineExecutionSummaries', [])