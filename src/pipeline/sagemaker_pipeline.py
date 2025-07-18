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

import logging
from typing import Dict, Any, Optional, List, Union

from src.pipeline.sagemaker_pipeline_factory import PipelineFactory
from src.pipeline.components.base import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pipeline_factory(aws_profile: str = "ab", region: str = "us-east-1") -> PipelineFactory:
    """
    Create a pipeline factory with the specified AWS profile and region.
    
    Args:
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Configured PipelineFactory
    """
    return PipelineFactory(aws_profile=aws_profile, region=region)


def create_end_to_end_pipeline(
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
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Create an end-to-end SageMaker pipeline with preprocessing, training, evaluation, and registration.
    
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
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Dictionary with pipeline details
    """
    logger.info(f"Creating end-to-end pipeline: {pipeline_name}")
    
    # Create pipeline factory
    factory = create_pipeline_factory(aws_profile=aws_profile, region=region)
    
    # Create pipeline
    pipeline = factory.create_complete_pipeline(
        pipeline_name=pipeline_name,
        preprocessing_script=preprocessing_script,
        training_script=training_script,
        evaluation_script=evaluation_script,
        input_data=input_data,
        model_name=model_name,
        hyperparameters=hyperparameters,
        min_accuracy=min_accuracy,
        instance_type_processing=instance_type_processing,
        instance_type_training=instance_type_training,
        instance_count_training=instance_count_training
    )
    
    # Return pipeline details
    return {
        "pipeline": pipeline,
        "pipeline_name": pipeline_name,
        "pipeline_arn": pipeline.arn,
        "pipeline_definition": pipeline.definition()
    }


def execute_pipeline(
    pipeline_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    wait: bool = False,
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Execute a SageMaker pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        parameters: Pipeline parameters (optional)
        wait: Whether to wait for the pipeline to complete
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Pipeline execution details
    """
    logger.info(f"Executing pipeline: {pipeline_name}")
    
    # Create pipeline factory
    factory = create_pipeline_factory(aws_profile=aws_profile, region=region)
    
    # Get pipeline
    pipeline = factory.sagemaker_client.describe_pipeline(PipelineName=pipeline_name)
    
    # Execute pipeline
    execution = factory.sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=[
            {"Name": name, "Value": value} for name, value in (parameters or {}).items()
        ]
    )
    
    execution_arn = execution["PipelineExecutionArn"]
    logger.info(f"Pipeline execution started: {execution_arn}")
    
    # Wait for pipeline to complete if requested
    if wait:
        logger.info(f"Waiting for pipeline execution to complete...")
        waiter = factory.sagemaker_client.get_waiter("pipeline_execution_complete")
        waiter.wait(PipelineExecutionArn=execution_arn)
        
        # Get execution status
        execution_details = factory.sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        
        logger.info(f"Pipeline execution completed with status: {execution_details['PipelineExecutionStatus']}")
        
        return {
            "pipeline_name": pipeline_name,
            "execution_arn": execution_arn,
            "status": execution_details["PipelineExecutionStatus"],
            "execution_details": execution_details
        }
    
    return {
        "pipeline_name": pipeline_name,
        "execution_arn": execution_arn,
        "status": "InProgress"
    }


def get_pipeline_execution_status(
    execution_arn: str,
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Get the status of a pipeline execution.
    
    Args:
        execution_arn: ARN of the pipeline execution
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Pipeline execution status details
    """
    logger.info(f"Getting pipeline execution status: {execution_arn}")
    
    # Create session manager
    session_manager = SessionManager(aws_profile=aws_profile, region=region)
    
    # Get SageMaker client
    sagemaker_client = session_manager.get_client("sagemaker")
    
    # Get pipeline execution
    response = sagemaker_client.describe_pipeline_execution(
        PipelineExecutionArn=execution_arn
    )
    
    logger.info(f"Pipeline execution status: {response['PipelineExecutionStatus']}")
    return response


def list_pipeline_executions(
    pipeline_name: str,
    max_results: int = 10,
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> List[Dict[str, Any]]:
    """
    List executions of a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        max_results: Maximum number of results
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        List of pipeline executions
    """
    logger.info(f"Listing pipeline executions: {pipeline_name}")
    
    # Create session manager
    session_manager = SessionManager(aws_profile=aws_profile, region=region)
    
    # Get SageMaker client
    sagemaker_client = session_manager.get_client("sagemaker")
    
    # List pipeline executions
    response = sagemaker_client.list_pipeline_executions(
        PipelineName=pipeline_name,
        MaxResults=max_results
    )
    
    return response.get("PipelineExecutionSummaries", [])