#!/usr/bin/env python3
"""
SageMaker Pipeline Creation Script

This script creates and deploys a SageMaker Pipeline for YOLOv11 training,
including preprocessing, training, evaluation, and deployment steps.

Usage:
    python scripts/training/create_sagemaker_pipeline.py --create-pipeline
    python scripts/training/create_sagemaker_pipeline.py --execute-pipeline
    python scripts/training/create_sagemaker_pipeline.py --list-executions
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.pipeline.sagemaker_pipeline import SageMakerPipelineBuilder
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create and manage SageMaker Pipeline')
    
    # Actions
    parser.add_argument('--create-pipeline', action='store_true',
                       help='Create SageMaker Pipeline')
    parser.add_argument('--execute-pipeline', action='store_true',
                       help='Execute SageMaker Pipeline')
    parser.add_argument('--list-executions', action='store_true',
                       help='List pipeline executions')
    parser.add_argument('--delete-pipeline', action='store_true',
                       help='Delete SageMaker Pipeline')
    
    # Pipeline configuration
    parser.add_argument('--pipeline-name', type=str,
                       help='Name of the pipeline')
    parser.add_argument('--input-data', type=str,
                       help='S3 URI of input data')
    
    # AWS configuration
    parser.add_argument('--aws-profile', type=str, default='ab',
                       help='AWS profile to use')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    
    return parser.parse_args()


def create_preprocessing_pipeline(args, config):
    """
    Create a SageMaker Pipeline with preprocessing, training, evaluation, registration, and deployment steps.
    
    Args:
        args: Command line arguments
        config: Project configuration
        
    Returns:
        Created pipeline
    """
    logger.info("Creating SageMaker Pipeline with preprocessing, training, evaluation, registration, and deployment steps")
    
    # Initialize pipeline builder
    pipeline_builder = SageMakerPipelineBuilder(
        aws_profile=args.aws_profile,
        region=args.region
    )
    
    # Create preprocessing script
    script_dir = os.path.join(project_root, 'scripts', 'preprocessing')
    script_path = os.path.join(script_dir, 'preprocess_yolo_data.py')
    
    # Create training script with observability
    training_script_dir = os.path.join(project_root, 'scripts', 'training')
    training_script_path = os.path.join(training_script_dir, 'train_sagemaker_observability.py')
    
    # Create evaluation script
    evaluation_script_path = os.path.join(training_script_dir, 'evaluate_model.py')
    
    # Create training script if it doesn't exist
    if not os.path.exists(training_script_path):
        pipeline_builder.create_training_script_with_observability(training_script_path)
    
    # Set pipeline name
    pipeline_name = args.pipeline_name or config['pipeline']['name']
    
    # Set input data
    input_data = args.input_data or f"s3://{config['aws']['data_bucket']}/data/"
    
    # Create preprocessing step
    preprocessing_step = pipeline_builder.create_preprocessing_step(
        step_name="YOLOv11DataPreprocessing",
        script_path=script_path,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        input_data=input_data,
        output_prefix="yolov11-preprocessing",
        base_job_name="yolov11-preprocessing",
        environment={
            "PYTHONPATH": "/opt/ml/processing/code",
            "AWS_DEFAULT_REGION": args.region,
            "AWS_PROFILE": args.aws_profile
        }
    )
    
    # Create training step with observability
    training_step = pipeline_builder.create_training_step(
        step_name="YOLOv11Training",
        preprocessing_step=preprocessing_step,
        script_path=training_script_path,
        instance_type=config['training']['instance_type'],
        instance_count=config['training']['instance_count'],
        volume_size=config['training']['volume_size'],
        hyperparameters={
            'model': 'yolov11n',
            'epochs': 100,
            'batch-size': 16,
            'learning-rate': 0.01,
            'img-size': 640
        },
        environment={
            "PYTHONPATH": "/opt/ml/code",
            "AWS_DEFAULT_REGION": args.region,
            "AWS_PROFILE": args.aws_profile
        },
        base_job_name="yolov11-training",
        enable_mlflow=True,
        enable_powertools=True
    )
    
    # Create evaluation step
    evaluation_step = pipeline_builder.create_evaluation_step(
        step_name="YOLOv11Evaluation",
        training_step=training_step,
        preprocessing_step=preprocessing_step,
        script_path=evaluation_script_path,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="yolov11-evaluation"
    )
    
    # Create registration step
    registration_step = pipeline_builder.create_registration_step(
        step_name="YOLOv11Registration",
        training_step=training_step,
        evaluation_step=evaluation_step,
        model_name=config['model']['name'],
        model_package_group_name=f"{config['model']['name']}-group",
        model_approval_status="PendingManualApproval"
    )
    
    # Create deployment step with AWS Solutions Constructs
    # Get Lambda function ARN from config or environment
    lambda_function_arn = os.environ.get(
        "DEPLOY_LAMBDA_ARN", 
        config.get('deployment', {}).get('lambda_arn', '')
    )
    
    # If Lambda function ARN is provided, add deployment step
    if lambda_function_arn:
        deployment_step = pipeline_builder.create_deployment_step(
            step_name="YOLOv11Deployment",
            registration_step=registration_step,
            lambda_function_arn=lambda_function_arn,
            model_name=config['model']['name'],
            instance_type=config['inference']['instance_type'],
            initial_instance_count=config['inference']['initial_instance_count'],
            max_instance_count=config['inference']['max_instance_count']
        )
        
        # Add deployment step to pipeline
        steps = [preprocessing_step, training_step, evaluation_step, registration_step, deployment_step]
    else:
        logger.warning("Lambda function ARN not provided, skipping deployment step")
        steps = [preprocessing_step, training_step, evaluation_step, registration_step]
    
    # Create pipeline with all steps
    pipeline = pipeline_builder.create_pipeline(
        pipeline_name=pipeline_name,
        steps=steps,
        pipeline_description="YOLOv11 Training Pipeline with Preprocessing, Evaluation, Registration, and Deployment"
    )
    
    # Create pipeline definition
    pipeline_definition = pipeline.definition()
    
    # Save pipeline definition to file
    definition_path = os.path.join(project_root, 'configs', f"{pipeline_name}-definition.json")
    with open(definition_path, 'w') as f:
        json.dump(json.loads(pipeline_definition), f, indent=2)
    
    logger.info(f"Pipeline definition saved to: {definition_path}")
    
    return pipeline


def execute_pipeline(args, config):
    """
    Execute a SageMaker Pipeline.
    
    Args:
        args: Command line arguments
        config: Project configuration
        
    Returns:
        Execution ARN
    """
    logger.info("Executing SageMaker Pipeline")
    
    # Initialize pipeline builder
    pipeline_builder = SageMakerPipelineBuilder(
        aws_profile=args.aws_profile,
        region=args.region
    )
    
    # Set pipeline name
    pipeline_name = args.pipeline_name or config['pipeline']['name']
    
    # Get pipeline
    response = pipeline_builder.sagemaker_client.describe_pipeline(
        PipelineName=pipeline_name
    )
    
    # Start execution
    execution_response = pipeline_builder.sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDescription=f"Execution of {pipeline_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    execution_arn = execution_response['PipelineExecutionArn']
    logger.info(f"Pipeline execution started: {execution_arn}")
    
    return execution_arn


def list_pipeline_executions(args, config):
    """
    List executions of a SageMaker Pipeline.
    
    Args:
        args: Command line arguments
        config: Project configuration
    """
    logger.info("Listing SageMaker Pipeline executions")
    
    # Initialize pipeline builder
    pipeline_builder = SageMakerPipelineBuilder(
        aws_profile=args.aws_profile,
        region=args.region
    )
    
    # Set pipeline name
    pipeline_name = args.pipeline_name or config['pipeline']['name']
    
    # List executions
    response = pipeline_builder.sagemaker_client.list_pipeline_executions(
        PipelineName=pipeline_name
    )
    
    executions = response.get('PipelineExecutionSummaries', [])
    
    if not executions:
        logger.info(f"No executions found for pipeline: {pipeline_name}")
        return
    
    # Print execution summary
    print("\n" + "="*100)
    print(f"PIPELINE EXECUTIONS FOR: {pipeline_name}")
    print("="*100)
    print(f"{'Execution Name':<40} {'Status':<15} {'Start Time':<25} {'End Time':<25}")
    print("-"*100)
    
    for execution in executions:
        name = execution.get('PipelineExecutionDisplayName', 'Unnamed')
        status = execution.get('PipelineExecutionStatus', 'Unknown')
        start_time = execution.get('StartTime', '').strftime('%Y-%m-%d %H:%M:%S') if execution.get('StartTime') else 'N/A'
        end_time = execution.get('EndTime', '').strftime('%Y-%m-%d %H:%M:%S') if execution.get('EndTime') else 'N/A'
        
        print(f"{name:<40} {status:<15} {start_time:<25} {end_time:<25}")
    
    print("="*100)
    print(f"Total executions: {len(executions)}")


def delete_pipeline(args, config):
    """
    Delete a SageMaker Pipeline.
    
    Args:
        args: Command line arguments
        config: Project configuration
    """
    logger.info("Deleting SageMaker Pipeline")
    
    # Initialize pipeline builder
    pipeline_builder = SageMakerPipelineBuilder(
        aws_profile=args.aws_profile,
        region=args.region
    )
    
    # Set pipeline name
    pipeline_name = args.pipeline_name or config['pipeline']['name']
    
    # Delete pipeline
    pipeline_builder.sagemaker_client.delete_pipeline(
        PipelineName=pipeline_name
    )
    
    logger.info(f"Pipeline deleted: {pipeline_name}")


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting SageMaker Pipeline management")
    
    try:
        # Get project configuration
        config = get_config()
        
        # Execute requested action
        if args.create_pipeline:
            pipeline = create_preprocessing_pipeline(args, config)
            pipeline.upsert(role_arn=config['pipeline']['role_arn'])
            logger.info(f"Pipeline created/updated: {pipeline.name}")
            
        elif args.execute_pipeline:
            execution_arn = execute_pipeline(args, config)
            logger.info(f"Pipeline execution started: {execution_arn}")
            
        elif args.list_executions:
            list_pipeline_executions(args, config)
            
        elif args.delete_pipeline:
            delete_pipeline(args, config)
            
        else:
            logger.error("No action specified. Use --create-pipeline, --execute-pipeline, --list-executions, or --delete-pipeline")
            return 1
        
        logger.info("SageMaker Pipeline management completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)