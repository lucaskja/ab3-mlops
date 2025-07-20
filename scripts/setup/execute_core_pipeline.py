#!/usr/bin/env python3
"""
Execute Core SageMaker Pipeline

This script allows ML Engineers to execute the core YOLOv11 training pipeline
with custom parameters.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import boto3
import sagemaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorePipelineExecutor:
    """Executes the core SageMaker Pipeline for YOLOv11 training."""
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the pipeline executor.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Set up AWS session
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker')
        
        logger.info(f"Initialized CorePipelineExecutor with profile: {aws_profile}")
        logger.info(f"Region: {region}")
    
    def list_pipelines(self) -> list:
        """List available SageMaker pipelines."""
        try:
            response = self.sagemaker_client.list_pipelines(
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=50
            )
            
            pipelines = response.get('PipelineSummaries', [])
            
            # Filter for core setup pipelines
            core_pipelines = [
                p for p in pipelines 
                if 'sagemaker-core-setup' in p['PipelineName'] or 'yolov11' in p['PipelineName'].lower()
            ]
            
            return core_pipelines
            
        except Exception as e:
            logger.error(f"Failed to list pipelines: {str(e)}")
            return []
    
    def get_pipeline_definition(self, pipeline_name: str) -> Dict[str, Any]:
        """Get pipeline definition."""
        try:
            response = self.sagemaker_client.describe_pipeline(
                PipelineName=pipeline_name
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get pipeline definition: {str(e)}")
            return {}
    
    def execute_pipeline(self, 
                        pipeline_name: str, 
                        parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a SageMaker pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to execute
            parameters: Pipeline parameters to override
            
        Returns:
            Pipeline execution ARN
        """
        logger.info(f"Executing pipeline: {pipeline_name}")
        
        # Prepare execution parameters
        execution_params = []
        if parameters:
            for key, value in parameters.items():
                execution_params.append({
                    'Name': key,
                    'Value': str(value)
                })
        
        # Generate execution name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        execution_name = f"{pipeline_name}-execution-{timestamp}"
        
        try:
            # Start pipeline execution
            response = self.sagemaker_client.start_pipeline_execution(
                PipelineName=pipeline_name,
                PipelineExecutionDisplayName=execution_name,
                PipelineParameters=execution_params
            )
            
            execution_arn = response['PipelineExecutionArn']
            logger.info(f"Pipeline execution started: {execution_arn}")
            
            return execution_arn
            
        except Exception as e:
            logger.error(f"Failed to execute pipeline: {str(e)}")
            raise
    
    def monitor_execution(self, execution_arn: str) -> Dict[str, Any]:
        """Monitor pipeline execution status."""
        try:
            response = self.sagemaker_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            status = response['PipelineExecutionStatus']
            creation_time = response['CreationTime']
            
            result = {
                'execution_arn': execution_arn,
                'status': status,
                'creation_time': creation_time.isoformat(),
                'pipeline_name': response['PipelineName']
            }
            
            if 'LastModifiedTime' in response:
                result['last_modified'] = response['LastModifiedTime'].isoformat()
            
            if 'FailureReason' in response:
                result['failure_reason'] = response['FailureReason']
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to monitor execution: {str(e)}")
            return {}
    
    def list_executions(self, pipeline_name: str) -> list:
        """List pipeline executions."""
        try:
            response = self.sagemaker_client.list_pipeline_executions(
                PipelineName=pipeline_name,
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=20
            )
            
            return response.get('PipelineExecutionSummaries', [])
            
        except Exception as e:
            logger.error(f"Failed to list executions: {str(e)}")
            return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Execute Core SageMaker Pipeline")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--pipeline-name", help="Pipeline name to execute")
    parser.add_argument("--list-pipelines", action="store_true", help="List available pipelines")
    parser.add_argument("--list-executions", help="List executions for a pipeline")
    parser.add_argument("--monitor", help="Monitor execution by ARN")
    
    # Pipeline parameters
    parser.add_argument("--input-data", help="S3 path to input data")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--instance-type", help="Training instance type")
    parser.add_argument("--model-variant", help="YOLOv11 model variant")
    
    args = parser.parse_args()
    
    try:
        # Verify AWS profile
        session = boto3.Session(profile_name=args.profile)
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        
        logger.info(f"Using AWS profile: {args.profile}")
        logger.info(f"Account ID: {identity['Account']}")
        
        # Create executor
        executor = CorePipelineExecutor(aws_profile=args.profile, region=args.region)
        
        # Handle different operations
        if args.list_pipelines:
            pipelines = executor.list_pipelines()
            
            if pipelines:
                print("\n📋 Available Core Pipelines:")
                print("="*60)
                for i, pipeline in enumerate(pipelines, 1):
                    print(f"{i}. {pipeline['PipelineName']}")
                    print(f"   Status: {pipeline['PipelineStatus']}")
                    print(f"   Created: {pipeline['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')}")
                    if 'PipelineDescription' in pipeline:
                        print(f"   Description: {pipeline['PipelineDescription']}")
                    print()
            else:
                print("No core pipelines found.")
                print("Run create_core_pipeline.py first to create a pipeline.")
            
            return
        
        if args.list_executions:
            executions = executor.list_executions(args.list_executions)
            
            if executions:
                print(f"\n📊 Recent Executions for {args.list_executions}:")
                print("="*60)
                for i, execution in enumerate(executions, 1):
                    print(f"{i}. {execution['PipelineExecutionDisplayName']}")
                    print(f"   Status: {execution['PipelineExecutionStatus']}")
                    print(f"   Started: {execution['StartTime'].strftime('%Y-%m-%d %H:%M:%S')}")
                    if 'EndTime' in execution:
                        print(f"   Ended: {execution['EndTime'].strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   ARN: {execution['PipelineExecutionArn']}")
                    print()
            else:
                print(f"No executions found for pipeline: {args.list_executions}")
            
            return
        
        if args.monitor:
            result = executor.monitor_execution(args.monitor)
            
            if result:
                print(f"\n📈 Pipeline Execution Status:")
                print("="*60)
                print(f"Pipeline: {result['pipeline_name']}")
                print(f"Status: {result['status']}")
                print(f"Created: {result['creation_time']}")
                if 'last_modified' in result:
                    print(f"Last Modified: {result['last_modified']}")
                if 'failure_reason' in result:
                    print(f"Failure Reason: {result['failure_reason']}")
                print(f"ARN: {result['execution_arn']}")
            else:
                print("Failed to get execution status")
            
            return
        
        if not args.pipeline_name:
            print("Please specify --pipeline-name or use --list-pipelines to see available pipelines")
            return
        
        # Prepare parameters
        parameters = {}
        if args.input_data:
            parameters['InputData'] = args.input_data
        if args.epochs:
            parameters['Epochs'] = args.epochs
        if args.batch_size:
            parameters['BatchSize'] = args.batch_size
        if args.learning_rate:
            parameters['LearningRate'] = args.learning_rate
        if args.instance_type:
            parameters['TrainingInstanceType'] = args.instance_type
        if args.model_variant:
            parameters['ModelVariant'] = args.model_variant
        
        # Execute pipeline
        execution_arn = executor.execute_pipeline(args.pipeline_name, parameters)
        
        print("\n" + "="*60)
        print("🚀 Pipeline Execution Started!")
        print("="*60)
        print(f"Pipeline: {args.pipeline_name}")
        print(f"Execution ARN: {execution_arn}")
        
        if parameters:
            print(f"\nParameters:")
            for key, value in parameters.items():
                print(f"  {key}: {value}")
        
        print(f"\n📋 Monitor Progress:")
        print(f"python {__file__} --monitor {execution_arn} --profile {args.profile}")
        
        print(f"\n💡 SageMaker Console:")
        print(f"https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/pipelines")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
