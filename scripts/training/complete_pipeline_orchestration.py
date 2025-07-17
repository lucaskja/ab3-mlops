#!/usr/bin/env python3
"""
Complete SageMaker Pipeline Orchestration

This script provides comprehensive orchestration for the YOLOv11 SageMaker Pipeline,
including parameter management, execution monitoring, and failure handling with notifications.

Usage:
    python scripts/training/complete_pipeline_orchestration.py --create-pipeline
    python scripts/training/complete_pipeline_orchestration.py --execute-pipeline
    python scripts/training/complete_pipeline_orchestration.py --monitor-execution [execution-id]
    python scripts/training/complete_pipeline_orchestration.py --list-executions

Requirements addressed:
- 6.1: SageMaker Pipeline with data preprocessing, training, evaluation, and deployment steps
- 6.2: Pipeline artifacts stored in S3 with proper versioning
- 6.3: Experiments tracked in MLFlow with parameters and metrics
"""

import os
import sys
import argparse
import logging
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from botocore.exceptions import ClientError

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


class PipelineOrchestrator:
    """
    Orchestrates the complete SageMaker Pipeline lifecycle including creation,
    execution, monitoring, and failure handling.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the Pipeline Orchestrator.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Get project configuration
        self.config = get_config()
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker', region_name=region)
        self.cloudwatch_client = self.session.client('cloudwatch', region_name=region)
        self.sns_client = self.session.client('sns', region_name=region)
        self.events_client = self.session.client('events', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Initialize pipeline builder
        self.pipeline_builder = SageMakerPipelineBuilder(
            aws_profile=aws_profile,
            region=region
        )
        
        # Set default pipeline name
        self.pipeline_name = self.config['pipeline']['name']
        
        logger.info(f"Pipeline Orchestrator initialized for region: {region}")
        logger.info(f"Using AWS profile: {aws_profile}")
        logger.info(f"Default pipeline name: {self.pipeline_name}")
    
    def create_pipeline(
        self,
        pipeline_name: Optional[str] = None,
        input_data: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Pipeline:
        """
        Create a complete SageMaker Pipeline with all steps.
        
        Args:
            pipeline_name: Optional name for the pipeline (uses default if None)
            input_data: Optional S3 URI for input data
            hyperparameters: Optional hyperparameters for training
            description: Optional pipeline description
            
        Returns:
            Created pipeline
        """
        logger.info("Creating complete SageMaker Pipeline")
        
        # Set pipeline name
        pipeline_name = pipeline_name or self.pipeline_name
        
        # Set input data
        input_data = input_data or f"s3://{self.config['aws']['data_bucket']}/data/"
        
        # Set default hyperparameters if not provided
        if not hyperparameters:
            hyperparameters = {
                'model': 'yolov11n',
                'epochs': 100,
                'batch-size': 16,
                'learning-rate': 0.01,
                'img-size': 640
            }
        
        # Set pipeline description
        description = description or f"YOLOv11 Training Pipeline for {self.config['project']['name']}"
        
        # Create preprocessing script path
        script_dir = os.path.join(project_root, 'scripts', 'preprocessing')
        preprocessing_script_path = os.path.join(script_dir, 'preprocess_yolo_data.py')
        
        # Create training script path
        training_script_dir = os.path.join(project_root, 'scripts', 'training')
        training_script_path = os.path.join(training_script_dir, 'train_sagemaker_observability.py')
        
        # Create evaluation script path
        evaluation_script_path = os.path.join(training_script_dir, 'evaluate_model.py')
        
        # Create preprocessing step
        preprocessing_step = self.pipeline_builder.create_preprocessing_step(
            step_name="YOLOv11DataPreprocessing",
            script_path=preprocessing_script_path,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            input_data=input_data,
            output_prefix="yolov11-preprocessing",
            base_job_name="yolov11-preprocessing",
            environment={
                "PYTHONPATH": "/opt/ml/processing/code",
                "AWS_DEFAULT_REGION": self.region,
                "AWS_PROFILE": self.aws_profile
            }
        )
        
        # Create training step with observability
        training_step = self.pipeline_builder.create_training_step(
            step_name="YOLOv11Training",
            preprocessing_step=preprocessing_step,
            script_path=training_script_path,
            instance_type=self.config['training']['instance_type'],
            instance_count=self.config['training']['instance_count'],
            volume_size=self.config['training']['volume_size'],
            hyperparameters=hyperparameters,
            environment={
                "PYTHONPATH": "/opt/ml/code",
                "AWS_DEFAULT_REGION": self.region,
                "AWS_PROFILE": self.aws_profile,
                "MLFLOW_TRACKING_URI": self.config['mlflow']['tracking_uri'],
                "MLFLOW_EXPERIMENT_NAME": self.config['mlflow']['experiment_name']
            },
            base_job_name="yolov11-training",
            enable_mlflow=True,
            enable_powertools=True
        )
        
        # Create evaluation step
        evaluation_step = self.pipeline_builder.create_evaluation_step(
            step_name="YOLOv11Evaluation",
            training_step=training_step,
            preprocessing_step=preprocessing_step,
            script_path=evaluation_script_path,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            base_job_name="yolov11-evaluation"
        )
        
        # Create registration step
        registration_step = self.pipeline_builder.create_registration_step(
            step_name="YOLOv11Registration",
            training_step=training_step,
            evaluation_step=evaluation_step,
            model_name=self.config['model']['name'],
            model_package_group_name=f"{self.config['model']['name']}-group",
            model_approval_status="PendingManualApproval"
        )
        
        # Get Lambda function ARN for deployment
        lambda_function_arn = os.environ.get(
            "DEPLOY_LAMBDA_ARN", 
            self.config.get('deployment', {}).get('lambda_arn', '')
        )
        
        # Add deployment step if Lambda function ARN is provided
        if lambda_function_arn:
            deployment_step = self.pipeline_builder.create_deployment_step(
                step_name="YOLOv11Deployment",
                registration_step=registration_step,
                lambda_function_arn=lambda_function_arn,
                model_name=self.config['model']['name'],
                instance_type=self.config['inference']['instance_type'],
                initial_instance_count=self.config['inference']['initial_instance_count'],
                max_instance_count=self.config['inference']['max_instance_count']
            )
            
            # Add deployment step to pipeline
            steps = [preprocessing_step, training_step, evaluation_step, registration_step, deployment_step]
        else:
            logger.warning("Lambda function ARN not provided, skipping deployment step")
            steps = [preprocessing_step, training_step, evaluation_step, registration_step]
        
        # Create pipeline with all steps
        pipeline = self.pipeline_builder.create_pipeline(
            pipeline_name=pipeline_name,
            steps=steps,
            pipeline_description=description
        )
        
        # Create pipeline definition
        pipeline_definition = pipeline.definition()
        
        # Save pipeline definition to file
        definition_path = os.path.join(project_root, 'configs', f"{pipeline_name}-definition.json")
        with open(definition_path, 'w') as f:
            json.dump(json.loads(pipeline_definition), f, indent=2)
        
        logger.info(f"Pipeline definition saved to: {definition_path}")
        
        return pipeline
    
    def execute_pipeline(
        self,
        pipeline_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        execution_display_name: Optional[str] = None
    ) -> str:
        """
        Execute a SageMaker Pipeline.
        
        Args:
            pipeline_name: Optional name of the pipeline to execute (uses default if None)
            parameters: Optional parameters for the pipeline execution
            execution_display_name: Optional display name for the execution
            
        Returns:
            Execution ARN
        """
        logger.info("Executing SageMaker Pipeline")
        
        # Set pipeline name
        pipeline_name = pipeline_name or self.pipeline_name
        
        # Set execution display name
        if not execution_display_name:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            execution_display_name = f"{pipeline_name}-{timestamp}"
        
        try:
            # Get pipeline
            response = self.sagemaker_client.describe_pipeline(
                PipelineName=pipeline_name
            )
            
            # Start execution
            execution_params = {
                'PipelineName': pipeline_name,
                'PipelineExecutionDisplayName': execution_display_name,
                'PipelineExecutionDescription': f"Execution of {pipeline_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            # Add parameters if provided
            if parameters:
                execution_params['PipelineParameters'] = [
                    {'Name': key, 'Value': str(value)} for key, value in parameters.items()
                ]
            
            execution_response = self.sagemaker_client.start_pipeline_execution(**execution_params)
            
            execution_arn = execution_response['PipelineExecutionArn']
            logger.info(f"Pipeline execution started: {execution_arn}")
            
            # Set up CloudWatch alarm for execution failure
            self._setup_execution_alarm(execution_arn)
            
            return execution_arn
            
        except ClientError as e:
            error_message = f"Error executing pipeline: {str(e)}"
            logger.error(error_message)
            self._send_notification(
                subject=f"Pipeline Execution Error: {pipeline_name}",
                message=error_message
            )
            raise
    
    def _setup_execution_alarm(self, execution_arn: str) -> None:
        """
        Set up CloudWatch alarm for pipeline execution failure.
        
        Args:
            execution_arn: Pipeline execution ARN
        """
        try:
            # Extract execution ID from ARN
            execution_id = execution_arn.split('/')[-1]
            
            # Create CloudWatch alarm
            alarm_name = f"{self.pipeline_name}-execution-{execution_id}-alarm"
            
            self.cloudwatch_client.put_metric_alarm(
                AlarmName=alarm_name,
                AlarmDescription=f"Alarm for pipeline execution {execution_id}",
                ActionsEnabled=True,
                MetricName="ExecutionFailed",
                Namespace="AWS/SageMaker/ModelBuildingPipeline",
                Statistic="Sum",
                Dimensions=[
                    {
                        'Name': 'PipelineExecutionArn',
                        'Value': execution_arn
                    }
                ],
                Period=60,
                EvaluationPeriods=1,
                Threshold=1,
                ComparisonOperator="GreaterThanOrEqualToThreshold",
                TreatMissingData="notBreaching"
            )
            
            logger.info(f"CloudWatch alarm created: {alarm_name}")
            
            # Create SNS topic if it doesn't exist
            self._ensure_sns_topic_exists()
            
            # Add SNS action to alarm
            sns_topic_arn = self._get_sns_topic_arn()
            if sns_topic_arn:
                self.cloudwatch_client.put_metric_alarm(
                    AlarmName=alarm_name,
                    AlarmActions=[sns_topic_arn]
                )
                logger.info(f"SNS action added to alarm: {alarm_name}")
            
        except Exception as e:
            logger.error(f"Error setting up execution alarm: {str(e)}")
    
    def _ensure_sns_topic_exists(self) -> None:
        """
        Ensure SNS topic exists for notifications.
        """
        try:
            topic_name = self.config['notifications']['sns_topic']
            
            # Check if topic exists
            topics = self.sns_client.list_topics()
            for topic in topics.get('Topics', []):
                if topic_name in topic['TopicArn']:
                    logger.info(f"SNS topic already exists: {topic_name}")
                    return
            
            # Create topic if it doesn't exist
            self.sns_client.create_topic(Name=topic_name)
            logger.info(f"SNS topic created: {topic_name}")
            
        except Exception as e:
            logger.error(f"Error ensuring SNS topic exists: {str(e)}")
    
    def _get_sns_topic_arn(self) -> Optional[str]:
        """
        Get SNS topic ARN.
        
        Returns:
            SNS topic ARN or None if not found
        """
        try:
            topic_name = self.config['notifications']['sns_topic']
            
            # List topics
            topics = self.sns_client.list_topics()
            for topic in topics.get('Topics', []):
                if topic_name in topic['TopicArn']:
                    return topic['TopicArn']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting SNS topic ARN: {str(e)}")
            return None
    
    def _send_notification(self, subject: str, message: str) -> None:
        """
        Send SNS notification.
        
        Args:
            subject: Notification subject
            message: Notification message
        """
        try:
            sns_topic_arn = self._get_sns_topic_arn()
            if sns_topic_arn:
                self.sns_client.publish(
                    TopicArn=sns_topic_arn,
                    Subject=subject,
                    Message=message
                )
                logger.info(f"Notification sent to SNS topic: {sns_topic_arn}")
            else:
                logger.warning("SNS topic not found, notification not sent")
                
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
    
    def monitor_execution(self, execution_id: str, wait: bool = False) -> Dict[str, Any]:
        """
        Monitor a pipeline execution.
        
        Args:
            execution_id: Pipeline execution ID
            wait: Whether to wait for execution to complete
            
        Returns:
            Execution details
        """
        logger.info(f"Monitoring pipeline execution: {execution_id}")
        
        try:
            # Get pipeline execution
            response = self.sagemaker_client.describe_pipeline_execution(
                PipelineExecutionArn=f"arn:aws:sagemaker:{self.region}:{self.config['aws']['account_id']}:pipeline/{self.pipeline_name}/execution/{execution_id}"
            )
            
            status = response['PipelineExecutionStatus']
            logger.info(f"Pipeline execution status: {status}")
            
            # Wait for execution to complete if requested
            if wait and status in ['Executing', 'Stopping']:
                logger.info("Waiting for pipeline execution to complete...")
                
                while status in ['Executing', 'Stopping']:
                    time.sleep(30)
                    response = self.sagemaker_client.describe_pipeline_execution(
                        PipelineExecutionArn=f"arn:aws:sagemaker:{self.region}:{self.config['aws']['account_id']}:pipeline/{self.pipeline_name}/execution/{execution_id}"
                    )
                    new_status = response['PipelineExecutionStatus']
                    
                    if new_status != status:
                        logger.info(f"Pipeline execution status changed: {status} -> {new_status}")
                        status = new_status
                
                logger.info(f"Pipeline execution completed with status: {status}")
                
                # Send notification on failure
                if status in ['Failed', 'Stopped']:
                    failure_reason = response.get('FailureReason', 'Unknown reason')
                    error_message = f"Pipeline execution {execution_id} {status.lower()}: {failure_reason}"
                    logger.error(error_message)
                    self._send_notification(
                        subject=f"Pipeline Execution {status}: {self.pipeline_name}",
                        message=error_message
                    )
            
            # Get step details
            step_details = []
            if 'PipelineExecutionArn' in response:
                steps_response = self.sagemaker_client.list_pipeline_execution_steps(
                    PipelineExecutionArn=response['PipelineExecutionArn']
                )
                step_details = steps_response.get('PipelineExecutionSteps', [])
            
            # Add step details to response
            response['StepDetails'] = step_details
            
            return response
            
        except ClientError as e:
            error_message = f"Error monitoring pipeline execution: {str(e)}"
            logger.error(error_message)
            self._send_notification(
                subject=f"Pipeline Monitoring Error: {self.pipeline_name}",
                message=error_message
            )
            raise
    
    def list_executions(
        self,
        pipeline_name: Optional[str] = None,
        max_results: int = 20,
        sort_by: str = "CreationTime",
        sort_order: str = "Descending"
    ) -> List[Dict[str, Any]]:
        """
        List pipeline executions.
        
        Args:
            pipeline_name: Optional name of the pipeline (uses default if None)
            max_results: Maximum number of results to return
            sort_by: Field to sort by
            sort_order: Sort order
            
        Returns:
            List of pipeline executions
        """
        logger.info("Listing pipeline executions")
        
        # Set pipeline name
        pipeline_name = pipeline_name or self.pipeline_name
        
        try:
            # List executions
            response = self.sagemaker_client.list_pipeline_executions(
                PipelineName=pipeline_name,
                SortBy=sort_by,
                SortOrder=sort_order,
                MaxResults=max_results
            )
            
            executions = response.get('PipelineExecutionSummaries', [])
            
            return executions
            
        except ClientError as e:
            error_message = f"Error listing pipeline executions: {str(e)}"
            logger.error(error_message)
            raise
    
    def print_execution_summary(self, executions: List[Dict[str, Any]]) -> None:
        """
        Print execution summary.
        
        Args:
            executions: List of pipeline executions
        """
        if not executions:
            logger.info(f"No executions found for pipeline: {self.pipeline_name}")
            return
        
        # Print execution summary
        print("\n" + "="*100)
        print(f"PIPELINE EXECUTIONS FOR: {self.pipeline_name}")
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
    
    def print_execution_details(self, execution_details: Dict[str, Any]) -> None:
        """
        Print execution details.
        
        Args:
            execution_details: Pipeline execution details
        """
        print("\n" + "="*100)
        print(f"PIPELINE EXECUTION DETAILS")
        print("="*100)
        
        # Print execution information
        print(f"Execution Name: {execution_details.get('PipelineExecutionDisplayName', 'Unnamed')}")
        print(f"Status: {execution_details.get('PipelineExecutionStatus', 'Unknown')}")
        print(f"Start Time: {execution_details.get('StartTime', '').strftime('%Y-%m-%d %H:%M:%S') if execution_details.get('StartTime') else 'N/A'}")
        print(f"End Time: {execution_details.get('EndTime', '').strftime('%Y-%m-%d %H:%M:%S') if execution_details.get('EndTime') else 'N/A'}")
        
        if 'FailureReason' in execution_details:
            print(f"Failure Reason: {execution_details['FailureReason']}")
        
        # Print step details
        step_details = execution_details.get('StepDetails', [])
        if step_details:
            print("\nStep Details:")
            print("-"*100)
            print(f"{'Step Name':<30} {'Status':<15} {'Start Time':<25} {'End Time':<25}")
            print("-"*100)
            
            for step in step_details:
                step_name = step.get('StepName', 'Unnamed')
                status = step.get('StepStatus', 'Unknown')
                start_time = step.get('StartTime', '').strftime('%Y-%m-%d %H:%M:%S') if step.get('StartTime') else 'N/A'
                end_time = step.get('EndTime', '').strftime('%Y-%m-%d %H:%M:%S') if step.get('EndTime') else 'N/A'
                
                print(f"{step_name:<30} {status:<15} {start_time:<25} {end_time:<25}")
                
                if 'FailureReason' in step:
                    print(f"  Failure Reason: {step['FailureReason']}")
        
        print("="*100)
    
    def setup_eventbridge_rule(self) -> None:
        """
        Set up EventBridge rule for pipeline notifications.
        """
        logger.info("Setting up EventBridge rule for pipeline notifications")
        
        try:
            rule_name = self.config['notifications']['eventbridge_rule']
            
            # Create EventBridge rule
            self.events_client.put_rule(
                Name=rule_name,
                EventPattern=json.dumps({
                    "source": ["aws.sagemaker"],
                    "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
                    "detail": {
                        "pipelineName": [self.pipeline_name],
                        "currentPipelineExecutionStatus": ["Failed", "Stopped", "Succeeded"]
                    }
                }),
                State="ENABLED",
                Description=f"Rule for {self.pipeline_name} pipeline status changes"
            )
            
            logger.info(f"EventBridge rule created: {rule_name}")
            
            # Add SNS target to rule
            sns_topic_arn = self._get_sns_topic_arn()
            if sns_topic_arn:
                self.events_client.put_targets(
                    Rule=rule_name,
                    Targets=[
                        {
                            'Id': f"{rule_name}-target",
                            'Arn': sns_topic_arn,
                            'InputTransformer': {
                                'InputPathsMap': {
                                    'pipeline': '$.detail.pipelineName',
                                    'execution': '$.detail.executionName',
                                    'status': '$.detail.currentPipelineExecutionStatus'
                                },
                                'InputTemplate': 'Pipeline <pipeline> execution <execution> status changed to <status>'
                            }
                        }
                    ]
                )
                logger.info(f"SNS target added to EventBridge rule: {rule_name}")
            else:
                logger.warning("SNS topic not found, target not added to EventBridge rule")
                
        except Exception as e:
            logger.error(f"Error setting up EventBridge rule: {str(e)}")
    
    def delete_pipeline(self, pipeline_name: Optional[str] = None) -> None:
        """
        Delete a SageMaker Pipeline.
        
        Args:
            pipeline_name: Optional name of the pipeline to delete (uses default if None)
        """
        logger.info("Deleting SageMaker Pipeline")
        
        # Set pipeline name
        pipeline_name = pipeline_name or self.pipeline_name
        
        try:
            # Delete pipeline
            self.sagemaker_client.delete_pipeline(
                PipelineName=pipeline_name
            )
            
            logger.info(f"Pipeline deleted: {pipeline_name}")
            
        except ClientError as e:
            error_message = f"Error deleting pipeline: {str(e)}"
            logger.error(error_message)
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Complete SageMaker Pipeline Orchestration')
    
    # Actions
    parser.add_argument('--create-pipeline', action='store_true',
                       help='Create SageMaker Pipeline')
    parser.add_argument('--execute-pipeline', action='store_true',
                       help='Execute SageMaker Pipeline')
    parser.add_argument('--monitor-execution', type=str, metavar='EXECUTION_ID',
                       help='Monitor pipeline execution')
    parser.add_argument('--list-executions', action='store_true',
                       help='List pipeline executions')
    parser.add_argument('--setup-notifications', action='store_true',
                       help='Set up pipeline notifications')
    parser.add_argument('--delete-pipeline', action='store_true',
                       help='Delete SageMaker Pipeline')
    
    # Pipeline configuration
    parser.add_argument('--pipeline-name', type=str,
                       help='Name of the pipeline')
    parser.add_argument('--input-data', type=str,
                       help='S3 URI of input data')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for execution to complete')
    
    # AWS configuration
    parser.add_argument('--aws-profile', type=str, default='ab',
                       help='AWS profile to use')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting SageMaker Pipeline orchestration")
    
    try:
        # Get project configuration
        config = get_config()
        
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator(
            aws_profile=args.aws_profile,
            region=args.region
        )
        
        # Execute requested action
        if args.create_pipeline:
            pipeline = orchestrator.create_pipeline(
                pipeline_name=args.pipeline_name,
                input_data=args.input_data
            )
            pipeline.upsert(role_arn=config['pipeline']['role_arn'])
            logger.info(f"Pipeline created/updated: {pipeline.name}")
            
        elif args.execute_pipeline:
            execution_arn = orchestrator.execute_pipeline(
                pipeline_name=args.pipeline_name
            )
            logger.info(f"Pipeline execution started: {execution_arn}")
            
            # Monitor execution if wait flag is set
            if args.wait:
                execution_id = execution_arn.split('/')[-1]
                execution_details = orchestrator.monitor_execution(execution_id, wait=True)
                orchestrator.print_execution_details(execution_details)
            
        elif args.monitor_execution:
            execution_details = orchestrator.monitor_execution(args.monitor_execution, wait=args.wait)
            orchestrator.print_execution_details(execution_details)
            
        elif args.list_executions:
            executions = orchestrator.list_executions(
                pipeline_name=args.pipeline_name
            )
            orchestrator.print_execution_summary(executions)
            
        elif args.setup_notifications:
            orchestrator._ensure_sns_topic_exists()
            orchestrator.setup_eventbridge_rule()
            logger.info("Pipeline notifications set up successfully")
            
        elif args.delete_pipeline:
            orchestrator.delete_pipeline(
                pipeline_name=args.pipeline_name
            )
            
        else:
            logger.error("No action specified. Use --create-pipeline, --execute-pipeline, --monitor-execution, --list-executions, --setup-notifications, or --delete-pipeline")
            return 1
        
        logger.info("SageMaker Pipeline orchestration completed successfully!")
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