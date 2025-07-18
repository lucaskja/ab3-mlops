#!/usr/bin/env python3
"""
Automated Model Retraining Trigger

This script implements automated model retraining triggers based on data drift detection
and model performance degradation. It sets up EventBridge rules to automatically start
SageMaker pipelines when drift is detected, and implements an approval workflow for model updates.

Requirements addressed:
- 5.2: Capture input data and predictions for monitoring
- 5.3: Trigger alerts through EventBridge when data drift is detected
- 6.1: Complete SageMaker Pipeline implementation with automated retraining
"""

import os
import json
import logging
import argparse
import boto3
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

# Import project modules
from src.monitoring.drift_detection import DriftDetector
from src.pipeline.event_bridge_integration import EventBridgeIntegration
from src.pipeline.sagemaker_pipeline import SageMakerPipelineBuilder
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedRetrainingManager:
    """
    Manages automated model retraining triggers based on drift detection.
    
    This class provides functionality for:
    - Setting up data drift detection thresholds
    - Creating scheduled model evaluation jobs
    - Configuring EventBridge rules for automated pipeline triggering
    - Implementing approval workflows for model updates
    - Sending notifications for retraining events
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the automated retraining manager.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker', region_name=region)
        self.events_client = self.session.client('events', region_name=region)
        self.lambda_client = self.session.client('lambda', region_name=region)
        self.sns_client = self.session.client('sns', region_name=region)
        self.s3_client = self.session.client('s3', region_name=region)
        
        # Get project configuration
        self.project_config = get_config()
        self.project_name = self.project_config['project']['name']
        
        # Initialize components
        self.event_bridge = EventBridgeIntegration(aws_profile=aws_profile, region=region)
        self.pipeline_builder = SageMakerPipelineBuilder(aws_profile=aws_profile, region=region)
        
        logger.info(f"Automated retraining manager initialized for region: {region}")    
        
    def create_drift_detection_lambda(
        self,
        endpoint_name: str,
        pipeline_name: str,
        drift_threshold: float = 0.1,
        lambda_name: Optional[str] = None,
        bucket: Optional[str] = None,
        prefix: str = "monitoring"
    ) -> str:
        """
        Create a Lambda function to evaluate drift and trigger retraining.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            pipeline_name: Name of the SageMaker pipeline to trigger
            drift_threshold: Threshold for drift detection
            lambda_name: Name for the Lambda function (optional)
            bucket: S3 bucket for monitoring data
            prefix: S3 prefix for monitoring data
            
        Returns:
            ARN of the created Lambda function
        """
        logger.info(f"Creating drift detection Lambda for endpoint: {endpoint_name}")
        
        # Generate Lambda name if not provided
        if not lambda_name:
            lambda_name = f"{self.project_name}-drift-detection-{endpoint_name}"
        
        # Set default bucket if not provided
        if not bucket:
            bucket = self.pipeline_builder.default_bucket
        
        # Create Lambda function code
        lambda_code = f"""
import json
import boto3
import os
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

# Configuration
ENDPOINT_NAME = "{endpoint_name}"
PIPELINE_NAME = "{pipeline_name}"
DRIFT_THRESHOLD = {drift_threshold}
BUCKET = "{bucket}"
PREFIX = "{prefix}"
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')

def lambda_handler(event, context):
    # Lambda handler for drift detection and retraining trigger.
    """
    logger.info(f"Received event: {{json.dumps(event)}}")
    
    try:
        # Extract monitoring schedule name from event
        monitoring_schedule_name = event.get('detail', {}).get('monitoringScheduleName', '')
        if not monitoring_schedule_name:
            logger.warning("No monitoring schedule name found in event")
            return {
                'statusCode': 400,
                'body': json.dumps('No monitoring schedule name found in event')
            }
        
        # Get the latest monitoring execution
        response = sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=monitoring_schedule_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        executions = response.get('MonitoringExecutionSummaries', [])
        if not executions:
            logger.warning(f"No monitoring executions found for schedule: {{monitoring_schedule_name}}")
            return {
                'statusCode': 404,
                'body': json.dumps('No monitoring executions found')
            }
        
        # Get the latest execution
        latest_execution = executions[0]
        execution_arn = latest_execution.get('MonitoringExecutionArn')
        status = latest_execution.get('MonitoringExecutionStatus')
        
        if status != 'Completed':
            logger.warning(f"Latest monitoring execution not completed. Status: {{status}}")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Monitoring execution status: {{status}}')
            }
        
        # Get execution details
        execution_details = sagemaker_client.describe_monitoring_execution(
            MonitoringExecutionArn=execution_arn
        )
        
        # Get output path
        output_path = None
        outputs = execution_details.get('ProcessingOutputConfig', {}).get('Outputs', [])
        for output in outputs:
            if output.get('OutputName') == 'evaluation':
                output_path = output.get('S3Output', {}).get('S3Uri')
                break
        
        if not output_path:
            logger.warning("No evaluation output found in execution details")
            return {
                'statusCode': 404,
                'body': json.dumps('No evaluation output found')
            }
        
        # Parse S3 URI
        s3_parts = output_path.replace('s3://', '').split('/')
        bucket_name = s3_parts[0]
        key_prefix = '/'.join(s3_parts[1:])
        
        # Download and parse violations file
        violations_key = f"{{key_prefix}}/constraint_violations.json"
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=violations_key
            )
            violations_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Check for violations
            violations = violations_data.get('violations', [])
            violations_count = len(violations)
            
            logger.info(f"Found {{violations_count}} violations")
            
            # Determine if retraining is needed
            retraining_needed = violations_count > 0
            
            if retraining_needed:
                logger.info(f"Drift detected. Triggering retraining pipeline: {{PIPELINE_NAME}}")
                
                # Start pipeline execution
                pipeline_response = sagemaker_client.start_pipeline_execution(
                    PipelineName=PIPELINE_NAME,
                    PipelineParameters=[
                        {
                            'Name': 'EndpointName',
                            'Value': ENDPOINT_NAME
                        },
                        {
                            'Name': 'RetrainingReason',
                            'Value': 'DriftDetected'
                        }
                    ]
                )
                
                execution_arn = pipeline_response['PipelineExecutionArn']
                
                # Send notification if SNS topic is configured
                if SNS_TOPIC_ARN:
                    sns_client.publish(
                        TopicArn=SNS_TOPIC_ARN,
                        Subject=f"Model Retraining Triggered for {{ENDPOINT_NAME}}",
                        Message=f"Model drift detected for endpoint {{ENDPOINT_NAME}}. "
                                f"Retraining pipeline {{PIPELINE_NAME}} has been triggered. "
                                f"Pipeline execution ARN: {{execution_arn}}. "
                                f"Number of violations: {{violations_count}}."
                    )
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'Retraining pipeline triggered',
                        'pipeline_name': PIPELINE_NAME,
                        'execution_arn': execution_arn,
                        'violations_count': violations_count
                    })
                }
            else:
                logger.info("No drift detected. Retraining not needed.")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'No drift detected',
                        'violations_count': violations_count
                    })
                }
                
        except Exception as e:
            logger.error(f"Error processing violations file: {{str(e)}}")
            return {
                'statusCode': 500,
                'body': json.dumps(f'Error: {{str(e)}}')
            }
            
    except Exception as e:
        logger.error(f"Error in lambda handler: {{str(e)}}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {{str(e)}}')
        }
"""
        
        # Create IAM role for Lambda
        iam_client = self.session.client('iam')
        role_name = f"{lambda_name}-role"
        
        try:
            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description=f"Role for {lambda_name} Lambda function"
            )
            
            role_arn = role_response['Role']['Arn']
            
            # Attach policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSNSFullAccess"
            )
            
            # Wait for role to propagate
            logger.info(f"Waiting for IAM role to propagate...")
            time.sleep(10)
            
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = iam_client.get_role(RoleName=role_name)
            role_arn = role_response['Role']['Arn']
            logger.info(f"Using existing IAM role: {role_name}")
        
        # Create Lambda function
        lambda_zip_file = "/tmp/lambda_function.zip"
        
        # Write Lambda code to file
        with open("/tmp/lambda_function.py", "w") as f:
            f.write(lambda_code)
        
        # Create ZIP file
        import zipfile
        with zipfile.ZipFile(lambda_zip_file, "w") as z:
            z.write("/tmp/lambda_function.py", "lambda_function.py")
        
        # Read ZIP file
        with open(lambda_zip_file, "rb") as f:
            zip_bytes = f.read()
        
        # Create Lambda function
        try:
            response = self.lambda_client.create_function(
                FunctionName=lambda_name,
                Runtime="python3.8",
                Role=role_arn,
                Handler="lambda_function.lambda_handler",
                Code={
                    "ZipFile": zip_bytes
                },
                Description=f"Lambda function to detect drift and trigger retraining for {endpoint_name}",
                Timeout=60,
                MemorySize=128,
                Publish=True
            )
            
            lambda_arn = response["FunctionArn"]
            logger.info(f"Created Lambda function: {lambda_name} (ARN: {lambda_arn})")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            # Function already exists, get its ARN
            response = self.lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn = response["Configuration"]["FunctionArn"]
            
            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=lambda_name,
                ZipFile=zip_bytes,
                Publish=True
            )
            
            logger.info(f"Updated existing Lambda function: {lambda_name} (ARN: {lambda_arn})")
        
        return lambda_arn
        
    def create_scheduled_evaluation_lambda(
        self,
        endpoint_name: str,
        pipeline_name: str,
        performance_threshold: float = 0.7,
        lambda_name: Optional[str] = None,
        evaluation_data: Optional[str] = None
    ) -> str:
        """
        Create a Lambda function for scheduled model evaluation.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            pipeline_name: Name of the SageMaker pipeline to trigger
            performance_threshold: Threshold for model performance
            lambda_name: Name for the Lambda function (optional)
            evaluation_data: S3 path to evaluation data (optional)
            
        Returns:
            ARN of the created Lambda function
        """
        logger.info(f"Creating scheduled evaluation Lambda for endpoint: {endpoint_name}")
        
        # Generate Lambda name if not provided
        if not lambda_name:
            lambda_name = f"{self.project_name}-scheduled-evaluation-{endpoint_name}"
        
        # Set default evaluation data if not provided
        if not evaluation_data:
            evaluation_data = f"s3://{self.pipeline_builder.default_bucket}/evaluation-data/{endpoint_name}"
        
        # Create Lambda function code
        lambda_code = f"""
import json
import boto3
import os
import logging
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

# Configuration
ENDPOINT_NAME = "{endpoint_name}"
PIPELINE_NAME = "{pipeline_name}"
PERFORMANCE_THRESHOLD = {performance_threshold}
EVALUATION_DATA = "{evaluation_data}"
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')

def lambda_handler(event, context):
    # Lambda handler for scheduled model evaluation.
    """
    logger.info(f"Starting scheduled evaluation for endpoint: {{ENDPOINT_NAME}}")
    
    try:
        # Parse S3 URI
        s3_parts = EVALUATION_DATA.replace('s3://', '').split('/')
        bucket_name = s3_parts[0]
        key_prefix = '/'.join(s3_parts[1:])
        
        # List evaluation data files
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=key_prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No evaluation data found at: {{EVALUATION_DATA}}")
            return {
                'statusCode': 404,
                'body': json.dumps('No evaluation data found')
            }
        
        # Get the latest evaluation data file
        latest_file = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
        latest_key = latest_file['Key']
        
        # Download evaluation data
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=latest_key
        )
        
        evaluation_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Prepare for batch inference
        test_data = evaluation_data.get('test_data', [])
        ground_truth = evaluation_data.get('ground_truth', [])
        
        if not test_data or not ground_truth:
            logger.warning("Invalid evaluation data format")
            return {
                'statusCode': 400,
                'body': json.dumps('Invalid evaluation data format')
            }
        
        # Perform batch inference
        predictions = []
        for data_point in test_data:
            try:
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='application/json',
                    Body=json.dumps(data_point)
                )
                
                result = json.loads(response['Body'].read().decode())
                predictions.append(result)
                
            except Exception as e:
                logger.error(f"Error invoking endpoint: {{str(e)}}")
                continue
        
        # Calculate performance metrics
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            if pred == truth:  # Simplified comparison, adjust based on your model output format
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"Model accuracy: {{accuracy:.4f}}")
        
        # Determine if retraining is needed
        retraining_needed = accuracy < PERFORMANCE_THRESHOLD
        
        if retraining_needed:
            logger.info(f"Performance below threshold. Triggering retraining pipeline: {{PIPELINE_NAME}}")
            
            # Start pipeline execution
            pipeline_response = sagemaker_client.start_pipeline_execution(
                PipelineName=PIPELINE_NAME,
                PipelineParameters=[
                    {
                        'Name': 'EndpointName',
                        'Value': ENDPOINT_NAME
                    },
                    {
                        'Name': 'RetrainingReason',
                        'Value': 'PerformanceDegradation'
                    },
                    {
                        'Name': 'CurrentAccuracy',
                        'Value': str(accuracy)
                    }
                ]
            )
            
            execution_arn = pipeline_response['PipelineExecutionArn']
            
            # Send notification if SNS topic is configured
            if SNS_TOPIC_ARN:
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Subject=f"Model Retraining Triggered for {{ENDPOINT_NAME}}",
                    Message=f"Model performance degradation detected for endpoint {{ENDPOINT_NAME}}. "
                            f"Current accuracy: {{accuracy:.4f}}, Threshold: {{PERFORMANCE_THRESHOLD}}. "
                            f"Retraining pipeline {{PIPELINE_NAME}} has been triggered. "
                            f"Pipeline execution ARN: {{execution_arn}}."
                )
            
            # Store evaluation results
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            evaluation_result = {
                'timestamp': timestamp,
                'endpoint_name': ENDPOINT_NAME,
                'accuracy': accuracy,
                'threshold': PERFORMANCE_THRESHOLD,
                'retraining_triggered': True,
                'pipeline_execution_arn': execution_arn
            }
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{{key_prefix}}/evaluation_results/{{timestamp}}.json",
                Body=json.dumps(evaluation_result)
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Retraining pipeline triggered',
                    'pipeline_name': PIPELINE_NAME,
                    'execution_arn': execution_arn,
                    'accuracy': accuracy,
                    'threshold': PERFORMANCE_THRESHOLD
                })
            }
        else:
            logger.info("Model performance above threshold. Retraining not needed.")
            
            # Store evaluation results
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            evaluation_result = {
                'timestamp': timestamp,
                'endpoint_name': ENDPOINT_NAME,
                'accuracy': accuracy,
                'threshold': PERFORMANCE_THRESHOLD,
                'retraining_triggered': False
            }
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{{key_prefix}}/evaluation_results/{{timestamp}}.json",
                Body=json.dumps(evaluation_result)
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Model performance satisfactory',
                    'accuracy': accuracy,
                    'threshold': PERFORMANCE_THRESHOLD
                })
            }
            
    except Exception as e:
        logger.error(f"Error in lambda handler: {{str(e)}}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {{str(e)}}')
        }
"""
        
        # Create IAM role for Lambda
        iam_client = self.session.client('iam')
        role_name = f"{lambda_name}-role"
        
        try:
            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description=f"Role for {lambda_name} Lambda function"
            )
            
            role_arn = role_response['Role']['Arn']
            
            # Attach policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSNSFullAccess"
            )
            
            # Wait for role to propagate
            logger.info(f"Waiting for IAM role to propagate...")
            time.sleep(10)
            
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = iam_client.get_role(RoleName=role_name)
            role_arn = role_response['Role']['Arn']
            logger.info(f"Using existing IAM role: {role_name}")
        
        # Create Lambda function
        lambda_zip_file = "/tmp/lambda_function.zip"
        
        # Write Lambda code to file
        with open("/tmp/lambda_function.py", "w") as f:
            f.write(lambda_code)
        
        # Create ZIP file
        import zipfile
        with zipfile.ZipFile(lambda_zip_file, "w") as z:
            z.write("/tmp/lambda_function.py", "lambda_function.py")
        
        # Read ZIP file
        with open(lambda_zip_file, "rb") as f:
            zip_bytes = f.read()
        
        # Create Lambda function
        try:
            response = self.lambda_client.create_function(
                FunctionName=lambda_name,
                Runtime="python3.8",
                Role=role_arn,
                Handler="lambda_function.lambda_handler",
                Code={
                    "ZipFile": zip_bytes
                },
                Description=f"Lambda function for scheduled model evaluation of {endpoint_name}",
                Timeout=300,  # 5 minutes
                MemorySize=256,
                Publish=True
            )
            
            lambda_arn = response["FunctionArn"]
            logger.info(f"Created Lambda function: {lambda_name} (ARN: {lambda_arn})")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            # Function already exists, get its ARN
            response = self.lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn = response["Configuration"]["FunctionArn"]
            
            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=lambda_name,
                ZipFile=zip_bytes,
                Publish=True
            )
            
            logger.info(f"Updated existing Lambda function: {lambda_name} (ARN: {lambda_arn})")
        
        return lambda_arn
        
    def create_scheduled_evaluation_rule(
        self,
        lambda_arn: str,
        schedule_expression: str = "rate(1 day)",
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule for scheduled model evaluation.
        
        Args:
            lambda_arn: ARN of the Lambda function to trigger
            schedule_expression: Schedule expression (rate or cron)
            rule_name: Name for the rule (optional)
            
        Returns:
            Rule details
        """
        logger.info(f"Creating scheduled evaluation rule with schedule: {schedule_expression}")
        
        # Generate rule name if not provided
        if not rule_name:
            lambda_name = lambda_arn.split(":")[-1]
            rule_name = f"{lambda_name}-schedule"
        
        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule_expression,
            State="ENABLED",
            Description=f"Scheduled rule for model evaluation: {schedule_expression}"
        )
        
        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")
        
        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    "Id": f"{rule_name}-target",
                    "Arn": lambda_arn
                }
            ]
        )
        
        # Add Lambda permission
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_arn,
                StatementId=f"{rule_name}-permission",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn
            )
            logger.info(f"Added permission for EventBridge to invoke Lambda: {lambda_arn}")
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"Permission already exists for Lambda: {lambda_arn}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "schedule_expression": schedule_expression,
            "lambda_arn": lambda_arn
        }
    
    def create_approval_workflow_lambda(
        self,
        pipeline_name: str,
        lambda_name: Optional[str] = None,
        approval_sns_topic_arn: Optional[str] = None,
        approval_email: Optional[str] = None
    ) -> str:
        """
        Create a Lambda function for model update approval workflow.
        
        Args:
            pipeline_name: Name of the SageMaker pipeline
            lambda_name: Name for the Lambda function (optional)
            approval_sns_topic_arn: ARN of the SNS topic for approval notifications (optional)
            approval_email: Email address for approval notifications (optional)
            
        Returns:
            ARN of the created Lambda function
        """
        logger.info(f"Creating approval workflow Lambda for pipeline: {pipeline_name}")
        
        # Generate Lambda name if not provided
        if not lambda_name:
            lambda_name = f"{self.project_name}-model-approval-{pipeline_name}"
        
        # Create SNS topic if not provided
        if not approval_sns_topic_arn and approval_email:
            topic_name = f"{self.project_name}-model-approval"
            response = self.sns_client.create_topic(Name=topic_name)
            approval_sns_topic_arn = response["TopicArn"]
            
            # Subscribe email to topic
            self.sns_client.subscribe(
                TopicArn=approval_sns_topic_arn,
                Protocol="email",
                Endpoint=approval_email
            )
            
            logger.info(f"Created SNS topic: {topic_name} (ARN: {approval_sns_topic_arn})")
            logger.info(f"Subscribed email to SNS topic: {approval_email}")
        
        # Create Lambda function code
        lambda_code = f"""
import json
import boto3
import os
import logging
import urllib.parse
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
sns_client = boto3.client('sns')

# Configuration
PIPELINE_NAME = "{pipeline_name}"
SNS_TOPIC_ARN = "{approval_sns_topic_arn or ''}"

def lambda_handler(event, context):
    # Lambda handler for model update approval workflow.
    """
    """
    logger.info(f"Received event: {{json.dumps(event)}}")
    
    try:
        # Check if this is a pipeline execution status change event
        if event.get('source') == 'aws.sagemaker' and event.get('detail-type') == 'SageMaker Model Building Pipeline Execution Status Change':
            return handle_pipeline_status_change(event)
        
        # Check if this is a model package status change event
        elif event.get('source') == 'aws.sagemaker' and event.get('detail-type') == 'SageMaker Model Package State Change':
            return handle_model_package_status_change(event)
        
        # Check if this is an approval response
        elif 'queryStringParameters' in event:
            return handle_approval_response(event)
        
        else:
            logger.warning("Unknown event type")
            return {
                'statusCode': 400,
                'body': json.dumps('Unknown event type')
            }
    
    except Exception as e:
        logger.error(f"Error in lambda handler: {{str(e)}}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {{str(e)}}')
        }

def handle_pipeline_status_change(event):
    """
    Handle SageMaker pipeline execution status change events.
    """
    detail = event.get('detail', {})
    pipeline_name = detail.get('pipelineName', '')
    execution_status = detail.get('currentPipelineExecutionStatus', '')
    
    if pipeline_name != PIPELINE_NAME:
        logger.info(f"Event for different pipeline: {{pipeline_name}}")
        return {
            'statusCode': 200,
            'body': json.dumps('Event for different pipeline')
        }
    
    if execution_status == 'Succeeded':
        logger.info(f"Pipeline execution succeeded: {{pipeline_name}}")
        
        # Get the model package group name from the pipeline
        pipeline_desc = sagemaker_client.describe_pipeline(PipelineName=pipeline_name)
        pipeline_definition = json.loads(pipeline_desc['PipelineDefinition'])
        
        # Extract model package group name (simplified, would need to parse the pipeline definition)
        model_package_group_name = f"{pipeline_name}-models"
        
        # Get the latest model package
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        model_packages = response.get('ModelPackageSummaryList', [])
        if not model_packages:
            logger.warning(f"No model packages found for group: {{model_package_group_name}}")
            return {
                'statusCode': 404,
                'body': json.dumps('No model packages found')
            }
        
        model_package_arn = model_packages[0]['ModelPackageArn']
        
        # Send approval notification
        if SNS_TOPIC_ARN:
            # Create approval/rejection URLs (these would point to an API Gateway endpoint that triggers this Lambda)
            api_gateway_url = "https://example.execute-api.{self.region}.amazonaws.com/prod/model-approval"
            approve_url = f"{{api_gateway_url}}?action=approve&model_package_arn={{urllib.parse.quote(model_package_arn)}}"
            reject_url = f"{{api_gateway_url}}?action=reject&model_package_arn={{urllib.parse.quote(model_package_arn)}}"
            
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Model Approval Required for {{pipeline_name}}",
                Message=f"A new model has been created by pipeline {{pipeline_name}} and requires approval. "
                        f"\\n\\nModel Package ARN: {{model_package_arn}} "
                        f"\\n\\nTo approve this model, click: {{approve_url}} "
                        f"\\n\\nTo reject this model, click: {{reject_url}} "
                        f"\\n\\nThis approval link will expire in 7 days."
            )
            
            logger.info(f"Sent approval notification for model package: {{model_package_arn}}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Pipeline execution succeeded',
                'pipeline_name': pipeline_name,
                'model_package_arn': model_package_arn
            })
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Pipeline execution status: {{execution_status}}',
            'pipeline_name': pipeline_name
        })
    }

def handle_model_package_status_change(event):
    """
    Handle SageMaker model package status change events.
    """
    detail = event.get('detail', {})
    model_package_arn = detail.get('ModelPackageArn', '')
    model_approval_status = detail.get('ModelApprovalStatus', '')
    
    logger.info(f"Model package status changed: {{model_package_arn}} -> {{model_approval_status}}")
    
    if model_approval_status == 'Approved':
        # Get model package details
        response = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        
        # Send notification
        if SNS_TOPIC_ARN:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Model Approved: {{model_package_arn.split('/')[-1]}}",
                Message=f"The model package {{model_package_arn}} has been approved. "
                        f"\\n\\nThe model will now be deployed to the endpoint."
            )
            
            logger.info(f"Sent approval confirmation for model package: {{model_package_arn}}")
    
    elif model_approval_status == 'Rejected':
        # Send notification
        if SNS_TOPIC_ARN:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Model Rejected: {{model_package_arn.split('/')[-1]}}",
                Message=f"The model package {{model_package_arn}} has been rejected. "
                        f"\\n\\nNo further action will be taken."
            )
            
            logger.info(f"Sent rejection confirmation for model package: {{model_package_arn}}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Model package status: {{model_approval_status}}',
            'model_package_arn': model_package_arn
        })
    }

def handle_approval_response(event):
    """
    Handle approval response from the API Gateway.
    """
    query_params = event.get('queryStringParameters', {})
    action = query_params.get('action', '')
    model_package_arn = query_params.get('model_package_arn', '')
    
    if not action or not model_package_arn:
        return {
            'statusCode': 400,
            'body': json.dumps('Missing required parameters')
        }
    
    if action == 'approve':
        # Update model package approval status
        sagemaker_client.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus='Approved'
        )
        
        logger.info(f"Model package approved: {{model_package_arn}}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Model approved successfully')
        }
    
    elif action == 'reject':
        # Update model package approval status
        sagemaker_client.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus='Rejected'
        )
        
        logger.info(f"Model package rejected: {{model_package_arn}}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Model rejected successfully')
        }
    
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid action')
        }
"""
        
        # Create IAM role for Lambda
        iam_client = self.session.client('iam')
        role_name = f"{lambda_name}-role"
        
        try:
            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description=f"Role for {lambda_name} Lambda function"
            )
            
            role_arn = role_response['Role']['Arn']
            
            # Attach policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSNSFullAccess"
            )
            
            # Wait for role to propagate
            logger.info(f"Waiting for IAM role to propagate...")
            time.sleep(10)
            
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = iam_client.get_role(RoleName=role_name)
            role_arn = role_response['Role']['Arn']
            logger.info(f"Using existing IAM role: {role_name}")
        
        # Create Lambda function
        lambda_zip_file = "/tmp/lambda_function.zip"
        
        # Write Lambda code to file
        with open("/tmp/lambda_function.py", "w") as f:
            f.write(lambda_code)
        
        # Create ZIP file
        import zipfile
        with zipfile.ZipFile(lambda_zip_file, "w") as z:
            z.write("/tmp/lambda_function.py", "lambda_function.py")
        
        # Read ZIP file
        with open(lambda_zip_file, "rb") as f:
            zip_bytes = f.read()
        
        # Create Lambda function
        try:
            response = self.lambda_client.create_function(
                FunctionName=lambda_name,
                Runtime="python3.8",
                Role=role_arn,
                Handler="lambda_function.lambda_handler",
                Code={
                    "ZipFile": zip_bytes
                },
                Description=f"Lambda function for model update approval workflow for {pipeline_name}",
                Timeout=60,
                MemorySize=128,
                Publish=True
            )
            
            lambda_arn = response["FunctionArn"]
            logger.info(f"Created Lambda function: {lambda_name} (ARN: {lambda_arn})")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            # Function already exists, get its ARN
            response = self.lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn = response["Configuration"]["FunctionArn"]
            
            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=lambda_name,
                ZipFile=zip_bytes,
                Publish=True
            )
            
            logger.info(f"Updated existing Lambda function: {lambda_name} (ARN: {lambda_arn})")
        
        return lambda_arn    d
ef create_pipeline_event_rule(
        self,
        pipeline_name: str,
        lambda_arn: str,
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule for pipeline execution events.
        
        Args:
            pipeline_name: Name of the SageMaker pipeline
            lambda_arn: ARN of the Lambda function to trigger
            rule_name: Name for the rule (optional)
            
        Returns:
            Rule details
        """
        logger.info(f"Creating pipeline event rule for pipeline: {pipeline_name}")
        
        # Generate rule name if not provided
        if not rule_name:
            rule_name = f"{self.project_name}-pipeline-events-{pipeline_name}"
        
        # Create rule pattern for pipeline events
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
            "detail": {
                "pipelineName": [pipeline_name]
            }
        }
        
        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Rule for pipeline execution events: {pipeline_name}"
        )
        
        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")
        
        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    "Id": f"{rule_name}-target",
                    "Arn": lambda_arn
                }
            ]
        )
        
        # Add Lambda permission
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_arn,
                StatementId=f"{rule_name}-permission",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn
            )
            logger.info(f"Added permission for EventBridge to invoke Lambda: {lambda_arn}")
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"Permission already exists for Lambda: {lambda_arn}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "lambda_arn": lambda_arn
        }
    
    def create_model_package_event_rule(
        self,
        lambda_arn: str,
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule for model package status change events.
        
        Args:
            lambda_arn: ARN of the Lambda function to trigger
            rule_name: Name for the rule (optional)
            
        Returns:
            Rule details
        """
        logger.info(f"Creating model package event rule")
        
        # Generate rule name if not provided
        if not rule_name:
            rule_name = f"{self.project_name}-model-package-events"
        
        # Create rule pattern for model package events
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Package State Change"]
        }
        
        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Rule for model package status change events"
        )
        
        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")
        
        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    "Id": f"{rule_name}-target",
                    "Arn": lambda_arn
                }
            ]
        )
        
        # Add Lambda permission
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_arn,
                StatementId=f"{rule_name}-permission",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn
            )
            logger.info(f"Added permission for EventBridge to invoke Lambda: {lambda_arn}")
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"Permission already exists for Lambda: {lambda_arn}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "lambda_arn": lambda_arn
        }
    
    def setup_complete_retraining_solution(
        self,
        endpoint_name: str,
        pipeline_name: str,
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.7,
        evaluation_schedule: str = "rate(1 day)",
        email_notifications: Optional[List[str]] = None,
        bucket: Optional[str] = None,
        evaluation_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set up a complete automated retraining solution.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            pipeline_name: Name of the SageMaker pipeline to trigger
            drift_threshold: Threshold for drift detection
            performance_threshold: Threshold for model performance
            evaluation_schedule: Schedule expression for model evaluation
            email_notifications: List of email addresses for notifications
            bucket: S3 bucket for monitoring data
            evaluation_data: S3 path to evaluation data
            
        Returns:
            Solution configuration details
        """
        logger.info(f"Setting up complete retraining solution for endpoint: {endpoint_name}")
        
        # Set default bucket if not provided
        if not bucket:
            bucket = self.pipeline_builder.default_bucket
        
        # Set default evaluation data if not provided
        if not evaluation_data:
            evaluation_data = f"s3://{bucket}/evaluation-data/{endpoint_name}"
        
        # Create SNS topic for notifications
        sns_topic_arn = None
        if email_notifications:
            topic_name = f"{self.project_name}-retraining-{endpoint_name}"
            response = self.sns_client.create_topic(Name=topic_name)
            sns_topic_arn = response["TopicArn"]
            
            # Subscribe emails to topic
            for email in email_notifications:
                self.sns_client.subscribe(
                    TopicArn=sns_topic_arn,
                    Protocol="email",
                    Endpoint=email
                )
            
            logger.info(f"Created SNS topic: {topic_name} (ARN: {sns_topic_arn})")
            logger.info(f"Subscribed {len(email_notifications)} emails to SNS topic")
        
        # Create drift detection Lambda
        drift_lambda_arn = self.create_drift_detection_lambda(
            endpoint_name=endpoint_name,
            pipeline_name=pipeline_name,
            drift_threshold=drift_threshold,
            bucket=bucket
        )
        
        # Create EventBridge rule for model drift
        drift_rule = self.event_bridge.create_rule_for_model_drift(
            endpoint_name=endpoint_name,
            target_lambda_arn=drift_lambda_arn,
            target_sns_arn=sns_topic_arn
        )
        
        # Create scheduled evaluation Lambda
        eval_lambda_arn = self.create_scheduled_evaluation_lambda(
            endpoint_name=endpoint_name,
            pipeline_name=pipeline_name,
            performance_threshold=performance_threshold,
            evaluation_data=evaluation_data
        )
        
        # Create scheduled evaluation rule
        eval_rule = self.create_scheduled_evaluation_rule(
            lambda_arn=eval_lambda_arn,
            schedule_expression=evaluation_schedule
        )
        
        # Create approval workflow Lambda
        approval_lambda_arn = self.create_approval_workflow_lambda(
            pipeline_name=pipeline_name,
            approval_sns_topic_arn=sns_topic_arn,
            approval_email=email_notifications[0] if email_notifications else None
        )
        
        # Create pipeline event rule
        pipeline_rule = self.create_pipeline_event_rule(
            pipeline_name=pipeline_name,
            lambda_arn=approval_lambda_arn
        )
        
        # Create model package event rule
        model_package_rule = self.create_model_package_event_rule(
            lambda_arn=approval_lambda_arn
        )
        
        logger.info(f"Complete retraining solution set up for endpoint: {endpoint_name}")
        
        return {
            "endpoint_name": endpoint_name,
            "pipeline_name": pipeline_name,
            "drift_threshold": drift_threshold,
            "performance_threshold": performance_threshold,
            "evaluation_schedule": evaluation_schedule,
            "sns_topic_arn": sns_topic_arn,
            "drift_lambda_arn": drift_lambda_arn,
            "drift_rule": drift_rule,
            "eval_lambda_arn": eval_lambda_arn,
            "eval_rule": eval_rule,
            "approval_lambda_arn": approval_lambda_arn,
            "pipeline_rule": pipeline_rule,
            "model_package_rule": model_package_rule
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up automated model retraining triggers")
    parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    parser.add_argument("--pipeline-name", required=True, help="Name of the SageMaker pipeline to trigger")
    parser.add_argument("--drift-threshold", type=float, default=0.1, help="Threshold for drift detection")
    parser.add_argument("--performance-threshold", type=float, default=0.7, help="Threshold for model performance")
    parser.add_argument("--evaluation-schedule", default="rate(1 day)", help="Schedule expression for model evaluation")
    parser.add_argument("--email", action="append", help="Email address for notifications")
    parser.add_argument("--bucket", help="S3 bucket for monitoring data")
    parser.add_argument("--evaluation-data", help="S3 path to evaluation data")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    
    args = parser.parse_args()
    
    # Initialize retraining manager
    retraining_manager = AutomatedRetrainingManager(
        aws_profile=args.profile,
        region=args.region
    )
    
    # Set up complete retraining solution
    result = retraining_manager.setup_complete_retraining_solution(
        endpoint_name=args.endpoint_name,
        pipeline_name=args.pipeline_name,
        drift_threshold=args.drift_threshold,
        performance_threshold=args.performance_threshold,
        evaluation_schedule=args.evaluation_schedule,
        email_notifications=args.email,
        bucket=args.bucket,
        evaluation_data=args.evaluation_data
    )
    
    # Print result
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()