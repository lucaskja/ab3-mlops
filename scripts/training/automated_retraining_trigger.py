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
from src.lambda_functions.utils import load_lambda_code, replace_placeholders
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        self.sagemaker_client = self.session.client("sagemaker", region_name=region)
        self.events_client = self.session.client("events", region_name=region)
        self.lambda_client = self.session.client("lambda", region_name=region)
        self.sns_client = self.session.client("sns", region_name=region)
        self.s3_client = self.session.client("s3", region_name=region)

        # Get project configuration
        self.project_config = get_config()
        self.project_name = self.project_config["project"]["name"]

        # Initialize components
        self.event_bridge = EventBridgeIntegration(
            aws_profile=aws_profile, region=region
        )
        self.pipeline_builder = SageMakerPipelineBuilder(
            aws_profile=aws_profile, region=region
        )

        logger.info(f"Automated retraining manager initialized for region: {region}")

    def create_drift_detection_lambda(
        self,
        endpoint_name: str,
        pipeline_name: str,
        drift_threshold: float = 0.1,
        lambda_name: Optional[str] = None,
        bucket: Optional[str] = None,
        prefix: str = "monitoring",
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

        # Load Lambda function code from file
        try:
            lambda_code = load_lambda_code("drift_detection")

            # Replace placeholders with actual values
            replacements = {
                "ENDPOINT_NAME_PLACEHOLDER": endpoint_name,
                "PIPELINE_NAME_PLACEHOLDER": pipeline_name,
                "DRIFT_THRESHOLD_PLACEHOLDER": drift_threshold,
                "BUCKET_PLACEHOLDER": bucket,
                "PREFIX_PLACEHOLDER": prefix,
            }

            lambda_code = replace_placeholders(lambda_code, replacements)

        except Exception as e:
            logger.error(f"Error loading Lambda code: {str(e)}")
            raise

        # Create IAM role for Lambda
        iam_client = self.session.client("iam")
        role_name = f"{lambda_name}-role"

        try:
            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description=f"Role for {lambda_name} Lambda function",
            )

            role_arn = role_response["Role"]["Arn"]

            # Attach policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSNSFullAccess",
            )

            # Wait for role to propagate
            logger.info(f"Waiting for IAM role to propagate...")
            time.sleep(10)

        except iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = iam_client.get_role(RoleName=role_name)
            role_arn = role_response["Role"]["Arn"]
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
                Code={"ZipFile": zip_bytes},
                Description=f"Lambda function to detect drift and trigger retraining for {endpoint_name}",
                Timeout=60,
                MemorySize=128,
                Publish=True,
            )

            lambda_arn = response["FunctionArn"]
            logger.info(f"Created Lambda function: {lambda_name} (ARN: {lambda_arn})")

        except self.lambda_client.exceptions.ResourceConflictException:
            # Function already exists, get its ARN
            response = self.lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn = response["Configuration"]["FunctionArn"]

            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=lambda_name, ZipFile=zip_bytes, Publish=True
            )

            logger.info(
                f"Updated existing Lambda function: {lambda_name} (ARN: {lambda_arn})"
            )

        return lambda_arn

    def create_scheduled_evaluation_lambda(
        self,
        endpoint_name: str,
        pipeline_name: str,
        performance_threshold: float = 0.7,
        lambda_name: Optional[str] = None,
        evaluation_data: Optional[str] = None,
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
        logger.info(
            f"Creating scheduled evaluation Lambda for endpoint: {endpoint_name}"
        )

        # Generate Lambda name if not provided
        if not lambda_name:
            lambda_name = f"{self.project_name}-scheduled-evaluation-{endpoint_name}"

        # Set default evaluation data if not provided
        if not evaluation_data:
            evaluation_data = f"s3://{self.pipeline_builder.default_bucket}/evaluation-data/{endpoint_name}"

        # Load Lambda function code from file
        try:
            lambda_code = load_lambda_code("scheduled_evaluation")

            # Replace placeholders with actual values
            replacements = {
                "ENDPOINT_NAME_PLACEHOLDER": endpoint_name,
                "PIPELINE_NAME_PLACEHOLDER": pipeline_name,
                "PERFORMANCE_THRESHOLD_PLACEHOLDER": performance_threshold,
                "EVALUATION_DATA_PLACEHOLDER": evaluation_data,
            }

            lambda_code = replace_placeholders(lambda_code, replacements)

        except Exception as e:
            logger.error(f"Error loading Lambda code: {str(e)}")
            raise

        # Create IAM role for Lambda
        iam_client = self.session.client("iam")
        role_name = f"{lambda_name}-role"

        try:
            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description=f"Role for {lambda_name} Lambda function",
            )

            role_arn = role_response["Role"]["Arn"]

            # Attach policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSNSFullAccess",
            )

            # Wait for role to propagate
            logger.info(f"Waiting for IAM role to propagate...")
            time.sleep(10)

        except iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = iam_client.get_role(RoleName=role_name)
            role_arn = role_response["Role"]["Arn"]
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
                Code={"ZipFile": zip_bytes},
                Description=f"Lambda function for scheduled model evaluation of {endpoint_name}",
                Timeout=300,  # 5 minutes
                MemorySize=256,
                Publish=True,
            )

            lambda_arn = response["FunctionArn"]
            logger.info(f"Created Lambda function: {lambda_name} (ARN: {lambda_arn})")

        except self.lambda_client.exceptions.ResourceConflictException:
            # Function already exists, get its ARN
            response = self.lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn = response["Configuration"]["FunctionArn"]

            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=lambda_name, ZipFile=zip_bytes, Publish=True
            )

            logger.info(
                f"Updated existing Lambda function: {lambda_name} (ARN: {lambda_arn})"
            )

        return lambda_arn

    def create_scheduled_evaluation_rule(
        self,
        lambda_arn: str,
        schedule_expression: str = "rate(1 day)",
        rule_name: Optional[str] = None,
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
        logger.info(
            f"Creating scheduled evaluation rule with schedule: {schedule_expression}"
        )

        # Generate rule name if not provided
        if not rule_name:
            lambda_name = lambda_arn.split(":")[-1]
            rule_name = f"{lambda_name}-schedule"

        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule_expression,
            State="ENABLED",
            Description=f"Scheduled rule for model evaluation: {schedule_expression}",
        )

        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")

        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name, Targets=[{"Id": f"{rule_name}-target", "Arn": lambda_arn}]
        )

        # Add Lambda permission
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_arn,
                StatementId=f"{rule_name}-permission",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn,
            )
            logger.info(
                f"Added permission for EventBridge to invoke Lambda: {lambda_arn}"
            )
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"Permission already exists for Lambda: {lambda_arn}")

        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "schedule_expression": schedule_expression,
            "lambda_arn": lambda_arn,
        }

    def create_approval_workflow_lambda(
        self,
        pipeline_name: str,
        lambda_name: Optional[str] = None,
        approval_sns_topic_arn: Optional[str] = None,
        approval_email: Optional[str] = None,
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
                Endpoint=approval_email,
            )

            logger.info(
                f"Created SNS topic: {topic_name} (ARN: {approval_sns_topic_arn})"
            )
            logger.info(f"Subscribed email to SNS topic: {approval_email}")

        # Load Lambda function code from file
        try:
            lambda_code = load_lambda_code("model_approval")

            # Replace placeholders with actual values
            replacements = {
                "PIPELINE_NAME_PLACEHOLDER": pipeline_name,
                "SNS_TOPIC_ARN_PLACEHOLDER": approval_sns_topic_arn or "",
                "REGION_PLACEHOLDER": self.region,
            }

            lambda_code = replace_placeholders(lambda_code, replacements)

        except Exception as e:
            logger.error(f"Error loading Lambda code: {str(e)}")
            raise

        # Create IAM role for Lambda
        iam_client = self.session.client("iam")
        role_name = f"{lambda_name}-role"

        try:
            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description=f"Role for {lambda_name} Lambda function",
            )

            role_arn = role_response["Role"]["Arn"]

            # Attach policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSNSFullAccess",
            )

            # Wait for role to propagate
            logger.info(f"Waiting for IAM role to propagate...")
            time.sleep(10)

        except iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = iam_client.get_role(RoleName=role_name)
            role_arn = role_response["Role"]["Arn"]
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
                Code={"ZipFile": zip_bytes},
                Description=f"Lambda function for model approval workflow of {pipeline_name}",
                Timeout=60,
                MemorySize=128,
                Publish=True,
            )

            lambda_arn = response["FunctionArn"]
            logger.info(f"Created Lambda function: {lambda_name} (ARN: {lambda_arn})")

        except self.lambda_client.exceptions.ResourceConflictException:
            # Function already exists, get its ARN
            response = self.lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn = response["Configuration"]["FunctionArn"]

            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=lambda_name, ZipFile=zip_bytes, Publish=True
            )

            logger.info(
                f"Updated existing Lambda function: {lambda_name} (ARN: {lambda_arn})"
            )

        return lambda_arn

    def create_pipeline_event_rule(
        self, pipeline_name: str, lambda_arn: str, rule_name: Optional[str] = None
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
            "detail-type": [
                "SageMaker Model Building Pipeline Execution Status Change"
            ],
            "detail": {"pipelineName": [pipeline_name]},
        }

        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Rule for pipeline execution events: {pipeline_name}",
        )

        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")

        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name, Targets=[{"Id": f"{rule_name}-target", "Arn": lambda_arn}]
        )

        # Add Lambda permission
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_arn,
                StatementId=f"{rule_name}-permission",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn,
            )
            logger.info(
                f"Added permission for EventBridge to invoke Lambda: {lambda_arn}"
            )
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"Permission already exists for Lambda: {lambda_arn}")

        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "lambda_arn": lambda_arn,
        }

    def create_model_package_event_rule(
        self, lambda_arn: str, rule_name: Optional[str] = None
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
            "detail-type": ["SageMaker Model Package State Change"],
        }

        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Rule for model package status change events",
        )

        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")

        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name, Targets=[{"Id": f"{rule_name}-target", "Arn": lambda_arn}]
        )

        # Add Lambda permission
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_arn,
                StatementId=f"{rule_name}-permission",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn,
            )
            logger.info(
                f"Added permission for EventBridge to invoke Lambda: {lambda_arn}"
            )
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"Permission already exists for Lambda: {lambda_arn}")

        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "lambda_arn": lambda_arn,
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
        evaluation_data: Optional[str] = None,
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
        logger.info(
            f"Setting up complete retraining solution for endpoint: {endpoint_name}"
        )

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
                    TopicArn=sns_topic_arn, Protocol="email", Endpoint=email
                )

            logger.info(f"Created SNS topic: {topic_name} (ARN: {sns_topic_arn})")
            logger.info(f"Subscribed {len(email_notifications)} emails to SNS topic")

        # Create drift detection Lambda
        drift_lambda_arn = self.create_drift_detection_lambda(
            endpoint_name=endpoint_name,
            pipeline_name=pipeline_name,
            drift_threshold=drift_threshold,
            bucket=bucket,
        )

        # Create EventBridge rule for model drift
        drift_rule = self.event_bridge.create_rule_for_model_drift(
            endpoint_name=endpoint_name,
            target_lambda_arn=drift_lambda_arn,
            target_sns_arn=sns_topic_arn,
        )

        # Create scheduled evaluation Lambda
        eval_lambda_arn = self.create_scheduled_evaluation_lambda(
            endpoint_name=endpoint_name,
            pipeline_name=pipeline_name,
            performance_threshold=performance_threshold,
            evaluation_data=evaluation_data,
        )

        # Create scheduled evaluation rule
        eval_rule = self.create_scheduled_evaluation_rule(
            lambda_arn=eval_lambda_arn, schedule_expression=evaluation_schedule
        )

        # Create approval workflow Lambda
        approval_lambda_arn = self.create_approval_workflow_lambda(
            pipeline_name=pipeline_name,
            approval_sns_topic_arn=sns_topic_arn,
            approval_email=email_notifications[0] if email_notifications else None,
        )

        # Create pipeline event rule
        pipeline_rule = self.create_pipeline_event_rule(
            pipeline_name=pipeline_name, lambda_arn=approval_lambda_arn
        )

        # Create model package event rule
        model_package_rule = self.create_model_package_event_rule(
            lambda_arn=approval_lambda_arn
        )

        logger.info(
            f"Complete retraining solution set up for endpoint: {endpoint_name}"
        )

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
            "model_package_rule": model_package_rule,
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up automated model retraining triggers"
    )
    parser.add_argument(
        "--endpoint-name", required=True, help="Name of the SageMaker endpoint"
    )
    parser.add_argument(
        "--pipeline-name",
        required=True,
        help="Name of the SageMaker pipeline to trigger",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.1,
        help="Threshold for drift detection",
    )
    parser.add_argument(
        "--performance-threshold",
        type=float,
        default=0.7,
        help="Threshold for model performance",
    )
    parser.add_argument(
        "--evaluation-schedule",
        default="rate(1 day)",
        help="Schedule expression for model evaluation",
    )
    parser.add_argument(
        "--email", action="append", help="Email address for notifications"
    )
    parser.add_argument("--bucket", help="S3 bucket for monitoring data")
    parser.add_argument("--evaluation-data", help="S3 path to evaluation data")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")

    args = parser.parse_args()

    # Create retraining manager
    retraining_manager = AutomatedRetrainingManager(
        aws_profile=args.profile, region=args.region
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
        evaluation_data=args.evaluation_data,
    )

    # Print result
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
