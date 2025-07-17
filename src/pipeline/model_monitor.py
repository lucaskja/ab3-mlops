"""
SageMaker Model Monitor Implementation

This module provides functionality for setting up and managing SageMaker Model Monitor
for data quality monitoring, drift detection, and alerting.

Requirements addressed:
- 5.1: Automatic configuration of SageMaker Model Monitor when a model is deployed
- 5.2: Capture input data and predictions for monitoring
- 5.3: Trigger alerts through EventBridge when data drift is detected
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    ModelQualityMonitor,
    CronExpressionGenerator,
    ModelMonitor
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.s3 import S3Uploader, S3Downloader

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitoringManager:
    """
    Manages SageMaker Model Monitor configurations and operations.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the Model Monitor manager.
        
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
        self.cloudwatch_client = self.session.client('cloudwatch', region_name=region)
        self.events_client = self.session.client('events', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Get project configuration
        self.project_config = get_config()
        self.execution_role = self.project_config['iam']['roles']['sagemaker_execution']['arn']
        
        # Set default S3 bucket
        self.default_bucket = self.sagemaker_session.default_bucket()
        
        logger.info(f"Model Monitor manager initialized for region: {region}")
        logger.info(f"Using execution role: {self.execution_role}")
        logger.info(f"Using default S3 bucket: {self.default_bucket}")
    
    def create_data_capture_config(
        self,
        endpoint_name: str,
        sampling_percentage: float = 100.0,
        enable_capture_input: bool = True,
        enable_capture_output: bool = True,
        initial_sampling_percentage: Optional[float] = None
    ) -> DataCaptureConfig:
        """
        Create a data capture configuration for a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            sampling_percentage: Percentage of data to capture (0-100)
            enable_capture_input: Whether to capture input data
            enable_capture_output: Whether to capture output data
            initial_sampling_percentage: Initial sampling percentage (optional)
            
        Returns:
            Configured DataCaptureConfig object
        """
        logger.info(f"Creating data capture config for endpoint: {endpoint_name}")
        
        # Create S3 prefix for captured data
        capture_prefix = f"data-capture/{endpoint_name}"
        
        # Create data capture config
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=sampling_percentage,
            destination_s3_uri=f"s3://{self.default_bucket}/{capture_prefix}",
            capture_options=["Input", "Output"] if enable_capture_input and enable_capture_output
                           else ["Input"] if enable_capture_input
                           else ["Output"] if enable_capture_output
                           else [],
            csv_content_types=["text/csv"],
            json_content_types=["application/json"]
        )
        
        logger.info(f"Data capture config created for endpoint: {endpoint_name}")
        logger.info(f"Capture destination: s3://{self.default_bucket}/{capture_prefix}")
        logger.info(f"Sampling percentage: {sampling_percentage}%")
        
        return data_capture_config
    
    def create_baseline_constraints(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_in_seconds: int = 1800
    ) -> str:
        """
        Create baseline constraints for model monitoring.
        
        Args:
            dataset_path: S3 path to the baseline dataset
            output_path: S3 path for output constraints (optional)
            features: List of feature names (optional)
            instance_type: Instance type for processing
            instance_count: Number of instances
            max_runtime_in_seconds: Maximum runtime in seconds
            
        Returns:
            S3 path to the generated constraints file
        """
        logger.info(f"Creating baseline constraints from dataset: {dataset_path}")
        
        # Set default output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"s3://{self.default_bucket}/model-monitor/baselines/{timestamp}"
        
        # Create default model monitor
        model_monitor = DefaultModelMonitor(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        # Suggest baseline constraints and statistics
        model_monitor.suggest_baseline(
            baseline_dataset=dataset_path,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=output_path,
            wait=True
        )
        
        # Get paths to generated files
        constraints_path = f"{output_path}/constraints.json"
        statistics_path = f"{output_path}/statistics.json"
        
        logger.info(f"Baseline constraints created at: {constraints_path}")
        logger.info(f"Baseline statistics created at: {statistics_path}")
        
        return constraints_path
    
    def create_data_quality_monitor(
        self,
        endpoint_name: str,
        baseline_constraints_path: str,
        baseline_statistics_path: Optional[str] = None,
        schedule_expression: str = "cron(0 * ? * * *)",  # Hourly
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_in_seconds: int = 1800,
        enable_cloudwatch_metrics: bool = True
    ) -> DefaultModelMonitor:
        """
        Create a data quality monitor for a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            baseline_constraints_path: S3 path to baseline constraints
            baseline_statistics_path: S3 path to baseline statistics (optional)
            schedule_expression: Schedule expression for monitoring
            instance_type: Instance type for monitoring
            instance_count: Number of instances
            max_runtime_in_seconds: Maximum runtime in seconds
            enable_cloudwatch_metrics: Whether to enable CloudWatch metrics
            
        Returns:
            Configured DefaultModelMonitor object
        """
        logger.info(f"Creating data quality monitor for endpoint: {endpoint_name}")
        
        # Set default statistics path if not provided
        if not baseline_statistics_path:
            baseline_statistics_path = baseline_constraints_path.replace(
                "constraints.json", "statistics.json"
            )
        
        # Create output path for monitoring results
        monitoring_output_path = f"s3://{self.default_bucket}/model-monitor/results/{endpoint_name}"
        
        # Create model monitor
        monitor = DefaultModelMonitor(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        # Set baseline resources
        monitor.latest_baselining_job_constraints_uri = baseline_constraints_path
        monitor.latest_baselining_job_statistics_uri = baseline_statistics_path
        
        # Create monitoring schedule
        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{endpoint_name}-data-quality-monitor",
            endpoint_input=endpoint_name,
            record_preprocessor_script=None,  # Use default preprocessor
            post_analytics_processor_script=None,  # Use default postprocessor
            output_s3_uri=monitoring_output_path,
            statistics=baseline_statistics_path,
            constraints=baseline_constraints_path,
            schedule_cron_expression=schedule_expression,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics
        )
        
        logger.info(f"Data quality monitor created for endpoint: {endpoint_name}")
        logger.info(f"Monitoring schedule: {schedule_expression}")
        logger.info(f"Monitoring output path: {monitoring_output_path}")
        
        return monitor
    
    def create_model_quality_monitor(
        self,
        endpoint_name: str,
        baseline_dataset: str,
        problem_type: str,
        ground_truth_attribute: str,
        inference_attribute: str,
        probability_attribute: Optional[str] = None,
        schedule_expression: str = "cron(0 0 ? * * *)",  # Daily
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_in_seconds: int = 1800,
        enable_cloudwatch_metrics: bool = True
    ) -> ModelQualityMonitor:
        """
        Create a model quality monitor for a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            baseline_dataset: S3 path to baseline dataset
            problem_type: Problem type (regression, binary_classification, multiclass_classification)
            ground_truth_attribute: Name of ground truth attribute
            inference_attribute: Name of inference attribute
            probability_attribute: Name of probability attribute (optional)
            schedule_expression: Schedule expression for monitoring
            instance_type: Instance type for monitoring
            instance_count: Number of instances
            max_runtime_in_seconds: Maximum runtime in seconds
            enable_cloudwatch_metrics: Whether to enable CloudWatch metrics
            
        Returns:
            Configured ModelQualityMonitor object
        """
        logger.info(f"Creating model quality monitor for endpoint: {endpoint_name}")
        
        # Create output path for monitoring results
        monitoring_output_path = f"s3://{self.default_bucket}/model-monitor/quality-results/{endpoint_name}"
        
        # Create model quality monitor
        monitor = ModelQualityMonitor(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        # Create baseline for model quality
        baseline_job_name = f"{endpoint_name}-quality-baseline-{int(time.time())}"
        monitor.suggest_baseline(
            baseline_dataset=baseline_dataset,
            dataset_format=DatasetFormat.csv(header=True),
            problem_type=problem_type,
            inference_attribute=inference_attribute,
            probability_attribute=probability_attribute,
            ground_truth_attribute=ground_truth_attribute,
            output_s3_uri=f"s3://{self.default_bucket}/model-monitor/quality-baselines/{endpoint_name}",
            wait=True,
            job_name=baseline_job_name
        )
        
        # Create monitoring schedule
        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{endpoint_name}-model-quality-monitor",
            endpoint_input=endpoint_name,
            record_preprocessor_script=None,  # Use default preprocessor
            post_analytics_processor_script=None,  # Use default postprocessor
            output_s3_uri=monitoring_output_path,
            problem_type=problem_type,
            ground_truth_input=None,  # Will be provided later through ground truth upload
            inference_attribute=inference_attribute,
            probability_attribute=probability_attribute,
            ground_truth_attribute=ground_truth_attribute,
            schedule_cron_expression=schedule_expression,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics
        )
        
        logger.info(f"Model quality monitor created for endpoint: {endpoint_name}")
        logger.info(f"Monitoring schedule: {schedule_expression}")
        logger.info(f"Monitoring output path: {monitoring_output_path}")
        
        return monitor
    
    def upload_ground_truth_to_monitor(
        self,
        monitor_schedule_name: str,
        ground_truth_s3_uri: str
    ) -> None:
        """
        Upload ground truth data to a model quality monitoring schedule.
        
        Args:
            monitor_schedule_name: Name of the monitoring schedule
            ground_truth_s3_uri: S3 URI to ground truth data
        """
        logger.info(f"Uploading ground truth data to monitor: {monitor_schedule_name}")
        
        # Create model quality monitor
        monitor = ModelQualityMonitor(
            role=self.execution_role,
            sagemaker_session=self.sagemaker_session
        )
        
        # Set monitoring schedule name
        monitor.monitoring_schedule_name = monitor_schedule_name
        
        # Upload ground truth
        monitor.update_monitoring_schedule(ground_truth_input=ground_truth_s3_uri)
        
        logger.info(f"Ground truth data uploaded to monitor: {monitor_schedule_name}")
        logger.info(f"Ground truth data path: {ground_truth_s3_uri}")
    
    def list_monitoring_schedules(
        self,
        endpoint_name: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List monitoring schedules.
        
        Args:
            endpoint_name: Filter by endpoint name (optional)
            max_results: Maximum number of results
            
        Returns:
            List of monitoring schedules
        """
        logger.info("Listing monitoring schedules")
        
        # Create filter if endpoint name is provided
        name_contains_filter = {
            'Name': 'MonitoringScheduleName',
            'Operator': 'Contains',
            'Value': endpoint_name
        } if endpoint_name else None
        
        # List monitoring schedules
        response = self.sagemaker_client.list_monitoring_schedules(
            MaxResults=max_results,
            NameContains=endpoint_name
        )
        
        schedules = response.get('MonitoringScheduleSummaries', [])
        logger.info(f"Found {len(schedules)} monitoring schedules")
        
        return schedules
    
    def get_monitoring_schedule(self, schedule_name: str) -> Dict[str, Any]:
        """
        Get details of a monitoring schedule.
        
        Args:
            schedule_name: Name of the monitoring schedule
            
        Returns:
            Monitoring schedule details
        """
        logger.info(f"Getting monitoring schedule: {schedule_name}")
        
        # Get monitoring schedule
        response = self.sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=schedule_name
        )
        
        logger.info(f"Retrieved monitoring schedule: {schedule_name}")
        return response
    
    def get_latest_monitoring_result(
        self,
        schedule_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest monitoring result for a schedule.
        
        Args:
            schedule_name: Name of the monitoring schedule
            
        Returns:
            Latest monitoring result or None if no executions
        """
        logger.info(f"Getting latest monitoring result for schedule: {schedule_name}")
        
        # List monitoring executions
        response = self.sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=schedule_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        executions = response.get('MonitoringExecutionSummaries', [])
        
        if not executions:
            logger.info(f"No monitoring executions found for schedule: {schedule_name}")
            return None
        
        # Get latest execution
        latest_execution = executions[0]
        execution_arn = latest_execution.get('MonitoringExecutionArn')
        
        # Get execution details
        execution_details = self.sagemaker_client.describe_monitoring_execution(
            MonitoringExecutionArn=execution_arn
        )
        
        logger.info(f"Latest monitoring execution: {execution_arn}")
        logger.info(f"Status: {execution_details.get('MonitoringExecutionStatus')}")
        
        return execution_details
    
    def analyze_monitoring_results(
        self,
        execution_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze monitoring results from an execution.
        
        Args:
            execution_details: Monitoring execution details
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing monitoring results")
        
        # Check execution status
        status = execution_details.get('MonitoringExecutionStatus')
        if status != 'Completed':
            logger.warning(f"Monitoring execution not completed. Status: {status}")
            return {
                'status': status,
                'violations_detected': False,
                'analysis_complete': False
            }
        
        # Get output path
        output_path = execution_details.get('ProcessingOutputConfig', {}).get('Outputs', [])
        if not output_path:
            logger.warning("No output path found in execution details")
            return {
                'status': status,
                'violations_detected': False,
                'analysis_complete': False
            }
        
        # Find the evaluation output
        evaluation_output = None
        for output in output_path:
            if output.get('OutputName') == 'evaluation':
                evaluation_output = output.get('S3Output', {}).get('S3Uri')
                break
        
        if not evaluation_output:
            logger.warning("No evaluation output found in execution details")
            return {
                'status': status,
                'violations_detected': False,
                'analysis_complete': False
            }
        
        # Download and parse constraints violations file
        violations_path = f"{evaluation_output}/constraint_violations.json"
        try:
            violations_local_path = "/tmp/constraint_violations.json"
            S3Downloader.download(
                s3_uri=violations_path,
                local_path=violations_local_path,
                sagemaker_session=self.sagemaker_session
            )
            
            # Parse violations file
            with open(violations_local_path, 'r') as f:
                violations_data = json.load(f)
            
            # Check for violations
            violations = violations_data.get('violations', [])
            has_violations = len(violations) > 0
            
            logger.info(f"Violations detected: {has_violations}")
            if has_violations:
                logger.info(f"Number of violations: {len(violations)}")
                for violation in violations:
                    logger.info(f"Violation: {violation.get('feature_name')} - {violation.get('constraint_check_type')}")
            
            # Return analysis results
            return {
                'status': status,
                'violations_detected': has_violations,
                'violations_count': len(violations),
                'violations': violations,
                'analysis_complete': True,
                'evaluation_path': evaluation_output
            }
            
        except Exception as e:
            logger.error(f"Error analyzing monitoring results: {str(e)}")
            return {
                'status': status,
                'violations_detected': False,
                'analysis_complete': False,
                'error': str(e)
            }
    
    def create_cloudwatch_alarm_for_violations(
        self,
        endpoint_name: str,
        threshold: int = 0,
        evaluation_period: int = 1,
        period: int = 3600,
        alarm_name: Optional[str] = None,
        sns_topic_arn: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a CloudWatch alarm for constraint violations.
        
        Args:
            endpoint_name: Name of the endpoint
            threshold: Threshold for violations (default: 0, any violation triggers alarm)
            evaluation_period: Number of periods to evaluate
            period: Period in seconds
            alarm_name: Name of the alarm (optional)
            sns_topic_arn: SNS topic ARN for notifications (optional)
            
        Returns:
            Alarm details
        """
        logger.info(f"Creating CloudWatch alarm for endpoint: {endpoint_name}")
        
        # Set default alarm name if not provided
        if not alarm_name:
            alarm_name = f"{endpoint_name}-constraint-violations-alarm"
        
        # Create alarm
        alarm_actions = [sns_topic_arn] if sns_topic_arn else []
        
        response = self.cloudwatch_client.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=evaluation_period,
            MetricName='feature_baseline_drift_check_violations',
            Namespace='aws/sagemaker/Endpoints/data-metrics',
            Period=period,
            Statistic='Maximum',
            Threshold=threshold,
            ActionsEnabled=True,
            AlarmActions=alarm_actions,
            AlarmDescription=f'Alarm for data drift violations on endpoint {endpoint_name}',
            Dimensions=[
                {
                    'Name': 'Endpoint',
                    'Value': endpoint_name
                },
                {
                    'Name': 'MonitoringSchedule',
                    'Value': f"{endpoint_name}-data-quality-monitor"
                }
            ]
        )
        
        logger.info(f"CloudWatch alarm created: {alarm_name}")
        if sns_topic_arn:
            logger.info(f"Alarm notifications will be sent to: {sns_topic_arn}")
        
        return {
            'alarm_name': alarm_name,
            'endpoint_name': endpoint_name,
            'threshold': threshold,
            'evaluation_period': evaluation_period,
            'period': period,
            'sns_topic_arn': sns_topic_arn
        }
    
    def create_sns_topic_for_alerts(
        self,
        topic_name: Optional[str] = None,
        email_subscriptions: Optional[List[str]] = None
    ) -> str:
        """
        Create an SNS topic for monitoring alerts.
        
        Args:
            topic_name: Name of the topic (optional)
            email_subscriptions: List of email addresses to subscribe (optional)
            
        Returns:
            SNS topic ARN
        """
        logger.info("Creating SNS topic for monitoring alerts")
        
        # Set default topic name if not provided
        if not topic_name:
            topic_name = f"sagemaker-model-monitor-alerts-{int(time.time())}"
        
        # Create SNS topic
        sns_client = self.session.client('sns', region_name=self.region)
        response = sns_client.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']
        
        # Add email subscriptions if provided
        if email_subscriptions:
            for email in email_subscriptions:
                sns_client.subscribe(
                    TopicArn=topic_arn,
                    Protocol='email',
                    Endpoint=email
                )
                logger.info(f"Subscribed email to SNS topic: {email}")
        
        logger.info(f"SNS topic created: {topic_arn}")
        return topic_arn
    
    def setup_complete_monitoring_solution(
        self,
        endpoint_name: str,
        baseline_dataset: str,
        schedule_expression: str = "cron(0 * ? * * *)",  # Hourly
        instance_type: str = "ml.m5.xlarge",
        email_notifications: Optional[List[str]] = None,
        enable_model_quality: bool = False,
        problem_type: Optional[str] = None,
        ground_truth_attribute: Optional[str] = None,
        inference_attribute: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set up a complete monitoring solution for a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            baseline_dataset: S3 path to baseline dataset
            schedule_expression: Schedule expression for monitoring
            instance_type: Instance type for monitoring
            email_notifications: List of email addresses for notifications
            enable_model_quality: Whether to enable model quality monitoring
            problem_type: Problem type for model quality monitoring
            ground_truth_attribute: Name of ground truth attribute
            inference_attribute: Name of inference attribute
            
        Returns:
            Monitoring configuration details
        """
        logger.info(f"Setting up complete monitoring solution for endpoint: {endpoint_name}")
        
        # Create data capture config
        data_capture_config = self.create_data_capture_config(
            endpoint_name=endpoint_name,
            sampling_percentage=100.0,
            enable_capture_input=True,
            enable_capture_output=True
        )
        
        # Create baseline constraints
        baseline_constraints_path = self.create_baseline_constraints(
            dataset_path=baseline_dataset,
            instance_type=instance_type
        )
        
        # Create SNS topic for alerts
        sns_topic_arn = self.create_sns_topic_for_alerts(
            topic_name=f"{endpoint_name}-monitor-alerts",
            email_subscriptions=email_notifications
        )
        
        # Create data quality monitor
        data_quality_monitor = self.create_data_quality_monitor(
            endpoint_name=endpoint_name,
            baseline_constraints_path=baseline_constraints_path,
            schedule_expression=schedule_expression,
            instance_type=instance_type,
            enable_cloudwatch_metrics=True
        )
        
        # Create CloudWatch alarm for violations
        alarm_details = self.create_cloudwatch_alarm_for_violations(
            endpoint_name=endpoint_name,
            threshold=0,  # Any violation triggers alarm
            sns_topic_arn=sns_topic_arn
        )
        
        # Create model quality monitor if enabled
        model_quality_monitor = None
        if enable_model_quality:
            if not all([problem_type, ground_truth_attribute, inference_attribute]):
                logger.warning("Model quality monitoring requires problem_type, ground_truth_attribute, and inference_attribute")
            else:
                model_quality_monitor = self.create_model_quality_monitor(
                    endpoint_name=endpoint_name,
                    baseline_dataset=baseline_dataset,
                    problem_type=problem_type,
                    ground_truth_attribute=ground_truth_attribute,
                    inference_attribute=inference_attribute,
                    schedule_expression="cron(0 0 ? * * *)",  # Daily
                    instance_type=instance_type,
                    enable_cloudwatch_metrics=True
                )
        
        # Return configuration details
        return {
            'endpoint_name': endpoint_name,
            'data_capture_config': {
                'destination_s3_uri': data_capture_config.destination_s3_uri,
                'sampling_percentage': data_capture_config.sampling_percentage
            },
            'baseline_constraints_path': baseline_constraints_path,
            'data_quality_monitor': {
                'schedule_name': f"{endpoint_name}-data-quality-monitor",
                'schedule_expression': schedule_expression
            },
            'model_quality_monitor': {
                'enabled': enable_model_quality,
                'schedule_name': f"{endpoint_name}-model-quality-monitor" if enable_model_quality else None
            },
            'alerts': {
                'sns_topic_arn': sns_topic_arn,
                'alarm_name': alarm_details['alarm_name']
            }
        }


def create_model_monitor_for_endpoint(
    endpoint_name: str,
    baseline_dataset: str,
    aws_profile: str = "ab",
    region: str = "us-east-1",
    email_notifications: Optional[List[str]] = None,
    enable_model_quality: bool = False,
    problem_type: Optional[str] = None,
    ground_truth_attribute: Optional[str] = None,
    inference_attribute: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create model monitoring for a SageMaker endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        baseline_dataset: S3 path to baseline dataset
        aws_profile: AWS profile to use
        region: AWS region
        email_notifications: List of email addresses for notifications
        enable_model_quality: Whether to enable model quality monitoring
        problem_type: Problem type for model quality monitoring
        ground_truth_attribute: Name of ground truth attribute
        inference_attribute: Name of inference attribute
        
    Returns:
        Monitoring configuration details
    """
    logger.info(f"Creating model monitoring for endpoint: {endpoint_name}")
    
    # Create model monitoring manager
    monitoring_manager = ModelMonitoringManager(
        aws_profile=aws_profile,
        region=region
    )
    
    # Set up complete monitoring solution
    monitoring_config = monitoring_manager.setup_complete_monitoring_solution(
        endpoint_name=endpoint_name,
        baseline_dataset=baseline_dataset,
        email_notifications=email_notifications,
        enable_model_quality=enable_model_quality,
        problem_type=problem_type,
        ground_truth_attribute=ground_truth_attribute,
        inference_attribute=inference_attribute
    )
    
    logger.info(f"Model monitoring created for endpoint: {endpoint_name}")
    return monitoring_config