#!/usr/bin/env python3
"""
Unit tests for Model Monitor

Tests the SageMaker Model Monitor functionality.
"""

import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock, call
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.model_monitor import (
    ModelMonitoringManager,
    create_model_monitor_for_endpoint
)


class TestModelMonitoringManager(unittest.TestCase):
    """Test ModelMonitoringManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock AWS session and clients
        self.mock_session = MagicMock()
        self.mock_sagemaker_client = MagicMock()
        self.mock_cloudwatch_client = MagicMock()
        self.mock_s3_client = MagicMock()
        self.mock_sns_client = MagicMock()
        self.mock_events_client = MagicMock()
        
        # Configure mock session to return mock clients
        self.mock_session.client.side_effect = lambda service, region_name=None: {
            'sagemaker': self.mock_sagemaker_client,
            'cloudwatch': self.mock_cloudwatch_client,
            's3': self.mock_s3_client,
            'sns': self.mock_sns_client,
            'events': self.mock_events_client
        }.get(service, MagicMock())
        
        # Create patch for boto3.Session
        self.boto3_session_patch = patch('boto3.Session', return_value=self.mock_session)
        self.boto3_session_patch.start()
        
        # Create ModelMonitoringManager instance
        self.endpoint_name = "test-endpoint"
        self.bucket = "test-bucket"
        self.prefix = "monitoring"
        self.manager = ModelMonitoringManager(
            endpoint_name=self.endpoint_name,
            bucket=self.bucket,
            prefix=self.prefix,
            aws_profile="ab",
            region="us-east-1"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.boto3_session_patch.stop()
    
    def test_initialization(self):
        """Test initialization of ModelMonitoringManager"""
        # Verify boto3 session was created with correct profile and region
        from boto3 import Session
        Session.assert_called_once_with(profile_name="ab", region_name="us-east-1")
        
        # Verify clients were created
        self.mock_session.client.assert_any_call('sagemaker', region_name="us-east-1")
        self.mock_session.client.assert_any_call('cloudwatch', region_name="us-east-1")
        self.mock_session.client.assert_any_call('s3', region_name="us-east-1")
        self.mock_session.client.assert_any_call('sns', region_name="us-east-1")
        self.mock_session.client.assert_any_call('events', region_name="us-east-1")
        
        # Verify instance variables
        self.assertEqual(self.manager.endpoint_name, self.endpoint_name)
        self.assertEqual(self.manager.bucket, self.bucket)
        self.assertEqual(self.manager.prefix, self.prefix)
        self.assertEqual(self.manager.sagemaker_client, self.mock_sagemaker_client)
        self.assertEqual(self.manager.cloudwatch_client, self.mock_cloudwatch_client)
        self.assertEqual(self.manager.s3_client, self.mock_s3_client)
        self.assertEqual(self.manager.sns_client, self.mock_sns_client)
        self.assertEqual(self.manager.events_client, self.mock_events_client)
    
    def test_enable_data_capture(self):
        """Test enabling data capture for an endpoint"""
        # Configure mock responses
        self.mock_sagemaker_client.describe_endpoint.return_value = {
            "EndpointName": self.endpoint_name,
            "EndpointConfigName": "test-endpoint-config"
        }
        
        self.mock_sagemaker_client.describe_endpoint_config.return_value = {
            "EndpointConfigName": "test-endpoint-config",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "test-model",
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge"
                }
            ]
        }
        
        self.mock_sagemaker_client.create_endpoint_config.return_value = {
            "EndpointConfigArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint-config/test-endpoint-config-with-capture"
        }
        
        self.mock_sagemaker_client.update_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"
        }
        
        # Call the method
        capture_config = self.manager.enable_data_capture(
            sampling_percentage=100.0,
            capture_options=["Input", "Output"]
        )
        
        # Verify describe_endpoint was called
        self.mock_sagemaker_client.describe_endpoint.assert_called_once_with(
            EndpointName=self.endpoint_name
        )
        
        # Verify describe_endpoint_config was called
        self.mock_sagemaker_client.describe_endpoint_config.assert_called_once_with(
            EndpointConfigName="test-endpoint-config"
        )
        
        # Verify create_endpoint_config was called
        self.mock_sagemaker_client.create_endpoint_config.assert_called_once()
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_endpoint_config.call_args[1]
        self.assertTrue(call_args["EndpointConfigName"].startswith("test-endpoint-config-with-capture"))
        self.assertEqual(len(call_args["ProductionVariants"]), 1)
        
        # Verify data capture config
        data_capture_config = call_args["DataCaptureConfig"]
        self.assertTrue(data_capture_config["EnableCapture"])
        self.assertEqual(data_capture_config["InitialSamplingPercentage"], 100.0)
        self.assertEqual(data_capture_config["CaptureOptions"], [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}])
        self.assertEqual(data_capture_config["CaptureContentTypeHeader"]["CsvContentTypes"], ["text/csv"])
        self.assertEqual(data_capture_config["CaptureContentTypeHeader"]["JsonContentTypes"], ["application/json"])
        
        # Verify update_endpoint was called
        self.mock_sagemaker_client.update_endpoint.assert_called_once()
        update_args = self.mock_sagemaker_client.update_endpoint.call_args[1]
        self.assertEqual(update_args["EndpointName"], self.endpoint_name)
        self.assertTrue(update_args["EndpointConfigName"].startswith("test-endpoint-config-with-capture"))
        
        # Verify return value
        self.assertIn("destination_s3_uri", capture_config)
        self.assertEqual(capture_config["sampling_percentage"], 100.0)
        self.assertEqual(capture_config["capture_options"], ["Input", "Output"])
    
    def test_create_baseline(self):
        """Test creating a baseline for model monitoring"""
        # Configure mock response
        self.mock_sagemaker_client.create_processing_job.return_value = {
            "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:processing-job/test-baseline-job"
        }
        
        # Call the method
        baseline_dataset = "s3://test-bucket/baseline-data"
        job_name = self.manager.create_baseline(
            baseline_dataset=baseline_dataset,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600,
            problem_type="BinaryClassification",
            ground_truth_attribute="target",
            inference_attribute="prediction",
            probability_attribute="probability",
            probability_threshold_attribute=0.5
        )
        
        # Verify create_processing_job was called
        self.mock_sagemaker_client.create_processing_job.assert_called_once()
        
        # Verify job name format
        self.assertTrue(job_name.startswith(f"{self.endpoint_name}-baseline-"))
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_processing_job.call_args[1]
        self.assertEqual(call_args["ProcessingJobName"], job_name)
        
        # Verify processing resources
        self.assertEqual(call_args["ProcessingResources"]["ClusterConfig"]["InstanceType"], "ml.m5.xlarge")
        self.assertEqual(call_args["ProcessingResources"]["ClusterConfig"]["InstanceCount"], 1)
        
        # Verify environment variables
        env = call_args["Environment"]
        self.assertEqual(env["dataset_format"], "csv/default")
        self.assertEqual(env["dataset_source"], "/opt/ml/processing/input/baseline")
        self.assertEqual(env["output_path"], "/opt/ml/processing/output")
        self.assertEqual(env["publish_cloudwatch_metrics"], "Enabled")
        self.assertEqual(env["problem_type"], "BinaryClassification")
        self.assertEqual(env["ground_truth_attribute"], "target")
        self.assertEqual(env["inference_attribute"], "prediction")
        self.assertEqual(env["probability_attribute"], "probability")
        self.assertEqual(env["probability_threshold_attribute"], "0.5")
    
    def test_create_monitoring_schedule(self):
        """Test creating a monitoring schedule"""
        # Configure mock response
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-schedule"
        }
        
        # Call the method
        baseline_constraints = "s3://test-bucket/constraints.json"
        baseline_statistics = "s3://test-bucket/statistics.json"
        schedule_name = self.manager.create_monitoring_schedule(
            baseline_constraints=baseline_constraints,
            baseline_statistics=baseline_statistics,
            schedule_expression="cron(0 * ? * * *)",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600,
            monitoring_type="DataQuality",
            record_preprocessor_script=None,
            post_analytics_processor_script=None,
            enable_cloudwatch_metrics=True
        )
        
        # Verify create_monitoring_schedule was called
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called_once()
        
        # Verify schedule name format
        self.assertTrue(schedule_name.startswith(f"{self.endpoint_name}-"))
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_monitoring_schedule.call_args[1]
        self.assertEqual(call_args["MonitoringScheduleName"], schedule_name)
        
        # Verify schedule config
        schedule_config = call_args["MonitoringScheduleConfig"]
        self.assertEqual(schedule_config["ScheduleConfig"]["ScheduleExpression"], "cron(0 * ? * * *)")
        
        # Verify monitoring job definition
        job_def = schedule_config["MonitoringJobDefinition"]
        
        # Verify baseline config
        self.assertEqual(job_def["BaselineConfig"]["ConstraintsResource"]["S3Uri"], baseline_constraints)
        self.assertEqual(job_def["BaselineConfig"]["StatisticsResource"]["S3Uri"], baseline_statistics)
        
        # Verify monitoring inputs
        self.assertIn("MonitoringInputs", job_def)
        self.assertEqual(len(job_def["MonitoringInputs"]), 1)
        self.assertEqual(job_def["MonitoringInputs"][0]["EndpointInput"]["EndpointName"], self.endpoint_name)
        
        # Verify environment variables
        env = job_def["Environment"]
        self.assertEqual(env["publish_cloudwatch_metrics"], "Enabled")
    
    def test_create_model_quality_monitoring_schedule(self):
        """Test creating a model quality monitoring schedule"""
        # Configure mock response
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-schedule"
        }
        
        # Call the method
        baseline_constraints = "s3://test-bucket/constraints.json"
        baseline_statistics = "s3://test-bucket/statistics.json"
        ground_truth_input = "s3://test-bucket/ground-truth"
        schedule_name = self.manager.create_model_quality_monitoring_schedule(
            baseline_constraints=baseline_constraints,
            baseline_statistics=baseline_statistics,
            ground_truth_input=ground_truth_input,
            problem_type="BinaryClassification",
            inference_attribute="prediction",
            ground_truth_attribute="target",
            probability_attribute="probability",
            probability_threshold_attribute=0.5,
            schedule_expression="cron(0 * ? * * *)",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600
        )
        
        # Verify create_monitoring_schedule was called
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called_once()
        
        # Verify schedule name format
        self.assertTrue(schedule_name.startswith(f"{self.endpoint_name}-model-quality-"))
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_monitoring_schedule.call_args[1]
        self.assertEqual(call_args["MonitoringScheduleName"], schedule_name)
        
        # Verify monitoring job definition
        job_def = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        
        # Verify monitoring inputs
        self.assertIn("MonitoringInputs", job_def)
        self.assertEqual(len(job_def["MonitoringInputs"]), 2)
        self.assertEqual(job_def["MonitoringInputs"][0]["EndpointInput"]["EndpointName"], self.endpoint_name)
        self.assertEqual(job_def["MonitoringInputs"][1]["BatchTransformInput"]["S3Uri"], ground_truth_input)
        
        # Verify environment variables
        env = job_def["Environment"]
        self.assertEqual(env["problem_type"], "BinaryClassification")
        self.assertEqual(env["inference_attribute"], "prediction")
        self.assertEqual(env["ground_truth_attribute"], "target")
        self.assertEqual(env["probability_attribute"], "probability")
        self.assertEqual(env["probability_threshold_attribute"], "0.5")
    
    def test_create_sns_topic(self):
        """Test creating an SNS topic for alerts"""
        # Configure mock response
        self.mock_sns_client.create_topic.return_value = {
            "TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic"
        }
        
        # Call the method
        topic_name = f"{self.endpoint_name}-alerts"
        email_notifications = ["user1@example.com", "user2@example.com"]
        topic_arn = self.manager.create_sns_topic(
            topic_name=topic_name,
            email_notifications=email_notifications
        )
        
        # Verify create_topic was called
        self.mock_sns_client.create_topic.assert_called_once_with(
            Name=topic_name
        )
        
        # Verify subscribe was called for each email
        self.assertEqual(self.mock_sns_client.subscribe.call_count, 2)
        self.mock_sns_client.subscribe.assert_has_calls([
            call(
                TopicArn="arn:aws:sns:us-east-1:123456789012:test-topic",
                Protocol="email",
                Endpoint="user1@example.com"
            ),
            call(
                TopicArn="arn:aws:sns:us-east-1:123456789012:test-topic",
                Protocol="email",
                Endpoint="user2@example.com"
            )
        ])
        
        # Verify return value
        self.assertEqual(topic_arn, "arn:aws:sns:us-east-1:123456789012:test-topic")
    
    def test_create_cloudwatch_alarm(self):
        """Test creating a CloudWatch alarm"""
        # Call the method
        metric_name = "feature_baseline_drift_check_violations"
        alarm_name = f"{self.endpoint_name}-{metric_name}-alarm"
        sns_topic_arn = "arn:aws:sns:us-east-1:123456789012:test-topic"
        
        alarm_arn = self.manager.create_cloudwatch_alarm(
            metric_name=metric_name,
            alarm_name=alarm_name,
            sns_topic_arn=sns_topic_arn,
            threshold=0,
            comparison_operator="GreaterThanThreshold",
            evaluation_periods=1,
            period=3600,
            statistic="Maximum"
        )
        
        # Verify put_metric_alarm was called
        self.mock_cloudwatch_client.put_metric_alarm.assert_called_once()
        
        # Verify call arguments
        call_args = self.mock_cloudwatch_client.put_metric_alarm.call_args[1]
        self.assertEqual(call_args["AlarmName"], alarm_name)
        self.assertEqual(call_args["MetricName"], metric_name)
        self.assertEqual(call_args["Namespace"], "aws/sagemaker/Endpoints/data-metrics")
        self.assertEqual(call_args["Dimensions"][0]["Name"], "Endpoint")
        self.assertEqual(call_args["Dimensions"][0]["Value"], self.endpoint_name)
        self.assertEqual(call_args["Statistic"], "Maximum")
        self.assertEqual(call_args["Period"], 3600)
        self.assertEqual(call_args["EvaluationPeriods"], 1)
        self.assertEqual(call_args["Threshold"], 0)
        self.assertEqual(call_args["ComparisonOperator"], "GreaterThanThreshold")
        self.assertEqual(call_args["AlarmActions"], [sns_topic_arn])
    
    def test_create_eventbridge_rule(self):
        """Test creating an EventBridge rule"""
        # Configure mock response
        self.mock_events_client.put_rule.return_value = {
            "RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        }
        
        # Call the method
        rule_name = f"{self.endpoint_name}-drift-detection-rule"
        monitoring_schedule_name = f"{self.endpoint_name}-data-quality-monitoring"
        sns_topic_arn = "arn:aws:sns:us-east-1:123456789012:test-topic"
        
        rule_arn = self.manager.create_eventbridge_rule(
            rule_name=rule_name,
            monitoring_schedule_name=monitoring_schedule_name,
            sns_topic_arn=sns_topic_arn
        )
        
        # Verify put_rule was called
        self.mock_events_client.put_rule.assert_called_once()
        
        # Verify call arguments
        call_args = self.mock_events_client.put_rule.call_args[1]
        self.assertEqual(call_args["Name"], rule_name)
        self.assertEqual(call_args["State"], "ENABLED")
        
        # Verify event pattern
        event_pattern = json.loads(call_args["EventPattern"])
        self.assertEqual(event_pattern["source"], ["aws.sagemaker"])
        self.assertEqual(event_pattern["detail-type"], ["SageMaker Model Monitor Violation"])
        self.assertIn(self.endpoint_name, event_pattern["resources"][0])
        self.assertEqual(event_pattern["detail"]["MonitoringScheduleName"], [monitoring_schedule_name])
        self.assertEqual(event_pattern["detail"]["ViolationsFound"], [True])
        
        # Verify put_targets was called
        self.mock_events_client.put_targets.assert_called_once()
        
        # Verify target arguments
        target_args = self.mock_events_client.put_targets.call_args[1]
        self.assertEqual(target_args["Rule"], rule_name)
        self.assertEqual(len(target_args["Targets"]), 1)
        self.assertEqual(target_args["Targets"][0]["Id"], f"{rule_name}-sns-target")
        self.assertEqual(target_args["Targets"][0]["Arn"], sns_topic_arn)
        
        # Verify return value
        self.assertEqual(rule_arn, "arn:aws:events:us-east-1:123456789012:rule/test-rule")


class TestCreateModelMonitorForEndpoint(unittest.TestCase):
    """Test create_model_monitor_for_endpoint function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock for ModelMonitoringManager
        self.mock_manager = MagicMock()
        self.mock_manager.enable_data_capture.return_value = {
            "destination_s3_uri": "s3://test-bucket/data-capture",
            "sampling_percentage": 100.0,
            "capture_options": ["Input", "Output"]
        }
        self.mock_manager.create_baseline.return_value = "test-baseline-job"
        self.mock_manager.create_monitoring_schedule.return_value = "test-endpoint-data-quality-monitoring"
        self.mock_manager.create_sns_topic.return_value = "arn:aws:sns:us-east-1:123456789012:test-topic"
        self.mock_manager.create_cloudwatch_alarm.return_value = "arn:aws:cloudwatch:us-east-1:123456789012:alarm:test-alarm"
        self.mock_manager.create_eventbridge_rule.return_value = "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        
        self.manager_patch = patch(
            'src.pipeline.model_monitor.ModelMonitoringManager',
            return_value=self.mock_manager
        )
        self.mock_manager_class = self.manager_patch.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.manager_patch.stop()
    
    def test_create_model_monitor_for_endpoint(self):
        """Test creating a model monitor for an endpoint"""
        # Call the function
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline-data"
        email_notifications = ["user@example.com"]
        
        monitoring_config = create_model_monitor_for_endpoint(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            aws_profile="ab",
            region="us-east-1",
            email_notifications=email_notifications
        )
        
        # Verify ModelMonitoringManager was created
        self.mock_manager_class.assert_called_once_with(
            endpoint_name=endpoint_name,
            bucket="test-bucket",
            prefix="monitoring",
            aws_profile="ab",
            region="us-east-1"
        )
        
        # Verify methods were called
        self.mock_manager.enable_data_capture.assert_called_once()
        self.mock_manager.create_baseline.assert_called_once_with(
            baseline_dataset=baseline_dataset,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600
        )
        self.mock_manager.create_monitoring_schedule.assert_called_once()
        self.mock_manager.create_sns_topic.assert_called_once_with(
            topic_name=f"{endpoint_name}-monitor-alerts",
            email_notifications=email_notifications
        )
        self.mock_manager.create_cloudwatch_alarm.assert_called_once()
        self.mock_manager.create_eventbridge_rule.assert_called_once()
        
        # Verify return value
        self.assertEqual(monitoring_config["endpoint_name"], endpoint_name)
        self.assertEqual(monitoring_config["data_capture_config"]["destination_s3_uri"], "s3://test-bucket/data-capture")
        self.assertEqual(monitoring_config["data_capture_config"]["sampling_percentage"], 100.0)
        self.assertEqual(monitoring_config["baseline_job"], "test-baseline-job")
        self.assertEqual(monitoring_config["monitoring_schedule"], "test-endpoint-data-quality-monitoring")
        self.assertEqual(monitoring_config["alerts"]["sns_topic_arn"], "arn:aws:sns:us-east-1:123456789012:test-topic")
        self.assertEqual(monitoring_config["alerts"]["alarm_arn"], "arn:aws:cloudwatch:us-east-1:123456789012:alarm:test-alarm")
        self.assertEqual(monitoring_config["alerts"]["rule_arn"], "arn:aws:events:us-east-1:123456789012:rule/test-rule")


if __name__ == '__main__':
    unittest.main()