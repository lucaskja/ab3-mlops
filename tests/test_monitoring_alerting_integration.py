#!/usr/bin/env python3
"""
Integration tests for Monitoring and Alerting

Tests the integration between SageMaker Model Monitor, CloudWatch, EventBridge, and SNS.
"""

import unittest
import os
import sys
import json
import boto3
from unittest.mock import patch, MagicMock, ANY
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.model_monitor_integration import add_monitoring_to_pipeline_endpoint
from src.pipeline.event_bridge_integration import create_model_drift_alert
from src.monitoring.drift_detection import DriftDetector


class TestMonitoringAlertingIntegration(unittest.TestCase):
    """Integration tests for monitoring and alerting"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock AWS session and clients
        self.session_patch = patch('boto3.Session')
        self.mock_session = self.session_patch.start()
        
        # Mock SageMaker client
        self.mock_sagemaker_client = MagicMock()
        
        # Mock CloudWatch client
        self.mock_cloudwatch_client = MagicMock()
        
        # Mock EventBridge client
        self.mock_events_client = MagicMock()
        
        # Mock SNS client
        self.mock_sns_client = MagicMock()
        
        # Mock S3 client
        self.mock_s3_client = MagicMock()
        
        # Configure mock session to return mock clients
        self.mock_session.return_value.client.side_effect = lambda service, region_name=None: {
            'sagemaker': self.mock_sagemaker_client,
            'cloudwatch': self.mock_cloudwatch_client,
            'events': self.mock_events_client,
            'sns': self.mock_sns_client,
            's3': self.mock_s3_client
        }.get(service, MagicMock())
        
        # Mock responses
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-schedule"
        }
        self.mock_cloudwatch_client.put_metric_alarm.return_value = {}
        self.mock_events_client.put_rule.return_value = {
            "RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        }
        self.mock_sns_client.create_topic.return_value = {
            "TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic"
        }
        
        # Test parameters
        self.endpoint_name = "test-endpoint"
        self.baseline_dataset = "s3://test-bucket/baseline-data"
        self.email_notifications = ["test@example.com"]
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.session_patch.stop()
    
    def test_add_monitoring_to_endpoint(self):
        """Test adding monitoring to an endpoint"""
        # Call the function
        monitoring_config = add_monitoring_to_pipeline_endpoint(
            pipeline_name="test-pipeline",
            endpoint_name=self.endpoint_name,
            baseline_dataset=self.baseline_dataset,
            aws_profile="ab",
            region="us-east-1",
            email_notifications=self.email_notifications
        )
        
        # Verify SageMaker client calls
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called()
        
        # Verify CloudWatch client calls
        self.mock_cloudwatch_client.put_metric_alarm.assert_called()
        
        # Verify SNS client calls
        self.mock_sns_client.create_topic.assert_called()
        self.mock_sns_client.subscribe.assert_called_with(
            TopicArn=ANY,
            Protocol="email",
            Endpoint="test@example.com"
        )
        
        # Verify monitoring config
        self.assertEqual(monitoring_config["endpoint_name"], self.endpoint_name)
        self.assertIn("data_capture_config", monitoring_config)
        self.assertIn("monitoring_schedule", monitoring_config)
        self.assertIn("alerts", monitoring_config)
    
    def test_create_model_drift_alert(self):
        """Test creating a model drift alert"""
        # Call the function
        alert_config = create_model_drift_alert(
            endpoint_name=self.endpoint_name,
            email_notifications=self.email_notifications,
            aws_profile="ab"
        )
        
        # Verify EventBridge client calls
        self.mock_events_client.put_rule.assert_called()
        self.mock_events_client.put_targets.assert_called()
        
        # Verify SNS client calls
        self.mock_sns_client.create_topic.assert_called()
        self.mock_sns_client.subscribe.assert_called_with(
            TopicArn=ANY,
            Protocol="email",
            Endpoint="test@example.com"
        )
        
        # Verify alert config
        self.assertEqual(alert_config["endpoint_name"], self.endpoint_name)
        self.assertIn("sns_topic_arn", alert_config)
        self.assertIn("rule_details", alert_config)
    
    def test_drift_detection_to_alert_integration(self):
        """Test integration from drift detection to alert"""
        # Create DriftDetector instance
        detector = DriftDetector(
            endpoint_name=self.endpoint_name,
            bucket="test-bucket",
            prefix="monitoring",
            profile_name="ab",
            region_name="us-east-1"
        )
        
        # Mock get_monitoring_results
        with patch.object(detector, 'get_monitoring_results') as mock_get_results:
            # Configure mock response
            mock_get_results.return_value = [
                {
                    'execution_date': datetime.now() - timedelta(days=1),
                    'violations': [
                        {"feature_name": "feature1", "constraint_check_type": "drift_check"},
                        {"feature_name": "feature2", "constraint_check_type": "drift_check"}
                    ],
                    's3_uri': 's3://test-bucket/result1'
                }
            ]
            
            # Analyze drift
            drift_metrics = detector.analyze_drift(
                monitoring_type="data-quality",
                days=7
            )
            
            # Verify drift metrics
            self.assertEqual(drift_metrics['total_executions'], 1)
            self.assertEqual(drift_metrics['executions_with_violations'], 1)
            self.assertEqual(drift_metrics['violation_counts']['feature1'], 1)
            self.assertEqual(drift_metrics['violation_counts']['feature2'], 1)
            
            # Mock trigger_retraining
            with patch.object(detector, 'trigger_retraining') as mock_trigger:
                mock_trigger.return_value = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
                
                # Trigger retraining if drift detected
                if drift_metrics['executions_with_violations'] > 0:
                    execution_arn = detector.trigger_retraining(
                        pipeline_name="test-pipeline",
                        pipeline_parameters={
                            "RetrainingReason": "DriftDetected",
                            "DriftMetrics": json.dumps(drift_metrics['violation_counts'])
                        }
                    )
                    
                    # Verify trigger_retraining was called
                    mock_trigger.assert_called_once()
                    
                    # Verify execution ARN
                    self.assertEqual(
                        execution_arn,
                        "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
                    )
    
    def test_cloudwatch_to_eventbridge_integration(self):
        """Test integration from CloudWatch to EventBridge"""
        # Mock CloudWatch alarm
        alarm_name = f"{self.endpoint_name}-feature_baseline_drift_check_violations-alarm"
        alarm_arn = f"arn:aws:cloudwatch:us-east-1:123456789012:alarm:{alarm_name}"
        
        # Mock EventBridge rule
        rule_name = f"mlops-sagemaker-demo-model-drift-{self.endpoint_name}"
        rule_arn = f"arn:aws:events:us-east-1:123456789012:rule/{rule_name}"
        
        # Mock SNS topic
        topic_name = f"model-drift-{self.endpoint_name}"
        topic_arn = f"arn:aws:sns:us-east-1:123456789012:{topic_name}"
        
        # Create CloudWatch alarm
        self.mock_cloudwatch_client.put_metric_alarm.return_value = {}
        
        # Create EventBridge rule
        self.mock_events_client.put_rule.return_value = {"RuleArn": rule_arn}
        
        # Create SNS topic
        self.mock_sns_client.create_topic.return_value = {"TopicArn": topic_arn}
        
        # Create DriftDetector instance
        detector = DriftDetector(
            endpoint_name=self.endpoint_name,
            bucket="test-bucket",
            prefix="monitoring",
            profile_name="ab",
            region_name="us-east-1"
        )
        
        # Create drift alert
        alarm_name = detector.create_drift_alert(
            metric_name="feature_baseline_drift_check_violations",
            threshold=0,
            comparison_operator="GreaterThanThreshold",
            evaluation_periods=1,
            period=3600,
            statistic="Maximum",
            sns_topic_arn=topic_arn
        )
        
        # Verify CloudWatch client calls
        self.mock_cloudwatch_client.put_metric_alarm.assert_called_once()
        
        # Verify alarm name
        expected_alarm_name = f"{self.endpoint_name}-feature_baseline_drift_check_violations-drift-alarm"
        self.assertEqual(alarm_name, expected_alarm_name)
        
        # Verify alarm configuration
        call_args = self.mock_cloudwatch_client.put_metric_alarm.call_args[1]
        self.assertEqual(call_args["AlarmName"], expected_alarm_name)
        self.assertEqual(call_args["MetricName"], "feature_baseline_drift_check_violations")
        self.assertEqual(call_args["AlarmActions"], [topic_arn])
    
    def test_eventbridge_to_sns_integration(self):
        """Test integration from EventBridge to SNS"""
        # Mock EventBridge rule
        rule_name = f"mlops-sagemaker-demo-model-drift-{self.endpoint_name}"
        rule_arn = f"arn:aws:events:us-east-1:123456789012:rule/{rule_name}"
        
        # Mock SNS topic
        topic_name = f"model-drift-{self.endpoint_name}"
        topic_arn = f"arn:aws:sns:us-east-1:123456789012:{topic_name}"
        
        # Create EventBridge rule
        self.mock_events_client.put_rule.return_value = {"RuleArn": rule_arn}
        
        # Create SNS topic
        self.mock_sns_client.create_topic.return_value = {"TopicArn": topic_arn}
        
        # Create EventBridge rule with SNS target
        self.mock_events_client.put_targets.return_value = {"FailedEntryCount": 0}
        
        # Create model drift alert
        alert_config = create_model_drift_alert(
            endpoint_name=self.endpoint_name,
            email_notifications=self.email_notifications,
            aws_profile="ab"
        )
        
        # Verify EventBridge client calls
        self.mock_events_client.put_rule.assert_called_once()
        self.mock_events_client.put_targets.assert_called_once()
        
        # Verify SNS client calls
        self.mock_sns_client.create_topic.assert_called_once()
        self.mock_sns_client.subscribe.assert_called_once()
        
        # Verify alert config
        self.assertEqual(alert_config["endpoint_name"], self.endpoint_name)
        self.assertIn("sns_topic_arn", alert_config)
        self.assertIn("rule_details", alert_config)


if __name__ == '__main__':
    unittest.main()