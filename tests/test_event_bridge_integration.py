"""
Unit tests for EventBridge integration with SageMaker Pipeline and Model Monitoring.

This module provides tests for the EventBridge integration functionality,
ensuring proper event rule creation, SNS topic management, and custom event publishing.
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import boto3

# Import the module to test
from src.pipeline.event_bridge_integration import (
    EventBridgeIntegration, 
    get_event_bridge_integration,
    create_pipeline_failure_alert,
    create_model_drift_alert
)


class TestEventBridgeIntegration(unittest.TestCase):
    """Test cases for EventBridge integration."""

    def setUp(self):
        """Set up test environment before each test."""
        # Mock AWS session and clients
        self.mock_session = MagicMock()
        self.mock_events_client = MagicMock()
        self.mock_lambda_client = MagicMock()
        self.mock_sns_client = MagicMock()
        
        # Set up patches
        self.boto3_session_patch = patch('boto3.Session', return_value=self.mock_session)
        self.get_config_patch = patch('src.pipeline.event_bridge_integration.get_config', return_value={
            'project': {'name': 'mlops-sagemaker-demo'},
            'aws': {'profile': 'ab', 'region': 'us-east-1'}
        })
        
        # Start patches
        self.mock_boto3_session = self.boto3_session_patch.start()
        self.mock_get_config = self.get_config_patch.start()
        
        # Configure mock session
        self.mock_session.client.side_effect = lambda service, region_name=None: {
            'events': self.mock_events_client,
            'lambda': self.mock_lambda_client,
            'sns': self.mock_sns_client
        }.get(service, MagicMock())
        
        # Create EventBridgeIntegration instance
        self.event_bridge = EventBridgeIntegration(aws_profile="ab", region="us-east-1")

    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        self.boto3_session_patch.stop()
        self.get_config_patch.stop()

    def test_init(self):
        """Test initialization of EventBridgeIntegration."""
        # Verify AWS session was created with correct profile
        self.mock_boto3_session.assert_called_once_with(profile_name="ab")
        
        # Verify clients were created
        self.mock_session.client.assert_any_call('events', region_name="us-east-1")
        self.mock_session.client.assert_any_call('lambda', region_name="us-east-1")
        self.mock_session.client.assert_any_call('sns', region_name="us-east-1")
        
        # Verify project name was set
        self.assertEqual(self.event_bridge.project_name, "mlops-sagemaker-demo")
        
        # Verify event bus name was set
        self.assertEqual(self.event_bridge.event_bus_name, "default")

    def test_create_rule_for_pipeline_failures(self):
        """Test creating an EventBridge rule for pipeline failures."""
        # Mock put_rule response
        self.mock_events_client.put_rule.return_value = {
            "RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        }
        
        # Call create_rule_for_pipeline_failures
        result = self.event_bridge.create_rule_for_pipeline_failures(
            pipeline_name="test-pipeline",
            target_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            target_sns_arn="arn:aws:sns:us-east-1:123456789012:test-topic"
        )
        
        # Verify put_rule was called with correct parameters
        self.mock_events_client.put_rule.assert_called_once()
        args, kwargs = self.mock_events_client.put_rule.call_args
        self.assertEqual(kwargs["Name"], "mlops-sagemaker-demo-pipeline-failure-test-pipeline")
        self.assertEqual(kwargs["State"], "ENABLED")
        
        # Verify event pattern
        event_pattern = json.loads(kwargs["EventPattern"])
        self.assertEqual(event_pattern["source"], ["aws.sagemaker"])
        self.assertEqual(event_pattern["detail-type"], ["SageMaker Model Building Pipeline Execution Status Change"])
        self.assertEqual(event_pattern["detail"]["currentPipelineExecutionStatus"], ["Failed"])
        
        # Verify put_targets was called with correct parameters
        self.mock_events_client.put_targets.assert_called_once()
        args, kwargs = self.mock_events_client.put_targets.call_args
        self.assertEqual(kwargs["Rule"], "mlops-sagemaker-demo-pipeline-failure-test-pipeline")
        self.assertEqual(len(kwargs["Targets"]), 2)  # Lambda and SNS targets
        
        # Verify Lambda permission was added
        self.mock_lambda_client.add_permission.assert_called_once()
        
        # Verify result
        self.assertEqual(result["rule_name"], "mlops-sagemaker-demo-pipeline-failure-test-pipeline")
        self.assertEqual(result["rule_arn"], "arn:aws:events:us-east-1:123456789012:rule/test-rule")
        self.assertEqual(len(result["targets"]), 2)

    def test_create_rule_for_model_drift(self):
        """Test creating an EventBridge rule for model drift."""
        # Mock put_rule response
        self.mock_events_client.put_rule.return_value = {
            "RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        }
        
        # Call create_rule_for_model_drift
        result = self.event_bridge.create_rule_for_model_drift(
            endpoint_name="test-endpoint",
            target_sns_arn="arn:aws:sns:us-east-1:123456789012:test-topic"
        )
        
        # Verify put_rule was called with correct parameters
        self.mock_events_client.put_rule.assert_called_once()
        args, kwargs = self.mock_events_client.put_rule.call_args
        self.assertEqual(kwargs["Name"], "mlops-sagemaker-demo-model-drift-test-endpoint")
        self.assertEqual(kwargs["State"], "ENABLED")
        
        # Verify event pattern
        event_pattern = json.loads(kwargs["EventPattern"])
        self.assertEqual(event_pattern["source"], ["aws.sagemaker"])
        self.assertEqual(event_pattern["detail-type"], ["SageMaker Model Monitor Scheduled Rule Status Change"])
        self.assertEqual(event_pattern["detail"]["monitoringExecutionStatus"], ["Completed"])
        
        # Verify put_targets was called with correct parameters
        self.mock_events_client.put_targets.assert_called_once()
        args, kwargs = self.mock_events_client.put_targets.call_args
        self.assertEqual(kwargs["Rule"], "mlops-sagemaker-demo-model-drift-test-endpoint")
        self.assertEqual(len(kwargs["Targets"]), 1)  # SNS target
        
        # Verify result
        self.assertEqual(result["rule_name"], "mlops-sagemaker-demo-model-drift-test-endpoint")
        self.assertEqual(result["rule_arn"], "arn:aws:events:us-east-1:123456789012:rule/test-rule")
        self.assertEqual(len(result["targets"]), 1)

    def test_create_sns_topic_for_alerts(self):
        """Test creating an SNS topic for alerts."""
        # Mock create_topic response
        self.mock_sns_client.create_topic.return_value = {
            "TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic"
        }
        
        # Call create_sns_topic_for_alerts
        result = self.event_bridge.create_sns_topic_for_alerts(
            topic_name="test-topic",
            email_subscriptions=["test@example.com"]
        )
        
        # Verify create_topic was called with correct parameters
        self.mock_sns_client.create_topic.assert_called_once_with(Name="test-topic")
        
        # Verify subscribe was called with correct parameters
        self.mock_sns_client.subscribe.assert_called_once_with(
            TopicArn="arn:aws:sns:us-east-1:123456789012:test-topic",
            Protocol="email",
            Endpoint="test@example.com"
        )
        
        # Verify result
        self.assertEqual(result, "arn:aws:sns:us-east-1:123456789012:test-topic")

    def test_publish_custom_event(self):
        """Test publishing a custom event to EventBridge."""
        # Mock put_events response
        self.mock_events_client.put_events.return_value = {
            "Entries": [{"EventId": "1234567890"}]
        }
        
        # Call publish_custom_event
        result = self.event_bridge.publish_custom_event(
            detail_type=EventBridgeIntegration.EVENT_TYPES["MODEL_DRIFT_DETECTED"],
            detail={"endpoint": "test-endpoint", "drift_value": 0.25},
            resources=["arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"]
        )
        
        # Verify put_events was called with correct parameters
        self.mock_events_client.put_events.assert_called_once()
        args, kwargs = self.mock_events_client.put_events.call_args
        entries = kwargs["Entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["Source"], "com.mlops.sagemaker.demo")
        self.assertEqual(entries[0]["DetailType"], "ModelDriftDetected")
        self.assertEqual(entries[0]["EventBusName"], "default")
        
        # Verify detail
        detail = json.loads(entries[0]["Detail"])
        self.assertEqual(detail["endpoint"], "test-endpoint")
        self.assertEqual(detail["drift_value"], 0.25)
        
        # Verify resources
        self.assertEqual(entries[0]["Resources"], ["arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"])

    def test_create_rule_for_custom_events(self):
        """Test creating an EventBridge rule for custom events."""
        # Mock put_rule response
        self.mock_events_client.put_rule.return_value = {
            "RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        }
        
        # Call create_rule_for_custom_events
        result = self.event_bridge.create_rule_for_custom_events(
            detail_type=EventBridgeIntegration.EVENT_TYPES["MODEL_DRIFT_DETECTED"],
            target_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            rule_name="test-rule",
            event_pattern_detail={"drift_value": [{"numeric": [">", 0.2]}]}
        )
        
        # Verify put_rule was called with correct parameters
        self.mock_events_client.put_rule.assert_called_once()
        args, kwargs = self.mock_events_client.put_rule.call_args
        self.assertEqual(kwargs["Name"], "test-rule")
        self.assertEqual(kwargs["State"], "ENABLED")
        
        # Verify event pattern
        event_pattern = json.loads(kwargs["EventPattern"])
        self.assertEqual(event_pattern["source"], ["com.mlops.sagemaker.demo"])
        self.assertEqual(event_pattern["detail-type"], ["ModelDriftDetected"])
        self.assertEqual(event_pattern["detail"]["drift_value"], [{"numeric": [">", 0.2]}])
        
        # Verify put_targets was called with correct parameters
        self.mock_events_client.put_targets.assert_called_once()
        args, kwargs = self.mock_events_client.put_targets.call_args
        self.assertEqual(kwargs["Rule"], "test-rule")
        self.assertEqual(len(kwargs["Targets"]), 1)  # Lambda target
        
        # Verify Lambda permission was added
        self.mock_lambda_client.add_permission.assert_called_once()
        
        # Verify result
        self.assertEqual(result["rule_name"], "test-rule")
        self.assertEqual(result["rule_arn"], "arn:aws:events:us-east-1:123456789012:rule/test-rule")
        self.assertEqual(len(result["targets"]), 1)

    def test_create_retraining_trigger(self):
        """Test creating a retraining trigger."""
        # Mock put_rule response
        self.mock_events_client.put_rule.return_value = {
            "RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"
        }
        
        # Call create_retraining_trigger
        result = self.event_bridge.create_retraining_trigger(
            endpoint_name="test-endpoint",
            pipeline_name="test-pipeline",
            drift_threshold=0.2,
            target_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function"
        )
        
        # Verify create_rule_for_model_drift was called
        self.mock_events_client.put_rule.assert_called_once()
        
        # Verify result
        self.assertEqual(result["endpoint_name"], "test-endpoint")
        self.assertEqual(result["pipeline_name"], "test-pipeline")
        self.assertEqual(result["drift_threshold"], 0.2)
        self.assertIn("rule_details", result)

    def test_list_rules(self):
        """Test listing EventBridge rules."""
        # Mock list_rules response
        self.mock_events_client.list_rules.return_value = {
            "Rules": [
                {"Name": "rule1", "Arn": "arn1"},
                {"Name": "rule2", "Arn": "arn2"}
            ]
        }
        
        # Call list_rules
        result = self.event_bridge.list_rules(name_prefix="test")
        
        # Verify list_rules was called with correct parameters
        self.mock_events_client.list_rules.assert_called_once_with(NamePrefix="test")
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["Name"], "rule1")
        self.assertEqual(result[1]["Name"], "rule2")

    def test_delete_rule(self):
        """Test deleting an EventBridge rule."""
        # Mock list_targets_by_rule response
        self.mock_events_client.list_targets_by_rule.return_value = {
            "Targets": [
                {"Id": "target1"},
                {"Id": "target2"}
            ]
        }
        
        # Call delete_rule
        self.event_bridge.delete_rule("test-rule")
        
        # Verify list_targets_by_rule was called with correct parameters
        self.mock_events_client.list_targets_by_rule.assert_called_once_with(Rule="test-rule")
        
        # Verify remove_targets was called with correct parameters
        self.mock_events_client.remove_targets.assert_called_once_with(
            Rule="test-rule",
            Ids=["target1", "target2"]
        )
        
        # Verify delete_rule was called with correct parameters
        self.mock_events_client.delete_rule.assert_called_once_with(Name="test-rule")

    def test_get_event_bridge_integration(self):
        """Test get_event_bridge_integration helper function."""
        with patch('src.pipeline.event_bridge_integration.EventBridgeIntegration') as mock_integration:
            # Call get_event_bridge_integration
            integration = get_event_bridge_integration(aws_profile="test-profile", region="us-west-2")
            
            # Verify EventBridgeIntegration was called with correct parameters
            mock_integration.assert_called_once_with(aws_profile="test-profile", region="us-west-2")

    def test_create_pipeline_failure_alert(self):
        """Test create_pipeline_failure_alert helper function."""
        with patch('src.pipeline.event_bridge_integration.get_event_bridge_integration') as mock_get_integration:
            mock_integration = MagicMock()
            mock_get_integration.return_value = mock_integration
            mock_integration.create_sns_topic_for_alerts.return_value = "arn:aws:sns:us-east-1:123456789012:test-topic"
            mock_integration.create_rule_for_pipeline_failures.return_value = {"rule_name": "test-rule"}
            
            # Call create_pipeline_failure_alert
            result = create_pipeline_failure_alert(
                pipeline_name="test-pipeline",
                email_notifications=["test@example.com"],
                aws_profile="test-profile"
            )
            
            # Verify get_event_bridge_integration was called with correct parameters
            mock_get_integration.assert_called_once_with(aws_profile="test-profile")
            
            # Verify create_sns_topic_for_alerts was called with correct parameters
            mock_integration.create_sns_topic_for_alerts.assert_called_once_with(
                topic_name="pipeline-failure-test-pipeline",
                email_subscriptions=["test@example.com"]
            )
            
            # Verify create_rule_for_pipeline_failures was called with correct parameters
            mock_integration.create_rule_for_pipeline_failures.assert_called_once_with(
                pipeline_name="test-pipeline",
                target_sns_arn="arn:aws:sns:us-east-1:123456789012:test-topic"
            )
            
            # Verify result
            self.assertEqual(result["pipeline_name"], "test-pipeline")
            self.assertEqual(result["sns_topic_arn"], "arn:aws:sns:us-east-1:123456789012:test-topic")
            self.assertEqual(result["rule_details"], {"rule_name": "test-rule"})

    def test_create_model_drift_alert(self):
        """Test create_model_drift_alert helper function."""
        with patch('src.pipeline.event_bridge_integration.get_event_bridge_integration') as mock_get_integration:
            mock_integration = MagicMock()
            mock_get_integration.return_value = mock_integration
            mock_integration.create_sns_topic_for_alerts.return_value = "arn:aws:sns:us-east-1:123456789012:test-topic"
            mock_integration.create_rule_for_model_drift.return_value = {"rule_name": "test-rule"}
            
            # Call create_model_drift_alert
            result = create_model_drift_alert(
                endpoint_name="test-endpoint",
                email_notifications=["test@example.com"],
                aws_profile="test-profile"
            )
            
            # Verify get_event_bridge_integration was called with correct parameters
            mock_get_integration.assert_called_once_with(aws_profile="test-profile")
            
            # Verify create_sns_topic_for_alerts was called with correct parameters
            mock_integration.create_sns_topic_for_alerts.assert_called_once_with(
                topic_name="model-drift-test-endpoint",
                email_subscriptions=["test@example.com"]
            )
            
            # Verify create_rule_for_model_drift was called with correct parameters
            mock_integration.create_rule_for_model_drift.assert_called_once_with(
                endpoint_name="test-endpoint",
                target_sns_arn="arn:aws:sns:us-east-1:123456789012:test-topic"
            )
            
            # Verify result
            self.assertEqual(result["endpoint_name"], "test-endpoint")
            self.assertEqual(result["sns_topic_arn"], "arn:aws:sns:us-east-1:123456789012:test-topic")
            self.assertEqual(result["rule_details"], {"rule_name": "test-rule"})


if __name__ == '__main__':
    unittest.main()