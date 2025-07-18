#!/usr/bin/env python3
"""
Unit tests for automated retraining trigger functionality.
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import boto3
from scripts.training.automated_retraining_trigger import AutomatedRetrainingManager


class TestAutomatedRetrainingManager(unittest.TestCase):
    """Test cases for AutomatedRetrainingManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock AWS clients
        self.mock_session_patcher = patch('boto3.Session')
        self.mock_session = self.mock_session_patcher.start()
        
        # Mock SageMaker client
        self.mock_sagemaker_client = MagicMock()
        self.mock_session.return_value.client.return_value = self.mock_sagemaker_client
        
        # Mock EventBridge integration
        self.mock_event_bridge_patcher = patch('scripts.training.automated_retraining_trigger.EventBridgeIntegration')
        self.mock_event_bridge = self.mock_event_bridge_patcher.start()
        self.mock_event_bridge_instance = MagicMock()
        self.mock_event_bridge.return_value = self.mock_event_bridge_instance
        
        # Mock SageMaker pipeline builder
        self.mock_pipeline_builder_patcher = patch('scripts.training.automated_retraining_trigger.SageMakerPipelineBuilder')
        self.mock_pipeline_builder = self.mock_pipeline_builder_patcher.start()
        self.mock_pipeline_builder_instance = MagicMock()
        self.mock_pipeline_builder.return_value = self.mock_pipeline_builder_instance
        self.mock_pipeline_builder_instance.default_bucket = "test-bucket"
        
        # Mock get_config
        self.mock_get_config_patcher = patch('scripts.training.automated_retraining_trigger.get_config')
        self.mock_get_config = self.mock_get_config_patcher.start()
        self.mock_get_config.return_value = {
            'project': {'name': 'test-project'}
        }
        
        # Create manager instance
        self.manager = AutomatedRetrainingManager(aws_profile="test", region="us-east-1")
        
        # Mock Lambda client
        self.manager.lambda_client = MagicMock()
        self.manager.lambda_client.create_function.return_value = {"FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:test-function"}
        
        # Mock SNS client
        self.manager.sns_client = MagicMock()
        self.manager.sns_client.create_topic.return_value = {"TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic"}
        
        # Mock IAM client
        self.manager.session.client.return_value = MagicMock()
        self.manager.session.client.return_value.create_role.return_value = {"Role": {"Arn": "arn:aws:iam::123456789012:role/test-role"}}
        
        # Mock EventBridge client
        self.manager.events_client = MagicMock()
        self.manager.events_client.put_rule.return_value = {"RuleArn": "arn:aws:events:us-east-1:123456789012:rule/test-rule"}
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_session_patcher.stop()
        self.mock_event_bridge_patcher.stop()
        self.mock_pipeline_builder_patcher.stop()
        self.mock_get_config_patcher.stop()
    
    @patch('scripts.training.automated_retraining_trigger.open')
    @patch('scripts.training.automated_retraining_trigger.zipfile.ZipFile')
    @patch('scripts.training.automated_retraining_trigger.time.sleep')
    def test_create_drift_detection_lambda(self, mock_sleep, mock_zipfile, mock_open):
        """Test creating a drift detection Lambda function."""
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = b"test"
        
        # Call the method
        lambda_arn = self.manager.create_drift_detection_lambda(
            endpoint_name="test-endpoint",
            pipeline_name="test-pipeline",
            drift_threshold=0.1
        )
        
        # Verify the result
        self.assertEqual(lambda_arn, "arn:aws:lambda:us-east-1:123456789012:function:test-function")
        
        # Verify Lambda creation
        self.manager.lambda_client.create_function.assert_called_once_with(
            FunctionName="test-project-drift-detection-test-endpoint",
            Runtime="python3.8",
            Role=ANY,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": b"test"},
            Description=ANY,
            Timeout=60,
            MemorySize=128,
            Publish=True
        )
    
    @patch('scripts.training.automated_retraining_trigger.open')
    @patch('scripts.training.automated_retraining_trigger.zipfile.ZipFile')
    @patch('scripts.training.automated_retraining_trigger.time.sleep')
    def test_create_scheduled_evaluation_lambda(self, mock_sleep, mock_zipfile, mock_open):
        """Test creating a scheduled evaluation Lambda function."""
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = b"test"
        
        # Call the method
        lambda_arn = self.manager.create_scheduled_evaluation_lambda(
            endpoint_name="test-endpoint",
            pipeline_name="test-pipeline",
            performance_threshold=0.7
        )
        
        # Verify the result
        self.assertEqual(lambda_arn, "arn:aws:lambda:us-east-1:123456789012:function:test-function")
        
        # Verify Lambda creation
        self.manager.lambda_client.create_function.assert_called_once_with(
            FunctionName="test-project-scheduled-evaluation-test-endpoint",
            Runtime="python3.8",
            Role=ANY,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": b"test"},
            Description=ANY,
            Timeout=300,
            MemorySize=256,
            Publish=True
        )
    
    def test_create_scheduled_evaluation_rule(self):
        """Test creating a scheduled evaluation rule."""
        # Call the method
        rule = self.manager.create_scheduled_evaluation_rule(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test-function",
            schedule_expression="rate(1 day)"
        )
        
        # Verify the result
        self.assertEqual(rule["rule_arn"], "arn:aws:events:us-east-1:123456789012:rule/test-rule")
        
        # Verify rule creation
        self.manager.events_client.put_rule.assert_called_once_with(
            Name=ANY,
            ScheduleExpression="rate(1 day)",
            State="ENABLED",
            Description=ANY
        )
        
        # Verify target addition
        self.manager.events_client.put_targets.assert_called_once()
    
    @patch('scripts.training.automated_retraining_trigger.open')
    @patch('scripts.training.automated_retraining_trigger.zipfile.ZipFile')
    @patch('scripts.training.automated_retraining_trigger.time.sleep')
    def test_create_approval_workflow_lambda(self, mock_sleep, mock_zipfile, mock_open):
        """Test creating an approval workflow Lambda function."""
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = b"test"
        
        # Call the method
        lambda_arn = self.manager.create_approval_workflow_lambda(
            pipeline_name="test-pipeline",
            approval_email="test@example.com"
        )
        
        # Verify the result
        self.assertEqual(lambda_arn, "arn:aws:lambda:us-east-1:123456789012:function:test-function")
        
        # Verify SNS topic creation
        self.manager.sns_client.create_topic.assert_called_once_with(Name="test-project-model-approval")
        
        # Verify email subscription
        self.manager.sns_client.subscribe.assert_called_once_with(
            TopicArn="arn:aws:sns:us-east-1:123456789012:test-topic",
            Protocol="email",
            Endpoint="test@example.com"
        )
    
    def test_setup_complete_retraining_solution(self):
        """Test setting up a complete retraining solution."""
        # Mock method calls
        self.manager.create_drift_detection_lambda = MagicMock(return_value="arn:aws:lambda:us-east-1:123456789012:function:drift-lambda")
        self.manager.create_scheduled_evaluation_lambda = MagicMock(return_value="arn:aws:lambda:us-east-1:123456789012:function:eval-lambda")
        self.manager.create_approval_workflow_lambda = MagicMock(return_value="arn:aws:lambda:us-east-1:123456789012:function:approval-lambda")
        self.manager.create_scheduled_evaluation_rule = MagicMock(return_value={"rule_arn": "arn:aws:events:us-east-1:123456789012:rule/eval-rule"})
        self.manager.create_pipeline_event_rule = MagicMock(return_value={"rule_arn": "arn:aws:events:us-east-1:123456789012:rule/pipeline-rule"})
        self.manager.create_model_package_event_rule = MagicMock(return_value={"rule_arn": "arn:aws:events:us-east-1:123456789012:rule/model-package-rule"})
        self.mock_event_bridge_instance.create_rule_for_model_drift.return_value = {"rule_arn": "arn:aws:events:us-east-1:123456789012:rule/drift-rule"}
        
        # Call the method
        result = self.manager.setup_complete_retraining_solution(
            endpoint_name="test-endpoint",
            pipeline_name="test-pipeline",
            email_notifications=["test@example.com"]
        )
        
        # Verify the result
        self.assertEqual(result["endpoint_name"], "test-endpoint")
        self.assertEqual(result["pipeline_name"], "test-pipeline")
        self.assertEqual(result["drift_lambda_arn"], "arn:aws:lambda:us-east-1:123456789012:function:drift-lambda")
        self.assertEqual(result["eval_lambda_arn"], "arn:aws:lambda:us-east-1:123456789012:function:eval-lambda")
        self.assertEqual(result["approval_lambda_arn"], "arn:aws:lambda:us-east-1:123456789012:function:approval-lambda")
        
        # Verify method calls
        self.manager.create_drift_detection_lambda.assert_called_once()
        self.manager.create_scheduled_evaluation_lambda.assert_called_once()
        self.manager.create_approval_workflow_lambda.assert_called_once()
        self.manager.create_scheduled_evaluation_rule.assert_called_once()
        self.manager.create_pipeline_event_rule.assert_called_once()
        self.manager.create_model_package_event_rule.assert_called_once()
        self.mock_event_bridge_instance.create_rule_for_model_drift.assert_called_once()


if __name__ == "__main__":
    unittest.main()