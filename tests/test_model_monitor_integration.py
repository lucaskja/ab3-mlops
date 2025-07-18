#!/usr/bin/env python3
"""
Unit tests for Model Monitor Integration

Tests the integration between SageMaker Pipelines and Model Monitor.
"""

import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock, call
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.model_monitor_integration import (
    create_monitoring_lambda_step,
    add_monitoring_to_pipeline_endpoint,
    add_clarify_to_endpoint,
    create_clarify_lambda_step
)


class TestModelMonitorIntegration(unittest.TestCase):
    """Test Model Monitor Integration functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock for Lambda helper
        self.mock_lambda = MagicMock()
        self.lambda_patch = patch('src.pipeline.model_monitor_integration.Lambda', return_value=self.mock_lambda)
        self.mock_lambda_class = self.lambda_patch.start()
        
        # Create mock for create_model_monitor_for_endpoint
        self.mock_create_monitor = MagicMock(return_value={
            "endpoint_name": "test-endpoint",
            "data_capture_config": {
                "destination_s3_uri": "s3://test-bucket/data-capture",
                "sampling_percentage": 100.0
            },
            "monitoring_schedule": "test-endpoint-data-quality-monitoring"
        })
        self.create_monitor_patch = patch(
            'src.pipeline.model_monitor_integration.create_model_monitor_for_endpoint',
            self.mock_create_monitor
        )
        self.mock_create_monitor_func = self.create_monitor_patch.start()
        
        # Create mock for ClarifyManager
        self.mock_clarify_manager = MagicMock()
        self.mock_clarify_manager.run_explainability_analysis.return_value = "s3://test-bucket/clarify/explainability"
        self.mock_clarify_manager.run_bias_analysis.return_value = "s3://test-bucket/clarify/bias"
        self.mock_clarify_manager.setup_clarify_monitoring.return_value = {"schedule_name": "test-endpoint-clarify-monitoring"}
        self.mock_clarify_manager.generate_explainability_report.return_value = "s3://test-bucket/clarify/report.html"
        self.mock_clarify_manager.create_monitoring_dashboard.return_value = {"dashboard_name": "test-endpoint-clarify-dashboard"}
        
        self.clarify_manager_patch = patch(
            'src.pipeline.model_monitor_integration.ClarifyManager',
            return_value=self.mock_clarify_manager
        )
        self.mock_clarify_manager_class = self.clarify_manager_patch.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.lambda_patch.stop()
        self.create_monitor_patch.stop()
        self.clarify_manager_patch.stop()
    
    def test_create_monitoring_lambda_step(self):
        """Test creating a Lambda step for model monitoring"""
        # Call the function
        lambda_function_name = "test-monitoring-lambda"
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline"
        email_notifications = ["user@example.com"]
        role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        lambda_step = create_monitoring_lambda_step(
            lambda_function_name=lambda_function_name,
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            email_notifications=email_notifications,
            aws_profile="ab",
            region="us-east-1",
            role=role,
            enable_model_quality=True,
            problem_type="BinaryClassification",
            ground_truth_attribute="target",
            inference_attribute="prediction"
        )
        
        # Verify Lambda was created
        self.mock_lambda_class.assert_called_once()
        
        # Verify Lambda parameters
        call_args = self.mock_lambda_class.call_args[1]
        self.assertEqual(call_args["function_name"], lambda_function_name)
        self.assertEqual(call_args["execution_role_arn"], role)
        self.assertEqual(call_args["handler"], "index.lambda_handler")
        self.assertEqual(call_args["timeout"], 900)
        self.assertEqual(call_args["memory_size"], 512)
        
        # Verify Lambda code contains key elements
        lambda_code = call_args["script"]
        self.assertIn(endpoint_name, lambda_code)
        self.assertIn(baseline_dataset, lambda_code)
        self.assertIn("us-east-1", lambda_code)
        self.assertIn("BinaryClassification", lambda_code)
        self.assertIn("target", lambda_code)
        self.assertIn("prediction", lambda_code)
        self.assertIn("user@example.com", lambda_code)
        self.assertIn("true", lambda_code)  # enable_model_quality=True
        
        # Verify return value
        self.assertEqual(lambda_step, self.mock_lambda)
    
    def test_add_monitoring_to_pipeline_endpoint(self):
        """Test adding model monitoring to a pipeline endpoint"""
        # Call the function
        pipeline_name = "test-pipeline"
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline"
        email_notifications = ["user@example.com"]
        
        add_monitoring_to_pipeline_endpoint(
            pipeline_name=pipeline_name,
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            aws_profile="ab",
            region="us-east-1",
            email_notifications=email_notifications
        )
        
        # Verify create_model_monitor_for_endpoint was called
        self.mock_create_monitor_func.assert_called_once_with(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            aws_profile="ab",
            region="us-east-1",
            email_notifications=email_notifications
        )
    
    def test_add_clarify_to_endpoint(self):
        """Test adding SageMaker Clarify to an endpoint"""
        # Call the function
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline"
        target_column = "target"
        features = ["feature1", "feature2", "feature3"]
        sensitive_columns = ["feature1"]
        
        clarify_config = add_clarify_to_endpoint(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            sensitive_columns=sensitive_columns,
            aws_profile="ab",
            region="us-east-1"
        )
        
        # Verify ClarifyManager was created
        self.mock_clarify_manager_class.assert_called_once_with(
            aws_profile="ab",
            region="us-east-1"
        )
        
        # Verify explainability analysis was run
        self.mock_clarify_manager.run_explainability_analysis.assert_called_once_with(
            dataset_path=baseline_dataset,
            model_name=endpoint_name,
            model_endpoint_name=endpoint_name,
            features=features,
            target_column=target_column,
            wait=True
        )
        
        # Verify bias analysis was run
        self.mock_clarify_manager.run_bias_analysis.assert_called_once_with(
            dataset_path=baseline_dataset,
            model_name=endpoint_name,
            model_endpoint_name=endpoint_name,
            target_column=target_column,
            sensitive_columns=sensitive_columns,
            wait=True
        )
        
        # Verify clarify monitoring was set up
        self.mock_clarify_manager.setup_clarify_monitoring.assert_called_once_with(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            sensitive_columns=sensitive_columns
        )
        
        # Verify explainability report was generated
        self.mock_clarify_manager.generate_explainability_report.assert_called_once_with(
            clarify_output_path="s3://test-bucket/clarify/explainability",
            model_name=endpoint_name
        )
        
        # Verify monitoring dashboard was created
        self.mock_clarify_manager.create_monitoring_dashboard.assert_called_once_with(
            endpoint_name=endpoint_name,
            clarify_output_path="s3://test-bucket/clarify/explainability"
        )
        
        # Verify return value
        self.assertEqual(clarify_config["endpoint_name"], endpoint_name)
        self.assertEqual(clarify_config["explainability"]["output_path"], "s3://test-bucket/clarify/explainability")
        self.assertEqual(clarify_config["explainability"]["report_path"], "s3://test-bucket/clarify/report.html")
        self.assertEqual(clarify_config["bias"]["enabled"], True)
        self.assertEqual(clarify_config["bias"]["output_path"], "s3://test-bucket/clarify/bias")
        self.assertEqual(clarify_config["monitoring"]["schedule_name"], "test-endpoint-clarify-monitoring")
        self.assertEqual(clarify_config["dashboard"]["dashboard_name"], "test-endpoint-clarify-dashboard")
    
    def test_add_clarify_to_endpoint_without_sensitive_columns(self):
        """Test adding SageMaker Clarify to an endpoint without sensitive columns"""
        # Call the function
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline"
        target_column = "target"
        features = ["feature1", "feature2", "feature3"]
        
        clarify_config = add_clarify_to_endpoint(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            aws_profile="ab",
            region="us-east-1"
        )
        
        # Verify bias analysis was not run
        self.mock_clarify_manager.run_bias_analysis.assert_not_called()
        
        # Verify return value
        self.assertEqual(clarify_config["bias"]["enabled"], False)
        self.assertIsNone(clarify_config["bias"]["output_path"])
    
    def test_create_clarify_lambda_step(self):
        """Test creating a Lambda step for SageMaker Clarify"""
        # Call the function
        lambda_function_name = "test-clarify-lambda"
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline"
        target_column = "target"
        features = ["feature1", "feature2", "feature3"]
        sensitive_columns = ["feature1"]
        role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        lambda_step = create_clarify_lambda_step(
            lambda_function_name=lambda_function_name,
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            sensitive_columns=sensitive_columns,
            aws_profile="ab",
            region="us-east-1",
            role=role
        )
        
        # Verify Lambda was created
        self.mock_lambda_class.assert_called_once()
        
        # Verify Lambda parameters
        call_args = self.mock_lambda_class.call_args[1]
        self.assertEqual(call_args["function_name"], lambda_function_name)
        self.assertEqual(call_args["execution_role_arn"], role)
        self.assertEqual(call_args["handler"], "index.lambda_handler")
        
        # Verify Lambda code contains key elements
        lambda_code = call_args["script"]
        self.assertIn(endpoint_name, lambda_code)
        self.assertIn(baseline_dataset, lambda_code)
        self.assertIn("us-east-1", lambda_code)
        self.assertIn(target_column, lambda_code)
        self.assertIn(json.dumps(features), lambda_code)
        self.assertIn(json.dumps(sensitive_columns), lambda_code)
        
        # Verify return value
        self.assertEqual(lambda_step, self.mock_lambda)
    
    def test_create_clarify_lambda_step_without_sensitive_columns(self):
        """Test creating a Lambda step for SageMaker Clarify without sensitive columns"""
        # Call the function
        lambda_function_name = "test-clarify-lambda"
        endpoint_name = "test-endpoint"
        baseline_dataset = "s3://test-bucket/baseline"
        target_column = "target"
        features = ["feature1", "feature2", "feature3"]
        role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        lambda_step = create_clarify_lambda_step(
            lambda_function_name=lambda_function_name,
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            aws_profile="ab",
            region="us-east-1",
            role=role
        )
        
        # Verify Lambda code contains key elements
        lambda_code = self.mock_lambda_class.call_args[1]["script"]
        self.assertIn("None", lambda_code)  # sensitive_columns=None


if __name__ == '__main__':
    unittest.main()