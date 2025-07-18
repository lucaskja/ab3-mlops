#!/usr/bin/env python3
"""
Unit tests for Drift Detection

Tests the drift detection functionality for SageMaker Model Monitor.
"""

import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.drift_detection import DriftDetector


class TestDriftDetector(unittest.TestCase):
    """Test DriftDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock AWS session and clients
        self.mock_session = MagicMock()
        self.mock_sagemaker_client = MagicMock()
        self.mock_cloudwatch_client = MagicMock()
        self.mock_s3_client = MagicMock()
        
        # Configure mock session to return mock clients
        self.mock_session.client.side_effect = lambda service, region_name=None: {
            'sagemaker': self.mock_sagemaker_client,
            'cloudwatch': self.mock_cloudwatch_client,
            's3': self.mock_s3_client
        }.get(service, MagicMock())
        
        # Create patch for boto3.Session
        self.boto3_session_patch = patch('boto3.Session', return_value=self.mock_session)
        self.boto3_session_patch.start()
        
        # Create DriftDetector instance
        self.endpoint_name = "test-endpoint"
        self.bucket = "test-bucket"
        self.prefix = "monitoring"
        self.detector = DriftDetector(
            endpoint_name=self.endpoint_name,
            bucket=self.bucket,
            prefix=self.prefix,
            profile_name="ab",
            region_name="us-east-1"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.boto3_session_patch.stop()
    
    def test_initialization(self):
        """Test initialization of DriftDetector"""
        # Verify boto3 session was created with correct profile and region
        from boto3 import Session
        Session.assert_called_once_with(profile_name="ab", region_name="us-east-1")
        
        # Verify clients were created
        self.mock_session.client.assert_any_call('sagemaker', region_name="us-east-1")
        self.mock_session.client.assert_any_call('cloudwatch', region_name="us-east-1")
        self.mock_session.client.assert_any_call('s3', region_name="us-east-1")
        
        # Verify instance variables
        self.assertEqual(self.detector.endpoint_name, self.endpoint_name)
        self.assertEqual(self.detector.bucket, self.bucket)
        self.assertEqual(self.detector.prefix, self.prefix)
        self.assertEqual(self.detector.sagemaker_client, self.mock_sagemaker_client)
        self.assertEqual(self.detector.cloudwatch_client, self.mock_cloudwatch_client)
        self.assertEqual(self.detector.s3_client, self.mock_s3_client)
    
    def test_create_data_quality_baseline(self):
        """Test creating a data quality baseline"""
        # Configure mock response
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-job"
        }
        
        # Call the method
        baseline_dataset = "s3://test-bucket/baseline-data"
        job_name = self.detector.create_data_quality_baseline(
            baseline_dataset=baseline_dataset,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600
        )
        
        # Verify create_monitoring_schedule was called
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called_once()
        
        # Verify job name format
        self.assertTrue(job_name.startswith(f"{self.endpoint_name}-data-quality-baseline-"))
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_monitoring_schedule.call_args[1]
        self.assertEqual(call_args["MonitoringScheduleName"], job_name)
        
        # Verify monitoring job definition
        job_def = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        
        # Verify baseline config
        self.assertIn("BaselineConfig", job_def)
        self.assertIn("ConstraintsResource", job_def["BaselineConfig"])
        self.assertIn("StatisticsResource", job_def["BaselineConfig"])
        
        # Verify monitoring inputs
        self.assertIn("MonitoringInputs", job_def)
        self.assertEqual(len(job_def["MonitoringInputs"]), 1)
        self.assertEqual(job_def["MonitoringInputs"][0]["S3Input"]["S3Uri"], baseline_dataset)
        
        # Verify resources
        self.assertEqual(job_def["MonitoringResources"]["ClusterConfig"]["InstanceType"], "ml.m5.xlarge")
        self.assertEqual(job_def["MonitoringResources"]["ClusterConfig"]["InstanceCount"], 1)
        
        # Verify stopping condition
        self.assertEqual(job_def["StoppingCondition"]["MaxRuntimeInSeconds"], 3600)
    
    def test_create_model_quality_baseline(self):
        """Test creating a model quality baseline"""
        # Configure mock response
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-job"
        }
        
        # Call the method
        baseline_dataset = "s3://test-bucket/baseline-data"
        problem_type = "BinaryClassification"
        job_name = self.detector.create_model_quality_baseline(
            baseline_dataset=baseline_dataset,
            problem_type=problem_type,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600
        )
        
        # Verify create_monitoring_schedule was called
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called_once()
        
        # Verify job name format
        self.assertTrue(job_name.startswith(f"{self.endpoint_name}-model-quality-baseline-"))
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_monitoring_schedule.call_args[1]
        self.assertEqual(call_args["MonitoringScheduleName"], job_name)
        
        # Verify monitoring job definition
        job_def = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        
        # Verify environment variables
        self.assertIn("Environment", job_def)
        self.assertEqual(job_def["Environment"]["PROBLEM_TYPE"], problem_type)
    
    def test_create_data_quality_monitoring_schedule(self):
        """Test creating a data quality monitoring schedule"""
        # Configure mock response
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-schedule"
        }
        
        # Call the method
        baseline_constraints_uri = "s3://test-bucket/constraints.json"
        baseline_statistics_uri = "s3://test-bucket/statistics.json"
        schedule_name = self.detector.create_data_quality_monitoring_schedule(
            baseline_constraints_uri=baseline_constraints_uri,
            baseline_statistics_uri=baseline_statistics_uri,
            schedule_expression="cron(0 * ? * * *)",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600
        )
        
        # Verify create_monitoring_schedule was called
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called_once()
        
        # Verify schedule name
        self.assertEqual(schedule_name, f"{self.endpoint_name}-data-quality-monitoring")
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.create_monitoring_schedule.call_args[1]
        self.assertEqual(call_args["MonitoringScheduleName"], schedule_name)
        
        # Verify schedule config
        schedule_config = call_args["MonitoringScheduleConfig"]
        self.assertEqual(schedule_config["ScheduleConfig"]["ScheduleExpression"], "cron(0 * ? * * *)")
        
        # Verify monitoring job definition
        job_def = schedule_config["MonitoringJobDefinition"]
        
        # Verify baseline config
        self.assertEqual(job_def["BaselineConfig"]["ConstraintsResource"]["S3Uri"], baseline_constraints_uri)
        self.assertEqual(job_def["BaselineConfig"]["StatisticsResource"]["S3Uri"], baseline_statistics_uri)
        
        # Verify monitoring inputs
        self.assertIn("MonitoringInputs", job_def)
        self.assertEqual(len(job_def["MonitoringInputs"]), 1)
        self.assertEqual(job_def["MonitoringInputs"][0]["EndpointInput"]["EndpointName"], self.endpoint_name)
        
        # Verify output config
        output_s3_uri = f"s3://{self.bucket}/{self.prefix}/monitoring/data-quality"
        self.assertEqual(job_def["MonitoringOutputConfig"]["MonitoringOutputs"][0]["S3Output"]["S3Uri"], output_s3_uri)
    
    def test_create_model_quality_monitoring_schedule(self):
        """Test creating a model quality monitoring schedule"""
        # Configure mock response
        self.mock_sagemaker_client.create_monitoring_schedule.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-east-1:123456789012:monitoring-schedule/test-schedule"
        }
        
        # Call the method
        baseline_constraints_uri = "s3://test-bucket/constraints.json"
        baseline_statistics_uri = "s3://test-bucket/statistics.json"
        ground_truth_input = "s3://test-bucket/ground-truth"
        problem_type = "BinaryClassification"
        schedule_name = self.detector.create_model_quality_monitoring_schedule(
            baseline_constraints_uri=baseline_constraints_uri,
            baseline_statistics_uri=baseline_statistics_uri,
            ground_truth_input=ground_truth_input,
            problem_type=problem_type,
            schedule_expression="cron(0 * ? * * *)",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            max_runtime_seconds=3600
        )
        
        # Verify create_monitoring_schedule was called
        self.mock_sagemaker_client.create_monitoring_schedule.assert_called_once()
        
        # Verify schedule name
        self.assertEqual(schedule_name, f"{self.endpoint_name}-model-quality-monitoring")
        
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
        self.assertIn("Environment", job_def)
        self.assertEqual(job_def["Environment"]["PROBLEM_TYPE"], problem_type)
    
    def test_create_drift_alert(self):
        """Test creating a CloudWatch alarm for drift detection"""
        # Call the method
        metric_name = "feature_baseline_drift_check_violations"
        threshold = 0
        sns_topic_arn = "arn:aws:sns:us-east-1:123456789012:test-topic"
        alarm_name = self.detector.create_drift_alert(
            metric_name=metric_name,
            threshold=threshold,
            comparison_operator="GreaterThanThreshold",
            evaluation_periods=1,
            period=3600,
            statistic="Maximum",
            sns_topic_arn=sns_topic_arn
        )
        
        # Verify put_metric_alarm was called
        self.mock_cloudwatch_client.put_metric_alarm.assert_called_once()
        
        # Verify alarm name
        expected_alarm_name = f"{self.endpoint_name}-{metric_name}-drift-alarm"
        self.assertEqual(alarm_name, expected_alarm_name)
        
        # Verify call arguments
        call_args = self.mock_cloudwatch_client.put_metric_alarm.call_args[1]
        self.assertEqual(call_args["AlarmName"], expected_alarm_name)
        self.assertEqual(call_args["MetricName"], metric_name)
        self.assertEqual(call_args["Namespace"], "AWS/SageMaker")
        self.assertEqual(call_args["Dimensions"][0]["Name"], "EndpointName")
        self.assertEqual(call_args["Dimensions"][0]["Value"], self.endpoint_name)
        self.assertEqual(call_args["Statistic"], "Maximum")
        self.assertEqual(call_args["Period"], 3600)
        self.assertEqual(call_args["EvaluationPeriods"], 1)
        self.assertEqual(call_args["Threshold"], 0)
        self.assertEqual(call_args["ComparisonOperator"], "GreaterThanThreshold")
        self.assertEqual(call_args["AlarmActions"], [sns_topic_arn])
    
    def test_get_monitoring_results(self):
        """Test getting monitoring results"""
        # Configure mock response
        self.mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'monitoring/data-quality/2023-01-01/violations.json',
                    'LastModified': datetime(2023, 1, 1, 12, 0, 0)
                },
                {
                    'Key': 'monitoring/data-quality/2023-01-02/violations.json',
                    'LastModified': datetime(2023, 1, 2, 12, 0, 0)
                }
            ]
        }
        
        # Mock get_object responses
        def mock_get_object(Bucket, Key):
            if Key == 'monitoring/data-quality/2023-01-01/violations.json':
                return {
                    'Body': MagicMock(read=lambda: json.dumps([
                        {"feature_name": "feature1", "constraint_check_type": "drift_check"}
                    ]).encode('utf-8'))
                }
            elif Key == 'monitoring/data-quality/2023-01-02/violations.json':
                return {
                    'Body': MagicMock(read=lambda: json.dumps([
                        {"feature_name": "feature2", "constraint_check_type": "drift_check"}
                    ]).encode('utf-8'))
                }
            return {'Body': MagicMock(read=lambda: b'{}')}
        
        self.mock_s3_client.get_object.side_effect = mock_get_object
        
        # Call the method
        results = self.detector.get_monitoring_results(
            monitoring_type="data-quality",
            max_results=10
        )
        
        # Verify list_objects_v2 was called
        self.mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket=self.bucket,
            Prefix="monitoring/data-quality",
            MaxKeys=100
        )
        
        # Verify get_object was called for each violations file
        self.assertEqual(self.mock_s3_client.get_object.call_count, 2)
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        # Results should be sorted by date (newest first)
        self.assertEqual(results[0]['execution_date'], datetime(2023, 1, 2, 12, 0, 0))
        self.assertEqual(results[1]['execution_date'], datetime(2023, 1, 1, 12, 0, 0))
        
        # Verify violations
        self.assertEqual(results[0]['violations'][0]['feature_name'], 'feature2')
        self.assertEqual(results[1]['violations'][0]['feature_name'], 'feature1')
    
    def test_analyze_drift(self):
        """Test analyzing drift over time"""
        # Mock get_monitoring_results
        with patch.object(self.detector, 'get_monitoring_results') as mock_get_results:
            # Configure mock response
            mock_get_results.return_value = [
                {
                    'execution_date': datetime.now() - timedelta(days=1),
                    'violations': [
                        {"feature_name": "feature1", "constraint_check_type": "drift_check"},
                        {"feature_name": "feature2", "constraint_check_type": "drift_check"}
                    ],
                    's3_uri': 's3://test-bucket/result1'
                },
                {
                    'execution_date': datetime.now() - timedelta(days=2),
                    'violations': [
                        {"feature_name": "feature1", "constraint_check_type": "drift_check"}
                    ],
                    's3_uri': 's3://test-bucket/result2'
                },
                {
                    'execution_date': datetime.now() - timedelta(days=3),
                    'violations': [],
                    's3_uri': 's3://test-bucket/result3'
                }
            ]
            
            # Call the method
            drift_metrics = self.detector.analyze_drift(
                monitoring_type="data-quality",
                days=7
            )
            
            # Verify get_monitoring_results was called
            mock_get_results.assert_called_once_with(
                monitoring_type="data-quality",
                max_results=100
            )
            
            # Verify drift metrics
            self.assertEqual(drift_metrics['total_executions'], 3)
            self.assertEqual(drift_metrics['executions_with_violations'], 2)
            self.assertEqual(drift_metrics['violation_counts']['feature1'], 2)
            self.assertEqual(drift_metrics['violation_counts']['feature2'], 1)
            self.assertEqual(drift_metrics['time_period'], "Last 7 days")
            self.assertEqual(drift_metrics['monitoring_type'], "data-quality")
    
    def test_trigger_retraining(self):
        """Test triggering model retraining"""
        # Configure mock response
        self.mock_sagemaker_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
        }
        
        # Call the method
        pipeline_name = "test-pipeline"
        pipeline_parameters = {
            "LearningRate": "0.001",
            "BatchSize": "32"
        }
        execution_arn = self.detector.trigger_retraining(
            pipeline_name=pipeline_name,
            pipeline_parameters=pipeline_parameters
        )
        
        # Verify start_pipeline_execution was called
        self.mock_sagemaker_client.start_pipeline_execution.assert_called_once()
        
        # Verify call arguments
        call_args = self.mock_sagemaker_client.start_pipeline_execution.call_args[1]
        self.assertEqual(call_args["PipelineName"], pipeline_name)
        
        # Verify pipeline parameters
        pipeline_params = call_args["PipelineParameters"]
        
        # Convert list of dicts to dict for easier verification
        param_dict = {param['Name']: param['Value'] for param in pipeline_params}
        
        # Verify default parameters
        self.assertEqual(param_dict["EndpointName"], self.endpoint_name)
        self.assertEqual(param_dict["RetrainingReason"], "DriftDetected")
        
        # Verify custom parameters
        self.assertEqual(param_dict["LearningRate"], "0.001")
        self.assertEqual(param_dict["BatchSize"], "32")
        
        # Verify return value
        self.assertEqual(execution_arn, "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution")


if __name__ == '__main__':
    unittest.main()