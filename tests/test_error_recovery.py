"""
Unit tests for SageMaker training job error recovery mechanisms.

This module provides tests for the error recovery functionality,
ensuring proper handling of failed training jobs, automatic retries,
and failure analysis.
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import boto3
import botocore
import json
import time
from datetime import datetime

# Import the module to test
from src.pipeline.error_recovery import TrainingJobRecoveryManager, get_recovery_manager, retry_failed_job, monitor_training_job


class TestTrainingJobRecoveryManager(unittest.TestCase):
    """Test cases for SageMaker training job recovery manager."""

    def setUp(self):
        """Set up test environment before each test."""
        # Mock AWS session and clients
        self.mock_session = MagicMock()
        self.mock_sagemaker_client = MagicMock()
        self.mock_cloudwatch_client = MagicMock()
        self.mock_logs_client = MagicMock()
        self.mock_s3_client = MagicMock()
        
        # Set up patches
        self.boto3_session_patch = patch('boto3.Session', return_value=self.mock_session)
        self.get_config_patch = patch('src.pipeline.error_recovery.get_config', return_value={
            'aws': {'profile': 'ab', 'region': 'us-east-1'},
            'training': {'instance_type': 'ml.g4dn.xlarge'}
        })
        
        # Start patches
        self.mock_boto3_session = self.boto3_session_patch.start()
        self.mock_get_config = self.get_config_patch.start()
        
        # Configure mock session
        self.mock_session.client.side_effect = lambda service, region_name=None: {
            'sagemaker': self.mock_sagemaker_client,
            'cloudwatch': self.mock_cloudwatch_client,
            'logs': self.mock_logs_client,
            's3': self.mock_s3_client
        }.get(service, MagicMock())
        
        # Create recovery manager instance
        self.recovery_manager = TrainingJobRecoveryManager(aws_profile="ab", region="us-east-1")

    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        self.boto3_session_patch.stop()
        self.get_config_patch.stop()

    def test_init(self):
        """Test initialization of TrainingJobRecoveryManager."""
        # Verify AWS session was created with correct profile
        self.mock_boto3_session.assert_called_once_with(profile_name="ab")
        
        # Verify clients were created
        self.mock_session.client.assert_any_call('sagemaker', region_name="us-east-1")
        self.mock_session.client.assert_any_call('cloudwatch', region_name="us-east-1")
        self.mock_session.client.assert_any_call('logs', region_name="us-east-1")

    def test_retry_with_backoff_success(self):
        """Test retry_with_backoff with successful function call."""
        # Mock function that succeeds
        mock_func = MagicMock(return_value="success")
        
        # Call retry_with_backoff
        result = self.recovery_manager.retry_with_backoff(mock_func)
        
        # Verify function was called once
        mock_func.assert_called_once()
        self.assertEqual(result, "success")

    def test_retry_with_backoff_retry(self):
        """Test retry_with_backoff with function that fails then succeeds."""
        # Mock function that fails once then succeeds
        mock_func = MagicMock(side_effect=[
            botocore.exceptions.ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
                "operation"
            ),
            "success"
        ])
        
        # Patch time.sleep to avoid waiting
        with patch('time.sleep'):
            # Call retry_with_backoff
            result = self.recovery_manager.retry_with_backoff(mock_func)
        
        # Verify function was called twice
        self.assertEqual(mock_func.call_count, 2)
        self.assertEqual(result, "success")

    def test_retry_with_backoff_max_retries(self):
        """Test retry_with_backoff with function that always fails."""
        # Mock function that always fails
        error = botocore.exceptions.ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "operation"
        )
        mock_func = MagicMock(side_effect=error)
        
        # Patch time.sleep to avoid waiting
        with patch('time.sleep'):
            # Call retry_with_backoff and verify it raises exception
            with self.assertRaises(botocore.exceptions.ClientError):
                self.recovery_manager.retry_with_backoff(mock_func, max_retries=2)
        
        # Verify function was called max_retries + 1 times
        self.assertEqual(mock_func.call_count, 3)

    def test_get_training_job_status(self):
        """Test getting training job status."""
        # Mock response
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed"
        }
        
        # Call get_training_job_status
        result = self.recovery_manager.get_training_job_status("test-job")
        
        # Verify client was called
        self.mock_sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        
        # Verify result
        self.assertEqual(result["TrainingJobStatus"], "Completed")

    def test_get_training_job_status_not_found(self):
        """Test getting status of non-existent training job."""
        # Mock response
        self.mock_sagemaker_client.describe_training_job.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "ResourceNotFound", "Message": "Resource not found"}},
            "DescribeTrainingJob"
        )
        
        # Call get_training_job_status
        result = self.recovery_manager.get_training_job_status("non-existent-job")
        
        # Verify client was called
        self.mock_sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="non-existent-job"
        )
        
        # Verify result
        self.assertEqual(result["TrainingJobStatus"], "NotFound")

    def test_analyze_job_failure(self):
        """Test analyzing job failure."""
        # Mock job details
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobName": "failed-job",
            "TrainingJobStatus": "Failed",
            "FailureReason": "OutOfMemoryError: Container killed due to memory usage",
            "ResourceConfig": {
                "InstanceType": "ml.g4dn.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400
            },
            "TrainingEndTime": datetime.now()
        }
        
        # Mock log streams
        self.mock_logs_client.describe_log_streams.return_value = {
            "logStreams": [
                {"logStreamName": "failed-job/algo-1-1234"}
            ]
        }
        
        # Mock log events
        self.mock_logs_client.get_log_events.return_value = {
            "events": [
                {
                    "timestamp": 1234567890,
                    "message": "Error: OutOfMemoryError: Container killed due to memory usage"
                }
            ]
        }
        
        # Call analyze_job_failure
        result = self.recovery_manager.analyze_job_failure("failed-job")
        
        # Verify clients were called
        self.mock_sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="failed-job"
        )
        self.mock_logs_client.describe_log_streams.assert_called_once()
        self.mock_logs_client.get_log_events.assert_called_once()
        
        # Verify result
        self.assertEqual(result["job_name"], "failed-job")
        self.assertEqual(result["status"], "Failed")
        self.assertEqual(result["error_type"], "OutOfMemoryError")
        self.assertTrue(result["logs_analysis"]["has_errors"])
        self.assertEqual(len(result["logs_analysis"]["error_messages"]), 1)

    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint."""
        # Mock job details
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "OutputDataConfig": {
                "S3OutputPath": "s3://test-bucket/output"
            }
        }
        
        # Mock S3 objects
        self.mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "output/test-job/checkpoints/checkpoint-1.pt",
                    "LastModified": datetime(2023, 1, 1),
                    "Size": 1000
                },
                {
                    "Key": "output/test-job/checkpoints/checkpoint-2.pt",
                    "LastModified": datetime(2023, 1, 2),
                    "Size": 1000
                }
            ]
        }
        
        # Call get_latest_checkpoint
        result = self.recovery_manager.get_latest_checkpoint("test-job")
        
        # Verify clients were called
        self.mock_sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        self.mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket",
            Prefix="output/test-job/checkpoints/"
        )
        
        # Verify result
        self.assertEqual(result, "s3://test-bucket/output/test-job/checkpoints/checkpoint-2.pt")

    def test_retry_failed_job(self):
        """Test retrying a failed job."""
        # Mock job details
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobName": "failed-job",
            "TrainingJobStatus": "Failed",
            "FailureReason": "OutOfMemoryError: Container killed due to memory usage",
            "AlgorithmSpecification": {
                "TrainingImage": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-pytorch:1.10.0-gpu-py38",
                "TrainingInputMode": "File"
            },
            "RoleArn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3Uri": "s3://test-bucket/train",
                            "S3DataType": "S3Prefix"
                        }
                    }
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": "s3://test-bucket/output"
            },
            "ResourceConfig": {
                "InstanceType": "ml.g4dn.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400
            },
            "HyperParameters": {
                "epochs": "10",
                "batch-size": "32"
            }
        }
        
        # Mock create_training_job response
        self.mock_sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/new-job"
        }
        
        # Mock get_latest_checkpoint
        with patch.object(self.recovery_manager, 'get_latest_checkpoint', return_value="s3://test-bucket/output/failed-job/checkpoints/checkpoint.pt"):
            # Call retry_failed_job
            result = self.recovery_manager.retry_failed_job(
                job_name="failed-job",
                new_job_name="new-job",
                use_checkpoints=True,
                adjust_resources=True
            )
        
        # Verify clients were called
        self.mock_sagemaker_client.describe_training_job.assert_called_with(
            TrainingJobName="failed-job"
        )
        self.mock_sagemaker_client.create_training_job.assert_called_once()
        
        # Verify create_training_job was called with correct parameters
        call_args = self.mock_sagemaker_client.create_training_job.call_args[1]
        self.assertEqual(call_args["TrainingJobName"], "new-job")
        self.assertEqual(call_args["ResourceConfig"]["InstanceType"], "ml.g4dn.2xlarge")  # Upgraded instance
        self.assertTrue("checkpoint_path" in call_args["HyperParameters"])
        self.assertTrue("resume_from_checkpoint" in call_args["HyperParameters"])
        
        # Verify result
        self.assertEqual(result["original_job_name"], "failed-job")
        self.assertEqual(result["new_job_name"], "new-job")
        self.assertTrue(result["checkpoint_used"])
        self.assertTrue(result["resources_adjusted"])

    def test_monitor_training_job_completed(self):
        """Test monitoring a training job that completes successfully."""
        # Mock job details
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed"
        }
        
        # Call monitor_training_job
        result = self.recovery_manager.monitor_training_job("test-job")
        
        # Verify client was called
        self.mock_sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        
        # Verify result
        self.assertEqual(result["final_status"], "Completed")
        self.assertEqual(result["job_name"], "test-job")
        self.assertEqual(result["retry_count"], 0)

    def test_monitor_training_job_failed_with_retry(self):
        """Test monitoring a training job that fails and is retried."""
        # Mock job details for first call (failed job)
        self.mock_sagemaker_client.describe_training_job.side_effect = [
            {
                "TrainingJobName": "test-job",
                "TrainingJobStatus": "Failed",
                "FailureReason": "ResourceLimitExceeded: Resource limit exceeded"
            },
            {
                "TrainingJobName": "test-job-retry-123",
                "TrainingJobStatus": "Completed"
            }
        ]
        
        # Mock analyze_job_failure
        with patch.object(self.recovery_manager, 'analyze_job_failure', return_value={"is_retryable": True}):
            # Mock retry_failed_job
            with patch.object(self.recovery_manager, 'retry_failed_job', return_value={"new_job_name": "test-job-retry-123"}):
                # Call monitor_training_job
                result = self.recovery_manager.monitor_training_job("test-job")
        
        # Verify result
        self.assertEqual(result["final_status"], "Completed")
        self.assertEqual(result["job_name"], "test-job-retry-123")
        self.assertEqual(result["original_job_name"], "test-job")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(len(result["job_history"]), 2)

    def test_get_recovery_manager(self):
        """Test get_recovery_manager helper function."""
        with patch('src.pipeline.error_recovery.TrainingJobRecoveryManager') as mock_manager:
            # Call get_recovery_manager
            recovery_manager = get_recovery_manager(aws_profile="test-profile", region="us-west-2")
            
            # Verify TrainingJobRecoveryManager was called with correct parameters
            mock_manager.assert_called_once_with(aws_profile="test-profile", region="us-west-2")

    def test_retry_failed_job_helper(self):
        """Test retry_failed_job helper function."""
        with patch('src.pipeline.error_recovery.get_recovery_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager
            mock_manager.retry_failed_job.return_value = {"new_job_name": "new-job"}
            
            # Call retry_failed_job
            result = retry_failed_job(
                job_name="failed-job",
                new_job_name="new-job",
                use_checkpoints=True,
                adjust_resources=False,
                aws_profile="test-profile"
            )
            
            # Verify get_recovery_manager was called with correct parameters
            mock_get_manager.assert_called_once_with(aws_profile="test-profile")
            
            # Verify retry_failed_job was called with correct parameters
            mock_manager.retry_failed_job.assert_called_once_with(
                job_name="failed-job",
                new_job_name="new-job",
                use_checkpoints=True,
                adjust_resources=False
            )
            
            # Verify result
            self.assertEqual(result, {"new_job_name": "new-job"})

    def test_monitor_training_job_helper(self):
        """Test monitor_training_job helper function."""
        with patch('src.pipeline.error_recovery.get_recovery_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager
            mock_manager.monitor_training_job.return_value = {"final_status": "Completed"}
            
            # Call monitor_training_job
            result = monitor_training_job(
                job_name="test-job",
                polling_interval=60,
                auto_retry=False,
                max_retries=2,
                aws_profile="test-profile"
            )
            
            # Verify get_recovery_manager was called with correct parameters
            mock_get_manager.assert_called_once_with(aws_profile="test-profile")
            
            # Verify monitor_training_job was called with correct parameters
            mock_manager.monitor_training_job.assert_called_once_with(
                job_name="test-job",
                polling_interval=60,
                auto_retry=False,
                max_retries=2
            )
            
            # Verify result
            self.assertEqual(result, {"final_status": "Completed"})


if __name__ == '__main__':
    unittest.main()