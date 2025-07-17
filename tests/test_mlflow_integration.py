"""
Unit tests for MLFlow integration with SageMaker.

This module provides comprehensive tests for the MLFlow integration functionality,
ensuring proper experiment tracking, parameter logging, and model artifact management.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, ANY
import tempfile
import json
import boto3
import mlflow
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException

# Import the module to test
from src.pipeline.mlflow_integration import MLFlowSageMakerIntegration, get_mlflow_integration, log_training_job


class TestMLFlowSageMakerIntegration(unittest.TestCase):
    """Test cases for MLFlow SageMaker integration."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for MLFlow tracking
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tracking_uri = f"file://{self.temp_dir.name}"
        
        # Mock AWS session and clients
        self.mock_session = MagicMock()
        self.mock_sagemaker_client = MagicMock()
        self.mock_s3_client = MagicMock()
        self.mock_cloudwatch_client = MagicMock()
        
        # Set up patches
        self.boto3_session_patch = patch('boto3.Session', return_value=self.mock_session)
        self.get_config_patch = patch('src.pipeline.mlflow_integration.get_config', return_value={
            'iam': {'roles': {'sagemaker_execution': {'arn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'}}},
            's3': {'default_bucket': 'test-bucket'},
            'mlflow': {'tracking_uri': self.tracking_uri, 'artifact_bucket': 'mlflow-artifacts-bucket'}
        })
        
        # Start patches
        self.mock_boto3_session = self.boto3_session_patch.start()
        self.mock_get_config = self.get_config_patch.start()
        
        # Configure mock session
        self.mock_session.client.side_effect = lambda service, region_name=None: {
            'sagemaker': self.mock_sagemaker_client,
            's3': self.mock_s3_client,
            'cloudwatch': self.mock_cloudwatch_client
        }.get(service, MagicMock())
        
        # Initialize MLFlow tracking
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create integration instance
        self.mlflow_integration = MLFlowSageMakerIntegration(aws_profile="ab", region="us-east-1")

    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        self.boto3_session_patch.stop()
        self.get_config_patch.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of MLFlowSageMakerIntegration."""
        # Verify AWS session was created with correct profile
        self.mock_boto3_session.assert_called_once_with(profile_name="ab")
        
        # Verify clients were created
        self.mock_session.client.assert_any_call('sagemaker', region_name="us-east-1")
        self.mock_session.client.assert_any_call('s3', region_name="us-east-1")
        
        # Verify tracking URI was set
        self.assertEqual(mlflow.get_tracking_uri(), self.tracking_uri)
        
        # Verify AWS profile environment variable was set
        self.assertEqual(os.environ.get("AWS_PROFILE"), "ab")

    def test_create_experiment(self):
        """Test creating an MLFlow experiment."""
        # Create experiment
        experiment_name = "test-experiment"
        experiment_id = self.mlflow_integration.create_experiment(experiment_name)
        
        # Verify experiment was created
        experiment = mlflow.get_experiment_by_name(experiment_name)
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.experiment_id, experiment_id)
        
        # Verify artifact location
        expected_artifact_location = f"s3://mlflow-artifacts-bucket/mlflow-artifacts/{experiment_name}"
        self.assertTrue(experiment.artifact_location.endswith(experiment_name))
        
        # Test getting existing experiment
        experiment_id2 = self.mlflow_integration.create_experiment(experiment_name)
        self.assertEqual(experiment_id, experiment_id2)

    def test_start_run(self):
        """Test starting an MLFlow run."""
        # Create experiment
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        # Start run
        run_name = "test-run"
        with self.mlflow_integration.start_run(experiment_name, run_name=run_name) as run:
            # Verify run was created
            self.assertIsNotNone(run)
            self.assertIsNotNone(run.info.run_id)
            
            # Verify run is active
            active_run = mlflow.active_run()
            self.assertEqual(active_run.info.run_id, run.info.run_id)
            
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            # Verify run belongs to correct experiment
            self.assertEqual(run.info.experiment_id, experiment.experiment_id)
        
        # Verify run is no longer active after context exit
        self.assertIsNone(mlflow.active_run())

    def test_log_parameters(self):
        """Test logging parameters to an MLFlow run."""
        # Create experiment and start run
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        with self.mlflow_integration.start_run(experiment_name) as run:
            # Log parameters
            params = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "model_type": "yolov11",
                "complex_param": {"key": "value"}  # Should be converted to string
            }
            self.mlflow_integration.log_parameters(params)
            
            # Get run data
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(run.info.run_id).data
            
            # Verify parameters were logged
            self.assertEqual(run_data.params.get("learning_rate"), "0.001")
            self.assertEqual(run_data.params.get("batch_size"), "32")
            self.assertEqual(run_data.params.get("epochs"), "100")
            self.assertEqual(run_data.params.get("model_type"), "yolov11")
            self.assertEqual(run_data.params.get("complex_param"), "{'key': 'value'}")

    def test_log_metrics(self):
        """Test logging metrics to an MLFlow run."""
        # Create experiment and start run
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        with self.mlflow_integration.start_run(experiment_name) as run:
            # Log metrics
            metrics = {
                "train_loss": 0.25,
                "val_loss": 0.3,
                "val_mAP": 0.85
            }
            self.mlflow_integration.log_metrics(metrics)
            
            # Log metrics with step
            metrics_step_1 = {
                "train_loss": 0.2,
                "val_loss": 0.25,
                "val_mAP": 0.87
            }
            self.mlflow_integration.log_metrics(metrics_step_1, step=1)
            
            # Get run data
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(run.info.run_id).data
            
            # Verify metrics were logged
            self.assertEqual(run_data.metrics.get("train_loss"), 0.2)  # Latest value
            self.assertEqual(run_data.metrics.get("val_loss"), 0.25)   # Latest value
            self.assertEqual(run_data.metrics.get("val_mAP"), 0.87)    # Latest value
            
            # Get metric history
            train_loss_history = client.get_metric_history(run.info.run_id, "train_loss")
            
            # Verify metric history
            self.assertEqual(len(train_loss_history), 2)
            self.assertEqual(train_loss_history[0].value, 0.25)
            self.assertEqual(train_loss_history[0].step, 0)
            self.assertEqual(train_loss_history[1].value, 0.2)
            self.assertEqual(train_loss_history[1].step, 1)

    def test_log_artifact(self):
        """Test logging artifacts to an MLFlow run."""
        # Create experiment and start run
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        with self.mlflow_integration.start_run(experiment_name) as run:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_file.write("Test artifact content")
                temp_file_path = temp_file.name
            
            try:
                # Log artifact
                self.mlflow_integration.log_artifact(temp_file_path)
                
                # Log artifact with custom path
                self.mlflow_integration.log_artifact(temp_file_path, "custom_path")
                
                # Get artifact URI
                artifact_uri = mlflow.get_artifact_uri()
                
                # Verify artifact was logged
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run.info.run_id)
                
                # Should have the file and the custom_path directory
                self.assertEqual(len(artifacts), 2)
                
                # Check custom path artifacts
                custom_path_artifacts = client.list_artifacts(run.info.run_id, "custom_path")
                self.assertEqual(len(custom_path_artifacts), 1)
                self.assertEqual(os.path.basename(custom_path_artifacts[0].path), 
                                os.path.basename(temp_file_path))
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

    @patch('mlflow.pytorch.log_model')
    def test_log_model(self, mock_log_model):
        """Test logging a model to an MLFlow run."""
        # Create experiment and start run
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        with self.mlflow_integration.start_run(experiment_name) as run:
            # Create a mock model
            mock_model = MagicMock()
            
            # Log model
            self.mlflow_integration.log_model(mock_model, flavor="pytorch")
            
            # Verify model was logged
            mock_log_model.assert_called_once_with(mock_model, "model", registered_model_name=None)
            
            # Log model with registration
            mock_log_model.reset_mock()
            self.mlflow_integration.log_model(mock_model, flavor="pytorch", registered_model_name="test-model")
            
            # Verify model was logged and registered
            mock_log_model.assert_called_once_with(mock_model, "model", registered_model_name="test-model")

    def test_end_run(self):
        """Test ending an MLFlow run."""
        # Create experiment and start run
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        # Start run
        run = self.mlflow_integration.start_run(experiment_name)
        run_id = run.info.run_id
        
        # End run
        self.mlflow_integration.end_run()
        
        # Verify run is no longer active
        self.assertIsNone(mlflow.active_run())
        
        # Verify run status
        client = mlflow.tracking.MlflowClient()
        run_info = client.get_run(run_id).info
        self.assertEqual(run_info.status, "FINISHED")
        
        # Start another run
        run = self.mlflow_integration.start_run(experiment_name)
        run_id = run.info.run_id
        
        # End run with failed status
        self.mlflow_integration.end_run(status="FAILED")
        
        # Verify run status
        run_info = client.get_run(run_id).info
        self.assertEqual(run_info.status, "FAILED")

    def test_list_experiments(self):
        """Test listing MLFlow experiments."""
        # Create experiments
        experiment_names = ["test-experiment-1", "test-experiment-2", "test-experiment-3"]
        for name in experiment_names:
            self.mlflow_integration.create_experiment(name)
        
        # List experiments
        experiments = self.mlflow_integration.list_experiments()
        
        # Verify experiments were listed
        self.assertGreaterEqual(len(experiments), len(experiment_names))
        
        # Verify experiment details
        experiment_names_found = [exp["name"] for exp in experiments]
        for name in experiment_names:
            self.assertIn(name, experiment_names_found)

    def test_list_runs(self):
        """Test listing runs for an experiment."""
        # Create experiment
        experiment_name = "test-experiment"
        self.mlflow_integration.create_experiment(experiment_name)
        
        # Create runs
        run_names = ["run-1", "run-2", "run-3"]
        run_ids = []
        
        for name in run_names:
            with self.mlflow_integration.start_run(experiment_name, run_name=name) as run:
                run_ids.append(run.info.run_id)
                # Log some data to make runs unique
                self.mlflow_integration.log_parameters({"run_name": name})
        
        # List runs
        runs = self.mlflow_integration.list_runs(experiment_name)
        
        # Verify runs were listed
        self.assertEqual(len(runs), len(run_names))
        
        # Verify run details
        run_ids_found = [run["run_id"] for run in runs]
        for run_id in run_ids:
            self.assertIn(run_id, run_ids_found)
        
        # Verify run parameters
        for run in runs:
            self.assertEqual(run["params"]["run_name"], f"run-{run['run_id'][-1]}")

    @patch('boto3.Session')
    def test_log_sagemaker_job(self, mock_boto3_session):
        """Test logging a SageMaker training job to MLFlow."""
        # Set up mock responses
        mock_session = MagicMock()
        mock_sagemaker_client = MagicMock()
        mock_cloudwatch_client = MagicMock()
        
        mock_boto3_session.return_value = mock_session
        mock_session.client.side_effect = lambda service, region_name=None: {
            'sagemaker': mock_sagemaker_client,
            'cloudwatch': mock_cloudwatch_client
        }.get(service, MagicMock())
        
        # Mock SageMaker describe_training_job response
        mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed",
            "ResourceConfig": {
                "InstanceType": "ml.p3.2xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 50
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400
            },
            "AlgorithmSpecification": {
                "TrainingImage": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-pytorch:1.10.0-gpu-py38",
                "TrainingInputMode": "File"
            },
            "HyperParameters": {
                "learning-rate": "0.001",
                "batch-size": "32",
                "epochs": "100"
            },
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://test-bucket/model.tar.gz"
            }
        }
        
        # Mock CloudWatch get_metric_data response
        mock_cloudwatch_client.get_metric_data.return_value = {
            "MetricDataResults": [
                {
                    "Id": "train_loss",
                    "Timestamps": [1, 2, 3],
                    "Values": [0.5, 0.4, 0.3]
                }
            ]
        }
        
        # Create MLFlow integration with mocks
        mlflow_integration = MLFlowSageMakerIntegration()
        
        # Create experiment
        experiment_name = "test-experiment"
        mlflow_integration.create_experiment(experiment_name)
        
        # Log SageMaker job
        job_name = "test-job"
        run_id = mlflow_integration.log_sagemaker_job(job_name, experiment_name)
        
        # Verify job was logged
        mock_sagemaker_client.describe_training_job.assert_called_once_with(TrainingJobName=job_name)
        
        # Verify run was created
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Verify parameters were logged
        self.assertEqual(run.data.params.get("job_name"), "test-job")
        self.assertEqual(run.data.params.get("instance_type"), "ml.p3.2xlarge")
        self.assertEqual(run.data.params.get("hp_learning-rate"), "0.001")
        self.assertEqual(run.data.params.get("hp_batch-size"), "32")
        self.assertEqual(run.data.params.get("hp_epochs"), "100")
        self.assertEqual(run.data.params.get("model_artifacts"), "s3://test-bucket/model.tar.gz")

    @patch('mlflow.tracking.MlflowClient')
    def test_register_model(self, mock_mlflow_client):
        """Test registering a model from an MLFlow run."""
        # Set up mock client
        mock_client = MagicMock()
        mock_mlflow_client.return_value = mock_client
        
        # Mock get_run response
        mock_run = MagicMock()
        mock_client.get_run.return_value = mock_run
        
        # Mock create_model_version response
        mock_model_version = MagicMock()
        mock_model_version.name = "test-model"
        mock_model_version.version = "1"
        mock_model_version.status = "READY"
        mock_model_version.run_id = "test-run-id"
        mock_client.create_model_version.return_value = mock_model_version
        
        # Mock get_registered_model to raise exception (model doesn't exist)
        mock_client.get_registered_model.side_effect = MlflowException("Model not found")
        
        # Create MLFlow integration
        mlflow_integration = MLFlowSageMakerIntegration()
        
        # Register model
        run_id = "test-run-id"
        model_name = "test-model"
        description = "Test model description"
        result = mlflow_integration.register_model(run_id, model_name, description)
        
        # Verify model was registered
        mock_client.get_run.assert_called_once_with(run_id)
        mock_client.create_registered_model.assert_called_once_with(model_name, description)
        mock_client.create_model_version.assert_called_once_with(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            description=description
        )
        
        # Verify result
        self.assertEqual(result["name"], "test-model")
        self.assertEqual(result["version"], "1")
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["run_id"], "test-run-id")

    @patch('sagemaker.mlflow.MlflowModelServer')
    def test_deploy_model_to_sagemaker(self, mock_mlflow_model_server):
        """Test deploying an MLFlow model to SageMaker."""
        # Set up mock model server
        mock_server = MagicMock()
        mock_mlflow_model_server.return_value = mock_server
        
        # Create MLFlow integration
        mlflow_integration = MLFlowSageMakerIntegration()
        
        # Deploy model
        model_name = "test-model"
        version = "1"
        instance_type = "ml.m5.large"
        instance_count = 1
        result = mlflow_integration.deploy_model_to_sagemaker(
            model_name, version, instance_type, instance_count
        )
        
        # Verify model was deployed
        mock_mlflow_model_server.assert_called_once_with(
            model_uri=f"models:/{model_name}/{version}",
            role=ANY,
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=ANY
        )
        
        mock_server.deploy.assert_called_once_with(endpoint_name=f"{model_name}-{version}")
        
        # Verify result
        self.assertEqual(result["model_name"], "test-model")
        self.assertEqual(result["model_version"], "1")
        self.assertEqual(result["endpoint_name"], "test-model-1")
        self.assertEqual(result["instance_type"], "ml.m5.large")
        self.assertEqual(result["instance_count"], 1)

    def test_delete_endpoint(self):
        """Test deleting a SageMaker endpoint."""
        # Create MLFlow integration
        mlflow_integration = MLFlowSageMakerIntegration()
        
        # Delete endpoint
        endpoint_name = "test-endpoint"
        mlflow_integration.delete_endpoint(endpoint_name)
        
        # Verify endpoint was deleted
        self.mock_sagemaker_client.delete_endpoint.assert_called_once_with(EndpointName=endpoint_name)

    def test_helper_functions(self):
        """Test helper functions."""
        # Test get_mlflow_integration
        with patch('src.pipeline.mlflow_integration.MLFlowSageMakerIntegration') as mock_integration:
            mock_instance = MagicMock()
            mock_integration.return_value = mock_instance
            
            integration = get_mlflow_integration(aws_profile="test-profile", region="us-west-2")
            
            mock_integration.assert_called_once_with(aws_profile="test-profile", region="us-west-2")
            self.assertEqual(integration, mock_instance)
        
        # Test log_training_job
        with patch('src.pipeline.mlflow_integration.get_mlflow_integration') as mock_get_integration:
            mock_instance = MagicMock()
            mock_get_integration.return_value = mock_instance
            mock_instance.log_sagemaker_job.return_value = "test-run-id"
            
            run_id = log_training_job(
                job_name="test-job",
                experiment_name="test-experiment",
                run_name="test-run",
                aws_profile="test-profile"
            )
            
            mock_get_integration.assert_called_once_with(aws_profile="test-profile")
            mock_instance.log_sagemaker_job.assert_called_once_with(
                "test-job", "test-experiment", "test-run"
            )
            self.assertEqual(run_id, "test-run-id")


if __name__ == '__main__':
    unittest.main()