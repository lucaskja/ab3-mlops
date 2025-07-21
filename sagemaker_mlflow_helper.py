"""
SageMaker Managed MLflow Helper for Core Setup

This module provides utilities for using SageMaker managed MLflow tracking server in notebooks.
"""

import os
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import boto3
from typing import Dict, Any, Optional


class SageMakerMLflowHelper:
    """Helper class for SageMaker managed MLflow operations in notebooks."""
    
    def __init__(self, aws_profile: str = "ab"):
        """Initialize SageMaker MLflow helper."""
        self.aws_profile = aws_profile
        self.tracking_server_name = "sagemaker-core-setup-mlflow-server"
        self.tracking_server_url = "https://t-2vktx6phiclp.us-east-1.experiments.sagemaker.aws"
        
        # Set up AWS session
        os.environ['AWS_PROFILE'] = aws_profile
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker')
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Set up MLflow tracking with SageMaker managed server."""
        # Set the tracking URI
        mlflow.set_tracking_uri(self.tracking_server_url)
        
        # Configure MLflow for SageMaker managed server
        # SageMaker managed MLflow requires AWS credentials and tracking server ARN
        try:
            # Get the tracking server ARN
            response = self.sagemaker_client.describe_mlflow_tracking_server(
                TrackingServerName=self.tracking_server_name
            )
            tracking_server_arn = response['TrackingServerArn']
            
            # Set environment variables for SageMaker MLflow authentication
            os.environ['MLFLOW_TRACKING_AWS_SIGV4'] = 'true'
            os.environ['MLFLOW_TRACKING_SERVER_ARN'] = tracking_server_arn
            
            print(f"Connected to SageMaker managed MLflow server: {self.tracking_server_url}")
            
        except Exception as e:
            print(f"Warning: Could not configure SageMaker MLflow authentication: {e}")
            print("MLflow may not work properly with the managed server")
    
    def get_tracking_server_info(self) -> Dict[str, Any]:
        """Get information about the tracking server."""
        try:
            response = self.sagemaker_client.describe_mlflow_tracking_server(
                TrackingServerName=self.tracking_server_name
            )
            return {
                "name": response.get('TrackingServerName'),
                "url": response.get('TrackingServerUrl'),
                "status": response.get('TrackingServerStatus'),
                "artifact_store": response.get('ArtifactStoreUri'),
                "mlflow_version": response.get('MlflowVersion'),
                "size": response.get('TrackingServerSize')
            }
        except Exception as e:
            print(f"Error getting tracking server info: {e}")
            return {}
    
    def create_experiment(self, experiment_name: str, artifact_location: Optional[str] = None) -> str:
        """Create or get an MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                print(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
                return experiment_id
            else:
                print(f"Using existing MLflow experiment: {experiment_name}")
                mlflow.set_experiment(experiment_name)
                return experiment.experiment_id
        except Exception as e:
            print(f"Error creating/getting experiment: {e}")
            return ""
    
    def start_run(self, run_name: Optional[str] = None, experiment_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start an MLflow run."""
        if experiment_name:
            self.create_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a model to MLflow."""
        if hasattr(model, 'state_dict'):  # PyTorch model
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        else:  # Sklearn or other models
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def list_experiments(self):
        """List all experiments."""
        return mlflow.search_experiments()
    
    def list_runs(self, experiment_name: str, max_results: int = 10):
        """List recent runs in an experiment."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            return runs
        return None
    
    def get_mlflow_ui_url(self) -> str:
        """Get the MLflow UI URL."""
        return self.tracking_server_url


# Convenience function for notebooks
def get_sagemaker_mlflow_helper(aws_profile: str = "ab") -> SageMakerMLflowHelper:
    """Get a SageMaker MLflow helper instance."""
    return SageMakerMLflowHelper(aws_profile=aws_profile)


# Example usage for notebooks
def example_usage():
    """Example of how to use SageMaker managed MLflow in notebooks."""
    # Initialize MLflow helper
    mlflow_helper = get_sagemaker_mlflow_helper()
    
    # Get tracking server info
    server_info = mlflow_helper.get_tracking_server_info()
    print(f"Tracking Server Info: {server_info}")
    
    # Create an experiment
    experiment_id = mlflow_helper.create_experiment("core-setup-experiment")
    
    # Start a run
    with mlflow_helper.start_run(run_name="example_run", experiment_name="core-setup-experiment") as run:
        # Log parameters
        mlflow_helper.log_params({
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10
        })
        
        # Log metrics
        mlflow_helper.log_metrics({
            "accuracy": 0.95,
            "loss": 0.05
        })
        
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow UI: {mlflow_helper.get_mlflow_ui_url()}")
    
    # List experiments
    experiments = mlflow_helper.list_experiments()
    print(f"Available experiments: {len(experiments)}")
