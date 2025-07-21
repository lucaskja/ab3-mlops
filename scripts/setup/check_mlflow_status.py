#!/usr/bin/env python3
"""
Check MLflow Tracking Server Status and Complete Setup

This script checks the status of the SageMaker managed MLflow tracking server
and completes the setup once it's ready.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, Any, Optional

import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_mlflow_status(aws_profile: str = "ab", region: str = "us-east-1", 
                       tracking_server_name: str = "sagemaker-core-setup-mlflow-server"):
    """Check the status of the MLflow tracking server."""
    
    # Set up AWS session
    os.environ['AWS_PROFILE'] = aws_profile
    session = boto3.Session(profile_name=aws_profile)
    sagemaker_client = session.client('sagemaker')
    s3_client = session.client('s3')
    
    try:
        response = sagemaker_client.describe_mlflow_tracking_server(
            TrackingServerName=tracking_server_name
        )
        
        status = response['TrackingServerStatus']
        url = response.get('TrackingServerUrl', '')
        mlflow_version = response.get('MlflowVersion', '')
        server_size = response.get('TrackingServerSize', '')
        artifact_store = response.get('ArtifactStoreUri', '')
        
        print(f"MLflow Tracking Server Status: {status}")
        print(f"Server Name: {tracking_server_name}")
        print(f"Server URL: {url}")
        print(f"MLflow Version: {mlflow_version}")
        print(f"Server Size: {server_size}")
        print(f"Artifact Store: {artifact_store}")
        print(f"Creation Time: {response.get('CreationTime', '')}")
        
        if status == 'Created':
            print("\n✅ MLflow tracking server is ready!")
            
            # Create helper script with the correct URL
            create_helper_script(s3_client, url, tracking_server_name, region)
            
            print("\n📋 Next Steps:")
            print("1. Open your enhanced notebooks:")
            print("   - notebooks/data-scientist-core-enhanced.ipynb")
            print("   - notebooks/ml-engineer-core-enhanced.ipynb")
            print("\n2. Add the MLflow integration code:")
            print("   - Download the helper script from S3")
            print("   - Use the SageMaker managed MLflow server")
            print(f"\n3. Access the MLflow UI: {url}")
            
            return True
            
        elif status in ['Failed', 'DeleteFailed']:
            print(f"\n❌ MLflow tracking server creation failed!")
            if 'FailureReason' in response:
                print(f"Failure reason: {response['FailureReason']}")
            return False
            
        else:
            print(f"\n⏳ MLflow tracking server is still {status.lower()}...")
            print("Please wait and check again in a few minutes.")
            return False
            
    except Exception as e:
        print(f"Error checking MLflow tracking server: {e}")
        return False


def create_helper_script(s3_client, tracking_server_url: str, tracking_server_name: str, region: str):
    """Create and upload the helper script with the correct tracking server URL."""
    
    helper_script = f'''"""
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
        self.tracking_server_name = "{tracking_server_name}"
        self.tracking_server_url = "{tracking_server_url}"
        
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
            
            print(f"Connected to SageMaker managed MLflow server: {{self.tracking_server_url}}")
            
        except Exception as e:
            print(f"Warning: Could not configure SageMaker MLflow authentication: {{e}}")
            print("MLflow may not work properly with the managed server")
    
    def get_tracking_server_info(self) -> Dict[str, Any]:
        """Get information about the tracking server."""
        try:
            response = self.sagemaker_client.describe_mlflow_tracking_server(
                TrackingServerName=self.tracking_server_name
            )
            return {{
                "name": response.get('TrackingServerName'),
                "url": response.get('TrackingServerUrl'),
                "status": response.get('TrackingServerStatus'),
                "artifact_store": response.get('ArtifactStoreUri'),
                "mlflow_version": response.get('MlflowVersion'),
                "size": response.get('TrackingServerSize')
            }}
        except Exception as e:
            print(f"Error getting tracking server info: {{e}}")
            return {{}}
    
    def create_experiment(self, experiment_name: str, artifact_location: Optional[str] = None) -> str:
        """Create or get an MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                print(f"Created MLflow experiment: {{experiment_name}} (ID: {{experiment_id}})")
                return experiment_id
            else:
                print(f"Using existing MLflow experiment: {{experiment_name}}")
                mlflow.set_experiment(experiment_name)
                return experiment.experiment_id
        except Exception as e:
            print(f"Error creating/getting experiment: {{e}}")
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
    print(f"Tracking Server Info: {{server_info}}")
    
    # Create an experiment
    experiment_id = mlflow_helper.create_experiment("core-setup-experiment")
    
    # Start a run
    with mlflow_helper.start_run(run_name="example_run", experiment_name="core-setup-experiment") as run:
        # Log parameters
        mlflow_helper.log_params({{
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10
        }})
        
        # Log metrics
        mlflow_helper.log_metrics({{
            "accuracy": 0.95,
            "loss": 0.05
        }})
        
        print(f"Run ID: {{run.info.run_id}}")
        print(f"MLflow UI: {{mlflow_helper.get_mlflow_ui_url()}}")
    
    # List experiments
    experiments = mlflow_helper.list_experiments()
    print(f"Available experiments: {{len(experiments)}}")
'''
    
    # Save helper script to S3
    bucket_name = "lucaskle-ab3-project-pv"
    helper_key = "mlflow-sagemaker/utils/sagemaker_mlflow_helper.py"
    
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=helper_key,
            Body=helper_script,
            ContentType='text/plain'
        )
        
        helper_uri = f"s3://{bucket_name}/{helper_key}"
        print(f"✓ Updated MLflow helper script: {helper_uri}")
        
        # Create notebook example
        notebook_example = f'''# SageMaker Managed MLflow Integration
# Add this to your enhanced notebooks

# Install MLflow if not already installed
# !pip install mlflow boto3

# Download the helper script
import boto3
s3_client = boto3.client('s3', region_name='{region}')
s3_client.download_file('{bucket_name}', '{helper_key}', 'sagemaker_mlflow_helper.py')

# Import and use the helper
from sagemaker_mlflow_helper import get_sagemaker_mlflow_helper

# Initialize MLflow helper
mlflow_helper = get_sagemaker_mlflow_helper(aws_profile="ab")

# Get server info
server_info = mlflow_helper.get_tracking_server_info()
print(f"MLflow Server: {{server_info['url']}}")
print(f"Status: {{server_info['status']}}")

# Create an experiment and start tracking
experiment_name = "my-experiment"
mlflow_helper.create_experiment(experiment_name)

with mlflow_helper.start_run(run_name="my_run", experiment_name=experiment_name) as run:
    # Log parameters and metrics
    mlflow_helper.log_params({{"param1": "value1", "param2": 42}})
    mlflow_helper.log_metrics({{"metric1": 0.95, "metric2": 0.05}})
    
    print(f"Run ID: {{run.info.run_id}}")
    print(f"MLflow UI: {{mlflow_helper.get_mlflow_ui_url()}}")
'''
        
        # Save notebook example
        example_key = "mlflow-sagemaker/examples/notebook_integration_example.py"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=example_key,
            Body=notebook_example,
            ContentType='text/plain'
        )
        
        print(f"✓ Updated notebook example: s3://{bucket_name}/{example_key}")
        
    except Exception as e:
        print(f"Warning: Could not update helper script: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check MLflow Tracking Server Status")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--server-name", default="sagemaker-core-setup-mlflow-server", help="Tracking server name")
    parser.add_argument("--wait", action="store_true", help="Wait for server to be ready")
    parser.add_argument("--wait-timeout", type=int, default=30, help="Max minutes to wait")
    
    args = parser.parse_args()
    
    if args.wait:
        print(f"Waiting for MLflow tracking server to be ready (max {args.wait_timeout} minutes)...")
        max_attempts = args.wait_timeout * 4  # Check every 15 seconds
        attempt = 0
        
        while attempt < max_attempts:
            if check_mlflow_status(args.profile, args.region, args.server_name):
                print("\n🎉 MLflow tracking server is ready and setup is complete!")
                sys.exit(0)
            
            print("Waiting 15 seconds before checking again...")
            time.sleep(15)
            attempt += 1
        
        print(f"\n⏰ Timeout after {args.wait_timeout} minutes. Server may still be creating.")
        print("You can check the status manually later using:")
        print(f"python {sys.argv[0]} --profile {args.profile}")
        sys.exit(1)
    else:
        success = check_mlflow_status(args.profile, args.region, args.server_name)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
