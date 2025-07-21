#!/usr/bin/env python3
"""
Setup SageMaker Managed MLflow Tracking Server

This script creates a managed MLflow tracking server using Amazon SageMaker AI with MLflow,
which provides a fully managed MLflow experience with web UI and better integration.
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
import botocore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SageMakerMLflowSetup:
    """Sets up SageMaker managed MLflow tracking server."""
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the SageMaker MLflow setup.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Set up AWS session
        os.environ['AWS_PROFILE'] = aws_profile
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker')
        self.s3_client = self.session.client('s3')
        self.iam_client = self.session.client('iam')
        
        # Project configuration
        self.project_name = "sagemaker-core-setup"
        self.bucket_name = "lucaskle-ab3-project-pv"
        self.tracking_server_name = f"{self.project_name}-mlflow-server"
        
        # Get account info
        sts_client = self.session.client('sts')
        identity = sts_client.get_caller_identity()
        self.account_id = identity['Account']
        
        logger.info(f"Initialized SageMaker MLflow setup with profile: {aws_profile}")
        logger.info(f"Account ID: {self.account_id}")
        logger.info(f"Region: {region}")
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        # Check S3 bucket access
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✓ S3 bucket access verified: {self.bucket_name}")
        except Exception as e:
            logger.error(f"✗ Cannot access S3 bucket {self.bucket_name}: {e}")
            return False
        
        # Check if SageMaker domain exists
        try:
            domains = self.sagemaker_client.list_domains()
            if not domains['Domains']:
                logger.error("✗ No SageMaker domains found. Please run core SageMaker setup first.")
                return False
            logger.info(f"✓ Found {len(domains['Domains'])} SageMaker domain(s)")
        except Exception as e:
            logger.error(f"✗ Error checking SageMaker domains: {e}")
            return False
        
        return True
    
    def get_execution_role_arn(self) -> str:
        """Get the SageMaker execution role ARN."""
        try:
            role_name = "mlops-sagemaker-demo-SageMaker-Execution-Role"
            role_response = self.iam_client.get_role(RoleName=role_name)
            return role_response['Role']['Arn']
        except Exception as e:
            logger.error(f"Could not find SageMaker execution role: {e}")
            # Fallback to constructing the ARN
            return f"arn:aws:iam::{self.account_id}:role/{role_name}"
    
    def check_existing_tracking_server(self) -> Optional[Dict[str, Any]]:
        """Check if MLflow tracking server already exists."""
        try:
            response = self.sagemaker_client.describe_mlflow_tracking_server(
                TrackingServerName=self.tracking_server_name
            )
            logger.info(f"Found existing MLflow tracking server: {self.tracking_server_name}")
            return response
        except self.sagemaker_client.exceptions.ResourceNotFound:
            logger.info(f"No existing MLflow tracking server found: {self.tracking_server_name}")
            return None
        except Exception as e:
            logger.warning(f"Error checking for existing tracking server: {e}")
            return None
    
    def create_mlflow_tracking_server(self) -> Dict[str, Any]:
        """Create SageMaker managed MLflow tracking server."""
        logger.info(f"Creating MLflow tracking server: {self.tracking_server_name}")
        
        # Get execution role
        execution_role = self.get_execution_role_arn()
        
        # Create artifact store URI
        artifact_store_uri = f"s3://{self.bucket_name}/mlflow-artifacts"
        
        try:
            response = self.sagemaker_client.create_mlflow_tracking_server(
                TrackingServerName=self.tracking_server_name,
                ArtifactStoreUri=artifact_store_uri,
                RoleArn=execution_role,
                TrackingServerSize="Small",  # Small, Medium, or Large
                # MlflowVersion="2.10.2",  # Let SageMaker use the default supported version
                Tags=[
                    {
                        'Key': 'Project',
                        'Value': self.project_name
                    },
                    {
                        'Key': 'Environment',
                        'Value': 'core-setup'
                    },
                    {
                        'Key': 'CreatedBy',
                        'Value': 'sagemaker-core-setup'
                    }
                ]
            )
            
            tracking_server_arn = response['TrackingServerArn']
            logger.info(f"✓ MLflow tracking server creation initiated: {tracking_server_arn}")
            
            return {
                'TrackingServerArn': tracking_server_arn,
                'TrackingServerName': self.tracking_server_name,
                'ArtifactStoreUri': artifact_store_uri,
                'RoleArn': execution_role,
                'Status': 'Creating'
            }
            
        except Exception as e:
            logger.error(f"Failed to create MLflow tracking server: {e}")
            raise
    
    def wait_for_tracking_server(self, max_wait_minutes: int = 15) -> Dict[str, Any]:
        """Wait for the tracking server to be ready."""
        logger.info(f"Waiting for tracking server to be ready (max {max_wait_minutes} minutes)...")
        
        max_attempts = max_wait_minutes * 4  # Check every 15 seconds
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = self.sagemaker_client.describe_mlflow_tracking_server(
                    TrackingServerName=self.tracking_server_name
                )
                
                status = response['TrackingServerStatus']
                logger.info(f"Tracking server status: {status}")
                
                if status == 'Created':
                    logger.info("✅ MLflow tracking server is ready!")
                    return response
                elif status in ['Failed', 'DeleteFailed']:
                    logger.error(f"❌ MLflow tracking server creation failed with status: {status}")
                    if 'FailureReason' in response:
                        logger.error(f"Failure reason: {response['FailureReason']}")
                    raise Exception(f"Tracking server creation failed: {status}")
                elif status in ['Creating', 'Updating']:
                    logger.info(f"Still {status.lower()}... waiting 15 seconds")
                    time.sleep(15)
                else:
                    logger.warning(f"Unknown status: {status}")
                    time.sleep(15)
                
                attempt += 1
                
            except Exception as e:
                if "ResourceNotFound" in str(e):
                    logger.error("Tracking server not found. It may have been deleted.")
                    raise
                else:
                    logger.error(f"Error checking tracking server status: {e}")
                    time.sleep(15)
                    attempt += 1
        
        logger.error(f"Timeout waiting for tracking server after {max_wait_minutes} minutes")
        raise Exception("Timeout waiting for tracking server to be ready")
    
    def create_notebook_helper(self, tracking_server_info: Dict[str, Any]) -> str:
        """Create a helper script for notebooks to use the managed MLflow server."""
        tracking_server_url = tracking_server_info.get('TrackingServerUrl', '')
        
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
        self.tracking_server_name = "{self.tracking_server_name}"
        self.tracking_server_url = "{tracking_server_url}"
        
        # Set up AWS session
        os.environ['AWS_PROFILE'] = aws_profile
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker')
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Set up MLflow tracking with SageMaker managed server."""
        if self.tracking_server_url:
            # Use the managed tracking server URL
            mlflow.set_tracking_uri(self.tracking_server_url)
            print(f"Using SageMaker managed MLflow server: {{self.tracking_server_url}}")
        else:
            # Get the tracking server URL dynamically
            try:
                response = self.sagemaker_client.describe_mlflow_tracking_server(
                    TrackingServerName=self.tracking_server_name
                )
                if response['TrackingServerStatus'] == 'Created':
                    self.tracking_server_url = response['TrackingServerUrl']
                    mlflow.set_tracking_uri(self.tracking_server_url)
                    print(f"Connected to SageMaker managed MLflow server: {{self.tracking_server_url}}")
                else:
                    print(f"Warning: Tracking server status is {{response['TrackingServerStatus']}}")
            except Exception as e:
                print(f"Warning: Could not connect to managed MLflow server: {{e}}")
                print("Falling back to local MLflow tracking")
    
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
        helper_key = "mlflow-sagemaker/utils/sagemaker_mlflow_helper.py"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=helper_key,
                Body=helper_script,
                ContentType='text/plain'
            )
            
            helper_uri = f"s3://{self.bucket_name}/{helper_key}"
            logger.info(f"✓ SageMaker MLflow helper script saved to: {helper_uri}")
            return helper_uri
            
        except Exception as e:
            logger.error(f"Failed to save SageMaker MLflow helper script: {e}")
            raise
    
    def create_notebook_examples(self, tracking_server_info: Dict[str, Any]) -> None:
        """Create example code snippets for notebooks."""
        tracking_server_url = tracking_server_info.get('TrackingServerUrl', '')
        
        # Data Scientist example
        ds_example = f'''# SageMaker Managed MLflow Integration for Data Scientists
# Add this to your data-scientist-core-enhanced.ipynb notebook

# Install MLflow if not already installed
# !pip install mlflow boto3

# Import SageMaker MLflow helper
import sys
import boto3

# Download SageMaker MLflow helper from S3
s3_client = boto3.client('s3', region_name='{self.region}')
s3_client.download_file(
    '{self.bucket_name}', 
    'mlflow-sagemaker/utils/sagemaker_mlflow_helper.py', 
    'sagemaker_mlflow_helper.py'
)

# Import the helper
from sagemaker_mlflow_helper import get_sagemaker_mlflow_helper

# Initialize SageMaker MLflow
mlflow_helper = get_sagemaker_mlflow_helper(aws_profile="ab")

# Get tracking server info
server_info = mlflow_helper.get_tracking_server_info()
print(f"MLflow Server: {{server_info['url']}}")
print(f"Status: {{server_info['status']}}")

# Example: Track data exploration
experiment_name = "data-exploration-experiments"
mlflow_helper.create_experiment(experiment_name)

with mlflow_helper.start_run(run_name="data_exploration", experiment_name=experiment_name) as run:
    # Log dataset parameters
    mlflow_helper.log_params({{
        "dataset_size": len(image_files) if 'image_files' in locals() else 1000,
        "image_format": "jpg",
        "data_source": "s3://{self.bucket_name}/datasets/"
    }})
    
    # Log data quality metrics
    mlflow_helper.log_metrics({{
        "avg_image_width": 640,  # Replace with actual values
        "avg_image_height": 480,  # Replace with actual values
        "total_images": 1000  # Replace with actual count
    }})
    
    print(f"Data exploration run: {{run.info.run_id}}")
    print(f"View in MLflow UI: {{mlflow_helper.get_mlflow_ui_url()}}")
'''
        
        # ML Engineer example
        ml_example = f'''# SageMaker Managed MLflow Integration for ML Engineers
# Add this to your ml-engineer-core-enhanced.ipynb notebook

# Install MLflow if not already installed
# !pip install mlflow boto3

# Import SageMaker MLflow helper
import sys
import boto3

# Download SageMaker MLflow helper from S3
s3_client = boto3.client('s3', region_name='{self.region}')
s3_client.download_file(
    '{self.bucket_name}', 
    'mlflow-sagemaker/utils/sagemaker_mlflow_helper.py', 
    'sagemaker_mlflow_helper.py'
)

# Import the helper
from sagemaker_mlflow_helper import get_sagemaker_mlflow_helper

# Initialize SageMaker MLflow
mlflow_helper = get_sagemaker_mlflow_helper(aws_profile="ab")

# Get tracking server info
server_info = mlflow_helper.get_tracking_server_info()
print(f"MLflow Server: {{server_info['url']}}")
print(f"Status: {{server_info['status']}}")

# Example: Track model training
experiment_name = "yolov11-training-experiments"
mlflow_helper.create_experiment(experiment_name)

with mlflow_helper.start_run(run_name="yolov11_training", experiment_name=experiment_name) as run:
    # Log training parameters
    mlflow_helper.log_params({{
        "model_variant": "yolov11n",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.001,
        "image_size": 640,
        "pipeline_name": "sagemaker-core-setup-yolov11-pipeline"
    }})
    
    # Log training metrics (these would come from your training job)
    mlflow_helper.log_metrics({{
        "mAP_0.5": 0.85,
        "mAP_0.5:0.95": 0.72,
        "precision": 0.88,
        "recall": 0.82,
        "training_loss": 0.15
    }})
    
    # Log model artifacts (if available locally)
    # mlflow_helper.log_artifact("model.pt", "models")
    
    print(f"Training run: {{run.info.run_id}}")
    print(f"View in MLflow UI: {{mlflow_helper.get_mlflow_ui_url()}}")

# List recent experiments
experiments = mlflow_helper.list_experiments()
print(f"Available experiments: {{len(experiments)}}")

# List recent runs
recent_runs = mlflow_helper.list_runs(experiment_name, max_results=10)
if recent_runs is not None:
    print("Recent MLflow runs:")
    print(recent_runs[['run_id', 'status', 'start_time']].head())
'''
        
        # Save examples to S3
        examples = {
            "mlflow-sagemaker/examples/data_scientist_example.py": ds_example,
            "mlflow-sagemaker/examples/ml_engineer_example.py": ml_example
        }
        
        for key, content in examples.items():
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=content,
                    ContentType='text/plain'
                )
                logger.info(f"✓ Example saved to: s3://{self.bucket_name}/{key}")
            except Exception as e:
                logger.warning(f"Could not save example {key}: {e}")
    
    def setup_sagemaker_mlflow(self) -> Dict[str, Any]:
        """Set up the complete SageMaker managed MLflow server."""
        logger.info("Setting up SageMaker managed MLflow server...")
        
        # Check prerequisites
        if not self.check_prerequisites():
            raise Exception("Prerequisites not met. Please run core SageMaker setup first.")
        
        # Check if tracking server already exists
        existing_server = self.check_existing_tracking_server()
        
        if existing_server:
            if existing_server['TrackingServerStatus'] == 'Created':
                logger.info("✅ MLflow tracking server already exists and is ready!")
                tracking_server_info = existing_server
            else:
                logger.info(f"Existing server status: {existing_server['TrackingServerStatus']}")
                logger.info("Waiting for existing server to be ready...")
                tracking_server_info = self.wait_for_tracking_server()
        else:
            # Create new tracking server
            tracking_server_info = self.create_mlflow_tracking_server()
            tracking_server_info = self.wait_for_tracking_server()
        
        # Create helper script
        helper_uri = self.create_notebook_helper(tracking_server_info)
        
        # Create notebook examples
        self.create_notebook_examples(tracking_server_info)
        
        # Return setup information
        setup_info = {
            "tracking_server_name": tracking_server_info['TrackingServerName'],
            "tracking_server_url": tracking_server_info.get('TrackingServerUrl', ''),
            "tracking_server_arn": tracking_server_info['TrackingServerArn'],
            "artifact_store_uri": tracking_server_info['ArtifactStoreUri'],
            "mlflow_version": tracking_server_info.get('MlflowVersion', ''),
            "server_size": tracking_server_info.get('TrackingServerSize', ''),
            "status": tracking_server_info['TrackingServerStatus'],
            "helper_uri": helper_uri,
            "bucket_name": self.bucket_name,
            "region": self.region,
            "setup_completed_at": datetime.now().isoformat()
        }
        
        logger.info("✅ SageMaker managed MLflow server setup completed successfully!")
        return setup_info


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup SageMaker Managed MLflow Tracking Server")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", default="lucaskle-ab3-project-pv", help="S3 bucket name")
    parser.add_argument("--wait-timeout", type=int, default=15, help="Max minutes to wait for server creation")
    
    args = parser.parse_args()
    
    try:
        # Initialize setup
        setup = SageMakerMLflowSetup(aws_profile=args.profile, region=args.region)
        
        # Override bucket if specified
        if args.bucket != "lucaskle-ab3-project-pv":
            setup.bucket_name = args.bucket
        
        # Run setup
        setup_info = setup.setup_sagemaker_mlflow()
        
        # Print results
        print("\n" + "="*70)
        print("🎉 SageMaker Managed MLflow Server Setup Complete!")
        print("="*70)
        print(f"Tracking Server Name: {setup_info['tracking_server_name']}")
        print(f"Tracking Server URL: {setup_info['tracking_server_url']}")
        print(f"MLflow Version: {setup_info['mlflow_version']}")
        print(f"Server Size: {setup_info['server_size']}")
        print(f"Status: {setup_info['status']}")
        print(f"Artifact Store: {setup_info['artifact_store_uri']}")
        print(f"Helper Script: {setup_info['helper_uri']}")
        print(f"Region: {setup_info['region']}")
        print(f"Setup Completed: {setup_info['setup_completed_at']}")
        
        print("\n📋 Next Steps:")
        print("1. Open your enhanced notebooks:")
        print("   - notebooks/data-scientist-core-enhanced.ipynb")
        print("   - notebooks/ml-engineer-core-enhanced.ipynb")
        print("\n2. Add SageMaker MLflow integration code from the examples:")
        print(f"   - s3://{setup_info['bucket_name']}/mlflow-sagemaker/examples/")
        print("\n3. Access the MLflow UI:")
        print(f"   - {setup_info['tracking_server_url']}")
        print("\n4. Start tracking your experiments with managed MLflow!")
        
        print("\n💡 Benefits of SageMaker Managed MLflow:")
        print("   ✅ Fully managed infrastructure")
        print("   ✅ Built-in authentication and authorization")
        print("   ✅ Web UI for experiment visualization")
        print("   ✅ Seamless SageMaker Studio integration")
        print("   ✅ Automatic scaling and high availability")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
