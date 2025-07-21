#!/usr/bin/env python3
"""
Setup MLFlow Server for Core SageMaker Setup

This script creates an MLFlow tracking server that can be used by the enhanced notebooks
for experiment tracking and model management.
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


class MLFlowServerSetup:
    """Sets up MLFlow tracking server for the core SageMaker setup."""
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the MLFlow server setup.
        
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
        self.mlflow_prefix = "mlflow"
        
        # Get account info
        sts_client = self.session.client('sts')
        identity = sts_client.get_caller_identity()
        self.account_id = identity['Account']
        
        logger.info(f"Initialized MLFlow setup with profile: {aws_profile}")
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
    
    def create_mlflow_s3_structure(self) -> None:
        """Create S3 directory structure for MLFlow."""
        logger.info("Creating MLFlow S3 structure...")
        
        # Create MLFlow directories in S3
        directories = [
            f"{self.mlflow_prefix}/",
            f"{self.mlflow_prefix}/artifacts/",
            f"{self.mlflow_prefix}/experiments/",
            f"{self.mlflow_prefix}/models/",
            f"{self.mlflow_prefix}/runs/"
        ]
        
        for directory in directories:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=directory,
                    Body=b''
                )
                logger.info(f"✓ Created S3 directory: s3://{self.bucket_name}/{directory}")
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
    
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
    
    def create_mlflow_experiment_config(self) -> Dict[str, Any]:
        """Create MLFlow experiment configuration."""
        config = {
            "mlflow_config": {
                "tracking_uri": f"s3://{self.bucket_name}/{self.mlflow_prefix}",
                "artifact_location": f"s3://{self.bucket_name}/{self.mlflow_prefix}/artifacts",
                "experiment_name": f"{self.project_name}-experiments",
                "backend_store_uri": f"s3://{self.bucket_name}/{self.mlflow_prefix}/experiments",
                "default_artifact_root": f"s3://{self.bucket_name}/{self.mlflow_prefix}/artifacts"
            },
            "aws_config": {
                "profile": self.aws_profile,
                "region": self.region,
                "bucket": self.bucket_name,
                "execution_role": self.get_execution_role_arn()
            },
            "project_config": {
                "project_name": self.project_name,
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
        
        return config
    
    def save_mlflow_config(self, config: Dict[str, Any]) -> str:
        """Save MLFlow configuration to S3."""
        config_key = f"{self.mlflow_prefix}/config/mlflow_config.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=config_key,
                Body=json.dumps(config, indent=2),
                ContentType='application/json'
            )
            
            config_uri = f"s3://{self.bucket_name}/{config_key}"
            logger.info(f"✓ MLFlow configuration saved to: {config_uri}")
            return config_uri
            
        except Exception as e:
            logger.error(f"Failed to save MLFlow configuration: {e}")
            raise
    
    def create_mlflow_notebook_helper(self) -> str:
        """Create a helper script for notebooks to use MLFlow."""
        helper_script = '''"""
MLFlow Helper for Core SageMaker Setup

This module provides utilities for using MLFlow in SageMaker notebooks.
"""

import os
import json
import boto3
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, Any, Optional


class MLFlowHelper:
    """Helper class for MLFlow operations in SageMaker notebooks."""
    
    def __init__(self, aws_profile: str = "ab", bucket_name: str = "lucaskle-ab3-project-pv"):
        """Initialize MLFlow helper."""
        self.aws_profile = aws_profile
        self.bucket_name = bucket_name
        self.mlflow_prefix = "mlflow"
        
        # Set up AWS session
        os.environ['AWS_PROFILE'] = aws_profile
        self.session = boto3.Session(profile_name=aws_profile)
        self.s3_client = self.session.client('s3')
        
        # Load MLFlow configuration
        self.config = self._load_config()
        self._setup_mlflow()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MLFlow configuration from S3."""
        try:
            config_key = f"{self.mlflow_prefix}/config/mlflow_config.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=config_key)
            config = json.loads(response['Body'].read().decode('utf-8'))
            return config
        except Exception as e:
            print(f"Warning: Could not load MLFlow config: {e}")
            # Return default configuration
            return {
                "mlflow_config": {
                    "tracking_uri": f"s3://{self.bucket_name}/{self.mlflow_prefix}",
                    "artifact_location": f"s3://{self.bucket_name}/{self.mlflow_prefix}/artifacts",
                    "experiment_name": "sagemaker-core-setup-experiments"
                }
            }
    
    def _setup_mlflow(self):
        """Set up MLFlow tracking."""
        mlflow_config = self.config['mlflow_config']
        
        # Set MLFlow tracking URI
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        
        # Set or create experiment
        experiment_name = mlflow_config['experiment_name']
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=mlflow_config['artifact_location']
                )
                print(f"Created MLFlow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                print(f"Using existing MLFlow experiment: {experiment_name}")
                mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set up MLFlow experiment: {e}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start an MLFlow run."""
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLFlow."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLFlow."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a model to MLFlow."""
        if hasattr(model, 'state_dict'):  # PyTorch model
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        else:  # Sklearn or other models
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    
    def end_run(self):
        """End the current MLFlow run."""
        mlflow.end_run()
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about the current experiment."""
        experiment = mlflow.get_experiment_by_name(self.config['mlflow_config']['experiment_name'])
        if experiment:
            return {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage
            }
        return {}
    
    def list_runs(self, max_results: int = 10):
        """List recent runs in the experiment."""
        experiment = mlflow.get_experiment_by_name(self.config['mlflow_config']['experiment_name'])
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            return runs
        return None


# Convenience function for notebooks
def get_mlflow_helper(aws_profile: str = "ab") -> MLFlowHelper:
    """Get an MLFlow helper instance."""
    return MLFlowHelper(aws_profile=aws_profile)


# Example usage for notebooks
def example_usage():
    """Example of how to use MLFlow in notebooks."""
    # Initialize MLFlow helper
    mlflow_helper = get_mlflow_helper()
    
    # Start a run
    with mlflow_helper.start_run(run_name="example_run") as run:
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
        
        # Log artifacts (files)
        # mlflow_helper.log_artifact("path/to/file.txt")
        
        print(f"Run ID: {run.info.run_id}")
    
    # Get experiment info
    exp_info = mlflow_helper.get_experiment_info()
    print(f"Experiment: {exp_info}")
    
    # List recent runs
    runs = mlflow_helper.list_runs(max_results=5)
    if runs is not None:
        print(f"Recent runs: {len(runs)} found")
'''
        
        # Save helper script to S3
        helper_key = f"{self.mlflow_prefix}/utils/mlflow_helper.py"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=helper_key,
                Body=helper_script,
                ContentType='text/plain'
            )
            
            helper_uri = f"s3://{self.bucket_name}/{helper_key}"
            logger.info(f"✓ MLFlow helper script saved to: {helper_uri}")
            return helper_uri
            
        except Exception as e:
            logger.error(f"Failed to save MLFlow helper script: {e}")
            raise
    
    def create_notebook_examples(self) -> None:
        """Create example code snippets for notebooks."""
        
        # Data Scientist example
        ds_example = '''# MLFlow Integration for Data Scientists
# Add this to your data-scientist-core-enhanced.ipynb notebook

# Install MLFlow if not already installed
# !pip install mlflow boto3

# Import MLFlow helper
import sys
import boto3

# Download MLFlow helper from S3
s3_client = boto3.client('s3', region_name='us-east-1')
s3_client.download_file(
    'lucaskle-ab3-project-pv', 
    'mlflow/utils/mlflow_helper.py', 
    'mlflow_helper.py'
)

# Import the helper
from mlflow_helper import get_mlflow_helper

# Initialize MLFlow
mlflow_helper = get_mlflow_helper(aws_profile="ab")

# Example: Track data exploration
with mlflow_helper.start_run(run_name="data_exploration") as run:
    # Log dataset parameters
    mlflow_helper.log_params({
        "dataset_size": len(image_files),
        "image_format": "jpg",
        "data_source": "s3://lucaskle-ab3-project-pv/datasets/"
    })
    
    # Log data quality metrics
    mlflow_helper.log_metrics({
        "avg_image_width": avg_width,
        "avg_image_height": avg_height,
        "total_images": total_images
    })
    
    print(f"Data exploration run: {run.info.run_id}")
'''
        
        # ML Engineer example
        ml_example = '''# MLFlow Integration for ML Engineers
# Add this to your ml-engineer-core-enhanced.ipynb notebook

# Install MLFlow if not already installed
# !pip install mlflow boto3

# Import MLFlow helper
import sys
import boto3

# Download MLFlow helper from S3
s3_client = boto3.client('s3', region_name='us-east-1')
s3_client.download_file(
    'lucaskle-ab3-project-pv', 
    'mlflow/utils/mlflow_helper.py', 
    'mlflow_helper.py'
)

# Import the helper
from mlflow_helper import get_mlflow_helper

# Initialize MLFlow
mlflow_helper = get_mlflow_helper(aws_profile="ab")

# Example: Track model training
with mlflow_helper.start_run(run_name="yolov11_training") as run:
    # Log training parameters
    mlflow_helper.log_params({
        "model_variant": "yolov11n",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.001,
        "image_size": 640
    })
    
    # Log training metrics (these would come from your training job)
    mlflow_helper.log_metrics({
        "mAP_0.5": 0.85,
        "mAP_0.5:0.95": 0.72,
        "precision": 0.88,
        "recall": 0.82,
        "training_loss": 0.15
    })
    
    # Log model artifacts (if available locally)
    # mlflow_helper.log_artifact("model.pt", "models")
    
    print(f"Training run: {run.info.run_id}")

# List recent experiments
recent_runs = mlflow_helper.list_runs(max_results=10)
if recent_runs is not None:
    print("Recent MLFlow runs:")
    print(recent_runs[['run_id', 'status', 'start_time', 'tags.mlflow.runName']].head())
'''
        
        # Save examples to S3
        examples = {
            f"{self.mlflow_prefix}/examples/data_scientist_example.py": ds_example,
            f"{self.mlflow_prefix}/examples/ml_engineer_example.py": ml_example
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
    
    def setup_mlflow_server(self) -> Dict[str, Any]:
        """Set up the complete MLFlow server configuration."""
        logger.info("Setting up MLFlow server...")
        
        # Check prerequisites
        if not self.check_prerequisites():
            raise Exception("Prerequisites not met. Please run core SageMaker setup first.")
        
        # Create S3 structure
        self.create_mlflow_s3_structure()
        
        # Create and save configuration
        config = self.create_mlflow_experiment_config()
        config_uri = self.save_mlflow_config(config)
        
        # Create helper script
        helper_uri = self.create_mlflow_notebook_helper()
        
        # Create notebook examples
        self.create_notebook_examples()
        
        # Return setup information
        setup_info = {
            "mlflow_tracking_uri": config['mlflow_config']['tracking_uri'],
            "artifact_location": config['mlflow_config']['artifact_location'],
            "experiment_name": config['mlflow_config']['experiment_name'],
            "config_uri": config_uri,
            "helper_uri": helper_uri,
            "bucket_name": self.bucket_name,
            "region": self.region,
            "setup_completed_at": datetime.now().isoformat()
        }
        
        logger.info("✅ MLFlow server setup completed successfully!")
        return setup_info


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup MLFlow Server for Core SageMaker Setup")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", default="lucaskle-ab3-project-pv", help="S3 bucket name")
    
    args = parser.parse_args()
    
    try:
        # Initialize setup
        setup = MLFlowServerSetup(aws_profile=args.profile, region=args.region)
        
        # Override bucket if specified
        if args.bucket != "lucaskle-ab3-project-pv":
            setup.bucket_name = args.bucket
        
        # Run setup
        setup_info = setup.setup_mlflow_server()
        
        # Print results
        print("\n" + "="*60)
        print("🎉 MLFlow Server Setup Complete!")
        print("="*60)
        print(f"MLFlow Tracking URI: {setup_info['mlflow_tracking_uri']}")
        print(f"Artifact Location: {setup_info['artifact_location']}")
        print(f"Experiment Name: {setup_info['experiment_name']}")
        print(f"Configuration: {setup_info['config_uri']}")
        print(f"Helper Script: {setup_info['helper_uri']}")
        print(f"Region: {setup_info['region']}")
        print(f"Setup Completed: {setup_info['setup_completed_at']}")
        
        print("\n📋 Next Steps:")
        print("1. Open your enhanced notebooks:")
        print("   - notebooks/data-scientist-core-enhanced.ipynb")
        print("   - notebooks/ml-engineer-core-enhanced.ipynb")
        print("\n2. Add MLFlow integration code from the examples:")
        print(f"   - s3://{setup_info['bucket_name']}/mlflow/examples/")
        print("\n3. Start tracking your experiments!")
        print("\n💡 MLFlow UI Access:")
        print("   Since this is an S3-based setup, you can view experiments by:")
        print("   - Using the MLFlow helper in notebooks")
        print("   - Accessing artifacts directly in S3")
        print("   - Setting up a local MLFlow UI pointing to the S3 backend")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
