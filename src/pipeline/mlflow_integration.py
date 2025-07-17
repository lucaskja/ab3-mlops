"""
MLFlow Integration with SageMaker

This module provides MLFlow integration with SageMaker for experiment tracking,
parameter logging, and model artifact management.

Requirements addressed:
- 7.1: MLFlow server configuration for SageMaker hosting
- 7.2: MLFlow experiment initialization and management
- 7.3: Automatic parameter and metric logging for training jobs
- Model artifact storage and retrieval
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException
import sagemaker
from sagemaker.mlflow import MlflowModelServer
from sagemaker.session import Session as SageMakerSession

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFlowSageMakerIntegration:
    """
    Integrates MLFlow with SageMaker for experiment tracking and model management.
    Uses SageMaker's built-in MLFlow tracking capabilities rather than a standalone server.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the MLFlow SageMaker integration.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker', region_name=region)
        self.s3_client = self.session.client('s3', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = SageMakerSession(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Get project configuration
        self.project_config = get_config()
        self.execution_role = self.project_config['iam']['roles']['sagemaker_execution']['arn']
        
        # Set up MLFlow tracking URI using SageMaker's default artifact store
        self.artifact_bucket = self.project_config.get('mlflow', {}).get(
            'artifact_bucket', 
            self.project_config['s3']['default_bucket']
        )
        self.artifact_prefix = self.project_config.get('mlflow', {}).get(
            'artifact_prefix', 
            'mlflow-artifacts'
        )
        
        # Configure MLFlow to use SageMaker's artifact store
        self._configure_mlflow()
        
        logger.info(f"MLFlow SageMaker integration initialized for region: {region}")
    
    def _configure_mlflow(self):
        """
        Configure MLFlow to use SageMaker's artifact store.
        
        This method checks for a SageMaker-hosted MLFlow tracking server first,
        then falls back to local tracking if not available.
        """
        # Get tracking URI from project config or environment variable
        tracking_uri = self.project_config.get('mlflow', {}).get('tracking_uri')
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", tracking_uri)
        
        if tracking_uri and "sagemaker" in tracking_uri:
            # Use SageMaker-hosted MLFlow tracking server
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Using SageMaker-hosted MLFlow tracking server: {tracking_uri}")
        else:
            # Fall back to local tracking
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            
            # Set tracking URI to local directory
            local_uri = f"file://{mlflow_dir.absolute()}"
            mlflow.set_tracking_uri(local_uri)
            logger.info(f"Using local MLFlow tracking: {local_uri}")
            logger.info("To use SageMaker-hosted MLFlow, run setup_mlflow_sagemaker.py")
        
        # Set default artifact location to S3
        artifact_location = f"s3://{self.artifact_bucket}/{self.artifact_prefix}"
        logger.info(f"MLFlow artifact location: {artifact_location}")
        
        # Set environment variables for boto3 to use the correct profile
        os.environ["AWS_PROFILE"] = self.aws_profile
    
    def create_experiment(self, experiment_name: str, artifact_location: Optional[str] = None) -> str:
        """
        Create or get an MLFlow experiment.
        
        Args:
            experiment_name: Name of the experiment
            artifact_location: Optional custom artifact location
            
        Returns:
            Experiment ID
        """
        try:
            # Set default artifact location if not provided
            if not artifact_location:
                artifact_location = f"s3://{self.artifact_bucket}/{self.artifact_prefix}/{experiment_name}"
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
                return experiment.experiment_id
            else:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
                return experiment_id
                
        except MlflowException as e:
            logger.error(f"Error creating MLFlow experiment: {str(e)}")
            raise
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None, 
                 nested: bool = False) -> mlflow.ActiveRun:
        """
        Start a new MLFlow run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            MLFlow active run object
        """
        try:
            # Get or create experiment
            experiment_id = self.create_experiment(experiment_name)
            
            # Start run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested
            )
            
            logger.info(f"Started MLFlow run: {run.info.run_id} (name: {run_name})")
            return run
            
        except MlflowException as e:
            logger.error(f"Error starting MLFlow run: {str(e)}")
            raise
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """
        Log parameters to the current MLFlow run.
        
        Args:
            parameters: Dictionary of parameters to log
        """
        try:
            # Convert all values to strings for MLFlow compatibility
            string_params = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                           for k, v in parameters.items()}
            
            mlflow.log_params(string_params)
            logger.info(f"Logged {len(parameters)} parameters to MLFlow")
            
        except MlflowException as e:
            logger.error(f"Error logging parameters to MLFlow: {str(e)}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current MLFlow run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged {len(metrics)} metrics to MLFlow" + 
                      (f" at step {step}" if step is not None else ""))
            
        except MlflowException as e:
            logger.error(f"Error logging metrics to MLFlow: {str(e)}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to the current MLFlow run.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Optional path within the artifact directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact from {local_path}" + 
                      (f" to {artifact_path}" if artifact_path else ""))
            
        except MlflowException as e:
            logger.error(f"Error logging artifact to MLFlow: {str(e)}")
            raise
    
    def log_model(self, model_path: str, flavor: str = "pytorch", 
                registered_model_name: Optional[str] = None):
        """
        Log a model to the current MLFlow run.
        
        Args:
            model_path: Path to the model file or directory
            flavor: Model flavor (pytorch, sklearn, etc.)
            registered_model_name: Optional name to register the model with
        """
        try:
            if flavor.lower() == "pytorch":
                import torch
                # Check if model_path is a PyTorch model or a path to a file
                if isinstance(model_path, str) and os.path.exists(model_path):
                    model = torch.load(model_path)
                else:
                    model = model_path
                
                mlflow.pytorch.log_model(
                    model, 
                    "model",
                    registered_model_name=registered_model_name
                )
            else:
                # Generic artifact logging for other model types
                mlflow.log_artifact(model_path, "model")
                
                # Register model if name provided
                if registered_model_name:
                    client = MlflowClient()
                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    client.create_model_version(
                        name=registered_model_name,
                        source=model_uri,
                        run_id=run_id
                    )
            
            logger.info(f"Logged model from {model_path}" + 
                      (f" and registered as {registered_model_name}" if registered_model_name else ""))
            
        except Exception as e:
            logger.error(f"Error logging model to MLFlow: {str(e)}")
            raise
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLFlow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            # Map string status to MLFlow RunStatus
            status_map = {
                "FINISHED": RunStatus.FINISHED,
                "FAILED": RunStatus.FAILED,
                "KILLED": RunStatus.KILLED
            }
            mlflow_status = status_map.get(status.upper(), RunStatus.FINISHED)
            
            mlflow.end_run(status=mlflow_status)
            logger.info(f"Ended MLFlow run with status: {status}")
            
        except MlflowException as e:
            logger.error(f"Error ending MLFlow run: {str(e)}")
            raise
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get details of an MLFlow run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Run details
        """
        try:
            client = MlflowClient()
            run = client.get_run(run_id)
            
            return {
                "info": {k: v for k, v in run.info.__dict__.items() if not k.startswith("_")},
                "data": {
                    "params": run.data.params,
                    "metrics": run.data.metrics
                }
            }
            
        except MlflowException as e:
            logger.error(f"Error getting MLFlow run: {str(e)}")
            raise
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all MLFlow experiments.
        
        Returns:
            List of experiment details
        """
        try:
            client = MlflowClient()
            experiments = client.list_experiments()
            
            return [{
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            } for exp in experiments]
            
        except MlflowException as e:
            logger.error(f"Error listing MLFlow experiments: {str(e)}")
            raise
    
    def list_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """
        List all runs for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            List of run details
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if not experiment:
                logger.warning(f"Experiment not found: {experiment_name}")
                return []
            
            client = MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"]
            )
            
            return [{
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "params": run.data.params
            } for run in runs]
            
        except MlflowException as e:
            logger.error(f"Error listing MLFlow runs: {str(e)}")
            raise
    
    def log_sagemaker_job(self, job_name: str, experiment_name: str, 
                        run_name: Optional[str] = None) -> str:
        """
        Log a SageMaker training job to MLFlow.
        
        Args:
            job_name: SageMaker training job name
            experiment_name: MLFlow experiment name
            run_name: Optional MLFlow run name
            
        Returns:
            MLFlow run ID
        """
        try:
            # Get job details
            response = self.sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            
            # Start MLFlow run
            if not run_name:
                run_name = job_name
                
            with self.start_run(experiment_name=experiment_name, run_name=run_name) as run:
                run_id = run.info.run_id
                
                # Log job parameters
                params = {
                    "job_name": job_name,
                    "instance_type": response.get("ResourceConfig", {}).get("InstanceType"),
                    "instance_count": response.get("ResourceConfig", {}).get("InstanceCount"),
                    "volume_size": response.get("ResourceConfig", {}).get("VolumeSizeInGB"),
                    "max_run_time": response.get("StoppingCondition", {}).get("MaxRuntimeInSeconds"),
                    "training_image": response.get("AlgorithmSpecification", {}).get("TrainingImage"),
                    "input_mode": response.get("AlgorithmSpecification", {}).get("TrainingInputMode")
                }
                
                # Add hyperparameters
                hyperparameters = response.get("HyperParameters", {})
                for key, value in hyperparameters.items():
                    params[f"hp_{key}"] = value
                
                self.log_parameters(params)
                
                # Log metrics if available
                metrics = self._get_job_metrics(job_name)
                if metrics:
                    for step, step_metrics in enumerate(metrics):
                        self.log_metrics(step_metrics, step=step)
                
                # Log model artifacts
                model_artifacts = response.get("ModelArtifacts", {}).get("S3ModelArtifacts")
                if model_artifacts:
                    self.log_parameters({"model_artifacts": model_artifacts})
                
                logger.info(f"Logged SageMaker job {job_name} to MLFlow run {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Error logging SageMaker job to MLFlow: {str(e)}")
            raise
    
    def _get_job_metrics(self, job_name: str) -> List[Dict[str, float]]:
        """
        Get metrics from a SageMaker training job.
        
        Args:
            job_name: SageMaker training job name
            
        Returns:
            List of metric dictionaries by step
        """
        try:
            # Get CloudWatch client
            cloudwatch = self.session.client('cloudwatch', region_name=self.region)
            
            # Define metrics to retrieve
            metric_names = [
                'train:loss', 
                'validation:loss', 
                'validation:mAP',
                'validation:precision', 
                'validation:recall'
            ]
            
            # Get metrics data
            all_metrics = []
            
            for metric_name in metric_names:
                response = cloudwatch.get_metric_data(
                    MetricDataQueries=[
                        {
                            'Id': metric_name.replace(':', '_'),
                            'MetricStat': {
                                'Metric': {
                                    'Namespace': '/aws/sagemaker/TrainingJobs',
                                    'MetricName': metric_name,
                                    'Dimensions': [
                                        {
                                            'Name': 'TrainingJobName',
                                            'Value': job_name
                                        }
                                    ]
                                },
                                'Period': 60,
                                'Stat': 'Average'
                            }
                        }
                    ],
                    StartTime=cloudwatch.meta.client.meta.region_name,
                    EndTime=cloudwatch.meta.client.meta.region_name
                )
                
                # Process and organize metrics by timestamp
                timestamps = response['MetricDataResults'][0]['Timestamps']
                values = response['MetricDataResults'][0]['Values']
                
                for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                    if i >= len(all_metrics):
                        all_metrics.append({})
                    
                    # Clean up metric name for MLFlow (remove namespace)
                    clean_name = metric_name.split(':')[-1]
                    all_metrics[i][clean_name] = value
            
            return all_metrics
            
        except Exception as e:
            logger.warning(f"Error getting job metrics: {str(e)}")
            return []
    
    def register_model(self, run_id: str, model_name: str, 
                     description: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a model from an MLFlow run.
        
        Args:
            run_id: MLFlow run ID
            model_name: Name to register the model with
            description: Optional model description
            
        Returns:
            Model version details
        """
        try:
            client = MlflowClient()
            
            # Get run details
            run = client.get_run(run_id)
            
            # Create model if it doesn't exist
            try:
                client.get_registered_model(model_name)
            except MlflowException:
                client.create_registered_model(model_name, description)
            
            # Register model version
            model_uri = f"runs:/{run_id}/model"
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=description
            )
            
            logger.info(f"Registered model {model_name} version {model_version.version} from run {run_id}")
            
            return {
                "name": model_version.name,
                "version": model_version.version,
                "status": model_version.status,
                "run_id": model_version.run_id
            }
            
        except MlflowException as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def deploy_model_to_sagemaker(self, model_name: str, version: str, 
                               instance_type: str = "ml.m5.large",
                               instance_count: int = 1) -> Dict[str, Any]:
        """
        Deploy an MLFlow model to SageMaker.
        
        Args:
            model_name: Registered model name
            version: Model version
            instance_type: SageMaker instance type
            instance_count: Number of instances
            
        Returns:
            Deployment details
        """
        try:
            # Get model URI
            model_uri = f"models:/{model_name}/{version}"
            
            # Create SageMaker model
            mlflow_model_server = MlflowModelServer(
                model_uri=model_uri,
                role=self.execution_role,
                instance_type=instance_type,
                instance_count=instance_count,
                sagemaker_session=self.sagemaker_session
            )
            
            # Deploy model
            endpoint_name = f"{model_name}-{version}".replace(".", "-").lower()
            mlflow_model_server.deploy(endpoint_name=endpoint_name)
            
            logger.info(f"Deployed model {model_name} version {version} to SageMaker endpoint {endpoint_name}")
            
            return {
                "model_name": model_name,
                "model_version": version,
                "endpoint_name": endpoint_name,
                "instance_type": instance_type,
                "instance_count": instance_count
            }
            
        except Exception as e:
            logger.error(f"Error deploying model to SageMaker: {str(e)}")
            raise
    
    def delete_endpoint(self, endpoint_name: str):
        """
        Delete a SageMaker endpoint.
        
        Args:
            endpoint_name: SageMaker endpoint name
        """
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Deleted SageMaker endpoint: {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Error deleting SageMaker endpoint: {str(e)}")
            raise


# Helper functions for common MLFlow operations
def get_mlflow_integration(aws_profile: str = "ab", region: str = "us-east-1") -> MLFlowSageMakerIntegration:
    """
    Get an MLFlow SageMaker integration instance.
    
    Args:
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        MLFlow SageMaker integration instance
    """
    return MLFlowSageMakerIntegration(aws_profile=aws_profile, region=region)


def log_training_job(job_name: str, experiment_name: str, 
                   run_name: Optional[str] = None, 
                   aws_profile: str = "ab") -> str:
    """
    Log a SageMaker training job to MLFlow.
    
    Args:
        job_name: SageMaker training job name
        experiment_name: MLFlow experiment name
        run_name: Optional MLFlow run name
        aws_profile: AWS profile to use
        
    Returns:
        MLFlow run ID
    """
    mlflow_integration = get_mlflow_integration(aws_profile=aws_profile)
    return mlflow_integration.log_sagemaker_job(job_name, experiment_name, run_name)


if __name__ == "__main__":
    # Example usage
    mlflow_integration = MLFlowSageMakerIntegration()
    
    # Create experiment
    experiment_name = "yolov11-drone-detection"
    mlflow_integration.create_experiment(experiment_name)
    
    # Start run and log parameters
    with mlflow_integration.start_run(experiment_name, run_name="example-run") as run:
        # Log parameters
        mlflow_integration.log_parameters({
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 100,
            "model_variant": "yolov11n"
        })
        
        # Log metrics
        mlflow_integration.log_metrics({
            "train_loss": 0.25,
            "val_loss": 0.3,
            "val_mAP": 0.85
        })
        
        print(f"MLFlow run ID: {run.info.run_id}")