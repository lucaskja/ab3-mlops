"""
SageMaker Training Job Orchestration

This module provides comprehensive SageMaker training job management including
job submission, monitoring, distributed training setup, and failure handling.

Requirements addressed:
- 4.1: SageMaker training job configuration and submission
- 4.4: Training job monitoring and logging functionality
- Distributed training setup for multi-GPU scenarios
- Training job failure handling and retry mechanisms
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SageMakerTrainingConfig:
    """Configuration for SageMaker training jobs"""
    
    # Job identification
    job_name: str = ""
    base_job_name: str = "yolov11-training"
    
    # Instance configuration
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    volume_size: int = 30
    max_run: int = 86400  # 24 hours in seconds
    
    # Training configuration
    framework_version: str = "2.0"
    python_version: str = "py310"
    
    # Data configuration
    training_data_s3_path: str = ""
    validation_data_s3_path: str = ""
    output_s3_path: str = ""
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = None
    
    # Environment variables
    environment: Dict[str, str] = None
    
    # Distributed training
    distribution: Dict[str, Any] = None
    
    # Retry configuration
    max_retry_attempts: int = 3
    retry_strategy: str = "exponential_backoff"
    
    # Monitoring
    enable_profiler: bool = True
    enable_debugger: bool = True
    
    # Tags
    tags: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.environment is None:
            self.environment = {}
        if self.tags is None:
            self.tags = []
        if not self.job_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.job_name = f"{self.base_job_name}-{timestamp}"


class SageMakerTrainingOrchestrator:
    """
    Orchestrates SageMaker training jobs with comprehensive monitoring and error handling.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the SageMaker training orchestrator.
        
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
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Get project configuration
        self.project_config = get_config()
        self.execution_role = self.project_config['iam']['roles']['sagemaker_execution']['arn']
        
        logger.info(f"SageMaker orchestrator initialized for region: {region}")
    
    def create_training_estimator(self, config: SageMakerTrainingConfig) -> PyTorch:
        """
        Create a SageMaker PyTorch estimator for YOLOv11 training.
        
        Args:
            config: Training configuration
            
        Returns:
            Configured PyTorch estimator
        """
        logger.info(f"Creating training estimator: {config.job_name}")
        
        # Prepare hyperparameters
        hyperparameters = {
            'epochs': 100,
            'batch-size': 16,
            'learning-rate': 0.01,
            'model': 'yolov11n',
            'img-size': 640,
            **config.hyperparameters
        }
        
        # Prepare environment variables
        environment = {
            'AWS_DEFAULT_REGION': self.region,
            'PYTHONPATH': '/opt/ml/code/src',
            **config.environment
        }
        
        # Configure distribution for multi-GPU training
        distribution_config = None
        if config.instance_count > 1 or config.distribution:
            distribution_config = {
                'pytorchddp': {
                    'enabled': True
                }
            }
            if config.distribution:
                distribution_config.update(config.distribution)
        
        # Create estimator
        estimator = PyTorch(
            entry_point='train_sagemaker.py',
            source_dir='scripts/training',
            role=self.execution_role,
            instance_type=config.instance_type,
            instance_count=config.instance_count,
            volume_size=config.volume_size,
            max_run=config.max_run,
            framework_version=config.framework_version,
            py_version=config.python_version,
            hyperparameters=hyperparameters,
            environment=environment,
            distribution=distribution_config,
            enable_sagemaker_metrics=True,
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
                {'Name': 'validation:loss', 'Regex': 'val_loss: ([0-9\\.]+)'},
                {'Name': 'validation:mAP', 'Regex': 'val_mAP: ([0-9\\.]+)'},
                {'Name': 'validation:precision', 'Regex': 'val_precision: ([0-9\\.]+)'},
                {'Name': 'validation:recall', 'Regex': 'val_recall: ([0-9\\.]+)'}
            ],
            tags=config.tags,
            sagemaker_session=self.sagemaker_session
        )
        
        # Configure profiler if enabled
        if config.enable_profiler:
            from sagemaker.debugger import ProfilerConfig, FrameworkProfile
            estimator.profiler_config = ProfilerConfig(
                framework_profile_params=FrameworkProfile(
                    local_path="/opt/ml/output/profiler/",
                    start_step=5,
                    num_steps=10
                )
            )
        
        # Configure debugger if enabled
        if config.enable_debugger:
            from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
            estimator.debugger_hook_config = DebuggerHookConfig(
                collection_configs=[
                    CollectionConfig(name="weights"),
                    CollectionConfig(name="gradients"),
                    CollectionConfig(name="losses")
                ]
            )
        
        return estimator
    
    def prepare_training_inputs(self, config: SageMakerTrainingConfig) -> Dict[str, TrainingInput]:
        """
        Prepare training input channels for SageMaker.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary of training inputs
        """
        inputs = {}
        
        if config.training_data_s3_path:
            inputs['training'] = TrainingInput(
                s3_data=config.training_data_s3_path,
                content_type='application/x-parquet'  # Adjust based on your data format
            )
        
        if config.validation_data_s3_path:
            inputs['validation'] = TrainingInput(
                s3_data=config.validation_data_s3_path,
                content_type='application/x-parquet'
            )
        
        return inputs
    
    def submit_training_job(self, config: SageMakerTrainingConfig, 
                           wait: bool = False, 
                           logs: bool = True) -> Dict[str, Any]:
        """
        Submit a training job to SageMaker.
        
        Args:
            config: Training configuration
            wait: Whether to wait for job completion
            logs: Whether to stream logs
            
        Returns:
            Job submission results
        """
        logger.info(f"Submitting training job: {config.job_name}")
        
        try:
            # Create estimator
            estimator = self.create_training_estimator(config)
            
            # Prepare inputs
            inputs = self.prepare_training_inputs(config)
            
            # Submit job
            estimator.fit(
                inputs=inputs if inputs else None,
                job_name=config.job_name,
                wait=wait,
                logs=logs
            )
            
            # Get job details
            job_details = self.get_training_job_details(config.job_name)
            
            logger.info(f"Training job submitted successfully: {config.job_name}")
            
            return {
                'job_name': config.job_name,
                'job_arn': job_details.get('TrainingJobArn'),
                'status': job_details.get('TrainingJobStatus'),
                'estimator': estimator,
                'submission_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to submit training job: {str(e)}")
            raise
    
    def get_training_job_details(self, job_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a training job.
        
        Args:
            job_name: Training job name
            
        Returns:
            Job details dictionary
        """
        try:
            response = self.sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            return response
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.warning(f"Training job not found: {job_name}")
                return {}
            else:
                logger.error(f"Error getting job details: {str(e)}")
                raise
    
    def monitor_training_job(self, job_name: str, 
                           callback: Optional[Callable] = None,
                           check_interval: int = 60) -> Dict[str, Any]:
        """
        Monitor a training job until completion.
        
        Args:
            job_name: Training job name
            callback: Optional callback function for status updates
            check_interval: Check interval in seconds
            
        Returns:
            Final job status and details
        """
        logger.info(f"Monitoring training job: {job_name}")
        
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                job_details = self.get_training_job_details(job_name)
                
                if not job_details:
                    logger.error(f"Job not found: {job_name}")
                    break
                
                current_status = job_details.get('TrainingJobStatus')
                
                # Log status changes
                if current_status != last_status:
                    logger.info(f"Job {job_name} status: {current_status}")
                    last_status = current_status
                    
                    # Call callback if provided
                    if callback:
                        callback(job_name, current_status, job_details)
                
                # Check if job is complete
                if current_status in ['Completed', 'Failed', 'Stopped']:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Job {job_name} finished with status: {current_status}")
                    logger.info(f"Total monitoring time: {elapsed_time:.2f} seconds")
                    
                    return {
                        'job_name': job_name,
                        'final_status': current_status,
                        'job_details': job_details,
                        'monitoring_time': elapsed_time
                    }
                
                # Wait before next check
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error monitoring job: {str(e)}")
                time.sleep(check_interval)
        
        return {
            'job_name': job_name,
            'final_status': 'Unknown',
            'monitoring_interrupted': True
        }
    
    def handle_training_failure(self, job_name: str, config: SageMakerTrainingConfig) -> Dict[str, Any]:
        """
        Handle training job failures with retry logic.
        
        Args:
            job_name: Failed job name
            config: Training configuration
            
        Returns:
            Failure handling results
        """
        logger.info(f"Handling training failure for job: {job_name}")
        
        try:
            # Get failure details
            job_details = self.get_training_job_details(job_name)
            failure_reason = job_details.get('FailureReason', 'Unknown failure')
            
            logger.error(f"Training job failed: {failure_reason}")
            
            # Analyze failure reason
            failure_analysis = self._analyze_failure_reason(failure_reason)
            
            # Determine if retry is appropriate
            should_retry = self._should_retry_job(failure_analysis, job_details)
            
            result = {
                'job_name': job_name,
                'failure_reason': failure_reason,
                'failure_analysis': failure_analysis,
                'should_retry': should_retry,
                'retry_attempted': False
            }
            
            if should_retry and config.max_retry_attempts > 0:
                logger.info(f"Attempting to retry failed job: {job_name}")
                
                # Create new job name for retry
                retry_job_name = f"{config.base_job_name}-retry-{int(time.time())}"
                retry_config = config
                retry_config.job_name = retry_job_name
                retry_config.max_retry_attempts -= 1
                
                # Apply failure-specific adjustments
                retry_config = self._adjust_config_for_retry(retry_config, failure_analysis)
                
                # Submit retry job
                retry_result = self.submit_training_job(retry_config)
                
                result.update({
                    'retry_attempted': True,
                    'retry_job_name': retry_job_name,
                    'retry_result': retry_result
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling training failure: {str(e)}")
            raise
    
    def _analyze_failure_reason(self, failure_reason: str) -> Dict[str, Any]:
        """
        Analyze training failure reason to determine appropriate response.
        
        Args:
            failure_reason: Failure reason string
            
        Returns:
            Failure analysis results
        """
        analysis = {
            'category': 'unknown',
            'retryable': False,
            'suggested_actions': []
        }
        
        failure_lower = failure_reason.lower()
        
        # Resource-related failures
        if any(keyword in failure_lower for keyword in ['capacity', 'insufficient', 'resource']):
            analysis.update({
                'category': 'resource',
                'retryable': True,
                'suggested_actions': ['retry_with_delay', 'try_different_instance_type']
            })
        
        # Memory-related failures
        elif any(keyword in failure_lower for keyword in ['memory', 'oom', 'out of memory']):
            analysis.update({
                'category': 'memory',
                'retryable': True,
                'suggested_actions': ['increase_instance_size', 'reduce_batch_size']
            })
        
        # Timeout failures
        elif any(keyword in failure_lower for keyword in ['timeout', 'exceeded']):
            analysis.update({
                'category': 'timeout',
                'retryable': True,
                'suggested_actions': ['increase_max_run_time', 'optimize_training_code']
            })
        
        # Code/configuration errors
        elif any(keyword in failure_lower for keyword in ['error', 'exception', 'failed']):
            analysis.update({
                'category': 'code_error',
                'retryable': False,
                'suggested_actions': ['fix_code', 'check_configuration']
            })
        
        return analysis
    
    def _should_retry_job(self, failure_analysis: Dict[str, Any], job_details: Dict[str, Any]) -> bool:
        """
        Determine if a failed job should be retried.
        
        Args:
            failure_analysis: Failure analysis results
            job_details: Job details from SageMaker
            
        Returns:
            True if job should be retried
        """
        # Don't retry if not retryable
        if not failure_analysis.get('retryable', False):
            return False
        
        # Don't retry if job ran for very short time (likely configuration issue)
        training_start = job_details.get('TrainingStartTime')
        training_end = job_details.get('TrainingEndTime')
        
        if training_start and training_end:
            duration = (training_end - training_start).total_seconds()
            if duration < 300:  # Less than 5 minutes
                logger.info("Job failed too quickly, likely configuration issue - not retrying")
                return False
        
        return True
    
    def _adjust_config_for_retry(self, config: SageMakerTrainingConfig, 
                                failure_analysis: Dict[str, Any]) -> SageMakerTrainingConfig:
        """
        Adjust configuration based on failure analysis for retry.
        
        Args:
            config: Original training configuration
            failure_analysis: Failure analysis results
            
        Returns:
            Adjusted configuration
        """
        adjusted_config = config
        
        failure_category = failure_analysis.get('category')
        suggested_actions = failure_analysis.get('suggested_actions', [])
        
        if 'increase_instance_size' in suggested_actions:
            # Upgrade instance type for memory issues
            instance_upgrades = {
                'ml.g4dn.xlarge': 'ml.g4dn.2xlarge',
                'ml.g4dn.2xlarge': 'ml.g4dn.4xlarge',
                'ml.g4dn.4xlarge': 'ml.g4dn.8xlarge'
            }
            
            current_instance = config.instance_type
            if current_instance in instance_upgrades:
                adjusted_config.instance_type = instance_upgrades[current_instance]
                logger.info(f"Upgraded instance type: {current_instance} -> {adjusted_config.instance_type}")
        
        if 'reduce_batch_size' in suggested_actions:
            # Reduce batch size in hyperparameters
            current_batch_size = config.hyperparameters.get('batch-size', 16)
            new_batch_size = max(1, current_batch_size // 2)
            adjusted_config.hyperparameters['batch-size'] = new_batch_size
            logger.info(f"Reduced batch size: {current_batch_size} -> {new_batch_size}")
        
        if 'increase_max_run_time' in suggested_actions:
            # Increase max run time for timeout issues
            adjusted_config.max_run = min(604800, config.max_run * 2)  # Max 7 days
            logger.info(f"Increased max run time: {config.max_run} -> {adjusted_config.max_run}")
        
        return adjusted_config
    
    def list_training_jobs(self, status_filter: Optional[str] = None, 
                          max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List training jobs with optional status filtering.
        
        Args:
            status_filter: Optional status filter (InProgress, Completed, Failed, etc.)
            max_results: Maximum number of results to return
            
        Returns:
            List of training job summaries
        """
        try:
            kwargs = {'MaxResults': max_results}
            if status_filter:
                kwargs['StatusEquals'] = status_filter
            
            response = self.sagemaker_client.list_training_jobs(**kwargs)
            return response.get('TrainingJobSummaries', [])
            
        except Exception as e:
            logger.error(f"Error listing training jobs: {str(e)}")
            raise
    
    def stop_training_job(self, job_name: str) -> Dict[str, Any]:
        """
        Stop a running training job.
        
        Args:
            job_name: Training job name
            
        Returns:
            Stop operation results
        """
        logger.info(f"Stopping training job: {job_name}")
        
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            
            # Wait for job to stop
            time.sleep(10)
            job_details = self.get_training_job_details(job_name)
            
            return {
                'job_name': job_name,
                'status': job_details.get('TrainingJobStatus'),
                'stopped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error stopping training job: {str(e)}")
            raise
    
    def get_training_metrics(self, job_name: str) -> Dict[str, List[Dict]]:
        """
        Get training metrics for a job.
        
        Args:
            job_name: Training job name
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Get metrics from CloudWatch
            cloudwatch = self.session.client('cloudwatch', region_name=self.region)
            
            metrics = {}
            metric_names = ['train:loss', 'validation:loss', 'validation:mAP', 
                          'validation:precision', 'validation:recall']
            
            for metric_name in metric_names:
                response = cloudwatch.get_metric_statistics(
                    Namespace='/aws/sagemaker/TrainingJobs',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'TrainingJobName', 'Value': job_name}
                    ],
                    StartTime=datetime.now() - timedelta(days=7),
                    EndTime=datetime.now(),
                    Period=300,
                    Statistics=['Average']
                )
                
                metrics[metric_name] = response.get('Datapoints', [])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting training metrics: {str(e)}")
            return {}


def create_sagemaker_training_config(**kwargs) -> SageMakerTrainingConfig:
    """
    Create a SageMaker training configuration with project defaults.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        SageMaker training configuration
    """
    project_config = get_config()
    
    defaults = {
        'instance_type': project_config['training']['instance_type'],
        'instance_count': project_config['training']['instance_count'],
        'volume_size': project_config['training']['volume_size'],
        'tags': [
            {'Key': k, 'Value': v} 
            for k, v in project_config['cost']['tags'].items()
        ]
    }
    
    # Merge defaults with provided kwargs
    config_dict = {**defaults, **kwargs}
    
    return SageMakerTrainingConfig(**config_dict)


if __name__ == "__main__":
    # Example usage
    orchestrator = SageMakerTrainingOrchestrator()
    
    config = create_sagemaker_training_config(
        base_job_name="yolov11-plant-disease",
        training_data_s3_path="s3://lucaskle-ab3-project-pv/data/plant-disease-detection/train/",
        validation_data_s3_path="s3://lucaskle-ab3-project-pv/data/plant-disease-detection/val/",
        hyperparameters={
            'epochs': 50,
            'batch-size': 16,
            'model': 'yolov11n'
        }
    )
    
    print("SageMaker Training Configuration:")
    print(json.dumps(asdict(config), indent=2, default=str))