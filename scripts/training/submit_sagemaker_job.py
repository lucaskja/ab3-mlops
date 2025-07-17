#!/usr/bin/env python3
"""
SageMaker Training Job Submission Script

This script provides a command-line interface for submitting and managing
SageMaker training jobs with comprehensive monitoring and error handling.

Usage:
    python scripts/training/submit_sagemaker_job.py --job-name my-training-job --epochs 50
    python scripts/training/submit_sagemaker_job.py --monitor-job existing-job-name
    python scripts/training/submit_sagemaker_job.py --list-jobs --status InProgress
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from pipeline.sagemaker_training import SageMakerTrainingOrchestrator, create_sagemaker_training_config
from pipeline.mlflow_integration import MLFlowSageMakerIntegration, log_training_job
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Submit and manage SageMaker training jobs')
    
    # Job identification
    parser.add_argument('--job-name', type=str, help='Training job name')
    parser.add_argument('--base-job-name', type=str, default='yolov11-training',
                       help='Base job name for auto-generated names')
    
    # Instance configuration
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                       help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=30,
                       help='EBS volume size in GB')
    parser.add_argument('--max-run', type=int, default=86400,
                       help='Maximum runtime in seconds')
    
    # Data configuration
    parser.add_argument('--training-data', type=str,
                       help='S3 path to training data')
    parser.add_argument('--validation-data', type=str,
                       help='S3 path to validation data')
    parser.add_argument('--output-path', type=str,
                       help='S3 path for output artifacts')
    
    # Training hyperparameters
    parser.add_argument('--model', type=str, default='yolov11n',
                       choices=['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
                       help='YOLOv11 model variant')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--num-classes', type=int, default=39,
                       help='Number of classes')
    parser.add_argument('--class-names', type=str,
                       help='Comma-separated list of class names')
    
    # AWS configuration
    parser.add_argument('--aws-profile', type=str, default='ab',
                       help='AWS profile to use')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    
    # Job management actions
    parser.add_argument('--submit', action='store_true',
                       help='Submit a new training job')
    parser.add_argument('--monitor-job', type=str,
                       help='Monitor an existing job')
    parser.add_argument('--stop-job', type=str,
                       help='Stop a running job')
    parser.add_argument('--list-jobs', action='store_true',
                       help='List training jobs')
    parser.add_argument('--job-details', type=str,
                       help='Get details for a specific job')
    parser.add_argument('--job-metrics', type=str,
                       help='Get metrics for a specific job')
    
    # Filtering and options
    parser.add_argument('--status', type=str,
                       choices=['InProgress', 'Completed', 'Failed', 'Stopping', 'Stopped'],
                       help='Filter jobs by status')
    parser.add_argument('--max-results', type=int, default=20,
                       help='Maximum number of results to return')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for job completion')
    parser.add_argument('--no-logs', action='store_true',
                       help='Disable log streaming')
    
    # Retry configuration
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts for failed jobs')
    parser.add_argument('--enable-profiler', action='store_true',
                       help='Enable SageMaker profiler')
    parser.add_argument('--enable-debugger', action='store_true',
                       help='Enable SageMaker debugger')
    
    # Configuration file
    parser.add_argument('--config-file', type=str,
                       help='JSON configuration file for job parameters')
    parser.add_argument('--save-config', type=str,
                       help='Save job configuration to file')
    
    return parser.parse_args()


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {str(e)}")
        raise


def save_config_to_file(config: Dict[str, Any], config_file: str):
    """Save configuration to JSON file."""
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
        logger.error(f"Failed to save configuration file: {str(e)}")
        raise


def create_job_config_from_args(args) -> Dict[str, Any]:
    """Create job configuration from command line arguments."""
    project_config = get_config()
    
    # Prepare hyperparameters
    hyperparameters = {
        'model': args.model,
        'epochs': args.epochs,
        'batch-size': args.batch_size,
        'learning-rate': args.learning_rate,
        'img-size': args.img_size,
        'num-classes': args.num_classes
    }
    
    if args.class_names:
        hyperparameters['class-names'] = args.class_names
    
    # Prepare data paths
    training_data = args.training_data
    validation_data = args.validation_data
    output_path = args.output_path
    
    if not training_data:
        # Use default data path from project config
        bucket = project_config['aws']['data_bucket']
        training_data = f"s3://{bucket}/data/plant-disease-detection/train/"
        validation_data = f"s3://{bucket}/data/plant-disease-detection/val/"
    
    if not output_path:
        bucket = project_config['aws']['data_bucket']
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"s3://{bucket}/models/yolov11-training/{timestamp}/"
    
    # Create configuration
    config = {
        'job_name': args.job_name,
        'base_job_name': args.base_job_name,
        'instance_type': args.instance_type,
        'instance_count': args.instance_count,
        'volume_size': args.volume_size,
        'max_run': args.max_run,
        'training_data_s3_path': training_data,
        'validation_data_s3_path': validation_data,
        'output_s3_path': output_path,
        'hyperparameters': hyperparameters,
        'max_retry_attempts': args.max_retries,
        'enable_profiler': args.enable_profiler,
        'enable_debugger': args.enable_debugger,
        'tags': [
            {'Key': k, 'Value': v} 
            for k, v in project_config['cost']['tags'].items()
        ]
    }
    
    return config


def submit_training_job(orchestrator: SageMakerTrainingOrchestrator, args) -> Dict[str, Any]:
    """Submit a new training job."""
    logger.info("Submitting new training job...")
    
    try:
        # Load configuration
        if args.config_file:
            config_dict = load_config_from_file(args.config_file)
        else:
            config_dict = create_job_config_from_args(args)
        
        # Save configuration if requested
        if args.save_config:
            save_config_to_file(config_dict, args.save_config)
        
        # Create training configuration
        training_config = create_sagemaker_training_config(**config_dict)
        
        logger.info(f"Job configuration: {training_config}")
        
        # Get project config for MLFlow experiment name
        project_config = get_config()
        experiment_name = project_config.get('mlflow', {}).get('experiment_name', 'yolov11-drone-detection')
        
        # Submit job
        result = orchestrator.submit_training_job(
            config=training_config,
            wait=args.wait,
            logs=not args.no_logs
        )
        
        job_name = result['job_name']
        logger.info(f"Training job submitted: {job_name}")
        
        # Log job to MLFlow
        try:
            # Initialize MLFlow integration
            mlflow_integration = MLFlowSageMakerIntegration(aws_profile=args.aws_profile)
            
            # Create run name from job name
            run_name = f"sagemaker-{job_name}"
            
            # Log SageMaker job to MLFlow
            run_id = log_training_job(job_name, experiment_name, run_name, args.aws_profile)
            logger.info(f"Job logged to MLFlow with run ID: {run_id}")
            
            # Add MLFlow info to result
            result['mlflow_run_id'] = run_id
            result['mlflow_experiment'] = experiment_name
            
        except Exception as e:
            logger.warning(f"Failed to log job to MLFlow: {str(e)}")
        
        # Monitor job if wait is enabled
        if args.wait:
            monitor_result = orchestrator.monitor_training_job(result['job_name'])
            result.update(monitor_result)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to submit training job: {str(e)}")
        raise


def monitor_training_job(orchestrator: SageMakerTrainingOrchestrator, job_name: str):
    """Monitor a training job."""
    logger.info(f"Monitoring training job: {job_name}")
    
    def status_callback(job_name: str, status: str, details: Dict):
        """Callback for job status updates."""
        logger.info(f"Job {job_name} status update: {status}")
        
        # Log additional details for certain statuses
        if status == 'Failed':
            failure_reason = details.get('FailureReason', 'Unknown')
            logger.error(f"Job failed: {failure_reason}")
        elif status == 'Completed':
            training_time = details.get('TrainingTimeInSeconds', 0)
            logger.info(f"Job completed in {training_time} seconds")
    
    try:
        result = orchestrator.monitor_training_job(
            job_name=job_name,
            callback=status_callback,
            check_interval=60
        )
        
        final_status = result.get('final_status')
        
        if final_status == 'Failed':
            # Handle failure
            logger.info("Handling training job failure...")
            failure_result = orchestrator.handle_training_failure(job_name, None)
            logger.info(f"Failure handling result: {failure_result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error monitoring job: {str(e)}")
        raise


def list_training_jobs(orchestrator: SageMakerTrainingOrchestrator, args):
    """List training jobs."""
    logger.info("Listing training jobs...")
    
    try:
        jobs = orchestrator.list_training_jobs(
            status_filter=args.status,
            max_results=args.max_results
        )
        
        if not jobs:
            logger.info("No training jobs found")
            return
        
        # Print job summary
        print("\n" + "="*80)
        print("SAGEMAKER TRAINING JOBS")
        print("="*80)
        print(f"{'Job Name':<40} {'Status':<15} {'Created':<20}")
        print("-"*80)
        
        for job in jobs:
            job_name = job['TrainingJobName']
            status = job['TrainingJobStatus']
            created = job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"{job_name:<40} {status:<15} {created:<20}")
        
        print("="*80)
        print(f"Total jobs: {len(jobs)}")
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise


def get_job_details(orchestrator: SageMakerTrainingOrchestrator, job_name: str):
    """Get detailed information about a training job."""
    logger.info(f"Getting details for job: {job_name}")
    
    try:
        details = orchestrator.get_training_job_details(job_name)
        
        if not details:
            logger.warning(f"Job not found: {job_name}")
            return
        
        # Print job details
        print("\n" + "="*60)
        print(f"TRAINING JOB DETAILS: {job_name}")
        print("="*60)
        
        print(f"Status: {details.get('TrainingJobStatus')}")
        print(f"Instance Type: {details.get('ResourceConfig', {}).get('InstanceType')}")
        print(f"Instance Count: {details.get('ResourceConfig', {}).get('InstanceCount')}")
        
        if 'CreationTime' in details:
            print(f"Created: {details['CreationTime']}")
        if 'TrainingStartTime' in details:
            print(f"Started: {details['TrainingStartTime']}")
        if 'TrainingEndTime' in details:
            print(f"Ended: {details['TrainingEndTime']}")
        
        if 'FailureReason' in details:
            print(f"Failure Reason: {details['FailureReason']}")
        
        # Print hyperparameters
        hyperparams = details.get('HyperParameters', {})
        if hyperparams:
            print(f"\nHyperparameters:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error getting job details: {str(e)}")
        raise


def get_job_metrics(orchestrator: SageMakerTrainingOrchestrator, job_name: str):
    """Get training metrics for a job."""
    logger.info(f"Getting metrics for job: {job_name}")
    
    try:
        metrics = orchestrator.get_training_metrics(job_name)
        
        if not metrics:
            logger.warning(f"No metrics found for job: {job_name}")
            return
        
        # Print metrics
        print("\n" + "="*60)
        print(f"TRAINING METRICS: {job_name}")
        print("="*60)
        
        for metric_name, datapoints in metrics.items():
            if datapoints:
                latest_value = datapoints[-1]['Average']
                timestamp = datapoints[-1]['Timestamp']
                print(f"{metric_name}: {latest_value:.4f} (at {timestamp})")
            else:
                print(f"{metric_name}: No data available")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error getting job metrics: {str(e)}")
        raise


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting SageMaker training job management...")
    
    try:
        # Initialize orchestrator
        orchestrator = SageMakerTrainingOrchestrator(
            aws_profile=args.aws_profile,
            region=args.region
        )
        
        # Execute requested action
        if args.submit or (not any([args.monitor_job, args.stop_job, args.list_jobs, 
                                   args.job_details, args.job_metrics])):
            # Submit new job (default action)
            result = submit_training_job(orchestrator, args)
            print(f"\nTraining job submitted successfully: {result['job_name']}")
            
        elif args.monitor_job:
            # Monitor existing job
            result = monitor_training_job(orchestrator, args.monitor_job)
            print(f"\nJob monitoring completed: {result.get('final_status')}")
            
        elif args.stop_job:
            # Stop running job
            result = orchestrator.stop_training_job(args.stop_job)
            print(f"\nJob stop requested: {result['status']}")
            
        elif args.list_jobs:
            # List jobs
            list_training_jobs(orchestrator, args)
            
        elif args.job_details:
            # Get job details
            get_job_details(orchestrator, args.job_details)
            
        elif args.job_metrics:
            # Get job metrics
            get_job_metrics(orchestrator, args.job_metrics)
        
        logger.info("Operation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)