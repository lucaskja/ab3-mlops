"""
Error Recovery Mechanisms for SageMaker Training Jobs

This module provides sophisticated error recovery mechanisms for SageMaker training jobs,
including automatic retries, checkpointing, and failure analysis.

Requirements addressed:
- Robust error handling for training jobs
- Automatic retry with exponential backoff
- Checkpoint-based resumption of failed jobs
- Detailed error analysis and reporting
"""

import os
import time
import json
import logging
import boto3
import botocore
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from datetime import datetime, timedelta

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingJobRecoveryManager:
    """
    Manages error recovery for SageMaker training jobs.
    
    This class provides mechanisms for:
    - Automatic retry of failed training jobs
    - Checkpoint-based resumption
    - Failure analysis and reporting
    - Resource cleanup after failures
    """
    
    # Error categories for different recovery strategies
    RETRYABLE_ERRORS = [
        "ResourceLimitExceeded",
        "InternalServerError",
        "ThrottlingException",
        "ServiceUnavailable",
        "ProvisionedThroughputExceededException"
    ]
    
    RESOURCE_ERRORS = [
        "ResourceLimitExceeded",
        "OutOfMemoryError",
        "InsufficientCapacity"
    ]
    
    PERMISSION_ERRORS = [
        "AccessDenied",
        "UnauthorizedOperation",
        "InvalidRoleException"
    ]
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the training job recovery manager.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker', region_name=region)
        self.cloudwatch_client = self.session.client('cloudwatch', region_name=region)
        self.logs_client = self.session.client('logs', region_name=region)
        
        # Get project configuration
        self.project_config = get_config()
        
        logger.info(f"Training job recovery manager initialized for region: {region}")
    
    def retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        retryable_exceptions: Optional[List[Exception]] = None
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplicative factor for backoff
            jitter: Random jitter factor to add to backoff
            retryable_exceptions: List of exceptions to retry on
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: Last exception encountered after max retries
        """
        import random
        
        if retryable_exceptions is None:
            retryable_exceptions = [
                botocore.exceptions.ClientError,
                botocore.exceptions.ConnectionError,
                botocore.exceptions.HTTPClientError
            ]
        
        last_exception = None
        backoff = initial_backoff
        
        for retry in range(max_retries + 1):
            try:
                return func()
            except tuple(retryable_exceptions) as e:
                last_exception = e
                
                if retry == max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded")
                    raise
                
                # Add jitter to backoff
                jitter_amount = random.uniform(-jitter, jitter)
                sleep_time = backoff * (1 + jitter_amount)
                
                logger.warning(f"Retry {retry + 1}/{max_retries} after error: {str(e)}")
                logger.warning(f"Waiting {sleep_time:.2f} seconds before retry")
                
                time.sleep(sleep_time)
                backoff *= backoff_factor
        
        # This should not be reached due to the raise in the loop
        raise last_exception if last_exception else RuntimeError("Unknown error in retry_with_backoff")
    
    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Get the status of a SageMaker training job with retry logic.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            Training job details
        """
        try:
            return self.retry_with_backoff(
                lambda: self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
            )
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFound':
                logger.error(f"Training job not found: {job_name}")
                return {"TrainingJobStatus": "NotFound"}
            raise
    
    def analyze_job_failure(self, job_name: str) -> Dict[str, Any]:
        """
        Analyze the failure of a SageMaker training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            Failure analysis results
        """
        try:
            # Get job details
            job_details = self.get_training_job_status(job_name)
            
            if job_details.get("TrainingJobStatus") != "Failed":
                logger.warning(f"Job {job_name} is not in Failed state, current status: {job_details.get('TrainingJobStatus')}")
                return {
                    "job_name": job_name,
                    "status": job_details.get("TrainingJobStatus"),
                    "failure_reason": None,
                    "error_type": None,
                    "is_retryable": False,
                    "logs_analysis": None
                }
            
            # Get failure reason
            failure_reason = job_details.get("FailureReason", "Unknown failure reason")
            
            # Determine error type
            error_type = self._categorize_error(failure_reason)
            
            # Check if error is retryable
            is_retryable = any(error in failure_reason for error in self.RETRYABLE_ERRORS)
            
            # Analyze logs
            logs_analysis = self._analyze_cloudwatch_logs(job_name)
            
            # Return analysis results
            analysis = {
                "job_name": job_name,
                "status": job_details.get("TrainingJobStatus"),
                "failure_reason": failure_reason,
                "error_type": error_type,
                "is_retryable": is_retryable,
                "logs_analysis": logs_analysis,
                "resource_config": job_details.get("ResourceConfig"),
                "stopping_condition": job_details.get("StoppingCondition"),
                "failure_time": job_details.get("TrainingEndTime")
            }
            
            logger.info(f"Failure analysis for job {job_name}: {error_type} (Retryable: {is_retryable})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing job failure: {str(e)}")
            return {
                "job_name": job_name,
                "status": "Unknown",
                "failure_reason": str(e),
                "error_type": "AnalysisError",
                "is_retryable": False,
                "logs_analysis": None
            }
    
    def _categorize_error(self, failure_reason: str) -> str:
        """
        Categorize the error based on the failure reason.
        
        Args:
            failure_reason: Failure reason from SageMaker
            
        Returns:
            Error category
        """
        if any(error in failure_reason for error in self.RESOURCE_ERRORS):
            return "ResourceError"
        elif any(error in failure_reason for error in self.PERMISSION_ERRORS):
            return "PermissionError"
        elif "algorithm-error" in failure_reason.lower():
            return "AlgorithmError"
        elif "timeout" in failure_reason.lower():
            return "TimeoutError"
        elif "out of memory" in failure_reason.lower() or "oom" in failure_reason.lower():
            return "OutOfMemoryError"
        elif "validation error" in failure_reason.lower():
            return "ValidationError"
        else:
            return "UnknownError"
    
    def _analyze_cloudwatch_logs(self, job_name: str) -> Dict[str, Any]:
        """
        Analyze CloudWatch logs for a failed training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            Log analysis results
        """
        try:
            # Get log group and stream names
            log_group = f"/aws/sagemaker/TrainingJobs"
            log_streams = self.logs_client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                limit=5
            ).get("logStreams", [])
            
            if not log_streams:
                logger.warning(f"No log streams found for job {job_name}")
                return {"error_messages": [], "has_errors": False}
            
            # Get error messages from logs
            error_messages = []
            for stream in log_streams:
                stream_name = stream.get("logStreamName")
                
                # Get log events
                log_events = self.logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=stream_name,
                    limit=1000
                ).get("events", [])
                
                # Filter for error messages
                for event in log_events:
                    message = event.get("message", "")
                    if any(keyword in message.lower() for keyword in ["error", "exception", "fail", "traceback"]):
                        error_messages.append({
                            "timestamp": event.get("timestamp"),
                            "message": message
                        })
            
            return {
                "error_messages": error_messages,
                "has_errors": len(error_messages) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CloudWatch logs: {str(e)}")
            return {"error": str(e), "has_errors": False}
    
    def get_latest_checkpoint(self, job_name: str) -> Optional[str]:
        """
        Get the latest checkpoint from a training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            S3 URI of the latest checkpoint or None if not found
        """
        try:
            # Get job details
            job_details = self.get_training_job_status(job_name)
            
            # Get output path
            output_path = job_details.get("OutputDataConfig", {}).get("S3OutputPath")
            
            if not output_path:
                logger.warning(f"No output path found for job {job_name}")
                return None
            
            # Parse S3 URI
            if output_path.startswith("s3://"):
                parts = output_path[5:].split("/", 1)
                bucket = parts[0]
                prefix = parts[1] if len(parts) > 1 else ""
                
                # List objects in checkpoint directory
                s3_client = self.session.client('s3')
                checkpoint_prefix = f"{prefix}/{job_name}/checkpoints/"
                
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=checkpoint_prefix
                )
                
                # Find latest checkpoint
                checkpoints = []
                for obj in response.get("Contents", []):
                    key = obj.get("Key")
                    if key and key.endswith(".pt"):  # PyTorch checkpoint
                        checkpoints.append({
                            "key": key,
                            "last_modified": obj.get("LastModified"),
                            "size": obj.get("Size")
                        })
                
                if checkpoints:
                    # Sort by last modified time (newest first)
                    checkpoints.sort(key=lambda x: x["last_modified"], reverse=True)
                    latest_checkpoint = checkpoints[0]
                    checkpoint_uri = f"s3://{bucket}/{latest_checkpoint['key']}"
                    
                    logger.info(f"Found latest checkpoint: {checkpoint_uri}")
                    return checkpoint_uri
            
            logger.warning(f"No checkpoints found for job {job_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest checkpoint: {str(e)}")
            return None
    
    def retry_failed_job(
        self,
        job_name: str,
        new_job_name: Optional[str] = None,
        use_checkpoints: bool = True,
        adjust_resources: bool = True,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Retry a failed training job with recovery mechanisms.
        
        Args:
            job_name: Name of the failed training job
            new_job_name: Name for the new job (optional)
            use_checkpoints: Whether to use checkpoints for resumption
            adjust_resources: Whether to adjust resources based on failure analysis
            max_retries: Maximum number of retries
            
        Returns:
            New job details
        """
        try:
            # Get job details
            job_details = self.get_training_job_status(job_name)
            
            if job_details.get("TrainingJobStatus") == "NotFound":
                raise ValueError(f"Training job {job_name} not found")
            
            # Analyze failure
            failure_analysis = self.analyze_job_failure(job_name)
            
            if not failure_analysis.get("is_retryable", False):
                logger.warning(f"Job {job_name} has a non-retryable error: {failure_analysis.get('error_type')}")
                
                # Still proceed but with a warning
                logger.warning("Proceeding with retry despite non-retryable error")
            
            # Generate new job name if not provided
            if not new_job_name:
                timestamp = int(time.time())
                new_job_name = f"{job_name}-retry-{timestamp}"
            
            # Get checkpoint if available and requested
            checkpoint_uri = None
            if use_checkpoints:
                checkpoint_uri = self.get_latest_checkpoint(job_name)
            
            # Create new job configuration
            new_job_config = self._create_retry_job_config(
                job_details=job_details,
                failure_analysis=failure_analysis,
                new_job_name=new_job_name,
                checkpoint_uri=checkpoint_uri,
                adjust_resources=adjust_resources
            )
            
            # Create new training job
            logger.info(f"Creating retry job {new_job_name} for failed job {job_name}")
            response = self.retry_with_backoff(
                lambda: self.sagemaker_client.create_training_job(**new_job_config)
            )
            
            logger.info(f"Successfully created retry job: {new_job_name}")
            return {
                "original_job_name": job_name,
                "new_job_name": new_job_name,
                "checkpoint_used": checkpoint_uri is not None,
                "resources_adjusted": adjust_resources,
                "job_arn": response.get("TrainingJobArn")
            }
            
        except Exception as e:
            logger.error(f"Error retrying failed job: {str(e)}")
            raise
    
    def _create_retry_job_config(
        self,
        job_details: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        new_job_name: str,
        checkpoint_uri: Optional[str] = None,
        adjust_resources: bool = True
    ) -> Dict[str, Any]:
        """
        Create configuration for a retry job based on the original job and failure analysis.
        
        Args:
            job_details: Original job details
            failure_analysis: Failure analysis results
            new_job_name: Name for the new job
            checkpoint_uri: URI of checkpoint to resume from (optional)
            adjust_resources: Whether to adjust resources based on failure analysis
            
        Returns:
            New job configuration
        """
        # Start with a copy of the original job configuration
        new_config = {
            "TrainingJobName": new_job_name,
            "AlgorithmSpecification": job_details.get("AlgorithmSpecification", {}),
            "RoleArn": job_details.get("RoleArn", ""),
            "InputDataConfig": job_details.get("InputDataConfig", []),
            "OutputDataConfig": job_details.get("OutputDataConfig", {}),
            "ResourceConfig": job_details.get("ResourceConfig", {}),
            "StoppingCondition": job_details.get("StoppingCondition", {}),
            "Tags": job_details.get("Tags", [])
        }
        
        # Add hyperparameters
        hyperparameters = job_details.get("HyperParameters", {})
        if hyperparameters:
            new_config["HyperParameters"] = hyperparameters
        
        # Add checkpoint configuration if available
        if checkpoint_uri:
            # Add or update checkpoint configuration
            if "HyperParameters" not in new_config:
                new_config["HyperParameters"] = {}
            
            new_config["HyperParameters"]["checkpoint_path"] = checkpoint_uri
            new_config["HyperParameters"]["resume_from_checkpoint"] = "True"
            
            logger.info(f"Configured job to resume from checkpoint: {checkpoint_uri}")
        
        # Adjust resources based on failure analysis
        if adjust_resources and "ResourceConfig" in new_config:
            error_type = failure_analysis.get("error_type")
            
            if error_type == "OutOfMemoryError":
                # Increase instance size for OOM errors
                instance_type = new_config["ResourceConfig"].get("InstanceType", "")
                new_instance_type = self._get_larger_instance(instance_type)
                
                if new_instance_type != instance_type:
                    new_config["ResourceConfig"]["InstanceType"] = new_instance_type
                    logger.info(f"Upgraded instance type from {instance_type} to {new_instance_type} due to OOM error")
            
            elif error_type == "TimeoutError":
                # Increase max runtime for timeout errors
                if "StoppingCondition" in new_config:
                    max_runtime = new_config["StoppingCondition"].get("MaxRuntimeInSeconds", 86400)
                    new_max_runtime = int(max_runtime * 1.5)  # Increase by 50%
                    new_config["StoppingCondition"]["MaxRuntimeInSeconds"] = new_max_runtime
                    logger.info(f"Increased max runtime from {max_runtime} to {new_max_runtime} seconds due to timeout")
        
        return new_config
    
    def _get_larger_instance(self, instance_type: str) -> str:
        """
        Get a larger instance type for the given instance type.
        
        Args:
            instance_type: Current instance type
            
        Returns:
            Larger instance type or the same if already at the largest
        """
        # Define instance type upgrade paths
        instance_upgrades = {
            # General purpose
            "ml.m5.large": "ml.m5.xlarge",
            "ml.m5.xlarge": "ml.m5.2xlarge",
            "ml.m5.2xlarge": "ml.m5.4xlarge",
            "ml.m5.4xlarge": "ml.m5.12xlarge",
            "ml.m5.12xlarge": "ml.m5.24xlarge",
            
            # Compute optimized
            "ml.c5.large": "ml.c5.xlarge",
            "ml.c5.xlarge": "ml.c5.2xlarge",
            "ml.c5.2xlarge": "ml.c5.4xlarge",
            "ml.c5.4xlarge": "ml.c5.9xlarge",
            "ml.c5.9xlarge": "ml.c5.18xlarge",
            
            # Memory optimized
            "ml.r5.large": "ml.r5.xlarge",
            "ml.r5.xlarge": "ml.r5.2xlarge",
            "ml.r5.2xlarge": "ml.r5.4xlarge",
            "ml.r5.4xlarge": "ml.r5.12xlarge",
            "ml.r5.12xlarge": "ml.r5.24xlarge",
            
            # GPU instances
            "ml.g4dn.xlarge": "ml.g4dn.2xlarge",
            "ml.g4dn.2xlarge": "ml.g4dn.4xlarge",
            "ml.g4dn.4xlarge": "ml.g4dn.8xlarge",
            "ml.g4dn.8xlarge": "ml.g4dn.16xlarge",
            
            "ml.p3.2xlarge": "ml.p3.8xlarge",
            "ml.p3.8xlarge": "ml.p3.16xlarge",
            
            "ml.g5.xlarge": "ml.g5.2xlarge",
            "ml.g5.2xlarge": "ml.g5.4xlarge",
            "ml.g5.4xlarge": "ml.g5.8xlarge",
            "ml.g5.8xlarge": "ml.g5.16xlarge",
            "ml.g5.16xlarge": "ml.g5.24xlarge"
        }
        
        return instance_upgrades.get(instance_type, instance_type)
    
    def monitor_training_job(
        self,
        job_name: str,
        polling_interval: int = 30,
        auto_retry: bool = True,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Monitor a training job with automatic recovery.
        
        Args:
            job_name: Name of the training job
            polling_interval: Interval in seconds between status checks
            auto_retry: Whether to automatically retry failed jobs
            max_retries: Maximum number of retries
            
        Returns:
            Final job status
        """
        retry_count = 0
        current_job_name = job_name
        job_history = [job_name]
        
        while True:
            try:
                # Get job status
                job_details = self.get_training_job_status(current_job_name)
                status = job_details.get("TrainingJobStatus")
                
                logger.info(f"Job {current_job_name} status: {status}")
                
                if status == "Completed":
                    return {
                        "final_status": "Completed",
                        "job_name": current_job_name,
                        "original_job_name": job_name,
                        "job_history": job_history,
                        "retry_count": retry_count,
                        "job_details": job_details
                    }
                
                elif status == "Failed":
                    if auto_retry and retry_count < max_retries:
                        # Analyze failure
                        failure_analysis = self.analyze_job_failure(current_job_name)
                        
                        # Check if error is retryable
                        if failure_analysis.get("is_retryable", False) or retry_count == 0:
                            # Retry job
                            retry_count += 1
                            logger.info(f"Retrying failed job {current_job_name} (Attempt {retry_count}/{max_retries})")
                            
                            retry_result = self.retry_failed_job(
                                job_name=current_job_name,
                                use_checkpoints=True,
                                adjust_resources=True
                            )
                            
                            current_job_name = retry_result.get("new_job_name")
                            job_history.append(current_job_name)
                            
                            logger.info(f"Created retry job: {current_job_name}")
                            time.sleep(polling_interval)
                            continue
                        else:
                            logger.warning(f"Job {current_job_name} failed with non-retryable error")
                    
                    return {
                        "final_status": "Failed",
                        "job_name": current_job_name,
                        "original_job_name": job_name,
                        "job_history": job_history,
                        "retry_count": retry_count,
                        "job_details": job_details,
                        "failure_analysis": self.analyze_job_failure(current_job_name)
                    }
                
                elif status == "Stopped":
                    return {
                        "final_status": "Stopped",
                        "job_name": current_job_name,
                        "original_job_name": job_name,
                        "job_history": job_history,
                        "retry_count": retry_count,
                        "job_details": job_details
                    }
                
                elif status == "NotFound":
                    raise ValueError(f"Training job {current_job_name} not found")
                
                # Wait for next polling interval
                time.sleep(polling_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring job {current_job_name}: {str(e)}")
                return {
                    "final_status": "Error",
                    "job_name": current_job_name,
                    "original_job_name": job_name,
                    "job_history": job_history,
                    "retry_count": retry_count,
                    "error": str(e)
                }


# Helper functions
def get_recovery_manager(aws_profile: str = "ab", region: str = "us-east-1") -> TrainingJobRecoveryManager:
    """
    Get a training job recovery manager instance.
    
    Args:
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        TrainingJobRecoveryManager instance
    """
    return TrainingJobRecoveryManager(aws_profile=aws_profile, region=region)


def retry_failed_job(
    job_name: str,
    new_job_name: Optional[str] = None,
    use_checkpoints: bool = True,
    adjust_resources: bool = True,
    aws_profile: str = "ab"
) -> Dict[str, Any]:
    """
    Retry a failed training job with recovery mechanisms.
    
    Args:
        job_name: Name of the failed training job
        new_job_name: Name for the new job (optional)
        use_checkpoints: Whether to use checkpoints for resumption
        adjust_resources: Whether to adjust resources based on failure analysis
        aws_profile: AWS profile to use
        
    Returns:
        New job details
    """
    recovery_manager = get_recovery_manager(aws_profile=aws_profile)
    return recovery_manager.retry_failed_job(
        job_name=job_name,
        new_job_name=new_job_name,
        use_checkpoints=use_checkpoints,
        adjust_resources=adjust_resources
    )


def monitor_training_job(
    job_name: str,
    polling_interval: int = 30,
    auto_retry: bool = True,
    max_retries: int = 3,
    aws_profile: str = "ab"
) -> Dict[str, Any]:
    """
    Monitor a training job with automatic recovery.
    
    Args:
        job_name: Name of the training job
        polling_interval: Interval in seconds between status checks
        auto_retry: Whether to automatically retry failed jobs
        max_retries: Maximum number of retries
        aws_profile: AWS profile to use
        
    Returns:
        Final job status
    """
    recovery_manager = get_recovery_manager(aws_profile=aws_profile)
    return recovery_manager.monitor_training_job(
        job_name=job_name,
        polling_interval=polling_interval,
        auto_retry=auto_retry,
        max_retries=max_retries
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker Training Job Recovery Tool")
    parser.add_argument("--job-name", required=True, help="Name of the training job")
    parser.add_argument("--action", choices=["analyze", "retry", "monitor"], default="monitor", 
                      help="Action to perform (analyze, retry, or monitor)")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint resumption")
    parser.add_argument("--no-resource-adjust", action="store_true", help="Disable resource adjustment")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries")
    
    args = parser.parse_args()
    
    recovery_manager = TrainingJobRecoveryManager(aws_profile=args.profile, region=args.region)
    
    if args.action == "analyze":
        analysis = recovery_manager.analyze_job_failure(args.job_name)
        print(json.dumps(analysis, indent=2, default=str))
    
    elif args.action == "retry":
        result = recovery_manager.retry_failed_job(
            job_name=args.job_name,
            use_checkpoints=not args.no_checkpoints,
            adjust_resources=not args.no_resource_adjust
        )
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "monitor":
        result = recovery_manager.monitor_training_job(
            job_name=args.job_name,
            max_retries=args.max_retries
        )
        print(json.dumps(result, indent=2, default=str))