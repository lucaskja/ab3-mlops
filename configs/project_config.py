#!/usr/bin/env python3
"""
Project Configuration for MLOps SageMaker Demo
Contains all project-specific settings and configurations
"""

import os
import boto3
from typing import Dict, Any, Optional

# Project Information
PROJECT_NAME = "mlops-sagemaker-demo"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "MLOps demonstration using AWS SageMaker with governance and monitoring"

# AWS Configuration
AWS_PROFILE = "ab"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DATA_BUCKET_NAME = "lucaskle-ab3-project-pv"

# IAM Role Configuration
IAM_STACK_NAME = f"{PROJECT_NAME}-iam-roles"
DATA_SCIENTIST_ROLE_NAME = f"{PROJECT_NAME}-DataScientist-Role"
ML_ENGINEER_ROLE_NAME = f"{PROJECT_NAME}-MLEngineer-Role"
SAGEMAKER_EXECUTION_ROLE_NAME = f"{PROJECT_NAME}-SageMaker-Execution-Role"
SAGEMAKER_STUDIO_POLICY_NAME = f"{PROJECT_NAME}-SageMaker-Studio-Policy"

# SageMaker Configuration
SAGEMAKER_DOMAIN_NAME = f"{PROJECT_NAME}-domain"

# MLFlow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Will be updated with SageMaker hosted MLFlow
MLFLOW_EXPERIMENT_NAME = "yolov11-drone-detection"
MLFLOW_ARTIFACT_BUCKET = f"{PROJECT_NAME}-mlflow-artifacts"

# Model Configuration
MODEL_NAME = "yolov11-drone-detection"
MODEL_FRAMEWORK = "pytorch"
MODEL_FRAMEWORK_VERSION = "2.0"
PYTHON_VERSION = "py310"

# Training Configuration
TRAINING_INSTANCE_TYPE = "ml.g4dn.xlarge"
TRAINING_INSTANCE_COUNT = 1
TRAINING_VOLUME_SIZE = 30  # GB

# Inference Configuration
INFERENCE_INSTANCE_TYPE = "ml.m5.large"
INFERENCE_INITIAL_INSTANCE_COUNT = 1
INFERENCE_MAX_INSTANCE_COUNT = 3

# Pipeline Configuration
PIPELINE_NAME = f"{PROJECT_NAME}-pipeline"

# Monitoring Configuration
MODEL_MONITOR_SCHEDULE_NAME = f"{PROJECT_NAME}-monitor"
DATA_CAPTURE_PERCENTAGE = 100
MONITORING_SCHEDULE_CRON = "cron(0 */6 * * ? *)"  # Every 6 hours

# Cost Configuration
COST_ALLOCATION_TAGS = {
    "Project": PROJECT_NAME,
    "Environment": "Development",
    "Owner": "MLOps-Team",
    "CostCenter": "Research-Development"
}

# Notification Configuration
SNS_TOPIC_NAME = f"{PROJECT_NAME}-notifications"
EVENTBRIDGE_RULE_NAME = f"{PROJECT_NAME}-pipeline-events"


def get_account_id() -> Optional[str]:
    """Get AWS account ID using the configured profile"""
    try:
        session = boto3.Session(profile_name=AWS_PROFILE)
        sts_client = session.client('sts')
        return sts_client.get_caller_identity()['Account']
    except Exception:
        return None


def get_role_arn(role_name: str) -> str:
    """Generate IAM role ARN"""
    account_id = get_account_id()
    if account_id:
        return f"arn:aws:iam::{account_id}:role/{role_name}"
    else:
        return f"arn:aws:iam::ACCOUNT_ID:role/{role_name}"


def get_policy_arn(policy_name: str) -> str:
    """Generate IAM policy ARN"""
    account_id = get_account_id()
    if account_id:
        return f"arn:aws:iam::{account_id}:policy/{policy_name}"
    else:
        return f"arn:aws:iam::ACCOUNT_ID:policy/{policy_name}"


def get_iam_roles() -> Dict[str, str]:
    """Get all IAM role ARNs"""
    return {
        "data_scientist_role_arn": get_role_arn(DATA_SCIENTIST_ROLE_NAME),
        "ml_engineer_role_arn": get_role_arn(ML_ENGINEER_ROLE_NAME),
        "sagemaker_execution_role_arn": get_role_arn(SAGEMAKER_EXECUTION_ROLE_NAME),
        "sagemaker_studio_policy_arn": get_policy_arn(SAGEMAKER_STUDIO_POLICY_NAME)
    }


def get_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """
    Return complete configuration dictionary with environment-specific overrides.
    
    Args:
        environment: Optional environment name (development, staging, production)
                    If None, uses the environment from environment_config.get_environment()
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary
    """
    # Import here to avoid circular imports
    from configs.environment_config import get_environment, load_environment_config, deep_merge
    
    # Get environment-specific configuration
    if environment is None:
        environment = get_environment()
    
    env_config = load_environment_config(environment)
    iam_roles = get_iam_roles()
    
    # Base configuration
    base_config = {
        "project": {
            "name": PROJECT_NAME,
            "version": PROJECT_VERSION,
            "description": PROJECT_DESCRIPTION,
            "environment": environment
        },
        "aws": {
            "profile": AWS_PROFILE,
            "region": AWS_REGION,
            "account_id": get_account_id(),
            "data_bucket": DATA_BUCKET_NAME
        },
        "iam": {
            "stack_name": IAM_STACK_NAME,
            "roles": {
                "data_scientist": {
                    "name": DATA_SCIENTIST_ROLE_NAME,
                    "arn": iam_roles["data_scientist_role_arn"]
                },
                "ml_engineer": {
                    "name": ML_ENGINEER_ROLE_NAME,
                    "arn": iam_roles["ml_engineer_role_arn"]
                },
                "sagemaker_execution": {
                    "name": SAGEMAKER_EXECUTION_ROLE_NAME,
                    "arn": iam_roles["sagemaker_execution_role_arn"]
                }
            },
            "policies": {
                "sagemaker_studio": {
                    "name": SAGEMAKER_STUDIO_POLICY_NAME,
                    "arn": iam_roles["sagemaker_studio_policy_arn"]
                }
            }
        },
        "sagemaker": {
            "domain_name": SAGEMAKER_DOMAIN_NAME,
            "execution_role_arn": iam_roles["sagemaker_execution_role_arn"]
        },
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "artifact_bucket": MLFLOW_ARTIFACT_BUCKET
        },
        "model": {
            "name": MODEL_NAME,
            "framework": MODEL_FRAMEWORK,
            "framework_version": MODEL_FRAMEWORK_VERSION,
            "python_version": PYTHON_VERSION
        },
        "training": {
            "instance_type": TRAINING_INSTANCE_TYPE,
            "instance_count": TRAINING_INSTANCE_COUNT,
            "volume_size": TRAINING_VOLUME_SIZE
        },
        "inference": {
            "instance_type": INFERENCE_INSTANCE_TYPE,
            "initial_instance_count": INFERENCE_INITIAL_INSTANCE_COUNT,
            "max_instance_count": INFERENCE_MAX_INSTANCE_COUNT
        },
        "pipeline": {
            "name": PIPELINE_NAME,
            "role_arn": iam_roles["sagemaker_execution_role_arn"]
        },
        "monitoring": {
            "schedule_name": MODEL_MONITOR_SCHEDULE_NAME,
            "data_capture_percentage": DATA_CAPTURE_PERCENTAGE,
            "schedule_cron": MONITORING_SCHEDULE_CRON
        },
        "cost": {
            "tags": COST_ALLOCATION_TAGS
        },
        "notifications": {
            "sns_topic": SNS_TOPIC_NAME,
            "eventbridge_rule": EVENTBRIDGE_RULE_NAME
        }
    }
    
    # Merge base config with environment-specific config
    return deep_merge(base_config, env_config)


if __name__ == "__main__":
    import json
    config = get_config()
    print(json.dumps(config, indent=2))