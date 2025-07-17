#!/usr/bin/env python3
"""
Environment-specific Configuration for MLOps SageMaker Demo

This module provides environment-specific configuration management for the project.
It supports different configurations for development, staging, and production environments.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Default environment
DEFAULT_ENV = "development"

# Environment configuration paths
CONFIG_DIR = Path(__file__).parent
ENV_CONFIG_DIR = CONFIG_DIR / "environments"
ENV_CONFIG_DIR.mkdir(exist_ok=True)

# Environment variable for setting the environment
ENV_VAR_NAME = "MLOPS_ENVIRONMENT"


def get_environment() -> str:
    """
    Get the current environment name from environment variable or default.
    
    Returns:
        str: Environment name (development, staging, or production)
    """
    env = os.environ.get(ENV_VAR_NAME, DEFAULT_ENV).lower()
    if env not in ["development", "staging", "production"]:
        print(f"Warning: Unknown environment '{env}', using '{DEFAULT_ENV}' instead")
        env = DEFAULT_ENV
    return env


def load_environment_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment-specific configuration.
    
    Args:
        environment: Optional environment name, defaults to get_environment()
        
    Returns:
        Dict[str, Any]: Environment-specific configuration
    """
    if environment is None:
        environment = get_environment()
    
    # Define file paths for different formats
    yaml_path = ENV_CONFIG_DIR / f"{environment}.yaml"
    yml_path = ENV_CONFIG_DIR / f"{environment}.yml"
    json_path = ENV_CONFIG_DIR / f"{environment}.json"
    
    # Try to load configuration from files
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    elif yml_path.exists():
        with open(yml_path, 'r') as f:
            return yaml.safe_load(f)
    elif json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: No configuration file found for environment '{environment}'")
        return {}


def create_default_environment_configs():
    """
    Create default environment configuration files if they don't exist.
    """
    # Development environment config
    dev_config = {
        "aws": {
            "region": "us-east-1",
            "data_bucket": "lucaskle-ab3-project-pv"
        },
        "sagemaker": {
            "domain_name": "mlops-sagemaker-demo-dev-domain"
        },
        "training": {
            "instance_type": "ml.g4dn.xlarge",
            "instance_count": 1
        },
        "inference": {
            "instance_type": "ml.m5.large",
            "initial_instance_count": 1
        },
        "cost": {
            "tags": {
                "Environment": "Development"
            }
        }
    }
    
    # Staging environment config
    staging_config = {
        "aws": {
            "region": "us-east-1",
            "data_bucket": "lucaskle-ab3-project-pv-staging"
        },
        "sagemaker": {
            "domain_name": "mlops-sagemaker-demo-staging-domain"
        },
        "training": {
            "instance_type": "ml.g4dn.2xlarge",
            "instance_count": 1
        },
        "inference": {
            "instance_type": "ml.m5.xlarge",
            "initial_instance_count": 2
        },
        "cost": {
            "tags": {
                "Environment": "Staging"
            }
        }
    }
    
    # Production environment config
    prod_config = {
        "aws": {
            "region": "us-east-1",
            "data_bucket": "lucaskle-ab3-project-pv-prod"
        },
        "sagemaker": {
            "domain_name": "mlops-sagemaker-demo-prod-domain"
        },
        "training": {
            "instance_type": "ml.g5.2xlarge",
            "instance_count": 2
        },
        "inference": {
            "instance_type": "ml.m5.2xlarge",
            "initial_instance_count": 2,
            "max_instance_count": 5
        },
        "monitoring": {
            "data_capture_percentage": 20,  # Lower percentage for production to reduce costs
            "schedule_cron": "cron(0 */4 * * ? *)"  # More frequent monitoring in production
        },
        "cost": {
            "tags": {
                "Environment": "Production"
            }
        }
    }
    
    # Write configuration files
    ENV_CONFIG_DIR.mkdir(exist_ok=True)
    
    with open(ENV_CONFIG_DIR / "development.yaml", 'w') as f:
        yaml.dump(dev_config, f, default_flow_style=False)
    
    with open(ENV_CONFIG_DIR / "staging.yaml", 'w') as f:
        yaml.dump(staging_config, f, default_flow_style=False)
    
    with open(ENV_CONFIG_DIR / "production.yaml", 'w') as f:
        yaml.dump(prod_config, f, default_flow_style=False)
    
    print("Created default environment configuration files")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary with values that take precedence
        
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


if __name__ == "__main__":
    # Create default environment configs if they don't exist
    create_default_environment_configs()
    
    # Print current environment and configuration
    env = get_environment()
    config = load_environment_config(env)
    print(f"Current environment: {env}")
    print(json.dumps(config, indent=2))