"""
Lambda function for deploying SageMaker endpoints with auto-scaling.

This Lambda function is used in the SageMaker pipeline to deploy models
to endpoints with auto-scaling configuration based on model performance.
"""

import os
import json
import logging
import boto3
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
application_autoscaling_client = boto3.client('application-autoscaling')


def lambda_handler(event, context):
    """
    Lambda handler for deploying SageMaker endpoints.
    
    Args:
        event: Lambda event containing model details
        context: Lambda context
        
    Returns:
        Deployment results
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Extract parameters from event
        model_name = event.get('model_name')
        model_version = event.get('model_version', '1')
        endpoint_name = event.get('endpoint_name', f"{model_name}-{model_version}".replace('.', '-').lower())
        instance_type = event.get('instance_type', 'ml.m5.large')
        initial_instance_count = event.get('initial_instance_count', 1)
        max_instance_count = event.get('max_instance_count', 3)
        role_arn = event.get('role_arn')
        
        # Check if endpoint already exists
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} already exists, updating...")
            
            # Update endpoint
            result = update_endpoint(
                endpoint_name=endpoint_name,
                model_name=model_name,
                model_version=model_version,
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                role_arn=role_arn
            )
        except sagemaker_client.exceptions.ClientError:
            logger.info(f"Endpoint {endpoint_name} does not exist, creating...")
            
            # Create endpoint
            result = create_endpoint(
                endpoint_name=endpoint_name,
                model_name=model_name,
                model_version=model_version,
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                role_arn=role_arn
            )
        
        # Configure auto-scaling
        configure_autoscaling(
            endpoint_name=endpoint_name,
            max_instance_count=max_instance_count
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Endpoint {endpoint_name} deployed successfully",
                'endpoint_name': endpoint_name,
                'endpoint_status': result.get('EndpointStatus', 'Creating'),
                'model_name': model_name,
                'model_version': model_version
            })
        }
        
    except Exception as e:
        logger.error(f"Error deploying endpoint: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f"Error deploying endpoint: {str(e)}"
            })
        }


def create_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    instance_type: str,
    initial_instance_count: int,
    role_arn: str
) -> Dict[str, Any]:
    """
    Create a SageMaker endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        model_name: Name of the model
        model_version: Version of the model
        instance_type: Instance type for the endpoint
        initial_instance_count: Initial number of instances
        role_arn: IAM role ARN for SageMaker
        
    Returns:
        Endpoint creation response
    """
    logger.info(f"Creating endpoint: {endpoint_name}")
    
    # Get model package ARN
    model_package_arn = get_model_package_arn(model_name, model_version)
    
    # Create model
    model_response = sagemaker_client.create_model(
        ModelName=f"{endpoint_name}-model",
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=role_arn
    )
    
    logger.info(f"Model created: {model_response['ModelArn']}")
    
    # Create endpoint config
    endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=f"{endpoint_name}-config",
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': f"{endpoint_name}-model",
                'InitialInstanceCount': initial_instance_count,
                'InstanceType': instance_type,
                'InitialVariantWeight': 1.0
            }
        ]
    )
    
    logger.info(f"Endpoint config created: {endpoint_config_response['EndpointConfigArn']}")
    
    # Create endpoint
    endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=f"{endpoint_name}-config"
    )
    
    logger.info(f"Endpoint creation initiated: {endpoint_response['EndpointArn']}")
    
    return endpoint_response


def update_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    instance_type: str,
    initial_instance_count: int,
    role_arn: str
) -> Dict[str, Any]:
    """
    Update an existing SageMaker endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        model_name: Name of the model
        model_version: Version of the model
        instance_type: Instance type for the endpoint
        initial_instance_count: Initial number of instances
        role_arn: IAM role ARN for SageMaker
        
    Returns:
        Endpoint update response
    """
    logger.info(f"Updating endpoint: {endpoint_name}")
    
    # Create new endpoint config
    new_config_name = f"{endpoint_name}-config-{int(time.time())}"
    
    # Get model package ARN
    model_package_arn = get_model_package_arn(model_name, model_version)
    
    # Create model
    model_response = sagemaker_client.create_model(
        ModelName=f"{endpoint_name}-model-{int(time.time())}",
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=role_arn
    )
    
    logger.info(f"Model created: {model_response['ModelArn']}")
    
    # Create endpoint config
    endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_response['ModelName'],
                'InitialInstanceCount': initial_instance_count,
                'InstanceType': instance_type,
                'InitialVariantWeight': 1.0
            }
        ]
    )
    
    logger.info(f"Endpoint config created: {endpoint_config_response['EndpointConfigArn']}")
    
    # Update endpoint
    endpoint_response = sagemaker_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=new_config_name
    )
    
    logger.info(f"Endpoint update initiated: {endpoint_response['EndpointArn']}")
    
    return endpoint_response


def get_model_package_arn(model_name: str, model_version: str) -> str:
    """
    Get the ARN of a model package.
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        
    Returns:
        Model package ARN
    """
    logger.info(f"Getting model package ARN for {model_name} version {model_version}")
    
    # Get model package group
    model_package_group_name = f"{model_name}-group"
    
    # List model packages in the group
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=100
    )
    
    # Find the requested version
    for package in response['ModelPackageSummaryList']:
        if package.get('ModelPackageVersion') == model_version:
            logger.info(f"Found model package: {package['ModelPackageArn']}")
            return package['ModelPackageArn']
    
    # If version not found, use the latest
    if response['ModelPackageSummaryList']:
        logger.warning(f"Model version {model_version} not found, using latest version")
        return response['ModelPackageSummaryList'][0]['ModelPackageArn']
    
    raise ValueError(f"No model packages found for {model_name}")


def configure_autoscaling(
    endpoint_name: str,
    max_instance_count: int,
    min_instance_count: int = 1,
    target_utilization: int = 70,
    scale_in_cooldown: int = 300,
    scale_out_cooldown: int = 60
) -> None:
    """
    Configure auto-scaling for a SageMaker endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        max_instance_count: Maximum number of instances
        min_instance_count: Minimum number of instances
        target_utilization: Target CPU utilization percentage
        scale_in_cooldown: Scale-in cooldown period in seconds
        scale_out_cooldown: Scale-out cooldown period in seconds
    """
    logger.info(f"Configuring auto-scaling for endpoint: {endpoint_name}")
    
    # Register scalable target
    try:
        application_autoscaling_client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_instance_count,
            MaxCapacity=max_instance_count
        )
        
        logger.info(f"Registered scalable target for endpoint: {endpoint_name}")
        
        # Configure scaling policy
        application_autoscaling_client.put_scaling_policy(
            PolicyName=f'{endpoint_name}-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': target_utilization,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': scale_in_cooldown,
                'ScaleOutCooldown': scale_out_cooldown
            }
        )
        
        logger.info(f"Configured auto-scaling policy for endpoint: {endpoint_name}")
        
    except Exception as e:
        logger.error(f"Error configuring auto-scaling: {str(e)}")
        raise