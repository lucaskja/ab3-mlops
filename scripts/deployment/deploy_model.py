#!/usr/bin/env python3
"""
Script to deploy a model to a SageMaker endpoint.
"""

import argparse
import json
import boto3
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy model to SageMaker endpoint')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--model-info', type=str, required=True, help='Path to model info JSON file')
    parser.add_argument('--endpoint-name', type=str, required=True, help='Name of the endpoint')
    parser.add_argument('--endpoint-type', type=str, choices=['staging', 'production'], required=True, help='Type of endpoint')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge', help='Instance type for the endpoint')
    parser.add_argument('--instance-count', type=int, default=1, help='Number of instances')
    parser.add_argument('--output', type=str, default='endpoint_info.json', help='Path to output JSON file')
    return parser.parse_args()

def deploy_model(args):
    """Deploy model to SageMaker endpoint."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    sagemaker_client = session.client('sagemaker')
    
    # Load model info
    with open(args.model_info, 'r') as f:
        model_info = json.load(f)
    
    # Create model
    model_name = f"{args.endpoint_name}-model-{int(time.time())}"
    logger.info(f"Creating model {model_name}")
    
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': model_info['inference_image'],
            'ModelDataUrl': model_info['model_artifact_path'],
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'SAGEMAKER_REGION': session.region_name
            }
        },
        ExecutionRoleArn=f"arn:aws:iam::{session.client('sts').get_caller_identity()['Account']}:role/mlops-sagemaker-demo-SageMaker-Execution-Role"
    )
    
    # Create endpoint configuration
    endpoint_config_name = f"{args.endpoint_name}-config-{int(time.time())}"
    logger.info(f"Creating endpoint configuration {endpoint_config_name}")
    
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': args.instance_type,
                'InitialInstanceCount': args.instance_count,
                'InitialVariantWeight': 1.0
            }
        ],
        DataCaptureConfig={
            'EnableCapture': True,
            'InitialSamplingPercentage': 100,
            'DestinationS3Uri': f"s3://lucaskle-ab3-project-pv/endpoint-data-capture/{args.endpoint_name}/",
            'CaptureOptions': [
                {'CaptureMode': 'Input'},
                {'CaptureMode': 'Output'}
            ],
            'CaptureContentTypeHeader': {
                'CsvContentTypes': ['text/csv'],
                'JsonContentTypes': ['application/json']
            }
        }
    )
    
    # Check if endpoint exists
    try:
        sagemaker_client.describe_endpoint(EndpointName=args.endpoint_name)
        endpoint_exists = True
    except sagemaker_client.exceptions.ClientError:
        endpoint_exists = False
    
    # Create or update endpoint
    if endpoint_exists:
        logger.info(f"Updating endpoint {args.endpoint_name}")
        sagemaker_client.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        logger.info(f"Creating endpoint {args.endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    
    # Wait for endpoint to be in service
    logger.info(f"Waiting for endpoint {args.endpoint_name} to be in service")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(
        EndpointName=args.endpoint_name,
        WaiterConfig={
            'Delay': 30,
            'MaxAttempts': 60
        }
    )
    
    # Get endpoint info
    endpoint_info = sagemaker_client.describe_endpoint(EndpointName=args.endpoint_name)
    
    # Save endpoint info
    endpoint_output = {
        'endpoint_name': args.endpoint_name,
        'endpoint_type': args.endpoint_type,
        'model_name': model_name,
        'endpoint_config_name': endpoint_config_name,
        'endpoint_status': endpoint_info['EndpointStatus'],
        'endpoint_arn': endpoint_info['EndpointArn'],
        'creation_time': endpoint_info['CreationTime'].isoformat(),
        'last_modified_time': endpoint_info['LastModifiedTime'].isoformat(),
        'model_info': model_info
    }
    
    with open(args.output, 'w') as f:
        json.dump(endpoint_output, f, indent=2)
    
    logger.info(f"Endpoint info saved to {args.output}")
    return endpoint_output

def main():
    """Main function."""
    args = parse_args()
    deploy_model(args)

if __name__ == '__main__':
    main()