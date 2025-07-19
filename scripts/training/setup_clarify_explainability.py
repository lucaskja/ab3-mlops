#!/usr/bin/env python3
"""
Script to set up SageMaker Clarify explainability for a model endpoint.
"""

import argparse
import json
import boto3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up SageMaker Clarify explainability')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--endpoint-name', type=str, required=True, help='Name of the endpoint')
    parser.add_argument('--clarify-job-name', type=str, required=True, help='Name of the Clarify job')
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge', help='Instance type for Clarify job')
    parser.add_argument('--instance-count', type=int, default=1, help='Number of instances for Clarify job')
    parser.add_argument('--output', type=str, default='clarify_info.json', help='Path to output JSON file')
    return parser.parse_args()

def setup_clarify_explainability(args):
    """Set up SageMaker Clarify explainability."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    sagemaker_client = session.client('sagemaker')
    
    # Create Clarify job
    logger.info(f"Creating Clarify job: {args.clarify_job_name}")
    
    # Define Clarify job configuration
    clarify_job_config = {
        'JobName': args.clarify_job_name,
        'RoleArn': f"arn:aws:iam::{session.client('sts').get_caller_identity()['Account']}:role/service-role/AmazonSageMaker-ExecutionRole",
        'ModelConfig': {
            'ModelName': args.endpoint_name,
            'EnvVariables': {
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'SAGEMAKER_REGION': session.region_name
            }
        },
        'ExplainerConfig': {
            'Shap': {
                'ShapBaselineConfig': {
                    'MimeType': 'image/jpeg',
                    'ShapBaseline': 's3://lucaskle-ab3-project-pv/clarify/baselines/black_image.jpg'
                },
                'NumberOfSamples': 100,
                'Seed': 42
            }
        },
        'DataConfig': {
            'S3Uri': f"s3://lucaskle-ab3-project-pv/clarify/input/{args.endpoint_name}/",
            'LocalPath': '/opt/ml/processing/input',
            'S3DataDistributionType': 'FullyReplicated',
            'S3InputMode': 'File'
        },
        'OutputConfig': {
            'S3OutputPath': f"s3://lucaskle-ab3-project-pv/clarify/output/{args.endpoint_name}/",
            'LocalPath': '/opt/ml/processing/output'
        },
        'ResourceConfig': {
            'InstanceType': args.instance_type,
            'InstanceCount': args.instance_count,
            'VolumeSizeInGB': 20
        },
        'NetworkConfig': {
            'EnableInterContainerTrafficEncryption': True,
            'EnableNetworkIsolation': True
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600
        }
    }
    
    # Create Clarify job
    response = sagemaker_client.create_processing_job(**clarify_job_config)
    
    # Save Clarify job info
    clarify_info = {
        'endpoint_name': args.endpoint_name,
        'clarify_job_name': args.clarify_job_name,
        'clarify_job_arn': response['ProcessingJobArn'],
        'output_s3_uri': f"s3://lucaskle-ab3-project-pv/clarify/output/{args.endpoint_name}/",
        'creation_time': datetime.now().isoformat()
    }
    
    with open(args.output, 'w') as f:
        json.dump(clarify_info, f, indent=2)
    
    logger.info(f"Clarify job info saved to {args.output}")
    return clarify_info

def main():
    """Main function."""
    args = parse_args()
    setup_clarify_explainability(args)

if __name__ == '__main__':
    main()