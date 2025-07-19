#!/usr/bin/env python3
"""
Script to get the latest approved model from the SageMaker Model Registry.
"""

import argparse
import json
import boto3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Get latest approved model from SageMaker Model Registry')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--model-package-group-name', type=str, required=True, help='Name of the model package group')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    return parser.parse_args()

def get_approved_model(args):
    """Get the latest approved model from the Model Registry."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    sagemaker_client = session.client('sagemaker')
    
    # List model packages
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=args.model_package_group_name,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10
    )
    
    if not response['ModelPackageSummaryList']:
        logger.error(f"No approved models found in model package group {args.model_package_group_name}")
        raise ValueError(f"No approved models found in model package group {args.model_package_group_name}")
    
    # Get the latest approved model
    latest_model = response['ModelPackageSummaryList'][0]
    model_package_arn = latest_model['ModelPackageArn']
    
    # Get model package details
    model_details = sagemaker_client.describe_model_package(
        ModelPackageName=model_package_arn
    )
    
    # Extract relevant information
    model_info = {
        'model_package_arn': model_package_arn,
        'model_name': model_details['ModelPackageName'].split('/')[-1],
        'model_artifact_path': model_details['InferenceSpecification']['Containers'][0]['ModelDataUrl'],
        'inference_image': model_details['InferenceSpecification']['Containers'][0]['Image'],
        'creation_time': model_details['CreationTime'].isoformat(),
        'model_metrics': {
            metric['Name']: {
                'value': metric['Value'],
                'standard_deviation': metric.get('StandardDeviation', 0.0)
            }
            for metric in model_details.get('ModelMetrics', {}).get('ModelQuality', {}).get('Statistics', {}).get('ContentType', [])
        }
    }
    
    # Save model info
    with open(args.output, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Latest approved model info saved to {args.output}")
    return model_info

def main():
    """Main function."""
    args = parse_args()
    get_approved_model(args)

if __name__ == '__main__':
    main()