#!/usr/bin/env python3
"""
Setup script for MLOps SageMaker Demo.

This script sets up the necessary resources for the MLOps SageMaker Demo,
including S3 buckets, IAM roles, and sample data.
"""

import os
import sys
import argparse
import boto3
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.project_config import PROJECT_NAME, DATA_BUCKET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup_demo')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup MLOps SageMaker Demo')
    parser.add_argument('--profile', type=str, default='ab', help='AWS CLI profile to use')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region to use')
    parser.add_argument('--create-bucket', action='store_true', help='Create S3 bucket')
    parser.add_argument('--upload-sample-data', action='store_true', help='Upload sample data to S3')
    parser.add_argument('--deploy-iam-roles', action='store_true', help='Deploy IAM roles')
    parser.add_argument('--all', action='store_true', help='Perform all setup steps')
    return parser.parse_args()

def create_s3_bucket(session, bucket_name, region):
    """Create S3 bucket for the demo."""
    logger.info(f"Creating S3 bucket: {bucket_name}")
    s3_client = session.client('s3')
    
    try:
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        # Enable versioning
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        # Add tags
        s3_client.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={
                'TagSet': [
                    {'Key': 'Project', 'Value': PROJECT_NAME},
                    {'Key': 'Environment', 'Value': 'demo'},
                    {'Key': 'CreatedBy', 'Value': 'setup_demo.py'}
                ]
            }
        )
        
        logger.info(f"Successfully created S3 bucket: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error creating S3 bucket: {e}")
        return False

def upload_sample_data(session, bucket_name):
    """Upload sample data to S3 bucket."""
    logger.info(f"Uploading sample data to S3 bucket: {bucket_name}")
    s3_client = session.client('s3')
    
    # Define local sample data directory
    sample_data_dir = Path(__file__).resolve().parent.parent.parent / 'sample_data'
    
    if not sample_data_dir.exists():
        logger.error(f"Sample data directory not found: {sample_data_dir}")
        return False
    
    try:
        # Create directory structure in S3
        directories = ['raw-images', 'labeled-data', 'training-data', 'model-artifacts', 'baseline-data']
        
        for directory in directories:
            logger.info(f"Creating directory: {directory}/")
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{directory}/",
                Body=''
            )
        
        # Upload sample images
        images_dir = sample_data_dir / 'images'
        if images_dir.exists():
            for image_file in images_dir.glob('*.jpg'):
                key = f"raw-images/{image_file.name}"
                logger.info(f"Uploading: {key}")
                s3_client.upload_file(
                    Filename=str(image_file),
                    Bucket=bucket_name,
                    Key=key
                )
        
        # Upload sample labeled data
        labels_dir = sample_data_dir / 'labels'
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.json'):
                key = f"labeled-data/sample-job/output/{label_file.name}"
                logger.info(f"Uploading: {key}")
                s3_client.upload_file(
                    Filename=str(label_file),
                    Bucket=bucket_name,
                    Key=key
                )
        
        logger.info(f"Successfully uploaded sample data to S3 bucket: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading sample data: {e}")
        return False

def deploy_iam_roles(session):
    """Deploy IAM roles using CloudFormation."""
    logger.info("Deploying IAM roles using CloudFormation")
    
    # Define CloudFormation template path
    cf_template_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'iam-roles-cloudformation.yaml'
    
    if not cf_template_path.exists():
        logger.error(f"CloudFormation template not found: {cf_template_path}")
        return False
    
    try:
        # Read CloudFormation template
        with open(cf_template_path, 'r') as f:
            template_body = f.read()
        
        # Create CloudFormation stack
        cf_client = session.client('cloudformation')
        stack_name = f"{PROJECT_NAME}-iam-roles"
        
        cf_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=[
                {'ParameterKey': 'ProjectName', 'ParameterValue': PROJECT_NAME},
                {'ParameterKey': 'DataBucketName', 'ParameterValue': DATA_BUCKET}
            ],
            Capabilities=['CAPABILITY_NAMED_IAM'],
            Tags=[
                {'Key': 'Project', 'Value': PROJECT_NAME},
                {'Key': 'Environment', 'Value': 'demo'},
                {'Key': 'CreatedBy', 'Value': 'setup_demo.py'}
            ]
        )
        
        logger.info(f"CloudFormation stack creation initiated: {stack_name}")
        
        # Wait for stack creation to complete
        logger.info("Waiting for stack creation to complete...")
        waiter = cf_client.get_waiter('stack_create_complete')
        waiter.wait(StackName=stack_name)
        
        logger.info(f"Successfully deployed IAM roles using CloudFormation: {stack_name}")
        return True
    except Exception as e:
        logger.error(f"Error deploying IAM roles: {e}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Create AWS session
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    
    # Print session info
    sts_client = session.client('sts')
    caller_identity = sts_client.get_caller_identity()
    logger.info(f"AWS Account ID: {caller_identity['Account']}")
    logger.info(f"AWS Region: {args.region}")
    
    # Perform setup steps
    if args.create_bucket or args.all:
        create_s3_bucket(session, DATA_BUCKET, args.region)
    
    if args.upload_sample_data or args.all:
        upload_sample_data(session, DATA_BUCKET)
    
    if args.deploy_iam_roles or args.all:
        deploy_iam_roles(session)
    
    logger.info("Setup complete!")

if __name__ == '__main__':
    main()