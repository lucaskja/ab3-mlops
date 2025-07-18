#!/usr/bin/env python3
"""
Teardown script for MLOps SageMaker Demo.

This script cleans up resources created for the MLOps SageMaker Demo,
including SageMaker endpoints, pipelines, and CloudFormation stacks.
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
logger = logging.getLogger('teardown_demo')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Teardown MLOps SageMaker Demo')
    parser.add_argument('--profile', type=str, default='ab', help='AWS CLI profile to use')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region to use')
    parser.add_argument('--delete-endpoints', action='store_true', help='Delete SageMaker endpoints')
    parser.add_argument('--delete-pipelines', action='store_true', help='Delete SageMaker pipelines')
    parser.add_argument('--delete-monitoring', action='store_true', help='Delete monitoring schedules')
    parser.add_argument('--delete-iam-roles', action='store_true', help='Delete IAM roles')
    parser.add_argument('--empty-bucket', action='store_true', help='Empty S3 bucket')
    parser.add_argument('--delete-bucket', action='store_true', help='Delete S3 bucket')
    parser.add_argument('--all', action='store_true', help='Perform all teardown steps')
    parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    return parser.parse_args()

def delete_sagemaker_endpoints(session):
    """Delete all SageMaker endpoints."""
    logger.info("Deleting SageMaker endpoints")
    sagemaker_client = session.client('sagemaker')
    
    try:
        # List all endpoints
        response = sagemaker_client.list_endpoints()
        endpoints = response['Endpoints']
        
        if not endpoints:
            logger.info("No SageMaker endpoints found")
            return True
        
        # Delete each endpoint
        for endpoint in endpoints:
            endpoint_name = endpoint['EndpointName']
            logger.info(f"Deleting endpoint: {endpoint_name}")
            
            try:
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                logger.info(f"Successfully deleted endpoint: {endpoint_name}")
            except Exception as e:
                logger.error(f"Error deleting endpoint {endpoint_name}: {e}")
        
        # List all endpoint configs
        response = sagemaker_client.list_endpoint_configs()
        endpoint_configs = response['EndpointConfigs']
        
        # Delete each endpoint config
        for config in endpoint_configs:
            config_name = config['EndpointConfigName']
            logger.info(f"Deleting endpoint config: {config_name}")
            
            try:
                sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                logger.info(f"Successfully deleted endpoint config: {config_name}")
            except Exception as e:
                logger.error(f"Error deleting endpoint config {config_name}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error deleting SageMaker endpoints: {e}")
        return False

def delete_sagemaker_pipelines(session):
    """Delete all SageMaker pipelines."""
    logger.info("Deleting SageMaker pipelines")
    sagemaker_client = session.client('sagemaker')
    
    try:
        # List all pipelines
        response = sagemaker_client.list_pipelines()
        pipelines = response['PipelineSummaries']
        
        if not pipelines:
            logger.info("No SageMaker pipelines found")
            return True
        
        # Delete each pipeline
        for pipeline in pipelines:
            pipeline_name = pipeline['PipelineName']
            logger.info(f"Deleting pipeline: {pipeline_name}")
            
            try:
                sagemaker_client.delete_pipeline(PipelineName=pipeline_name)
                logger.info(f"Successfully deleted pipeline: {pipeline_name}")
            except Exception as e:
                logger.error(f"Error deleting pipeline {pipeline_name}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error deleting SageMaker pipelines: {e}")
        return False

def delete_monitoring_schedules(session):
    """Delete all monitoring schedules."""
    logger.info("Deleting monitoring schedules")
    sagemaker_client = session.client('sagemaker')
    
    try:
        # List all monitoring schedules
        response = sagemaker_client.list_monitoring_schedules()
        schedules = response['MonitoringScheduleSummaries']
        
        if not schedules:
            logger.info("No monitoring schedules found")
            return True
        
        # Delete each monitoring schedule
        for schedule in schedules:
            schedule_name = schedule['MonitoringScheduleName']
            logger.info(f"Deleting monitoring schedule: {schedule_name}")
            
            try:
                sagemaker_client.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
                logger.info(f"Successfully deleted monitoring schedule: {schedule_name}")
            except Exception as e:
                logger.error(f"Error deleting monitoring schedule {schedule_name}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error deleting monitoring schedules: {e}")
        return False

def delete_iam_roles(session):
    """Delete IAM roles using CloudFormation."""
    logger.info("Deleting IAM roles using CloudFormation")
    cf_client = session.client('cloudformation')
    
    try:
        # Delete CloudFormation stack
        stack_name = f"{PROJECT_NAME}-iam-roles"
        logger.info(f"Deleting CloudFormation stack: {stack_name}")
        
        cf_client.delete_stack(StackName=stack_name)
        
        # Wait for stack deletion to complete
        logger.info("Waiting for stack deletion to complete...")
        waiter = cf_client.get_waiter('stack_delete_complete')
        waiter.wait(StackName=stack_name)
        
        logger.info(f"Successfully deleted CloudFormation stack: {stack_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting IAM roles: {e}")
        return False

def empty_s3_bucket(session, bucket_name):
    """Empty S3 bucket."""
    logger.info(f"Emptying S3 bucket: {bucket_name}")
    s3_client = session.client('s3')
    
    try:
        # List all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        delete_list = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    delete_list.append({'Key': obj['Key']})
        
        if not delete_list:
            logger.info(f"S3 bucket {bucket_name} is already empty")
            return True
        
        # Delete objects in batches of 1000
        for i in range(0, len(delete_list), 1000):
            batch = delete_list[i:i+1000]
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': batch}
            )
            logger.info(f"Deleted {len(batch)} objects from S3 bucket")
        
        # List and delete all object versions
        paginator = s3_client.get_paginator('list_object_versions')
        pages = paginator.paginate(Bucket=bucket_name)
        
        delete_list = []
        for page in pages:
            if 'Versions' in page:
                for version in page['Versions']:
                    delete_list.append({
                        'Key': version['Key'],
                        'VersionId': version['VersionId']
                    })
            
            if 'DeleteMarkers' in page:
                for marker in page['DeleteMarkers']:
                    delete_list.append({
                        'Key': marker['Key'],
                        'VersionId': marker['VersionId']
                    })
        
        if delete_list:
            # Delete object versions in batches of 1000
            for i in range(0, len(delete_list), 1000):
                batch = delete_list[i:i+1000]
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': batch}
                )
                logger.info(f"Deleted {len(batch)} object versions from S3 bucket")
        
        logger.info(f"Successfully emptied S3 bucket: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error emptying S3 bucket: {e}")
        return False

def delete_s3_bucket(session, bucket_name):
    """Delete S3 bucket."""
    logger.info(f"Deleting S3 bucket: {bucket_name}")
    s3_client = session.client('s3')
    
    try:
        s3_client.delete_bucket(Bucket=bucket_name)
        logger.info(f"Successfully deleted S3 bucket: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting S3 bucket: {e}")
        return False

def confirm_teardown():
    """Confirm teardown with user."""
    print("\nWARNING: This will delete all resources created for the MLOps SageMaker Demo.")
    print("This action cannot be undone.")
    response = input("Are you sure you want to proceed? (y/n): ")
    return response.lower() == 'y'

def main():
    """Main function."""
    args = parse_args()
    
    # Confirm teardown unless --force is specified
    if not args.force and not confirm_teardown():
        logger.info("Teardown cancelled")
        return
    
    # Create AWS session
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    
    # Print session info
    sts_client = session.client('sts')
    caller_identity = sts_client.get_caller_identity()
    logger.info(f"AWS Account ID: {caller_identity['Account']}")
    logger.info(f"AWS Region: {args.region}")
    
    # Perform teardown steps
    if args.delete_endpoints or args.all:
        delete_sagemaker_endpoints(session)
    
    if args.delete_pipelines or args.all:
        delete_sagemaker_pipelines(session)
    
    if args.delete_monitoring or args.all:
        delete_monitoring_schedules(session)
    
    if args.delete_iam_roles or args.all:
        delete_iam_roles(session)
    
    if args.empty_bucket or args.all:
        empty_s3_bucket(session, DATA_BUCKET)
    
    if args.delete_bucket or args.all:
        delete_s3_bucket(session, DATA_BUCKET)
    
    logger.info("Teardown complete!")

if __name__ == '__main__':
    main()