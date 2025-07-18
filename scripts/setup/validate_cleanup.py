#!/usr/bin/env python3
"""
Validation script for MLOps SageMaker Demo cleanup.

This script checks if any resources created by the MLOps SageMaker Demo still exist
and provides a report of remaining resources.
"""

import os
import sys
import argparse
import boto3
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.project_config import PROJECT_NAME, DATA_BUCKET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validate_cleanup')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate MLOps SageMaker Demo cleanup')
    parser.add_argument('--profile', type=str, default='ab', help='AWS CLI profile to use')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region to use')
    parser.add_argument('--output', type=str, choices=['text', 'json', 'csv'], default='text', help='Output format')
    parser.add_argument('--output-file', type=str, help='Output file path')
    return parser.parse_args()

def check_sagemaker_endpoints(session):
    """Check for SageMaker endpoints."""
    sagemaker_client = session.client('sagemaker')
    
    try:
        response = sagemaker_client.list_endpoints()
        endpoints = [endpoint for endpoint in response['Endpoints'] 
                    if PROJECT_NAME.lower() in endpoint['EndpointName'].lower()]
        
        return [
            {
                'resource_type': 'SageMaker Endpoint',
                'name': endpoint['EndpointName'],
                'arn': endpoint['EndpointArn'],
                'status': endpoint['EndpointStatus'],
                'created': endpoint['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for endpoint in endpoints
        ]
    except Exception as e:
        logger.error(f"Error checking SageMaker endpoints: {e}")
        return []

def check_sagemaker_models(session):
    """Check for SageMaker models."""
    sagemaker_client = session.client('sagemaker')
    
    try:
        response = sagemaker_client.list_models()
        models = [model for model in response['Models'] 
                 if PROJECT_NAME.lower() in model['ModelName'].lower()]
        
        return [
            {
                'resource_type': 'SageMaker Model',
                'name': model['ModelName'],
                'arn': model['ModelArn'],
                'created': model['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for model in models
        ]
    except Exception as e:
        logger.error(f"Error checking SageMaker models: {e}")
        return []

def check_sagemaker_pipelines(session):
    """Check for SageMaker pipelines."""
    sagemaker_client = session.client('sagemaker')
    
    try:
        response = sagemaker_client.list_pipelines()
        pipelines = [pipeline for pipeline in response['PipelineSummaries'] 
                    if PROJECT_NAME.lower() in pipeline['PipelineName'].lower()]
        
        return [
            {
                'resource_type': 'SageMaker Pipeline',
                'name': pipeline['PipelineName'],
                'arn': pipeline['PipelineArn'],
                'created': pipeline['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for pipeline in pipelines
        ]
    except Exception as e:
        logger.error(f"Error checking SageMaker pipelines: {e}")
        return []

def check_monitoring_schedules(session):
    """Check for monitoring schedules."""
    sagemaker_client = session.client('sagemaker')
    
    try:
        response = sagemaker_client.list_monitoring_schedules()
        schedules = [schedule for schedule in response['MonitoringScheduleSummaries'] 
                    if PROJECT_NAME.lower() in schedule['MonitoringScheduleName'].lower()]
        
        return [
            {
                'resource_type': 'Monitoring Schedule',
                'name': schedule['MonitoringScheduleName'],
                'status': schedule['MonitoringScheduleStatus'],
                'created': schedule['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for schedule in schedules
        ]
    except Exception as e:
        logger.error(f"Error checking monitoring schedules: {e}")
        return []

def check_cloudwatch_dashboards(session):
    """Check for CloudWatch dashboards."""
    cloudwatch_client = session.client('cloudwatch')
    
    try:
        response = cloudwatch_client.list_dashboards()
        dashboards = [dashboard for dashboard in response['DashboardEntries'] 
                     if PROJECT_NAME.lower() in dashboard['DashboardName'].lower()]
        
        return [
            {
                'resource_type': 'CloudWatch Dashboard',
                'name': dashboard['DashboardName'],
                'arn': dashboard['DashboardArn'],
                'last_modified': dashboard['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for dashboard in dashboards
        ]
    except Exception as e:
        logger.error(f"Error checking CloudWatch dashboards: {e}")
        return []

def check_cloudwatch_alarms(session):
    """Check for CloudWatch alarms."""
    cloudwatch_client = session.client('cloudwatch')
    
    try:
        response = cloudwatch_client.describe_alarms()
        alarms = [alarm for alarm in response['MetricAlarms'] 
                 if PROJECT_NAME.lower() in alarm['AlarmName'].lower()]
        
        return [
            {
                'resource_type': 'CloudWatch Alarm',
                'name': alarm['AlarmName'],
                'arn': alarm['AlarmArn'],
                'state': alarm['StateValue']
            }
            for alarm in alarms
        ]
    except Exception as e:
        logger.error(f"Error checking CloudWatch alarms: {e}")
        return []

def check_eventbridge_rules(session):
    """Check for EventBridge rules."""
    events_client = session.client('events')
    
    try:
        response = events_client.list_rules()
        rules = [rule for rule in response['Rules'] 
                if PROJECT_NAME.lower() in rule['Name'].lower()]
        
        return [
            {
                'resource_type': 'EventBridge Rule',
                'name': rule['Name'],
                'arn': rule['Arn'],
                'state': rule['State']
            }
            for rule in rules
        ]
    except Exception as e:
        logger.error(f"Error checking EventBridge rules: {e}")
        return []

def check_cloudformation_stacks(session):
    """Check for CloudFormation stacks."""
    cf_client = session.client('cloudformation')
    
    try:
        response = cf_client.list_stacks(StackStatusFilter=['CREATE_COMPLETE', 'UPDATE_COMPLETE', 'ROLLBACK_COMPLETE'])
        stacks = [stack for stack in response['StackSummaries'] 
                 if PROJECT_NAME.lower() in stack['StackName'].lower()]
        
        return [
            {
                'resource_type': 'CloudFormation Stack',
                'name': stack['StackName'],
                'id': stack['StackId'],
                'status': stack['StackStatus'],
                'created': stack['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for stack in stacks
        ]
    except Exception as e:
        logger.error(f"Error checking CloudFormation stacks: {e}")
        return []

def check_s3_bucket(session, bucket_name):
    """Check S3 bucket contents."""
    s3_client = session.client('s3')
    
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        
        # List objects
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        object_count = 0
        total_size = 0
        
        for page in pages:
            if 'Contents' in page:
                object_count += len(page['Contents'])
                total_size += sum(obj['Size'] for obj in page['Contents'])
        
        if object_count > 0:
            return [
                {
                    'resource_type': 'S3 Bucket',
                    'name': bucket_name,
                    'status': f'Contains {object_count} objects ({total_size / (1024 * 1024):.2f} MB)',
                    'created': 'N/A'
                }
            ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error checking S3 bucket: {e}")
        return []

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
    
    # Check resources
    logger.info("Checking for remaining resources...")
    
    resources = []
    resources.extend(check_sagemaker_endpoints(session))
    resources.extend(check_sagemaker_models(session))
    resources.extend(check_sagemaker_pipelines(session))
    resources.extend(check_monitoring_schedules(session))
    resources.extend(check_cloudwatch_dashboards(session))
    resources.extend(check_cloudwatch_alarms(session))
    resources.extend(check_eventbridge_rules(session))
    resources.extend(check_cloudformation_stacks(session))
    resources.extend(check_s3_bucket(session, DATA_BUCKET))
    
    # Generate report
    if not resources:
        logger.info("No resources found. Cleanup is complete!")
        return
    
    logger.info(f"Found {len(resources)} remaining resources:")
    
    # Create DataFrame
    df = pd.DataFrame(resources)
    
    # Output report
    if args.output == 'json':
        report = df.to_json(orient='records', indent=2)
        print(report)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(report)
    elif args.output == 'csv':
        report = df.to_csv(index=False)
        print(report)
        if args.output_file:
            df.to_csv(args.output_file, index=False)
    else:  # text
        report = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        print(report)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(report)
    
    # Print summary
    resource_types = df['resource_type'].value_counts().to_dict()
    print("\nSummary:")
    for resource_type, count in resource_types.items():
        print(f"- {resource_type}: {count}")
    
    print("\nCleanup is not complete. Please delete the remaining resources.")

if __name__ == '__main__':
    main()