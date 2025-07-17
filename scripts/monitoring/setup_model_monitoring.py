#!/usr/bin/env python
"""
Script to set up SageMaker Model Monitor for an endpoint.

This script sets up data quality and model quality monitoring for a SageMaker endpoint,
including baseline creation, monitoring schedule configuration, and alert setup.

Usage:
    python setup_model_monitoring.py --endpoint-name my-endpoint --bucket my-bucket --profile ab
"""

import argparse
import boto3
import json
import logging
import os
import sys
import time
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.drift_detection import DriftDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up SageMaker Model Monitor for an endpoint")
    parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    parser.add_argument("--bucket", required=True, help="S3 bucket for monitoring data")
    parser.add_argument("--prefix", default="monitoring", help="S3 prefix for monitoring data")
    parser.add_argument("--baseline-dataset", required=True, help="S3 URI to the baseline dataset")
    parser.add_argument("--ground-truth-dataset", help="S3 URI to the ground truth dataset (for model quality)")
    parser.add_argument("--schedule", default="cron(0 * ? * * *)", help="Schedule expression (cron format)")
    parser.add_argument("--instance-type", default="ml.m5.xlarge", help="Instance type for monitoring jobs")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances for monitoring jobs")
    parser.add_argument("--sns-topic-arn", help="ARN of the SNS topic for notifications")
    parser.add_argument("--profile", default="ab", help="AWS profile name")
    parser.add_argument("--region", default="us-east-1", help="AWS region name")
    parser.add_argument("--data-quality", action="store_true", help="Set up data quality monitoring")
    parser.add_argument("--model-quality", action="store_true", help="Set up model quality monitoring")
    parser.add_argument("--problem-type", default="BinaryClassification", 
                        choices=["BinaryClassification", "MulticlassClassification", "Regression"],
                        help="Type of ML problem (for model quality)")
    return parser.parse_args()

def setup_data_quality_monitoring(args, drift_detector):
    """Set up data quality monitoring."""
    logger.info("Setting up data quality monitoring...")
    
    # Create data quality baseline
    baseline_job_name = drift_detector.create_data_quality_baseline(
        baseline_dataset=args.baseline_dataset,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
    
    # Wait for baseline job to complete
    logger.info(f"Waiting for baseline job {baseline_job_name} to complete...")
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    sagemaker_client = session.client('sagemaker')
    
    while True:
        response = sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=baseline_job_name
        )
        status = response['MonitoringScheduleStatus']
        if status == 'Scheduled':
            logger.info("Baseline job completed successfully")
            break
        elif status == 'Failed':
            logger.error("Baseline job failed")
            logger.error(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
            sys.exit(1)
        else:
            logger.info(f"Baseline job status: {status}")
            time.sleep(60)
    
    # Get baseline artifacts
    baseline_output_uri = f"s3://{args.bucket}/{args.prefix}/baselines/data-quality/{baseline_job_name}"
    baseline_constraints_uri = f"{baseline_output_uri}/constraints.json"
    baseline_statistics_uri = f"{baseline_output_uri}/statistics.json"
    
    # Create data quality monitoring schedule
    schedule_name = drift_detector.create_data_quality_monitoring_schedule(
        baseline_constraints_uri=baseline_constraints_uri,
        baseline_statistics_uri=baseline_statistics_uri,
        schedule_expression=args.schedule,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
    
    # Create drift alerts
    if args.sns_topic_arn:
        drift_detector.create_drift_alert(
            metric_name="feature_drift_score",
            threshold=0.7,
            sns_topic_arn=args.sns_topic_arn
        )
        
        drift_detector.create_drift_alert(
            metric_name="data_quality_violations",
            threshold=1,
            comparison_operator="GreaterThanOrEqualToThreshold",
            sns_topic_arn=args.sns_topic_arn
        )
    
    logger.info(f"Data quality monitoring setup complete. Schedule name: {schedule_name}")
    return schedule_name

def setup_model_quality_monitoring(args, drift_detector):
    """Set up model quality monitoring."""
    if not args.ground_truth_dataset:
        logger.error("Ground truth dataset is required for model quality monitoring")
        sys.exit(1)
    
    logger.info("Setting up model quality monitoring...")
    
    # Create model quality baseline
    baseline_job_name = drift_detector.create_model_quality_baseline(
        baseline_dataset=args.ground_truth_dataset,
        problem_type=args.problem_type,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
    
    # Wait for baseline job to complete
    logger.info(f"Waiting for baseline job {baseline_job_name} to complete...")
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    sagemaker_client = session.client('sagemaker')
    
    while True:
        response = sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=baseline_job_name
        )
        status = response['MonitoringScheduleStatus']
        if status == 'Scheduled':
            logger.info("Baseline job completed successfully")
            break
        elif status == 'Failed':
            logger.error("Baseline job failed")
            logger.error(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
            sys.exit(1)
        else:
            logger.info(f"Baseline job status: {status}")
            time.sleep(60)
    
    # Get baseline artifacts
    baseline_output_uri = f"s3://{args.bucket}/{args.prefix}/baselines/model-quality/{baseline_job_name}"
    baseline_constraints_uri = f"{baseline_output_uri}/constraints.json"
    baseline_statistics_uri = f"{baseline_output_uri}/statistics.json"
    
    # Create model quality monitoring schedule
    schedule_name = drift_detector.create_model_quality_monitoring_schedule(
        baseline_constraints_uri=baseline_constraints_uri,
        baseline_statistics_uri=baseline_statistics_uri,
        ground_truth_input=args.ground_truth_dataset,
        problem_type=args.problem_type,
        schedule_expression=args.schedule,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
    
    # Create drift alerts
    if args.sns_topic_arn:
        drift_detector.create_drift_alert(
            metric_name="model_quality_score",
            threshold=0.7,
            comparison_operator="LessThanThreshold",
            sns_topic_arn=args.sns_topic_arn
        )
        
        drift_detector.create_drift_alert(
            metric_name="model_quality_violations",
            threshold=1,
            comparison_operator="GreaterThanOrEqualToThreshold",
            sns_topic_arn=args.sns_topic_arn
        )
    
    logger.info(f"Model quality monitoring setup complete. Schedule name: {schedule_name}")
    return schedule_name

def main():
    """Main function."""
    args = parse_args()
    
    # Create drift detector
    drift_detector = DriftDetector(
        endpoint_name=args.endpoint_name,
        bucket=args.bucket,
        prefix=args.prefix,
        profile_name=args.profile,
        region_name=args.region
    )
    
    # Set up monitoring based on arguments
    if not args.data_quality and not args.model_quality:
        # Default to data quality if neither is specified
        args.data_quality = True
    
    # Set up data quality monitoring
    if args.data_quality:
        data_quality_schedule = setup_data_quality_monitoring(args, drift_detector)
    
    # Set up model quality monitoring
    if args.model_quality:
        model_quality_schedule = setup_model_quality_monitoring(args, drift_detector)
    
    logger.info("Model monitoring setup complete")

if __name__ == "__main__":
    main()