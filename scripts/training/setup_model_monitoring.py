#!/usr/bin/env python3
"""
SageMaker Model Monitor Setup Script

This script demonstrates how to set up SageMaker Model Monitor for an existing endpoint.
It configures data quality monitoring, drift detection, and alerting.

Usage:
    python setup_model_monitoring.py --endpoint-name <endpoint_name> --baseline-dataset <s3_path> [--email <email>]

Requirements addressed:
- 5.1: Automatic configuration of SageMaker Model Monitor when a model is deployed
- 5.2: Capture input data and predictions for monitoring
- 5.3: Trigger alerts through EventBridge when data drift is detected
"""

import argparse
import json
import logging
import sys
from typing import List, Optional

# Add parent directory to path to import project modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pipeline.model_monitor import create_model_monitor_for_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up SageMaker Model Monitor for an endpoint")
    
    parser.add_argument("--endpoint-name", type=str, required=True,
                       help="Name of the SageMaker endpoint")
    parser.add_argument("--baseline-dataset", type=str, required=True,
                       help="S3 path to baseline dataset for monitoring")
    parser.add_argument("--email", type=str, action="append",
                       help="Email address for monitoring alerts (can be specified multiple times)")
    parser.add_argument("--aws-profile", type=str, default="ab",
                       help="AWS profile to use")
    parser.add_argument("--region", type=str, default="us-east-1",
                       help="AWS region")
    parser.add_argument("--enable-model-quality", action="store_true",
                       help="Enable model quality monitoring")
    parser.add_argument("--problem-type", type=str, choices=["regression", "binary_classification", "multiclass_classification"],
                       help="Problem type for model quality monitoring")
    parser.add_argument("--ground-truth-attribute", type=str,
                       help="Name of ground truth attribute for model quality monitoring")
    parser.add_argument("--inference-attribute", type=str,
                       help="Name of inference attribute for model quality monitoring")
    parser.add_argument("--lambda-target-arn", type=str,
                       help="ARN of Lambda function to trigger for drift detection")
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if args.enable_model_quality:
        if not args.problem_type:
            raise ValueError("--problem-type is required when --enable-model-quality is specified")
        if not args.ground_truth_attribute:
            raise ValueError("--ground-truth-attribute is required when --enable-model-quality is specified")
        if not args.inference_attribute:
            raise ValueError("--inference-attribute is required when --enable-model-quality is specified")


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        logger.info(f"Setting up model monitoring for endpoint: {args.endpoint_name}")
        logger.info(f"Using baseline dataset: {args.baseline_dataset}")
        
        # Set up model monitoring
        monitoring_config = create_model_monitor_for_endpoint(
            endpoint_name=args.endpoint_name,
            baseline_dataset=args.baseline_dataset,
            aws_profile=args.aws_profile,
            region=args.region,
            email_notifications=args.email,
            enable_model_quality=args.enable_model_quality,
            problem_type=args.problem_type,
            ground_truth_attribute=args.ground_truth_attribute,
            inference_attribute=args.inference_attribute,
            lambda_target_arn=args.lambda_target_arn
        )
        
        # Print monitoring configuration
        logger.info("Model monitoring setup complete")
        logger.info(f"Configuration: {json.dumps(monitoring_config, indent=2)}")
        
        # Save configuration to file
        output_file = f"{args.endpoint_name}_monitoring_config.json"
        with open(output_file, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        logger.info(f"Configuration saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error setting up model monitoring: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()