#!/usr/bin/env python3
"""
Setup SageMaker Clarify Explainability

This script sets up SageMaker Clarify for model explainability and bias detection
on an existing SageMaker endpoint.

Requirements addressed:
- 5.4: Generate monitoring reports accessible through SageMaker Clarify
"""

import argparse
import logging
import json
import sys
from typing import Dict, Any, List, Optional

from src.pipeline.model_monitor_integration import add_clarify_to_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup SageMaker Clarify for model explainability")
    
    parser.add_argument("--endpoint-name", type=str, required=True,
                       help="Name of the SageMaker endpoint")
    parser.add_argument("--baseline-dataset", type=str, required=True,
                       help="S3 path to baseline dataset")
    parser.add_argument("--target-column", type=str, required=True,
                       help="Name of the target column")
    parser.add_argument("--features", type=str, required=True,
                       help="Comma-separated list of feature names")
    parser.add_argument("--sensitive-columns", type=str, default=None,
                       help="Comma-separated list of sensitive column names (optional)")
    parser.add_argument("--aws-profile", type=str, default="ab",
                       help="AWS profile to use")
    parser.add_argument("--region", type=str, default="us-east-1",
                       help="AWS region")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Parse features and sensitive columns
        features = [f.strip() for f in args.features.split(",")]
        sensitive_columns = None
        if args.sensitive_columns:
            sensitive_columns = [f.strip() for f in args.sensitive_columns.split(",")]
        
        logger.info(f"Setting up SageMaker Clarify for endpoint: {args.endpoint_name}")
        logger.info(f"Baseline dataset: {args.baseline_dataset}")
        logger.info(f"Target column: {args.target_column}")
        logger.info(f"Features: {features}")
        logger.info(f"Sensitive columns: {sensitive_columns}")
        
        # Add Clarify to endpoint
        clarify_config = add_clarify_to_endpoint(
            endpoint_name=args.endpoint_name,
            baseline_dataset=args.baseline_dataset,
            target_column=args.target_column,
            features=features,
            sensitive_columns=sensitive_columns,
            aws_profile=args.aws_profile,
            region=args.region
        )
        
        # Print configuration
        logger.info("SageMaker Clarify setup completed successfully")
        logger.info(f"Explainability report: {clarify_config['explainability']['report_path']}")
        logger.info(f"Dashboard: {clarify_config['dashboard']['dashboard_name']}")
        
        # Print instructions
        print("\n=== SageMaker Clarify Setup Complete ===")
        print(f"Explainability report: {clarify_config['explainability']['report_path']}")
        print(f"CloudWatch dashboard: {clarify_config['dashboard']['dashboard_name']}")
        print("\nTo view the report:")
        print(f"  aws s3 cp {clarify_config['explainability']['report_path']} ./report.html --profile {args.aws_profile}")
        print("  open ./report.html")
        print("\nTo view the dashboard:")
        print(f"  Open AWS CloudWatch console and navigate to Dashboards > {clarify_config['dashboard']['dashboard_name']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error setting up SageMaker Clarify: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())