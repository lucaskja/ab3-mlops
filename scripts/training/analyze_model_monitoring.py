#!/usr/bin/env python3
"""
SageMaker Model Monitor Analysis Script

This script analyzes monitoring results from SageMaker Model Monitor and generates reports.
It can be used to check for data drift and model quality issues.

Usage:
    python analyze_model_monitoring.py --endpoint-name <endpoint_name> [--days <days>]

Requirements addressed:
- 5.2: Analyze captured input data and predictions
- 5.3: Detect data drift and generate alerts
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add parent directory to path to import project modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pipeline.model_monitor import ModelMonitoringManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze SageMaker Model Monitor results")
    
    parser.add_argument("--endpoint-name", type=str, required=True,
                       help="Name of the SageMaker endpoint")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to analyze (default: 7)")
    parser.add_argument("--aws-profile", type=str, default="ab",
                       help="AWS profile to use")
    parser.add_argument("--region", type=str, default="us-east-1",
                       help="AWS region")
    parser.add_argument("--output-path", type=str,
                       help="S3 path for output report (optional)")
    
    return parser.parse_args()


def format_report_summary(report: Dict[str, Any]) -> str:
    """
    Format a monitoring report summary for display.
    
    Args:
        report: Monitoring report
        
    Returns:
        Formatted summary
    """
    summary = []
    
    # Add header
    summary.append("=" * 80)
    summary.append(f"Model Monitoring Report for Endpoint: {report['endpoint_name']}")
    summary.append("=" * 80)
    
    # Add report period
    start_time = report['report_period']['start_time']
    end_time = report['report_period']['end_time']
    summary.append(f"Report Period: {start_time} to {end_time}")
    summary.append("")
    
    # Add executions summary
    summary.append("Monitoring Executions:")
    summary.append("-" * 40)
    total_executions = report['executions_summary']['total_executions']
    summary.append(f"Total Executions: {total_executions}")
    for schedule_name, count in report['executions_summary']['executions_by_schedule'].items():
        summary.append(f"  - {schedule_name}: {count} executions")
    summary.append("")
    
    # Add violations summary
    summary.append("Violations Summary:")
    summary.append("-" * 40)
    total_violations = report['violations_summary']['total_violations']
    summary.append(f"Total Violations: {total_violations}")
    for schedule_name, count in report['violations_summary']['violations_by_schedule'].items():
        summary.append(f"  - {schedule_name}: {count} violations")
    summary.append("")
    
    # Add detailed violations if any
    if total_violations > 0:
        summary.append("Detailed Violations:")
        summary.append("-" * 40)
        for schedule_name, violations in report['detailed_violations'].items():
            if violations:
                summary.append(f"Schedule: {schedule_name}")
                for violation_record in violations:
                    summary.append(f"  - Execution Time: {violation_record['scheduled_time']}")
                    summary.append(f"    Violations Count: {violation_record['violations_count']}")
                    for violation in violation_record['violations'][:5]:  # Show first 5 violations
                        summary.append(f"    - Feature: {violation.get('feature_name')}")
                        summary.append(f"      Check Type: {violation.get('constraint_check_type')}")
                    if len(violation_record['violations']) > 5:
                        summary.append(f"    ... and {len(violation_record['violations']) - 5} more violations")
                summary.append("")
    
    # Add report location
    summary.append(f"Full Report: {report.get('report_s3_path', 'N/A')}")
    summary.append("=" * 80)
    
    return "\n".join(summary)


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        logger.info(f"Analyzing model monitoring for endpoint: {args.endpoint_name}")
        
        # Create model monitoring manager
        monitor_manager = ModelMonitoringManager(
            aws_profile=args.aws_profile,
            region=args.region
        )
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=args.days)
        
        # Generate monitoring report
        report = monitor_manager.generate_monitoring_report(
            endpoint_name=args.endpoint_name,
            start_time=start_time,
            end_time=end_time,
            output_path=args.output_path
        )
        
        # Print report summary
        summary = format_report_summary(report)
        print(summary)
        
        # Save summary to file
        summary_file = f"{args.endpoint_name}_monitoring_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error analyzing model monitoring: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()