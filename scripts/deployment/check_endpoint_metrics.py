#!/usr/bin/env python3
"""
Script to check metrics for a SageMaker endpoint.
"""

import argparse
import json
import boto3
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check metrics for SageMaker endpoint')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--endpoint-name', type=str, required=True, help='Name of the endpoint')
    parser.add_argument('--period', type=int, default=300, help='Period in seconds for metrics')
    parser.add_argument('--duration', type=int, default=3600, help='Duration in seconds to look back')
    parser.add_argument('--output', type=str, default='endpoint_metrics.json', help='Path to output JSON file')
    return parser.parse_args()

def check_endpoint_metrics(args):
    """Check metrics for SageMaker endpoint."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    cloudwatch = session.client('cloudwatch')
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=args.duration)
    
    # Get metrics
    metrics = {}
    
    # Invocation metrics
    invocation_metrics = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'invocations',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'Invocations',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': args.endpoint_name
                            },
                            {
                                'Name': 'VariantName',
                                'Value': 'AllTraffic'
                            }
                        ]
                    },
                    'Period': args.period,
                    'Stat': 'Sum'
                },
                'ReturnData': True
            },
            {
                'Id': 'invocation_errors',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'ModelLatency',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': args.endpoint_name
                            },
                            {
                                'Name': 'VariantName',
                                'Value': 'AllTraffic'
                            }
                        ]
                    },
                    'Period': args.period,
                    'Stat': 'Average'
                },
                'ReturnData': True
            },
            {
                'Id': 'model_latency',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'ModelLatency',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': args.endpoint_name
                            },
                            {
                                'Name': 'VariantName',
                                'Value': 'AllTraffic'
                            }
                        ]
                    },
                    'Period': args.period,
                    'Stat': 'p95'
                },
                'ReturnData': True
            },
            {
                'Id': 'overhead_latency',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'OverheadLatency',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': args.endpoint_name
                            },
                            {
                                'Name': 'VariantName',
                                'Value': 'AllTraffic'
                            }
                        ]
                    },
                    'Period': args.period,
                    'Stat': 'p95'
                },
                'ReturnData': True
            }
        ],
        StartTime=start_time,
        EndTime=end_time
    )
    
    # Process metrics
    total_invocations = sum(invocation_metrics['MetricDataResults'][0]['Values']) if invocation_metrics['MetricDataResults'][0]['Values'] else 0
    total_errors = sum(invocation_metrics['MetricDataResults'][1]['Values']) if invocation_metrics['MetricDataResults'][1]['Values'] else 0
    error_rate = total_errors / total_invocations if total_invocations > 0 else 0
    
    model_latency_values = invocation_metrics['MetricDataResults'][2]['Values']
    model_latency_p95 = max(model_latency_values) if model_latency_values else 0
    
    overhead_latency_values = invocation_metrics['MetricDataResults'][3]['Values']
    overhead_latency_p95 = max(overhead_latency_values) if overhead_latency_values else 0
    
    # Compile metrics
    metrics = {
        'endpoint_name': args.endpoint_name,
        'total_invocations': total_invocations,
        'total_errors': total_errors,
        'error_rate': error_rate,
        'latency_p95': model_latency_p95,
        'overhead_latency_p95': overhead_latency_p95,
        'time_period': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'period_seconds': args.period,
            'duration_seconds': args.duration
        }
    }
    
    # Save metrics
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Endpoint metrics saved to {args.output}")
    return metrics

def main():
    """Main function."""
    args = parse_args()
    check_endpoint_metrics(args)

if __name__ == '__main__':
    main()