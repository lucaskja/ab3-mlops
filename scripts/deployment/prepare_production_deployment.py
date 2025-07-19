#!/usr/bin/env python3
"""
Script to prepare for production deployment after successful staging tests.
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
    parser = argparse.ArgumentParser(description='Prepare for production deployment')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--endpoint-metrics', type=str, required=True, help='Path to endpoint metrics JSON file')
    parser.add_argument('--output', type=str, default='production_deployment_info.json', help='Path to output JSON file')
    return parser.parse_args()

def prepare_production_deployment(args):
    """Prepare for production deployment."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    
    # Load endpoint metrics
    with open(args.endpoint_metrics, 'r') as f:
        endpoint_metrics = json.load(f)
    
    # Check if metrics meet production criteria
    is_ready_for_production = True
    reasons = []
    
    if endpoint_metrics['error_rate'] > 0.01:
        is_ready_for_production = False
        reasons.append(f"Error rate too high: {endpoint_metrics['error_rate']:.4f} > 0.01")
    
    if endpoint_metrics['latency_p95'] > 200:
        is_ready_for_production = False
        reasons.append(f"Latency too high: {endpoint_metrics['latency_p95']:.2f} ms > 200 ms")
    
    if endpoint_metrics['total_invocations'] < 100:
        is_ready_for_production = False
        reasons.append(f"Not enough invocations: {endpoint_metrics['total_invocations']} < 100")
    
    # Prepare production deployment info
    production_deployment_info = {
        'endpoint_name': endpoint_metrics['endpoint_name'],
        'is_ready_for_production': is_ready_for_production,
        'reasons': reasons,
        'metrics': endpoint_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save production deployment info
    with open(args.output, 'w') as f:
        json.dump(production_deployment_info, f, indent=2)
    
    logger.info(f"Production deployment info saved to {args.output}")
    
    if is_ready_for_production:
        logger.info("Endpoint is ready for production deployment")
    else:
        logger.warning("Endpoint is NOT ready for production deployment")
        for reason in reasons:
            logger.warning(f"  - {reason}")
    
    return production_deployment_info

def main():
    """Main function."""
    args = parse_args()
    prepare_production_deployment(args)

if __name__ == '__main__':
    main()