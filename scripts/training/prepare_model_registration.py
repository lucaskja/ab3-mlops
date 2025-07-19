#!/usr/bin/env python3
"""
Script to prepare model registration information for the SageMaker Model Registry.
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
    parser = argparse.ArgumentParser(description='Prepare model for registration in SageMaker Model Registry')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--model-info', type=str, required=True, help='Path to model info JSON file')
    parser.add_argument('--evaluation-results', type=str, required=True, help='Path to evaluation results JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    return parser.parse_args()

def prepare_model_registration(args):
    """Prepare model registration information."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    
    # Load model info and evaluation results
    with open(args.model_info, 'r') as f:
        model_info = json.load(f)
    
    with open(args.evaluation_results, 'r') as f:
        evaluation_results = json.load(f)
    
    # Prepare model package info
    model_package_info = {
        'model_name': model_info['model_name'],
        'model_artifact_path': model_info['model_artifact_path'],
        'inference_image': model_info.get('inference_image', 'amazonaws.com/sagemaker-pytorch:1.12.0-gpu-py38'),
        'model_metrics': {
            'mAP_0.5': {
                'value': evaluation_results['mAP_0.5'],
                'standard_deviation': 0.0
            },
            'precision': {
                'value': evaluation_results['precision'],
                'standard_deviation': 0.0
            },
            'recall': {
                'value': evaluation_results['recall'],
                'standard_deviation': 0.0
            }
        },
        'approval_status': 'PendingManualApproval',
        'created_at': datetime.now().isoformat()
    }
    
    # Save model package info
    with open(args.output, 'w') as f:
        json.dump(model_package_info, f, indent=2)
    
    logger.info(f"Model package info saved to {args.output}")
    return model_package_info

def main():
    """Main function."""
    args = parse_args()
    prepare_model_registration(args)

if __name__ == '__main__':
    main()