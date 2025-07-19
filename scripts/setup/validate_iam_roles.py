#!/usr/bin/env python3
"""
IAM Role Validation Script for MLOps SageMaker Demo
Tests role-based access controls and governance policies
"""

import boto3
import json
import sys
import argparse
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate IAM roles for MLOps SageMaker Demo')
    parser.add_argument('--profile', default='ab', help='AWS CLI profile to use (default: ab)')
    parser.add_argument('--project-name', default='mlops-sagemaker-demo', help='Project name (default: mlops-sagemaker-demo)')
    return parser.parse_args()

def validate_iam_roles(profile_name, project_name):
    """Validate IAM roles."""
    print(f"Starting IAM Role Validation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize AWS clients
        session = boto3.Session(profile_name=profile_name)
        iam_client = session.client('iam')
        sts_client = session.client('sts')
        
        # Get current account info
        account_id = sts_client.get_caller_identity()['Account']
        region = session.region_name or 'us-east-1'
        
        print(f"✓ Initialized validator for account {account_id} in region {region}")
        
        # Define roles to check
        roles_to_check = [
            f"{project_name}-DataScientist-Role",
            f"{project_name}-MLEngineer-Role",
            f"{project_name}-SageMaker-Execution-Role"
        ]
        
        # Check roles
        for role_name in roles_to_check:
            try:
                role = iam_client.get_role(RoleName=role_name)
                print(f"✓ Role '{role_name}' exists with ARN: {role['Role']['Arn']}")
            except iam_client.exceptions.NoSuchEntityException:
                print(f"✗ Role '{role_name}' does not exist")
            except Exception as e:
                print(f"✗ Error checking role '{role_name}': {str(e)}")
        
        # Check policy
        policy_name = f"{project_name}-SageMaker-Studio-Policy"
        try:
            policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
            policy = iam_client.get_policy(PolicyArn=policy_arn)
            print(f"✓ Policy '{policy_name}' exists with ARN: {policy_arn}")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"✗ Policy '{policy_name}' does not exist")
        except Exception as e:
            print(f"✗ Error checking policy '{policy_name}': {str(e)}")
        
        print("\n" + "=" * 60)
        print("IAM Role Validation Complete")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_args()
    success = validate_iam_roles(args.profile, args.project_name)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()