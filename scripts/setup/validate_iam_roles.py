#!/usr/bin/env python3
"""
IAM Role Validation Script for MLOps SageMaker Demo
Tests role-based access controls and governance policies
"""

import boto3
import json
import sys
from typing import Dict, List, Tuple, Any
from botocore.exceptions import ClientError, NoCredentialsError
import argparse
from datetime import datetime


class IAMRoleValidator:
    """Validates IAM roles and policies for MLOps governance"""
    
    def __init__(self, profile_name: str = "ab"):
        """Initialize the validator with AWS profile"""
        try:
            self.session = boto3.Session(profile_name=profile_name)
            self.iam_client = self.session.client('iam')
            self.sts_client = self.session.client('sts')
            self.sagemaker_client = self.session.client('sagemaker')
            self.s3_client = self.session.client('s3')
            
            # Get current account info
            self.account_id = self.sts_client.get_caller_identity()['Account']
            self.region = self.session.region_name or 'us-east-1'
            
            print(f"✓ Initialized validator for account {self.account_id} in region {self.region}")
            
        except NoCredentialsError:
            print(f"✗ Error: AWS credentials not found for profile '{profile_name}'")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error initializing AWS clients: {str(e)}")
            sys.exit(1)
    
    def validate_role_exists(self, role_name: str) -> bool:
        """Check if an IAM role exists"""
        try:
            self.iam_client.get_role(RoleName=role_name)
            print(f"✓ Role '{role_name}' exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                print(f"✗ Role '{role_name}' does not exist")
                return False
            else:
                print(f"✗ Error checking role '{role_name}': {str(e)}")
                return False
    
    def validate_role_policies(self, role_name: str, expected_policies: List[str]) -> bool:
        """Validate that a role has the expected managed policies attached"""
        try:
            response = self.iam_client.list_attached_role_policies(RoleName=role_name)
            attached_policies = [policy['PolicyName'] for policy in response['AttachedPolicies']]
            
            all_policies_found = True
            for policy in expected_policies:
                if policy in attached_policies:
                    print(f"✓ Role '{role_name}' has policy '{policy}' attached")
                else:
                    print(f"✗ Role '{role_name}' missing policy '{policy}'")
                    all_policies_found = False
            
            return all_policies_found
            
        except ClientError as e:
            print(f"✗ Error checking policies for role '{role_name}': {str(e)}")
            return False
    
    def validate_role_trust_policy(self, role_name: str, expected_principals: List[str]) -> bool:
        """Validate the trust policy of a role"""
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            trust_policy = response['Role']['AssumeRolePolicyDocument']
            
            # Extract principals from trust policy
            principals = []
            for statement in trust_policy.get('Statement', []):
                principal = statement.get('Principal', {})
                if isinstance(principal, dict):
                    for key, value in principal.items():
                        if isinstance(value, list):
                            principals.extend(value)
                        else:
                            principals.append(value)
            
            all_principals_found = True
            for expected_principal in expected_principals:
                if any(expected_principal in principal for principal in principals):
                    print(f"✓ Role '{role_name}' trusts '{expected_principal}'")
                else:
                    print(f"✗ Role '{role_name}' does not trust '{expected_principal}'")
                    all_principals_found = False
            
            return all_principals_found
            
        except ClientError as e:
            print(f"✗ Error checking trust policy for role '{role_name}': {str(e)}")
            return False
    
    def simulate_role_permissions(self, role_arn: str, actions: List[str], resources: List[str]) -> Dict[str, bool]:
        """Simulate IAM permissions for a role"""
        results = {}
        
        try:
            for action in actions:
                for resource in resources:
                    try:
                        response = self.iam_client.simulate_principal_policy(
                            PolicySourceArn=role_arn,
                            ActionNames=[action],
                            ResourceArns=[resource] if resource != '*' else []
                        )
                        
                        evaluation_results = response.get('EvaluationResults', [])
                        if evaluation_results:
                            decision = evaluation_results[0]['EvalDecision']
                            is_allowed = decision == 'allowed'
                            results[f"{action}:{resource}"] = is_allowed
                            
                            status = "✓" if is_allowed else "✗"
                            print(f"{status} {action} on {resource}: {decision}")
                        
                    except ClientError as e:
                        print(f"✗ Error simulating {action} on {resource}: {str(e)}")
                        results[f"{action}:{resource}"] = False
            
            return results
            
        except ClientError as e:
            print(f"✗ Error in permission simulation: {str(e)}")
            return {}
    
    def validate_data_scientist_role(self) -> bool:
        """Validate Data Scientist role permissions and restrictions"""
        print("\n=== Validating Data Scientist Role ===")
        
        role_name = "mlops-sagemaker-demo-DataScientist-Role"
        role_arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
        
        # Check if role exists
        if not self.validate_role_exists(role_name):
            return False
        
        # Check managed policies
        expected_policies = ["AmazonSageMakerReadOnly"]
        policies_valid = self.validate_role_policies(role_name, expected_policies)
        
        # Check trust policy
        expected_principals = ["sagemaker.amazonaws.com", f"arn:aws:iam::{self.account_id}:root"]
        trust_valid = self.validate_role_trust_policy(role_name, expected_principals)
        
        # Test permissions - should be allowed
        allowed_actions = [
            "s3:GetObject",
            "s3:ListBucket",
            "sagemaker:DescribeUserProfile",
            "sagemaker:CreatePresignedDomainUrl"
        ]
        
        allowed_resources = [
            "arn:aws:s3:::lucaskle-ab3-project-pv",
            "arn:aws:s3:::lucaskle-ab3-project-pv/*",
            "*"
        ]
        
        print("\nTesting allowed permissions:")
        allowed_results = self.simulate_role_permissions(role_arn, allowed_actions, allowed_resources)
        
        # Test permissions - should be denied
        denied_actions = [
            "sagemaker:CreateEndpoint",
            "sagemaker:CreateModel",
            "sagemaker:CreatePipeline",
            "sagemaker:StartPipelineExecution"
        ]
        
        denied_resources = ["*"]
        
        print("\nTesting denied permissions:")
        denied_results = self.simulate_role_permissions(role_arn, denied_actions, denied_resources)
        
        # Validate results
        allowed_count = sum(1 for result in allowed_results.values() if result)
        denied_count = sum(1 for result in denied_results.values() if not result)
        
        print(f"\nData Scientist Role Summary:")
        print(f"✓ Allowed permissions working: {allowed_count}/{len(allowed_results)}")
        print(f"✓ Denied permissions working: {denied_count}/{len(denied_results)}")
        
        return policies_valid and trust_valid and allowed_count > 0 and denied_count > 0
    
    def validate_ml_engineer_role(self) -> bool:
        """Validate ML Engineer role permissions"""
        print("\n=== Validating ML Engineer Role ===")
        
        role_name = "mlops-sagemaker-demo-MLEngineer-Role"
        role_arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
        
        # Check if role exists
        if not self.validate_role_exists(role_name):
            return False
        
        # Check managed policies
        expected_policies = ["AmazonSageMakerFullAccess", "AmazonS3FullAccess"]
        policies_valid = self.validate_role_policies(role_name, expected_policies)
        
        # Check trust policy
        expected_principals = ["sagemaker.amazonaws.com", f"arn:aws:iam::{self.account_id}:root"]
        trust_valid = self.validate_role_trust_policy(role_name, expected_principals)
        
        # Test full access permissions
        full_access_actions = [
            "sagemaker:CreateEndpoint",
            "sagemaker:CreateModel",
            "sagemaker:CreatePipeline",
            "sagemaker:StartPipelineExecution",
            "events:PutEvents",
            "cloudwatch:PutMetricData"
        ]
        
        resources = ["*"]
        
        print("\nTesting full access permissions:")
        results = self.simulate_role_permissions(role_arn, full_access_actions, resources)
        
        allowed_count = sum(1 for result in results.values() if result)
        
        print(f"\nML Engineer Role Summary:")
        print(f"✓ Full access permissions working: {allowed_count}/{len(results)}")
        
        return policies_valid and trust_valid and allowed_count > 0
    
    def validate_sagemaker_execution_role(self) -> bool:
        """Validate SageMaker execution role"""
        print("\n=== Validating SageMaker Execution Role ===")
        
        role_name = "mlops-sagemaker-demo-SageMaker-Execution-Role"
        
        # Check if role exists
        if not self.validate_role_exists(role_name):
            return False
        
        # Check managed policies
        expected_policies = ["AmazonSageMakerFullAccess"]
        policies_valid = self.validate_role_policies(role_name, expected_policies)
        
        # Check trust policy - should only trust SageMaker service
        expected_principals = ["sagemaker.amazonaws.com"]
        trust_valid = self.validate_role_trust_policy(role_name, expected_principals)
        
        return policies_valid and trust_valid
    
    def validate_sagemaker_studio_policy(self) -> bool:
        """Validate SageMaker Studio managed policy"""
        print("\n=== Validating SageMaker Studio Policy ===")
        
        policy_name = "mlops-sagemaker-demo-SageMaker-Studio-Policy"
        
        try:
            response = self.iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{self.account_id}:policy/{policy_name}"
            )
            print(f"✓ Policy '{policy_name}' exists")
            
            # Get policy version
            policy_version = response['Policy']['DefaultVersionId']
            version_response = self.iam_client.get_policy_version(
                PolicyArn=f"arn:aws:iam::{self.account_id}:policy/{policy_name}",
                VersionId=policy_version
            )
            
            policy_document = version_response['PolicyVersion']['Document']
            
            # Check for key statements
            statements = policy_document.get('Statement', [])
            has_studio_permissions = any(
                'sagemaker:CreatePresignedDomainUrl' in stmt.get('Action', [])
                for stmt in statements
            )
            
            has_instance_restrictions = any(
                'sagemaker:InstanceTypes' in str(stmt.get('Condition', {}))
                for stmt in statements
            )
            
            if has_studio_permissions:
                print("✓ Policy contains SageMaker Studio permissions")
            else:
                print("✗ Policy missing SageMaker Studio permissions")
            
            if has_instance_restrictions:
                print("✓ Policy contains instance type restrictions")
            else:
                print("✗ Policy missing instance type restrictions")
            
            return has_studio_permissions and has_instance_restrictions
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                print(f"✗ Policy '{policy_name}' does not exist")
            else:
                print(f"✗ Error checking policy '{policy_name}': {str(e)}")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete validation of all IAM roles and policies"""
        print(f"Starting IAM Role Validation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        results = []
        
        # Validate each component
        results.append(self.validate_data_scientist_role())
        results.append(self.validate_ml_engineer_role())
        results.append(self.validate_sagemaker_execution_role())
        results.append(self.validate_sagemaker_studio_policy())
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(results)
        total = len(results)
        
        print(f"✓ Passed: {passed}/{total} validations")
        
        if passed == total:
            print("🎉 All IAM roles and policies are properly configured!")
            return True
        else:
            print("❌ Some validations failed. Please check the configuration.")
            return False


def main():
    """Main function to run IAM role validation"""
    parser = argparse.ArgumentParser(description='Validate IAM roles for MLOps SageMaker Demo')
    parser.add_argument('--profile', default='ab', help='AWS CLI profile to use (default: ab)')
    parser.add_argument('--role', choices=['data-scientist', 'ml-engineer', 'execution', 'studio-policy', 'all'], 
                       default='all', help='Specific role to validate (default: all)')
    
    args = parser.parse_args()
    
    validator = IAMRoleValidator(profile_name=args.profile)
    
    if args.role == 'all':
        success = validator.run_full_validation()
    elif args.role == 'data-scientist':
        success = validator.validate_data_scientist_role()
    elif args.role == 'ml-engineer':
        success = validator.validate_ml_engineer_role()
    elif args.role == 'execution':
        success = validator.validate_sagemaker_execution_role()
    elif args.role == 'studio-policy':
        success = validator.validate_sagemaker_studio_policy()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()