#!/usr/bin/env python3
"""
Validation script for core SageMaker infrastructure deployment.
This script checks that all required resources are properly deployed and configured.
"""

import argparse
import boto3
import logging
import sys
import json
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate core SageMaker infrastructure deployment')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--project-name', type=str, default='sagemaker-core-setup', help='Project name')
    parser.add_argument('--data-bucket', type=str, default='lucaskle-ab3-project-pv', help='Data bucket name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def setup_aws_session(profile_name, region_name):
    """Set up AWS session with the specified profile and region."""
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        # Verify that the session is valid by getting the caller identity
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        logger.info(f"AWS session established for account {identity['Account']} using profile '{profile_name}'")
        return session
    except Exception as e:
        logger.error(f"Failed to set up AWS session with profile '{profile_name}': {str(e)}")
        sys.exit(1)

def validate_iam_roles(session, project_name):
    """Validate that the required IAM roles exist and have the correct permissions."""
    logger.info("Validating IAM roles...")
    
    iam_client = session.client('iam')
    cloudformation_client = session.client('cloudformation')
    
    # Check if the CloudFormation stack exists
    stack_name = f"{project_name}-iam-roles"
    try:
        stack_response = cloudformation_client.describe_stacks(StackName=stack_name)
        stack = stack_response['Stacks'][0]
        logger.info(f"CloudFormation stack '{stack_name}' exists with status: {stack['StackStatus']}")
        
        # Check if the stack is in a good state
        if stack['StackStatus'] not in ['CREATE_COMPLETE', 'UPDATE_COMPLETE', 'UPDATE_ROLLBACK_COMPLETE']:
            logger.warning(f"CloudFormation stack '{stack_name}' is in an unexpected state: {stack['StackStatus']}")
    except ClientError as e:
        if 'does not exist' in str(e):
            logger.error(f"CloudFormation stack '{stack_name}' does not exist")
            return False
        else:
            logger.error(f"Error checking CloudFormation stack: {str(e)}")
            return False
    
    # Get the role names from the CloudFormation stack outputs
    try:
        outputs = {output['OutputKey']: output['OutputValue'] for output in stack['Outputs']}
        data_scientist_role_arn = outputs.get('DataScientistRoleArn', '')
        ml_engineer_role_arn = outputs.get('MLEngineerRoleArn', '')
        sagemaker_execution_role_arn = outputs.get('SageMakerExecutionRoleArn', '')
        
        # Extract role names from ARNs
        data_scientist_role_name = data_scientist_role_arn.split('/')[-1] if data_scientist_role_arn else f"{project_name}-DataScientist-Role"
        ml_engineer_role_name = ml_engineer_role_arn.split('/')[-1] if ml_engineer_role_arn else f"{project_name}-MLEngineer-Role"
        sagemaker_execution_role_name = sagemaker_execution_role_arn.split('/')[-1] if sagemaker_execution_role_arn else f"{project_name}-SageMaker-Execution-Role"
        
        role_names = [
            data_scientist_role_name,
            ml_engineer_role_name,
            sagemaker_execution_role_name
        ]
        
        logger.info(f"Using role names from CloudFormation outputs: {role_names}")
    except Exception as e:
        logger.warning(f"Failed to get role names from CloudFormation outputs: {str(e)}")
        # Fallback to default role names
        role_names = [
            f"{project_name}-DataScientist-Role",
            f"{project_name}-MLEngineer-Role",
            f"{project_name}-SageMaker-Execution-Role"
        ]
        logger.info(f"Using default role names: {role_names}")
    
    roles_exist = True
    for role_name in role_names:
        try:
            role = iam_client.get_role(RoleName=role_name)
            logger.info(f"IAM role '{role_name}' exists")
            
            # Check if the role has the correct trust relationship
            trust_policy = role['Role']['AssumeRolePolicyDocument']
            if isinstance(trust_policy, str):
                trust_policy = json.loads(trust_policy)
            if 'sagemaker.amazonaws.com' not in str(trust_policy):
                logger.warning(f"IAM role '{role_name}' may not have the correct trust relationship with SageMaker")
        except ClientError as e:
            if 'NoSuchEntity' in str(e):
                logger.error(f"IAM role '{role_name}' does not exist")
                roles_exist = False
            else:
                logger.error(f"Error checking IAM role '{role_name}': {str(e)}")
                roles_exist = False
    
    return roles_exist

def validate_sagemaker_domain(session, project_name):
    """Validate that the SageMaker domain and user profiles exist and are properly configured."""
    logger.info("Validating SageMaker domain and user profiles...")
    
    sagemaker_client = session.client('sagemaker')
    
    # Check if the SageMaker domain exists
    try:
        domains = sagemaker_client.list_domains()
        domain_name = f"{project_name}-domain"
        domain = next((d for d in domains['Domains'] if d['DomainName'] == domain_name), None)
        
        if domain is None:
            logger.error(f"SageMaker domain '{domain_name}' does not exist")
            return False
        
        domain_id = domain['DomainId']
        domain_status = domain['Status']
        logger.info(f"SageMaker domain '{domain_name}' exists with ID: {domain_id} and status: {domain_status}")
        
        if domain_status != 'InService':
            logger.warning(f"SageMaker domain '{domain_name}' is not in service (status: {domain_status})")
        
        # Check for the required user profiles
        user_profiles = sagemaker_client.list_user_profiles(DomainIdEquals=domain_id)
        
        # Check for Data Scientist user profile
        ds_profile = next((p for p in user_profiles['UserProfiles'] if p['UserProfileName'] == 'data-scientist'), None)
        if ds_profile is None:
            logger.error("Data Scientist user profile does not exist")
            return False
        
        logger.info(f"Data Scientist user profile exists with status: {ds_profile['Status']}")
        
        # Check for ML Engineer user profile
        ml_profile = next((p for p in user_profiles['UserProfiles'] if p['UserProfileName'] == 'ml-engineer'), None)
        if ml_profile is None:
            logger.error("ML Engineer user profile does not exist")
            return False
        
        logger.info(f"ML Engineer user profile exists with status: {ml_profile['Status']}")
        
        return True
    except Exception as e:
        logger.error(f"Error validating SageMaker domain: {str(e)}")
        return False

def validate_s3_bucket(session, bucket_name):
    """Validate that the S3 bucket exists and is accessible."""
    logger.info(f"Validating S3 bucket: {bucket_name}...")
    
    s3_client = session.client('s3')
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"S3 bucket '{bucket_name}' exists and is accessible")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"S3 bucket '{bucket_name}' does not exist")
        elif e.response['Error']['Code'] == '403':
            logger.error(f"S3 bucket '{bucket_name}' exists but is not accessible (permission denied)")
        else:
            logger.error(f"Error checking S3 bucket '{bucket_name}': {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting validation of core SageMaker infrastructure...")
    logger.info(f"Configuration: project_name={args.project_name}, profile={args.profile}, region={args.region}")
    
    # Set up AWS session
    session = setup_aws_session(args.profile, args.region)
    
    # Validate components
    iam_valid = validate_iam_roles(session, args.project_name)
    sagemaker_valid = validate_sagemaker_domain(session, args.project_name)
    s3_valid = validate_s3_bucket(session, args.data_bucket)
    
    # Print summary
    logger.info("\n=== Validation Summary ===")
    logger.info(f"IAM Roles: {'✅ Valid' if iam_valid else '❌ Invalid'}")
    logger.info(f"SageMaker Domain: {'✅ Valid' if sagemaker_valid else '❌ Invalid'}")
    logger.info(f"S3 Bucket: {'✅ Valid' if s3_valid else '❌ Invalid'}")
    
    # Determine overall status
    if iam_valid and sagemaker_valid and s3_valid:
        logger.info("\n✅ All components are valid. The core SageMaker infrastructure is properly deployed.")
        return 0
    else:
        logger.error("\n❌ Some components are invalid. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())