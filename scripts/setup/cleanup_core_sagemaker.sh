#!/bin/bash
# Core SageMaker Infrastructure Cleanup Script
# This script removes the resources created by deploy_core_sagemaker.sh

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with colors
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_header() {
    echo -e "\n${GREEN}========== $1 ==========${NC}\n"
}

# Default values
AWS_PROFILE="ab"
AWS_REGION="us-east-1"
PROJECT_NAME="sagemaker-core-setup"
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --profile)
      AWS_PROFILE="$2"
      shift
      shift
      ;;
    --region)
      AWS_REGION="$2"
      shift
      shift
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Clean up core SageMaker infrastructure"
      echo ""
      echo "Options:"
      echo "  --profile PROFILE            AWS CLI profile to use (default: ab)"
      echo "  --region REGION              AWS region (default: us-east-1)"
      echo "  --project-name NAME          Project name (default: sagemaker-core-setup)"
      echo "  --force                      Skip confirmation prompts"
      echo "  --help                       Show this help message"
      exit 0
      ;;
    *)
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to check if AWS CLI is configured with the "ab" profile
check_aws_profile() {
    print_header "Checking AWS Profile Configuration"
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        print_error "AWS CLI profile '$AWS_PROFILE' is not configured or invalid."
        print_info "Please run: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
    REGION=$(aws configure get region --profile "$AWS_PROFILE")
    
    print_success "AWS CLI configured for account $ACCOUNT_ID in region $REGION using profile '$AWS_PROFILE'"
    
    # Set AWS profile for the session
    export AWS_PROFILE=$AWS_PROFILE
    export AWS_DEFAULT_REGION=$AWS_REGION
}

# Function to confirm action
confirm_action() {
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    read -p "$1 (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to delete SageMaker domain and user profiles
delete_sagemaker_domain() {
    print_header "Deleting SageMaker Domain and User Profiles"
    
    # Check if SageMaker domain exists
    DOMAIN_ID=$(aws sagemaker list-domains --profile $AWS_PROFILE --query "Domains[?DomainName=='${PROJECT_NAME}-domain'].DomainId" --output text)
    
    if [ -z "$DOMAIN_ID" ] || [ "$DOMAIN_ID" == "None" ]; then
        print_info "No SageMaker domain found with name: ${PROJECT_NAME}-domain"
        return
    fi
    
    print_info "Found SageMaker domain with ID: $DOMAIN_ID"
    
    # List user profiles
    USER_PROFILES=$(aws sagemaker list-user-profiles --domain-id $DOMAIN_ID --profile $AWS_PROFILE --query "UserProfiles[].UserProfileName" --output text)
    
    # Delete user profiles
    for PROFILE in $USER_PROFILES; do
        print_info "Checking apps for user profile: $PROFILE"
        
        # List apps for the user profile
        APPS=$(aws sagemaker list-apps --domain-id $DOMAIN_ID --user-profile-name $PROFILE --profile $AWS_PROFILE --query "Apps[].AppName" --output text)
        
        # Delete apps
        for APP in $APPS; do
            print_info "Deleting app: $APP for user profile: $PROFILE"
            
            aws sagemaker delete-app \
                --domain-id $DOMAIN_ID \
                --user-profile-name $PROFILE \
                --app-name $APP \
                --app-type JupyterServer \
                --profile $AWS_PROFILE
            
            print_info "Waiting for app deletion to complete..."
            aws sagemaker wait app-deleted \
                --domain-id $DOMAIN_ID \
                --user-profile-name $PROFILE \
                --app-name $APP \
                --app-type JupyterServer \
                --profile $AWS_PROFILE
        done
        
        print_info "Deleting user profile: $PROFILE"
        
        aws sagemaker delete-user-profile \
            --domain-id $DOMAIN_ID \
            --user-profile-name $PROFILE \
            --profile $AWS_PROFILE
        
        print_success "Deleted user profile: $PROFILE"
    done
    
    # Delete domain
    print_info "Deleting SageMaker domain: $DOMAIN_ID"
    
    aws sagemaker delete-domain \
        --domain-id $DOMAIN_ID \
        --profile $AWS_PROFILE
    
    print_success "Deleted SageMaker domain: $DOMAIN_ID"
}

# Function to delete CloudFormation stack
delete_cloudformation_stack() {
    print_header "Deleting CloudFormation Stack"
    
    STACK_NAME="${PROJECT_NAME}-iam-roles"
    
    # Check if stack exists
    if ! aws cloudformation describe-stacks --stack-name $STACK_NAME --profile $AWS_PROFILE &> /dev/null; then
        print_info "CloudFormation stack $STACK_NAME does not exist"
        return
    fi
    
    print_info "Deleting CloudFormation stack: $STACK_NAME"
    
    aws cloudformation delete-stack \
        --stack-name $STACK_NAME \
        --profile $AWS_PROFILE
    
    print_info "Waiting for stack deletion to complete..."
    aws cloudformation wait stack-delete-complete \
        --stack-name $STACK_NAME \
        --profile $AWS_PROFILE
    
    print_success "Deleted CloudFormation stack: $STACK_NAME"
}

# Main function
main() {
    print_header "Core SageMaker Infrastructure Cleanup"
    
    print_info "Configuration:"
    echo "  Project Name: $PROJECT_NAME"
    echo "  AWS Profile: $AWS_PROFILE"
    echo "  AWS Region: $AWS_REGION"
    echo ""
    
    # Check AWS profile configuration
    check_aws_profile
    
    # Confirm cleanup
    if ! confirm_action "This will delete all SageMaker resources created by the deployment script. Are you sure you want to continue?"; then
        print_info "Cleanup cancelled"
        exit 0
    fi
    
    # Delete SageMaker domain and user profiles
    delete_sagemaker_domain
    
    # Confirm CloudFormation stack deletion
    if confirm_action "Do you want to delete the IAM roles CloudFormation stack as well?"; then
        delete_cloudformation_stack
    else
        print_info "Skipping CloudFormation stack deletion"
    fi
    
    print_header "Cleanup Summary"
    print_success "Core SageMaker infrastructure has been cleaned up successfully!"
}

# Run main function
main