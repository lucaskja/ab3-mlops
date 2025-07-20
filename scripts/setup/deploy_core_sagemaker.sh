#!/bin/bash
# Core SageMaker Infrastructure Deployment Script
# This script deploys the essential components needed for SageMaker Studio and YOLOv11 training

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
DATA_BUCKET_NAME="lucaskle-ab3-project-pv"
SKIP_VALIDATION=false
SKIP_STACK_WAIT=false

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
    --data-bucket)
      DATA_BUCKET_NAME="$2"
      shift
      shift
      ;;
    --skip-validation)
      SKIP_VALIDATION=true
      shift
      ;;
    --skip-stack-wait)
      SKIP_STACK_WAIT=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Deploy core SageMaker infrastructure"
      echo ""
      echo "Options:"
      echo "  --profile PROFILE            AWS CLI profile to use (default: ab)"
      echo "  --region REGION              AWS region (default: us-east-1)"
      echo "  --project-name NAME          Project name (default: sagemaker-core-setup)"
      echo "  --data-bucket BUCKET         Data bucket name (default: lucaskle-ab3-project-pv)"
      echo "  --skip-validation            Skip validation steps"
      echo "  --skip-stack-wait            Skip waiting for CloudFormation stack updates"
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

# Function to deploy IAM roles
deploy_iam_roles() {
    print_header "Deploying IAM Roles and Policies"
    
    # Pass --skip-wait flag if SKIP_STACK_WAIT is true
    SKIP_WAIT_FLAG=""
    if [ "$SKIP_STACK_WAIT" = true ]; then
        SKIP_WAIT_FLAG="--skip-wait"
    fi
    
    # Get the directory of this script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    # Deploy IAM roles using the existing script
    bash "$SCRIPT_DIR/deploy_iam_roles.sh" --profile $AWS_PROFILE --stack-name "${PROJECT_NAME}-iam-roles" --bucket $DATA_BUCKET_NAME $SKIP_WAIT_FLAG || {
        ERROR_CODE=$?
        if grep -q "No updates are to be performed" <<< "$(cat /tmp/deploy_iam_error 2>/dev/null)"; then
            print_warning "No updates to be performed on the IAM stack. Continuing..."
        else
            print_error "IAM roles deployment failed with code $ERROR_CODE. Exiting..."
            exit 1
        fi
    }
    
    # Validate IAM roles if not skipped
    if [ "$SKIP_VALIDATION" = false ]; then
        print_header "Validating IAM Roles"
        
        # Check if validate_iam_roles.py exists
        if [ -f "$SCRIPT_DIR/validate_iam_roles.py" ]; then
            print_info "Running IAM role validation..."
            python3 "$SCRIPT_DIR/validate_iam_roles.py" --profile $AWS_PROFILE
        else
            print_warning "IAM role validation script not found. Skipping validation."
        fi
    fi
}

# Function to create SageMaker domain and user profiles
create_sagemaker_domain() {
    print_header "Creating SageMaker Domain and User Profiles"
    
    # Get IAM role ARNs
    DATA_SCIENTIST_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "${PROJECT_NAME}-iam-roles" --profile $AWS_PROFILE --query "Stacks[0].Outputs[?OutputKey=='DataScientistRoleArn'].OutputValue" --output text)
    ML_ENGINEER_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "${PROJECT_NAME}-iam-roles" --profile $AWS_PROFILE --query "Stacks[0].Outputs[?OutputKey=='MLEngineerRoleArn'].OutputValue" --output text)
    SAGEMAKER_EXECUTION_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "${PROJECT_NAME}-iam-roles" --profile $AWS_PROFILE --query "Stacks[0].Outputs[?OutputKey=='SageMakerExecutionRoleArn'].OutputValue" --output text)
    
    if [ -z "$DATA_SCIENTIST_ROLE_ARN" ] || [ -z "$ML_ENGINEER_ROLE_ARN" ] || [ -z "$SAGEMAKER_EXECUTION_ROLE_ARN" ]; then
        print_error "Failed to retrieve IAM role ARNs from CloudFormation stack. Exiting..."
        exit 1
    fi
    
    print_info "Retrieved IAM role ARNs:"
    print_info "Data Scientist Role ARN: $DATA_SCIENTIST_ROLE_ARN"
    print_info "ML Engineer Role ARN: $ML_ENGINEER_ROLE_ARN"
    print_info "SageMaker Execution Role ARN: $SAGEMAKER_EXECUTION_ROLE_ARN"
    
    # Check if SageMaker domain already exists
    DOMAIN_ID=$(aws sagemaker list-domains --profile $AWS_PROFILE --query "Domains[?DomainName=='${PROJECT_NAME}-domain'].DomainId" --output text)
    
    if [ -z "$DOMAIN_ID" ] || [ "$DOMAIN_ID" == "None" ]; then
        print_info "Creating new SageMaker domain: ${PROJECT_NAME}-domain"
        
        # Get default VPC and subnet information
        DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --profile $AWS_PROFILE --query "Vpcs[0].VpcId" --output text)
        
        # Get the first available subnet in the default VPC
        SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC_ID" --profile $AWS_PROFILE --query "Subnets[0].SubnetId" --output text)
        
        if [ -z "$DEFAULT_VPC_ID" ] || [ -z "$SUBNET_ID" ]; then
            print_error "Failed to retrieve default VPC or subnet information. Exiting..."
            exit 1
        fi
        
        # Create SageMaker domain
        print_info "Creating SageMaker domain. This may take a few minutes..."
        
        # Try to create the domain and capture any errors
        if DOMAIN_ID=$(aws sagemaker create-domain \
            --domain-name "${PROJECT_NAME}-domain" \
            --auth-mode IAM \
            --vpc-id $DEFAULT_VPC_ID \
            --subnet-ids $SUBNET_ID \
            --default-user-settings "{\"ExecutionRole\": \"$SAGEMAKER_EXECUTION_ROLE_ARN\"}" \
            --profile $AWS_PROFILE \
            --query DomainId \
            --output text 2>/tmp/domain_error); then
            
            print_success "Created SageMaker domain with ID: $DOMAIN_ID"
        else
            ERROR_CODE=$?
            if grep -q "ResourceInUse" /tmp/domain_error; then
                print_warning "A SageMaker domain with this name already exists."
                # Try to get the domain ID
                DOMAIN_ID=$(aws sagemaker list-domains --profile $AWS_PROFILE --query "Domains[?DomainName=='${PROJECT_NAME}-domain'].DomainId" --output text)
                print_info "Using existing domain with ID: $DOMAIN_ID"
            else
                print_error "Failed to create SageMaker domain: $(cat /tmp/domain_error)"
                print_info "You may need to create the domain manually through the AWS console."
                DOMAIN_ID=""
            fi
        fi
    else
        print_info "SageMaker domain already exists with ID: $DOMAIN_ID"
    fi
    
    # Create Data Scientist user profile
    DS_USER_PROFILE_NAME="data-scientist"
    
    # Check if domain ID is valid before creating user profiles
    if [ -z "$DOMAIN_ID" ] || [ "$DOMAIN_ID" == "None" ]; then
        print_warning "Domain ID is not available yet. User profiles will need to be created manually later."
        print_info "Wait for the domain to be created, then run the following commands:"
        print_info "aws sagemaker create-user-profile --domain-id <DOMAIN_ID> --user-profile-name data-scientist --user-settings '{\"ExecutionRole\": \"$DATA_SCIENTIST_ROLE_ARN\"}' --profile $AWS_PROFILE"
        print_info "aws sagemaker create-user-profile --domain-id <DOMAIN_ID> --user-profile-name ml-engineer --user-settings '{\"ExecutionRole\": \"$ML_ENGINEER_ROLE_ARN\"}' --profile $AWS_PROFILE"
    else
        if ! aws sagemaker describe-user-profile --domain-id $DOMAIN_ID --user-profile-name $DS_USER_PROFILE_NAME --profile $AWS_PROFILE &> /dev/null; then
            print_info "Creating Data Scientist user profile..."
            
            aws sagemaker create-user-profile \
                --domain-id $DOMAIN_ID \
                --user-profile-name $DS_USER_PROFILE_NAME \
                --user-settings "{\"ExecutionRole\": \"$DATA_SCIENTIST_ROLE_ARN\"}" \
                --profile $AWS_PROFILE
            
            print_success "Created Data Scientist user profile"
        else
            print_info "Data Scientist user profile already exists"
        fi
        
        # Create ML Engineer user profile
        ML_USER_PROFILE_NAME="ml-engineer"
        if ! aws sagemaker describe-user-profile --domain-id $DOMAIN_ID --user-profile-name $ML_USER_PROFILE_NAME --profile $AWS_PROFILE &> /dev/null; then
            print_info "Creating ML Engineer user profile..."
            
            aws sagemaker create-user-profile \
                --domain-id $DOMAIN_ID \
                --user-profile-name $ML_USER_PROFILE_NAME \
                --user-settings "{\"ExecutionRole\": \"$ML_ENGINEER_ROLE_ARN\"}" \
                --profile $AWS_PROFILE
            
            print_success "Created ML Engineer user profile"
        else
            print_info "ML Engineer user profile already exists"
        fi
    fi
    
    print_success "SageMaker domain and user profiles setup completed"
    
    # Return domain ID for further use
    echo $DOMAIN_ID
}

# Function to create S3 buckets if they don't exist
create_s3_buckets() {
    print_header "Setting up S3 Buckets"
    
    # Check if data bucket exists (should already exist)
    if aws s3 ls "s3://$DATA_BUCKET_NAME" --profile $AWS_PROFILE &> /dev/null; then
        print_success "Data bucket $DATA_BUCKET_NAME is accessible"
    else
        print_warning "Data bucket $DATA_BUCKET_NAME is not accessible. Please verify permissions."
    fi
    
    # Create notebooks directory in S3 bucket
    print_info "Creating notebooks directory in S3 bucket..."
    aws s3api put-object --bucket $DATA_BUCKET_NAME --key notebooks/ --profile $AWS_PROFILE
    
    print_success "S3 bucket setup completed"
}

# Function to validate the deployment
validate_deployment() {
    print_header "Validating Deployment"
    
    # Check SageMaker domain status
    print_info "Checking SageMaker domain status..."
    DOMAIN_STATUS=$(aws sagemaker list-domains --profile $AWS_PROFILE --query "Domains[?DomainName=='${PROJECT_NAME}-domain'].Status" --output text)
    
    if [ "$DOMAIN_STATUS" == "InService" ]; then
        print_success "SageMaker domain is in service"
    else
        print_warning "SageMaker domain status: $DOMAIN_STATUS"
        print_info "SageMaker domain may still be creating. Check the AWS console for status."
        print_info "You can run the validation script later to verify the deployment."
        return
    fi
    
    # Check user profiles
    print_info "Checking user profiles..."
    DOMAIN_ID=$(aws sagemaker list-domains --profile $AWS_PROFILE --query "Domains[?DomainName=='${PROJECT_NAME}-domain'].DomainId" --output text)
    
    if [ -z "$DOMAIN_ID" ]; then
        print_warning "Could not retrieve domain ID. Skipping user profile checks."
        return
    fi
    
    DS_STATUS=$(aws sagemaker list-user-profiles --domain-id $DOMAIN_ID --profile $AWS_PROFILE --query "UserProfiles[?UserProfileName=='data-scientist'].Status" --output text)
    ML_STATUS=$(aws sagemaker list-user-profiles --domain-id $DOMAIN_ID --profile $AWS_PROFILE --query "UserProfiles[?UserProfileName=='ml-engineer'].Status" --output text)
    
    if [ "$DS_STATUS" == "InService" ]; then
        print_success "Data Scientist user profile is in service"
    else
        print_warning "Data Scientist user profile status: $DS_STATUS"
    fi
    
    if [ "$ML_STATUS" == "InService" ]; then
        print_success "ML Engineer user profile is in service"
    else
        print_warning "ML Engineer user profile status: $ML_STATUS"
    fi
    
    # Check IAM roles
    print_info "Checking IAM roles..."
    
    # Get the role names from the CloudFormation stack outputs
    DATA_SCIENTIST_ROLE_NAME=$(aws cloudformation describe-stacks --stack-name "${PROJECT_NAME}-iam-roles" --profile $AWS_PROFILE --query "Stacks[0].Outputs[?OutputKey=='DataScientistRoleArn'].OutputValue" --output text | cut -d'/' -f2)
    ML_ENGINEER_ROLE_NAME=$(aws cloudformation describe-stacks --stack-name "${PROJECT_NAME}-iam-roles" --profile $AWS_PROFILE --query "Stacks[0].Outputs[?OutputKey=='MLEngineerRoleArn'].OutputValue" --output text | cut -d'/' -f2)
    SAGEMAKER_EXECUTION_ROLE_NAME=$(aws cloudformation describe-stacks --stack-name "${PROJECT_NAME}-iam-roles" --profile $AWS_PROFILE --query "Stacks[0].Outputs[?OutputKey=='SageMakerExecutionRoleArn'].OutputValue" --output text | cut -d'/' -f2)
    
    if aws iam get-role --role-name "$DATA_SCIENTIST_ROLE_NAME" --profile $AWS_PROFILE &> /dev/null; then
        print_success "Data Scientist IAM role exists: $DATA_SCIENTIST_ROLE_NAME"
    else
        print_warning "Data Scientist IAM role does not exist: $DATA_SCIENTIST_ROLE_NAME"
    fi
    
    if aws iam get-role --role-name "$ML_ENGINEER_ROLE_NAME" --profile $AWS_PROFILE &> /dev/null; then
        print_success "ML Engineer IAM role exists: $ML_ENGINEER_ROLE_NAME"
    else
        print_warning "ML Engineer IAM role does not exist: $ML_ENGINEER_ROLE_NAME"
    fi
    
    if aws iam get-role --role-name "$SAGEMAKER_EXECUTION_ROLE_NAME" --profile $AWS_PROFILE &> /dev/null; then
        print_success "SageMaker Execution IAM role exists: $SAGEMAKER_EXECUTION_ROLE_NAME"
    else
        print_warning "SageMaker Execution IAM role does not exist: $SAGEMAKER_EXECUTION_ROLE_NAME"
    fi
    
    print_success "Validation completed"
}

# Main function
main() {
    print_header "Core SageMaker Infrastructure Deployment"
    
    print_info "Configuration:"
    echo "  Project Name: $PROJECT_NAME"
    echo "  AWS Profile: $AWS_PROFILE"
    echo "  AWS Region: $AWS_REGION"
    echo "  Data Bucket: $DATA_BUCKET_NAME"
    echo ""
    
    # Check AWS profile configuration
    check_aws_profile
    
    # Deploy IAM roles
    deploy_iam_roles
    
    # Create S3 buckets
    create_s3_buckets
    
    # Create SageMaker domain and user profiles
    DOMAIN_ID=$(create_sagemaker_domain)
    
    # Validate deployment if not skipped
    if [ "$SKIP_VALIDATION" = false ]; then
        validate_deployment
    fi
    
    print_header "Deployment Summary"
    print_success "Core SageMaker infrastructure has been deployed successfully!"
    print_info "SageMaker Domain ID: $DOMAIN_ID"
    print_info "SageMaker Domain Name: ${PROJECT_NAME}-domain"
    print_info "User Profiles:"
    print_info "  - data-scientist (Data Scientist role)"
    print_info "  - ml-engineer (ML Engineer role)"
    print_info ""
    print_info "Next Steps:"
    print_info "1. Access SageMaker Studio: https://$AWS_REGION.console.aws.amazon.com/sagemaker/home?region=$AWS_REGION#/studio"
    print_info "2. Select the domain '${PROJECT_NAME}-domain'"
    print_info "3. Launch Studio for either 'data-scientist' or 'ml-engineer' user profile"
    print_info "4. Upload and run the example notebooks for YOLOv11 training"
    print_info ""
    print_info "To clean up resources when no longer needed, run: ./scripts/setup/cleanup_core_sagemaker.sh --profile $AWS_PROFILE"
}

# Run main function
main