#!/bin/bash

# Deploy IAM Roles for MLOps SageMaker Demo
# This script deploys the CloudFormation stack with IAM roles and policies

set -e

# Configuration
PROJECT_NAME="mlops-sagemaker-demo"
STACK_NAME="${PROJECT_NAME}-iam-roles"
TEMPLATE_FILE="configs/iam-roles-cloudformation.yaml"
AWS_PROFILE="ab"
DATA_BUCKET_NAME="lucaskle-ab3-project-pv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if AWS CLI is configured
check_aws_cli() {
    print_status "Checking AWS CLI configuration..."
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        print_error "AWS CLI profile '$AWS_PROFILE' is not configured or invalid."
        print_status "Please run: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
    REGION=$(aws configure get region --profile "$AWS_PROFILE")
    
    print_success "AWS CLI configured for account $ACCOUNT_ID in region $REGION"
}

# Function to validate CloudFormation template
validate_template() {
    print_status "Validating CloudFormation template..."
    
    if [ ! -f "$TEMPLATE_FILE" ]; then
        print_error "Template file $TEMPLATE_FILE not found!"
        exit 1
    fi
    
    if aws cloudformation validate-template \
        --template-body file://"$TEMPLATE_FILE" \
        --profile "$AWS_PROFILE" > /dev/null; then
        print_success "CloudFormation template is valid"
    else
        print_error "CloudFormation template validation failed"
        exit 1
    fi
}

# Function to check if stack exists
stack_exists() {
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --profile "$AWS_PROFILE" \
        --query 'Stacks[0].StackStatus' \
        --output text 2>/dev/null || echo "DOES_NOT_EXIST"
}

# Function to deploy CloudFormation stack
deploy_stack() {
    print_status "Deploying IAM roles CloudFormation stack..."
    
    STACK_STATUS=$(stack_exists)
    
    if [ "$STACK_STATUS" = "DOES_NOT_EXIST" ]; then
        print_status "Creating new CloudFormation stack: $STACK_NAME"
        
        aws cloudformation create-stack \
            --stack-name "$STACK_NAME" \
            --template-body file://"$TEMPLATE_FILE" \
            --parameters ParameterKey=ProjectName,ParameterValue="$PROJECT_NAME" \
                        ParameterKey=DataBucketName,ParameterValue="$DATA_BUCKET_NAME" \
            --capabilities CAPABILITY_NAMED_IAM \
            --tags Key=Project,Value="$PROJECT_NAME" \
                   Key=Environment,Value=Development \
                   Key=ManagedBy,Value=CloudFormation \
            --profile "$AWS_PROFILE"
        
        print_status "Waiting for stack creation to complete..."
        aws cloudformation wait stack-create-complete \
            --stack-name "$STACK_NAME" \
            --profile "$AWS_PROFILE"
        
    else
        print_status "Updating existing CloudFormation stack: $STACK_NAME"
        
        aws cloudformation update-stack \
            --stack-name "$STACK_NAME" \
            --template-body file://"$TEMPLATE_FILE" \
            --parameters ParameterKey=ProjectName,ParameterValue="$PROJECT_NAME" \
                        ParameterKey=DataBucketName,ParameterValue="$DATA_BUCKET_NAME" \
            --capabilities CAPABILITY_NAMED_IAM \
            --profile "$AWS_PROFILE" || {
                if [ $? -eq 255 ]; then
                    print_warning "No updates to be performed on the stack"
                else
                    print_error "Stack update failed"
                    exit 1
                fi
            }
        
        if [ $? -ne 255 ]; then
            print_status "Waiting for stack update to complete..."
            aws cloudformation wait stack-update-complete \
                --stack-name "$STACK_NAME" \
                --profile "$AWS_PROFILE"
        fi
    fi
    
    print_success "CloudFormation stack deployment completed"
}

# Function to display stack outputs
show_outputs() {
    print_status "Retrieving stack outputs..."
    
    OUTPUTS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --profile "$AWS_PROFILE" \
        --query 'Stacks[0].Outputs' \
        --output table)
    
    if [ -n "$OUTPUTS" ]; then
        echo ""
        echo "Stack Outputs:"
        echo "$OUTPUTS"
    fi
}

# Function to run validation
run_validation() {
    print_status "Running IAM role validation..."
    
    if [ -f "scripts/setup/validate_iam_roles.py" ]; then
        python3 scripts/setup/validate_iam_roles.py --profile "$AWS_PROFILE"
    else
        print_warning "Validation script not found. Skipping validation."
    fi
}

# Function to create S3 buckets if they don't exist
create_s3_buckets() {
    print_status "Checking S3 buckets..."
    
    # Check if MLFlow artifacts bucket exists
    MLFLOW_BUCKET="${PROJECT_NAME}-mlflow-artifacts"
    
    if ! aws s3 ls "s3://$MLFLOW_BUCKET" --profile "$AWS_PROFILE" &> /dev/null; then
        print_status "Creating MLFlow artifacts bucket: $MLFLOW_BUCKET"
        aws s3 mb "s3://$MLFLOW_BUCKET" --profile "$AWS_PROFILE"
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "$MLFLOW_BUCKET" \
            --versioning-configuration Status=Enabled \
            --profile "$AWS_PROFILE"
        
        print_success "Created MLFlow artifacts bucket with versioning enabled"
    else
        print_success "MLFlow artifacts bucket already exists"
    fi
    
    # Check if data bucket exists (should already exist)
    if aws s3 ls "s3://$DATA_BUCKET_NAME" --profile "$AWS_PROFILE" &> /dev/null; then
        print_success "Data bucket $DATA_BUCKET_NAME is accessible"
    else
        print_warning "Data bucket $DATA_BUCKET_NAME is not accessible. Please verify permissions."
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy IAM roles for MLOps SageMaker Demo"
    echo ""
    echo "Options:"
    echo "  -p, --profile PROFILE    AWS CLI profile to use (default: ab)"
    echo "  -s, --stack-name NAME    CloudFormation stack name (default: $STACK_NAME)"
    echo "  -b, --bucket NAME        Data bucket name (default: $DATA_BUCKET_NAME)"
    echo "  --skip-validation        Skip IAM role validation after deployment"
    echo "  --skip-buckets          Skip S3 bucket creation"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Deploy with default settings"
    echo "  $0 --profile myprofile  # Use different AWS profile"
    echo "  $0 --skip-validation    # Deploy without validation"
}

# Parse command line arguments
SKIP_VALIDATION=false
SKIP_BUCKETS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        -s|--stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        -b|--bucket)
            DATA_BUCKET_NAME="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-buckets)
            SKIP_BUCKETS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=================================================="
    echo "MLOps SageMaker Demo - IAM Roles Deployment"
    echo "=================================================="
    echo ""
    
    print_status "Configuration:"
    echo "  Project Name: $PROJECT_NAME"
    echo "  Stack Name: $STACK_NAME"
    echo "  AWS Profile: $AWS_PROFILE"
    echo "  Data Bucket: $DATA_BUCKET_NAME"
    echo "  Template: $TEMPLATE_FILE"
    echo ""
    
    # Run deployment steps
    check_aws_cli
    validate_template
    
    if [ "$SKIP_BUCKETS" = false ]; then
        create_s3_buckets
    fi
    
    deploy_stack
    show_outputs
    
    if [ "$SKIP_VALIDATION" = false ]; then
        echo ""
        run_validation
    fi
    
    echo ""
    print_success "IAM roles deployment completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Review the stack outputs above"
    echo "2. Configure SageMaker Studio with the created roles"
    echo "3. Test role-based access with the validation script"
    echo ""
    print_status "To validate roles manually, run:"
    echo "  python3 scripts/setup/validate_iam_roles.py --profile $AWS_PROFILE"
}

# Run main function
main