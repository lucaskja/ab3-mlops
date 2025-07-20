#!/bin/bash

# Setup Core SageMaker Pipeline
# This script sets up the simplified YOLOv11 training pipeline for the core setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
AWS_PROFILE="ab"
AWS_REGION="us-east-1"
DRY_RUN=false
PIPELINE_NAME=""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup Core SageMaker Pipeline for YOLOv11 Training

OPTIONS:
    --profile PROFILE       AWS CLI profile to use (default: ab)
    --region REGION         AWS region (default: us-east-1)
    --pipeline-name NAME    Custom pipeline name (optional)
    --dry-run              Validate setup without creating resources
    --help                 Show this help message

EXAMPLES:
    # Create pipeline with default settings
    $0

    # Create pipeline with custom name
    $0 --pipeline-name my-yolov11-pipeline

    # Dry run to validate setup
    $0 --dry-run

    # Use different AWS profile and region
    $0 --profile my-profile --region us-west-2

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --pipeline-name)
            PIPELINE_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

print_header "🚀 Core SageMaker Pipeline Setup"
echo "=================================="

# Verify AWS profile
print_status "Verifying AWS profile: $AWS_PROFILE"
if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
    print_error "AWS profile '$AWS_PROFILE' not found or not configured"
    print_error "Please configure the profile using: aws configure --profile $AWS_PROFILE"
    exit 1
fi

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
USER_ARN=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Arn --output text)

print_status "AWS Account ID: $ACCOUNT_ID"
print_status "User/Role: $USER_ARN"
print_status "Region: $AWS_REGION"

# Check if virtual environment exists
if [[ -d "$PROJECT_ROOT/venv" ]]; then
    print_status "Using virtual environment: $PROJECT_ROOT/venv"
    PYTHON_CMD="$PROJECT_ROOT/venv/bin/python"
else
    print_warning "Virtual environment not found, using system Python"
    PYTHON_CMD="python3"
fi

# Verify Python dependencies
print_status "Checking Python dependencies..."
if ! $PYTHON_CMD -c "import boto3, sagemaker" >/dev/null 2>&1; then
    print_error "Required Python packages not found"
    print_error "Please install: pip install boto3 sagemaker"
    exit 1
fi

# Check if core SageMaker infrastructure exists
print_status "Checking core SageMaker infrastructure..."
DOMAIN_COUNT=$(aws sagemaker list-domains --profile "$AWS_PROFILE" --region "$AWS_REGION" --query 'length(Domains)' --output text 2>/dev/null || echo "0")

if [[ "$DOMAIN_COUNT" -eq "0" ]]; then
    print_warning "No SageMaker domains found"
    print_warning "Please run deploy_core_sagemaker.sh first to set up the infrastructure"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Exiting..."
            exit 0
        fi
    fi
else
    print_status "Found $DOMAIN_COUNT SageMaker domain(s)"
fi

# Check S3 bucket access
print_status "Checking S3 bucket access..."
BUCKET_NAME="lucaskle-ab3-project-pv"
if ! aws s3 ls "s3://$BUCKET_NAME" --profile "$AWS_PROFILE" >/dev/null 2>&1; then
    print_error "Cannot access S3 bucket: $BUCKET_NAME"
    print_error "Please ensure the bucket exists and you have access"
    exit 1
fi

print_status "S3 bucket access verified: $BUCKET_NAME"

if [[ "$DRY_RUN" == "true" ]]; then
    print_header "✅ Dry Run Complete"
    print_status "All prerequisites verified"
    print_status "Ready to create core SageMaker pipeline"
    exit 0
fi

# Create the pipeline
print_header "📋 Creating Core SageMaker Pipeline"

PIPELINE_ARGS="--profile $AWS_PROFILE --region $AWS_REGION"
if [[ -n "$PIPELINE_NAME" ]]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --pipeline-name $PIPELINE_NAME"
fi

print_status "Running pipeline creation script..."
if $PYTHON_CMD "$SCRIPT_DIR/create_core_pipeline.py" $PIPELINE_ARGS; then
    print_header "✅ Pipeline Setup Complete!"
    
    echo
    print_status "Next Steps:"
    echo "1. Use the ML Engineer notebook to execute the pipeline"
    echo "2. Or use the command line execution script:"
    echo "   $SCRIPT_DIR/execute_core_pipeline.py --list-pipelines --profile $AWS_PROFILE"
    echo
    echo "3. To execute a pipeline:"
    echo "   $SCRIPT_DIR/execute_core_pipeline.py --pipeline-name PIPELINE_NAME --profile $AWS_PROFILE"
    echo
    echo "4. Monitor execution:"
    echo "   $SCRIPT_DIR/execute_core_pipeline.py --monitor EXECUTION_ARN --profile $AWS_PROFILE"
    
    print_header "📚 Documentation"
    echo "- Pipeline execution guide: notebooks/README_ml-engineer.md"
    echo "- SageMaker Console: https://$AWS_REGION.console.aws.amazon.com/sagemaker/home?region=$AWS_REGION#/pipelines"
    
else
    print_error "Pipeline creation failed"
    exit 1
fi
