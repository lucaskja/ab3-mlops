#!/bin/bash
# CDK Deployment Validation Script for MLOps SageMaker Demo
# This script validates the CDK deployment by checking if resources were created correctly

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print with colors
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
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

# Check if AWS profile is set
if [ -z "$AWS_PROFILE" ]; then
    print_warning "AWS_PROFILE environment variable not set. Using --profile ab flag."
    PROFILE="--profile ab"
else
    print_info "Using AWS_PROFILE: $AWS_PROFILE"
    PROFILE=""
fi

# Project name
PROJECT_NAME="mlops-sagemaker-demo"

# Validate IAM roles
print_info "Validating IAM roles..."

# Check Data Scientist Role
aws iam get-role --role-name "${PROJECT_NAME}-DataScientist-Role" $PROFILE &> /dev/null && {
    print_success "Data Scientist Role exists."
} || {
    print_error "Data Scientist Role not found."
    exit 1
}

# Check ML Engineer Role
aws iam get-role --role-name "${PROJECT_NAME}-MLEngineer-Role" $PROFILE &> /dev/null && {
    print_success "ML Engineer Role exists."
} || {
    print_error "ML Engineer Role not found."
    exit 1
}

# Check SageMaker Execution Role
aws iam get-role --role-name "${PROJECT_NAME}-SageMaker-Execution-Role" $PROFILE &> /dev/null && {
    print_success "SageMaker Execution Role exists."
} || {
    print_error "SageMaker Execution Role not found."
    exit 1
}

# Check SageMaker Studio Policy
aws iam get-policy --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text $PROFILE):policy/${PROJECT_NAME}-SageMaker-Studio-Policy" $PROFILE &> /dev/null && {
    print_success "SageMaker Studio Policy exists."
} || {
    print_error "SageMaker Studio Policy not found."
    exit 1
}

# Validate SageMaker endpoint
print_info "Validating SageMaker endpoint..."
aws sagemaker describe-endpoint --endpoint-name "${PROJECT_NAME}-yolov11-endpoint" $PROFILE &> /dev/null && {
    print_success "SageMaker endpoint exists."
    
    # Check endpoint status
    ENDPOINT_STATUS=$(aws sagemaker describe-endpoint --endpoint-name "${PROJECT_NAME}-yolov11-endpoint" --query "EndpointStatus" --output text $PROFILE)
    print_info "Endpoint status: $ENDPOINT_STATUS"
    
    if [ "$ENDPOINT_STATUS" != "InService" ]; then
        print_warning "Endpoint is not in service yet. Current status: $ENDPOINT_STATUS"
    fi
} || {
    print_warning "SageMaker endpoint not found or still being created."
}

# Validate Lambda functions
print_info "Validating Lambda functions..."
aws lambda get-function --function-name "${PROJECT_NAME}-endpoint-invocation" $PROFILE &> /dev/null && {
    print_success "Endpoint invocation Lambda function exists."
} || {
    print_error "Endpoint invocation Lambda function not found."
    exit 1
}

aws lambda get-function --function-name "${PROJECT_NAME}-endpoint-status-handler" $PROFILE &> /dev/null && {
    print_success "Endpoint status handler Lambda function exists."
} || {
    print_error "Endpoint status handler Lambda function not found."
    exit 1
}

# Validate API Gateway
print_info "Validating API Gateway..."
API_ID=$(aws apigateway get-rest-apis --query "items[?name=='${PROJECT_NAME}-endpoint-api'].id" --output text $PROFILE)
if [ -n "$API_ID" ]; then
    print_success "API Gateway exists with ID: $API_ID"
    API_URL="https://${API_ID}.execute-api.$(aws configure get region $PROFILE).amazonaws.com/prod/"
    print_info "API URL: $API_URL"
else
    print_error "API Gateway not found."
    exit 1
fi

# Validate SNS topic
print_info "Validating SNS topic..."
SNS_TOPIC_ARN=$(aws sns list-topics --query "Topics[?contains(TopicArn, '${PROJECT_NAME}-endpoint-monitoring')].TopicArn" --output text $PROFILE)
if [ -n "$SNS_TOPIC_ARN" ]; then
    print_success "SNS topic exists with ARN: $SNS_TOPIC_ARN"
else
    print_error "SNS topic not found."
    exit 1
fi

# Validate EventBridge rule
print_info "Validating EventBridge rule..."
aws events describe-rule --name "${PROJECT_NAME}-endpoint-status-changes" $PROFILE &> /dev/null && {
    print_success "EventBridge rule exists."
} || {
    print_error "EventBridge rule not found."
    exit 1
}

print_success "All resources validated successfully!"
print_info "CDK deployment is complete and validated."