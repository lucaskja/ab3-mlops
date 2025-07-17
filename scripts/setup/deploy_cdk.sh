#!/bin/bash
# CDK Deployment Script for MLOps SageMaker Demo
# This script deploys the CDK stacks for the MLOps SageMaker Demo project

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

# Check if AWS profile is set
if [ -z "$AWS_PROFILE" ]; then
    print_warning "AWS_PROFILE environment variable not set. Using --profile ab flag."
    PROFILE_FLAG="--profile ab"
else
    print_info "Using AWS_PROFILE: $AWS_PROFILE"
    PROFILE_FLAG=""
fi

# Check if CDK is installed
if ! command -v cdk &> /dev/null; then
    print_error "AWS CDK is not installed. Please install it with 'npm install -g aws-cdk'"
    exit 1
fi

# Navigate to CDK directory
print_info "Navigating to CDK directory..."
cd "$(dirname "$0")/../../configs/cdk" || exit 1

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_info "Installing dependencies..."
    npm install
fi

# Build TypeScript code
print_info "Building TypeScript code..."
npm run build

# Bootstrap CDK (if needed)
print_info "Checking if CDK bootstrap is needed..."
cdk bootstrap $PROFILE_FLAG || {
    print_error "CDK bootstrap failed. Please check your AWS credentials and permissions."
    exit 1
}

# Synthesize CloudFormation templates
print_info "Synthesizing CloudFormation templates..."
cdk synth

# Deploy stacks
print_info "Deploying IAM stack..."
cdk deploy MLOpsSageMakerIAMStack --require-approval never $PROFILE_FLAG || {
    print_error "IAM stack deployment failed."
    exit 1
}

print_info "Deploying Endpoint stack..."
cdk deploy MLOpsSageMakerEndpointStack --require-approval never $PROFILE_FLAG || {
    print_error "Endpoint stack deployment failed."
    exit 1
}

print_info "Deployment completed successfully!"
print_info "Running validation..."

# Run validation script
"$(dirname "$0")/validate_cdk_deployment.sh"

print_info "All done!"