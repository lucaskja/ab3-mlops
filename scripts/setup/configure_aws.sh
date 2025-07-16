#!/bin/bash

# AWS CLI Configuration Script for MLOps SageMaker Demo
# Uses "ab" profile for cost tracking and resource allocation

set -e

echo "Configuring AWS CLI with 'ab' profile for MLOps SageMaker Demo..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install AWS CLI first."
    echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Configure the 'ab' profile
echo "Setting up AWS CLI profile 'ab'..."
echo "Please provide your AWS credentials when prompted."

aws configure --profile ab

# Verify the configuration
echo "Verifying AWS CLI configuration..."
aws sts get-caller-identity --profile ab

# Set default region if not already set
if ! aws configure get region --profile ab &> /dev/null; then
    echo "Setting default region to us-east-1..."
    aws configure set region us-east-1 --profile ab
fi

# Export the profile for current session
export AWS_PROFILE=ab

echo "AWS CLI configuration completed successfully!"
echo "Profile 'ab' is now configured and set as the active profile."
echo ""
echo "To use this profile in future sessions, run:"
echo "export AWS_PROFILE=ab"
echo ""
echo "Or use --profile ab with AWS CLI commands."