#!/bin/bash
# Deploy CDK stack for Lambda-SageMaker endpoint deployment

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default values
AWS_PROFILE="ab"
AWS_REGION="us-east-1"
STACK_NAME=""
LAMBDA_CODE_PATH="$PROJECT_ROOT/scripts/training"

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
    --stack-name)
      STACK_NAME="$2"
      shift
      shift
      ;;
    --lambda-code-path)
      LAMBDA_CODE_PATH="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set AWS profile
export AWS_PROFILE=$AWS_PROFILE
export AWS_DEFAULT_REGION=$AWS_REGION

# Load project configuration
echo "Loading project configuration..."
# Use Python from venv if available
if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON="python3"
fi

PROJECT_NAME=$($PYTHON -c "from configs.project_config import PROJECT_NAME; print(PROJECT_NAME)")
MODEL_NAME=$($PYTHON -c "from configs.project_config import MODEL_NAME; print(MODEL_NAME)")
SAGEMAKER_ROLE_ARN=$($PYTHON -c "from configs.project_config import get_config; print(get_config()['iam']['roles']['sagemaker_execution']['arn'])")

# Set stack name if not provided
if [ -z "$STACK_NAME" ]; then
  STACK_NAME="${PROJECT_NAME}-endpoint-stack"
fi

echo "Project name: $PROJECT_NAME"
echo "Model name: $MODEL_NAME"
echo "SageMaker role ARN: $SAGEMAKER_ROLE_ARN"
echo "Stack name: $STACK_NAME"
echo "Lambda code path: $LAMBDA_CODE_PATH"

# Check if Lambda code exists and warn if not
if [ ! -f "$LAMBDA_CODE_PATH/deploy_endpoint_lambda.py" ]; then
  echo "Warning: Lambda code not found at $LAMBDA_CODE_PATH/deploy_endpoint_lambda.py"
  echo "Using default inline Lambda code instead."
fi

# Navigate to CDK directory
cd "$PROJECT_ROOT/configs/cdk"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing CDK dependencies..."
  npm install
fi

# Deploy CDK stack
echo "Deploying CDK stack..."
export LAMBDA_CODE_PATH=$LAMBDA_CODE_PATH

# First try to synthesize the stack to check for errors
echo "Synthesizing CDK stack to check for errors..."
if npx cdk synth $STACK_NAME --app "npx ts-node bin/app.ts"; then
  echo "CDK synthesis successful, proceeding with deployment..."
  npx cdk deploy $STACK_NAME \
    --app "npx ts-node bin/app.ts" \
    --parameters projectName=$PROJECT_NAME \
    --parameters modelName=$MODEL_NAME \
    --parameters sagemakerRoleArn=$SAGEMAKER_ROLE_ARN \
    --parameters lambdaCodePath=$LAMBDA_CODE_PATH \
    --require-approval never
else
  echo "CDK synthesis failed. Skipping deployment."
  exit 1
fi

# Get Lambda function ARN
LAMBDA_ARN=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='DeployEndpointLambdaArn'].OutputValue" --output text)

echo "Lambda function ARN: $LAMBDA_ARN"

# Update project configuration with Lambda ARN
echo "Updating project configuration with Lambda ARN..."
CONFIG_FILE="$PROJECT_ROOT/configs/project_config.py"

# Check if deployment section exists in config
if grep -q "deployment" "$CONFIG_FILE"; then
  # Update existing deployment section
  sed -i '' "s|'lambda_arn': '.*'|'lambda_arn': '$LAMBDA_ARN'|g" "$CONFIG_FILE"
else
  # Add deployment section
  awk -v arn="$LAMBDA_ARN" '
  /COST_ALLOCATION_TAGS = {/ {
    print;
    print "}\n";
    print "# Deployment Configuration";
    print "DEPLOYMENT_CONFIG = {";
    print "    \"lambda_arn\": \"" arn "\"";
    print "}";
    next;
  }
  /}/ && !done {
    if (prev ~ /CostCenter/) {
      done = 1;
      next;
    }
  }
  { prev = $0; print }
  ' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
fi

echo "Deployment completed successfully!"
echo "You can now use the Lambda function ARN in your SageMaker pipeline."
echo "Lambda ARN: $LAMBDA_ARN"