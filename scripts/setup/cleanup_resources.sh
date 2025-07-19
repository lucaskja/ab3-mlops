#!/bin/bash
# Cleanup script for MLOps SageMaker Demo resources
# This script removes all AWS resources created for the demo

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

print_header() {
    echo -e "\n${GREEN}========== $1 ==========${NC}\n"
}

# Check if venv exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Use Python from venv
PYTHON="venv/bin/python"
PIP="venv/bin/pip"

# Check and install dependencies
print_info "Checking dependencies..."
if [ -f "requirements.txt" ]; then
    print_info "Installing dependencies from requirements.txt..."
    $PIP install -r requirements.txt
    print_success "Dependencies installed"
else
    print_info "Installing required packages..."
    $PIP install boto3 sagemaker
    print_success "Required packages installed"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default values
AWS_PROFILE="ab"
AWS_REGION="us-east-1"
PROJECT_NAME="mlops-sagemaker-demo"
ENDPOINT_NAME="${PROJECT_NAME}-yolov11-endpoint"
FORCE=false
SKIP_CONFIRMATION=false

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
    --endpoint-name)
      ENDPOINT_NAME="$2"
      shift
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --yes)
      SKIP_CONFIRMATION=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Clean up AWS resources for MLOps SageMaker Demo"
      echo ""
      echo "Options:"
      echo "  --profile PROFILE            AWS CLI profile to use (default: ab)"
      echo "  --region REGION              AWS region (default: us-east-1)"
      echo "  --project-name NAME          Project name (default: mlops-sagemaker-demo)"
      echo "  --endpoint-name NAME         SageMaker endpoint name (default: mlops-sagemaker-demo-yolov11-endpoint)"
      echo "  --force                      Force cleanup even if resources are in use"
      echo "  --yes                        Skip confirmation prompts"
      echo "  --help                       Show this help message"
      exit 0
      ;;
    *)
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set AWS profile
export AWS_PROFILE=$AWS_PROFILE
export AWS_DEFAULT_REGION=$AWS_REGION

# Print configuration
print_header "Configuration"
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Project Name: $PROJECT_NAME"
echo "Endpoint Name: $ENDPOINT_NAME"
echo "Force: $FORCE"
echo "Skip Confirmation: $SKIP_CONFIRMATION"
echo ""

# Confirm cleanup
if [ "$SKIP_CONFIRMATION" = false ]; then
    echo -e "${RED}WARNING: This script will delete all AWS resources created for the MLOps SageMaker Demo.${NC}"
    echo -e "${RED}This action cannot be undone.${NC}"
    echo ""
    read -p "Are you sure you want to proceed? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleanup cancelled."
        exit 0
    fi
fi

# Delete SageMaker endpoint
print_header "Deleting SageMaker Endpoint"
if aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --profile $AWS_PROFILE &> /dev/null; then
    print_info "Deleting endpoint: $ENDPOINT_NAME"
    aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME --profile $AWS_PROFILE
    print_success "Endpoint deletion initiated"
    
    # Get endpoint config name
    ENDPOINT_CONFIG_NAME=$(aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --query 'EndpointConfigName' --output text --profile $AWS_PROFILE 2>/dev/null || echo "")
    
    if [ -n "$ENDPOINT_CONFIG_NAME" ]; then
        print_info "Waiting for endpoint to be deleted..."
        aws sagemaker wait endpoint-deleted --endpoint-name $ENDPOINT_NAME --profile $AWS_PROFILE
        
        print_info "Deleting endpoint config: $ENDPOINT_CONFIG_NAME"
        aws sagemaker delete-endpoint-config --endpoint-config-name $ENDPOINT_CONFIG_NAME --profile $AWS_PROFILE
        print_success "Endpoint config deleted"
        
        # Get model names
        MODEL_NAMES=$(aws sagemaker describe-endpoint-config --endpoint-config-name $ENDPOINT_CONFIG_NAME --query 'ProductionVariants[*].ModelName' --output text --profile $AWS_PROFILE 2>/dev/null || echo "")
        
        if [ -n "$MODEL_NAMES" ]; then
            for MODEL_NAME in $MODEL_NAMES; do
                print_info "Deleting model: $MODEL_NAME"
                aws sagemaker delete-model --model-name $MODEL_NAME --profile $AWS_PROFILE
                print_success "Model deleted"
            done
        fi
    fi
else
    print_warning "Endpoint $ENDPOINT_NAME does not exist"
fi

# Delete monitoring schedules
print_header "Deleting Monitoring Schedules"
MONITORING_SCHEDULES=$(aws sagemaker list-monitoring-schedules --query 'MonitoringScheduleSummaries[?contains(MonitoringScheduleName, `'$PROJECT_NAME'`)].MonitoringScheduleName' --output text --profile $AWS_PROFILE)

if [ -n "$MONITORING_SCHEDULES" ]; then
    for SCHEDULE_NAME in $MONITORING_SCHEDULES; do
        print_info "Deleting monitoring schedule: $SCHEDULE_NAME"
        aws sagemaker delete-monitoring-schedule --monitoring-schedule-name $SCHEDULE_NAME --profile $AWS_PROFILE
        print_success "Monitoring schedule deletion initiated"
    done
else
    print_warning "No monitoring schedules found for $PROJECT_NAME"
fi

# Delete MLFlow tracking server
print_header "Deleting MLFlow Tracking Server"
MLFLOW_ENDPOINT=$(aws sagemaker list-endpoints --query 'Endpoints[?contains(EndpointName, `mlflow`)].EndpointName' --output text --profile $AWS_PROFILE)

if [ -n "$MLFLOW_ENDPOINT" ]; then
    print_info "Deleting MLFlow endpoint: $MLFLOW_ENDPOINT"
    aws sagemaker delete-endpoint --endpoint-name $MLFLOW_ENDPOINT --profile $AWS_PROFILE
    print_success "MLFlow endpoint deletion initiated"
    
    # Get endpoint config name
    MLFLOW_CONFIG_NAME=$(aws sagemaker describe-endpoint --endpoint-name $MLFLOW_ENDPOINT --query 'EndpointConfigName' --output text --profile $AWS_PROFILE 2>/dev/null || echo "")
    
    if [ -n "$MLFLOW_CONFIG_NAME" ]; then
        print_info "Waiting for MLFlow endpoint to be deleted..."
        aws sagemaker wait endpoint-deleted --endpoint-name $MLFLOW_ENDPOINT --profile $AWS_PROFILE
        
        print_info "Deleting MLFlow endpoint config: $MLFLOW_CONFIG_NAME"
        aws sagemaker delete-endpoint-config --endpoint-config-name $MLFLOW_CONFIG_NAME --profile $AWS_PROFILE
        print_success "MLFlow endpoint config deleted"
        
        # Get model names
        MLFLOW_MODEL_NAMES=$(aws sagemaker describe-endpoint-config --endpoint-config-name $MLFLOW_CONFIG_NAME --query 'ProductionVariants[*].ModelName' --output text --profile $AWS_PROFILE 2>/dev/null || echo "")
        
        if [ -n "$MLFLOW_MODEL_NAMES" ]; then
            for MODEL_NAME in $MLFLOW_MODEL_NAMES; do
                print_info "Deleting MLFlow model: $MODEL_NAME"
                aws sagemaker delete-model --model-name $MODEL_NAME --profile $AWS_PROFILE
                print_success "MLFlow model deleted"
            done
        fi
    fi
else
    print_warning "No MLFlow endpoint found"
fi

# Delete SageMaker Project
print_header "Deleting SageMaker Project"
PROJECT_ID=$(aws sagemaker list-projects --query "ProjectSummaryList[?ProjectName=='$PROJECT_NAME'].ProjectId" --output text --profile $AWS_PROFILE)

if [ -n "$PROJECT_ID" ]; then
    print_info "Deleting SageMaker project: $PROJECT_NAME (ID: $PROJECT_ID)"
    aws sagemaker delete-project --project-name $PROJECT_NAME --profile $AWS_PROFILE
    print_success "SageMaker project deletion initiated"
else
    print_warning "No SageMaker project found with name $PROJECT_NAME"
fi

# Delete EventBridge rules
print_header "Deleting EventBridge Rules"
RULES=$(aws events list-rules --name-prefix $PROJECT_NAME --query 'Rules[*].Name' --output text --profile $AWS_PROFILE)

if [ -n "$RULES" ]; then
    for RULE_NAME in $RULES; do
        # Remove targets
        TARGETS=$(aws events list-targets-by-rule --rule $RULE_NAME --query 'Targets[*].Id' --output text --profile $AWS_PROFILE)
        
        if [ -n "$TARGETS" ]; then
            print_info "Removing targets from rule: $RULE_NAME"
            aws events remove-targets --rule $RULE_NAME --ids $TARGETS --profile $AWS_PROFILE
            print_success "Targets removed"
        fi
        
        print_info "Deleting EventBridge rule: $RULE_NAME"
        aws events delete-rule --name $RULE_NAME --profile $AWS_PROFILE
        print_success "EventBridge rule deleted"
    done
else
    print_warning "No EventBridge rules found with prefix $PROJECT_NAME"
fi

# Delete SNS topics
print_header "Deleting SNS Topics"
TOPICS=$(aws sns list-topics --query "Topics[?contains(TopicArn, '$PROJECT_NAME')].TopicArn" --output text --profile $AWS_PROFILE)

if [ -n "$TOPICS" ]; then
    for TOPIC_ARN in $TOPICS; do
        print_info "Deleting SNS topic: $TOPIC_ARN"
        aws sns delete-topic --topic-arn $TOPIC_ARN --profile $AWS_PROFILE
        print_success "SNS topic deleted"
    done
else
    print_warning "No SNS topics found for $PROJECT_NAME"
fi

# Delete Lambda functions
print_header "Deleting Lambda Functions"
FUNCTIONS=$(aws lambda list-functions --query "Functions[?contains(FunctionName, '$PROJECT_NAME')].FunctionName" --output text --profile $AWS_PROFILE)

if [ -n "$FUNCTIONS" ]; then
    for FUNCTION_NAME in $FUNCTIONS; do
        print_info "Deleting Lambda function: $FUNCTION_NAME"
        aws lambda delete-function --function-name $FUNCTION_NAME --profile $AWS_PROFILE
        print_success "Lambda function deleted"
    done
else
    print_warning "No Lambda functions found for $PROJECT_NAME"
fi

# Delete S3 buckets
print_header "Deleting S3 Buckets"
MLFLOW_BUCKET="${PROJECT_NAME}-mlflow-artifacts"

if aws s3 ls "s3://$MLFLOW_BUCKET" --profile $AWS_PROFILE &> /dev/null; then
    print_info "Emptying bucket: $MLFLOW_BUCKET"
    aws s3 rm "s3://$MLFLOW_BUCKET" --recursive --profile $AWS_PROFILE
    
    print_info "Deleting bucket: $MLFLOW_BUCKET"
    aws s3 rb "s3://$MLFLOW_BUCKET" --profile $AWS_PROFILE
    print_success "S3 bucket deleted"
else
    print_warning "S3 bucket $MLFLOW_BUCKET does not exist"
fi

# Delete CDK stacks
print_header "Deleting CDK Stacks"
ENDPOINT_STACK="${PROJECT_NAME}-endpoint-stack"

if aws cloudformation describe-stacks --stack-name $ENDPOINT_STACK --profile $AWS_PROFILE &> /dev/null; then
    print_info "Deleting CloudFormation stack: $ENDPOINT_STACK"
    aws cloudformation delete-stack --stack-name $ENDPOINT_STACK --profile $AWS_PROFILE
    
    print_info "Waiting for stack deletion to complete..."
    aws cloudformation wait stack-delete-complete --stack-name $ENDPOINT_STACK --profile $AWS_PROFILE
    print_success "CloudFormation stack deleted"
else
    print_warning "CloudFormation stack $ENDPOINT_STACK does not exist"
fi

# Delete IAM roles and policies
print_header "Deleting IAM Roles and Policies"
IAM_STACK="${PROJECT_NAME}-iam-roles"

if aws cloudformation describe-stacks --stack-name $IAM_STACK --profile $AWS_PROFILE &> /dev/null; then
    print_info "Deleting CloudFormation stack: $IAM_STACK"
    aws cloudformation delete-stack --stack-name $IAM_STACK --profile $AWS_PROFILE
    
    print_info "Waiting for stack deletion to complete..."
    aws cloudformation wait stack-delete-complete --stack-name $IAM_STACK --profile $AWS_PROFILE
    print_success "CloudFormation stack deleted"
else
    print_warning "CloudFormation stack $IAM_STACK does not exist"
fi

# Delete cost budgets
print_header "Deleting Cost Budgets"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile $AWS_PROFILE)
BUDGETS=$(aws budgets describe-budgets --account-id $ACCOUNT_ID --query "Budgets[?contains(BudgetName, '$PROJECT_NAME')].BudgetName" --output text --profile $AWS_PROFILE 2>/dev/null || echo "")

if [ -n "$BUDGETS" ]; then
    for BUDGET_NAME in $BUDGETS; do
        print_info "Deleting budget: $BUDGET_NAME"
        aws budgets delete-budget --account-id $ACCOUNT_ID --budget-name "$BUDGET_NAME" --profile $AWS_PROFILE
        print_success "Budget deleted"
    done
else
    print_warning "No budgets found for $PROJECT_NAME"
fi

print_header "Cleanup Complete"
print_success "All resources for $PROJECT_NAME have been deleted or deletion has been initiated."
print_info "Some resources may take time to be fully deleted."
print_info "Check the AWS Management Console to verify all resources have been properly cleaned up."