#!/bin/bash
# Complete Infrastructure Deployment Script for MLOps SageMaker Demo
# This script deploys all required AWS infrastructure components

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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
    $PIP install boto3 sagemaker pandas matplotlib seaborn numpy PyYAML xmltodict
    print_success "Required packages installed"
fi

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

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default values
AWS_PROFILE="ab"
AWS_REGION="us-east-1"
PROJECT_NAME="mlops-sagemaker-demo"
DATA_BUCKET_NAME="lucaskle-ab3-project-pv"
ENDPOINT_NAME="${PROJECT_NAME}-yolov11-endpoint"
MONITORING_SCHEDULE_NAME="${PROJECT_NAME}-monitoring"
EMAIL_NOTIFICATIONS=""
SKIP_VALIDATION=false
SKIP_MONITORING=false
SKIP_MLFLOW=false
SKIP_EVENTBRIDGE=false
SKIP_COST_MONITORING=false

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
    --endpoint-name)
      ENDPOINT_NAME="$2"
      shift
      shift
      ;;
    --email)
      EMAIL_NOTIFICATIONS="$2"
      shift
      shift
      ;;
    --skip-validation)
      SKIP_VALIDATION=true
      shift
      ;;
    --skip-monitoring)
      SKIP_MONITORING=true
      shift
      ;;
    --skip-mlflow)
      SKIP_MLFLOW=true
      shift
      ;;
    --skip-eventbridge)
      SKIP_EVENTBRIDGE=true
      shift
      ;;
    --skip-cost-monitoring)
      SKIP_COST_MONITORING=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Deploy complete infrastructure for MLOps SageMaker Demo"
      echo ""
      echo "Options:"
      echo "  --profile PROFILE            AWS CLI profile to use (default: ab)"
      echo "  --region REGION              AWS region (default: us-east-1)"
      echo "  --project-name NAME          Project name (default: mlops-sagemaker-demo)"
      echo "  --data-bucket BUCKET         Data bucket name (default: lucaskle-ab3-project-pv)"
      echo "  --endpoint-name NAME         SageMaker endpoint name (default: mlops-sagemaker-demo-yolov11-endpoint)"
      echo "  --email EMAIL                Email address for notifications"
      echo "  --skip-validation            Skip validation steps"
      echo "  --skip-monitoring            Skip model monitoring setup"
      echo "  --skip-mlflow                Skip MLFlow setup"
      echo "  --skip-eventbridge           Skip EventBridge setup"
      echo "  --skip-cost-monitoring       Skip cost monitoring setup"
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
echo "Data Bucket: $DATA_BUCKET_NAME"
echo "Endpoint Name: $ENDPOINT_NAME"
echo "Email Notifications: $EMAIL_NOTIFICATIONS"
echo ""

# Check AWS CLI configuration
print_header "Checking AWS CLI Configuration"
bash "$SCRIPT_DIR/configure_aws.sh"

# Deploy IAM roles
print_header "Deploying IAM Roles and Policies"
bash "$SCRIPT_DIR/deploy_iam_roles.sh" --profile $AWS_PROFILE --stack-name "${PROJECT_NAME}-iam-roles" --bucket $DATA_BUCKET_NAME

# Validate IAM roles if not skipped
if [ "$SKIP_VALIDATION" = false ]; then
    print_header "Validating IAM Roles"
    $PYTHON "$SCRIPT_DIR/validate_iam_roles.py" --profile $AWS_PROFILE
fi

# Deploy CDK stacks
print_header "Deploying CDK Stacks"
bash "$SCRIPT_DIR/deploy_cdk.sh" --profile $AWS_PROFILE --region $AWS_REGION --stack-name "${PROJECT_NAME}-endpoint-stack"

# Validate CDK deployment if not skipped
if [ "$SKIP_VALIDATION" = false ]; then
    print_header "Validating CDK Deployment"
    bash "$SCRIPT_DIR/validate_cdk_deployment.sh"
    
    print_header "Validating CDK Security"
    bash "$SCRIPT_DIR/validate_cdk_security.sh"
fi

# Setup MLFlow if not skipped
if [ "$SKIP_MLFLOW" = false ]; then
    print_header "Setting up MLFlow on SageMaker"
    $PYTHON "$SCRIPT_DIR/setup_mlflow_sagemaker.py" --create-server --aws-profile $AWS_PROFILE --region $AWS_REGION --artifact-bucket "${PROJECT_NAME}-mlflow-artifacts"
fi

# Setup SageMaker Project if not skipped
print_header "Setting up SageMaker Project"
$PYTHON "$SCRIPT_DIR/setup_sagemaker_project.py" --profile $AWS_PROFILE --project-name $PROJECT_NAME --s3-bucket $DATA_BUCKET_NAME

# Deploy model to endpoint
print_header "Deploying Model to SageMaker Endpoint"
MODEL_INFO_PATH="$PROJECT_ROOT/configs/model_info.json"

# Check if model info exists, if not create a placeholder
if [ ! -f "$MODEL_INFO_PATH" ]; then
    print_warning "Model info file not found. Creating placeholder..."
    cat > "$MODEL_INFO_PATH" << EOF
{
    "model_name": "yolov11-drone-detection",
    "model_artifact_path": "s3://$DATA_BUCKET_NAME/models/yolov11/model.tar.gz",
    "inference_image": "763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/pytorch-inference:1.12.1-gpu-py38",
    "training_image": "763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/pytorch-training:1.12.1-gpu-py38",
    "framework": "pytorch",
    "framework_version": "1.12.1"
}
EOF
fi

$PYTHON "$PROJECT_ROOT/scripts/deployment/deploy_model.py" --profile $AWS_PROFILE --model-info $MODEL_INFO_PATH --endpoint-name $ENDPOINT_NAME --endpoint-type staging --instance-type ml.g4dn.xlarge --instance-count 1

# Setup model monitoring if not skipped
if [ "$SKIP_MONITORING" = false ]; then
    print_header "Setting up Model Monitoring"
    $PYTHON "$PROJECT_ROOT/scripts/monitoring/setup_model_monitoring.py" --profile $AWS_PROFILE --endpoint-name $ENDPOINT_NAME --monitoring-schedule-name $MONITORING_SCHEDULE_NAME
    
    print_header "Setting up SageMaker Clarify"
    $PYTHON "$PROJECT_ROOT/scripts/training/setup_clarify_explainability.py" --profile $AWS_PROFILE --endpoint-name $ENDPOINT_NAME
fi

# Setup EventBridge rules and SNS notifications if not skipped
if [ "$SKIP_EVENTBRIDGE" = false ] && [ -n "$EMAIL_NOTIFICATIONS" ]; then
    print_header "Setting up EventBridge Rules and SNS Notifications"
    
    # Create pipeline failure alert
    $PYTHON -c "
import sys
sys.path.append('$PROJECT_ROOT')
from src.pipeline.event_bridge_integration import create_pipeline_failure_alert
create_pipeline_failure_alert('${PROJECT_NAME}-pipeline', ['$EMAIL_NOTIFICATIONS'], '$AWS_PROFILE')
"
    
    # Create model drift alert
    $PYTHON -c "
import sys
sys.path.append('$PROJECT_ROOT')
from src.pipeline.event_bridge_integration import create_model_drift_alert
create_model_drift_alert('$ENDPOINT_NAME', ['$EMAIL_NOTIFICATIONS'], '$AWS_PROFILE')
"
fi

# Setup cost monitoring if not skipped
if [ "$SKIP_COST_MONITORING" = false ]; then
    print_header "Setting up Cost Monitoring"
    
    # Create cost monitoring dashboard
    $PYTHON -c "
import sys
sys.path.append('$PROJECT_ROOT')
from src.monitoring.cost_tracking import CostTracker
tracker = CostTracker(profile_name='$AWS_PROFILE')
report = tracker.generate_cost_report(output_format='html', output_file='$PROJECT_ROOT/cost_report.html')
print('Cost report generated at $PROJECT_ROOT/cost_report.html')

# Create cost budget if email is provided
if '$EMAIL_NOTIFICATIONS':
    tracker.create_cost_budget(
        budget_name='${PROJECT_NAME}-monthly-budget',
        budget_amount=100.0,  # $100 monthly budget
        notification_email='$EMAIL_NOTIFICATIONS',
        threshold_percent=80.0
    )
    print('Cost budget created with notification to $EMAIL_NOTIFICATIONS')
"
fi

# Execute infrastructure validation and health checks
if [ "$SKIP_VALIDATION" = false ]; then
    print_header "Executing Infrastructure Validation and Health Checks"
    
    # Check endpoint status
    $PYTHON "$PROJECT_ROOT/scripts/deployment/check_endpoint_metrics.py" --profile $AWS_PROFILE --endpoint-name $ENDPOINT_NAME
    
    # Validate pipeline integration
    $PYTHON -c "
import sys
sys.path.append('$PROJECT_ROOT')
import boto3
session = boto3.Session(profile_name='$AWS_PROFILE')
sagemaker_client = session.client('sagemaker')

# Check SageMaker domain
try:
    domains = sagemaker_client.list_domains()
    print(f'Found {len(domains.get(\"Domains\", []))} SageMaker domains')
    for domain in domains.get('Domains', []):
        print(f'Domain: {domain[\"DomainName\"]} (Status: {domain[\"Status\"]})')
except Exception as e:
    print(f'Error checking SageMaker domains: {str(e)}')

# Check SageMaker pipelines
try:
    pipelines = sagemaker_client.list_pipelines()
    print(f'Found {len(pipelines.get(\"PipelineSummaries\", []))} SageMaker pipelines')
    for pipeline in pipelines.get('PipelineSummaries', []):
        print(f'Pipeline: {pipeline[\"PipelineName\"]} (Status: {pipeline[\"PipelineStatus\"]})')
except Exception as e:
    print(f'Error checking SageMaker pipelines: {str(e)}')

# Check model registry
try:
    model_packages = sagemaker_client.list_model_packages(
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10
    )
    print(f'Found {len(model_packages.get(\"ModelPackageSummaryList\", []))} approved model packages')
    for model_package in model_packages.get('ModelPackageSummaryList', []):
        print(f'Model Package: {model_package[\"ModelPackageName\"]} (Status: {model_package[\"ModelApprovalStatus\"]})')
except Exception as e:
    print(f'Error checking model registry: {str(e)}')
"
fi

print_header "Infrastructure Deployment Complete"
print_success "All infrastructure components have been deployed successfully!"
print_info "Next steps:"
print_info "1. Access SageMaker Studio to start using the MLOps environment"
print_info "2. Run the demo notebooks in notebooks/demo/ to explore the functionality"
print_info "3. Monitor costs and performance using the generated dashboards"
print_info "4. Check your email for subscription confirmation if you provided an email address"

if [ "$SKIP_VALIDATION" = true ]; then
    print_warning "Validation was skipped. Run validation scripts manually if needed."
fi

if [ "$SKIP_MONITORING" = true ]; then
    print_warning "Model monitoring was skipped. Set up monitoring manually if needed."
fi

if [ "$SKIP_MLFLOW" = true ]; then
    print_warning "MLFlow setup was skipped. Set up MLFlow manually if needed."
fi

if [ "$SKIP_EVENTBRIDGE" = true ] || [ -z "$EMAIL_NOTIFICATIONS" ]; then
    print_warning "EventBridge setup was skipped. Set up EventBridge rules manually if needed."
fi

if [ "$SKIP_COST_MONITORING" = true ]; then
    print_warning "Cost monitoring was skipped. Set up cost monitoring manually if needed."
fi

print_info "To clean up resources when no longer needed, run: ./scripts/setup/cleanup_resources.sh"

# Usage examples
echo ""
print_info "Usage examples:"
print_info "1. Deploy all infrastructure:"
print_info "   ./scripts/setup/deploy_complete_infrastructure.sh --profile ab --email your.email@example.com"
print_info ""
print_info "2. Deploy without monitoring:"
print_info "   ./scripts/setup/deploy_complete_infrastructure.sh --profile ab --skip-monitoring"
print_info ""
print_info "3. Validate deployment:"
print_info "   venv/bin/python scripts/setup/validate_deployment.py --profile ab"
print_info ""
print_info "4. Clean up resources:"
print_info "   ./scripts/setup/cleanup_resources.sh --profile ab"
print_info ""
print_info "Note: All Python commands use the virtual environment in 'venv' directory."