#!/bin/bash
# Cleanup script for MLOps SageMaker Demo resources
# This script deletes all AWS resources created by the MLOps SageMaker Demo

# Set strict error handling
set -e

# Set default AWS profile
AWS_PROFILE=${AWS_PROFILE:-ab}
AWS_REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME="mlops-sagemaker-demo"
DATA_BUCKET="lucaskle-ab3-project-pv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section header
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning message
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to confirm action
confirm_action() {
    read -p "Are you sure you want to proceed with cleanup? This will delete all resources created by the MLOps SageMaker Demo. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 1
    fi
}

# Function to delete SageMaker endpoints
delete_sagemaker_endpoints() {
    print_section "Deleting SageMaker Endpoints"
    
    # List endpoints
    endpoints=$(aws sagemaker list-endpoints --profile $AWS_PROFILE --region $AWS_REGION --query "Endpoints[?contains(EndpointName, '$PROJECT_NAME')].EndpointName" --output text)
    
    if [ -z "$endpoints" ]; then
        print_warning "No SageMaker endpoints found"
        return
    fi
    
    # Delete endpoints
    for endpoint in $endpoints; do
        echo "Deleting endpoint: $endpoint"
        aws sagemaker delete-endpoint --endpoint-name $endpoint --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted endpoint: $endpoint"
    done
}

# Function to delete SageMaker endpoint configs
delete_sagemaker_endpoint_configs() {
    print_section "Deleting SageMaker Endpoint Configs"
    
    # List endpoint configs
    endpoint_configs=$(aws sagemaker list-endpoint-configs --profile $AWS_PROFILE --region $AWS_REGION --query "EndpointConfigs[?contains(EndpointConfigName, '$PROJECT_NAME')].EndpointConfigName" --output text)
    
    if [ -z "$endpoint_configs" ]; then
        print_warning "No SageMaker endpoint configs found"
        return
    fi
    
    # Delete endpoint configs
    for config in $endpoint_configs; do
        echo "Deleting endpoint config: $config"
        aws sagemaker delete-endpoint-config --endpoint-config-name $config --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted endpoint config: $config"
    done
}

# Function to delete SageMaker models
delete_sagemaker_models() {
    print_section "Deleting SageMaker Models"
    
    # List models
    models=$(aws sagemaker list-models --profile $AWS_PROFILE --region $AWS_REGION --query "Models[?contains(ModelName, '$PROJECT_NAME')].ModelName" --output text)
    
    if [ -z "$models" ]; then
        print_warning "No SageMaker models found"
        return
    fi
    
    # Delete models
    for model in $models; do
        echo "Deleting model: $model"
        aws sagemaker delete-model --model-name $model --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted model: $model"
    done
}

# Function to delete SageMaker pipelines
delete_sagemaker_pipelines() {
    print_section "Deleting SageMaker Pipelines"
    
    # List pipelines
    pipelines=$(aws sagemaker list-pipelines --profile $AWS_PROFILE --region $AWS_REGION --query "PipelineSummaries[?contains(PipelineName, '$PROJECT_NAME')].PipelineName" --output text)
    
    if [ -z "$pipelines" ]; then
        print_warning "No SageMaker pipelines found"
        return
    fi
    
    # Delete pipelines
    for pipeline in $pipelines; do
        echo "Deleting pipeline: $pipeline"
        aws sagemaker delete-pipeline --pipeline-name $pipeline --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted pipeline: $pipeline"
    done
}

# Function to delete monitoring schedules
delete_monitoring_schedules() {
    print_section "Deleting Monitoring Schedules"
    
    # List monitoring schedules
    schedules=$(aws sagemaker list-monitoring-schedules --profile $AWS_PROFILE --region $AWS_REGION --query "MonitoringScheduleSummaries[?contains(MonitoringScheduleName, '$PROJECT_NAME')].MonitoringScheduleName" --output text)
    
    if [ -z "$schedules" ]; then
        print_warning "No monitoring schedules found"
        return
    fi
    
    # Delete monitoring schedules
    for schedule in $schedules; do
        echo "Deleting monitoring schedule: $schedule"
        aws sagemaker delete-monitoring-schedule --monitoring-schedule-name $schedule --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted monitoring schedule: $schedule"
    done
}

# Function to delete CloudWatch dashboards
delete_cloudwatch_dashboards() {
    print_section "Deleting CloudWatch Dashboards"
    
    # List dashboards
    dashboards=$(aws cloudwatch list-dashboards --profile $AWS_PROFILE --region $AWS_REGION --query "DashboardEntries[?contains(DashboardName, '$PROJECT_NAME')].DashboardName" --output text)
    
    if [ -z "$dashboards" ]; then
        print_warning "No CloudWatch dashboards found"
        return
    fi
    
    # Delete dashboards
    for dashboard in $dashboards; do
        echo "Deleting dashboard: $dashboard"
        aws cloudwatch delete-dashboards --dashboard-names $dashboard --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted dashboard: $dashboard"
    done
}

# Function to delete CloudWatch alarms
delete_cloudwatch_alarms() {
    print_section "Deleting CloudWatch Alarms"
    
    # List alarms
    alarms=$(aws cloudwatch describe-alarms --profile $AWS_PROFILE --region $AWS_REGION --query "MetricAlarms[?contains(AlarmName, '$PROJECT_NAME')].AlarmName" --output text)
    
    if [ -z "$alarms" ]; then
        print_warning "No CloudWatch alarms found"
        return
    fi
    
    # Delete alarms
    echo "Deleting alarms: $alarms"
    aws cloudwatch delete-alarms --alarm-names $alarms --profile $AWS_PROFILE --region $AWS_REGION
    print_success "Deleted alarms"
}

# Function to delete EventBridge rules
delete_eventbridge_rules() {
    print_section "Deleting EventBridge Rules"
    
    # List rules
    rules=$(aws events list-rules --profile $AWS_PROFILE --region $AWS_REGION --query "Rules[?contains(Name, '$PROJECT_NAME')].Name" --output text)
    
    if [ -z "$rules" ]; then
        print_warning "No EventBridge rules found"
        return
    fi
    
    # Delete rules
    for rule in $rules; do
        # List targets
        targets=$(aws events list-targets-by-rule --rule $rule --profile $AWS_PROFILE --region $AWS_REGION --query "Targets[].Id" --output text)
        
        # Remove targets
        if [ ! -z "$targets" ]; then
            echo "Removing targets from rule: $rule"
            aws events remove-targets --rule $rule --ids $targets --profile $AWS_PROFILE --region $AWS_REGION
        fi
        
        # Delete rule
        echo "Deleting rule: $rule"
        aws events delete-rule --name $rule --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted rule: $rule"
    done
}

# Function to empty S3 bucket
empty_s3_bucket() {
    print_section "Emptying S3 Bucket: $DATA_BUCKET"
    
    # Check if bucket exists
    if ! aws s3api head-bucket --bucket $DATA_BUCKET --profile $AWS_PROFILE 2>/dev/null; then
        print_warning "Bucket $DATA_BUCKET does not exist or you don't have access"
        return
    fi
    
    # Empty bucket
    echo "Emptying bucket: $DATA_BUCKET"
    aws s3 rm s3://$DATA_BUCKET --recursive --profile $AWS_PROFILE
    print_success "Emptied bucket: $DATA_BUCKET"
}

# Function to delete CloudFormation stacks
delete_cloudformation_stacks() {
    print_section "Deleting CloudFormation Stacks"
    
    # List stacks
    stacks=$(aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE --profile $AWS_PROFILE --region $AWS_REGION --query "StackSummaries[?contains(StackName, '$PROJECT_NAME')].StackName" --output text)
    
    if [ -z "$stacks" ]; then
        print_warning "No CloudFormation stacks found"
        return
    fi
    
    # Delete stacks
    for stack in $stacks; do
        echo "Deleting stack: $stack"
        aws cloudformation delete-stack --stack-name $stack --profile $AWS_PROFILE --region $AWS_REGION
        
        # Wait for stack deletion to complete
        echo "Waiting for stack deletion to complete..."
        aws cloudformation wait stack-delete-complete --stack-name $stack --profile $AWS_PROFILE --region $AWS_REGION
        print_success "Deleted stack: $stack"
    done
}

# Function to verify cleanup
verify_cleanup() {
    print_section "Verifying Cleanup"
    
    # Check endpoints
    endpoints=$(aws sagemaker list-endpoints --profile $AWS_PROFILE --region $AWS_REGION --query "Endpoints[?contains(EndpointName, '$PROJECT_NAME')].EndpointName" --output text)
    if [ ! -z "$endpoints" ]; then
        print_warning "Some SageMaker endpoints still exist: $endpoints"
    else
        print_success "No SageMaker endpoints found"
    fi
    
    # Check models
    models=$(aws sagemaker list-models --profile $AWS_PROFILE --region $AWS_REGION --query "Models[?contains(ModelName, '$PROJECT_NAME')].ModelName" --output text)
    if [ ! -z "$models" ]; then
        print_warning "Some SageMaker models still exist: $models"
    else
        print_success "No SageMaker models found"
    fi
    
    # Check pipelines
    pipelines=$(aws sagemaker list-pipelines --profile $AWS_PROFILE --region $AWS_REGION --query "PipelineSummaries[?contains(PipelineName, '$PROJECT_NAME')].PipelineName" --output text)
    if [ ! -z "$pipelines" ]; then
        print_warning "Some SageMaker pipelines still exist: $pipelines"
    else
        print_success "No SageMaker pipelines found"
    fi
    
    # Check stacks
    stacks=$(aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE --profile $AWS_PROFILE --region $AWS_REGION --query "StackSummaries[?contains(StackName, '$PROJECT_NAME')].StackName" --output text)
    if [ ! -z "$stacks" ]; then
        print_warning "Some CloudFormation stacks still exist: $stacks"
    else
        print_success "No CloudFormation stacks found"
    fi
}

# Main function
main() {
    echo -e "${BLUE}MLOps SageMaker Demo Cleanup Script${NC}"
    echo -e "${YELLOW}This script will delete all resources created by the MLOps SageMaker Demo.${NC}"
    echo -e "${YELLOW}AWS Profile: $AWS_PROFILE${NC}"
    echo -e "${YELLOW}AWS Region: $AWS_REGION${NC}"
    
    # Confirm action
    confirm_action
    
    # Delete resources
    delete_sagemaker_endpoints
    delete_sagemaker_endpoint_configs
    delete_sagemaker_models
    delete_sagemaker_pipelines
    delete_monitoring_schedules
    delete_cloudwatch_dashboards
    delete_cloudwatch_alarms
    delete_eventbridge_rules
    empty_s3_bucket
    delete_cloudformation_stacks
    
    # Verify cleanup
    verify_cleanup
    
    print_section "Cleanup Complete"
    echo -e "${GREEN}Cleanup process has completed. Please check the output for any warnings or errors.${NC}"
    echo -e "${YELLOW}To verify that all costs have been eliminated, check the AWS Cost Explorer.${NC}"
}

# Run main function
main