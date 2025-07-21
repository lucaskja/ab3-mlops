#!/bin/bash

# Setup Core SageMaker Pipeline with MLFlow Integration
# This script sets up both the core SageMaker pipeline and MLFlow server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
AWS_PROFILE="ab"
REGION="us-east-1"
SKIP_PIPELINE=false
SKIP_MLFLOW=false

# Function to print colored output
print_info() {
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

Setup Core SageMaker Pipeline with MLFlow Integration

OPTIONS:
    --profile PROFILE       AWS CLI profile to use (default: ab)
    --region REGION         AWS region (default: us-east-1)
    --skip-pipeline         Skip SageMaker pipeline setup
    --skip-mlflow          Skip MLFlow server setup
    --help                 Show this help message

EXAMPLES:
    $0 --profile ab
    $0 --profile ab --region us-west-2
    $0 --skip-pipeline --profile ab
    $0 --skip-mlflow --profile ab

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
            REGION="$2"
            shift 2
            ;;
        --skip-pipeline)
            SKIP_PIPELINE=true
            shift
            ;;
        --skip-mlflow)
            SKIP_MLFLOW=true
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

# Main setup function
main() {
    print_header "🚀 Core SageMaker Setup with MLFlow Integration"
    echo "=================================================="
    
    print_info "Configuration:"
    print_info "  AWS Profile: $AWS_PROFILE"
    print_info "  Region: $REGION"
    print_info "  Skip Pipeline: $SKIP_PIPELINE"
    print_info "  Skip MLFlow: $SKIP_MLFLOW"
    echo
    
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    
    print_info "Project root: $PROJECT_ROOT"
    print_info "Script directory: $SCRIPT_DIR"
    echo
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        print_error "Cannot find requirements.txt. Please run this script from the project root or check the path."
        exit 1
    fi
    
    # Verify AWS profile
    print_info "Verifying AWS profile: $AWS_PROFILE"
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
        print_error "Cannot access AWS with profile '$AWS_PROFILE'. Please check your AWS configuration."
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Arn --output text)
    print_info "AWS Account ID: $ACCOUNT_ID"
    print_info "User/Role: $USER_ARN"
    echo
    
    # Setup SageMaker Pipeline
    if [[ "$SKIP_PIPELINE" == "false" ]]; then
        print_header "📋 Setting up SageMaker Pipeline"
        print_info "Running core pipeline setup..."
        
        if [[ -f "$SCRIPT_DIR/setup_core_pipeline.sh" ]]; then
            "$SCRIPT_DIR/setup_core_pipeline.sh" --profile "$AWS_PROFILE"
            if [[ $? -eq 0 ]]; then
                print_info "✅ SageMaker pipeline setup completed successfully"
            else
                print_error "❌ SageMaker pipeline setup failed"
                exit 1
            fi
        else
            print_error "Cannot find setup_core_pipeline.sh script"
            exit 1
        fi
        echo
    else
        print_warning "Skipping SageMaker pipeline setup"
        echo
    fi
    
    # Setup MLFlow Server
    if [[ "$SKIP_MLFLOW" == "false" ]]; then
        print_header "🔬 Setting up MLFlow Server"
        print_info "Running MLFlow server setup..."
        
        # Check if virtual environment exists and activate it
        if [[ -d "$PROJECT_ROOT/venv" ]]; then
            print_info "Activating virtual environment..."
            source "$PROJECT_ROOT/venv/bin/activate"
        else
            print_warning "Virtual environment not found. Using system Python."
        fi
        
        if [[ -f "$SCRIPT_DIR/setup_mlflow_server.py" ]]; then
            python "$SCRIPT_DIR/setup_mlflow_server.py" --profile "$AWS_PROFILE" --region "$REGION"
            if [[ $? -eq 0 ]]; then
                print_info "✅ MLFlow server setup completed successfully"
            else
                print_error "❌ MLFlow server setup failed"
                exit 1
            fi
        else
            print_error "Cannot find setup_mlflow_server.py script"
            exit 1
        fi
        echo
    else
        print_warning "Skipping MLFlow server setup"
        echo
    fi
    
    # Final summary
    print_header "🎉 Setup Complete!"
    echo "=================="
    
    if [[ "$SKIP_PIPELINE" == "false" ]]; then
        print_info "✅ SageMaker Pipeline: Ready for execution"
    fi
    
    if [[ "$SKIP_MLFLOW" == "false" ]]; then
        print_info "✅ MLFlow Server: Ready for experiment tracking"
    fi
    
    echo
    print_header "📚 Next Steps"
    echo "1. Open SageMaker Studio and access your user profile"
    echo "2. Use the enhanced notebooks with MLFlow integration:"
    echo "   - notebooks/data-scientist-core-enhanced.ipynb"
    echo "   - notebooks/ml-engineer-core-enhanced.ipynb"
    echo "3. Start tracking your experiments with MLFlow!"
    echo
    print_header "🔗 Useful Links"
    echo "- SageMaker Console: https://$REGION.console.aws.amazon.com/sagemaker/home?region=$REGION"
    echo "- S3 Console: https://s3.console.aws.amazon.com/s3/buckets/lucaskle-ab3-project-pv"
    echo "- MLFlow Artifacts: s3://lucaskle-ab3-project-pv/mlflow/"
    echo
}

# Run main function
main "$@"
