# Technology Stack

## Core Technologies
- **Python 3.10+**: Primary development language
- **AWS SageMaker**: ML platform for training, deployment, and monitoring
- **YOLOv11 (Ultralytics)**: Object detection framework for drone detection
- **PyTorch**: Deep learning framework
- **MLFlow**: Experiment tracking and model registry
- **Jupyter Notebooks**: Interactive development and exploration

## AWS Services
- **S3**: Data storage and artifact management
- **SageMaker**: ML workflows, training, and inference
- **IAM**: Role-based access control
- **CloudFormation**: Infrastructure as code
- **EventBridge**: Pipeline event monitoring
- **SNS**: Notifications

## Key Libraries
- **Data Processing**: pandas, numpy, PIL, opencv-python
- **ML/AI**: scikit-learn, torch, torchvision, ultralytics
- **AWS Integration**: boto3, sagemaker SDK
- **Visualization**: matplotlib, seaborn
- **Validation**: PyYAML, xmltodict

## AWS Configuration
- **Profile**: Always use `ab` profile for AWS CLI operations
- **Region**: `us-east-1` (default)
- **Data Bucket**: `lucaskle-ab3-project-pv`

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS CLI
./scripts/setup/configure_aws.sh

# Deploy IAM roles
./scripts/setup/deploy_iam_roles.sh
```

### Development Workflow
```bash
# Set AWS profile for session
export AWS_PROFILE=ab

# Validate IAM setup
python3 scripts/setup/validate_iam_roles.py --profile ab

# Start Jupyter for notebook development
jupyter notebook
```

### AWS Operations
```bash
# List S3 bucket contents
aws s3 ls s3://lucaskle-ab3-project-pv --profile ab

# Check CloudFormation stack status
aws cloudformation describe-stacks --stack-name mlops-sagemaker-demo-iam-roles --profile ab
```

## Build System
- **Package Management**: pip with requirements.txt
- **Infrastructure**: CloudFormation templates and bash scripts
- **Testing**: Manual validation scripts (no automated test suite)
- **Deployment**: Shell scripts for AWS resource provisioning