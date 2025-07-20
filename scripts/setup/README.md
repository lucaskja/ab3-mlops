# SageMaker Core Setup Scripts

This directory contains scripts for deploying, validating, and cleaning up the core SageMaker infrastructure needed for YOLOv11 training.

## Scripts

### `deploy_core_sagemaker.sh`

Deploys the essential components needed for SageMaker Studio and YOLOv11 training.

```bash
./deploy_core_sagemaker.sh [OPTIONS]
```

Options:
- `--profile PROFILE`: AWS CLI profile to use (default: ab)
- `--region REGION`: AWS region (default: us-east-1)
- `--project-name NAME`: Project name (default: sagemaker-core-setup)
- `--data-bucket BUCKET`: Data bucket name (default: lucaskle-ab3-project-pv)
- `--skip-validation`: Skip validation steps
- `--skip-stack-wait`: Skip waiting for CloudFormation stack updates
- `--help`: Show help message

### `validate_core_sagemaker.py`

Validates that all required resources are properly deployed.

```bash
./validate_core_sagemaker.py [OPTIONS]
```

Options:
- `--profile PROFILE`: AWS CLI profile to use (default: ab)
- `--region REGION`: AWS region (default: us-east-1)
- `--project-name PROJECT_NAME`: Project name (default: sagemaker-core-setup)
- `--data-bucket DATA_BUCKET`: Data bucket name (default: lucaskle-ab3-project-pv)
- `--verbose`: Enable verbose output
- `-h, --help`: Show help message

### `cleanup_core_sagemaker.sh`

Removes all resources created by the deployment script.

```bash
./cleanup_core_sagemaker.sh [OPTIONS]
```

Options:
- `--profile PROFILE`: AWS CLI profile to use (default: ab)
- `--region REGION`: AWS region (default: us-east-1)
- `--project-name NAME`: Project name (default: sagemaker-core-setup)
- `--force`: Skip confirmation prompts
- `--help`: Show help message

## Usage Workflow

1. Deploy the core SageMaker infrastructure:
   ```bash
   ./deploy_core_sagemaker.sh --profile ab
   ```

2. Validate the deployment:
   ```bash
   ./validate_core_sagemaker.py --profile ab
   ```

3. When no longer needed, clean up the resources:
   ```bash
   ./cleanup_core_sagemaker.sh --profile ab
   ```

## Requirements

- AWS CLI configured with the "ab" profile
- Python 3.6+ with boto3 installed
- Proper IAM permissions to create and manage SageMaker resources