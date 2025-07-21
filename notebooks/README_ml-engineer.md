# ML Engineer Notebooks

Core notebooks for ML Engineers with MLflow integration and Model Registry support.

## Notebooks Included

### ml-engineer-core-enhanced.ipynb

- **Purpose**: Pipeline execution and model management with comprehensive MLflow tracking
- **Location**: `/home/sagemaker-user/ml-engineer-notebooks/ml-engineer-core-enhanced.ipynb`

## Getting Started

1. **Open SageMaker Studio**
   - Navigate to your SageMaker Studio environment
   - Go to the file browser

2. **Navigate to Notebooks**
   - Go to: `/home/sagemaker-user/ml-engineer-notebooks/`
   - You'll find all the notebooks for your role

3. **Prerequisites**
   - AWS CLI configured with "ab" profile
   - Access to S3 bucket: `lucaskle-ab3-project-pv`
   - Appropriate IAM permissions for your role (MLEngineerRole)
   - SageMaker managed MLflow tracking server access

4. **Running Notebooks**
   - Open any notebook by double-clicking
   - Select the appropriate kernel (Python 3)
   - Run cells sequentially

## Notebook Descriptions

### ML Engineer Core Enhanced
- **Purpose**: Training pipeline execution with MLflow and Model Registry integration
- **Features**:
  - Pipeline configuration and execution
  - SageMaker training job management
  - MLflow experiment tracking with SageMaker managed server
  - Model Registry integration with approval workflows
  - Training monitoring and metrics

## MLFlow Integration

All notebooks include MLFlow integration for experiment tracking:

- **Connection Method**: Uses tracking server ARN format for SageMaker managed MLflow
- **Experiment Tracking**: All training runs are automatically tracked
- **Parameter Logging**: Training parameters and configurations
- **Metric Logging**: Performance metrics and results
- **Artifact Storage**: Model artifacts and training outputs
- **Model Registry**: Automatic model registration in SageMaker Model Registry
- **Authentication**: Automatic AWS authentication through SageMaker roles

### MLflow Configuration

The notebooks are configured to use the SageMaker managed MLflow tracking server:
- **Tracking Server ARN**: `arn:aws:sagemaker:us-east-1:192771711075:mlflow-tracking-server/sagemaker-core-setup-mlflow-server`
- **Authentication**: Handled automatically through IAM roles with `sagemaker-mlflow:*` permissions
- **Fallback**: Local MLflow tracking if SageMaker managed server is unavailable

### Accessing MLFlow UI

1. In SageMaker Studio, go to "Experiments and trials"
2. Click on "MLflow" to access the tracking UI
3. View experiments, runs, and artifacts
4. Compare different training runs and model versions

## Support

For issues or questions:
1. Check the notebook documentation
2. Review the MLOps SageMaker Demo project documentation
3. Contact your team lead or DevOps team

## Last Updated

Generated on: 2025-07-20 13:29:32
