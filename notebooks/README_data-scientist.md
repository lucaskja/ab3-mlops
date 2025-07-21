# Data Scientist Notebooks

Core notebooks for Data Scientists with MLflow integration for experiment tracking.

## Notebooks Included

### data-scientist-core-enhanced.ipynb

- **Purpose**: Core functionality for data scientist with comprehensive MLflow tracking
- **Location**: `/home/sagemaker-user/data-scientist-notebooks/data-scientist-core-enhanced.ipynb`

## Getting Started

1. **Open SageMaker Studio**
   - Navigate to your SageMaker Studio environment
   - Go to the file browser

2. **Navigate to Notebooks**
   - Go to: `/home/sagemaker-user/data-scientist-notebooks/`
   - You'll find all the notebooks for your role

3. **Prerequisites**
   - AWS CLI configured with "ab" profile
   - Access to S3 bucket: `lucaskle-ab3-project-pv`
   - Appropriate IAM permissions for your role (DataScientistRole)
   - SageMaker managed MLflow tracking server access

4. **Running Notebooks**
   - Open any notebook by double-clicking
   - Select the appropriate kernel (Python 3)
   - Run cells sequentially

## Notebook Descriptions

### Data Scientist Core Enhanced
- **Purpose**: Data exploration and preparation with comprehensive MLflow tracking
- **Features**:
  - S3 data access and exploration
  - Image analysis and visualization
  - Data preparation for YOLOv11
  - MLflow experiment tracking with SageMaker managed server
  - Dataset structure creation
  - Ground Truth labeling job integration

## MLflow Integration

All notebooks include MLflow integration using SageMaker managed MLflow tracking server:

- **Connection Method**: Uses tracking server ARN format for SageMaker managed MLflow
- **Experiment Tracking**: All data exploration runs are automatically tracked
- **Parameter Logging**: Dataset characteristics and exploration parameters
- **Metric Logging**: Data quality metrics and statistics
- **Artifact Storage**: Visualizations and analysis results
- **Authentication**: Automatic AWS authentication through SageMaker roles

### MLflow Configuration

The notebooks are configured to use the SageMaker managed MLflow tracking server:
- **Tracking Server ARN**: `arn:aws:sagemaker:us-east-1:192771711075:mlflow-tracking-server/sagemaker-core-setup-mlflow-server`
- **Authentication**: Handled automatically through IAM roles with `sagemaker-mlflow:*` permissions
- **Fallback**: Local MLflow tracking if SageMaker managed server is unavailable

### Accessing MLflow UI

1. In SageMaker Studio, go to "Experiments and trials"
2. Click on "MLflow" to access the tracking UI
3. View experiments, runs, and artifacts
4. Compare different data exploration runs

## Troubleshooting

### MLflow Connection Issues
- Ensure your IAM role has `sagemaker-mlflow:*` permissions
- Verify the tracking server is running and accessible
- Check that you're using the correct tracking server ARN format

### S3 Access Issues
- Verify your role has read access to the `lucaskle-ab3-project-pv` bucket
- Ensure AWS profile "ab" is configured correctly
- Check that the bucket exists and contains the expected data

## Support

For issues or questions:
1. Check the notebook documentation and error messages
2. Review the MLOps SageMaker Demo project documentation
3. Verify IAM permissions and MLflow server status
4. Contact your team lead or DevOps team

## Last Updated

Generated on: 2025-07-21 15:00:00
