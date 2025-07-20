# Ml Engineer Notebooks

Core notebooks for ML Engineers

## Notebooks Included

### ml-engineer-core-enhanced.ipynb

- **Purpose**: Core functionality for ml engineer
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
   - Appropriate IAM permissions for your role

4. **Running Notebooks**
   - Open any notebook by double-clicking
   - Select the appropriate kernel (Python 3)
   - Run cells sequentially

## Notebook Descriptions

### ML Engineer Core Enhanced
- **Purpose**: Training pipeline execution with MLFlow and Model Registry
- **Features**:
  - Pipeline configuration and execution
  - SageMaker training job management
  - MLFlow experiment tracking
  - Model Registry integration
  - Training monitoring and metrics

## MLFlow Integration

All notebooks include MLFlow integration for experiment tracking:

- **Experiment Tracking**: All runs are automatically tracked
- **Parameter Logging**: Training parameters and configurations
- **Metric Logging**: Performance metrics and results
- **Artifact Storage**: Visualizations and model artifacts
- **Model Registry**: Automatic model registration (ML Engineer)

### Accessing MLFlow UI

1. In SageMaker Studio, go to "Experiments and trials"
2. Click on "MLflow" to access the tracking UI
3. View experiments, runs, and artifacts

## Support

For issues or questions:
1. Check the notebook documentation
2. Review the MLOps SageMaker Demo project documentation
3. Contact your team lead or DevOps team

## Last Updated

Generated on: 2025-07-20 13:29:32
