# Data Scientist Notebooks

Core notebooks for Data Scientists

## Notebooks Included

### data-scientist-core-enhanced.ipynb

- **Purpose**: Core functionality for data scientist
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
   - Appropriate IAM permissions for your role

4. **Running Notebooks**
   - Open any notebook by double-clicking
   - Select the appropriate kernel (Python 3)
   - Run cells sequentially

## Notebook Descriptions

### Data Scientist Core Enhanced
- **Purpose**: Data exploration and preparation with MLFlow tracking
- **Features**:
  - S3 data access and exploration
  - Image analysis and visualization
  - Data preparation for YOLOv11
  - MLFlow experiment tracking
  - Dataset structure creation

### Create Labeling Job
- **Purpose**: Create SageMaker Ground Truth labeling jobs
- **Features**:
  - Interactive labeling job creation
  - Cost estimation
  - Job monitoring
  - Label format conversion

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
