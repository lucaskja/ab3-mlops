# SageMaker Studio Notebook Deployment Instructions

## Overview

This guide explains how to access and use the core notebooks deployed to your SageMaker Studio environment.

## For Data Scientists

### Accessing Your Notebooks

1. **Open SageMaker Studio**
   - Go to the AWS Console
   - Navigate to SageMaker > Studio
   - Click "Open Studio" for your user profile

2. **Navigate to Your Notebooks**
   - In the file browser, go to: `/home/sagemaker-user/data-scientist-notebooks/`
   - You'll find these notebooks:
     - `data-scientist-core-enhanced.ipynb` - Core data exploration with MLFlow
     - `create_labeling_job.ipynb` - Ground Truth labeling job creation

3. **Getting Started**
   - Open `data-scientist-core-enhanced.ipynb` first
   - Follow the instructions in the notebook
   - Ensure your AWS CLI is configured with the "ab" profile

### Key Features for Data Scientists

- **Data Exploration**: Analyze drone imagery datasets
- **MLFlow Tracking**: All exploration activities are tracked
- **Ground Truth Integration**: Create labeling jobs for annotation
- **Data Preparation**: Prepare data for YOLOv11 training

## For ML Engineers

### Accessing Your Notebooks

1. **Open SageMaker Studio**
   - Go to the AWS Console
   - Navigate to SageMaker > Studio
   - Click "Open Studio" for your user profile

2. **Navigate to Your Notebooks**
   - In the file browser, go to: `/home/sagemaker-user/ml-engineer-notebooks/`
   - You'll find this notebook:
     - `ml-engineer-core-enhanced.ipynb` - Training pipeline with MLFlow and Model Registry

3. **Getting Started**
   - Open `ml-engineer-core-enhanced.ipynb`
   - Follow the instructions in the notebook
   - Ensure your AWS CLI is configured with the "ab" profile

### Key Features for ML Engineers

- **Training Pipeline**: Execute YOLOv11 training jobs
- **MLFlow Integration**: Complete experiment tracking
- **Model Registry**: Automatic model registration
- **Pipeline Monitoring**: Real-time training monitoring

## MLFlow Integration

### Accessing MLFlow UI

1. In SageMaker Studio, look for "Experiments and trials" in the left sidebar
2. Click on "MLflow" to access the MLFlow tracking UI
3. Here you can:
   - View all experiments and runs
   - Compare different training runs
   - Access model artifacts and visualizations
   - Track model lineage

### Benefits of MLFlow Integration

- **Experiment Tracking**: Every data exploration and training run is tracked
- **Reproducibility**: All parameters and configurations are logged
- **Collaboration**: Team members can view and compare experiments
- **Model Management**: Complete model lifecycle tracking

## Prerequisites

### AWS Configuration

Ensure your environment has:
- AWS CLI configured with "ab" profile
- Access to S3 bucket: `lucaskle-ab3-project-pv`
- Appropriate IAM permissions for your role

### Checking AWS Configuration

Run this in a notebook cell to verify:

```python
import boto3
session = boto3.Session(profile_name='ab')
print(f"Account ID: {session.client('sts').get_caller_identity()['Account']}")
print(f"Region: {session.region_name}")
```

## Troubleshooting

### Common Issues

1. **AWS Profile Not Found**
   - Ensure AWS CLI is configured with "ab" profile
   - Run `aws configure list --profile ab` in terminal

2. **S3 Access Denied**
   - Check IAM permissions for your role
   - Ensure you have access to the data bucket

3. **MLFlow Connection Issues**
   - Verify SageMaker Studio has MLFlow enabled
   - Check network connectivity

4. **Notebook Kernel Issues**
   - Select "Python 3" kernel when opening notebooks
   - Restart kernel if needed

### Getting Help

1. Check notebook documentation and comments
2. Review error messages carefully
3. Contact your team lead or DevOps team
4. Refer to AWS SageMaker documentation

## Best Practices

### For Data Scientists

1. **Start Small**: Begin with small datasets for exploration
2. **Track Everything**: Use MLFlow to track all experiments
3. **Document Findings**: Add markdown cells with insights
4. **Collaborate**: Share interesting findings with the team

### For ML Engineers

1. **Monitor Resources**: Keep an eye on training costs
2. **Use Spot Instances**: Enable spot instances for cost savings
3. **Track Experiments**: Log all training parameters and metrics
4. **Model Governance**: Use Model Registry approval workflow

## Next Steps

1. **Complete Setup**: Ensure all prerequisites are met
2. **Run Notebooks**: Start with the core notebooks for your role
3. **Explore MLFlow**: Familiarize yourself with the MLFlow UI
4. **Collaborate**: Share experiments and findings with your team

---

Generated on: 2025-07-20 13:29:32
