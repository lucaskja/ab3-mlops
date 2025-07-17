# Data Scientist Guide

This guide provides instructions for Data Scientists working with the MLOps SageMaker Demo project, focusing on data exploration, labeling, and model development.

## Role Overview

As a Data Scientist, you have:
- Read-only access to raw data in S3
- Full access to SageMaker Studio notebooks
- Access to MLFlow for experiment tracking
- Permissions to create Ground Truth labeling jobs

## Getting Started

### 1. Access SageMaker Studio

1. Log in to the AWS Management Console
2. Navigate to Amazon SageMaker
3. Select "Studio" from the left navigation
4. Open Studio with the "DataScientist" user profile

### 2. Clone the Repository

In a Terminal within SageMaker Studio:

```bash
git clone https://github.com/yourusername/mlops-sagemaker-demo.git
cd mlops-sagemaker-demo
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Exploration Workflow

### 1. Access the Dataset

The drone imagery dataset is stored in the S3 bucket `lucaskle-ab3-project-pv`. You can explore it using the data exploration notebooks:

1. Open `notebooks/data-exploration/dataset-analysis.ipynb`
2. Run the notebook to analyze dataset characteristics
3. Use the visualization tools to understand the data distribution

### 2. Profile the Data

1. Open `notebooks/data-exploration/data-profiling.ipynb`
2. Run the notebook to generate data quality profiles
3. Identify potential issues in the dataset

## Data Labeling Workflow

### 1. Create a Ground Truth Labeling Job

1. Open `notebooks/data-labeling/create_labeling_job.ipynb`
2. Configure the labeling job parameters:
   - Select images for labeling
   - Define object categories
   - Configure worker settings
   - Set budget limits
3. Run the notebook to create and start the labeling job

### 2. Monitor Labeling Progress

1. Use the monitoring cells in the notebook to check job status
2. View completion metrics and progress
3. Analyze labeling quality metrics

### 3. Process Labeled Data

1. Once labeling is complete, run the conversion cells
2. Convert Ground Truth output to YOLOv11 format
3. Visualize the labeled data to verify quality

## Model Development Workflow

### 1. Train YOLOv11 Models

1. Open `notebooks/model-development/yolov11-training.ipynb`
2. Configure training parameters:
   - Set hyperparameters using interactive widgets
   - Select training and validation datasets
   - Configure model architecture
3. Run the training cells to train the model
4. Monitor training progress with real-time metrics

### 2. Tune Hyperparameters

1. Open `notebooks/model-development/hyperparameter-tuning.ipynb`
2. Configure the hyperparameter ranges
3. Run the notebook to perform hyperparameter optimization
4. Analyze results to find optimal configurations

### 3. Evaluate Models

1. Open `notebooks/model-development/model-evaluation.ipynb`
2. Select models to evaluate
3. Run evaluation on test datasets
4. Compare performance metrics across models
5. Visualize detection results

## Experiment Tracking with MLFlow

### 1. View Experiments

1. Open the MLFlow UI from SageMaker Studio
2. Browse experiments and runs
3. Compare metrics across different runs
4. Analyze parameter importance

### 2. Log New Experiments

When training models, experiments are automatically logged to MLFlow:

```python
from src.pipeline.mlflow_integration import MLFlowSageMakerIntegration

# Initialize MLFlow integration
mlflow_integration = MLFlowSageMakerIntegration(experiment_name="yolov11-drone-detection")

# Start a run
with mlflow_integration.start_run():
    # Log parameters
    mlflow_integration.log_parameters({
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 100
    })
    
    # Train model
    # ...
    
    # Log metrics
    mlflow_integration.log_metrics({
        "mAP_0.5": 0.85,
        "precision": 0.88,
        "recall": 0.82
    })
    
    # Log model
    mlflow_integration.log_model("model", "yolov11_model")
```

## Best Practices

### Data Management

- Always use the provided utility functions for data access
- Document any data quality issues you discover
- Version your datasets appropriately

### Experimentation

- Keep experiments organized with clear naming conventions
- Log all parameters and metrics to MLFlow
- Document your findings in notebook markdown cells

### Collaboration

- Share insights with ML Engineers through MLFlow experiments
- Document model performance characteristics
- Provide clear requirements for model deployment

## Troubleshooting

### Common Issues

#### Access Denied Errors

If you encounter "Access Denied" errors when accessing S3:

1. Verify you're using the correct AWS profile:
   ```python
   import boto3
   session = boto3.Session(profile_name='ab')
   ```

2. Check that you're accessing the correct bucket:
   ```python
   bucket_name = "lucaskle-ab3-project-pv"
   ```

#### SageMaker Studio Kernel Issues

If Jupyter kernels are not starting:

1. Try restarting the kernel
2. If that fails, restart the SageMaker Studio application
3. Ensure all dependencies are installed in the environment

#### Ground Truth Job Failures

If Ground Truth labeling jobs fail:

1. Check the job status in the AWS console
2. Verify the input manifest format
3. Ensure the IAM role has appropriate permissions

## Getting Help

If you encounter issues not covered in this guide:

1. Check the project documentation in the `docs/` directory
2. Review AWS SageMaker documentation
3. Contact the ML Engineering team for assistance