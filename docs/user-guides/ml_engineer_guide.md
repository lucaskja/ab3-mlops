# ML Engineer Guide

This guide provides instructions for ML Engineers working with the MLOps SageMaker Demo project, focusing on pipeline development, model deployment, and monitoring.

## Role Overview

As an ML Engineer, you have:
- Full access to SageMaker Pipelines
- Access to Model Registry and deployment resources
- Permission to create and manage endpoints
- Access to monitoring and production resources

## Getting Started

### 1. Access SageMaker Studio

1. Log in to the AWS Management Console
2. Navigate to Amazon SageMaker
3. Select "Studio" from the left navigation
4. Open Studio with the "MLEngineer" user profile

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

## Pipeline Development Workflow

### 1. Create SageMaker Pipelines

1. Open `notebooks/pipeline-development/training-pipeline.ipynb`
2. Configure the pipeline parameters:
   - Set input and output data paths
   - Configure instance types
   - Set hyperparameters
3. Run the notebook to create and register the pipeline

### 2. Customize Pipeline Steps

Modify the pipeline steps in `src/pipeline/sagemaker_pipeline.py`:

```python
from src.pipeline.sagemaker_pipeline import create_training_pipeline

# Create a custom pipeline
pipeline = create_training_pipeline(
    pipeline_name="custom-yolov11-pipeline",
    role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    preprocessing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.g4dn.xlarge",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 100
    }
)

# Register the pipeline
pipeline.upsert(role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole")
```

### 3. Execute and Monitor Pipelines

1. Start pipeline executions:
   ```python
   execution = pipeline.start()
   ```

2. Monitor execution status:
   ```python
   from src.pipeline.sagemaker_pipeline import monitor_pipeline_execution
   
   status = monitor_pipeline_execution(execution.arn)
   print(f"Pipeline status: {status}")
   ```

## Model Deployment Workflow

### 1. Register Models in the Model Registry

1. Open `notebooks/pipeline-development/model-registry.ipynb`
2. Configure model registration parameters
3. Run the notebook to register models

### 2. Deploy Models to Endpoints

1. Open `notebooks/pipeline-development/model-deployment.ipynb`
2. Configure endpoint parameters:
   - Instance type
   - Auto-scaling settings
   - Monitoring configuration
3. Run the notebook to deploy the model

### 3. Test Deployed Endpoints

```python
import boto3
import json
import numpy as np
from PIL import Image
import io

# Load test image
image = Image.open("test_image.jpg")
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', profile_name='ab')

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName="yolov11-endpoint",
    ContentType="image/jpeg",
    Body=image_bytes
)

# Parse response
result = json.loads(response['Body'].read().decode())
print(result)
```

## Monitoring and Governance

### 1. Set Up Model Monitoring

1. Open `notebooks/pipeline-development/model-monitoring.ipynb`
2. Configure monitoring parameters:
   - Data quality baselines
   - Model quality thresholds
   - Monitoring schedule
3. Run the notebook to set up monitoring

### 2. Configure Alerts

```python
from src.monitoring.alerting import create_model_drift_alert

# Create a model drift alert
create_model_drift_alert(
    endpoint_name="yolov11-endpoint",
    metric_name="data_drift_score",
    threshold=0.7,
    evaluation_period=3600,
    notification_topic="arn:aws:sns:us-east-1:123456789012:model-alerts"
)
```

### 3. Create Dashboards

```python
from src.monitoring.dashboards import create_model_monitoring_dashboard

# Create a monitoring dashboard
create_model_monitoring_dashboard(
    endpoint_name="yolov11-endpoint",
    dashboard_name="yolov11-monitoring"
)
```

## CI/CD Integration

### 1. Set Up SageMaker Projects

1. Navigate to SageMaker Studio
2. Select "Projects" from the left navigation
3. Create a new project using the MLOps template
4. Configure the project settings:
   - Source repository
   - Build pipeline
   - Deployment pipeline

### 2. Configure CodePipeline

1. Define the build and test stages in `buildspec.yml`
2. Configure the deployment stages in `pipeline.yml`
3. Set up environment-specific configurations

## Infrastructure Management

### 1. Deploy CDK Stacks

```bash
cd configs/cdk
npm install
cdk deploy --profile ab
```

### 2. Validate Infrastructure

```bash
./scripts/setup/validate_cdk_deployment.sh
./scripts/setup/validate_cdk_security.sh
```

## Best Practices

### Pipeline Development

- Use modular pipeline components for reusability
- Implement comprehensive error handling
- Log all pipeline steps for observability
- Use conditional steps for dynamic behavior

### Model Deployment

- Implement blue/green deployments for zero downtime
- Use auto-scaling for cost optimization
- Set up comprehensive monitoring from day one
- Implement circuit breakers for endpoint protection

### Monitoring

- Define clear thresholds for alerts
- Implement automated remediation where possible
- Create comprehensive dashboards for visibility
- Set up regular monitoring reviews

## Troubleshooting

### Common Issues

#### Pipeline Execution Failures

If pipeline executions fail:

1. Check the execution logs in SageMaker Studio
2. Verify IAM permissions for all pipeline steps
3. Check resource availability in the AWS region

#### Endpoint Deployment Issues

If endpoint deployments fail:

1. Verify the model artifact is valid
2. Check instance availability in the region
3. Verify IAM permissions for endpoint creation

#### Monitoring Configuration Issues

If monitoring setup fails:

1. Verify baseline data is available
2. Check IAM permissions for monitoring jobs
3. Verify CloudWatch permissions for metrics

## Getting Help

If you encounter issues not covered in this guide:

1. Check the project documentation in the `docs/` directory
2. Review AWS SageMaker documentation
3. Contact the DevOps team for infrastructure assistance