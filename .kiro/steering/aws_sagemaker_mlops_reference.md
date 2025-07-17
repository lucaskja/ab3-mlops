# AWS SageMaker MLOps Reference Implementation

This steering document incorporates guidance from the official AWS reference implementation for MLOps with SageMaker: [Amazon SageMaker MLOps: from idea to production in six steps](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production).

## Six-Step Process

Our MLOps SageMaker Demo follows the six-step process outlined in the AWS reference implementation:

1. **Experiment in a notebook** - Initial development and experimentation with YOLOv11 for drone imagery
2. **Scale with SageMaker processing jobs and Python SDK** - Moving computation to SageMaker for data preprocessing
3. **Operationalize with ML pipeline, model registry, and feature store** - Building automation with SageMaker Pipelines
4. **Add a model building CI/CD pipeline** - Automating the model building process
5. **Add a model deployment pipeline** - Automating model deployment with conditional steps
6. **Add model and data monitoring** - Ensuring ongoing quality with Model Monitor and EventBridge alerts

## Key Components

Our implementation incorporates the following key components from the reference implementation:

### 1. SageMaker Processing Jobs
- Using SageMaker Processing for data preprocessing of drone imagery
- Containerized preprocessing scripts with YOLOv11 dependencies
- Data validation and quality checks for image data

### 2. SageMaker Pipelines
- End-to-end ML workflows with preprocessing, training, evaluation, and deployment steps
- S3 artifact storage with proper versioning
- Conditional steps based on model performance metrics (mAP, precision, recall)

### 3. MLFlow Integration
- SageMaker-managed MLflow for experiment tracking
- Logging of hyperparameters, metrics, and artifacts
- Model version comparison and hyperparameter tuning

### 4. Model Registry
- Model registration with proper versioning
- Approval workflows for production deployment
- Model lineage tracking

### 5. Model Monitoring
- Data quality monitoring for input drift detection
- Model quality monitoring for performance degradation
- Drift detection and EventBridge alerting

### 6. Error Recovery
- Sophisticated error recovery mechanisms for training jobs
- Automatic retry with exponential backoff
- Checkpoint-based resumption of failed jobs

## Implementation Approach

Our implementation follows these principles from the reference:

1. **Modularity**: Reusable components for each stage of the ML lifecycle
2. **Automation**: Minimized manual steps through pipelines
3. **Governance**: Role-based access control and audit logging
4. **Monitoring**: Comprehensive monitoring of models and data
5. **Testing**: Tests at all levels (unit, integration, system)

## Key Differences from Reference Implementation

While we follow the same general approach as the reference implementation, our MLOps SageMaker Demo has some key differences:

1. **Focus on Computer Vision**: We're using YOLOv11 for object detection on drone imagery, while the reference uses a tabular dataset
2. **Enhanced Error Recovery**: We've implemented sophisticated error recovery mechanisms for training jobs
3. **Environment-Specific Configuration**: We've added robust configuration management for different environments
4. **Comprehensive Testing**: We've added extensive unit and integration tests for all components
5. **Role Separation**: We've implemented clear separation between Data Scientist and ML Engineer roles

## Reference Links

- [GitHub Repository](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production)
- [Workshop Guide](https://catalog.workshops.aws/mlops-from-idea-to-production)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
- [MLflow Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html)