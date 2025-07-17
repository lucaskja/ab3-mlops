# Source Code Modules

This directory contains the core implementation of the MLOps SageMaker Demo project.

## Directory Structure

```
src/
├── data/                # Data processing and validation modules
│   ├── data_profiler.py # Data profiling utilities
│   ├── data_validator.py # Data validation utilities
│   ├── ground_truth_utils.py # Ground Truth labeling utilities
│   ├── s3_utils.py      # S3 access utilities
│   └── yolo_preprocessor.py # YOLO data preprocessing
├── models/              # Model implementation modules
│   └── yolov11_trainer.py # YOLOv11 training implementation
├── pipeline/            # Pipeline orchestration modules
│   ├── clarify_integration.py # SageMaker Clarify integration
│   ├── error_recovery.py # Pipeline error handling
│   ├── mlflow_integration.py # MLFlow experiment tracking
│   ├── mlflow_visualization.py # MLFlow visualization utilities
│   ├── model_monitor.py # SageMaker Model Monitor
│   ├── model_monitor_integration.py # Model monitoring integration
│   ├── sagemaker_pipeline.py # SageMaker Pipeline implementation
│   └── sagemaker_training.py # SageMaker training job management
└── monitoring/          # Monitoring and observability modules (to be implemented)
    ├── alerting.py      # EventBridge alerting integration
    ├── dashboards.py    # CloudWatch dashboard generation
    └── drift_detection.py # Model drift detection utilities
```

## Module Descriptions

### Data Modules

- **data_profiler.py**: Utilities for profiling and analyzing drone imagery datasets
- **data_validator.py**: Data validation functions for ensuring data quality
- **ground_truth_utils.py**: Utilities for creating and managing Ground Truth labeling jobs
- **s3_utils.py**: S3 access patterns and utilities for data management
- **yolo_preprocessor.py**: Data preprocessing for YOLOv11 format

### Model Modules

- **yolov11_trainer.py**: YOLOv11 model training implementation with SageMaker integration

### Pipeline Modules

- **clarify_integration.py**: Integration with SageMaker Clarify for model explainability
- **error_recovery.py**: Error handling and recovery mechanisms for pipelines
- **mlflow_integration.py**: MLFlow experiment tracking integration
- **mlflow_visualization.py**: Visualization utilities for MLFlow experiments
- **model_monitor.py**: SageMaker Model Monitor configuration and management
- **model_monitor_integration.py**: Integration of Model Monitor with pipelines
- **sagemaker_pipeline.py**: SageMaker Pipeline implementation and orchestration
- **sagemaker_training.py**: SageMaker training job configuration and management

### Monitoring Modules (To Be Implemented)

- **alerting.py**: EventBridge alerting integration for model monitoring
- **dashboards.py**: CloudWatch dashboard generation for monitoring visualization
- **drift_detection.py**: Model drift detection utilities and thresholds

## Usage Patterns

### Data Access Pattern

```python
from src.data.s3_utils import S3DataAccess

# Initialize S3 data access with the project bucket
s3_access = S3DataAccess(bucket_name="lucaskle-ab3-project-pv", profile_name="ab")

# List available datasets
datasets = s3_access.list_datasets()

# Access a specific dataset
dataset = s3_access.get_dataset("drone-imagery")
```

### Ground Truth Labeling Pattern

```python
from src.data.ground_truth_utils import create_labeling_job_config, monitor_labeling_job

# Create a labeling job configuration
config = create_labeling_job_config(
    job_name="drone-detection-job",
    input_path="s3://lucaskle-ab3-project-pv/raw-images/manifest.json",
    output_path="s3://lucaskle-ab3-project-pv/labeled-data/",
    task_type="BoundingBox",
    worker_type="private",
    labels=["drone", "vehicle", "person", "building"]
)

# Monitor job progress
status = monitor_labeling_job("drone-detection-job")
```

### Pipeline Orchestration Pattern

```python
from src.pipeline.sagemaker_pipeline import create_training_pipeline

# Create a training pipeline
pipeline = create_training_pipeline(
    pipeline_name="yolov11-training-pipeline",
    role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    input_data_uri="s3://lucaskle-ab3-project-pv/processed-data/",
    output_path="s3://lucaskle-ab3-project-pv/model-artifacts/"
)

# Execute the pipeline
execution = pipeline.start()
```

## Error Handling

All modules implement comprehensive error handling with:

- Specific exception types for different error scenarios
- Detailed error messages with context information
- Automatic retry mechanisms for transient failures
- Logging of all errors with appropriate severity levels

## Logging

All modules use a standardized logging approach:

```python
import logging

# Get a logger for the current module
logger = logging.getLogger(__name__)

# Log messages with appropriate levels
logger.debug("Detailed debug information")
logger.info("General information about operation progress")
logger.warning("Warning about potential issues")
logger.error("Error information with exception details")
```

## Testing

Each module has corresponding unit tests in the `tests/` directory. Run tests with:

```bash
python -m unittest discover tests
```