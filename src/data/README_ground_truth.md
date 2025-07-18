# Ground Truth Utilities

This module provides comprehensive utilities for creating, monitoring, and processing SageMaker Ground Truth labeling jobs for drone imagery object detection.

## Overview

The `ground_truth_utils.py` module enables Data Scientists to easily create and manage labeling jobs directly from Jupyter notebooks in SageMaker Studio. It provides a complete set of functions for the entire labeling workflow, from job creation to result processing.

## Features

- **Job Configuration**: Create and configure labeling jobs with customizable parameters
- **Job Monitoring**: Track job progress and status with detailed metrics
- **Format Conversion**: Convert Ground Truth output to YOLOv11 format for object detection
- **Annotation Quality**: Validate annotation quality with customizable criteria
- **Cost Estimation**: Estimate and control labeling costs with budget limits
- **Visualization**: Visualize annotations and labeling results
- **Instruction Creation**: Generate custom labeling instructions with templates

## Usage Examples

### Creating a Labeling Job

```python
from src.data.ground_truth_utils import create_labeling_job_config

# Configure the labeling job
job_name = "drone-detection-001"
input_path = "s3://lucaskle-ab3-project-pv/manifests/input.json"
output_path = "s3://lucaskle-ab3-project-pv/output/"

# Create labeling job configuration
config = create_labeling_job_config(
    job_name=job_name,
    input_path=input_path,
    output_path=output_path,
    task_type="BoundingBox",
    worker_type="private",
    labels=["drone", "vehicle", "person", "building"],
    instructions="Label all drones and other objects visible in the image.",
    max_budget_usd=100.0
)

# Submit the job using the SageMaker client
import boto3
session = boto3.Session(profile_name='ab')
sagemaker_client = session.client('sagemaker')
response = sagemaker_client.create_labeling_job(**config)
```

### Monitoring Job Progress

```python
from src.data.ground_truth_utils import monitor_labeling_job, get_labeling_job_metrics

# Get basic job status
job_name = "drone-detection-001"
status = monitor_labeling_job(job_name)
print(f"Job Status: {status['LabelingJobStatus']}")
print(f"Progress: {status['CompletionPercentage']:.2f}%")

# Get detailed metrics
metrics = get_labeling_job_metrics(job_name)
print(f"Labeling Speed: {metrics['basic_status']['LabelingSpeed']:.2f} objects/hour")
print(f"Estimated Completion Time: {metrics['basic_status']['EstimatedCompletionTime']}")
```

### Converting to YOLOv11 Format

```python
from src.data.ground_truth_utils import convert_ground_truth_to_yolo

# Convert Ground Truth output to YOLOv11 format
input_manifest = "s3://lucaskle-ab3-project-pv/output/drone-detection-001/manifests/output/output.manifest"
output_directory = "s3://lucaskle-ab3-project-pv/training-data/yolo-format/"

# Define class mapping
class_mapping = {
    "drone": 0,
    "vehicle": 1,
    "person": 2,
    "building": 3
}

# Perform conversion
convert_ground_truth_to_yolo(
    input_manifest=input_manifest,
    output_directory=output_directory,
    class_mapping=class_mapping
)
```

### Validating Annotation Quality

```python
from src.data.ground_truth_utils import validate_annotation_quality

# Define validation criteria
validation_criteria = {
    "min_boxes_per_image": 1,
    "max_boxes_per_image": 50,
    "min_box_size": 0.01,  # 1% of image area
    "max_box_overlap": 0.8  # 80% IoU threshold
}

# Validate annotations
manifest_file = "s3://lucaskle-ab3-project-pv/output/drone-detection-001/manifests/output/output.manifest"
results = validate_annotation_quality(
    manifest_file=manifest_file,
    validation_criteria=validation_criteria
)

# Print results
print(f"Total Images: {results['total_images']}")
print(f"Total Annotations: {results['total_annotations']}")
print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
print(f"Images with Issues: {results['images_with_issues']}")
```

### Estimating Labeling Costs

```python
from src.data.ground_truth_utils import estimate_labeling_cost

# Estimate cost for a labeling job
cost_estimate = estimate_labeling_cost(
    num_images=1000,
    task_type="BoundingBox",
    worker_type="public",
    objects_per_image=3.0
)

# Print cost breakdown
print(f"Base Cost: ${cost_estimate['base_cost']:.2f}")
print(f"Adjusted Cost: ${cost_estimate['adjusted_cost']:.2f}")
print(f"Service Cost: ${cost_estimate['service_cost']:.2f}")
print(f"Storage Cost: ${cost_estimate['storage_cost']:.2f}")
print(f"Total Cost: ${cost_estimate['total_cost']:.2f}")
print(f"Cost per Image: ${cost_estimate['cost_per_image']:.4f}")

# Print cost control recommendations
for recommendation in cost_estimate['cost_control_recommendations']:
    print(f"- {recommendation}")
```

### Creating Labeling Instructions

```python
from src.data.ground_truth_utils import create_labeling_instructions

# Create instructions for a bounding box task
instructions = create_labeling_instructions(
    task_type="BoundingBox",
    categories=["drone", "vehicle", "person", "building"],
    example_images=[
        "s3://lucaskle-ab3-project-pv/examples/example1.jpg",
        "s3://lucaskle-ab3-project-pv/examples/example2.jpg"
    ],
    detailed_instructions=True
)

# Display instructions in a notebook
from IPython.display import HTML
display(HTML(instructions))
```

### Visualizing Annotations

```python
from src.data.ground_truth_utils import visualize_annotations
import matplotlib.pyplot as plt
from PIL import Image

# Visualize annotations
manifest_file = "s3://lucaskle-ab3-project-pv/output/drone-detection-001/manifests/output/output.manifest"
visualization_paths = visualize_annotations(
    manifest_file=manifest_file,
    output_directory="/tmp/visualizations",
    max_images=5
)

# Display visualizations
for path in visualization_paths:
    img = Image.open(path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Annotated: {path.split('/')[-1]}")
    plt.show()
```

## Function Reference

### Job Configuration and Submission

- `create_labeling_job_config`: Create a configuration for a SageMaker Ground Truth labeling job
- `create_manifest_file`: Create a manifest file for a Ground Truth labeling job

### Job Monitoring and Status Tracking

- `monitor_labeling_job`: Monitor the status of a SageMaker Ground Truth labeling job
- `get_labeling_job_metrics`: Get detailed metrics for a SageMaker Ground Truth labeling job
- `list_labeling_jobs`: List SageMaker Ground Truth labeling jobs

### Format Conversion

- `convert_ground_truth_to_yolo`: Convert SageMaker Ground Truth output to YOLOv11 format

### Annotation Quality Validation

- `validate_annotation_quality`: Validate the quality of annotations in a Ground Truth output manifest
- `calculate_iou`: Calculate Intersection over Union (IoU) between two bounding boxes

### Cost Estimation and Budget Control

- `estimate_labeling_cost`: Estimate the cost of a labeling job
- `calculate_max_objects`: Calculate the maximum number of objects that can be labeled within the budget

### Visualization and Instructions

- `visualize_annotations`: Visualize annotations from a Ground Truth output manifest
- `create_labeling_instructions`: Create HTML instructions for a labeling job

### Utility Functions

- `parse_s3_uri`: Parse an S3 URI into bucket and key
- `get_pre_human_task_lambda_arn`: Get the appropriate pre-human task Lambda ARN for the task type and region
- `get_annotation_consolidation_lambda_arn`: Get the appropriate annotation consolidation Lambda ARN for the task type and region
- `get_ui_template_s3_uri`: Get the appropriate UI template S3 URI for the task type and region
- `get_default_workteam_arn`: Get the default workteam ARN for the region

## Integration with SageMaker Studio

This module is designed to be used within SageMaker Studio notebooks. For a complete example of integration, see the notebook at `notebooks/data-labeling/create_labeling_job_enhanced.ipynb`.