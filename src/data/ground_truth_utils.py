"""
Ground Truth utilities for creating, monitoring, and processing labeling jobs.

This module provides utilities for:
- Creating and configuring SageMaker Ground Truth labeling jobs
- Monitoring job progress and status
- Converting Ground Truth output to YOLOv11 format
- Validating annotation quality
- Estimating and controlling labeling costs
"""

import boto3
import json
import time
import os
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

def create_labeling_job_config(
    job_name: str,
    input_path: str,
    output_path: str,
    task_type: str = "BoundingBox",
    worker_type: str = "private",
    labels: List[str] = None,
    instructions: str = "",
    max_budget_usd: float = 100.0,
    role_arn: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a SageMaker Ground Truth labeling job.
    
    Args:
        job_name: Unique name for the labeling job
        input_path: S3 URI to the input manifest file
        output_path: S3 URI where output will be stored
        task_type: Type of labeling task (BoundingBox, ImageClassification, etc.)
        worker_type: Type of workforce (private, public)
        labels: List of label categories
        instructions: Instructions for workers
        max_budget_usd: Maximum budget in USD
        role_arn: IAM role ARN for SageMaker to assume
        
    Returns:
        Dictionary containing the labeling job configuration
    """
    if labels is None:
        labels = ["drone", "vehicle", "person", "building"]
    
    # Get the execution role if not provided
    if role_arn is None:
        # Use the SageMaker execution role
        import sagemaker
        role_arn = sagemaker.get_execution_role()
    
    # Create label category configuration
    label_category_config = {
        "LabelCategoryConfigS3Uri": None,  # We'll use inline configuration
        "CatalogConfig": None
    }
    
    # Create the labeling job configuration
    labeling_job_config = {
        "LabelingJobName": job_name,
        "LabelAttributeName": f"{task_type.lower()}-labels",
        "InputConfig": {
            "DataSource": {
                "S3DataSource": {
                    "ManifestS3Uri": input_path
                }
            },
            "DataAttributes": {
                "ContentClassifiers": [
                    "FreeOfPersonallyIdentifiableInformation",
                    "FreeOfAdultContent"
                ]
            }
        },
        "OutputConfig": {
            "S3OutputPath": output_path,
            "KmsKeyId": None
        },
        "RoleArn": role_arn,
        "LabelCategoryConfigS3Uri": None,
        "StoppingConditions": {
            "MaxHumanLabeledObjectCount": 10000,  # Adjust as needed
            "MaxPercentageOfInputDatasetLabeled": 100
        },
        "LabelingJobAlgorithmsConfig": {
            "LabelingJobAlgorithm": "BOUNDING_BOX" if task_type == "BoundingBox" else task_type.upper()
        },
        "HumanTaskConfig": {
            "WorkteamArn": None,  # Will be set based on worker_type
            "UiConfig": {
                "UiTemplateS3Uri": None  # Will be set based on task_type
            },
            "PreHumanTaskLambdaArn": None,  # Will be set based on task_type and region
            "TaskKeywords": [
                "image labeling",
                "object detection",
                "bounding box"
            ],
            "TaskTitle": f"Drone Imagery {task_type} Labeling",
            "TaskDescription": instructions or f"Draw bounding boxes around objects in drone imagery",
            "NumberOfHumanWorkersPerDataObject": 1,
            "TaskTimeLimitInSeconds": 3600,
            "TaskAvailabilityLifetimeInSeconds": 864000,
            "MaxConcurrentTaskCount": 1000,
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": None  # Will be set based on task_type and region
            },
            "PublicWorkforceTaskPrice": {
                "AmountInUsd": {
                    "Dollars": int(max_budget_usd),
                    "Cents": int((max_budget_usd % 1) * 100),
                    "TenthFractionsOfACent": 0
                }
            } if worker_type == "public" else None
        },
        "Tags": [
            {
                "Key": "Project",
                "Value": "mlops-sagemaker-demo"
            },
            {
                "Key": "MaxBudgetUSD",
                "Value": str(max_budget_usd)
            }
        ]
    }
    
    # This is a simplified version - in a real implementation, you would:
    # 1. Set the correct WorkteamArn based on worker_type
    # 2. Set the correct UiTemplateS3Uri based on task_type
    # 3. Set the correct Lambda ARNs based on task_type and region
    # 4. Add label categories configuration
    
    logger.info(f"Created labeling job configuration for job: {job_name}")
    return labeling_job_config


def monitor_labeling_job(job_name: str, sagemaker_client=None) -> Dict[str, Any]:
    """
    Monitor the status of a SageMaker Ground Truth labeling job.
    
    Args:
        job_name: Name of the labeling job to monitor
        sagemaker_client: Boto3 SageMaker client (optional)
        
    Returns:
        Dictionary containing the labeling job status
    """
    if sagemaker_client is None:
        session = boto3.Session(profile_name='ab')
        sagemaker_client = session.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_labeling_job(
            LabelingJobName=job_name
        )
        
        # Extract relevant information
        status = {
            "LabelingJobName": response["LabelingJobName"],
            "LabelingJobStatus": response["LabelingJobStatus"],
            "LabelingJobArn": response["LabelingJobArn"],
            "CreationTime": response["CreationTime"],
            "LastModifiedTime": response["LastModifiedTime"],
            "LabelCounters": response["LabelCounters"],
            "FailureReason": response.get("FailureReason", None)
        }
        
        return status
    
    except Exception as e:
        logger.error(f"Error monitoring labeling job {job_name}: {str(e)}")
        raise


def convert_ground_truth_to_yolo(
    input_manifest: str,
    output_directory: str,
    class_mapping: Dict[str, int]
) -> bool:
    """
    Convert SageMaker Ground Truth output to YOLOv11 format.
    
    Args:
        input_manifest: S3 URI to the output manifest file from Ground Truth
        output_directory: S3 URI where YOLOv11 format data will be stored
        class_mapping: Dictionary mapping class names to class IDs
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Download the manifest file from S3
        # 2. Parse the JSON lines to extract annotations
        # 3. Convert bounding box coordinates to YOLO format
        # 4. Create label files in YOLO format
        # 5. Upload the label files to S3
        
        logger.info(f"Converting Ground Truth output from {input_manifest} to YOLOv11 format")
        logger.info(f"Output will be stored at {output_directory}")
        logger.info(f"Using class mapping: {class_mapping}")
        
        # Placeholder for conversion logic
        # In a real implementation, this would contain the actual conversion code
        
        return True
    
    except Exception as e:
        logger.error(f"Error converting Ground Truth output to YOLOv11 format: {str(e)}")
        return False


def validate_annotation_quality(
    manifest_file: str,
    validation_criteria: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate the quality of annotations from a Ground Truth labeling job.
    
    Args:
        manifest_file: S3 URI to the output manifest file
        validation_criteria: Dictionary of validation criteria
        
    Returns:
        Dictionary containing validation results
    """
    if validation_criteria is None:
        validation_criteria = {
            "min_boxes_per_image": 0,
            "max_boxes_per_image": 100,
            "min_box_size": 0.01,  # As percentage of image size
            "max_box_overlap": 0.8  # IOU threshold
        }
    
    try:
        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Download the manifest file from S3
        # 2. Parse the JSON lines to extract annotations
        # 3. Apply validation criteria to each annotation
        # 4. Generate validation statistics
        
        logger.info(f"Validating annotations in {manifest_file}")
        
        # Placeholder for validation logic
        validation_results = {
            "total_images": 0,
            "total_annotations": 0,
            "images_with_issues": 0,
            "annotations_with_issues": 0,
            "issues_by_type": {
                "too_few_boxes": 0,
                "too_many_boxes": 0,
                "box_too_small": 0,
                "excessive_overlap": 0
            },
            "pass_rate": 0.0
        }
        
        return validation_results
    
    except Exception as e:
        logger.error(f"Error validating annotations: {str(e)}")
        raise


def estimate_labeling_cost(
    num_images: int,
    task_type: str = "BoundingBox",
    worker_type: str = "private",
    objects_per_image: float = 3.0
) -> Dict[str, float]:
    """
    Estimate the cost of a Ground Truth labeling job.
    
    Args:
        num_images: Number of images to label
        task_type: Type of labeling task
        worker_type: Type of workforce
        objects_per_image: Average number of objects per image
        
    Returns:
        Dictionary containing cost estimates
    """
    # These are placeholder values - actual costs would depend on many factors
    cost_per_image = {
        "BoundingBox": {
            "private": 0.0,  # Private workforce has no direct cost
            "public": 0.036   # $0.036 per image for public workforce (example)
        },
        "ImageClassification": {
            "private": 0.0,
            "public": 0.012
        }
    }
    
    # Calculate base cost
    base_cost = num_images * cost_per_image.get(task_type, {}).get(worker_type, 0.0)
    
    # Adjust for complexity (more objects per image increases cost)
    complexity_factor = max(1.0, objects_per_image / 2.0)
    adjusted_cost = base_cost * complexity_factor
    
    # Add AWS service costs (simplified)
    service_cost = num_images * 0.001  # $0.001 per image (example)
    
    # Total cost
    total_cost = adjusted_cost + service_cost
    
    return {
        "base_cost": base_cost,
        "adjusted_cost": adjusted_cost,
        "service_cost": service_cost,
        "total_cost": total_cost
    }


def create_labeling_instructions(
    task_type: str,
    categories: List[str],
    example_images: List[str] = None
) -> str:
    """
    Create HTML instructions for labeling workers.
    
    Args:
        task_type: Type of labeling task
        categories: List of label categories
        example_images: List of S3 URIs to example images
        
    Returns:
        HTML string containing instructions
    """
    # Create basic instructions template
    instructions = f"""
    <h1>Drone Imagery {task_type} Instructions</h1>
    
    <p>Please label the following objects in the drone imagery:</p>
    
    <ul>
    """
    
    # Add categories
    for category in categories:
        instructions += f"<li><strong>{category}</strong></li>\n"
    
    instructions += "</ul>\n\n"
    
    # Add task-specific instructions
    if task_type == "BoundingBox":
        instructions += """
        <h2>Bounding Box Instructions</h2>
        
        <p>Draw a tight rectangle around each object:</p>
        <ol>
            <li>Click and drag to draw a box around the object</li>
            <li>Select the appropriate label from the dropdown</li>
            <li>Adjust the box if needed by dragging the corners</li>
            <li>Continue until all objects are labeled</li>
        </ol>
        """
    
    # Add examples if provided
    if example_images:
        instructions += "<h2>Examples</h2>\n"
        for i, image_uri in enumerate(example_images):
            instructions += f'<div><img src="{image_uri}" style="max-width: 500px; margin: 10px;"><p>Example {i+1}</p></div>\n'
    
    return instructions