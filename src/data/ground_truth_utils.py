"""
Ground Truth utilities for creating, monitoring, and processing labeling jobs.

This module provides comprehensive utilities for:
- Creating and configuring SageMaker Ground Truth labeling jobs
- Monitoring job progress and status with detailed metrics
- Converting Ground Truth output to YOLOv11 format for object detection
- Validating annotation quality with customizable criteria
- Estimating and controlling labeling costs with budget limits
- Tracking job metrics and completion statistics
- Managing cost control mechanisms for labeling jobs
- Visualizing annotations and labeling results
- Creating custom labeling instructions with templates
- Managing labeling workforces and job submissions

This module is designed to be used by Data Scientists in SageMaker Studio
to easily create and manage labeling jobs for drone imagery object detection.
"""

import boto3
import json
import time
import os
import logging
import re
import io
import csv
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont

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
    role_arn: Optional[str] = None,
    region_name: str = "us-east-1",
    workteam_arn: Optional[str] = None,
    ui_template_s3_uri: Optional[str] = None
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
        region_name: AWS region name
        workteam_arn: ARN of the workteam (required for private workforce)
        ui_template_s3_uri: S3 URI to custom UI template (optional)
        
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
    label_category_config_content = {
        "document-version": "2018-11-28",
        "categoryGlobalAttributes": [],
        "labels": [{"label": label} for label in labels]
    }
    
    # Create a temporary file for the label category configuration
    label_config_file = f"/tmp/label_config_{job_name}.json"
    with open(label_config_file, 'w') as f:
        json.dump(label_category_config_content, f)
    
    # Upload the label category configuration to S3
    session = boto3.Session(profile_name='ab', region_name=region_name)
    s3_client = session.client('s3')
    
    # Parse the output path to get bucket and prefix
    output_bucket = output_path.replace('s3://', '').split('/')[0]
    output_key_prefix = '/'.join(output_path.replace('s3://', '').split('/')[1:])
    label_config_s3_key = f"{output_key_prefix}/label-config/{job_name}.json"
    
    try:
        s3_client.upload_file(label_config_file, output_bucket, label_config_s3_key)
        label_category_config_s3_uri = f"s3://{output_bucket}/{label_config_s3_key}"
    except Exception as e:
        logger.error(f"Error uploading label category configuration: {str(e)}")
        label_category_config_s3_uri = None
    
    # Get the appropriate Lambda ARNs for the task type
    pre_human_task_lambda_arn = get_pre_human_task_lambda_arn(task_type, region_name)
    annotation_consolidation_lambda_arn = get_annotation_consolidation_lambda_arn(task_type, region_name)
    
    # Get the appropriate UI template S3 URI if not provided
    if ui_template_s3_uri is None:
        ui_template_s3_uri = get_ui_template_s3_uri(task_type, region_name)
    
    # Get the appropriate workteam ARN if not provided
    if workteam_arn is None and worker_type == "private":
        workteam_arn = get_default_workteam_arn(region_name)
    
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
        "LabelCategoryConfigS3Uri": label_category_config_s3_uri,
        "StoppingConditions": {
            "MaxHumanLabeledObjectCount": 10000,  # Adjust as needed
            "MaxPercentageOfInputDatasetLabeled": 100
        },
        "HumanTaskConfig": {
            "WorkteamArn": workteam_arn if worker_type == "private" else None,
            "UiConfig": {
                "UiTemplateS3Uri": ui_template_s3_uri
            },
            "PreHumanTaskLambdaArn": pre_human_task_lambda_arn,
            "TaskKeywords": [
                "image labeling",
                "object detection",
                "drone imagery"
            ],
            "TaskTitle": f"Drone Imagery {task_type} Labeling",
            "TaskDescription": instructions or f"Draw bounding boxes around objects in drone imagery",
            "NumberOfHumanWorkersPerDataObject": 1,
            "TaskTimeLimitInSeconds": 3600,
            "TaskAvailabilityLifetimeInSeconds": 864000,
            "MaxConcurrentTaskCount": 1000,
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": annotation_consolidation_lambda_arn
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
            },
            {
                "Key": "CreatedBy",
                "Value": "data-scientist"
            },
            {
                "Key": "TaskType",
                "Value": task_type
            }
        ]
    }
    
    # Add cost control mechanisms
    if max_budget_usd > 0:
        # Add budget constraints to the configuration
        labeling_job_config["StoppingConditions"]["MaxHumanLabeledObjectCount"] = calculate_max_objects(
            max_budget_usd, task_type, worker_type
        )
    
    logger.info(f"Created labeling job configuration for job: {job_name}")
    return labeling_job_config


def get_pre_human_task_lambda_arn(task_type: str, region_name: str) -> str:
    """
    Get the appropriate pre-human task Lambda ARN for the task type and region.
    
    Args:
        task_type: Type of labeling task
        region_name: AWS region name
        
    Returns:
        ARN of the pre-human task Lambda function
    """
    # These are the standard ARNs for Ground Truth built-in task types
    # Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-task-types.html
    lambda_arns = {
        "BoundingBox": {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-BoundingBox",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-BoundingBox",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-BoundingBox",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-BoundingBox",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-BoundingBox"
        },
        "ImageClassification": {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-ImageMultiClass",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-ImageMultiClass",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-ImageMultiClass",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-ImageMultiClass",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-ImageMultiClass"
        }
    }
    
    # Return the appropriate ARN or raise an error if not found
    if task_type in lambda_arns and region_name in lambda_arns[task_type]:
        return lambda_arns[task_type][region_name]
    else:
        raise ValueError(f"No pre-human task Lambda ARN found for task type {task_type} in region {region_name}")


def get_annotation_consolidation_lambda_arn(task_type: str, region_name: str) -> str:
    """
    Get the appropriate annotation consolidation Lambda ARN for the task type and region.
    
    Args:
        task_type: Type of labeling task
        region_name: AWS region name
        
    Returns:
        ARN of the annotation consolidation Lambda function
    """
    # These are the standard ARNs for Ground Truth built-in task types
    # Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-task-types.html
    lambda_arns = {
        "BoundingBox": {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-BoundingBox",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-BoundingBox",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-BoundingBox",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-BoundingBox",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-BoundingBox"
        },
        "ImageClassification": {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-ImageMultiClass",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-ImageMultiClass",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-ImageMultiClass",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-ImageMultiClass",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-ImageMultiClass"
        }
    }
    
    # Return the appropriate ARN or raise an error if not found
    if task_type in lambda_arns and region_name in lambda_arns[task_type]:
        return lambda_arns[task_type][region_name]
    else:
        raise ValueError(f"No annotation consolidation Lambda ARN found for task type {task_type} in region {region_name}")


def get_ui_template_s3_uri(task_type: str, region_name: str) -> str:
    """
    Get the appropriate UI template S3 URI for the task type and region.
    
    Args:
        task_type: Type of labeling task
        region_name: AWS region name
        
    Returns:
        S3 URI of the UI template
    """
    # These are the standard S3 URIs for Ground Truth built-in task types
    # Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-task-types.html
    ui_templates = {
        "BoundingBox": f"s3://sagemaker-task-uis-{region_name}/bounding-box/index.html",
        "ImageClassification": f"s3://sagemaker-task-uis-{region_name}/image-classification/index.html"
    }
    
    # Return the appropriate URI or raise an error if not found
    if task_type in ui_templates:
        return ui_templates[task_type]
    else:
        raise ValueError(f"No UI template S3 URI found for task type {task_type}")


def get_default_workteam_arn(region_name: str) -> str:
    """
    Get the default workteam ARN for the region.
    
    Args:
        region_name: AWS region name
        
    Returns:
        ARN of the default workteam
    """
    # In a real implementation, you would query the SageMaker API to get the workteam ARN
    # For now, we'll return a placeholder ARN
    return f"arn:aws:sagemaker:{region_name}:123456789012:workteam/private-crowd/default-workteam"


def calculate_max_objects(max_budget_usd: float, task_type: str, worker_type: str) -> int:
    """
    Calculate the maximum number of objects that can be labeled within the budget.
    
    Args:
        max_budget_usd: Maximum budget in USD
        task_type: Type of labeling task
        worker_type: Type of workforce
        
    Returns:
        Maximum number of objects that can be labeled
    """
    # These are placeholder values - actual costs would depend on many factors
    cost_per_object = {
        "BoundingBox": {
            "private": 0.0,  # Private workforce has no direct cost
            "public": 0.036   # $0.036 per image for public workforce (example)
        },
        "ImageClassification": {
            "private": 0.0,
            "public": 0.012
        }
    }
    
    # Get the cost per object
    cost = cost_per_object.get(task_type, {}).get(worker_type, 0.0)
    
    # Calculate the maximum number of objects
    if cost > 0:
        # Add a 10% buffer for AWS service costs
        max_objects = int(max_budget_usd / (cost * 1.1))
    else:
        # For private workforce, use a high number
        max_objects = 10000
    
    return max_objects


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
            "FailureReason": response.get("FailureReason", None),
            "HumanTaskConfig": response.get("HumanTaskConfig", {}),
            "InputConfig": response.get("InputConfig", {}),
            "OutputConfig": response.get("OutputConfig", {})
        }
        
        # Calculate completion percentage
        if status["LabelCounters"]["TotalObjects"] > 0:
            completion_percentage = (status["LabelCounters"]["LabeledObjects"] / 
                                    status["LabelCounters"]["TotalObjects"]) * 100
            status["CompletionPercentage"] = completion_percentage
        else:
            status["CompletionPercentage"] = 0.0
        
        # Calculate labeling speed (objects per hour)
        if status["LabelingJobStatus"] != "Initializing":
            time_elapsed = (status["LastModifiedTime"] - status["CreationTime"]).total_seconds() / 3600  # hours
            if time_elapsed > 0:
                labeling_speed = status["LabelCounters"]["LabeledObjects"] / time_elapsed
                status["LabelingSpeed"] = labeling_speed
            else:
                status["LabelingSpeed"] = 0.0
        else:
            status["LabelingSpeed"] = 0.0
        
        # Calculate estimated completion time
        if status["LabelingSpeed"] > 0 and status["CompletionPercentage"] < 100:
            remaining_objects = status["LabelCounters"]["TotalObjects"] - status["LabelCounters"]["LabeledObjects"]
            estimated_hours = remaining_objects / status["LabelingSpeed"]
            estimated_completion_time = datetime.now() + timedelta(hours=estimated_hours)
            status["EstimatedCompletionTime"] = estimated_completion_time
        else:
            status["EstimatedCompletionTime"] = None
        
        # Get cost information if available
        try:
            # Extract cost information from tags
            tags_response = sagemaker_client.list_tags(
                ResourceArn=status["LabelingJobArn"]
            )
            
            tags = {tag["Key"]: tag["Value"] for tag in tags_response["Tags"]}
            
            if "MaxBudgetUSD" in tags:
                status["MaxBudgetUSD"] = float(tags["MaxBudgetUSD"])
                
                # Calculate estimated cost based on progress
                if status["CompletionPercentage"] > 0:
                    estimated_total_cost = (status["MaxBudgetUSD"] * status["LabelCounters"]["LabeledObjects"] / 
                                          status["LabelCounters"]["TotalObjects"])
                    status["EstimatedTotalCost"] = estimated_total_cost
                else:
                    status["EstimatedTotalCost"] = 0.0
        except Exception as e:
            logger.warning(f"Could not retrieve cost information: {str(e)}")
        
        return status
    
    except Exception as e:
        logger.error(f"Error monitoring labeling job {job_name}: {str(e)}")
        raise


def get_labeling_job_metrics(job_name: str, sagemaker_client=None) -> Dict[str, Any]:
    """
    Get detailed metrics for a SageMaker Ground Truth labeling job.
    
    Args:
        job_name: Name of the labeling job
        sagemaker_client: Boto3 SageMaker client (optional)
        
    Returns:
        Dictionary containing detailed metrics
    """
    # Get basic job status
    status = monitor_labeling_job(job_name, sagemaker_client)
    
    # Initialize metrics dictionary
    metrics = {
        "basic_status": status,
        "worker_metrics": {},
        "annotation_metrics": {},
        "cost_metrics": {},
        "time_metrics": {}
    }
    
    # Get the output manifest location
    output_path = status["OutputConfig"]["S3OutputPath"]
    output_manifest = f"{output_path}{job_name}/manifests/output/output.manifest"
    
    try:
        # Parse the output manifest to get annotation metrics
        session = boto3.Session(profile_name='ab')
        s3_client = session.client('s3')
        
        # Parse the S3 URI
        bucket, key = parse_s3_uri(output_manifest)
        
        try:
            # Download the manifest file
            response = s3_client.get_object(Bucket=bucket, Key=key)
            manifest_content = response['Body'].read().decode('utf-8')
            
            # Parse the manifest (each line is a JSON object)
            annotations = []
            for line in manifest_content.strip().split('\n'):
                if line:
                    annotations.append(json.loads(line))
            
            # Calculate annotation metrics
            metrics["annotation_metrics"] = calculate_annotation_metrics(annotations)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Output manifest not found: {output_manifest}")
            else:
                logger.error(f"Error accessing output manifest: {str(e)}")
        
        # Calculate time metrics
        metrics["time_metrics"] = {
            "total_time_hours": (status["LastModifiedTime"] - status["CreationTime"]).total_seconds() / 3600,
            "average_time_per_object": (status["LastModifiedTime"] - status["CreationTime"]).total_seconds() / 
                                      max(status["LabelCounters"]["LabeledObjects"], 1) if status["LabelCounters"]["LabeledObjects"] > 0 else 0,
            "estimated_completion_time": status.get("EstimatedCompletionTime", None)
        }
        
        # Calculate cost metrics
        if "MaxBudgetUSD" in status:
            metrics["cost_metrics"] = {
                "max_budget_usd": status["MaxBudgetUSD"],
                "estimated_total_cost": status.get("EstimatedTotalCost", 0.0),
                "cost_per_object": status.get("EstimatedTotalCost", 0.0) / 
                                  max(status["LabelCounters"]["LabeledObjects"], 1) if status["LabelCounters"]["LabeledObjects"] > 0 else 0
            }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting labeling job metrics: {str(e)}")
        # Return basic status if detailed metrics cannot be calculated
        return {"basic_status": status}


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI into bucket and key.
    
    Args:
        s3_uri: S3 URI (e.g., s3://bucket/key)
        
    Returns:
        Tuple of (bucket, key)
    """
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    path_without_scheme = s3_uri[5:]
    parts = path_without_scheme.split('/', 1)
    
    if len(parts) == 1:
        return parts[0], ''
    else:
        return parts[0], parts[1]


def calculate_annotation_metrics(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics from annotation data.
    
    Args:
        annotations: List of annotation objects from manifest
        
    Returns:
        Dictionary containing annotation metrics
    """
    # Initialize metrics
    metrics = {
        "total_annotations": len(annotations),
        "annotations_per_image": {},
        "label_distribution": {},
        "box_size_distribution": {},
        "worker_performance": {}
    }
    
    # Process each annotation
    for annotation in annotations:
        # This is a simplified implementation
        # In a real implementation, you would extract and analyze the actual annotation data
        
        # Example: Count annotations per image
        image_id = annotation.get("source-ref", "unknown")
        if image_id not in metrics["annotations_per_image"]:
            metrics["annotations_per_image"][image_id] = 0
        metrics["annotations_per_image"][image_id] += 1
        
        # Example: Count label distribution
        # Assuming the annotation has a "labels" field with a list of labels
        labels = annotation.get("labels", [])
        for label in labels:
            label_name = label.get("label", "unknown")
            if label_name not in metrics["label_distribution"]:
                metrics["label_distribution"][label_name] = 0
            metrics["label_distribution"][label_name] += 1
    
    # Calculate summary statistics
    if metrics["annotations_per_image"]:
        metrics["avg_annotations_per_image"] = sum(metrics["annotations_per_image"].values()) / len(metrics["annotations_per_image"])
        metrics["max_annotations_per_image"] = max(metrics["annotations_per_image"].values())
        metrics["min_annotations_per_image"] = min(metrics["annotations_per_image"].values())
    
    return metrics


def convert_ground_truth_to_yolo(
    input_manifest: str,
    output_directory: str,
    class_mapping: Dict[str, int],
    region_name: str = "us-east-1",
    download_images: bool = False
) -> bool:
    """
    Convert SageMaker Ground Truth output to YOLOv11 format.
    
    Args:
        input_manifest: S3 URI to the output manifest file from Ground Truth
        output_directory: S3 URI where YOLOv11 format data will be stored
        class_mapping: Dictionary mapping class names to class IDs
        region_name: AWS region name
        download_images: Whether to download and copy images to the output directory
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        logger.info(f"Converting Ground Truth output from {input_manifest} to YOLOv11 format")
        logger.info(f"Output will be stored at {output_directory}")
        logger.info(f"Using class mapping: {class_mapping}")
        
        # Initialize AWS clients
        session = boto3.Session(profile_name='ab', region_name=region_name)
        s3_client = session.client('s3')
        
        # Parse the S3 URIs
        input_bucket, input_key = parse_s3_uri(input_manifest)
        output_bucket, output_prefix = parse_s3_uri(output_directory)
        
        # Create local directories for processing
        local_temp_dir = f"/tmp/ground_truth_to_yolo_{int(time.time())}"
        os.makedirs(local_temp_dir, exist_ok=True)
        os.makedirs(f"{local_temp_dir}/labels", exist_ok=True)
        if download_images:
            os.makedirs(f"{local_temp_dir}/images", exist_ok=True)
        
        # Download the manifest file
        local_manifest_path = f"{local_temp_dir}/manifest.json"
        s3_client.download_file(input_bucket, input_key, local_manifest_path)
        
        # Parse the manifest file
        annotations = []
        with open(local_manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
        
        logger.info(f"Processing {len(annotations)} annotations")
        
        # Process each annotation
        for i, annotation in enumerate(annotations):
            # Get the image URI
            image_uri = annotation.get("source-ref")
            if not image_uri:
                logger.warning(f"No source-ref found in annotation {i}")
                continue
            
            # Extract the image filename
            image_bucket, image_key = parse_s3_uri(image_uri)
            image_filename = os.path.basename(image_key)
            image_name = os.path.splitext(image_filename)[0]
            
            # Get the image dimensions
            try:
                # Download the image if needed to get dimensions
                if download_images:
                    local_image_path = f"{local_temp_dir}/images/{image_filename}"
                    s3_client.download_file(image_bucket, image_key, local_image_path)
                    with Image.open(local_image_path) as img:
                        img_width, img_height = img.size
                else:
                    # Get image dimensions from metadata if available
                    img_width = annotation.get("image_width", 0)
                    img_height = annotation.get("image_height", 0)
                    
                    # If dimensions not in metadata, download image temporarily to get dimensions
                    if img_width == 0 or img_height == 0:
                        temp_image_path = f"{local_temp_dir}/temp_image.jpg"
                        s3_client.download_file(image_bucket, image_key, temp_image_path)
                        with Image.open(temp_image_path) as img:
                            img_width, img_height = img.size
                        os.remove(temp_image_path)
            except Exception as e:
                logger.warning(f"Could not get image dimensions for {image_uri}: {str(e)}")
                continue
            
            # Extract bounding box annotations
            # The exact structure depends on the Ground Truth output format
            # This is for the BoundingBox task type
            bounding_boxes = []
            
            # Look for the bounding box annotations in the expected format
            for key, value in annotation.items():
                if key.endswith("-labels") and isinstance(value, dict) and "annotations" in value:
                    for box_annotation in value["annotations"]:
                        if "label" in box_annotation and "left" in box_annotation and "top" in box_annotation and "width" in box_annotation and "height" in box_annotation:
                            label = box_annotation["label"]
                            left = float(box_annotation["left"])
                            top = float(box_annotation["top"])
                            width = float(box_annotation["width"])
                            height = float(box_annotation["height"])
                            
                            # Convert to YOLO format (class_id, x_center, y_center, width, height)
                            # All values normalized to [0, 1]
                            class_id = class_mapping.get(label, -1)
                            if class_id == -1:
                                logger.warning(f"Unknown class label: {label}")
                                continue
                            
                            x_center = (left + width / 2) / img_width
                            y_center = (top + height / 2) / img_height
                            norm_width = width / img_width
                            norm_height = height / img_height
                            
                            # Ensure values are within [0, 1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            norm_width = max(0, min(1, norm_width))
                            norm_height = max(0, min(1, norm_height))
                            
                            bounding_boxes.append((class_id, x_center, y_center, norm_width, norm_height))
            
            # Create YOLO format label file
            label_file_path = f"{local_temp_dir}/labels/{image_name}.txt"
            with open(label_file_path, 'w') as f:
                for box in bounding_boxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
            
            # Upload the label file to S3
            s3_label_key = f"{output_prefix}/labels/{image_name}.txt"
            s3_client.upload_file(label_file_path, output_bucket, s3_label_key)
            
            # Upload the image if requested
            if download_images:
                s3_image_key = f"{output_prefix}/images/{image_filename}"
                s3_client.upload_file(f"{local_temp_dir}/images/{image_filename}", output_bucket, s3_image_key)
        
        # Create dataset.yaml file for YOLOv11
        dataset_yaml = {
            "path": output_directory,
            "train": "images",
            "val": "images",
            "names": {v: k for k, v in class_mapping.items()}
        }
        
        dataset_yaml_path = f"{local_temp_dir}/dataset.yaml"
        with open(dataset_yaml_path, 'w') as f:
            yaml_content = "# YOLOv11 dataset configuration\n"
            yaml_content += f"path: {output_directory}\n"
            yaml_content += "train: images\n"
            yaml_content += "val: images\n\n"
            yaml_content += "names:\n"
            for class_id, class_name in sorted({v: k for k, v in class_mapping.items()}.items()):
                yaml_content += f"  {class_id}: {class_name}\n"
            f.write(yaml_content)
        
        # Upload the dataset.yaml file to S3
        s3_dataset_yaml_key = f"{output_prefix}/dataset.yaml"
        s3_client.upload_file(dataset_yaml_path, output_bucket, s3_dataset_yaml_key)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(local_temp_dir)
        
        logger.info(f"Conversion complete. YOLO format data stored at {output_directory}")
        return True
    
    except Exception as e:
        logger.error(f"Error converting Ground Truth output to YOLOv11 format: {str(e)}")
        return False


def validate_annotation_quality(
    manifest_file: str,
    validation_criteria: Dict[str, Any] = None,
    region_name: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Validate the quality of annotations from a Ground Truth labeling job.
    
    Args:
        manifest_file: S3 URI to the output manifest file
        validation_criteria: Dictionary of validation criteria
        region_name: AWS region name
        
    Returns:
        Dictionary containing validation results
    """
    if validation_criteria is None:
        validation_criteria = {
            "min_boxes_per_image": 0,
            "max_boxes_per_image": 100,
            "min_box_size": 0.01,  # As percentage of image size
            "max_box_overlap": 0.8,  # IOU threshold
            "min_box_aspect_ratio": 0.1,  # Min width/height ratio
            "max_box_aspect_ratio": 10.0  # Max width/height ratio
        }
    
    try:
        logger.info(f"Validating annotations in {manifest_file}")
        
        # Initialize AWS clients
        session = boto3.Session(profile_name='ab', region_name=region_name)
        s3_client = session.client('s3')
        
        # Parse the S3 URI
        bucket, key = parse_s3_uri(manifest_file)
        
        # Download the manifest file
        local_manifest_path = f"/tmp/manifest_{int(time.time())}.json"
        s3_client.download_file(bucket, key, local_manifest_path)
        
        # Parse the manifest file
        annotations = []
        with open(local_manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
        
        # Initialize validation results
        validation_results = {
            "total_images": len(annotations),
            "total_annotations": 0,
            "images_with_issues": 0,
            "annotations_with_issues": 0,
            "issues_by_type": {
                "too_few_boxes": 0,
                "too_many_boxes": 0,
                "box_too_small": 0,
                "box_aspect_ratio": 0,
                "excessive_overlap": 0
            },
            "issues_by_image": {},
            "pass_rate": 0.0
        }
        
        # Process each annotation
        for annotation in annotations:
            image_uri = annotation.get("source-ref", "unknown")
            image_id = os.path.basename(image_uri)
            
            # Extract bounding box annotations
            boxes = []
            box_count = 0
            
            # Look for the bounding box annotations in the expected format
            for key, value in annotation.items():
                if key.endswith("-labels") and isinstance(value, dict) and "annotations" in value:
                    for box_annotation in value["annotations"]:
                        if "label" in box_annotation and "left" in box_annotation and "top" in box_annotation and "width" in box_annotation and "height" in box_annotation:
                            box_count += 1
                            
                            # Extract box coordinates
                            left = float(box_annotation["left"])
                            top = float(box_annotation["top"])
                            width = float(box_annotation["width"])
                            height = float(box_annotation["height"])
                            label = box_annotation["label"]
                            
                            boxes.append({
                                "left": left,
                                "top": top,
                                "width": width,
                                "height": height,
                                "right": left + width,
                                "bottom": top + height,
                                "label": label
                            })
            
            # Update total annotation count
            validation_results["total_annotations"] += box_count
            
            # Initialize issues for this image
            image_issues = []
            
            # Check box count
            if box_count < validation_criteria["min_boxes_per_image"]:
                validation_results["issues_by_type"]["too_few_boxes"] += 1
                image_issues.append(f"Too few boxes: {box_count} < {validation_criteria['min_boxes_per_image']}")
            
            if box_count > validation_criteria["max_boxes_per_image"]:
                validation_results["issues_by_type"]["too_many_boxes"] += 1
                image_issues.append(f"Too many boxes: {box_count} > {validation_criteria['max_boxes_per_image']}")
            
            # Get image dimensions
            try:
                # Try to get dimensions from annotation metadata
                img_width = annotation.get("image_width", 0)
                img_height = annotation.get("image_height", 0)
                
                # If dimensions not available, download image temporarily to get dimensions
                if img_width == 0 or img_height == 0:
                    image_bucket, image_key = parse_s3_uri(image_uri)
                    temp_image_path = f"/tmp/temp_image_{int(time.time())}.jpg"
                    s3_client.download_file(image_bucket, image_key, temp_image_path)
                    with Image.open(temp_image_path) as img:
                        img_width, img_height = img.size
                    os.remove(temp_image_path)
                
                # Check each box
                for i, box in enumerate(boxes):
                    # Check box size
                    box_area = box["width"] * box["height"]
                    image_area = img_width * img_height
                    box_size_ratio = box_area / image_area
                    
                    if box_size_ratio < validation_criteria["min_box_size"]:
                        validation_results["issues_by_type"]["box_too_small"] += 1
                        image_issues.append(f"Box {i} too small: {box_size_ratio:.4f} < {validation_criteria['min_box_size']}")
                    
                    # Check box aspect ratio
                    if box["width"] > 0 and box["height"] > 0:
                        aspect_ratio = box["width"] / box["height"]
                        if aspect_ratio < validation_criteria["min_box_aspect_ratio"] or aspect_ratio > validation_criteria["max_box_aspect_ratio"]:
                            validation_results["issues_by_type"]["box_aspect_ratio"] += 1
                            image_issues.append(f"Box {i} has unusual aspect ratio: {aspect_ratio:.2f}")
                    
                    # Check overlap with other boxes
                    for j, other_box in enumerate(boxes):
                        if i != j:
                            overlap = calculate_iou(box, other_box)
                            if overlap > validation_criteria["max_box_overlap"]:
                                validation_results["issues_by_type"]["excessive_overlap"] += 1
                                image_issues.append(f"Boxes {i} and {j} have excessive overlap: {overlap:.2f} > {validation_criteria['max_box_overlap']}")
            
            except Exception as e:
                logger.warning(f"Could not validate boxes for {image_uri}: {str(e)}")
                image_issues.append(f"Validation error: {str(e)}")
            
            # Update image issues count
            if image_issues:
                validation_results["images_with_issues"] += 1
                validation_results["annotations_with_issues"] += len(image_issues)
                validation_results["issues_by_image"][image_id] = image_issues
        
        # Calculate pass rate
        if validation_results["total_images"] > 0:
            validation_results["pass_rate"] = (validation_results["total_images"] - validation_results["images_with_issues"]) / validation_results["total_images"]
        
        # Clean up
        os.remove(local_manifest_path)
        
        logger.info(f"Validation complete: {validation_results['pass_rate']:.2%} pass rate")
        return validation_results
    
    except Exception as e:
        logger.error(f"Error validating annotations: {str(e)}")
        raise


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box with left, top, right, bottom coordinates
        box2: Second bounding box with left, top, right, bottom coordinates
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x_left = max(box1["left"], box2["left"])
    y_top = max(box1["top"], box2["top"])
    x_right = min(box1["right"], box2["right"])
    y_bottom = min(box1["bottom"], box2["bottom"])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1["right"] - box1["left"]) * (box1["bottom"] - box1["top"])
    box2_area = (box2["right"] - box2["left"]) * (box2["bottom"] - box2["top"])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def estimate_labeling_cost(
    num_images: int,
    task_type: str = "BoundingBox",
    worker_type: str = "private",
    objects_per_image: float = 3.0,
    region_name: str = "us-east-1"
) -> Dict[str, float]:
    """
    Estimate the cost of a Ground Truth labeling job.
    
    Args:
        num_images: Number of images to label
        task_type: Type of labeling task
        worker_type: Type of workforce
        objects_per_image: Average number of objects per image
        region_name: AWS region name
        
    Returns:
        Dictionary containing cost estimates
    """
    # These are the standard costs for Ground Truth built-in task types
    # Reference: https://aws.amazon.com/sagemaker/groundtruth/pricing/
    cost_per_image = {
        "BoundingBox": {
            "private": 0.0,  # Private workforce has no direct cost
            "public": 0.036,  # $0.036 per image for public workforce (Mechanical Turk)
            "vendor": 0.12    # $0.12 per image for vendor workforce
        },
        "ImageClassification": {
            "private": 0.0,
            "public": 0.012,
            "vendor": 0.04
        },
        "SemanticSegmentation": {
            "private": 0.0,
            "public": 0.072,
            "vendor": 0.24
        }
    }
    
    # AWS service costs per unit
    # These are approximate values and may vary by region
    aws_service_costs = {
        "us-east-1": 0.00040,  # $0.0004 per object
        "us-west-2": 0.00042,
        "eu-west-1": 0.00044,
        "default": 0.00040
    }
    
    # Get the cost per image
    base_cost_per_image = cost_per_image.get(task_type, {}).get(worker_type, 0.0)
    
    # Calculate base cost
    base_cost = num_images * base_cost_per_image
    
    # Adjust for complexity (more objects per image increases cost)
    complexity_factor = max(1.0, objects_per_image / 2.0)
    adjusted_cost = base_cost * complexity_factor
    
    # Add AWS service costs
    service_cost_per_unit = aws_service_costs.get(region_name, aws_service_costs["default"])
    service_cost = num_images * service_cost_per_unit
    
    # Calculate storage costs (assuming average image size of 2MB)
    avg_image_size_gb = 2 / 1024  # 2MB in GB
    storage_cost_per_gb_month = 0.023  # S3 Standard storage cost per GB-month
    storage_cost = num_images * avg_image_size_gb * storage_cost_per_gb_month
    
    # Calculate data transfer costs (simplified)
    data_transfer_cost = 0.0
    if num_images > 1000:
        data_transfer_gb = num_images * avg_image_size_gb
        data_transfer_cost = max(0, data_transfer_gb - 1) * 0.09  # First 1GB free, then $0.09 per GB
    
    # Total cost
    total_cost = adjusted_cost + service_cost + storage_cost + data_transfer_cost
    
    # Add cost breakdown by category
    cost_breakdown = {
        "labeling_cost": adjusted_cost,
        "service_cost": service_cost,
        "storage_cost": storage_cost,
        "data_transfer_cost": data_transfer_cost
    }
    
    # Add cost control recommendations
    cost_control_recommendations = []
    if worker_type == "public" and num_images > 500:
        cost_control_recommendations.append("Consider using a private workforce for large jobs")
    if objects_per_image > 5:
        cost_control_recommendations.append("High object density may increase labeling time and cost")
    if total_cost > 100:
        cost_control_recommendations.append("Consider splitting the job into smaller batches")
    
    return {
        "base_cost": base_cost,
        "adjusted_cost": adjusted_cost,
        "service_cost": service_cost,
        "storage_cost": storage_cost,
        "data_transfer_cost": data_transfer_cost,
        "total_cost": total_cost,
        "cost_per_image": total_cost / num_images,
        "cost_breakdown": cost_breakdown,
        "cost_control_recommendations": cost_control_recommendations
    }


def create_labeling_instructions(
    task_type: str,
    categories: List[str],
    example_images: List[str] = None,
    detailed_instructions: bool = True
) -> str:
    """
    Create HTML instructions for labeling workers.
    
    Args:
        task_type: Type of labeling task
        categories: List of label categories
        example_images: List of S3 URIs to example images
        detailed_instructions: Whether to include detailed instructions
        
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
        
        if detailed_instructions:
            instructions += """
            <h3>Detailed Guidelines</h3>
            
            <h4>For Drones:</h4>
            <ul>
                <li>Include the entire drone body and propellers in the bounding box</li>
                <li>If the drone is partially obscured, draw the box around the visible part</li>
                <li>For very small drones in the distance, ensure the box is at least 10x10 pixels</li>
            </ul>
            
            <h4>For Vehicles:</h4>
            <ul>
                <li>Include cars, trucks, motorcycles, and other ground vehicles</li>
                <li>Draw the box around the entire vehicle</li>
                <li>If multiple vehicles are close together, label each one separately</li>
            </ul>
            
            <h4>For People:</h4>
            <ul>
                <li>Include the entire person from head to feet</li>
                <li>If a person is partially visible, draw the box around the visible part</li>
                <li>Label groups of people individually if they are distinguishable</li>
            </ul>
            
            <h4>For Buildings:</h4>
            <ul>
                <li>Include the entire visible structure</li>
                <li>For large buildings, include all visible walls and roof</li>
                <li>Separate adjacent buildings with individual boxes</li>
            </ul>
            """
    elif task_type == "ImageClassification":
        instructions += """
        <h2>Image Classification Instructions</h2>
        
        <p>Select the appropriate category for the entire image:</p>
        <ol>
            <li>Review the entire image carefully</li>
            <li>Select the category that best describes the main subject</li>
            <li>If multiple categories apply, select the most prominent one</li>
        </ol>
        """
        
        if detailed_instructions:
            instructions += """
            <h3>Classification Guidelines</h3>
            <ul>
                <li>Focus on the most prominent objects in the image</li>
                <li>Consider the context and environment</li>
                <li>If unsure between two categories, select the one that covers more area in the image</li>
            </ul>
            """
    
    # Add quality control information
    instructions += """
    <h2>Quality Guidelines</h2>
    <ul>
        <li>Be precise with your annotations</li>
        <li>Label all relevant objects in the image</li>
        <li>If you're unsure about an object, it's better to label it than to miss it</li>
        <li>Take your time to ensure accuracy</li>
    </ul>
    """
    
    # Add examples if provided
    if example_images:
        instructions += "<h2>Examples</h2>\n"
        for i, image_uri in enumerate(example_images):
            instructions += f'<div><img src="{image_uri}" style="max-width: 500px; margin: 10px;"><p>Example {i+1}</p></div>\n'
    
    return instructions


def create_manifest_file(
    image_files: List[str],
    output_bucket: str,
    output_prefix: str,
    job_name: str,
    region_name: str = "us-east-1"
) -> str:
    """
    Create a manifest file for Ground Truth labeling job.
    
    Args:
        image_files: List of S3 URIs to images
        output_bucket: S3 bucket for output
        output_prefix: S3 prefix for output
        job_name: Name of the labeling job
        region_name: AWS region name
        
    Returns:
        S3 URI to the created manifest file
    """
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    s3_client = session.client('s3')
    
    # Create a temporary file for the manifest
    local_manifest_path = f"/tmp/manifest_{job_name}.json"
    
    # Create manifest data
    with open(local_manifest_path, 'w') as f:
        for image_uri in image_files:
            # Ensure the image URI is properly formatted
            if not image_uri.startswith('s3://'):
                image_uri = f"s3://{image_uri}"
            
            # Create a JSON object for each image
            manifest_entry = {
                "source-ref": image_uri
            }
            
            # Write the JSON object as a line in the manifest file
            f.write(json.dumps(manifest_entry) + '\n')
    
    # Upload the manifest file to S3
    manifest_s3_key = f"{output_prefix}/manifests/{job_name}/manifest.json"
    s3_client.upload_file(local_manifest_path, output_bucket, manifest_s3_key)
    
    # Clean up the local file
    os.remove(local_manifest_path)
    
    # Return the S3 URI to the manifest file
    manifest_uri = f"s3://{output_bucket}/{manifest_s3_key}"
    logger.info(f"Created manifest file at {manifest_uri}")
    
    return manifest_uri


def list_labeling_jobs(
    max_results: int = 10,
    status_filter: Optional[str] = None,
    name_contains: Optional[str] = None,
    region_name: str = "us-east-1"
) -> List[Dict[str, Any]]:
    """
    List SageMaker Ground Truth labeling jobs.
    
    Args:
        max_results: Maximum number of results to return
        status_filter: Filter by job status (InProgress, Completed, Failed, Stopping, Stopped)
        name_contains: Filter by job name containing this string
        region_name: AWS region name
        
    Returns:
        List of labeling job summaries
    """
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    sagemaker_client = session.client('sagemaker')
    
    # Create filter parameters
    filters = []
    if status_filter:
        filters.append({
            "Name": "LabelingJobStatus",
            "Operator": "Equals",
            "Value": status_filter
        })
    if name_contains:
        filters.append({
            "Name": "LabelingJobName",
            "Operator": "Contains",
            "Value": name_contains
        })
    
    # List labeling jobs
    try:
        if filters:
            response = sagemaker_client.list_labeling_jobs(
                MaxResults=max_results,
                StatusEquals=status_filter if status_filter else None,
                NameContains=name_contains if name_contains else None
            )
        else:
            response = sagemaker_client.list_labeling_jobs(
                MaxResults=max_results
            )
        
        # Extract job summaries
        job_summaries = response.get("LabelingJobSummaryList", [])
        
        # Add additional information to each summary
        for summary in job_summaries:
            # Calculate completion percentage
            if summary["LabelCounters"]["TotalObjects"] > 0:
                completion_percentage = (summary["LabelCounters"]["LabeledObjects"] / 
                                        summary["LabelCounters"]["TotalObjects"]) * 100
                summary["CompletionPercentage"] = completion_percentage
            else:
                summary["CompletionPercentage"] = 0.0
        
        return job_summaries
    
    except Exception as e:
        logger.error(f"Error listing labeling jobs: {str(e)}")
        return []


def visualize_annotations(
    manifest_file: str,
    output_dir: str = "/tmp/visualized_annotations",
    max_images: int = 5,
    region_name: str = "us-east-1"
) -> List[str]:
    """
    Visualize annotations from a Ground Truth output manifest.
    
    Args:
        manifest_file: S3 URI to the output manifest file
        output_dir: Local directory to save visualized images
        max_images: Maximum number of images to visualize
        region_name: AWS region name
        
    Returns:
        List of paths to visualized images
    """
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    s3_client = session.client('s3')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the S3 URI
    bucket, key = parse_s3_uri(manifest_file)
    
    # Download the manifest file
    local_manifest_path = f"{output_dir}/manifest.json"
    s3_client.download_file(bucket, key, local_manifest_path)
    
    # Parse the manifest file
    annotations = []
    with open(local_manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    
    # Limit the number of images
    annotations = annotations[:max_images]
    
    # Visualize each annotation
    visualized_images = []
    for i, annotation in enumerate(annotations):
        # Get the image URI
        image_uri = annotation.get("source-ref")
        if not image_uri:
            logger.warning(f"No source-ref found in annotation {i}")
            continue
        
        # Download the image
        image_bucket, image_key = parse_s3_uri(image_uri)
        image_filename = os.path.basename(image_key)
        local_image_path = f"{output_dir}/{image_filename}"
        s3_client.download_file(image_bucket, image_key, local_image_path)
        
        # Open the image
        try:
            image = Image.open(local_image_path)
            draw = ImageDraw.Draw(image)
            
            # Extract bounding box annotations
            for key, value in annotation.items():
                if key.endswith("-labels") and isinstance(value, dict) and "annotations" in value:
                    for box_annotation in value["annotations"]:
                        if "label" in box_annotation and "left" in box_annotation and "top" in box_annotation and "width" in box_annotation and "height" in box_annotation:
                            label = box_annotation["label"]
                            left = float(box_annotation["left"])
                            top = float(box_annotation["top"])
                            width = float(box_annotation["width"])
                            height = float(box_annotation["height"])
                            
                            # Draw the bounding box
                            draw.rectangle([left, top, left + width, top + height], outline="red", width=3)
                            
                            # Draw the label
                            draw.text((left, top - 10), label, fill="red")
            
            # Save the visualized image
            visualized_path = f"{output_dir}/visualized_{image_filename}"
            image.save(visualized_path)
            visualized_images.append(visualized_path)
            
        except Exception as e:
            logger.warning(f"Could not visualize annotations for {image_uri}: {str(e)}")
    
    return visualized_images
def validate_annotation_quality(
    manifest_file: str,
    validation_criteria: Dict[str, float],
    region_name: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Validate the quality of annotations in a Ground Truth output manifest.
    
    Args:
        manifest_file: S3 URI to the output manifest file
        validation_criteria: Dictionary of validation criteria
            - min_boxes_per_image: Minimum number of bounding boxes per image
            - max_boxes_per_image: Maximum number of bounding boxes per image
            - min_box_size: Minimum size of bounding box as fraction of image (area)
            - max_box_overlap: Maximum allowed overlap between boxes (IoU)
        region_name: AWS region name
        
    Returns:
        Dictionary containing validation results
    """
    logger.info(f"Validating annotation quality for manifest: {manifest_file}")
    logger.info(f"Using validation criteria: {validation_criteria}")
    
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    s3_client = session.client('s3')
    
    # Parse the S3 URI
    bucket, key = parse_s3_uri(manifest_file)
    
    try:
        # Download the manifest file
        response = s3_client.get_object(Bucket=bucket, Key=key)
        manifest_content = response['Body'].read().decode('utf-8')
        
        # Parse the manifest (each line is a JSON object)
        annotations = []
        for line in manifest_content.strip().split('\n'):
            if line:
                annotations.append(json.loads(line))
        
        # Initialize validation results
        results = {
            "total_images": len(annotations),
            "total_annotations": 0,
            "images_with_issues": 0,
            "issues": [],
            "pass_rate": 0.0,
            "validation_criteria": validation_criteria
        }
        
        # Process each annotation
        for i, annotation in enumerate(annotations):
            image_uri = annotation.get("source-ref", "")
            image_filename = os.path.basename(image_uri) if image_uri else f"image_{i}"
            
            # Extract bounding box annotations
            boxes = []
            for key, value in annotation.items():
                if key.endswith("-labels") and isinstance(value, dict) and "annotations" in value:
                    for box in value["annotations"]:
                        if "label" in box and "left" in box and "top" in box and "width" in box and "height" in box:
                            boxes.append({
                                "label": box["label"],
                                "left": float(box["left"]),
                                "top": float(box["top"]),
                                "width": float(box["width"]),
                                "height": float(box["height"])
                            })
            
            # Update total annotations count
            results["total_annotations"] += len(boxes)
            
            # Check number of boxes
            image_issues = []
            if len(boxes) < validation_criteria.get("min_boxes_per_image", 0):
                image_issues.append(f"Too few boxes: {len(boxes)} < {validation_criteria.get('min_boxes_per_image', 0)}")
            
            if len(boxes) > validation_criteria.get("max_boxes_per_image", float('inf')):
                image_issues.append(f"Too many boxes: {len(boxes)} > {validation_criteria.get('max_boxes_per_image', float('inf'))}")
            
            # Get image dimensions
            img_width = annotation.get("image_width", 0)
            img_height = annotation.get("image_height", 0)
            
            if img_width == 0 or img_height == 0:
                # Try to extract dimensions from metadata
                for key, value in annotation.items():
                    if key.endswith("-metadata") and isinstance(value, dict):
                        if "internal-dimensions" in value:
                            dims = value["internal-dimensions"]
                            img_width = dims.get("width", 0)
                            img_height = dims.get("height", 0)
            
            # Check box sizes and overlaps
            if img_width > 0 and img_height > 0:
                image_area = img_width * img_height
                min_box_size = validation_criteria.get("min_box_size", 0.0) * image_area
                
                # Check each box size
                for j, box in enumerate(boxes):
                    box_area = box["width"] * box["height"]
                    if box_area < min_box_size:
                        image_issues.append(f"Box {j} too small: {box_area/image_area:.4f} < {validation_criteria.get('min_box_size', 0.0)}")
                
                # Check box overlaps
                max_overlap = validation_criteria.get("max_box_overlap", 1.0)
                for j in range(len(boxes)):
                    for k in range(j+1, len(boxes)):
                        iou = calculate_iou(boxes[j], boxes[k])
                        if iou > max_overlap:
                            image_issues.append(f"Boxes {j} and {k} overlap too much: IoU = {iou:.4f} > {max_overlap}")
            
            # Add issues to results if any
            if image_issues:
                results["images_with_issues"] += 1
                results["issues"].append({
                    "image": image_filename,
                    "issues": image_issues
                })
        
        # Calculate pass rate
        if results["total_images"] > 0:
            results["pass_rate"] = (results["total_images"] - results["images_with_issues"]) / results["total_images"]
        
        logger.info(f"Validation complete: {results['pass_rate']*100:.2f}% of images passed validation")
        logger.info(f"Found {results['images_with_issues']} images with issues out of {results['total_images']}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error validating annotations: {str(e)}")
        return {
            "error": str(e),
            "total_images": 0,
            "total_annotations": 0,
            "images_with_issues": 0,
            "issues": [],
            "pass_rate": 0.0
        }

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box with left, top, width, height
        box2: Second bounding box with left, top, width, height
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate box coordinates
    box1_x1 = box1["left"]
    box1_y1 = box1["top"]
    box1_x2 = box1["left"] + box1["width"]
    box1_y2 = box1["top"] + box1["height"]
    
    box2_x1 = box2["left"]
    box2_y1 = box2["top"]
    box2_x2 = box2["left"] + box2["width"]
    box2_y2 = box2["top"] + box2["height"]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        # No intersection
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def estimate_labeling_cost(
    num_images: int,
    task_type: str = "BoundingBox",
    worker_type: str = "private",
    objects_per_image: float = 3.0
) -> Dict[str, Any]:
    """
    Estimate the cost of a labeling job.
    
    Args:
        num_images: Number of images to label
        task_type: Type of labeling task
        worker_type: Type of workforce
        objects_per_image: Average number of objects per image
        
    Returns:
        Dictionary containing cost estimates
    """
    logger.info(f"Estimating labeling cost for {num_images} images")
    logger.info(f"Task type: {task_type}, Worker type: {worker_type}, Objects per image: {objects_per_image}")
    
    # Define base costs per image for different task types and worker types
    # These are approximate costs and may vary
    base_costs = {
        "BoundingBox": {
            "private": 0.0,  # Private workforce has no direct cost
            "public": 0.036   # $0.036 per image for public workforce (example)
        },
        "ImageClassification": {
            "private": 0.0,
            "public": 0.012
        },
        "SemanticSegmentation": {
            "private": 0.0,
            "public": 0.072
        }
    }
    
    # Get the base cost per image
    base_cost_per_image = base_costs.get(task_type, {}).get(worker_type, 0.0)
    
    # Calculate base cost
    base_cost = base_cost_per_image * num_images
    
    # Adjust cost based on objects per image
    # More objects typically means more time and potentially higher cost
    if worker_type == "public" and objects_per_image > 1.0:
        # Adjust cost based on number of objects
        # This is a simplified model - actual costs may vary
        adjustment_factor = min(2.0, 0.7 + (objects_per_image / 10.0))
        adjusted_cost = base_cost * adjustment_factor
    else:
        adjustment_factor = 1.0
        adjusted_cost = base_cost
    
    # Calculate AWS service costs (approximate)
    # These include SageMaker Ground Truth service fees
    service_cost = 0.0
    if worker_type == "public":
        # AWS adds a service fee on top of worker payments
        service_cost = adjusted_cost * 0.12
    
    # Storage costs for input/output data (approximate)
    # Assuming average image size of 2MB and annotation size of 5KB
    avg_image_size_mb = 2.0
    avg_annotation_size_mb = 0.005
    storage_cost_per_gb_month = 0.023  # S3 Standard storage cost
    
    # Calculate storage size in GB
    input_storage_gb = (num_images * avg_image_size_mb) / 1024
    output_storage_gb = (num_images * avg_annotation_size_mb) / 1024
    total_storage_gb = input_storage_gb + output_storage_gb
    
    # Calculate storage cost for 1 month
    storage_cost = total_storage_gb * storage_cost_per_gb_month
    
    # Data transfer costs (approximate)
    # Assuming all data is transferred within AWS
    data_transfer_cost = 0.0
    
    # Calculate total cost
    total_cost = adjusted_cost + service_cost + storage_cost + data_transfer_cost
    
    # Calculate cost per image
    cost_per_image = total_cost / num_images if num_images > 0 else 0.0
    
    # Generate cost control recommendations
    cost_control_recommendations = []
    
    if worker_type == "public" and num_images > 100:
        cost_control_recommendations.append(
            "Consider using a private workforce for large labeling jobs to reduce costs."
        )
    
    if objects_per_image > 5.0:
        cost_control_recommendations.append(
            "Images with many objects may take longer to label. Consider simplifying the task or breaking it down."
        )
    
    if total_cost > 1000:
        cost_control_recommendations.append(
            "For large jobs, consider labeling in batches to control costs and evaluate quality incrementally."
        )
    
    # Create cost breakdown
    cost_breakdown = {
        "labeling_cost": adjusted_cost,
        "service_cost": service_cost,
        "storage_cost": storage_cost,
        "data_transfer_cost": data_transfer_cost
    }
    
    # Create result dictionary
    result = {
        "num_images": num_images,
        "task_type": task_type,
        "worker_type": worker_type,
        "objects_per_image": objects_per_image,
        "base_cost_per_image": base_cost_per_image,
        "base_cost": base_cost,
        "adjustment_factor": adjustment_factor,
        "adjusted_cost": adjusted_cost,
        "service_cost": service_cost,
        "storage_cost": storage_cost,
        "data_transfer_cost": data_transfer_cost,
        "total_cost": total_cost,
        "cost_per_image": cost_per_image,
        "cost_breakdown": cost_breakdown,
        "cost_control_recommendations": cost_control_recommendations
    }
    
    logger.info(f"Estimated total cost: ${total_cost:.2f}")
    logger.info(f"Cost per image: ${cost_per_image:.4f}")
    
    return result

def create_labeling_instructions(
    task_type: str,
    categories: List[str],
    example_images: List[str] = None,
    detailed_instructions: bool = True
) -> str:
    """
    Create HTML instructions for a labeling job.
    
    Args:
        task_type: Type of labeling task
        categories: List of label categories
        example_images: List of example image URIs
        detailed_instructions: Whether to include detailed instructions
        
    Returns:
        HTML string with instructions
    """
    logger.info(f"Creating labeling instructions for {task_type} task")
    logger.info(f"Categories: {categories}")
    
    # Create basic instructions
    title = f"Drone Imagery {task_type} Instructions"
    
    # Create category list
    category_list = "".join([f"<li><strong>{category}</strong></li>" for category in categories])
    
    # Create basic instructions
    basic_instructions = f"""
    <h1>{title}</h1>
    <p>Please label all objects in the drone imagery according to the following categories:</p>
    <ul>
        {category_list}
    </ul>
    <p>Be as precise as possible when drawing bounding boxes around objects.</p>
    """
    
    # Add detailed instructions if requested
    if detailed_instructions:
        if task_type == "BoundingBox":
            detailed_content = """
            <h2>Detailed Instructions</h2>
            <h3>How to Draw Bounding Boxes</h3>
            <ol>
                <li>Click and drag to draw a box around each object</li>
                <li>Make sure the box completely contains the object</li>
                <li>Make the box as tight as possible around the object</li>
                <li>If objects overlap, draw separate boxes for each object</li>
                <li>If an object is partially visible, only label the visible part</li>
                <li>If you're unsure about an object, label it with your best guess</li>
            </ol>
            
            <h3>Tips for Accurate Labeling</h3>
            <ul>
                <li>Zoom in to see small objects more clearly</li>
                <li>Take your time to ensure accuracy</li>
                <li>If an object is cut off at the edge of the image, include the visible portion</li>
                <li>If you make a mistake, you can delete a box and redraw it</li>
            </ul>
            """
        elif task_type == "ImageClassification":
            detailed_content = """
            <h2>Detailed Instructions</h2>
            <h3>How to Classify Images</h3>
            <ol>
                <li>Look at the entire image carefully</li>
                <li>Select the most appropriate category for the image</li>
                <li>If multiple categories apply, select the most prominent one</li>
                <li>If you're unsure, select your best guess</li>
            </ol>
            
            <h3>Tips for Accurate Classification</h3>
            <ul>
                <li>Take your time to examine the entire image</li>
                <li>Consider the main subject of the image</li>
                <li>If the image is unclear, make your best judgment</li>
            </ul>
            """
        else:
            detailed_content = ""
    else:
        detailed_content = ""
    
    # Add example images if provided
    example_content = ""
    if example_images and len(example_images) > 0:
        example_list = "".join([f'<li><img src="{img}" alt="Example" style="max-width: 300px; margin: 10px;"></li>' for img in example_images])
        example_content = f"""
        <h2>Example Images</h2>
        <p>Here are some examples of correctly labeled images:</p>
        <ul style="list-style-type: none;">
            {example_list}
        </ul>
        """
    
    # Combine all sections
    instructions = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        {basic_instructions}
        {detailed_content}
        {example_content}
        
        <h2>Thank You!</h2>
        <p>Your careful labeling helps train accurate machine learning models for drone imagery analysis.</p>
    </div>
    """
    
    return instructions

def create_manifest_file(
    image_files: List[str],
    output_bucket: str,
    output_prefix: str,
    job_name: str,
    region_name: str = "us-east-1"
) -> str:
    """
    Create a manifest file for a Ground Truth labeling job.
    
    Args:
        image_files: List of S3 URIs to images
        output_bucket: S3 bucket for output
        output_prefix: S3 prefix for output
        job_name: Name of the labeling job
        region_name: AWS region name
        
    Returns:
        S3 URI to the created manifest file
    """
    logger.info(f"Creating manifest file for {len(image_files)} images")
    
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    s3_client = session.client('s3')
    
    # Create a temporary file for the manifest
    manifest_file = f"/tmp/manifest_{job_name}.json"
    
    # Create manifest content
    with open(manifest_file, 'w') as f:
        for image_uri in image_files:
            # Create a JSON object for each image
            manifest_entry = {
                "source-ref": image_uri
            }
            f.write(json.dumps(manifest_entry) + '\n')
    
    # Upload the manifest file to S3
    manifest_key = f"{output_prefix}/manifests/{job_name}/manifest.json"
    s3_client.upload_file(manifest_file, output_bucket, manifest_key)
    
    # Clean up the temporary file
    os.remove(manifest_file)
    
    # Return the S3 URI to the manifest file
    manifest_uri = f"s3://{output_bucket}/{manifest_key}"
    logger.info(f"Manifest file created at: {manifest_uri}")
    
    return manifest_uri

def list_labeling_jobs(
    max_results: int = 10,
    name_contains: str = None,
    status_equals: str = None,
    sort_by: str = "CreationTime",
    sort_order: str = "Descending",
    region_name: str = "us-east-1"
) -> List[Dict[str, Any]]:
    """
    List SageMaker Ground Truth labeling jobs.
    
    Args:
        max_results: Maximum number of results to return
        name_contains: Filter jobs by name containing this string
        status_equals: Filter jobs by status (InProgress, Completed, Failed, Stopping, Stopped)
        sort_by: Sort by field (CreationTime, LastModifiedTime)
        sort_order: Sort order (Ascending, Descending)
        region_name: AWS region name
        
    Returns:
        List of labeling job metadata
    """
    logger.info(f"Listing labeling jobs (max: {max_results})")
    
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    sagemaker_client = session.client('sagemaker')
    
    # Create filter parameters
    params = {
        "MaxResults": max_results,
        "SortBy": sort_by,
        "SortOrder": sort_order
    }
    
    if name_contains:
        params["NameContains"] = name_contains
    
    if status_equals:
        params["StatusEquals"] = status_equals
    
    # List labeling jobs
    try:
        response = sagemaker_client.list_labeling_jobs(**params)
        
        # Extract job metadata
        jobs = []
        for job in response.get("LabelingJobSummaryList", []):
            jobs.append({
                "name": job["LabelingJobName"],
                "status": job["LabelingJobStatus"],
                "creation_time": job["CreationTime"],
                "last_modified_time": job["LastModifiedTime"],
                "label_counters": job.get("LabelCounters", {}),
                "failure_reason": job.get("FailureReason", None),
                "job_arn": job["LabelingJobArn"]
            })
        
        logger.info(f"Found {len(jobs)} labeling jobs")
        return jobs
    
    except Exception as e:
        logger.error(f"Error listing labeling jobs: {str(e)}")
        return []

def visualize_annotations(
    manifest_file: str,
    output_directory: str = "/tmp/visualizations",
    max_images: int = 10,
    region_name: str = "us-east-1"
) -> List[str]:
    """
    Visualize annotations from a Ground Truth output manifest.
    
    Args:
        manifest_file: S3 URI to the output manifest file
        output_directory: Local directory to save visualizations
        max_images: Maximum number of images to visualize
        region_name: AWS region name
        
    Returns:
        List of paths to visualization images
    """
    logger.info(f"Visualizing annotations from {manifest_file}")
    
    # Initialize AWS clients
    session = boto3.Session(profile_name='ab', region_name=region_name)
    s3_client = session.client('s3')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Parse the S3 URI
    bucket, key = parse_s3_uri(manifest_file)
    
    try:
        # Download the manifest file
        response = s3_client.get_object(Bucket=bucket, Key=key)
        manifest_content = response['Body'].read().decode('utf-8')
        
        # Parse the manifest (each line is a JSON object)
        annotations = []
        for line in manifest_content.strip().split('\n'):
            if line:
                annotations.append(json.loads(line))
        
        # Limit to max_images
        annotations = annotations[:max_images]
        
        # Process each annotation
        visualization_paths = []
        for i, annotation in enumerate(annotations):
            # Get the image URI
            image_uri = annotation.get("source-ref", "")
            if not image_uri:
                logger.warning(f"No source-ref found in annotation {i}")
                continue
            
            # Extract the image filename
            image_bucket, image_key = parse_s3_uri(image_uri)
            image_filename = os.path.basename(image_key)
            
            # Download the image
            local_image_path = f"{output_directory}/{image_filename}"
            s3_client.download_file(image_bucket, image_key, local_image_path)
            
            # Open the image
            with Image.open(local_image_path) as img:
                draw = ImageDraw.Draw(img)
                
                # Extract bounding box annotations
                for key, value in annotation.items():
                    if key.endswith("-labels") and isinstance(value, dict) and "annotations" in value:
                        for box in value["annotations"]:
                            if "label" in box and "left" in box and "top" in box and "width" in box and "height" in box:
                                label = box["label"]
                                left = float(box["left"])
                                top = float(box["top"])
                                width = float(box["width"])
                                height = float(box["height"])
                                
                                # Draw the bounding box
                                draw.rectangle(
                                    [(left, top), (left + width, top + height)],
                                    outline="red",
                                    width=3
                                )
                                
                                # Draw the label
                                draw.text((left, top - 10), label, fill="red")
                
                # Save the annotated image
                visualization_path = f"{output_directory}/annotated_{image_filename}"
                img.save(visualization_path)
                visualization_paths.append(visualization_path)
        
        logger.info(f"Created {len(visualization_paths)} visualization images")
        return visualization_paths
    
    except Exception as e:
        logger.error(f"Error visualizing annotations: {str(e)}")
        return []