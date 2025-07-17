#!/usr/bin/env python
"""
Example script for creating and managing SageMaker Ground Truth labeling jobs.

This script demonstrates how to:
1. Create a manifest file for labeling
2. Configure and start a Ground Truth labeling job
3. Monitor job progress
4. Process completed labels

Usage:
    python ground_truth_example.py --bucket lucaskle-ab3-project-pv --prefix raw-images
"""

import argparse
import boto3
import json
import time
import os
import datetime
from typing import List, Dict, Any

from src.data.ground_truth_utils import (
    create_labeling_job_config,
    monitor_labeling_job,
    convert_ground_truth_to_yolo,
    create_labeling_instructions
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a Ground Truth labeling job")
    parser.add_argument("--bucket", required=True, help="S3 bucket containing images")
    parser.add_argument("--prefix", required=True, help="S3 prefix for images")
    parser.add_argument("--categories", nargs="+", default=["drone", "vehicle", "person", "building"],
                        help="Object categories for labeling")
    parser.add_argument("--job-name", help="Custom job name (default: auto-generated)")
    parser.add_argument("--worker-type", choices=["private", "public"], default="private",
                        help="Type of workforce to use")
    parser.add_argument("--max-budget", type=float, default=100.0,
                        help="Maximum budget in USD for the labeling job")
    return parser.parse_args()


def create_manifest_file(bucket: str, prefix: str) -> str:
    """
    Create a manifest file for Ground Truth labeling.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix for images
        
    Returns:
        S3 URI to the created manifest file
    """
    print(f"Creating manifest file for images in s3://{bucket}/{prefix}")
    
    # Initialize S3 client
    session = boto3.Session(profile_name='ab')
    s3_client = session.client('s3')
    
    # List objects in the bucket with the given prefix
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix
    )
    
    # Filter for image files
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []
    
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if any(key.lower().endswith(ext) for ext in image_extensions):
                image_files.append(key)
    
    print(f"Found {len(image_files)} images for labeling")
    
    # Create manifest data
    manifest_data = []
    for image_file in image_files:
        manifest_data.append({
            "source-ref": f"s3://{bucket}/{image_file}"
        })
    
    # Write manifest to a local file
    manifest_file = "manifest.json"
    with open(manifest_file, 'w') as f:
        for item in manifest_data:
            f.write(json.dumps(item) + '\n')
    
    # Upload manifest to S3
    manifest_s3_key = f"ground-truth-jobs/manifest/manifest-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    s3_client.upload_file(manifest_file, bucket, manifest_s3_key)
    
    # Clean up local file
    os.remove(manifest_file)
    
    manifest_s3_uri = f"s3://{bucket}/{manifest_s3_key}"
    print(f"Manifest file uploaded to {manifest_s3_uri}")
    
    return manifest_s3_uri


def create_and_start_labeling_job(
    manifest_uri: str,
    bucket: str,
    job_name: str = None,
    categories: List[str] = None,
    worker_type: str = "private",
    max_budget_usd: float = 100.0
) -> str:
    """
    Create and start a Ground Truth labeling job.
    
    Args:
        manifest_uri: S3 URI to the manifest file
        bucket: S3 bucket name
        job_name: Custom job name (optional)
        categories: List of object categories (optional)
        worker_type: Type of workforce to use (optional)
        max_budget_usd: Maximum budget in USD (optional)
        
    Returns:
        Name of the created labeling job
    """
    # Set default categories if not provided
    if categories is None:
        categories = ["drone", "vehicle", "person", "building"]
    
    # Generate job name if not provided
    if job_name is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        job_name = f"drone-detection-{timestamp}"
    
    print(f"Creating labeling job: {job_name}")
    
    # Define output path
    output_path = f"s3://{bucket}/ground-truth-jobs/output/"
    
    # Create labeling instructions
    instructions = create_labeling_instructions(
        task_type="BoundingBox",
        categories=categories,
        example_images=[
            f"s3://{bucket}/examples/example1.jpg",
            f"s3://{bucket}/examples/example2.jpg"
        ]
    )
    
    # Create labeling job configuration
    labeling_job_config = create_labeling_job_config(
        job_name=job_name,
        input_path=manifest_uri,
        output_path=output_path,
        task_type="BoundingBox",
        worker_type=worker_type,
        labels=categories,
        instructions=instructions,
        max_budget_usd=max_budget_usd
    )
    
    # Create SageMaker client
    session = boto3.Session(profile_name='ab')
    sagemaker_client = session.client('sagemaker')
    
    # Start the labeling job
    response = sagemaker_client.create_labeling_job(**labeling_job_config)
    
    print(f"Labeling job created: {job_name}")
    print(f"Job ARN: {response['LabelingJobArn']}")
    
    return job_name


def monitor_job_until_complete(job_name: str, polling_interval: int = 60) -> Dict[str, Any]:
    """
    Monitor a labeling job until it completes.
    
    Args:
        job_name: Name of the labeling job
        polling_interval: Time between status checks in seconds
        
    Returns:
        Final job status
    """
    print(f"Monitoring labeling job: {job_name}")
    
    # Create SageMaker client
    session = boto3.Session(profile_name='ab')
    sagemaker_client = session.client('sagemaker')
    
    # Monitor the job until it completes
    while True:
        # Get job status
        status = monitor_labeling_job(job_name, sagemaker_client)
        
        # Print status information
        print(f"Job Status: {status['LabelingJobStatus']}")
        print(f"Total Objects: {status['LabelCounters']['TotalObjects']}")
        print(f"Labeled Objects: {status['LabelCounters']['LabeledObjects']}")
        print(f"Failed Objects: {status['LabelCounters']['FailedObjects']}")
        
        # Calculate completion percentage
        if status['LabelCounters']['TotalObjects'] > 0:
            completion_percentage = (status['LabelCounters']['LabeledObjects'] / 
                                    status['LabelCounters']['TotalObjects']) * 100
            print(f"Completion: {completion_percentage:.2f}%")
        
        # Check if job is complete
        if status['LabelingJobStatus'] in ['Completed', 'Failed', 'Stopped']:
            print(f"Job {job_name} finished with status: {status['LabelingJobStatus']}")
            return status
        
        # Wait before checking again
        print(f"Waiting {polling_interval} seconds before next status check...")
        time.sleep(polling_interval)


def process_completed_labels(job_name: str, bucket: str, categories: List[str]) -> str:
    """
    Process completed labels and convert to YOLOv11 format.
    
    Args:
        job_name: Name of the completed labeling job
        bucket: S3 bucket name
        categories: List of object categories
        
    Returns:
        S3 URI to the converted YOLO format data
    """
    print(f"Processing completed labels for job: {job_name}")
    
    # Define paths
    output_path = f"s3://{bucket}/ground-truth-jobs/output/"
    output_manifest = f"{output_path}{job_name}/manifests/output/output.manifest"
    yolo_output_dir = f"s3://{bucket}/ground-truth-jobs/yolo-format/{job_name}/"
    
    # Create class mapping
    class_mapping = {category: i for i, category in enumerate(categories)}
    
    print(f"Converting Ground Truth output to YOLOv11 format")
    print(f"Class mapping: {class_mapping}")
    
    # Convert to YOLOv11 format
    convert_ground_truth_to_yolo(
        input_manifest=output_manifest,
        output_directory=yolo_output_dir,
        class_mapping=class_mapping
    )
    
    print(f"Converted annotations saved to: {yolo_output_dir}")
    
    return yolo_output_dir


def main():
    """Main function to run the example."""
    # Parse command line arguments
    args = parse_args()
    
    # Create manifest file
    manifest_uri = create_manifest_file(args.bucket, args.prefix)
    
    # Create and start labeling job
    job_name = create_and_start_labeling_job(
        manifest_uri=manifest_uri,
        bucket=args.bucket,
        job_name=args.job_name,
        categories=args.categories,
        worker_type=args.worker_type,
        max_budget_usd=args.max_budget
    )
    
    # Monitor job until complete
    final_status = monitor_job_until_complete(job_name)
    
    # Process completed labels if job was successful
    if final_status['LabelingJobStatus'] == 'Completed':
        yolo_output_dir = process_completed_labels(
            job_name=job_name,
            bucket=args.bucket,
            categories=args.categories
        )
        
        print(f"Labeling job complete. YOLO format data available at: {yolo_output_dir}")
    else:
        print(f"Labeling job did not complete successfully. Final status: {final_status['LabelingJobStatus']}")
        if 'FailureReason' in final_status:
            print(f"Failure reason: {final_status['FailureReason']}")


if __name__ == "__main__":
    main()