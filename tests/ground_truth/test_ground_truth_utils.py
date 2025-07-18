"""
Unit tests for Ground Truth utilities.

This module contains comprehensive tests for the ground_truth_utils.py module,
which provides utilities for creating, monitoring, and processing SageMaker
Ground Truth labeling jobs for drone imagery object detection.
"""

import unittest
import json
import os
import boto3
import pytest
from unittest.mock import patch, MagicMock
from src.data.ground_truth_utils import (
    create_labeling_job_config,
    monitor_labeling_job,
    convert_ground_truth_to_yolo,
    validate_annotation_quality,
    estimate_labeling_cost,
    create_labeling_instructions,
    create_manifest_file,
    list_labeling_jobs,
    visualize_annotations,
    calculate_iou,
    get_labeling_job_metrics,
    parse_s3_uri
)


class TestGroundTruthUtils(unittest.TestCase):
    """Test cases for Ground Truth utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.job_name = "test-labeling-job"
        self.input_path = "s3://test-bucket/input/manifest.json"
        self.output_path = "s3://test-bucket/output/"
        self.role_arn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.labels = ["drone", "vehicle", "person", "building"]
        
    def test_create_labeling_job_config(self):
        """Test creating a labeling job configuration."""
        config = create_labeling_job_config(
            job_name=self.job_name,
            input_path=self.input_path,
            output_path=self.output_path,
            task_type="BoundingBox",
            worker_type="private",
            labels=self.labels,
            instructions="Test instructions",
            max_budget_usd=100.0,
            role_arn=self.role_arn
        )
        
        # Verify basic configuration
        self.assertEqual(config["LabelingJobName"], self.job_name)
        self.assertEqual(config["InputConfig"]["DataSource"]["S3DataSource"]["ManifestS3Uri"], self.input_path)
        self.assertEqual(config["OutputConfig"]["S3OutputPath"], self.output_path)
        self.assertEqual(config["RoleArn"], self.role_arn)
        
        # Verify task configuration
        self.assertEqual(config["HumanTaskConfig"]["TaskDescription"], "Test instructions")
        
        # Verify budget configuration
        budget_tag = next((tag for tag in config["Tags"] if tag["Key"] == "MaxBudgetUSD"), None)
        self.assertIsNotNone(budget_tag)
        self.assertEqual(budget_tag["Value"], "100.0")
    
    @patch('boto3.Session')
    def test_monitor_labeling_job(self, mock_session):
        """Test monitoring a labeling job."""
        # Create mock response with datetime objects
        from datetime import datetime
        creation_time = datetime(2023, 1, 1, 0, 0, 0)
        last_modified_time = datetime(2023, 1, 1, 1, 0, 0)
        
        mock_response = {
            "LabelingJobName": self.job_name,
            "LabelingJobStatus": "InProgress",
            "LabelingJobArn": f"arn:aws:sagemaker:us-east-1:123456789012:labeling-job/{self.job_name}",
            "CreationTime": creation_time,
            "LastModifiedTime": last_modified_time,
            "LabelCounters": {
                "TotalObjects": 100,
                "LabeledObjects": 50,
                "FailedObjects": 0
            }
        }
        
        # Set up mock client
        mock_client = MagicMock()
        mock_client.describe_labeling_job.return_value = mock_response
        mock_session.return_value.client.return_value = mock_client
        
        # Call the function
        status = monitor_labeling_job(self.job_name, mock_client)
        
        # Verify the function called the API correctly
        mock_client.describe_labeling_job.assert_called_once_with(
            LabelingJobName=self.job_name
        )
        
        # Verify the returned status
        self.assertEqual(status["LabelingJobName"], self.job_name)
        self.assertEqual(status["LabelingJobStatus"], "InProgress")
        self.assertEqual(status["LabelCounters"]["TotalObjects"], 100)
        self.assertEqual(status["LabelCounters"]["LabeledObjects"], 50)
    
    @patch('boto3.Session')
    def test_convert_ground_truth_to_yolo(self, mock_session):
        """Test converting Ground Truth output to YOLOv11 format."""
        input_manifest = "s3://test-bucket/output/manifest.json"
        output_directory = "s3://test-bucket/yolo-format/"
        class_mapping = {
            "drone": 0,
            "vehicle": 1,
            "person": 2,
            "building": 3
        }
        
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_session.return_value.client.return_value = mock_s3_client
        
        # Mock the get_object response
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "source-ref": "s3://test-bucket/images/image1.jpg",
                "bounding-box-labels": {
                    "annotations": [
                        {
                            "label": "drone",
                            "left": 10,
                            "top": 10,
                            "width": 20,
                            "height": 20
                        }
                    ]
                },
                "image_width": 640,
                "image_height": 480
            }).encode())
        }
        
        # Mock the file operations
        with patch('os.makedirs'), patch('builtins.open', unittest.mock.mock_open()), patch('PIL.Image.open'):
            # Call the function
            result = convert_ground_truth_to_yolo(
                input_manifest=input_manifest,
                output_directory=output_directory,
                class_mapping=class_mapping
            )
            
            # Verify the result
            self.assertTrue(result)
    
    def test_validate_annotation_quality(self):
        """Test validating annotation quality."""
        manifest_file = "s3://test-bucket/output/manifest.json"
        validation_criteria = {
            "min_boxes_per_image": 1,
            "max_boxes_per_image": 50,
            "min_box_size": 0.01,
            "max_box_overlap": 0.8
        }
        
        # Since this is a placeholder implementation, we just verify it returns a dictionary
        results = validate_annotation_quality(
            manifest_file=manifest_file,
            validation_criteria=validation_criteria
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn("total_images", results)
        self.assertIn("total_annotations", results)
        self.assertIn("pass_rate", results)
    
    def test_estimate_labeling_cost(self):
        """Test estimating labeling costs."""
        num_images = 1000
        task_type = "BoundingBox"
        worker_type = "public"
        objects_per_image = 5.0
        
        cost_estimate = estimate_labeling_cost(
            num_images=num_images,
            task_type=task_type,
            worker_type=worker_type,
            objects_per_image=objects_per_image
        )
        
        self.assertIsInstance(cost_estimate, dict)
        self.assertIn("base_cost", cost_estimate)
        self.assertIn("total_cost", cost_estimate)
        self.assertGreater(cost_estimate["total_cost"], 0)
    
    def test_create_labeling_instructions(self):
        """Test creating labeling instructions."""
        task_type = "BoundingBox"
        categories = ["drone", "vehicle", "person", "building"]
        example_images = [
            "s3://test-bucket/examples/example1.jpg",
            "s3://test-bucket/examples/example2.jpg"
        ]
        
        instructions = create_labeling_instructions(
            task_type=task_type,
            categories=categories,
            example_images=example_images
        )
        
        self.assertIsInstance(instructions, str)
        self.assertIn("Drone Imagery BoundingBox Instructions", instructions)
        for category in categories:
            self.assertIn(category, instructions)
        for image in example_images:
            self.assertIn(image, instructions)
    
    def test_calculate_iou(self):
        """Test calculating IoU between two bounding boxes."""
        # Test case 1: No overlap
        box1 = {"left": 0, "top": 0, "width": 10, "height": 10}
        box2 = {"left": 20, "top": 20, "width": 10, "height": 10}
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
        
        # Test case 2: Complete overlap (identical boxes)
        box1 = {"left": 10, "top": 10, "width": 10, "height": 10}
        box2 = {"left": 10, "top": 10, "width": 10, "height": 10}
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 1.0)
        
        # Test case 3: Partial overlap
        box1 = {"left": 10, "top": 10, "width": 10, "height": 10}
        box2 = {"left": 15, "top": 15, "width": 10, "height": 10}
        iou = calculate_iou(box1, box2)
        # Expected IoU: area of intersection (5x5) / area of union (10x10 + 10x10 - 5x5)
        expected_iou = 25 / 175
        self.assertAlmostEqual(iou, expected_iou)
    
    @patch('boto3.Session')
    def test_create_manifest_file(self, mock_session):
        """Test creating a manifest file for a labeling job."""
        # Set up mock S3 client
        mock_s3_client = MagicMock()
        mock_session.return_value.client.return_value = mock_s3_client
        
        # Test data
        image_files = [
            "s3://test-bucket/images/image1.jpg",
            "s3://test-bucket/images/image2.jpg",
            "s3://test-bucket/images/image3.jpg"
        ]
        output_bucket = "test-bucket"
        output_prefix = "ground-truth-jobs"
        job_name = "test-job"
        
        # Call the function
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            manifest_uri = create_manifest_file(
                image_files=image_files,
                output_bucket=output_bucket,
                output_prefix=output_prefix,
                job_name=job_name
            )
        
        # Verify the function called the S3 client correctly
        mock_s3_client.upload_file.assert_called_once()
        
        # Verify the returned URI
        expected_uri = f"s3://{output_bucket}/{output_prefix}/manifests/{job_name}/manifest.json"
        self.assertEqual(manifest_uri, expected_uri)
    
    @patch('boto3.Session')
    def test_list_labeling_jobs(self, mock_session):
        """Test listing labeling jobs."""
        # Set up mock SageMaker client
        mock_sagemaker_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sagemaker_client
        
        # Create mock response
        mock_response = {
            "LabelingJobSummaryList": [
                {
                    "LabelingJobName": "job-1",
                    "LabelingJobStatus": "Completed",
                    "CreationTime": "2023-01-01T00:00:00Z",
                    "LastModifiedTime": "2023-01-01T01:00:00Z",
                    "LabelCounters": {
                        "TotalObjects": 100,
                        "LabeledObjects": 100,
                        "FailedObjects": 0
                    },
                    "LabelingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:labeling-job/job-1"
                },
                {
                    "LabelingJobName": "job-2",
                    "LabelingJobStatus": "InProgress",
                    "CreationTime": "2023-01-02T00:00:00Z",
                    "LastModifiedTime": "2023-01-02T01:00:00Z",
                    "LabelCounters": {
                        "TotalObjects": 100,
                        "LabeledObjects": 50,
                        "FailedObjects": 0
                    },
                    "LabelingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:labeling-job/job-2"
                }
            ]
        }
        mock_sagemaker_client.list_labeling_jobs.return_value = mock_response
        
        # Call the function
        jobs = list_labeling_jobs(
            max_results=10,
            name_contains="job",
            status_equals="InProgress"
        )
        
        # Verify the function called the SageMaker client correctly
        mock_sagemaker_client.list_labeling_jobs.assert_called_once_with(
            MaxResults=10,
            NameContains="job",
            StatusEquals="InProgress",
            SortBy="CreationTime",
            SortOrder="Descending"
        )
        
        # Verify the returned jobs
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["name"], "job-1")
        self.assertEqual(jobs[1]["name"], "job-2")
        self.assertEqual(jobs[0]["status"], "Completed")
        self.assertEqual(jobs[1]["status"], "InProgress")
    
    @patch('boto3.Session')
    @patch('PIL.Image.open')
    def test_visualize_annotations(self, mock_image_open, mock_session):
        """Test visualizing annotations."""
        # Set up mock S3 client
        mock_s3_client = MagicMock()
        mock_session.return_value.client.return_value = mock_s3_client
        
        # Create mock response for get_object
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "source-ref": "s3://test-bucket/images/image1.jpg",
                "bounding-box-labels": {
                    "annotations": [
                        {
                            "label": "drone",
                            "left": 10,
                            "top": 10,
                            "width": 20,
                            "height": 20
                        }
                    ]
                }
            }).encode())
        }
        
        # Set up mock image and draw
        mock_image = MagicMock()
        mock_draw = MagicMock()
        mock_image.__enter__.return_value = mock_image
        mock_image.save = MagicMock()
        mock_image_open.return_value = mock_image
        
        # Mock ImageDraw.Draw
        with patch('PIL.ImageDraw.Draw', return_value=mock_draw):
            # Call the function
            with patch('os.makedirs') as mock_makedirs:
                visualization_paths = visualize_annotations(
                    manifest_file="s3://test-bucket/output/manifest.json",
                    output_directory="/tmp/visualizations",
                    max_images=1
                )
        
        # Verify the function called the S3 client correctly
        mock_s3_client.get_object.assert_called_once()
        mock_s3_client.download_file.assert_called_once()
        
        # Verify the returned paths
        self.assertEqual(len(visualization_paths), 1)
        self.assertTrue(visualization_paths[0].startswith("/tmp/visualizations/annotated_"))
    
    @patch('boto3.Session')
    def test_get_labeling_job_metrics(self, mock_session):
        """Test getting detailed metrics for a labeling job."""
        # Set up mock clients
        mock_sagemaker_client = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.return_value.client.side_effect = lambda service: {
            'sagemaker': mock_sagemaker_client,
            's3': mock_s3_client
        }[service]
        
        # Create mock response for describe_labeling_job with datetime objects
        from datetime import datetime
        creation_time = datetime(2023, 1, 1, 0, 0, 0)
        last_modified_time = datetime(2023, 1, 1, 2, 0, 0)
        
        mock_sagemaker_client.describe_labeling_job.return_value = {
            "LabelingJobName": self.job_name,
            "LabelingJobStatus": "Completed",
            "LabelingJobArn": f"arn:aws:sagemaker:us-east-1:123456789012:labeling-job/{self.job_name}",
            "CreationTime": creation_time,
            "LastModifiedTime": last_modified_time,
            "LabelCounters": {
                "TotalObjects": 100,
                "LabeledObjects": 100,
                "FailedObjects": 0
            },
            "OutputConfig": {
                "S3OutputPath": "s3://test-bucket/output/"
            }
        }
        
        # Create mock response for get_object
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: "".encode())
        }
        
        # Call the function with a mock client
        metrics = get_labeling_job_metrics(self.job_name, mock_sagemaker_client)
        
        # Verify the function called the SageMaker client correctly
        mock_sagemaker_client.describe_labeling_job.assert_called_once_with(
            LabelingJobName=self.job_name
        )
        
        # Verify the returned metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn("basic_status", metrics)
        self.assertEqual(metrics["basic_status"]["LabelingJobName"], self.job_name)
        self.assertEqual(metrics["basic_status"]["LabelingJobStatus"], "Completed")
    
    def test_parse_s3_uri(self):
        """Test parsing S3 URIs."""
        # Test case 1: Standard URI
        uri = "s3://test-bucket/path/to/file.json"
        bucket, key = parse_s3_uri(uri)
        self.assertEqual(bucket, "test-bucket")
        self.assertEqual(key, "path/to/file.json")
        
        # Test case 2: Bucket only
        uri = "s3://test-bucket"
        bucket, key = parse_s3_uri(uri)
        self.assertEqual(bucket, "test-bucket")
        self.assertEqual(key, "")
        
        # Test case 3: Invalid URI
        with self.assertRaises(ValueError):
            parse_s3_uri("invalid-uri")


if __name__ == '__main__':
    unittest.main()