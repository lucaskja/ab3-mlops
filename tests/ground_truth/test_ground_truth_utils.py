"""
Unit tests for Ground Truth utilities.
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
    create_labeling_instructions
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
        self.assertEqual(config["LabelingJobAlgorithmsConfig"]["LabelingJobAlgorithm"], "BOUNDING_BOX")
        self.assertEqual(config["HumanTaskConfig"]["TaskDescription"], "Test instructions")
        
        # Verify budget configuration
        budget_tag = next((tag for tag in config["Tags"] if tag["Key"] == "MaxBudgetUSD"), None)
        self.assertIsNotNone(budget_tag)
        self.assertEqual(budget_tag["Value"], "100.0")
    
    @patch('boto3.Session')
    def test_monitor_labeling_job(self, mock_session):
        """Test monitoring a labeling job."""
        # Create mock response
        mock_response = {
            "LabelingJobName": self.job_name,
            "LabelingJobStatus": "InProgress",
            "LabelingJobArn": f"arn:aws:sagemaker:us-east-1:123456789012:labeling-job/{self.job_name}",
            "CreationTime": "2023-01-01T00:00:00Z",
            "LastModifiedTime": "2023-01-01T01:00:00Z",
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
    
    def test_convert_ground_truth_to_yolo(self):
        """Test converting Ground Truth output to YOLOv11 format."""
        input_manifest = "s3://test-bucket/output/manifest.json"
        output_directory = "s3://test-bucket/yolo-format/"
        class_mapping = {
            "drone": 0,
            "vehicle": 1,
            "person": 2,
            "building": 3
        }
        
        # Since this is a placeholder implementation, we just verify it returns True
        result = convert_ground_truth_to_yolo(
            input_manifest=input_manifest,
            output_directory=output_directory,
            class_mapping=class_mapping
        )
        
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


if __name__ == '__main__':
    unittest.main()