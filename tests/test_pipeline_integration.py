#!/usr/bin/env python3
"""
Integration tests for SageMaker Pipeline

Tests the end-to-end pipeline functionality with sample data.
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, ANY

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.sagemaker_pipeline import SageMakerPipeline
from src.pipeline.sagemaker_pipeline_factory import SageMakerPipelineFactory
from src.pipeline.components.preprocessing import PreprocessingStep
from src.pipeline.components.training import TrainingStep
from src.pipeline.components.evaluation import EvaluationStep


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for SageMaker Pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temp directory for test artifacts
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock AWS session and clients
        self.session_patch = patch('boto3.Session')
        self.mock_session = self.session_patch.start()
        
        # Mock SageMaker client
        self.mock_sagemaker_client = MagicMock()
        self.mock_session.return_value.client.return_value = self.mock_sagemaker_client
        
        # Mock SageMaker session
        self.sagemaker_session_patch = patch('sagemaker.Session')
        self.mock_sagemaker_session = self.sagemaker_session_patch.start()
        
        # Mock pipeline execution
        self.mock_sagemaker_client.create_pipeline.return_value = {
            "PipelineArn": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline"
        }
        self.mock_sagemaker_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
        }
        self.mock_sagemaker_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Succeeded"
        }
        
        # Create sample data
        self.sample_data_path = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(self.sample_data_path, exist_ok=True)
        
        # Create sample config
        self.config = {
            "pipeline": {
                "name": "test-pipeline",
                "role": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
                "default_bucket": "test-bucket",
                "tags": [{"Key": "Project", "Value": "MLOpsSageMakerDemo"}]
            },
            "preprocessing": {
                "instance_type": "ml.m5.xlarge",
                "instance_count": 1,
                "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:latest"
            },
            "training": {
                "instance_type": "ml.g4dn.xlarge",
                "instance_count": 1,
                "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/yolov11-training:latest",
                "hyperparameters": {
                    "epochs": "100",
                    "batch-size": "16",
                    "learning-rate": "0.01"
                }
            },
            "evaluation": {
                "instance_type": "ml.m5.xlarge",
                "instance_count": 1,
                "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/yolov11-evaluation:latest"
            },
            "model": {
                "approval_status": "PendingManualApproval",
                "registry_name": "yolov11-models"
            },
            "endpoint": {
                "instance_type": "ml.g4dn.xlarge",
                "instance_count": 1,
                "auto_scaling": {
                    "min_capacity": 1,
                    "max_capacity": 4,
                    "target_utilization": 75
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
        # Stop patches
        self.session_patch.stop()
        self.sagemaker_session_patch.stop()
    
    def test_end_to_end_pipeline_execution(self):
        """Test end-to-end pipeline execution with sample data"""
        # Create pipeline factory
        factory = SageMakerPipelineFactory(
            aws_profile="ab",
            region="us-east-1",
            config=self.config
        )
        
        # Create pipeline
        pipeline = factory.create_pipeline()
        
        # Verify pipeline was created
        self.assertIsInstance(pipeline, SageMakerPipeline)
        
        # Execute pipeline
        execution_arn = pipeline.execute(
            parameters={
                "InputDataPath": "s3://test-bucket/input-data",
                "ModelName": "yolov11-model",
                "BatchSize": "32"
            }
        )
        
        # Verify pipeline execution
        self.assertEqual(
            execution_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
        )
        
        # Verify create_pipeline was called
        self.mock_sagemaker_client.create_pipeline.assert_called_once()
        
        # Verify start_pipeline_execution was called
        self.mock_sagemaker_client.start_pipeline_execution.assert_called_once()
        
        # Verify parameters were passed correctly
        call_args = self.mock_sagemaker_client.start_pipeline_execution.call_args[1]
        self.assertEqual(call_args["PipelineName"], "test-pipeline")
        
        # Convert parameters to dict for easier verification
        params = {p["Name"]: p["Value"] for p in call_args["PipelineParameters"]}
        self.assertEqual(params["InputDataPath"], "s3://test-bucket/input-data")
        self.assertEqual(params["ModelName"], "yolov11-model")
        self.assertEqual(params["BatchSize"], "32")
    
    def test_pipeline_component_integration(self):
        """Test integration between pipeline components"""
        # Create pipeline factory
        factory = SageMakerPipelineFactory(
            aws_profile="ab",
            region="us-east-1",
            config=self.config
        )
        
        # Create preprocessing step
        preprocessing_step = factory.create_preprocessing_step(
            input_data="s3://test-bucket/input-data"
        )
        
        # Verify preprocessing step
        self.assertIsInstance(preprocessing_step, PreprocessingStep)
        
        # Create training step
        training_step = factory.create_training_step(
            preprocessed_data=preprocessing_step.get_output()
        )
        
        # Verify training step
        self.assertIsInstance(training_step, TrainingStep)
        
        # Create evaluation step
        evaluation_step = factory.create_evaluation_step(
            model=training_step.get_model(),
            test_data=preprocessing_step.get_test_data_output()
        )
        
        # Verify evaluation step
        self.assertIsInstance(evaluation_step, EvaluationStep)
        
        # Verify step dependencies
        self.assertEqual(training_step.get_dependencies()[0], preprocessing_step.get_step())
        self.assertEqual(evaluation_step.get_dependencies()[0], training_step.get_step())
    
    def test_pipeline_with_custom_components(self):
        """Test pipeline with custom components"""
        # Create pipeline factory
        factory = SageMakerPipelineFactory(
            aws_profile="ab",
            region="us-east-1",
            config=self.config
        )
        
        # Create pipeline with custom components
        pipeline = factory.create_custom_pipeline(
            steps=[
                {
                    "type": "preprocessing",
                    "input_data": "s3://test-bucket/input-data"
                },
                {
                    "type": "training",
                    "hyperparameters": {
                        "epochs": "200",
                        "batch-size": "32"
                    }
                },
                {
                    "type": "evaluation",
                    "threshold": 0.75
                }
            ]
        )
        
        # Verify pipeline was created
        self.assertIsInstance(pipeline, SageMakerPipeline)
        
        # Execute pipeline
        execution_arn = pipeline.execute()
        
        # Verify pipeline execution
        self.assertEqual(
            execution_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
        )
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        # Mock pipeline execution failure
        self.mock_sagemaker_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Failed",
            "FailureReason": "Step failed: PreprocessingStep"
        }
        
        # Create pipeline factory
        factory = SageMakerPipelineFactory(
            aws_profile="ab",
            region="us-east-1",
            config=self.config
        )
        
        # Create pipeline
        pipeline = factory.create_pipeline()
        
        # Execute pipeline
        execution_arn = pipeline.execute()
        
        # Verify pipeline execution
        self.assertEqual(
            execution_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
        )
        
        # Verify pipeline status
        status = pipeline.get_execution_status(execution_arn)
        self.assertEqual(status["status"], "Failed")
        self.assertEqual(status["reason"], "Step failed: PreprocessingStep")
    
    def test_pipeline_with_mlflow_integration(self):
        """Test pipeline with MLFlow integration"""
        # Mock MLFlow integration
        mlflow_patch = patch('src.pipeline.mlflow_integration.MLFlowSageMakerIntegration')
        mock_mlflow = mlflow_patch.start()
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        try:
            # Create pipeline factory with MLFlow integration
            factory = SageMakerPipelineFactory(
                aws_profile="ab",
                region="us-east-1",
                config=self.config,
                enable_mlflow=True
            )
            
            # Create pipeline
            pipeline = factory.create_pipeline()
            
            # Execute pipeline
            execution_arn = pipeline.execute()
            
            # Verify MLFlow integration
            mock_mlflow.assert_called_once()
            mock_mlflow_instance.log_pipeline_execution.assert_called_once()
            
            # Verify pipeline execution
            self.assertEqual(
                execution_arn,
                "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
            )
        finally:
            # Stop patch
            mlflow_patch.stop()


if __name__ == '__main__':
    unittest.main()