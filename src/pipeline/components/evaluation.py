"""
Evaluation component for SageMaker Pipelines.

This module provides components for evaluation steps in SageMaker Pipelines.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

from src.pipeline.components.base import StepComponent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationComponent(StepComponent):
    """Component for creating evaluation steps in SageMaker Pipelines."""
    
    def create(self, **kwargs) -> ScriptProcessor:
        """
        Create a script processor for evaluation.
        
        Args:
            **kwargs: Processor-specific parameters
            
        Returns:
            Configured ScriptProcessor
        """
        # Extract parameters
        image_uri = kwargs.get('image_uri')
        instance_type = kwargs.get('instance_type', 'ml.m5.xlarge')
        instance_count = kwargs.get('instance_count', 1)
        volume_size_in_gb = kwargs.get('volume_size_in_gb', 30)
        max_runtime_in_seconds = kwargs.get('max_runtime_in_seconds', 3600)
        environment = kwargs.get('environment', {})
        
        # Create script processor
        script_processor = ScriptProcessor(
            command=["python3"],
            image_uri=image_uri,
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=self.pipeline_session,
            env=environment
        )
        
        return script_processor
    
    def create_step(self, 
                   step_name: str, 
                   script_path: str,
                   model_path: str,
                   test_data: str,
                   output_path: Optional[str] = None,
                   **kwargs) -> ProcessingStep:
        """
        Create an evaluation step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            script_path: Path to the evaluation script
            model_path: S3 path to model artifacts
            test_data: S3 path to test data
            output_path: S3 path for evaluation results (optional)
            **kwargs: Additional parameters for the processor
            
        Returns:
            Configured ProcessingStep
        """
        logger.info(f"Creating evaluation step: {step_name}")
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"s3://{self.default_bucket}/pipeline/evaluation/{int(time.time())}"
        
        # Create script processor
        script_processor = self.create(**kwargs)
        
        # Extract additional parameters
        arguments = kwargs.get('arguments', [])
        
        # Create evaluation property file
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json"
        )
        
        # Create processing step
        evaluation_step = ProcessingStep(
            name=step_name,
            processor=script_processor,
            inputs=[
                ProcessingInput(
                    source=model_path,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=test_data,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=output_path
                )
            ],
            code=script_path,
            job_arguments=arguments,
            property_files=[evaluation_report]
        )
        
        logger.info(f"Evaluation step created: {step_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Test data: {test_data}")
        logger.info(f"Output path: {output_path}")
        
        return evaluation_step