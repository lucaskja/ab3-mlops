"""
Preprocessing component for SageMaker Pipelines.

This module provides components for data preprocessing steps in SageMaker Pipelines.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

from src.pipeline.components.base import StepComponent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingComponent(StepComponent):
    """Component for creating preprocessing steps in SageMaker Pipelines."""
    
    def create(self, **kwargs) -> ScriptProcessor:
        """
        Create a script processor for preprocessing.
        
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
                   input_data: str,
                   output_path: Optional[str] = None,
                   **kwargs) -> ProcessingStep:
        """
        Create a preprocessing step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            script_path: Path to the preprocessing script
            input_data: S3 path to input data
            output_path: S3 path for output data (optional)
            **kwargs: Additional parameters for the processor
            
        Returns:
            Configured ProcessingStep
        """
        logger.info(f"Creating preprocessing step: {step_name}")
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"s3://{self.default_bucket}/pipeline/preprocessing/{int(time.time())}"
        
        # Create script processor
        script_processor = self.create(**kwargs)
        
        # Extract additional parameters
        arguments = kwargs.get('arguments', [])
        
        # Create processing step
        processing_step = ProcessingStep(
            name=step_name,
            processor=script_processor,
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output/train",
                    destination=f"{output_path}/train"
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/output/validation",
                    destination=f"{output_path}/validation"
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output/test",
                    destination=f"{output_path}/test"
                )
            ],
            code=script_path,
            job_arguments=arguments
        )
        
        logger.info(f"Preprocessing step created: {step_name}")
        logger.info(f"Input data: {input_data}")
        logger.info(f"Output path: {output_path}")
        
        return processing_step