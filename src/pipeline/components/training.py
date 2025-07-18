"""
Training component for SageMaker Pipelines.

This module provides components for training steps in SageMaker Pipelines.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union

from sagemaker.pytorch import PyTorch
from sagemaker.workflow.steps import TrainingStep

from src.pipeline.components.base import StepComponent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingComponent(StepComponent):
    """Component for creating training steps in SageMaker Pipelines."""
    
    def create(self, 
              script_path: str,
              **kwargs) -> PyTorch:
        """
        Create a PyTorch estimator for training.
        
        Args:
            script_path: Path to the training script
            **kwargs: Estimator-specific parameters
            
        Returns:
            Configured PyTorch estimator
        """
        # Extract parameters
        instance_type = kwargs.get('instance_type', 'ml.p3.2xlarge')
        instance_count = kwargs.get('instance_count', 1)
        volume_size = kwargs.get('volume_size', 50)
        max_run = kwargs.get('max_run', 86400)  # 24 hours
        framework_version = kwargs.get('framework_version', '1.10.0')
        py_version = kwargs.get('py_version', 'py38')
        hyperparameters = kwargs.get('hyperparameters', {})
        environment = kwargs.get('environment', {})
        dependencies = kwargs.get('dependencies', [])
        distribution = kwargs.get('distribution')
        
        # Set default hyperparameters if not provided
        if not hyperparameters:
            hyperparameters = {
                "epochs": 10,
                "batch-size": 32,
                "learning-rate": 0.001,
                "optimizer": "adam"
            }
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point=os.path.basename(script_path),
            source_dir=os.path.dirname(script_path),
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            max_run=max_run,
            framework_version=framework_version,
            py_version=py_version,
            hyperparameters=hyperparameters,
            environment=environment,
            sagemaker_session=self.pipeline_session,
            dependencies=dependencies,
            distribution=distribution
        )
        
        return estimator
    
    def create_step(self, 
                   step_name: str, 
                   script_path: str,
                   input_train: str,
                   input_validation: str,
                   **kwargs) -> TrainingStep:
        """
        Create a training step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            script_path: Path to the training script
            input_train: S3 path to training data
            input_validation: S3 path to validation data
            **kwargs: Additional parameters for the estimator
            
        Returns:
            Configured TrainingStep
        """
        logger.info(f"Creating training step: {step_name}")
        
        # Create PyTorch estimator
        estimator = self.create(script_path=script_path, **kwargs)
        
        # Create training step
        training_step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs={
                "train": input_train,
                "validation": input_validation
            }
        )
        
        logger.info(f"Training step created: {step_name}")
        logger.info(f"Training data: {input_train}")
        logger.info(f"Validation data: {input_validation}")
        
        return training_step