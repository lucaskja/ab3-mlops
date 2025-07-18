"""
Base classes for SageMaker Pipeline components.

This module provides base classes and interfaces for pipeline components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineComponent(ABC):
    """Base class for all pipeline components."""
    
    def __init__(self, 
                 aws_profile: str = "ab", 
                 region: str = "us-east-1",
                 sagemaker_session: Optional[sagemaker.Session] = None,
                 pipeline_session: Optional[PipelineSession] = None,
                 execution_role: Optional[str] = None):
        """
        Initialize the pipeline component.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
            sagemaker_session: SageMaker session (optional)
            pipeline_session: Pipeline session (optional)
            execution_role: IAM execution role (optional)
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize sessions if not provided
        if sagemaker_session is None or pipeline_session is None:
            # Initialize AWS clients
            self.session = boto3.Session(profile_name=aws_profile)
            self.sagemaker_client = self.session.client('sagemaker', region_name=region)
            
            # Initialize SageMaker session if not provided
            if sagemaker_session is None:
                self.sagemaker_session = sagemaker.Session(
                    boto_session=self.session,
                    sagemaker_client=self.sagemaker_client
                )
            else:
                self.sagemaker_session = sagemaker_session
            
            # Initialize Pipeline session if not provided
            if pipeline_session is None:
                self.pipeline_session = PipelineSession(
                    boto_session=self.session,
                    sagemaker_client=self.sagemaker_client
                )
            else:
                self.pipeline_session = pipeline_session
        else:
            self.sagemaker_session = sagemaker_session
            self.pipeline_session = pipeline_session
        
        # Set execution role
        self.execution_role = execution_role
        
        # Set default S3 bucket
        self.default_bucket = self.sagemaker_session.default_bucket()
    
    @abstractmethod
    def create(self, **kwargs) -> Any:
        """
        Create the pipeline component.
        
        Args:
            **kwargs: Component-specific parameters
            
        Returns:
            Created component
        """
        pass


class StepComponent(PipelineComponent):
    """Base class for pipeline step components."""
    
    @abstractmethod
    def create_step(self, step_name: str, **kwargs) -> Any:
        """
        Create a pipeline step.
        
        Args:
            step_name: Name of the step
            **kwargs: Step-specific parameters
            
        Returns:
            Created pipeline step
        """
        pass


class SessionManager:
    """
    Manages AWS and SageMaker sessions for pipeline components.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the session manager.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS session
        self.aws_session = boto3.Session(profile_name=aws_profile)
        
        # Initialize AWS clients
        self.sagemaker_client = self.aws_session.client('sagemaker', region_name=region)
        self.s3_client = self.aws_session.client('s3', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.aws_session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Initialize Pipeline session
        self.pipeline_session = PipelineSession(
            boto_session=self.aws_session,
            sagemaker_client=self.sagemaker_client
        )
        
        logger.info(f"Session manager initialized for region: {region}")
    
    def get_aws_session(self) -> boto3.Session:
        """Get the AWS session."""
        return self.aws_session
    
    def get_sagemaker_session(self) -> sagemaker.Session:
        """Get the SageMaker session."""
        return self.sagemaker_session
    
    def get_pipeline_session(self) -> PipelineSession:
        """Get the Pipeline session."""
        return self.pipeline_session
    
    def get_client(self, service_name: str) -> Any:
        """
        Get an AWS client for the specified service.
        
        Args:
            service_name: Name of the AWS service
            
        Returns:
            AWS client
        """
        return self.aws_session.client(service_name, region_name=self.region)