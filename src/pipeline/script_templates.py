"""
Script templates for SageMaker Pipeline components.

This module provides utilities for generating scripts used in SageMaker Pipeline steps
by using template files from the templates directory.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from string import Template
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScriptTemplateManager:
    """Manager for script templates used in SageMaker Pipeline steps."""
    
    def __init__(self):
        """Initialize the script template manager."""
        # Initialize template registry
        self.templates = {}
        
        # Load templates from files
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from files in the templates directory."""
        # Get the templates directory path
        templates_dir = Path(__file__).parent / "templates"
        
        # Load preprocessing template
        preprocessing_path = templates_dir / "preprocessing_template.txt"
        if preprocessing_path.exists():
            with open(preprocessing_path, "r") as f:
                self.register_template("preprocessing", f.read())
        else:
            logger.warning(f"Template file not found: {preprocessing_path}")
        
        # Load evaluation template
        evaluation_path = templates_dir / "evaluation_template.txt"
        if evaluation_path.exists():
            with open(evaluation_path, "r") as f:
                self.register_template("evaluation", f.read())
        else:
            logger.warning(f"Template file not found: {evaluation_path}")
    
    def register_template(self, name: str, template_str: str):
        """
        Register a script template.
        
        Args:
            name: Template name
            template_str: Template string
        """
        self.templates[name] = Template(template_str)
        logger.info(f"Registered template: {name}")
    
    def get_template(self, name: str) -> Optional[Template]:
        """
        Get a script template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template object or None if not found
        """
        return self.templates.get(name)
    
    def generate_script(self, template_name: str, **kwargs) -> str:
        """
        Generate a script from a template.
        
        Args:
            template_name: Template name
            **kwargs: Template parameters
            
        Returns:
            Generated script
        """
        template = self.get_template(template_name)
        if template is None:
            raise ValueError(f"Template not found: {template_name}")
        
        # Set default values for missing parameters
        if "preprocessing_logic" not in kwargs:
            kwargs["preprocessing_logic"] = "# TODO: Implement preprocessing logic"
        if "evaluation_logic" not in kwargs:
            kwargs["evaluation_logic"] = "# TODO: Implement evaluation logic"
        if "additional_args" not in kwargs:
            kwargs["additional_args"] = ""
        if "kwargs_extraction" not in kwargs:
            kwargs["kwargs_extraction"] = ""
        
        return template.substitute(**kwargs)
    
    def generate_preprocessing_script(self, 
                                     preprocessing_logic: str = "",
                                     additional_args: str = "",
                                     kwargs_extraction: str = "") -> str:
        """
        Generate a preprocessing script.
        
        Args:
            preprocessing_logic: Custom preprocessing logic
            additional_args: Additional command-line arguments
            kwargs_extraction: Code for extracting additional arguments
            
        Returns:
            Generated preprocessing script
        """
        return self.generate_script(
            "preprocessing",
            preprocessing_logic=preprocessing_logic or "# TODO: Implement preprocessing logic",
            additional_args=additional_args,
            kwargs_extraction=kwargs_extraction
        )
    
    def generate_evaluation_script(self, 
                                  evaluation_logic: str = "",
                                  additional_args: str = "",
                                  kwargs_extraction: str = "") -> str:
        """
        Generate an evaluation script.
        
        Args:
            evaluation_logic: Custom evaluation logic
            additional_args: Additional command-line arguments
            kwargs_extraction: Code for extracting additional arguments
            
        Returns:
            Generated evaluation script
        """
        return self.generate_script(
            "evaluation",
            evaluation_logic=evaluation_logic or "# TODO: Implement evaluation logic",
            additional_args=additional_args,
            kwargs_extraction=kwargs_extraction
        )


# Singleton instance
script_template_manager = ScriptTemplateManager()