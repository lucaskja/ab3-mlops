#!/usr/bin/env python3
"""
Lambda Function Utilities

This module provides utilities for working with Lambda functions,
including loading Lambda code from files and replacing placeholders.
"""

import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def load_lambda_code(lambda_type: str) -> str:
    """
    Load Lambda function code from a file.
    
    Args:
        lambda_type: Type of Lambda function (drift_detection, scheduled_evaluation, model_approval)
        
    Returns:
        Lambda function code as a string
    """
    # Get the path to the Lambda function code
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lambda_path = os.path.join(base_dir, lambda_type, "lambda_function.py")
    
    # Check if the file exists
    if not os.path.exists(lambda_path):
        raise FileNotFoundError(f"Lambda function code not found at: {lambda_path}")
    
    # Read the Lambda function code
    with open(lambda_path, "r") as f:
        lambda_code = f.read()
    
    logger.info(f"Loaded Lambda function code from: {lambda_path}")
    return lambda_code

def replace_placeholders(code: str, replacements: Dict[str, Any]) -> str:
    """
    Replace placeholders in Lambda code with actual values.
    
    Args:
        code: Lambda function code
        replacements: Dictionary of placeholder replacements
        
    Returns:
        Lambda function code with placeholders replaced
    """
    result = code
    
    for placeholder, value in replacements.items():
        # Handle different types of values
        if isinstance(value, str):
            # For string values, wrap in quotes
            result = result.replace(f'"{placeholder}"', f'"{value}"')
            result = result.replace(f"'{placeholder}'", f"'{value}'")
            # Also replace without quotes for cases where the placeholder is used in a string
            result = result.replace(placeholder, value)
        elif isinstance(value, (int, float, bool)):
            # For numeric and boolean values, don't wrap in quotes
            result = result.replace(f"{placeholder}", str(value))
        elif value is None:
            # For None values, use "None"
            result = result.replace(f"{placeholder}", "None")
    
    return result