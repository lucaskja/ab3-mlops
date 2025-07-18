#!/usr/bin/env python3
"""
Unit tests for Script Templates

Tests the script template generation functionality.
"""

import unittest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.script_templates import ScriptTemplateManager


class TestScriptTemplateManager(unittest.TestCase):
    """Test ScriptTemplateManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.template_manager = ScriptTemplateManager()
    
    def test_template_loading(self):
        """Test that templates are loaded correctly"""
        self.assertIn("preprocessing", self.template_manager.templates)
        self.assertIn("evaluation", self.template_manager.templates)
    
    def test_generate_preprocessing_script(self):
        """Test generating a preprocessing script"""
        # Generate a script with custom logic
        script = self.template_manager.generate_preprocessing_script(
            preprocessing_logic="print('Custom preprocessing logic')",
            additional_args='parser.add_argument("--custom-arg", type=str, default="")',
            kwargs_extraction='"custom_arg": args.custom_arg'
        )
        
        # Verify the script contains the custom logic
        self.assertIn("print('Custom preprocessing logic')", script)
        self.assertIn('parser.add_argument("--custom-arg", type=str, default="")', script)
        self.assertIn('"custom_arg": args.custom_arg', script)
    
    def test_generate_evaluation_script(self):
        """Test generating an evaluation script"""
        # Generate a script with custom logic
        script = self.template_manager.generate_evaluation_script(
            evaluation_logic="print('Custom evaluation logic')",
            additional_args='parser.add_argument("--threshold", type=float, default=0.5)',
            kwargs_extraction='"threshold": args.threshold'
        )
        
        # Verify the script contains the custom logic
        self.assertIn("print('Custom evaluation logic')", script)
        self.assertIn('parser.add_argument("--threshold", type=float, default=0.5)', script)
        self.assertIn('"threshold": args.threshold', script)
    
    def test_default_values(self):
        """Test that default values are used when parameters are not provided"""
        # Generate a script without custom logic
        script = self.template_manager.generate_preprocessing_script()
        
        # Verify the script contains the default logic
        self.assertIn("# TODO: Implement preprocessing logic", script)


if __name__ == '__main__':
    unittest.main()