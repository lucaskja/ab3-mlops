#!/usr/bin/env python3
"""
Unit tests for Data Profiler

Tests the data profiling functionality for drone imagery analysis.
"""

import unittest
import tempfile
import shutil
import os
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import io
from PIL import Image
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_profiler import DroneImageryProfiler


class TestDroneImageryProfiler(unittest.TestCase):
    """Test DroneImageryProfiler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock S3 access
        self.mock_s3_access = Mock()
        
        # Create profiler instance
        self.profiler = DroneImageryProfiler(self.mock_s3_access)
        
        # Create temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        self.test_images = []
        self.image_keys = []
        
        # Create 3 test images with different properties
        for i in range(3):
            # Create image with different dimensions
            width = 640 + i * 100
            height = 480 + i * 80
            
            # Create a test image
            img = Image.new('RGB', (width, height), color=(i * 50, 100, 150))
            
            # Save image to a bytes buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            self.test_images.append(img_buffer)
            self.image_keys.append(f"test_image_{i}.jpg")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        
        # Close image buffers
        for img_buffer in self.test_images:
            img_buffer.close()
    
    def test_profiler_initialization(self):
        """Test profiler initialization"""
        self.assertEqual(self.profiler.s3_access, self.mock_s3_access)
        self.assertEqual(self.profiler.profile_cache, {})
    
    def test_profile_images(self):
        """Test profiling a collection of images"""
        # Configure mock S3 access to return test images
        self.mock_s3_access.download_file_to_memory.side_effect = self.test_images
        
        # Profile images
        profile_data = self.profiler.profile_images(self.image_keys)
        
        # Verify S3 access was called for each image
        self.assertEqual(self.mock_s3_access.download_file_to_memory.call_count, len(self.image_keys))
        
        # Verify profile data contains expected keys
        expected_keys = [
            'widths', 'heights', 'file_sizes', 'color_modes', 'formats',
            'aspect_ratios', 'brightness_scores', 'contrast_scores',
            'sharpness_scores', 'color_diversity_scores', 'errors',
            'min_width', 'max_width', 'avg_width', 'median_width',
            'min_height', 'max_height', 'avg_height', 'median_height',
            'avg_file_size', 'min_file_size_mb', 'max_file_size_mb',
            'avg_aspect_ratio', 'aspect_ratio_std',
            'avg_brightness', 'min_brightness', 'max_brightness',
            'avg_contrast', 'min_contrast', 'max_contrast',
            'avg_sharpness', 'min_sharpness', 'max_sharpness',
            'avg_color_diversity', 'min_color_diversity', 'max_color_diversity',
            'color_modes', 'formats', 'total_images_processed',
            'processing_errors', 'success_rate'
        ]
        
        for key in expected_keys:
            self.assertIn(key, profile_data)
        
        # Verify image dimensions were correctly extracted
        self.assertEqual(len(profile_data['widths']), 3)
        self.assertEqual(len(profile_data['heights']), 3)
        self.assertEqual(profile_data['widths'][0], 640)
        self.assertEqual(profile_data['widths'][1], 740)
        self.assertEqual(profile_data['widths'][2], 840)
        self.assertEqual(profile_data['heights'][0], 480)
        self.assertEqual(profile_data['heights'][1], 560)
        self.assertEqual(profile_data['heights'][2], 640)
        
        # Verify summary statistics
        self.assertEqual(profile_data['min_width'], 640)
        self.assertEqual(profile_data['max_width'], 840)
        self.assertEqual(profile_data['avg_width'], 740)
        self.assertEqual(profile_data['min_height'], 480)
        self.assertEqual(profile_data['max_height'], 640)
        self.assertEqual(profile_data['avg_height'], 560)
        
        # Verify color modes
        self.assertEqual(profile_data['color_modes'], ['RGB'])
        
        # Verify success rate
        self.assertEqual(profile_data['total_images_processed'], 3)
        self.assertEqual(profile_data['processing_errors'], 0)
        self.assertEqual(profile_data['success_rate'], 1.0)
    
    def test_profile_images_with_sample_size(self):
        """Test profiling with sample size limit"""
        # Configure mock S3 access to return test images
        self.mock_s3_access.download_file_to_memory.side_effect = self.test_images
        
        # Profile images with sample size of 2
        profile_data = self.profiler.profile_images(self.image_keys, sample_size=2)
        
        # Verify S3 access was called only for the sample
        self.assertEqual(self.mock_s3_access.download_file_to_memory.call_count, 2)
        
        # Verify profile data contains expected number of images
        self.assertEqual(len(profile_data['widths']), 2)
        self.assertEqual(len(profile_data['heights']), 2)
        self.assertEqual(profile_data['total_images_processed'], 2)
    
    def test_profile_images_with_errors(self):
        """Test profiling with errors"""
        # Configure mock S3 access to raise an exception for one image
        def mock_download(key):
            if key == self.image_keys[1]:
                raise Exception("Test error")
            return self.test_images[self.image_keys.index(key)]
        
        self.mock_s3_access.download_file_to_memory.side_effect = mock_download
        
        # Profile images
        profile_data = self.profiler.profile_images(self.image_keys)
        
        # Verify S3 access was called for each image
        self.assertEqual(self.mock_s3_access.download_file_to_memory.call_count, 3)
        
        # Verify error was recorded
        self.assertEqual(len(profile_data['errors']), 1)
        self.assertTrue("Test error" in profile_data['errors'][0])
        
        # Verify success rate
        self.assertEqual(profile_data['total_images_processed'], 2)
        self.assertEqual(profile_data['processing_errors'], 1)
        self.assertAlmostEqual(profile_data['success_rate'], 2/3)
    
    def test_analyze_image_quality(self):
        """Test analyzing image quality"""
        # Create a test image
        img = Image.new('RGB', (100, 100), color=(100, 150, 200))
        
        # Analyze image quality
        quality_metrics = self.profiler._analyze_image_quality(img)
        
        # Verify quality metrics
        self.assertIn('brightness', quality_metrics)
        self.assertIn('contrast', quality_metrics)
        self.assertIn('sharpness', quality_metrics)
        self.assertIn('color_diversity', quality_metrics)
        
        # Verify brightness calculation (100+150+200)/(3*255) = 0.5882
        expected_brightness = (100 + 150 + 200) / (3 * 255)
        self.assertAlmostEqual(quality_metrics['brightness'], expected_brightness, places=4)
        
        # Verify contrast is 0 for a solid color image
        self.assertEqual(quality_metrics['contrast'], 0.0)
    
    def test_analyze_image_quality_non_rgb(self):
        """Test analyzing image quality for non-RGB images"""
        # Create a grayscale test image
        img = Image.new('L', (100, 100), color=128)
        
        # Analyze image quality
        quality_metrics = self.profiler._analyze_image_quality(img)
        
        # Verify quality metrics
        self.assertIn('brightness', quality_metrics)
        self.assertIn('contrast', quality_metrics)
        self.assertIn('sharpness', quality_metrics)
        self.assertIn('color_diversity', quality_metrics)
        
        # Verify brightness calculation 128/255 = 0.502
        expected_brightness = 128 / 255
        self.assertAlmostEqual(quality_metrics['brightness'], expected_brightness, places=4)
    
    def test_estimate_sharpness(self):
        """Test estimating image sharpness"""
        # Create a test image with some edges
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        # Draw a black rectangle to create edges
        for x in range(25, 75):
            for y in range(25, 75):
                img.putpixel((x, y), (0, 0, 0))
        
        # Estimate sharpness
        sharpness = self.profiler._estimate_sharpness(img)
        
        # Verify sharpness is a float between 0 and 1
        self.assertIsInstance(sharpness, float)
        self.assertGreaterEqual(sharpness, 0.0)
        self.assertLessEqual(sharpness, 1.0)
        
        # Sharpness should be higher for an image with clear edges
        self.assertGreater(sharpness, 0.01)
    
    def test_calculate_color_diversity(self):
        """Test calculating color diversity"""
        # Create a test image with multiple colors
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        # Add different colored regions
        for x in range(50):
            for y in range(50):
                img.putpixel((x, y), (255, 0, 0))  # Red
        for x in range(50, 100):
            for y in range(50):
                img.putpixel((x, y), (0, 255, 0))  # Green
        for x in range(50):
            for y in range(50, 100):
                img.putpixel((x, y), (0, 0, 255))  # Blue
        
        # Calculate color diversity
        diversity = self.profiler._calculate_color_diversity(img)
        
        # Verify diversity is a float between 0 and 1
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
        
        # Diversity should be higher for an image with multiple colors
        self.assertGreater(diversity, 0.1)
    
    def test_generate_profile_report(self):
        """Test generating a profile report"""
        # Create mock profile data
        profile_data = {
            'total_images_processed': 100,
            'processing_errors': 2,
            'success_rate': 0.98,
            'min_width': 640,
            'max_width': 1920,
            'avg_width': 1280,
            'median_width': 1280,
            'min_height': 480,
            'max_height': 1080,
            'avg_height': 720,
            'median_height': 720,
            'avg_aspect_ratio': 1.78,
            'avg_file_size': 2.5,
            'min_file_size_mb': 1.2,
            'max_file_size_mb': 5.0,
            'color_modes': ['RGB', 'RGBA'],
            'formats': ['JPEG', 'PNG'],
            'avg_brightness': 0.6,
            'min_brightness': 0.3,
            'max_brightness': 0.9,
            'avg_contrast': 0.4,
            'min_contrast': 0.2,
            'max_contrast': 0.7,
            'avg_sharpness': 0.5,
            'min_sharpness': 0.2,
            'max_sharpness': 0.8,
            'avg_color_diversity': 0.7,
            'min_color_diversity': 0.4,
            'max_color_diversity': 0.9
        }
        
        # Generate report
        report = self.profiler.generate_profile_report(profile_data)
        
        # Verify report is a string
        self.assertIsInstance(report, str)
        
        # Verify report contains key information
        self.assertIn("DRONE IMAGERY DATASET PROFILE REPORT", report)
        self.assertIn("Total images processed: 100", report)
        self.assertIn("Processing errors: 2", report)
        self.assertIn("Success rate: 98.0%", report)
        self.assertIn("Resolution range: 640x480 to 1920x1080", report)
        self.assertIn("Average resolution: 1280x720", report)
        self.assertIn("Average aspect ratio: 1.78", report)
        self.assertIn("Average file size: 2.5 MB", report)
        self.assertIn("Color modes: RGB, RGBA", report)
        self.assertIn("File formats: JPEG, PNG", report)
        self.assertIn("Brightness: 0.600", report)
        self.assertIn("Contrast: 0.400", report)
        self.assertIn("Sharpness: 0.500", report)
        self.assertIn("Color Diversity: 0.700", report)
    
    def test_get_recommendations(self):
        """Test getting recommendations based on profile data"""
        # Test case 1: Low resolution images
        profile_data_1 = {
            'avg_width': 320,
            'avg_height': 240,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.5,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['JPEG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_1 = self.profiler.get_recommendations(profile_data_1)
        self.assertIsInstance(recommendations_1, list)
        self.assertIn("Consider upscaling images to at least 640x640 for optimal YOLOv11 performance", recommendations_1)
        
        # Test case 2: High aspect ratio variation
        profile_data_2 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.6,
            'avg_brightness': 0.5,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['JPEG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_2 = self.profiler.get_recommendations(profile_data_2)
        self.assertIn("High aspect ratio variation detected - consider standardizing image dimensions", recommendations_2)
        
        # Test case 3: Dark images
        profile_data_3 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.2,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['JPEG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_3 = self.profiler.get_recommendations(profile_data_3)
        self.assertIn("Images appear dark - consider brightness enhancement preprocessing", recommendations_3)
        
        # Test case 4: Overexposed images
        profile_data_4 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.9,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['JPEG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_4 = self.profiler.get_recommendations(profile_data_4)
        self.assertIn("Images appear overexposed - consider brightness normalization", recommendations_4)
        
        # Test case 5: Low contrast images
        profile_data_5 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.5,
            'avg_contrast': 0.1,
            'avg_sharpness': 0.6,
            'formats': ['JPEG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_5 = self.profiler.get_recommendations(profile_data_5)
        self.assertIn("Low contrast detected - consider contrast enhancement techniques", recommendations_5)
        
        # Test case 6: Blurry images
        profile_data_6 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.5,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.2,
            'formats': ['JPEG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_6 = self.profiler.get_recommendations(profile_data_6)
        self.assertIn("Images may be blurry - consider sharpening filters or quality filtering", recommendations_6)
        
        # Test case 7: Unsupported formats
        profile_data_7 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.5,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['BMP', 'TIFF'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_7 = self.profiler.get_recommendations(profile_data_7)
        self.assertIn("Consider converting images to JPEG or PNG format for better compatibility", recommendations_7)
        
        # Test case 8: High error rate
        profile_data_8 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.5,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['JPEG'],
            'processing_errors': 15,
            'total_images_processed': 100
        }
        
        recommendations_8 = self.profiler.get_recommendations(profile_data_8)
        self.assertIn("High error rate detected - review data quality and file integrity", recommendations_8)
        
        # Test case 9: Good quality dataset
        profile_data_9 = {
            'avg_width': 800,
            'avg_height': 600,
            'aspect_ratio_std': 0.2,
            'avg_brightness': 0.5,
            'avg_contrast': 0.4,
            'avg_sharpness': 0.6,
            'formats': ['JPEG', 'PNG'],
            'processing_errors': 1,
            'total_images_processed': 100
        }
        
        recommendations_9 = self.profiler.get_recommendations(profile_data_9)
        self.assertIn("Dataset appears to be in good condition for ML training", recommendations_9)


if __name__ == '__main__':
    unittest.main()