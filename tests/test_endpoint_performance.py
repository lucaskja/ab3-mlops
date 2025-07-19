#!/usr/bin/env python3
"""
Test script for SageMaker endpoint performance.
"""

import argparse
import json
import boto3
import numpy as np
import time
import logging
import unittest
from PIL import Image
import io
import os
import sys
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndpointPerformanceTest(unittest.TestCase):
    """Test class for SageMaker endpoint performance."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Test SageMaker endpoint performance')
        parser.add_argument('--endpoint-name', type=str, required=True, help='Name of the endpoint')
        parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
        parser.add_argument('--num-tests', type=int, default=10, help='Number of test invocations')
        parser.add_argument('--test-image-dir', type=str, default='tests/test_images', help='Directory with test images')
        
        # Parse args from command line or use defaults for running in IDE
        if 'pytest' in sys.modules:
            cls.args, _ = parser.parse_known_args()
        else:
            cls.args = parser.parse_args()
        
        # Set up AWS session
        cls.session = boto3.Session(profile_name=cls.args.profile)
        cls.runtime = cls.session.client('sagemaker-runtime')
        
        # Create test image directory if it doesn't exist
        os.makedirs(cls.args.test_image_dir, exist_ok=True)
        
        # Check if test images exist, if not create dummy images
        if not os.listdir(cls.args.test_image_dir):
            logger.info(f"Creating dummy test images in {cls.args.test_image_dir}")
            for i in range(5):
                img = Image.new('RGB', (640, 640), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                img.save(f"{cls.args.test_image_dir}/test_image_{i}.jpg")
    
    def test_endpoint_latency(self):
        """Test endpoint latency."""
        latencies = []
        
        # Get list of test images
        test_images = [os.path.join(self.args.test_image_dir, f) for f in os.listdir(self.args.test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # If no test images found, create a dummy image
        if not test_images:
            logger.warning("No test images found, creating a dummy image")
            dummy_image_path = f"{self.args.test_image_dir}/dummy_image.jpg"
            img = Image.new('RGB', (640, 640), color=(73, 109, 137))
            img.save(dummy_image_path)
            test_images = [dummy_image_path]
        
        # Run tests
        for _ in range(self.args.num_tests):
            # Select a random test image
            test_image_path = random.choice(test_images)
            
            # Load image
            with Image.open(test_image_path) as img:
                # Resize image to expected input size
                img = img.resize((640, 640))
                
                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()
            
            # Invoke endpoint
            start_time = time.time()
            response = self.runtime.invoke_endpoint(
                EndpointName=self.args.endpoint_name,
                ContentType='image/jpeg',
                Body=image_bytes
            )
            end_time = time.time()
            
            # Calculate latency
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            # Parse response
            response_body = response['Body'].read().decode('utf-8')
            result = json.loads(response_body)
            
            # Log results
            logger.info(f"Test {_ + 1}/{self.args.num_tests}: Latency = {latency:.2f} ms, Detections = {len(result.get('predictions', []))}")
            
            # Verify response format
            self.assertIn('predictions', result, "Response should contain 'predictions' key")
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Log statistics
        logger.info(f"Latency statistics (ms): Avg = {avg_latency:.2f}, P95 = {p95_latency:.2f}, P99 = {p99_latency:.2f}")
        
        # Assert performance requirements
        self.assertLess(p95_latency, 500, "P95 latency should be less than 500ms")
    
    def test_endpoint_throughput(self):
        """Test endpoint throughput."""
        # Create a dummy image
        img = Image.new('RGB', (640, 640), color=(73, 109, 137))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        
        # Run concurrent requests
        start_time = time.time()
        num_requests = 20
        
        for _ in range(num_requests):
            self.runtime.invoke_endpoint(
                EndpointName=self.args.endpoint_name,
                ContentType='image/jpeg',
                Body=image_bytes
            )
        
        end_time = time.time()
        
        # Calculate throughput
        duration = end_time - start_time
        throughput = num_requests / duration
        
        # Log results
        logger.info(f"Throughput: {throughput:.2f} requests/second ({num_requests} requests in {duration:.2f} seconds)")
        
        # Assert throughput requirements
        self.assertGreater(throughput, 1.0, "Throughput should be greater than 1 request/second")

if __name__ == '__main__':
    unittest.main()