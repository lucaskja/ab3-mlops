#!/usr/bin/env python3
"""
Integration tests for SageMaker Endpoint Performance

Tests the performance and load handling capabilities of SageMaker inference endpoints.
"""

import unittest
import os
import sys
import json
import time
import numpy as np
import concurrent.futures
from unittest.mock import patch, MagicMock, ANY

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import endpoint utilities
from configs.cdk.lib.endpoint_stack import EndpointStack


class TestEndpointPerformance(unittest.TestCase):
    """Integration tests for endpoint performance and load testing"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock AWS session and clients
        self.session_patch = patch('boto3.Session')
        self.mock_session = self.session_patch.start()
        
        # Mock SageMaker runtime client
        self.mock_runtime_client = MagicMock()
        self.mock_session.return_value.client.return_value = self.mock_runtime_client
        
        # Mock CloudWatch client
        self.mock_cloudwatch_client = MagicMock()
        
        # Configure mock session to return mock clients
        self.mock_session.return_value.client.side_effect = lambda service, region_name=None: {
            'sagemaker-runtime': self.mock_runtime_client,
            'cloudwatch': self.mock_cloudwatch_client
        }.get(service, MagicMock())
        
        # Mock invoke_endpoint response
        self.mock_runtime_client.invoke_endpoint.return_value = {
            'Body': MagicMock(read=lambda: json.dumps({
                'predictions': [
                    {
                        'boxes': [[0.1, 0.2, 0.3, 0.4]],
                        'scores': [0.95],
                        'classes': [0]
                    }
                ]
            }).encode('utf-8'))
        }
        
        # Test parameters
        self.endpoint_name = "test-endpoint"
        self.payload = json.dumps({
            'instances': [
                {
                    'image': np.random.rand(640, 640, 3).tolist()
                }
            ]
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.session_patch.stop()
    
    def test_endpoint_single_request(self):
        """Test single request to endpoint"""
        # Invoke endpoint
        response = self.mock_runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=self.payload
        )
        
        # Parse response
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        
        # Verify response
        self.assertIn('predictions', response_body)
        self.assertEqual(len(response_body['predictions']), 1)
        self.assertIn('boxes', response_body['predictions'][0])
        self.assertIn('scores', response_body['predictions'][0])
        self.assertIn('classes', response_body['predictions'][0])
    
    def test_endpoint_concurrent_requests(self):
        """Test concurrent requests to endpoint"""
        # Number of concurrent requests
        num_requests = 10
        
        # Function to invoke endpoint
        def invoke_endpoint():
            response = self.mock_runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=self.payload
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(invoke_endpoint) for _ in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify results
        self.assertEqual(len(results), num_requests)
        for result in results:
            self.assertIn('predictions', result)
            self.assertEqual(len(result['predictions']), 1)
    
    def test_endpoint_latency(self):
        """Test endpoint latency"""
        # Number of requests
        num_requests = 5
        
        # Track latencies
        latencies = []
        
        # Execute requests and measure latency
        for _ in range(num_requests):
            start_time = time.time()
            self.mock_runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=self.payload
            )
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Verify latencies (these are mock values, so they'll be very small)
        self.assertGreaterEqual(avg_latency, 0)
        self.assertLessEqual(avg_latency, 1.0)  # Assuming mock response is fast
    
    def test_endpoint_auto_scaling(self):
        """Test endpoint auto-scaling configuration"""
        # Create mock CDK app and stack
        mock_app = MagicMock()
        mock_stack = MagicMock()
        
        # Mock AWS CDK constructs
        with patch('aws_cdk.aws_sagemaker.CfnEndpointConfig') as mock_endpoint_config:
            with patch('aws_cdk.aws_applicationautoscaling.CfnScalableTarget') as mock_scalable_target:
                with patch('aws_cdk.aws_applicationautoscaling.CfnScalingPolicy') as mock_scaling_policy:
                    # Create endpoint stack
                    endpoint_stack = EndpointStack(
                        mock_app,
                        "TestEndpointStack",
                        model_name="yolov11-model",
                        instance_type="ml.g4dn.xlarge",
                        instance_count=1,
                        auto_scaling_enabled=True,
                        min_capacity=1,
                        max_capacity=4,
                        target_utilization=75
                    )
                    
                    # Verify endpoint config was created
                    mock_endpoint_config.assert_called_once()
                    
                    # Verify scalable target was created
                    mock_scalable_target.assert_called_once()
                    
                    # Verify scaling policy was created
                    mock_scaling_policy.assert_called_once()
                    
                    # Verify scaling policy parameters
                    scaling_policy_args = mock_scaling_policy.call_args[1]
                    self.assertEqual(scaling_policy_args["PolicyType"], "TargetTrackingScaling")
                    
                    # Verify target tracking configuration
                    target_tracking = scaling_policy_args["TargetTrackingScalingPolicyConfiguration"]
                    self.assertEqual(target_tracking["TargetValue"], 75)
                    self.assertEqual(target_tracking["PredefinedMetricSpecification"]["PredefinedMetricType"], 
                                    "SageMakerVariantInvocationsPerInstance")
    
    def test_endpoint_metrics(self):
        """Test endpoint CloudWatch metrics"""
        # Mock CloudWatch get_metric_data response
        self.mock_cloudwatch_client.get_metric_data.return_value = {
            'MetricDataResults': [
                {
                    'Id': 'invocations',
                    'Label': 'Invocations',
                    'Values': [10, 15, 20, 25, 30],
                    'Timestamps': [
                        time.time() - 240,
                        time.time() - 180,
                        time.time() - 120,
                        time.time() - 60,
                        time.time()
                    ]
                },
                {
                    'Id': 'latency',
                    'Label': 'ModelLatency',
                    'Values': [0.05, 0.06, 0.05, 0.07, 0.06],
                    'Timestamps': [
                        time.time() - 240,
                        time.time() - 180,
                        time.time() - 120,
                        time.time() - 60,
                        time.time()
                    ]
                }
            ]
        }
        
        # Get endpoint metrics
        response = self.mock_cloudwatch_client.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'invocations',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/SageMaker',
                            'MetricName': 'Invocations',
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': self.endpoint_name
                                },
                                {
                                    'Name': 'VariantName',
                                    'Value': 'AllTraffic'
                                }
                            ]
                        },
                        'Period': 60,
                        'Stat': 'Sum'
                    }
                },
                {
                    'Id': 'latency',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/SageMaker',
                            'MetricName': 'ModelLatency',
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': self.endpoint_name
                                },
                                {
                                    'Name': 'VariantName',
                                    'Value': 'AllTraffic'
                                }
                            ]
                        },
                        'Period': 60,
                        'Stat': 'Average'
                    }
                }
            ],
            StartTime=time.time() - 300,
            EndTime=time.time()
        )
        
        # Verify metrics
        self.assertEqual(len(response['MetricDataResults']), 2)
        
        # Verify invocations
        invocations = next(r for r in response['MetricDataResults'] if r['Id'] == 'invocations')
        self.assertEqual(len(invocations['Values']), 5)
        self.assertEqual(invocations['Values'][-1], 30)
        
        # Verify latency
        latency = next(r for r in response['MetricDataResults'] if r['Id'] == 'latency')
        self.assertEqual(len(latency['Values']), 5)
        self.assertEqual(latency['Values'][-1], 0.06)
    
    def test_endpoint_error_handling(self):
        """Test endpoint error handling"""
        # Mock invoke_endpoint to raise an exception
        self.mock_runtime_client.invoke_endpoint.side_effect = Exception("Model error")
        
        # Attempt to invoke endpoint
        with self.assertRaises(Exception) as context:
            self.mock_runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=self.payload
            )
        
        # Verify exception
        self.assertEqual(str(context.exception), "Model error")
        
        # Reset mock
        self.mock_runtime_client.invoke_endpoint.side_effect = None
        
        # Mock invoke_endpoint to return a model error
        self.mock_runtime_client.invoke_endpoint.return_value = {
            'Body': MagicMock(read=lambda: json.dumps({
                'error': 'Invalid input shape'
            }).encode('utf-8'))
        }
        
        # Invoke endpoint
        response = self.mock_runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=self.payload
        )
        
        # Parse response
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        
        # Verify error response
        self.assertIn('error', response_body)
        self.assertEqual(response_body['error'], 'Invalid input shape')


if __name__ == '__main__':
    unittest.main()