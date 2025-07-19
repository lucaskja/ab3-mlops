#!/usr/bin/env python3
"""
Integration tests for Role-Based Access Control

Tests the IAM role-based access control for Data Scientists and ML Engineers.
"""

import unittest
import os
import sys
import json
import boto3
from unittest.mock import patch, MagicMock, ANY

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import IAM validation utilities
from scripts.setup.validate_iam_roles import (
    validate_data_scientist_role,
    validate_ml_engineer_role,
    validate_role_separation
)


class TestRoleBasedAccess(unittest.TestCase):
    """Integration tests for role-based access control"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock AWS session and clients
        self.session_patch = patch('boto3.Session')
        self.mock_session = self.session_patch.start()
        
        # Mock IAM client
        self.mock_iam_client = MagicMock()
        self.mock_session.return_value.client.return_value = self.mock_iam_client
        
        # Mock role responses
        self.mock_iam_client.get_role.side_effect = self._mock_get_role
        self.mock_iam_client.list_attached_role_policies.side_effect = self._mock_list_attached_role_policies
        self.mock_iam_client.get_policy.side_effect = self._mock_get_policy
        self.mock_iam_client.get_policy_version.side_effect = self._mock_get_policy_version
        
        # Define test roles
        self.data_scientist_role = "mlops-sagemaker-demo-data-scientist"
        self.ml_engineer_role = "mlops-sagemaker-demo-ml-engineer"
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop patches
        self.session_patch.stop()
    
    def _mock_get_role(self, RoleName):
        """Mock get_role response"""
        if RoleName == self.data_scientist_role:
            return {
                "Role": {
                    "RoleName": self.data_scientist_role,
                    "Arn": f"arn:aws:iam::123456789012:role/{self.data_scientist_role}",
                    "Path": "/",
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "sagemaker.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    }
                }
            }
        elif RoleName == self.ml_engineer_role:
            return {
                "Role": {
                    "RoleName": self.ml_engineer_role,
                    "Arn": f"arn:aws:iam::123456789012:role/{self.ml_engineer_role}",
                    "Path": "/",
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "sagemaker.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    }
                }
            }
        else:
            raise Exception(f"Role {RoleName} not found")
    
    def _mock_list_attached_role_policies(self, RoleName):
        """Mock list_attached_role_policies response"""
        if RoleName == self.data_scientist_role:
            return {
                "AttachedPolicies": [
                    {
                        "PolicyName": "DataScientistPolicy",
                        "PolicyArn": "arn:aws:iam::123456789012:policy/DataScientistPolicy"
                    },
                    {
                        "PolicyName": "AmazonSageMakerFullAccess",
                        "PolicyArn": "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                    }
                ]
            }
        elif RoleName == self.ml_engineer_role:
            return {
                "AttachedPolicies": [
                    {
                        "PolicyName": "MLEngineerPolicy",
                        "PolicyArn": "arn:aws:iam::123456789012:policy/MLEngineerPolicy"
                    },
                    {
                        "PolicyName": "AmazonSageMakerFullAccess",
                        "PolicyArn": "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                    }
                ]
            }
        else:
            return {"AttachedPolicies": []}
    
    def _mock_get_policy(self, PolicyArn):
        """Mock get_policy response"""
        return {
            "Policy": {
                "PolicyName": PolicyArn.split("/")[-1],
                "PolicyId": "ABCDEFGHIJKLMNOPQRSTU",
                "Arn": PolicyArn,
                "DefaultVersionId": "v1"
            }
        }
    
    def _mock_get_policy_version(self, PolicyArn, VersionId):
        """Mock get_policy_version response"""
        if "DataScientistPolicy" in PolicyArn:
            return {
                "PolicyVersion": {
                    "Document": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "s3:GetObject",
                                    "s3:ListBucket"
                                ],
                                "Resource": [
                                    "arn:aws:s3:::lucaskle-ab3-project-pv",
                                    "arn:aws:s3:::lucaskle-ab3-project-pv/*"
                                ]
                            },
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "sagemaker:CreateExperiment",
                                    "sagemaker:CreateTrial",
                                    "sagemaker:DescribeExperiment",
                                    "sagemaker:DescribeTrial",
                                    "sagemaker:ListExperiments",
                                    "sagemaker:ListTrials"
                                ],
                                "Resource": "*"
                            },
                            {
                                "Effect": "Deny",
                                "Action": [
                                    "sagemaker:CreateEndpoint",
                                    "sagemaker:UpdateEndpoint",
                                    "sagemaker:DeleteEndpoint"
                                ],
                                "Resource": "*"
                            }
                        ]
                    }
                }
            }
        elif "MLEngineerPolicy" in PolicyArn:
            return {
                "PolicyVersion": {
                    "Document": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "s3:*"
                                ],
                                "Resource": [
                                    "arn:aws:s3:::lucaskle-ab3-project-pv",
                                    "arn:aws:s3:::lucaskle-ab3-project-pv/*"
                                ]
                            },
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "sagemaker:*"
                                ],
                                "Resource": "*"
                            },
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "iam:PassRole"
                                ],
                                "Resource": "*",
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": "sagemaker.amazonaws.com"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        else:
            return {
                "PolicyVersion": {
                    "Document": {
                        "Version": "2012-10-17",
                        "Statement": []
                    }
                }
            }
    
    def test_data_scientist_role_permissions(self):
        """Test Data Scientist role permissions"""
        # Call validation function
        result = validate_data_scientist_role(
            role_name=self.data_scientist_role,
            aws_profile="ab"
        )
        
        # Verify validation result
        self.assertTrue(result["valid"])
        self.assertTrue(result["has_read_access_to_data"])
        self.assertTrue(result["has_experiment_permissions"])
        self.assertFalse(result["has_endpoint_permissions"])
        
        # Verify IAM client calls
        self.mock_iam_client.get_role.assert_called_with(RoleName=self.data_scientist_role)
        self.mock_iam_client.list_attached_role_policies.assert_called_with(RoleName=self.data_scientist_role)
    
    def test_ml_engineer_role_permissions(self):
        """Test ML Engineer role permissions"""
        # Call validation function
        result = validate_ml_engineer_role(
            role_name=self.ml_engineer_role,
            aws_profile="ab"
        )
        
        # Verify validation result
        self.assertTrue(result["valid"])
        self.assertTrue(result["has_full_access_to_data"])
        self.assertTrue(result["has_pipeline_permissions"])
        self.assertTrue(result["has_endpoint_permissions"])
        self.assertTrue(result["has_pass_role_permission"])
        
        # Verify IAM client calls
        self.mock_iam_client.get_role.assert_called_with(RoleName=self.ml_engineer_role)
        self.mock_iam_client.list_attached_role_policies.assert_called_with(RoleName=self.ml_engineer_role)
    
    def test_role_separation(self):
        """Test role separation between Data Scientist and ML Engineer"""
        # Call validation function
        result = validate_role_separation(
            data_scientist_role=self.data_scientist_role,
            ml_engineer_role=self.ml_engineer_role,
            aws_profile="ab"
        )
        
        # Verify validation result
        self.assertTrue(result["valid"])
        self.assertTrue(result["data_scientist_restricted"])
        self.assertTrue(result["ml_engineer_privileged"])
        
        # Verify IAM client calls
        self.mock_iam_client.get_role.assert_any_call(RoleName=self.data_scientist_role)
        self.mock_iam_client.get_role.assert_any_call(RoleName=self.ml_engineer_role)
    
    @patch('boto3.client')
    def test_data_scientist_s3_access(self, mock_boto3_client):
        """Test Data Scientist S3 access permissions"""
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client
        
        # Mock S3 responses
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test-file.txt'}
            ]
        }
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'test content')
        }
        mock_s3_client.put_object.side_effect = boto3.exceptions.ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'PutObject'
        )
        
        # Test S3 read access
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.client.return_value = mock_s3_client
            
            # Test list_objects (should succeed)
            objects = mock_s3_client.list_objects_v2(
                Bucket="lucaskle-ab3-project-pv",
                Prefix="data/"
            )
            self.assertEqual(len(objects['Contents']), 1)
            
            # Test get_object (should succeed)
            response = mock_s3_client.get_object(
                Bucket="lucaskle-ab3-project-pv",
                Key="test-file.txt"
            )
            self.assertEqual(response['Body'].read(), b'test content')
            
            # Test put_object (should fail)
            with self.assertRaises(boto3.exceptions.ClientError) as context:
                mock_s3_client.put_object(
                    Bucket="lucaskle-ab3-project-pv",
                    Key="new-file.txt",
                    Body=b'new content'
                )
            
            self.assertEqual(context.exception.response['Error']['Code'], 'AccessDenied')
    
    @patch('boto3.client')
    def test_ml_engineer_sagemaker_pipeline_access(self, mock_boto3_client):
        """Test ML Engineer SageMaker Pipeline access permissions"""
        # Mock SageMaker client
        mock_sagemaker_client = MagicMock()
        mock_boto3_client.return_value = mock_sagemaker_client
        
        # Mock SageMaker responses
        mock_sagemaker_client.create_pipeline.return_value = {
            "PipelineArn": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline"
        }
        mock_sagemaker_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
        }
        
        # Test pipeline creation and execution
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.client.return_value = mock_sagemaker_client
            
            # Test create_pipeline (should succeed)
            response = mock_sagemaker_client.create_pipeline(
                PipelineName="test-pipeline",
                PipelineDefinition="{}",
                RoleArn="arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )
            self.assertEqual(
                response["PipelineArn"],
                "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline"
            )
            
            # Test start_pipeline_execution (should succeed)
            response = mock_sagemaker_client.start_pipeline_execution(
                PipelineName="test-pipeline"
            )
            self.assertEqual(
                response["PipelineExecutionArn"],
                "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-pipeline/execution/test-execution"
            )


if __name__ == '__main__':
    unittest.main()