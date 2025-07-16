"""
S3 Data Access Utilities for MLOps SageMaker Demo

This module provides utilities for accessing and managing data in S3 buckets
with proper error handling and AWS profile support.

Requirements addressed:
- 1.1: Data Scientist read-only access to S3 dataset
- Proper error handling for S3 operations
- Support for AWS CLI profile configuration
"""

import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
from typing import List, Dict, Optional, Any, Union
import io
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataAccess:
    """
    Utility class for accessing S3 data with proper error handling and governance.
    
    Provides read-only access to S3 buckets for Data Scientists with comprehensive
    error handling and logging capabilities.
    """
    
    def __init__(self, bucket_name: str, aws_profile: Optional[str] = None, region: str = 'us-east-1'):
        """
        Initialize S3 data access client.
        
        Args:
            bucket_name: Name of the S3 bucket to access
            aws_profile: AWS CLI profile to use (defaults to 'ab' if not specified)
            region: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name
        self.aws_profile = aws_profile or 'ab'
        self.region = region
        
        try:
            # Initialize boto3 session with specified profile
            self.session = boto3.Session(profile_name=self.aws_profile)
            self.s3_client = self.session.client('s3', region_name=self.region)
            self.s3_resource = self.session.resource('s3', region_name=self.region)
            
            # Validate bucket access
            self._validate_bucket_access()
            
            logger.info(f"Successfully initialized S3 access for bucket: {bucket_name}")
            
        except ProfileNotFound:
            raise ValueError(f"AWS profile '{self.aws_profile}' not found. Please configure AWS CLI.")
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Please configure AWS CLI.")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def _validate_bucket_access(self) -> None:
        """
        Validate that the bucket exists and is accessible.
        
        Raises:
            ValueError: If bucket is not accessible
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                raise ValueError(f"Access denied to bucket '{self.bucket_name}'. Check IAM permissions.")
            else:
                raise ValueError(f"Error accessing bucket '{self.bucket_name}': {str(e)}")
    
    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in the S3 bucket with optional prefix filtering.
        
        Args:
            prefix: Object key prefix to filter by
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object metadata dictionaries
            
        Raises:
            RuntimeError: If listing objects fails
        """
        try:
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'Key': obj['Key'],
                            'Size': obj['Size'],
                            'LastModified': obj['LastModified'],
                            'ETag': obj['ETag'].strip('"')
                        })
            
            logger.info(f"Listed {len(objects)} objects with prefix '{prefix}'")
            return objects
            
        except ClientError as e:
            logger.error(f"Error listing objects: {str(e)}")
            raise RuntimeError(f"Failed to list objects in bucket: {str(e)}")
    
    def download_file_to_memory(self, key: str) -> io.BytesIO:
        """
        Download a file from S3 to memory.
        
        Args:
            key: S3 object key
            
        Returns:
            BytesIO object containing file data
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            file_obj = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, key, file_obj)
            file_obj.seek(0)
            
            logger.debug(f"Downloaded file '{key}' to memory")
            return file_obj
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise RuntimeError(f"File '{key}' not found in bucket")
            else:
                logger.error(f"Error downloading file '{key}': {str(e)}")
                raise RuntimeError(f"Failed to download file '{key}': {str(e)}")
    
    def download_file_to_local(self, key: str, local_path: str) -> None:
        """
        Download a file from S3 to local filesystem.
        
        Args:
            key: S3 object key
            local_path: Local file path to save to
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, key, local_path)
            logger.info(f"Downloaded file '{key}' to '{local_path}'")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise RuntimeError(f"File '{key}' not found in bucket")
            else:
                logger.error(f"Error downloading file '{key}': {str(e)}")
                raise RuntimeError(f"Failed to download file '{key}': {str(e)}")
    
    def get_object_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata for an S3 object.
        
        Args:
            key: S3 object key
            
        Returns:
            Dictionary containing object metadata
            
        Raises:
            RuntimeError: If getting metadata fails
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            
            metadata = {
                'ContentLength': response['ContentLength'],
                'ContentType': response.get('ContentType', 'unknown'),
                'LastModified': response['LastModified'],
                'ETag': response['ETag'].strip('"'),
                'Metadata': response.get('Metadata', {})
            }
            
            logger.debug(f"Retrieved metadata for '{key}'")
            return metadata
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise RuntimeError(f"File '{key}' not found in bucket")
            else:
                logger.error(f"Error getting metadata for '{key}': {str(e)}")
                raise RuntimeError(f"Failed to get metadata for '{key}': {str(e)}")
    
    def check_object_exists(self, key: str) -> bool:
        """
        Check if an object exists in the S3 bucket.
        
        Args:
            key: S3 object key
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                logger.error(f"Error checking if object '{key}' exists: {str(e)}")
                raise RuntimeError(f"Failed to check object existence: {str(e)}")
    
    def get_bucket_size(self, prefix: str = '') -> Dict[str, Union[int, float]]:
        """
        Calculate total size of objects in bucket with optional prefix.
        
        Args:
            prefix: Object key prefix to filter by
            
        Returns:
            Dictionary with total size in bytes and human-readable format
        """
        try:
            total_size = 0
            object_count = 0
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
                        object_count += 1
            
            # Convert to human-readable format
            size_gb = total_size / (1024**3)
            size_mb = total_size / (1024**2)
            
            result = {
                'total_bytes': total_size,
                'total_mb': round(size_mb, 2),
                'total_gb': round(size_gb, 2),
                'object_count': object_count
            }
            
            logger.info(f"Bucket size calculation: {result}")
            return result
            
        except ClientError as e:
            logger.error(f"Error calculating bucket size: {str(e)}")
            raise RuntimeError(f"Failed to calculate bucket size: {str(e)}")
    
    def filter_objects_by_extension(self, extensions: List[str], prefix: str = '') -> List[str]:
        """
        Filter objects by file extension.
        
        Args:
            extensions: List of file extensions to filter by (without dots)
            prefix: Object key prefix to filter by
            
        Returns:
            List of object keys matching the extensions
        """
        try:
            objects = self.list_objects(prefix=prefix)
            filtered_keys = []
            
            for obj in objects:
                key = obj['Key']
                if '.' in key:
                    file_ext = key.split('.')[-1].lower()
                    if file_ext in [ext.lower() for ext in extensions]:
                        filtered_keys.append(key)
            
            logger.info(f"Found {len(filtered_keys)} objects with extensions {extensions}")
            return filtered_keys
            
        except Exception as e:
            logger.error(f"Error filtering objects by extension: {str(e)}")
            raise RuntimeError(f"Failed to filter objects: {str(e)}")
    
    def get_connection_info(self) -> Dict[str, str]:
        """
        Get information about the current S3 connection.
        
        Returns:
            Dictionary with connection details
        """
        return {
            'bucket_name': self.bucket_name,
            'aws_profile': self.aws_profile,
            'region': self.region,
            'connection_time': datetime.now().isoformat()
        }


# Utility functions for common operations
def create_s3_client(bucket_name: str, aws_profile: str = 'ab') -> S3DataAccess:
    """
    Factory function to create S3DataAccess client.
    
    Args:
        bucket_name: Name of the S3 bucket
        aws_profile: AWS CLI profile to use
        
    Returns:
        Configured S3DataAccess instance
    """
    return S3DataAccess(bucket_name=bucket_name, aws_profile=aws_profile)


def validate_s3_access(bucket_name: str, aws_profile: str = 'ab') -> bool:
    """
    Validate S3 access without creating a persistent client.
    
    Args:
        bucket_name: Name of the S3 bucket
        aws_profile: AWS CLI profile to use
        
    Returns:
        True if access is valid, False otherwise
    """
    try:
        client = S3DataAccess(bucket_name=bucket_name, aws_profile=aws_profile)
        return True
    except Exception as e:
        logger.error(f"S3 access validation failed: {str(e)}")
        return False