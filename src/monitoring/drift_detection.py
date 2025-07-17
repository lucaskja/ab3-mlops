"""
Drift detection utilities for SageMaker Model Monitor.

This module provides utilities for:
- Setting up data and model quality monitoring
- Detecting drift in input data and model predictions
- Configuring automated alerts for drift detection
- Analyzing monitoring results
"""

import boto3
import json
import logging
import os
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

class DriftDetector:
    """Class for detecting and managing data and model drift."""
    
    def __init__(
        self,
        endpoint_name: str,
        bucket: str,
        prefix: str = "monitoring",
        profile_name: str = "ab",
        region_name: str = "us-east-1"
    ):
        """
        Initialize the drift detector.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint to monitor
            bucket: S3 bucket for storing monitoring data
            prefix: S3 prefix for monitoring data
            profile_name: AWS profile name
            region_name: AWS region name
        """
        self.endpoint_name = endpoint_name
        self.bucket = bucket
        self.prefix = prefix
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.sagemaker_client = session.client('sagemaker')
        self.cloudwatch_client = session.client('cloudwatch')
        self.s3_client = session.client('s3')
        
        logger.info(f"Initialized drift detector for endpoint: {endpoint_name}")
    
    def create_data_quality_baseline(
        self,
        baseline_dataset: str,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_seconds: int = 3600
    ) -> str:
        """
        Create a data quality baseline for drift detection.
        
        Args:
            baseline_dataset: S3 URI to the baseline dataset
            instance_type: Instance type for processing job
            instance_count: Number of instances for processing job
            max_runtime_seconds: Maximum runtime in seconds
            
        Returns:
            Job name of the baseline processing job
        """
        # Generate a unique job name
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        job_name = f"{self.endpoint_name}-data-quality-baseline-{timestamp}"
        
        # Define output path
        output_s3_uri = f"s3://{self.bucket}/{self.prefix}/baselines/data-quality/{job_name}"
        
        # Create baseline job
        response = self.sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=job_name,
            MonitoringScheduleConfig={
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {
                            "S3Uri": f"{output_s3_uri}/constraints.json"
                        },
                        "StatisticsResource": {
                            "S3Uri": f"{output_s3_uri}/statistics.json"
                        }
                    },
                    "MonitoringInputs": [
                        {
                            "DatasetFormat": {
                                "Csv": {
                                    "Header": True
                                }
                            },
                            "S3Input": {
                                "LocalPath": "/opt/ml/processing/input",
                                "S3Uri": baseline_dataset,
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "LocalPath": "/opt/ml/processing/output",
                                    "S3Uri": output_s3_uri,
                                    "S3UploadMode": "EndOfJob"
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": instance_count,
                            "InstanceType": instance_type,
                            "VolumeSizeInGB": 30
                        }
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": max_runtime_seconds
                    }
                }
            }
        )
        
        logger.info(f"Created data quality baseline job: {job_name}")
        logger.info(f"Baseline output will be stored at: {output_s3_uri}")
        
        return job_name
    
    def create_model_quality_baseline(
        self,
        baseline_dataset: str,
        problem_type: str = "BinaryClassification",
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_seconds: int = 3600
    ) -> str:
        """
        Create a model quality baseline for drift detection.
        
        Args:
            baseline_dataset: S3 URI to the baseline dataset with ground truth labels
            problem_type: Type of ML problem (BinaryClassification, Regression, etc.)
            instance_type: Instance type for processing job
            instance_count: Number of instances for processing job
            max_runtime_seconds: Maximum runtime in seconds
            
        Returns:
            Job name of the baseline processing job
        """
        # Generate a unique job name
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        job_name = f"{self.endpoint_name}-model-quality-baseline-{timestamp}"
        
        # Define output path
        output_s3_uri = f"s3://{self.bucket}/{self.prefix}/baselines/model-quality/{job_name}"
        
        # Create baseline job
        response = self.sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=job_name,
            MonitoringScheduleConfig={
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {
                            "S3Uri": f"{output_s3_uri}/constraints.json"
                        },
                        "StatisticsResource": {
                            "S3Uri": f"{output_s3_uri}/statistics.json"
                        }
                    },
                    "MonitoringInputs": [
                        {
                            "DatasetFormat": {
                                "Csv": {
                                    "Header": True
                                }
                            },
                            "S3Input": {
                                "LocalPath": "/opt/ml/processing/input",
                                "S3Uri": baseline_dataset,
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "LocalPath": "/opt/ml/processing/output",
                                    "S3Uri": output_s3_uri,
                                    "S3UploadMode": "EndOfJob"
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": instance_count,
                            "InstanceType": instance_type,
                            "VolumeSizeInGB": 30
                        }
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": max_runtime_seconds
                    },
                    "Environment": {
                        "PROBLEM_TYPE": problem_type
                    }
                }
            }
        )
        
        logger.info(f"Created model quality baseline job: {job_name}")
        logger.info(f"Baseline output will be stored at: {output_s3_uri}")
        
        return job_name
    
    def create_data_quality_monitoring_schedule(
        self,
        baseline_constraints_uri: str,
        baseline_statistics_uri: str,
        schedule_expression: str = "cron(0 * ? * * *)",  # Hourly
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_seconds: int = 3600
    ) -> str:
        """
        Create a data quality monitoring schedule for an endpoint.
        
        Args:
            baseline_constraints_uri: S3 URI to baseline constraints
            baseline_statistics_uri: S3 URI to baseline statistics
            schedule_expression: Schedule expression (cron format)
            instance_type: Instance type for monitoring job
            instance_count: Number of instances for monitoring job
            max_runtime_seconds: Maximum runtime in seconds
            
        Returns:
            Name of the created monitoring schedule
        """
        # Generate a unique schedule name
        schedule_name = f"{self.endpoint_name}-data-quality-monitoring"
        
        # Define output path
        output_s3_uri = f"s3://{self.bucket}/{self.prefix}/monitoring/data-quality"
        
        # Create monitoring schedule
        response = self.sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {
                    "ScheduleExpression": schedule_expression
                },
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {
                            "S3Uri": baseline_constraints_uri
                        },
                        "StatisticsResource": {
                            "S3Uri": baseline_statistics_uri
                        }
                    },
                    "MonitoringInputs": [
                        {
                            "EndpointInput": {
                                "EndpointName": self.endpoint_name,
                                "LocalPath": "/opt/ml/processing/input",
                                "S3InputMode": "File",
                                "S3DataDistributionType": "FullyReplicated"
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "LocalPath": "/opt/ml/processing/output",
                                    "S3Uri": output_s3_uri,
                                    "S3UploadMode": "EndOfJob"
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": instance_count,
                            "InstanceType": instance_type,
                            "VolumeSizeInGB": 30
                        }
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": max_runtime_seconds
                    }
                }
            }
        )
        
        logger.info(f"Created data quality monitoring schedule: {schedule_name}")
        logger.info(f"Monitoring output will be stored at: {output_s3_uri}")
        
        return schedule_name
    
    def create_model_quality_monitoring_schedule(
        self,
        baseline_constraints_uri: str,
        baseline_statistics_uri: str,
        ground_truth_input: str,
        problem_type: str = "BinaryClassification",
        schedule_expression: str = "cron(0 * ? * * *)",  # Hourly
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_seconds: int = 3600
    ) -> str:
        """
        Create a model quality monitoring schedule for an endpoint.
        
        Args:
            baseline_constraints_uri: S3 URI to baseline constraints
            baseline_statistics_uri: S3 URI to baseline statistics
            ground_truth_input: S3 URI to ground truth data
            problem_type: Type of ML problem (BinaryClassification, Regression, etc.)
            schedule_expression: Schedule expression (cron format)
            instance_type: Instance type for monitoring job
            instance_count: Number of instances for monitoring job
            max_runtime_seconds: Maximum runtime in seconds
            
        Returns:
            Name of the created monitoring schedule
        """
        # Generate a unique schedule name
        schedule_name = f"{self.endpoint_name}-model-quality-monitoring"
        
        # Define output path
        output_s3_uri = f"s3://{self.bucket}/{self.prefix}/monitoring/model-quality"
        
        # Create monitoring schedule
        response = self.sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {
                    "ScheduleExpression": schedule_expression
                },
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {
                            "S3Uri": baseline_constraints_uri
                        },
                        "StatisticsResource": {
                            "S3Uri": baseline_statistics_uri
                        }
                    },
                    "MonitoringInputs": [
                        {
                            "EndpointInput": {
                                "EndpointName": self.endpoint_name,
                                "LocalPath": "/opt/ml/processing/input/endpoint",
                                "S3InputMode": "File",
                                "S3DataDistributionType": "FullyReplicated"
                            }
                        },
                        {
                            "BatchTransformInput": {
                                "DatasetFormat": {
                                    "Csv": {
                                        "Header": True
                                    }
                                },
                                "LocalPath": "/opt/ml/processing/input/groundtruth",
                                "S3Uri": ground_truth_input,
                                "S3DataDistributionType": "FullyReplicated",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "LocalPath": "/opt/ml/processing/output",
                                    "S3Uri": output_s3_uri,
                                    "S3UploadMode": "EndOfJob"
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": instance_count,
                            "InstanceType": instance_type,
                            "VolumeSizeInGB": 30
                        }
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": max_runtime_seconds
                    },
                    "Environment": {
                        "PROBLEM_TYPE": problem_type
                    }
                }
            }
        )
        
        logger.info(f"Created model quality monitoring schedule: {schedule_name}")
        logger.info(f"Monitoring output will be stored at: {output_s3_uri}")
        
        return schedule_name
    
    def create_drift_alert(
        self,
        metric_name: str,
        threshold: float,
        comparison_operator: str = "GreaterThanThreshold",
        evaluation_periods: int = 1,
        period: int = 3600,
        statistic: str = "Maximum",
        sns_topic_arn: Optional[str] = None
    ) -> str:
        """
        Create a CloudWatch alarm for drift detection.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold: Threshold value for the alarm
            comparison_operator: Comparison operator for the alarm
            evaluation_periods: Number of periods to evaluate
            period: Period in seconds
            statistic: Statistic to use for the alarm
            sns_topic_arn: ARN of the SNS topic for notifications
            
        Returns:
            Name of the created alarm
        """
        # Generate a unique alarm name
        alarm_name = f"{self.endpoint_name}-{metric_name}-drift-alarm"
        
        # Create alarm configuration
        alarm_config = {
            "AlarmName": alarm_name,
            "AlarmDescription": f"Drift detection alarm for {self.endpoint_name} on {metric_name}",
            "MetricName": metric_name,
            "Namespace": "AWS/SageMaker",
            "Dimensions": [
                {
                    "Name": "EndpointName",
                    "Value": self.endpoint_name
                },
                {
                    "Name": "MonitoringSchedule",
                    "Value": f"{self.endpoint_name}-data-quality-monitoring"
                }
            ],
            "Statistic": statistic,
            "Period": period,
            "EvaluationPeriods": evaluation_periods,
            "Threshold": threshold,
            "ComparisonOperator": comparison_operator,
            "TreatMissingData": "notBreaching"
        }
        
        # Add SNS topic if provided
        if sns_topic_arn:
            alarm_config["AlarmActions"] = [sns_topic_arn]
        
        # Create the alarm
        response = self.cloudwatch_client.put_metric_alarm(**alarm_config)
        
        logger.info(f"Created drift detection alarm: {alarm_name}")
        logger.info(f"Alarm will trigger when {metric_name} is {comparison_operator} {threshold}")
        
        return alarm_name
    
    def get_monitoring_results(
        self,
        monitoring_type: str = "data-quality",
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent monitoring results.
        
        Args:
            monitoring_type: Type of monitoring (data-quality or model-quality)
            max_results: Maximum number of results to return
            
        Returns:
            List of monitoring results
        """
        # Define the S3 prefix for monitoring results
        results_prefix = f"{self.prefix}/monitoring/{monitoring_type}"
        
        # List objects in the S3 bucket
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=results_prefix,
            MaxKeys=100
        )
        
        # Extract monitoring results
        results = []
        if 'Contents' in response:
            # Sort by last modified date (newest first)
            sorted_contents = sorted(
                response['Contents'],
                key=lambda x: x['LastModified'],
                reverse=True
            )
            
            # Process each monitoring result
            for obj in sorted_contents[:max_results]:
                if obj['Key'].endswith('violations.json'):
                    # Get the violations file
                    violations_response = self.s3_client.get_object(
                        Bucket=self.bucket,
                        Key=obj['Key']
                    )
                    
                    # Parse the violations
                    violations = json.loads(violations_response['Body'].read().decode('utf-8'))
                    
                    # Extract execution date from the key
                    execution_date = obj['LastModified']
                    
                    # Add to results
                    results.append({
                        'execution_date': execution_date,
                        'violations': violations,
                        's3_uri': f"s3://{self.bucket}/{obj['Key']}"
                    })
        
        logger.info(f"Retrieved {len(results)} monitoring results for {monitoring_type}")
        
        return results
    
    def analyze_drift(
        self,
        monitoring_type: str = "data-quality",
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze drift over time.
        
        Args:
            monitoring_type: Type of monitoring (data-quality or model-quality)
            days: Number of days to analyze
            
        Returns:
            Dictionary with drift analysis results
        """
        # Get monitoring results
        results = self.get_monitoring_results(monitoring_type=monitoring_type, max_results=100)
        
        # Filter results by date
        start_date = datetime.now() - timedelta(days=days)
        filtered_results = [
            r for r in results if r['execution_date'] >= start_date
        ]
        
        # Count violations by type
        violation_counts = {}
        for result in filtered_results:
            for violation in result.get('violations', []):
                violation_type = violation.get('feature_name', 'unknown')
                if violation_type not in violation_counts:
                    violation_counts[violation_type] = 0
                violation_counts[violation_type] += 1
        
        # Calculate drift metrics
        drift_metrics = {
            'total_executions': len(filtered_results),
            'executions_with_violations': sum(1 for r in filtered_results if r.get('violations')),
            'violation_counts': violation_counts,
            'time_period': f"Last {days} days",
            'monitoring_type': monitoring_type
        }
        
        logger.info(f"Analyzed drift for {monitoring_type} over the last {days} days")
        logger.info(f"Found {drift_metrics['executions_with_violations']} executions with violations")
        
        return drift_metrics
    
    def trigger_retraining(
        self,
        pipeline_name: str,
        pipeline_parameters: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Trigger model retraining based on drift detection.
        
        Args:
            pipeline_name: Name of the SageMaker Pipeline to execute
            pipeline_parameters: Parameters for the pipeline execution
            
        Returns:
            ARN of the pipeline execution
        """
        # Set default parameters if not provided
        if pipeline_parameters is None:
            pipeline_parameters = {}
        
        # Add default parameters
        pipeline_parameters['EndpointName'] = self.endpoint_name
        pipeline_parameters['RetrainingReason'] = 'DriftDetected'
        
        # Start pipeline execution
        response = self.sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=[
                {
                    'Name': name,
                    'Value': value
                }
                for name, value in pipeline_parameters.items()
            ]
        )
        
        execution_arn = response['PipelineExecutionArn']
        
        logger.info(f"Triggered retraining pipeline: {pipeline_name}")
        logger.info(f"Pipeline execution ARN: {execution_arn}")
        
        return execution_arn