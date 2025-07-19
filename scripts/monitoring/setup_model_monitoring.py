#!/usr/bin/env python3
"""
Script to set up model monitoring for a SageMaker endpoint.
"""

import argparse
import json
import boto3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up model monitoring for SageMaker endpoint')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--endpoint-name', type=str, required=True, help='Name of the endpoint')
    parser.add_argument('--monitoring-schedule-name', type=str, required=True, help='Name of the monitoring schedule')
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge', help='Instance type for monitoring jobs')
    parser.add_argument('--instance-count', type=int, default=1, help='Number of instances for monitoring jobs')
    parser.add_argument('--output', type=str, default='monitoring_info.json', help='Path to output JSON file')
    return parser.parse_args()

def setup_model_monitoring(args):
    """Set up model monitoring for SageMaker endpoint."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    sagemaker_client = session.client('sagemaker')
    
    # Get endpoint info
    endpoint_info = sagemaker_client.describe_endpoint(EndpointName=args.endpoint_name)
    endpoint_config_name = endpoint_info['EndpointConfigName']
    endpoint_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    
    # Get data capture config
    data_capture_config = endpoint_config.get('DataCaptureConfig', {})
    data_capture_s3_uri = data_capture_config.get('DestinationS3Uri', f"s3://lucaskle-ab3-project-pv/endpoint-data-capture/{args.endpoint_name}/")
    
    # Set up data quality monitoring
    data_quality_job_definition_name = f"{args.monitoring_schedule_name}-data-quality"
    logger.info(f"Creating data quality job definition {data_quality_job_definition_name}")
    
    sagemaker_client.create_data_quality_job_definition(
        JobDefinitionName=data_quality_job_definition_name,
        DataQualityBaselineConfig={
            'BaseliningJobName': f"{data_quality_job_definition_name}-baseline-{int(datetime.now().timestamp())}",
            'ConstraintsResource': {
                'S3Uri': f"s3://lucaskle-ab3-project-pv/monitoring/baselines/{args.endpoint_name}/data_quality/constraints.json"
            }
        },
        DataQualityAppSpecification={
            'ImageUri': f"amazonaws.com/sagemaker-model-monitor-analyzer:latest",
            'ContainerEntrypoint': [
                '/opt/program/run'
            ],
            'ContainerArguments': [
                'data-quality-monitoring'
            ],
            'RecordPreprocessorSourceUri': f"s3://lucaskle-ab3-project-pv/monitoring/preprocessors/{args.endpoint_name}/data_quality_preprocessor.py",
            'PostAnalyticsProcessorSourceUri': f"s3://lucaskle-ab3-project-pv/monitoring/postprocessors/{args.endpoint_name}/data_quality_postprocessor.py"
        },
        DataQualityJobInput={
            'EndpointInput': {
                'EndpointName': args.endpoint_name,
                'LocalPath': '/opt/ml/processing/input/endpoint',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        DataQualityJobOutputConfig={
            'MonitoringOutputs': [
                {
                    'S3Output': {
                        'S3Uri': f"s3://lucaskle-ab3-project-pv/monitoring/output/{args.endpoint_name}/data-quality",
                        'LocalPath': '/opt/ml/processing/output',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },
        JobResources={
            'ClusterConfig': {
                'InstanceCount': args.instance_count,
                'InstanceType': args.instance_type,
                'VolumeSizeInGB': 20
            }
        },
        RoleArn=f"arn:aws:iam::{session.client('sts').get_caller_identity()['Account']}:role/service-role/AmazonSageMaker-ExecutionRole",
        NetworkConfig={
            'EnableInterContainerTrafficEncryption': True,
            'EnableNetworkIsolation': True
        }
    )
    
    # Set up model quality monitoring
    model_quality_job_definition_name = f"{args.monitoring_schedule_name}-model-quality"
    logger.info(f"Creating model quality job definition {model_quality_job_definition_name}")
    
    sagemaker_client.create_model_quality_job_definition(
        JobDefinitionName=model_quality_job_definition_name,
        ModelQualityBaselineConfig={
            'BaseliningJobName': f"{model_quality_job_definition_name}-baseline-{int(datetime.now().timestamp())}",
            'ConstraintsResource': {
                'S3Uri': f"s3://lucaskle-ab3-project-pv/monitoring/baselines/{args.endpoint_name}/model_quality/constraints.json"
            }
        },
        ModelQualityAppSpecification={
            'ImageUri': f"amazonaws.com/sagemaker-model-monitor-analyzer:latest",
            'ContainerEntrypoint': [
                '/opt/program/run'
            ],
            'ContainerArguments': [
                'model-quality-monitoring'
            ],
            'ProblemType': 'BinaryClassification',
            'ModelQualityJobInput': {
                'EndpointInput': {
                    'EndpointName': args.endpoint_name,
                    'LocalPath': '/opt/ml/processing/input/endpoint',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'FeaturesAttribute': 'features',
                    'InferenceAttribute': 'predictions',
                    'ProbabilityAttribute': 'probability',
                    'ProbabilityThresholdAttribute': 0.5
                },
                'GroundTruthS3Input': {
                    'S3Uri': f"s3://lucaskle-ab3-project-pv/monitoring/ground-truth/{args.endpoint_name}/"
                }
            }
        },
        ModelQualityJobOutputConfig={
            'MonitoringOutputs': [
                {
                    'S3Output': {
                        'S3Uri': f"s3://lucaskle-ab3-project-pv/monitoring/output/{args.endpoint_name}/model-quality",
                        'LocalPath': '/opt/ml/processing/output',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },
        JobResources={
            'ClusterConfig': {
                'InstanceCount': args.instance_count,
                'InstanceType': args.instance_type,
                'VolumeSizeInGB': 20
            }
        },
        RoleArn=f"arn:aws:iam::{session.client('sts').get_caller_identity()['Account']}:role/service-role/AmazonSageMaker-ExecutionRole",
        NetworkConfig={
            'EnableInterContainerTrafficEncryption': True,
            'EnableNetworkIsolation': True
        }
    )
    
    # Create monitoring schedules
    data_quality_schedule_name = f"{args.monitoring_schedule_name}-data-quality"
    logger.info(f"Creating data quality monitoring schedule {data_quality_schedule_name}")
    
    sagemaker_client.create_monitoring_schedule(
        MonitoringScheduleName=data_quality_schedule_name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': 'cron(0 0 * * ? *)'  # Daily at midnight
            },
            'MonitoringJobDefinition': {
                'MonitoringType': 'DataQuality',
                'DataQualityJobDefinitionName': data_quality_job_definition_name
            }
        }
    )
    
    model_quality_schedule_name = f"{args.monitoring_schedule_name}-model-quality"
    logger.info(f"Creating model quality monitoring schedule {model_quality_schedule_name}")
    
    sagemaker_client.create_monitoring_schedule(
        MonitoringScheduleName=model_quality_schedule_name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': 'cron(0 0 * * ? *)'  # Daily at midnight
            },
            'MonitoringJobDefinition': {
                'MonitoringType': 'ModelQuality',
                'ModelQualityJobDefinitionName': model_quality_job_definition_name
            }
        }
    )
    
    # Save monitoring info
    monitoring_info = {
        'endpoint_name': args.endpoint_name,
        'data_capture_s3_uri': data_capture_s3_uri,
        'data_quality_job_definition_name': data_quality_job_definition_name,
        'model_quality_job_definition_name': model_quality_job_definition_name,
        'data_quality_schedule_name': data_quality_schedule_name,
        'model_quality_schedule_name': model_quality_schedule_name,
        'monitoring_output_s3_uri': f"s3://lucaskle-ab3-project-pv/monitoring/output/{args.endpoint_name}/",
        'creation_time': datetime.now().isoformat()
    }
    
    with open(args.output, 'w') as f:
        json.dump(monitoring_info, f, indent=2)
    
    logger.info(f"Monitoring info saved to {args.output}")
    return monitoring_info

def main():
    """Main function."""
    args = parse_args()
    setup_model_monitoring(args)

if __name__ == '__main__':
    main()