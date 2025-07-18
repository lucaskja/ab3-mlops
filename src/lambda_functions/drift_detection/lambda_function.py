#!/usr/bin/env python3
"""
Drift Detection Lambda Function

This Lambda function is triggered by EventBridge when a SageMaker Model Monitor
execution completes. It checks for data drift violations and triggers a retraining
pipeline if drift is detected.
"""

import json
import boto3
import os
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

# Configuration - these values will be replaced during deployment
ENDPOINT_NAME = "ENDPOINT_NAME_PLACEHOLDER"
PIPELINE_NAME = "PIPELINE_NAME_PLACEHOLDER"
DRIFT_THRESHOLD = DRIFT_THRESHOLD_PLACEHOLDER  # Numeric value, no quotes
BUCKET = "BUCKET_PLACEHOLDER"
PREFIX = "PREFIX_PLACEHOLDER"
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')

def lambda_handler(event, context):
    """Lambda handler for drift detection and retraining trigger."""
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Extract monitoring schedule name from event
        monitoring_schedule_name = event.get('detail', {}).get('monitoringScheduleName', '')
        if not monitoring_schedule_name:
            logger.warning("No monitoring schedule name found in event")
            return {
                'statusCode': 400,
                'body': json.dumps('No monitoring schedule name found in event')
            }
        
        # Get the latest monitoring execution
        response = sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=monitoring_schedule_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        executions = response.get('MonitoringExecutionSummaries', [])
        if not executions:
            logger.warning(f"No monitoring executions found for schedule: {monitoring_schedule_name}")
            return {
                'statusCode': 404,
                'body': json.dumps('No monitoring executions found')
            }
        
        # Get the latest execution
        latest_execution = executions[0]
        execution_arn = latest_execution.get('MonitoringExecutionArn')
        status = latest_execution.get('MonitoringExecutionStatus')
        
        if status != 'Completed':
            logger.warning(f"Latest monitoring execution not completed. Status: {status}")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Monitoring execution status: {status}')
            }
        
        # Get execution details
        execution_details = sagemaker_client.describe_monitoring_execution(
            MonitoringExecutionArn=execution_arn
        )
        
        # Get output path
        output_path = None
        outputs = execution_details.get('ProcessingOutputConfig', {}).get('Outputs', [])
        for output in outputs:
            if output.get('OutputName') == 'evaluation':
                output_path = output.get('S3Output', {}).get('S3Uri')
                break
        
        if not output_path:
            logger.warning("No evaluation output found in execution details")
            return {
                'statusCode': 404,
                'body': json.dumps('No evaluation output found')
            }
        
        # Parse S3 URI
        s3_parts = output_path.replace('s3://', '').split('/')
        bucket_name = s3_parts[0]
        key_prefix = '/'.join(s3_parts[1:])
        
        # Download and parse violations file
        violations_key = f"{key_prefix}/constraint_violations.json"
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=violations_key
            )
            violations_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Check for violations
            violations = violations_data.get('violations', [])
            violations_count = len(violations)
            
            logger.info(f"Found {violations_count} violations")
            
            # Determine if retraining is needed
            retraining_needed = violations_count > 0
            
            if retraining_needed:
                logger.info(f"Drift detected. Triggering retraining pipeline: {PIPELINE_NAME}")
                
                # Start pipeline execution
                pipeline_response = sagemaker_client.start_pipeline_execution(
                    PipelineName=PIPELINE_NAME,
                    PipelineParameters=[
                        {
                            'Name': 'EndpointName',
                            'Value': ENDPOINT_NAME
                        },
                        {
                            'Name': 'RetrainingReason',
                            'Value': 'DriftDetected'
                        }
                    ]
                )
                
                execution_arn = pipeline_response['PipelineExecutionArn']
                
                # Send notification if SNS topic is configured
                if SNS_TOPIC_ARN:
                    sns_client.publish(
                        TopicArn=SNS_TOPIC_ARN,
                        Subject=f"Model Retraining Triggered for {ENDPOINT_NAME}",
                        Message=f"Model drift detected for endpoint {ENDPOINT_NAME}. "
                                f"Retraining pipeline {PIPELINE_NAME} has been triggered. "
                                f"Pipeline execution ARN: {execution_arn}. "
                                f"Number of violations: {violations_count}."
                    )
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'Retraining pipeline triggered',
                        'pipeline_name': PIPELINE_NAME,
                        'execution_arn': execution_arn,
                        'violations_count': violations_count
                    })
                }
            else:
                logger.info("No drift detected. Retraining not needed.")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'No drift detected',
                        'violations_count': violations_count
                    })
                }
                
        except Exception as e:
            logger.error(f"Error processing violations file: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps(f'Error: {str(e)}')
            }
            
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }