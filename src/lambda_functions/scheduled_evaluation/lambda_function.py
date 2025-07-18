#!/usr/bin/env python3
"""
Scheduled Evaluation Lambda Function

This Lambda function is triggered on a schedule to evaluate model performance
against a threshold and trigger retraining if performance has degraded.
"""

import json
import boto3
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

# Configuration - these values will be replaced during deployment
ENDPOINT_NAME = "ENDPOINT_NAME_PLACEHOLDER"
PIPELINE_NAME = "PIPELINE_NAME_PLACEHOLDER"
PERFORMANCE_THRESHOLD = PERFORMANCE_THRESHOLD_PLACEHOLDER  # Numeric value, no quotes
EVALUATION_DATA = "EVALUATION_DATA_PLACEHOLDER"
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')

def lambda_handler(event, context):
    """Lambda handler for scheduled model evaluation."""
    logger.info(f"Starting scheduled evaluation for endpoint: {ENDPOINT_NAME}")
    
    try:
        # Parse S3 URI
        s3_parts = EVALUATION_DATA.replace('s3://', '').split('/')
        bucket_name = s3_parts[0]
        key_prefix = '/'.join(s3_parts[1:])
        
        # List evaluation data files
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=key_prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No evaluation data found at: {EVALUATION_DATA}")
            return {
                'statusCode': 404,
                'body': json.dumps('No evaluation data found')
            }
        
        # Get the latest evaluation data file
        latest_file = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
        latest_key = latest_file['Key']
        
        # Download evaluation data
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=latest_key
        )
        
        evaluation_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Prepare for batch inference
        test_data = evaluation_data.get('test_data', [])
        ground_truth = evaluation_data.get('ground_truth', [])
        
        if not test_data or not ground_truth:
            logger.warning("Invalid evaluation data format")
            return {
                'statusCode': 400,
                'body': json.dumps('Invalid evaluation data format')
            }
        
        # Perform batch inference
        predictions = []
        for data_point in test_data:
            try:
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='application/json',
                    Body=json.dumps(data_point)
                )
                
                result = json.loads(response['Body'].read().decode())
                predictions.append(result)
                
            except Exception as e:
                logger.error(f"Error invoking endpoint: {str(e)}")
                continue
        
        # Calculate performance metrics
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            if pred == truth:  # Simplified comparison, adjust based on your model output format
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Determine if retraining is needed
        retraining_needed = accuracy < PERFORMANCE_THRESHOLD
        
        if retraining_needed:
            logger.info(f"Performance below threshold. Triggering retraining pipeline: {PIPELINE_NAME}")
            
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
                        'Value': 'PerformanceDegradation'
                    },
                    {
                        'Name': 'CurrentAccuracy',
                        'Value': str(accuracy)
                    }
                ]
            )
            
            execution_arn = pipeline_response['PipelineExecutionArn']
            
            # Send notification if SNS topic is configured
            if SNS_TOPIC_ARN:
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Subject=f"Model Retraining Triggered for {ENDPOINT_NAME}",
                    Message=f"Model performance degradation detected for endpoint {ENDPOINT_NAME}. "
                            f"Current accuracy: {accuracy:.4f}, Threshold: {PERFORMANCE_THRESHOLD}. "
                            f"Retraining pipeline {PIPELINE_NAME} has been triggered. "
                            f"Pipeline execution ARN: {execution_arn}."
                )
            
            # Store evaluation results
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            evaluation_result = {
                'timestamp': timestamp,
                'endpoint_name': ENDPOINT_NAME,
                'accuracy': accuracy,
                'threshold': PERFORMANCE_THRESHOLD,
                'retraining_triggered': True,
                'pipeline_execution_arn': execution_arn
            }
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{key_prefix}/evaluation_results/{timestamp}.json",
                Body=json.dumps(evaluation_result)
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Retraining pipeline triggered',
                    'pipeline_name': PIPELINE_NAME,
                    'execution_arn': execution_arn,
                    'accuracy': accuracy,
                    'threshold': PERFORMANCE_THRESHOLD
                })
            }
        else:
            logger.info("Model performance above threshold. Retraining not needed.")
            
            # Store evaluation results
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            evaluation_result = {
                'timestamp': timestamp,
                'endpoint_name': ENDPOINT_NAME,
                'accuracy': accuracy,
                'threshold': PERFORMANCE_THRESHOLD,
                'retraining_triggered': False
            }
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{key_prefix}/evaluation_results/{timestamp}.json",
                Body=json.dumps(evaluation_result)
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Model performance satisfactory',
                    'accuracy': accuracy,
                    'threshold': PERFORMANCE_THRESHOLD
                })
            }
            
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }