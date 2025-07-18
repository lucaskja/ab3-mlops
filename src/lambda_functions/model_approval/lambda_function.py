#!/usr/bin/env python3
"""
Model Approval Workflow Lambda Function

This Lambda function handles model approval workflows, including:
- Processing pipeline execution status changes
- Processing model package status changes
- Handling approval/rejection responses from users
"""

import json
import boto3
import os
import logging
import urllib.parse

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
sagemaker_client = boto3.client('sagemaker')
sns_client = boto3.client('sns')

# Configuration - these values will be replaced during deployment
PIPELINE_NAME = "PIPELINE_NAME_PLACEHOLDER"
SNS_TOPIC_ARN = "SNS_TOPIC_ARN_PLACEHOLDER"
REGION = "REGION_PLACEHOLDER"

def lambda_handler(event, context):
    """Lambda handler for model update approval workflow."""
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Check if this is a pipeline execution status change event
        if event.get('source') == 'aws.sagemaker' and event.get('detail-type') == 'SageMaker Model Building Pipeline Execution Status Change':
            return handle_pipeline_status_change(event)
        
        # Check if this is a model package status change event
        elif event.get('source') == 'aws.sagemaker' and event.get('detail-type') == 'SageMaker Model Package State Change':
            return handle_model_package_status_change(event)
        
        # Check if this is an approval response
        elif 'queryStringParameters' in event:
            return handle_approval_response(event)
        
        else:
            logger.warning("Unknown event type")
            return {
                'statusCode': 400,
                'body': json.dumps('Unknown event type')
            }
    
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def handle_pipeline_status_change(event):
    """
    Handle SageMaker pipeline execution status change events.
    """
    detail = event.get('detail', {})
    pipeline_name = detail.get('pipelineName', '')
    execution_status = detail.get('currentPipelineExecutionStatus', '')
    
    if pipeline_name != PIPELINE_NAME:
        logger.info(f"Event for different pipeline: {pipeline_name}")
        return {
            'statusCode': 200,
            'body': json.dumps('Event for different pipeline')
        }
    
    if execution_status == 'Succeeded':
        logger.info(f"Pipeline execution succeeded: {pipeline_name}")
        
        # Get the model package group name from the pipeline
        pipeline_desc = sagemaker_client.describe_pipeline(PipelineName=pipeline_name)
        pipeline_definition = json.loads(pipeline_desc['PipelineDefinition'])
        
        # Extract model package group name (simplified, would need to parse the pipeline definition)
        model_package_group_name = f"{pipeline_name}-models"
        
        # Get the latest model package
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        model_packages = response.get('ModelPackageSummaryList', [])
        if not model_packages:
            logger.warning(f"No model packages found for group: {model_package_group_name}")
            return {
                'statusCode': 404,
                'body': json.dumps('No model packages found')
            }
        
        model_package_arn = model_packages[0]['ModelPackageArn']
        
        # Send approval notification
        if SNS_TOPIC_ARN:
            # Create approval/rejection URLs (these would point to an API Gateway endpoint that triggers this Lambda)
            api_gateway_url = f"https://example.execute-api.{REGION}.amazonaws.com/prod/model-approval"
            approve_url = f"{api_gateway_url}?action=approve&model_package_arn={urllib.parse.quote(model_package_arn)}"
            reject_url = f"{api_gateway_url}?action=reject&model_package_arn={urllib.parse.quote(model_package_arn)}"
            
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Model Approval Required for {pipeline_name}",
                Message=f"A new model has been created by pipeline {pipeline_name} and requires approval. "
                        f"\n\nModel Package ARN: {model_package_arn} "
                        f"\n\nTo approve this model, click: {approve_url} "
                        f"\n\nTo reject this model, click: {reject_url} "
                        f"\n\nThis approval link will expire in 7 days."
            )
            
            logger.info(f"Sent approval notification for model package: {model_package_arn}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Pipeline execution succeeded',
                'pipeline_name': pipeline_name,
                'model_package_arn': model_package_arn
            })
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Pipeline execution status: {execution_status}',
            'pipeline_name': pipeline_name
        })
    }

def handle_model_package_status_change(event):
    """
    Handle SageMaker model package status change events.
    """
    detail = event.get('detail', {})
    model_package_arn = detail.get('ModelPackageArn', '')
    model_approval_status = detail.get('ModelApprovalStatus', '')
    
    logger.info(f"Model package status changed: {model_package_arn} -> {model_approval_status}")
    
    if model_approval_status == 'Approved':
        # Get model package details
        response = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        
        # Send notification
        if SNS_TOPIC_ARN:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Model Approved: {model_package_arn.split('/')[-1]}",
                Message=f"The model package {model_package_arn} has been approved. "
                        f"\n\nThe model will now be deployed to the endpoint."
            )
            
            logger.info(f"Sent approval confirmation for model package: {model_package_arn}")
    
    elif model_approval_status == 'Rejected':
        # Send notification
        if SNS_TOPIC_ARN:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Model Rejected: {model_package_arn.split('/')[-1]}",
                Message=f"The model package {model_package_arn} has been rejected. "
                        f"\n\nNo further action will be taken."
            )
            
            logger.info(f"Sent rejection confirmation for model package: {model_package_arn}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Model package status: {model_approval_status}',
            'model_package_arn': model_package_arn
        })
    }

def handle_approval_response(event):
    """
    Handle approval response from the API Gateway.
    """
    query_params = event.get('queryStringParameters', {})
    action = query_params.get('action', '')
    model_package_arn = query_params.get('model_package_arn', '')
    
    if not action or not model_package_arn:
        return {
            'statusCode': 400,
            'body': json.dumps('Missing required parameters')
        }
    
    if action == 'approve':
        # Update model package approval status
        sagemaker_client.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus='Approved'
        )
        
        logger.info(f"Model package approved: {model_package_arn}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Model approved successfully')
        }
    
    elif action == 'reject':
        # Update model package approval status
        sagemaker_client.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus='Rejected'
        )
        
        logger.info(f"Model package rejected: {model_package_arn}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Model rejected successfully')
        }
    
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid action')
        }