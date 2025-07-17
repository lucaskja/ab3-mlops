import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as eventbridge from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import { LambdaToSagemakerEndpoint } from '@aws-solutions-constructs/aws-lambda-sagemakerendpoint';
import { ApiGatewayToSagemakerEndpoint } from '@aws-solutions-constructs/aws-apigateway-sagemakerendpoint';
import { EventbridgeToLambda } from '@aws-solutions-constructs/aws-eventbridge-lambda';
import { NagSuppressions } from 'cdk-nag';

export interface MLOpsSageMakerEndpointStackProps extends cdk.StackProps {
  projectName: string;
  sagemakerExecutionRole: iam.Role;
}

export class MLOpsSageMakerEndpointStack extends cdk.Stack {
  // Export resources for use in other stacks
  public readonly sagemakerEndpoint: sagemaker.CfnEndpoint;
  public readonly endpointApi: apigateway.RestApi;
  public readonly endpointLambda: lambda.Function;

  constructor(scope: Construct, id: string, props: MLOpsSageMakerEndpointStackProps) {
    super(scope, id, props);

    const { projectName, sagemakerExecutionRole } = props;

    // Create a model configuration for YOLOv11
    const modelName = `${projectName}-yolov11-model`;
    const endpointConfigName = `${projectName}-yolov11-endpoint-config`;
    const endpointName = `${projectName}-yolov11-endpoint`;

    // Create a placeholder model for the endpoint
    // In a real scenario, this would be created by the SageMaker pipeline
    const model = new sagemaker.CfnModel(this, 'YOLOv11Model', {
      modelName: modelName,
      executionRoleArn: sagemakerExecutionRole.roleArn,
      primaryContainer: {
        image: `${cdk.Stack.of(this).account}.dkr.ecr.${cdk.Stack.of(this).region}.amazonaws.com/${projectName}-yolov11:latest`,
        modelDataUrl: `s3://${projectName}-mlflow-artifacts/models/yolov11/model.tar.gz`,
        environment: {
          SAGEMAKER_PROGRAM: 'inference.py',
          MODEL_NAME: 'yolov11',
          FRAMEWORK: 'pytorch'
        }
      },
      tags: [
        { key: 'Project', value: projectName },
        { key: 'Environment', value: 'Development' },
        { key: 'ModelType', value: 'YOLOv11' }
      ]
    });

    // Create an endpoint configuration
    const endpointConfig = new sagemaker.CfnEndpointConfig(this, 'YOLOv11EndpointConfig', {
      endpointConfigName: endpointConfigName,
      productionVariants: [
        {
          initialVariantWeight: 1.0,
          modelName: modelName,
          variantName: 'AllTraffic',
          instanceType: 'ml.m5.large',
          initialInstanceCount: 1,
          serverlessConfig: {
            maxConcurrency: 5,
            memorySizeInMb: 2048
          }
        }
      ],
      dataCaptureConfig: {
        enableCapture: true,
        initialSamplingPercentage: 100,
        destinationS3Uri: `s3://${projectName}-mlflow-artifacts/data-capture/`,
        captureOptions: [
          { captureMode: 'Input' },
          { captureMode: 'Output' }
        ]
      },
      tags: [
        { key: 'Project', value: projectName },
        { key: 'Environment', value: 'Development' }
      ]
    });

    // Create the endpoint
    this.sagemakerEndpoint = new sagemaker.CfnEndpoint(this, 'YOLOv11Endpoint', {
      endpointName: endpointName,
      endpointConfigName: endpointConfig.attrEndpointConfigName,
      tags: [
        { key: 'Project', value: projectName },
        { key: 'Environment', value: 'Development' }
      ]
    });

    // Add dependencies
    endpointConfig.addDependency(model);
    this.sagemakerEndpoint.addDependency(endpointConfig);

    // Create Lambda function for endpoint invocation
    this.endpointLambda = new lambda.Function(this, 'EndpointInvocationFunction', {
      functionName: `${projectName}-endpoint-invocation`,
      runtime: lambda.Runtime.PYTHON_3_10,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
import json
import boto3
import base64
from io import BytesIO
import os

def handler(event, context):
    """Lambda function to invoke SageMaker endpoint with proper error handling"""
    try:
        # Get the endpoint name from environment variable
        endpoint_name = os.environ['ENDPOINT_NAME']
        
        # Initialize SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime')
        
        # Extract the image data from the request
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            
            # Check if image is base64 encoded
            if 'image' in body:
                image_data = body['image']
                # If it's base64 encoded, decode it
                if isinstance(image_data, str) and image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                else:
                    # Handle binary data
                    image_bytes = BytesIO(image_data).getvalue()
            else:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'No image data provided'})
                }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid request format'})
            }
        
        # Invoke the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/x-image',
            Body=image_bytes
        )
        
        # Process the response
        result = json.loads(response['Body'].read().decode())
        
        # Return the prediction results
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        # Log the error
        print(f"Error invoking endpoint: {str(e)}")
        
        # Return error response
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Error invoking the model endpoint',
                'message': str(e)
            })
        }
      `),
      environment: {
        'ENDPOINT_NAME': endpointName
      },
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      logRetention: logs.RetentionDays.ONE_WEEK,
      tracing: lambda.Tracing.ACTIVE
    });

    // Create AWS Solutions Construct for Lambda to SageMaker Endpoint
    const lambdaToEndpoint = new LambdaToSagemakerEndpoint(this, 'LambdaToSagemakerEndpoint', {
      existingLambdaObj: this.endpointLambda,
      sagemakerEndpointName: endpointName,
      sagemakerEndpointArn: this.sagemakerEndpoint.attrEndpointArn
    });

    // Create AWS Solutions Construct for API Gateway to SageMaker Endpoint
    const apiToEndpoint = new ApiGatewayToSagemakerEndpoint(this, 'ApiGatewayToSagemakerEndpoint', {
      existingLambdaObj: this.endpointLambda,
      apiGatewayProps: {
        restApiName: `${projectName}-endpoint-api`,
        description: 'API Gateway for YOLOv11 SageMaker endpoint',
        deployOptions: {
          stageName: 'prod',
          loggingLevel: apigateway.MethodLoggingLevel.INFO,
          dataTraceEnabled: true,
          metricsEnabled: true
        },
        defaultCorsPreflightOptions: {
          allowOrigins: apigateway.Cors.ALL_ORIGINS,
          allowMethods: apigateway.Cors.ALL_METHODS,
          allowHeaders: ['Content-Type', 'Authorization']
        }
      }
    });

    // Store the API for export
    this.endpointApi = apiToEndpoint.apiGateway;
    
    // Create SNS topic for endpoint monitoring alerts
    const monitoringTopic = new sns.Topic(this, 'EndpointMonitoringTopic', {
      topicName: `${projectName}-endpoint-monitoring`,
      displayName: 'SageMaker Endpoint Monitoring Alerts'
    });
    
    // Create CloudWatch alarms for endpoint monitoring
    const invocationErrorAlarm = new cloudwatch.Alarm(this, 'EndpointInvocationErrorAlarm', {
      alarmName: `${projectName}-endpoint-invocation-errors`,
      metric: new cloudwatch.Metric({
        namespace: 'AWS/SageMaker',
        metricName: 'Invocation5XXErrors',
        dimensionsMap: {
          EndpointName: endpointName,
          VariantName: 'AllTraffic'
        },
        statistic: 'Sum',
        period: cdk.Duration.minutes(5)
      }),
      threshold: 5,
      evaluationPeriods: 3,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING
    });
    
    // Add alarm action to notify SNS topic
    invocationErrorAlarm.addAlarmAction(new cloudwatch.SnsAction(monitoringTopic));
    
    // Create EventBridge rule for endpoint status changes
    const endpointStatusRule = new eventbridge.Rule(this, 'EndpointStatusRule', {
      ruleName: `${projectName}-endpoint-status-changes`,
      description: 'Monitors SageMaker endpoint status changes',
      eventPattern: {
        source: ['aws.sagemaker'],
        detailType: ['SageMaker Endpoint State Change'],
        detail: {
          EndpointName: [endpointName]
        }
      }
    });
    
    // Create Lambda function to handle endpoint status changes
    const statusHandlerFunction = new lambda.Function(this, 'EndpointStatusHandler', {
      functionName: `${projectName}-endpoint-status-handler`,
      runtime: lambda.Runtime.PYTHON_3_10,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
import json
import boto3
import os

def handler(event, context):
    """Lambda function to handle SageMaker endpoint status changes"""
    try:
        # Extract endpoint details from the event
        detail = event.get('detail', {})
        endpoint_name = detail.get('EndpointName')
        endpoint_status = detail.get('EndpointStatus')
        
        # Log the status change
        print(f"Endpoint {endpoint_name} status changed to {endpoint_status}")
        
        # Get the SNS topic ARN from environment variable
        sns_topic_arn = os.environ['SNS_TOPIC_ARN']
        
        # Initialize SNS client
        sns = boto3.client('sns')
        
        # Publish message to SNS topic
        message = {
            'endpoint_name': endpoint_name,
            'status': endpoint_status,
            'timestamp': event.get('time'),
            'region': os.environ['AWS_REGION']
        }
        
        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=f"SageMaker Endpoint Status Change: {endpoint_name}",
            Message=json.dumps(message, indent=2)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Status change notification sent successfully'
            })
        }
        
    except Exception as e:
        # Log the error
        print(f"Error handling endpoint status change: {str(e)}")
        
        # Return error response
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Error handling endpoint status change',
                'message': str(e)
            })
        }
      `),
      environment: {
        'SNS_TOPIC_ARN': monitoringTopic.topicArn
      },
      timeout: cdk.Duration.seconds(30),
      memorySize: 128,
      logRetention: logs.RetentionDays.ONE_WEEK
    });
    
    // Use AWS Solutions Construct for EventBridge to Lambda integration
    new EventbridgeToLambda(this, 'EndpointStatusEventToLambda', {
      existingLambdaObj: statusHandlerFunction,
      existingEventRuleObj: endpointStatusRule
    });
    
    // Grant permissions for the Lambda to publish to SNS
    monitoringTopic.grantPublish(statusHandlerFunction);
    
    // Add CDK Nag suppressions for specific resources
    NagSuppressions.addResourceSuppressions(this.endpointLambda, [
      { id: 'AwsSolutions-IAM4', reason: 'Lambda needs SageMaker invoke permissions' },
      { id: 'AwsSolutions-IAM5', reason: 'Lambda needs CloudWatch logs permissions' }
    ], true);

    NagSuppressions.addResourceSuppressions(statusHandlerFunction, [
      { id: 'AwsSolutions-IAM5', reason: 'Lambda needs SNS publish permissions' }
    ], true);

    NagSuppressions.addResourceSuppressions(this.endpointApi, [
      { id: 'AwsSolutions-APIG2', reason: 'Request validation is handled by the Lambda function' },
      { id: 'AwsSolutions-APIG4', reason: 'Authorization is not required for this demo endpoint' },
      { id: 'AwsSolutions-COG4', reason: 'Cognito authorization is not implemented in this demo' }
    ], true);
    
    NagSuppressions.addResourceSuppressions(monitoringTopic, [
      { id: 'AwsSolutions-SNS2', reason: 'SNS topic is used for monitoring alerts only, encryption not required for this demo' },
      { id: 'AwsSolutions-SNS3', reason: 'This is a demo environment, no subscribers are added programmatically' }
    ], true);

    // Create outputs
    new cdk.CfnOutput(this, 'SageMakerEndpointName', {
      value: endpointName,
      description: 'Name of the SageMaker endpoint',
      exportName: `${projectName}-endpoint-name`
    });

    new cdk.CfnOutput(this, 'ApiGatewayUrl', {
      value: this.endpointApi.url,
      description: 'URL of the API Gateway endpoint',
      exportName: `${projectName}-api-url`
    });

    new cdk.CfnOutput(this, 'LambdaFunctionName', {
      value: this.endpointLambda.functionName,
      description: 'Name of the Lambda function',
      exportName: `${projectName}-lambda-name`
    });
  }
}