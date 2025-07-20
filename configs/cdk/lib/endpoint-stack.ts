import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import { LambdaToSagemakerEndpoint } from '@aws-solutions-constructs/aws-lambda-sagemakerendpoint';

export interface EndpointStackProps extends cdk.StackProps {
  projectName: string;
  modelName: string;
  sagemakerRoleArn: string;
  lambdaCodePath: string;
}

export class EndpointStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: EndpointStackProps) {
    super(scope, id, props);

    // No need for S3 bucket since we're using inline code

    // Create IAM role for Lambda
    const lambdaRole = new iam.Role(this, 'LambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    // Add SageMaker permissions to Lambda role
    lambdaRole.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'sagemaker:CreateModel',
          'sagemaker:CreateEndpoint',
          'sagemaker:CreateEndpointConfig',
          'sagemaker:UpdateEndpoint',
          'sagemaker:DescribeEndpoint',
          'sagemaker:ListModelPackages',
          'sagemaker:DescribeModelPackage',
          'sagemaker:InvokeEndpoint',
          'application-autoscaling:RegisterScalableTarget',
          'application-autoscaling:PutScalingPolicy',
          'cloudwatch:PutMetricAlarm',
          'cloudwatch:DescribeAlarms',
          'iam:PassRole',
        ],
        resources: ['*'],
      })
    );

    // Create Lambda function for endpoint deployment
    const deployEndpointLambda = new lambda.Function(this, 'DeployEndpointLambda', {
      runtime: lambda.Runtime.PYTHON_3_10,
      handler: 'deploy_endpoint_lambda.lambda_handler',
      code: lambda.Code.fromInline(`
def lambda_handler(event, context):
    """
    Default Lambda handler for SageMaker endpoint invocation.
    """
    import boto3
    import os
    import json
    
    # Get the SageMaker endpoint name from environment variable
    endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
    
    # Create SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    # Log the event for debugging
    print(f"Received event: {json.dumps(event)}")
    
    # Extract payload from event
    payload = event.get('body', '{}')
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    
    # Invoke SageMaker endpoint
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = response['Body'].read().decode('utf-8')
        return {
            'statusCode': 200,
            'body': result,
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        print(f"Error invoking endpoint: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
`),
      timeout: cdk.Duration.minutes(10),
      memorySize: 256,
      role: lambdaRole,
      environment: {
        MODEL_NAME: props.modelName,
        SAGEMAKER_ROLE_ARN: props.sagemakerRoleArn,
      },
    });

    // Create Lambda-SageMaker endpoint construct
    const lambdaToSagemakerEndpoint = new LambdaToSagemakerEndpoint(
      this,
      'LambdaToSagemakerEndpoint',
      {
        existingLambdaObj: deployEndpointLambda,
        modelProps: {
          modelName: `${props.modelName}-model`,
          executionRoleArn: props.sagemakerRoleArn,
          primaryContainer: {
            image: `${cdk.Aws.ACCOUNT_ID}.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com/${props.modelName}:latest`,
            modelDataUrl: `s3://${props.projectName}-models/${props.modelName}/model.tar.gz`,
            environment: {
              ENABLE_CLOUDWATCH_METRICS: 'true',
              ENABLE_METRICS_EMISSION: 'true',
            }
          }
        },
        endpointProps: {
          endpointName: `${props.modelName}-endpoint`,
          endpointConfigName: `${props.modelName}-config`
        },
        endpointConfigProps: {
          productionVariants: [
            {
              initialInstanceCount: 1,
              instanceType: 'ml.m5.large',
              modelName: `${props.modelName}-model`,
              variantName: 'AllTraffic'
            }
          ]
        }
      }
    );

    // Output the Lambda function ARN
    new cdk.CfnOutput(this, 'DeployEndpointLambdaArn', {
      value: deployEndpointLambda.functionArn,
      description: 'ARN of the Lambda function for deploying SageMaker endpoints',
    });

    // Output the SageMaker endpoint name
    new cdk.CfnOutput(this, 'SageMakerEndpointName', {
      value: `${props.modelName}-endpoint`,
      description: 'Name of the SageMaker endpoint',
    });
  }
}