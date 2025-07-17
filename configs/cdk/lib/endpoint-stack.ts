import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as path from 'path';
import * as fs from 'fs';
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

    // Create S3 bucket for Lambda code
    const lambdaCodeBucket = new s3.Bucket(this, 'LambdaCodeBucket', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      versioned: true,
    });

    // Upload Lambda code to S3
    new s3deploy.BucketDeployment(this, 'DeployLambdaCode', {
      sources: [s3deploy.Source.asset(props.lambdaCodePath)],
      destinationBucket: lambdaCodeBucket,
      destinationKeyPrefix: 'lambda',
    });

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
      code: lambda.Code.fromBucket(
        lambdaCodeBucket,
        'lambda/deploy_endpoint_lambda.py'
      ),
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
        sagemakerEndpointProps: {
          modelName: `${props.modelName}-model`,
          endpointName: `${props.modelName}-endpoint`,
          instanceType: 'ml.m5.large',
          initialInstanceCount: 1,
          environment: {
            ENABLE_CLOUDWATCH_METRICS: 'true',
            ENABLE_METRICS_EMISSION: 'true',
          },
        },
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