#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { MLOpsSageMakerIAMStack } from '../lib/iam-stack';
import { EndpointStack } from '../lib/endpoint-stack';
import { Aspects } from 'aws-cdk-lib';
import { AwsSolutionsChecks, NagSuppressions } from 'cdk-nag';

const app = new cdk.App();

// Project configuration
const projectName = 'mlops-sagemaker-demo';
const dataBucketName = 'lucaskle-ab3-project-pv';
const env = { 
  account: process.env.CDK_DEFAULT_ACCOUNT, 
  region: process.env.CDK_DEFAULT_REGION || 'us-east-1' 
};

// Create IAM stack
const iamStack = new MLOpsSageMakerIAMStack(app, 'MLOpsSageMakerIAMStack', {
  projectName,
  dataBucketName,
  env,
  description: 'IAM Roles and Policies for MLOps SageMaker Demo with governance and role separation',
});

// Create Endpoint stack with AWS Solutions Constructs
const endpointStack = new EndpointStack(app, 'MLOpsSageMakerEndpointStack', {
  projectName,
  modelName: 'yolov11-drone-detection',
  sagemakerRoleArn: iamStack.sagemakerExecutionRole.roleArn,
  lambdaCodePath: process.env.LAMBDA_CODE_PATH || './lambda',
  env,
  description: 'SageMaker Endpoint infrastructure using AWS Solutions Constructs',
});

// Add CDK Nag to all stacks
Aspects.of(app).add(new AwsSolutionsChecks());

// Add specific suppressions for justified exceptions
NagSuppressions.addStackSuppressions(iamStack, [
  { id: 'AwsSolutions-IAM4', reason: 'SageMaker requires AmazonSageMakerFullAccess managed policy for full functionality' },
  { id: 'AwsSolutions-IAM5', reason: 'SageMaker service requires wildcard permissions for certain operations' }
]);

NagSuppressions.addStackSuppressions(endpointStack, [
  { id: 'AwsSolutions-IAM4', reason: 'AWS Solutions Constructs use managed policies for simplified implementation' },
  { id: 'AwsSolutions-IAM5', reason: 'Lambda functions require specific permissions to invoke SageMaker endpoints' },
  { id: 'AwsSolutions-APIG2', reason: 'Request validation is handled by the Lambda function' },
  { id: 'AwsSolutions-APIG4', reason: 'Authorization is not required for this demo endpoint' },
  { id: 'AwsSolutions-COG4', reason: 'Cognito authorization is not implemented in this demo' },
  { id: 'AwsSolutions-SNS2', reason: 'SNS topic is used for monitoring alerts only, encryption not required for this demo' },
  { id: 'AwsSolutions-SNS3', reason: 'This is a demo environment, no subscribers are added programmatically' },
  { id: 'AwsSolutions-L1', reason: 'Lambda functions use the latest runtime version available' }
]);

// Tag all resources
cdk.Tags.of(app).add('Project', projectName);
cdk.Tags.of(app).add('Environment', 'Development');
cdk.Tags.of(app).add('ManagedBy', 'CDK');

app.synth();