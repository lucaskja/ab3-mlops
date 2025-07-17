import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import { NagSuppressions } from 'cdk-nag';

export interface MLOpsSageMakerIAMStackProps extends cdk.StackProps {
  projectName: string;
  dataBucketName: string;
}

export class MLOpsSageMakerIAMStack extends cdk.Stack {
  // Export roles for use in other stacks
  public readonly dataScientistRole: iam.Role;
  public readonly mlEngineerRole: iam.Role;
  public readonly sagemakerExecutionRole: iam.Role;
  public readonly sagemakerStudioPolicy: iam.ManagedPolicy;

  constructor(scope: Construct, id: string, props: MLOpsSageMakerIAMStackProps) {
    super(scope, id, props);

    const { projectName, dataBucketName } = props;

    // Create Data Scientist Role with restricted permissions
    this.dataScientistRole = new iam.Role(this, 'DataScientistRole', {
      roleName: `${projectName}-DataScientist-Role`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('sagemaker.amazonaws.com'),
        new iam.AccountRootPrincipal()
      ),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerReadOnly')
      ],
      description: 'Role for Data Scientists with restricted permissions'
    });

    // Add custom policy for Data Scientist role
    this.dataScientistRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'sagemaker:CreatePresignedDomainUrl',
          'sagemaker:CreatePresignedNotebookInstanceUrl',
          'sagemaker:DescribeUserProfile',
          'sagemaker:DescribeDomain',
          'sagemaker:ListUserProfiles'
        ],
        resources: ['*']
      })
    );

    // Read-only S3 access to dataset
    this.dataScientistRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:GetObject',
          's3:ListBucket',
          's3:GetBucketLocation'
        ],
        resources: [
          `arn:aws:s3:::${dataBucketName}`,
          `arn:aws:s3:::${dataBucketName}/*`
        ]
      })
    );

    // MLFlow experiment tracking access
    this.dataScientistRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:GetObject',
          's3:PutObject',
          's3:DeleteObject',
          's3:ListBucket'
        ],
        resources: [
          `arn:aws:s3:::${projectName}-mlflow-artifacts`,
          `arn:aws:s3:::${projectName}-mlflow-artifacts/*`
        ]
      })
    );

    // CloudWatch Logs for notebook instances
    this.dataScientistRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          'logs:DescribeLogStreams'
        ],
        resources: [`arn:aws:logs:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:log-group:/aws/sagemaker/*`]
      })
    );

    // Explicit deny for production resources
    this.dataScientistRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.DENY,
        actions: [
          'sagemaker:CreateEndpoint*',
          'sagemaker:UpdateEndpoint*',
          'sagemaker:DeleteEndpoint*',
          'sagemaker:CreateModel',
          'sagemaker:DeleteModel',
          'sagemaker:CreatePipeline',
          'sagemaker:UpdatePipeline',
          'sagemaker:DeletePipeline',
          'sagemaker:StartPipelineExecution'
        ],
        resources: ['*']
      })
    );

    // Create ML Engineer Role with full pipeline access
    this.mlEngineerRole = new iam.Role(this, 'MLEngineerRole', {
      roleName: `${projectName}-MLEngineer-Role`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('sagemaker.amazonaws.com'),
        new iam.AccountRootPrincipal()
      ),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess')
      ],
      description: 'Role for ML Engineers with full pipeline access'
    });

    // Full SageMaker access
    this.mlEngineerRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['sagemaker:*'],
        resources: ['*']
      })
    );

    // EventBridge for pipeline notifications
    this.mlEngineerRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'events:PutEvents',
          'events:PutRule',
          'events:PutTargets',
          'events:DeleteRule',
          'events:RemoveTargets'
        ],
        resources: ['*']
      })
    );

    // CloudWatch for monitoring and logging
    this.mlEngineerRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'cloudwatch:*',
          'logs:*'
        ],
        resources: ['*']
      })
    );

    // IAM PassRole for SageMaker execution roles
    this.mlEngineerRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['iam:PassRole'],
        resources: ['*'],
        conditions: {
          StringEquals: {
            'iam:PassedToService': 'sagemaker.amazonaws.com'
          }
        }
      })
    );

    // Cost Explorer for cost monitoring
    this.mlEngineerRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'ce:GetCostAndUsage',
          'ce:GetUsageReport',
          'ce:ListCostCategoryDefinitions'
        ],
        resources: ['*']
      })
    );

    // Create SageMaker Execution Role for training jobs and pipelines
    this.sagemakerExecutionRole = new iam.Role(this, 'SageMakerExecutionRole', {
      roleName: `${projectName}-SageMaker-Execution-Role`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess')
      ],
      description: 'Execution role for SageMaker training jobs and pipelines'
    });

    // S3 access for all project buckets
    this.sagemakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:GetObject',
          's3:PutObject',
          's3:DeleteObject',
          's3:ListBucket',
          's3:GetBucketLocation'
        ],
        resources: [
          `arn:aws:s3:::${dataBucketName}`,
          `arn:aws:s3:::${dataBucketName}/*`,
          `arn:aws:s3:::${projectName}-*`,
          `arn:aws:s3:::${projectName}-*/*`
        ]
      })
    );

    // ECR access for custom containers
    this.sagemakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'ecr:GetAuthorizationToken',
          'ecr:BatchCheckLayerAvailability',
          'ecr:GetDownloadUrlForLayer',
          'ecr:BatchGetImage'
        ],
        resources: ['*']
      })
    );

    // CloudWatch Logs
    this.sagemakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents'
        ],
        resources: ['*']
      })
    );

    // Create SageMaker Studio Policy with role-based restrictions
    this.sagemakerStudioPolicy = new iam.ManagedPolicy(this, 'SageMakerStudioPolicy', {
      managedPolicyName: `${projectName}-SageMaker-Studio-Policy`,
      description: 'Policy for SageMaker Studio access with role-based restrictions',
      statements: [
        // Basic SageMaker Studio permissions
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'sagemaker:CreatePresignedDomainUrl',
            'sagemaker:DescribeDomain',
            'sagemaker:ListDomains',
            'sagemaker:DescribeUserProfile',
            'sagemaker:ListUserProfiles',
            'sagemaker:DescribeApp',
            'sagemaker:ListApps'
          ],
          resources: ['*']
        }),
        // Notebook instance permissions with instance type restrictions
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'sagemaker:CreateNotebookInstance',
            'sagemaker:DescribeNotebookInstance',
            'sagemaker:StartNotebookInstance',
            'sagemaker:StopNotebookInstance',
            'sagemaker:DeleteNotebookInstance'
          ],
          resources: ['*'],
          conditions: {
            StringEquals: {
              'sagemaker:InstanceTypes': [
                'ml.t3.medium',
                'ml.t3.large',
                'ml.m5.large',
                'ml.m5.xlarge'
              ]
            }
          }
        }),
        // Resource tagging for governance
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'sagemaker:AddTags',
            'sagemaker:ListTags'
          ],
          resources: ['*'],
          conditions: {
            StringEquals: {
              'sagemaker:TagKeys': [
                'Project',
                'Environment',
                'Owner'
              ]
            }
          }
        })
      ]
    });

    // Add CDK Nag suppressions for specific resources
    NagSuppressions.addResourceSuppressions(this.dataScientistRole, [
      { id: 'AwsSolutions-IAM5', reason: 'SageMaker requires wildcard permissions for certain operations' }
    ], true);

    NagSuppressions.addResourceSuppressions(this.mlEngineerRole, [
      { id: 'AwsSolutions-IAM4', reason: 'SageMaker requires AmazonSageMakerFullAccess managed policy for full functionality' },
      { id: 'AwsSolutions-IAM5', reason: 'ML Engineers need broad permissions to manage SageMaker resources' }
    ], true);

    NagSuppressions.addResourceSuppressions(this.sagemakerExecutionRole, [
      { id: 'AwsSolutions-IAM4', reason: 'SageMaker requires AmazonSageMakerFullAccess managed policy for full functionality' },
      { id: 'AwsSolutions-IAM5', reason: 'SageMaker execution role needs CloudWatch and ECR permissions' }
    ], true);

    // Create outputs
    new cdk.CfnOutput(this, 'DataScientistRoleArn', {
      value: this.dataScientistRole.roleArn,
      description: 'ARN of the Data Scientist IAM Role',
      exportName: `${projectName}-DataScientist-Role-Arn`
    });

    new cdk.CfnOutput(this, 'MLEngineerRoleArn', {
      value: this.mlEngineerRole.roleArn,
      description: 'ARN of the ML Engineer IAM Role',
      exportName: `${projectName}-MLEngineer-Role-Arn`
    });

    new cdk.CfnOutput(this, 'SageMakerExecutionRoleArn', {
      value: this.sagemakerExecutionRole.roleArn,
      description: 'ARN of the SageMaker Execution Role',
      exportName: `${projectName}-SageMaker-Execution-Role-Arn`
    });

    new cdk.CfnOutput(this, 'SageMakerStudioPolicyArn', {
      value: this.sagemakerStudioPolicy.managedPolicyArn,
      description: 'ARN of the SageMaker Studio Policy',
      exportName: `${projectName}-SageMaker-Studio-Policy-Arn`
    });
  }
}