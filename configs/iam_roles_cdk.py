#!/usr/bin/env python3
"""
CDK Stack for MLOps SageMaker Demo IAM Roles and Policies
Provides governance and role separation between Data Scientists and ML Engineers
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    Tags,
    aws_iam as iam,
)
from constructs import Construct


class MLOpsIAMStack(Stack):
    """CDK Stack for MLOps IAM roles and policies with governance controls"""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Parameters
        self.project_name = "mlops-sagemaker-demo"
        self.data_bucket_name = "lucaskle-ab3-project-pv"

        # Create IAM roles and policies
        self.data_scientist_role = self._create_data_scientist_role()
        self.ml_engineer_role = self._create_ml_engineer_role()
        self.sagemaker_execution_role = self._create_sagemaker_execution_role()
        self.sagemaker_studio_policy = self._create_sagemaker_studio_policy()

        # Add tags to all resources
        self._add_tags()

        # Create outputs
        self._create_outputs()

    def _create_data_scientist_role(self) -> iam.Role:
        """Create Data Scientist IAM role with restricted permissions"""
        
        # Create the role
        role = iam.Role(
            self,
            "DataScientistRole",
            role_name=f"{self.project_name}-DataScientist-Role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.AccountRootPrincipal()
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerReadOnly")
            ]
        )

        # Add custom policy for restricted access
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreatePresignedDomainUrl",
                    "sagemaker:CreatePresignedNotebookInstanceUrl",
                    "sagemaker:DescribeUserProfile",
                    "sagemaker:DescribeDomain",
                    "sagemaker:ListUserProfiles"
                ],
                resources=["*"]
            )
        )

        # Read-only S3 access to dataset
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation"
                ],
                resources=[
                    f"arn:aws:s3:::{self.data_bucket_name}",
                    f"arn:aws:s3:::{self.data_bucket_name}/*"
                ]
            )
        )

        # MLFlow experiment tracking access
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                resources=[
                    f"arn:aws:s3:::{self.project_name}-mlflow-artifacts",
                    f"arn:aws:s3:::{self.project_name}-mlflow-artifacts/*"
                ]
            )
        )

        # CloudWatch Logs for notebook instances
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams"
                ],
                resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"]
            )
        )

        # Explicit deny for production resources
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.DENY,
                actions=[
                    "sagemaker:CreateEndpoint*",
                    "sagemaker:UpdateEndpoint*",
                    "sagemaker:DeleteEndpoint*",
                    "sagemaker:CreateModel",
                    "sagemaker:DeleteModel",
                    "sagemaker:CreatePipeline",
                    "sagemaker:UpdatePipeline",
                    "sagemaker:DeletePipeline",
                    "sagemaker:StartPipelineExecution"
                ],
                resources=["*"]
            )
        )

        return role

    def _create_ml_engineer_role(self) -> iam.Role:
        """Create ML Engineer IAM role with full pipeline access"""
        
        role = iam.Role(
            self,
            "MLEngineerRole",
            role_name=f"{self.project_name}-MLEngineer-Role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.AccountRootPrincipal()
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )

        # Full SageMaker access
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:*"],
                resources=["*"]
            )
        )

        # EventBridge for pipeline notifications
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "events:PutEvents",
                    "events:PutRule",
                    "events:PutTargets",
                    "events:DeleteRule",
                    "events:RemoveTargets"
                ],
                resources=["*"]
            )
        )

        # CloudWatch for monitoring and logging
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:*",
                    "logs:*"
                ],
                resources=["*"]
            )
        )

        # Cost Explorer for cost monitoring
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ce:GetCostAndUsage",
                    "ce:GetUsageReport",
                    "ce:ListCostCategoryDefinitions"
                ],
                resources=["*"]
            )
        )

        return role

    def _create_sagemaker_execution_role(self) -> iam.Role:
        """Create SageMaker execution role for training jobs and pipelines"""
        
        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=f"{self.project_name}-SageMaker-Execution-Role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ]
        )

        # S3 access for all project buckets
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation"
                ],
                resources=[
                    f"arn:aws:s3:::{self.data_bucket_name}",
                    f"arn:aws:s3:::{self.data_bucket_name}/*",
                    f"arn:aws:s3:::{self.project_name}-*",
                    f"arn:aws:s3:::{self.project_name}-*/*"
                ]
            )
        )

        # ECR access for custom containers
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                resources=["*"]
            )
        )

        # CloudWatch Logs
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                resources=["*"]
            )
        )

        return role

    def _create_sagemaker_studio_policy(self) -> iam.ManagedPolicy:
        """Create SageMaker Studio policy with role-based restrictions"""
        
        policy = iam.ManagedPolicy(
            self,
            "SageMakerStudioPolicy",
            managed_policy_name=f"{self.project_name}-SageMaker-Studio-Policy",
            description="Policy for SageMaker Studio access with role-based restrictions",
            statements=[
                # Basic SageMaker Studio permissions
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:CreatePresignedDomainUrl",
                        "sagemaker:DescribeDomain",
                        "sagemaker:ListDomains",
                        "sagemaker:DescribeUserProfile",
                        "sagemaker:ListUserProfiles",
                        "sagemaker:DescribeApp",
                        "sagemaker:ListApps"
                    ],
                    resources=["*"]
                ),
                # Notebook instance permissions with instance type restrictions
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:CreateNotebookInstance",
                        "sagemaker:DescribeNotebookInstance",
                        "sagemaker:StartNotebookInstance",
                        "sagemaker:StopNotebookInstance",
                        "sagemaker:DeleteNotebookInstance"
                    ],
                    resources=["*"],
                    conditions={
                        "StringEquals": {
                            "sagemaker:InstanceTypes": [
                                "ml.t3.medium",
                                "ml.t3.large",
                                "ml.m5.large",
                                "ml.m5.xlarge"
                            ]
                        }
                    }
                ),
                # Resource tagging for governance
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:AddTags",
                        "sagemaker:ListTags"
                    ],
                    resources=["*"],
                    conditions={
                        "StringEquals": {
                            "sagemaker:TagKeys": [
                                "Project",
                                "Environment",
                                "Owner"
                            ]
                        }
                    }
                )
            ]
        )

        return policy

    def _add_tags(self) -> None:
        """Add tags to all resources in the stack"""
        Tags.of(self).add("Project", self.project_name)
        Tags.of(self).add("Environment", "Development")
        Tags.of(self).add("ManagedBy", "CDK")

    def _create_outputs(self) -> None:
        """Create CloudFormation outputs for the created resources"""
        
        CfnOutput(
            self,
            "DataScientistRoleArn",
            value=self.data_scientist_role.role_arn,
            description="ARN of the Data Scientist IAM Role",
            export_name=f"{self.project_name}-DataScientist-Role-Arn"
        )

        CfnOutput(
            self,
            "MLEngineerRoleArn",
            value=self.ml_engineer_role.role_arn,
            description="ARN of the ML Engineer IAM Role",
            export_name=f"{self.project_name}-MLEngineer-Role-Arn"
        )

        CfnOutput(
            self,
            "SageMakerExecutionRoleArn",
            value=self.sagemaker_execution_role.role_arn,
            description="ARN of the SageMaker Execution Role",
            export_name=f"{self.project_name}-SageMaker-Execution-Role-Arn"
        )

        CfnOutput(
            self,
            "SageMakerStudioPolicyArn",
            value=self.sagemaker_studio_policy.managed_policy_arn,
            description="ARN of the SageMaker Studio Policy",
            export_name=f"{self.project_name}-SageMaker-Studio-Policy-Arn"
        )


# CDK App definition
if __name__ == "__main__":
    from aws_cdk import App
    
    app = App()
    MLOpsIAMStack(app, "MLOpsIAMStack")
    app.synth()