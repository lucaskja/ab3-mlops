# IAM Roles and Policies Setup Guide

This directory contains the infrastructure-as-code templates and scripts for setting up IAM roles and policies that provide governance and role separation for the MLOps SageMaker Demo.

## Overview

The IAM setup implements a role-based access control (RBAC) system with three distinct roles:

1. **Data Scientist Role** - Restricted permissions for data exploration and model development
2. **ML Engineer Role** - Full permissions for pipeline deployment and production management  
3. **SageMaker Execution Role** - Service role for running training jobs and pipelines

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IAM Governance Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Data Scientist │    │   ML Engineer   │    │  SageMaker   │ │
│  │      Role       │    │      Role       │    │ Execution    │ │
│  │                 │    │                 │    │    Role      │ │
│  │ • Read-only S3  │    │ • Full SageMaker│    │ • Service    │ │
│  │ • Studio access │    │ • Pipeline mgmt │    │   role for   │ │
│  │ • MLFlow exp.   │    │ • Deployment    │    │   training   │ │
│  │ • NO production │    │ • Monitoring    │    │   jobs       │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                     │       │
│           └───────────────────────┼─────────────────────┘       │
│                                   │                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              SageMaker Studio Policy                        │ │
│  │  • Instance type restrictions                               │ │
│  │  • Resource tagging requirements                            │ │
│  │  • Cost optimization controls                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Files Description

### Infrastructure Templates

- **`iam-roles-cloudformation.yaml`** - CloudFormation template defining all IAM roles and policies
- **`iam_roles_cdk.py`** - AWS CDK equivalent for users preferring CDK over CloudFormation

### Deployment Scripts

- **`../scripts/setup/deploy_iam_roles.sh`** - Automated deployment script for CloudFormation stack
- **`../scripts/setup/validate_iam_roles.py`** - Comprehensive validation script for testing role permissions

### Configuration

- **`project_config.py`** - Python configuration file with project settings and role ARNs

## Quick Start

### Prerequisites

1. AWS CLI installed and configured with the "ab" profile
2. Python 3.8+ with boto3 installed
3. Appropriate AWS permissions to create IAM roles and policies

### Deployment

1. **Deploy using the automated script (Recommended):**
   ```bash
   ./scripts/setup/deploy_iam_roles.sh
   ```

2. **Deploy using AWS CLI directly:**
   ```bash
   aws cloudformation create-stack \
     --stack-name mlops-sagemaker-demo-iam-roles \
     --template-body file://configs/iam-roles-cloudformation.yaml \
     --parameters ParameterKey=ProjectName,ParameterValue=mlops-sagemaker-demo \
                  ParameterKey=DataBucketName,ParameterValue=lucaskle-ab3-project-pv \
     --capabilities CAPABILITY_NAMED_IAM \
     --profile ab
   ```

3. **Deploy using CDK:**
   ```bash
   cd configs
   pip install aws-cdk-lib constructs
   python iam_roles_cdk.py
   ```

### Validation

After deployment, validate the roles and permissions:

```bash
python3 scripts/setup/validate_iam_roles.py --profile ab
```

## Role Details

### Data Scientist Role

**Purpose:** Enables data scientists to explore data and develop models without access to production resources.

**Permissions:**
- ✅ Read-only access to dataset S3 bucket (`lucaskle-ab3-project-pv`)
- ✅ SageMaker Studio notebook access
- ✅ MLFlow experiment tracking (read/write)
- ✅ CloudWatch logs for notebooks
- ❌ **DENIED:** Endpoint creation/deployment
- ❌ **DENIED:** Pipeline creation/execution
- ❌ **DENIED:** Production resource access

**Trust Policy:** Can be assumed by SageMaker service and account root

### ML Engineer Role

**Purpose:** Provides full access for operationalizing models and managing production pipelines.

**Permissions:**
- ✅ Full SageMaker access (pipelines, endpoints, models)
- ✅ Full S3 access for artifacts and data
- ✅ EventBridge for pipeline notifications
- ✅ CloudWatch for monitoring and logging
- ✅ Cost Explorer for cost monitoring
- ✅ IAM PassRole for execution roles

**Trust Policy:** Can be assumed by SageMaker service and account root

### SageMaker Execution Role

**Purpose:** Service role for SageMaker training jobs and pipeline execution.

**Permissions:**
- ✅ Full SageMaker service access
- ✅ S3 access to project buckets
- ✅ ECR access for custom containers
- ✅ CloudWatch logs

**Trust Policy:** Can only be assumed by SageMaker service

### SageMaker Studio Policy

**Purpose:** Managed policy with governance controls for Studio access.

**Features:**
- Instance type restrictions (cost optimization)
- Required resource tagging
- Governance controls

## Security Considerations

### Principle of Least Privilege

Each role is granted only the minimum permissions necessary for its function:

- Data Scientists cannot access production resources
- Service roles cannot be assumed by users directly
- Instance types are restricted to prevent cost overruns

### Audit and Compliance

- All roles include comprehensive CloudWatch logging
- Resource tagging is enforced for cost tracking
- IAM policies follow AWS security best practices

### Cost Controls

- Instance type restrictions prevent expensive resource usage
- Resource tagging enables cost allocation and monitoring
- Automatic cleanup policies can be implemented

## Troubleshooting

### Common Issues

1. **"Access Denied" errors:**
   - Verify the AWS profile has sufficient permissions
   - Check that roles are properly deployed
   - Validate trust relationships

2. **Stack deployment fails:**
   - Ensure unique role names across regions
   - Verify CloudFormation template syntax
   - Check for existing resources with same names

3. **Validation script fails:**
   - Confirm AWS CLI profile is configured
   - Verify boto3 is installed
   - Check network connectivity to AWS APIs

### Validation Commands

```bash
# Test specific role
python3 scripts/setup/validate_iam_roles.py --role data-scientist

# Test with different profile
python3 scripts/setup/validate_iam_roles.py --profile myprofile

# Full validation
python3 scripts/setup/validate_iam_roles.py --role all
```

### Manual Verification

You can manually verify role permissions using the AWS CLI:

```bash
# List role policies
aws iam list-attached-role-policies --role-name mlops-sagemaker-demo-DataScientist-Role --profile ab

# Get role trust policy
aws iam get-role --role-name mlops-sagemaker-demo-DataScientist-Role --profile ab

# Simulate permissions
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::ACCOUNT:role/mlops-sagemaker-demo-DataScientist-Role \
  --action-names s3:GetObject \
  --resource-arns arn:aws:s3:::lucaskle-ab3-project-pv/* \
  --profile ab
```

## Integration with SageMaker Studio

After deploying the IAM roles, configure SageMaker Studio:

1. Create a SageMaker Domain with the appropriate execution role
2. Create user profiles for Data Scientists and ML Engineers
3. Assign the respective IAM roles to each user profile
4. Configure Studio apps with the SageMaker Studio Policy

## Cost Monitoring

The roles are configured to work with the "ab" AWS CLI profile for cost allocation:

- All resources are tagged with project identifiers
- Cost Explorer permissions enable cost tracking
- Instance type restrictions prevent cost overruns
- Cleanup procedures are documented for cost optimization

## Next Steps

After successful IAM setup:

1. Configure SageMaker Studio Domain and user profiles
2. Set up MLFlow tracking server with appropriate permissions
3. Create S3 buckets for artifacts and experiments
4. Test role-based access with sample notebooks
5. Implement pipeline deployment with proper role separation

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review AWS CloudFormation events for deployment issues
3. Use the validation script to identify permission problems
4. Consult AWS documentation for SageMaker IAM best practices