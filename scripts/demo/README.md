# Demo Scripts

This directory contains scripts for setting up and tearing down the MLOps SageMaker Demo environment.

## Setup Script

The `setup_demo.py` script sets up the necessary resources for the MLOps SageMaker Demo, including:

- S3 buckets
- IAM roles
- Sample data

### Usage

```bash
# Set up everything
python setup_demo.py --all --profile ab

# Set up specific components
python setup_demo.py --create-bucket --profile ab
python setup_demo.py --upload-sample-data --profile ab
python setup_demo.py --deploy-iam-roles --profile ab
```

### Options

- `--profile`: AWS CLI profile to use (default: `ab`)
- `--region`: AWS region to use (default: `us-east-1`)
- `--create-bucket`: Create S3 bucket
- `--upload-sample-data`: Upload sample data to S3
- `--deploy-iam-roles`: Deploy IAM roles
- `--all`: Perform all setup steps

## Teardown Script

The `teardown_demo.py` script cleans up resources created for the MLOps SageMaker Demo, including:

- SageMaker endpoints
- SageMaker pipelines
- Monitoring schedules
- IAM roles
- S3 buckets

### Usage

```bash
# Tear down everything
python teardown_demo.py --all --profile ab --force

# Tear down specific components
python teardown_demo.py --delete-endpoints --profile ab
python teardown_demo.py --delete-pipelines --profile ab
python teardown_demo.py --delete-monitoring --profile ab
python teardown_demo.py --delete-iam-roles --profile ab
python teardown_demo.py --empty-bucket --profile ab
python teardown_demo.py --delete-bucket --profile ab
```

### Options

- `--profile`: AWS CLI profile to use (default: `ab`)
- `--region`: AWS region to use (default: `us-east-1`)
- `--delete-endpoints`: Delete SageMaker endpoints
- `--delete-pipelines`: Delete SageMaker pipelines
- `--delete-monitoring`: Delete monitoring schedules
- `--delete-iam-roles`: Delete IAM roles
- `--empty-bucket`: Empty S3 bucket
- `--delete-bucket`: Delete S3 bucket
- `--all`: Perform all teardown steps
- `--force`: Force deletion without confirmation

## Cost Monitoring

Both scripts include proper resource tagging to enable cost tracking using the "ab" AWS CLI profile. All resources created by these scripts are tagged with:

- `Project`: The project name from `configs/project_config.py`
- `Environment`: `demo`
- `CreatedBy`: The script name

This enables detailed cost tracking and allocation in AWS Cost Explorer.

## Best Practices

1. Always run the teardown script when you're done with the demo to avoid ongoing costs
2. Use the `--force` flag with caution, as it bypasses the confirmation prompt
3. Consider running the teardown script with specific components rather than `--all` if you want to preserve certain resources
4. Monitor costs regularly using AWS Cost Explorer with the "ab" profile

## Troubleshooting

If you encounter issues with the scripts:

1. Check the logs for error messages
2. Verify that your AWS CLI profile has the necessary permissions
3. Ensure that the AWS region specified matches the region where your resources are deployed
4. If a resource fails to delete, try deleting it manually through the AWS console