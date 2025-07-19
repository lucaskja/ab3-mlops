# Model Deployment Pipeline

This repository contains code for the SageMaker model deployment pipeline for YOLOv11 object detection on drone imagery.

## Directory Structure

```
.
├── staging-buildspec.yml         # Build specification for staging deployment
├── staging-test-buildspec.yml    # Test specification for staging environment
├── prod-buildspec.yml            # Build specification for production deployment
├── requirements.txt              # Python dependencies
├── scripts/                      # Scripts for deployment
│   ├── deployment/
│   │   ├── get_approved_model.py          # Fetches approved model from registry
│   │   ├── deploy_model.py                # Deploys model to endpoint
│   │   ├── check_endpoint_metrics.py      # Checks endpoint performance metrics
│   │   └── prepare_production_deployment.py # Prepares for production deployment
│   └── monitoring/
│       └── setup_model_monitoring.py      # Sets up model monitoring
├── src/                          # Source code modules
│   ├── deployment/               # Deployment modules
│   └── monitoring/               # Monitoring modules
└── tests/                        # Tests for deployment
    └── test_endpoint_performance.py # Tests endpoint performance
```

## Deployment Workflow

This repository implements a multi-stage deployment workflow:

1. **Staging Deployment**: Deploys the model to a staging endpoint
2. **Staging Testing**: Tests the model in the staging environment
3. **Manual Approval**: Requires manual approval for production deployment
4. **Production Deployment**: Deploys the model to the production endpoint

## CI/CD Pipeline

This repository is integrated with AWS CodePipeline for continuous deployment:

1. **Source**: Code changes and new approved models trigger the pipeline
2. **Staging Deploy**: Deploys the model to the staging environment
3. **Staging Test**: Tests the model in the staging environment
4. **Approval**: Requires manual approval for production deployment
5. **Production Deploy**: Deploys the model to the production environment

## Configuration

The deployment pipeline can be configured through environment variables:

- `SAGEMAKER_PROJECT_NAME`: Name of the SageMaker project
- `SAGEMAKER_PROJECT_ID`: ID of the SageMaker project
- `MODEL_PACKAGE_GROUP_NAME`: Name of the model package group
- `ENDPOINT_NAME`: Base name for the endpoint
- `ENDPOINT_TYPE`: Type of endpoint (staging or production)
- `AWS_REGION`: AWS region for deployment
- `AWS_PROFILE`: AWS CLI profile to use (default: 'ab')

## Testing

Run endpoint tests:

```bash
python -m pytest tests/test_endpoint_performance.py --endpoint-name <endpoint-name> --profile ab
```

## Monitoring

The deployment pipeline automatically sets up model monitoring for:

- Data quality
- Model quality
- Bias drift
- Feature attribution drift

Monitoring results are stored in S3 and can be visualized in SageMaker Studio.