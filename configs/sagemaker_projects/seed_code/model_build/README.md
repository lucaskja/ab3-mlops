# Model Building Pipeline

This repository contains code for the SageMaker model building pipeline for YOLOv11 object detection on drone imagery.

## Directory Structure

```
.
├── buildspec.yml                 # Build specification for CodeBuild
├── test-buildspec.yml            # Test specification for CodeBuild
├── requirements.txt              # Python dependencies
├── scripts/                      # Scripts for pipeline execution
│   └── training/
│       ├── create_sagemaker_pipeline.py       # Creates the SageMaker Pipeline
│       ├── complete_pipeline_orchestration.py # Orchestrates the pipeline execution
│       ├── evaluate_model.py                  # Evaluates model performance
│       └── prepare_model_registration.py      # Prepares model for registration
├── src/                          # Source code modules
│   ├── data/                     # Data processing modules
│   ├── models/                   # Model implementation modules
│   └── pipeline/                 # Pipeline orchestration modules
└── tests/                        # Unit and integration tests
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure AWS CLI: `aws configure --profile ab`
4. Create the pipeline: `python scripts/training/create_sagemaker_pipeline.py --profile ab`
5. Execute the pipeline: `python scripts/training/complete_pipeline_orchestration.py --profile ab`

## CI/CD Pipeline

This repository is integrated with AWS CodePipeline for continuous integration and delivery:

1. **Source**: Code changes trigger the pipeline
2. **Build**: Creates and executes the SageMaker Pipeline
3. **Test**: Runs unit tests and evaluates model performance
4. **Register**: Registers the model in the SageMaker Model Registry

## Development Workflow

1. Make changes to the code
2. Run tests locally: `python -m pytest tests/`
3. Commit and push changes
4. CodePipeline automatically builds and tests the model
5. If tests pass, the model is registered in the Model Registry

## Configuration

The pipeline can be configured through environment variables:

- `SAGEMAKER_PROJECT_NAME`: Name of the SageMaker project
- `SAGEMAKER_PROJECT_ID`: ID of the SageMaker project
- `MODEL_PACKAGE_GROUP_NAME`: Name of the model package group
- `AWS_REGION`: AWS region for deployment
- `AWS_PROFILE`: AWS CLI profile to use (default: 'ab')

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Run with coverage:

```bash
python -m pytest tests/ --cov=src --cov-report=term
```