# Project Structure

## Directory Organization

```
├── configs/                 # Configuration files and infrastructure
│   ├── project_config.py    # Centralized project configuration
│   ├── environment_config.py # Environment-specific configuration
│   ├── iam-roles-cloudformation.yaml  # IAM infrastructure
│   ├── cdk/                 # AWS CDK infrastructure code
│   │   ├── lib/             # CDK stack definitions
│   │   └── bin/             # CDK entry points
│   └── environments/        # Environment-specific configurations
├── notebooks/               # Jupyter notebooks for development
│   ├── data-exploration/    # Data analysis and profiling notebooks
│   ├── data-labeling/       # Ground Truth labeling job notebooks
│   ├── model-development/   # Model training and experimentation
│   └── pipeline-development/ # Pipeline development notebooks
├── scripts/                 # Utility and setup scripts
│   ├── monitoring/          # Model monitoring scripts
│   ├── preprocessing/       # Data preprocessing scripts
│   ├── setup/               # Environment and AWS setup scripts
│   └── training/            # Training execution scripts
├── src/                    # Source code modules
│   ├── data/               # Data processing and validation
│   │   ├── data_profiler.py # Data profiling utilities
│   │   ├── data_validator.py # Data validation utilities
│   │   ├── ground_truth_utils.py # Ground Truth labeling utilities
│   │   ├── s3_utils.py     # S3 access utilities
│   │   └── yolo_preprocessor.py # YOLO data preprocessing
│   ├── models/             # Model implementation modules
│   │   └── yolov11_trainer.py # YOLOv11 training implementation
│   ├── monitoring/         # Model monitoring modules
│   │   └── drift_detection.py # Drift detection utilities
│   └── pipeline/           # Pipeline orchestration modules
│       ├── clarify_integration.py # SageMaker Clarify integration
│       ├── error_recovery.py # Pipeline error handling
│       ├── mlflow_integration.py # MLFlow experiment tracking
│       ├── mlflow_visualization.py # MLFlow visualization utilities
│       ├── model_monitor.py # SageMaker Model Monitor
│       ├── model_monitor_integration.py # Model monitoring integration
│       ├── sagemaker_pipeline.py # SageMaker Pipeline implementation
│       └── sagemaker_training.py # SageMaker training job management
├── tests/                  # Unit and integration tests
│   ├── ground_truth/       # Tests for Ground Truth utilities
│   │   └── test_ground_truth_utils.py # Tests for Ground Truth utilities
│   ├── test_error_recovery.py # Tests for error recovery functionality
│   ├── test_event_bridge_integration.py # Tests for EventBridge integration
│   ├── test_mlflow_integration.py # Tests for MLFlow integration
│   ├── test_yolo_preprocessor.py # Tests for YOLO preprocessing
│   └── test_yolov11_trainer.py # Tests for YOLOv11 trainer
├── mlruns/                 # Local MLFlow tracking data
└── logs/                   # Application logs
```

## Code Organization Patterns

### Configuration Management
- **Central Config**: All project settings in `configs/project_config.py`
- **Environment Config**: Environment-specific settings in `configs/environment_config.py`
- **Environment Variables**: AWS profile and region configuration
- **Constants**: Project names, bucket names, and resource identifiers

### Module Structure
- **Data Layer** (`src/data/`): S3 utilities, validation, and profiling
- **Model Layer** (`src/models/`): Model implementations and utilities
- **Pipeline Layer** (`src/pipeline/`): Orchestration and workflow management
- **Testing Layer** (`tests/`): Unit and integration tests for all components

### Naming Conventions
- **Files**: snake_case for Python files
- **Classes**: PascalCase (e.g., `DroneImageryProfiler`, `YOLOv11Validator`, `MLFlowSageMakerIntegration`)
- **Functions**: snake_case with descriptive names
- **Constants**: UPPER_SNAKE_CASE in configuration files
- **AWS Resources**: Prefixed with project name (`mlops-sagemaker-demo-`)
- **Test Files**: Prefixed with `test_` followed by the module name being tested

### Import Patterns
- **Relative Imports**: Use relative imports within modules
- **Configuration**: Import from `configs.project_config`
- **Logging**: Configure at module level with `logging.getLogger(__name__)`
- **AWS SDK**: Import boto3 and sagemaker modules with explicit session management

## File Conventions

### Python Files
- **Docstrings**: Comprehensive module and function documentation
- **Type Hints**: Use typing module for function signatures
- **Error Handling**: Comprehensive exception handling with logging
- **AWS Integration**: Use boto3 with profile-based authentication

### Test Files
- **Test Classes**: Inherit from `unittest.TestCase`
- **Test Methods**: Prefixed with `test_` followed by the functionality being tested
- **Mocking**: Use `unittest.mock` for external dependencies
- **Setup/Teardown**: Use `setUp` and `tearDown` methods for test environment management
- **Assertions**: Use appropriate assertion methods for different validation types

### Notebooks
- **Purpose-Driven**: Each notebook has a specific analytical purpose
- **Documentation**: Clear markdown cells explaining analysis steps
- **Modular**: Import utilities from `src/` modules rather than duplicating code

### Scripts
- **Executable**: All scripts have proper shebang and execute permissions
- **Error Handling**: Use `set -e` for bash scripts
- **Logging**: Colored output for status, warnings, and errors
- **Validation**: Check prerequisites before execution

## Development Workflow

### New Feature Development
1. Create/update configuration in `configs/project_config.py`
2. Implement core logic in appropriate `src/` module
3. Write unit tests in `tests/` directory
4. Create notebook for experimentation in relevant `notebooks/` subdirectory
5. Add setup/deployment scripts in `scripts/` if needed

### Testing Workflow
1. Write unit tests for all new functionality
2. Run tests with `python -m unittest discover tests`
3. Ensure all tests pass before committing changes
4. Use mocking for external dependencies like AWS services
5. Implement integration tests for end-to-end validation

### AWS Resource Management
- **Infrastructure**: Define in CloudFormation templates or CDK stacks
- **Deployment**: Use provided shell scripts with `ab` profile
- **Validation**: Run validation scripts after deployment

## Command Execution Rules

### Virtual Environment
- **ALWAYS** activate and use the `venv` virtual environment for all Python commands
- **ALWAYS** use `venv/bin/python` or `venv/bin/pip` for Python operations
- **Never** run Python commands without the virtual environment activated

### AWS Profile
- **ALWAYS** use the AWS profile called `ab` for all AWS operations
- **ALWAYS** set `AWS_PROFILE=ab` environment variable or use `--profile ab` flag
- **Never** use default AWS profile or other profiles

### Testing
- **Run Tests**: Use `python -m unittest discover tests` to run all tests
- **Single Test**: Use `python -m unittest tests/test_file.py` for specific test files
- **Test Coverage**: Use `coverage run -m unittest discover` followed by `coverage report`

### MLFlow Operations
- **Tracking**: Use `MLFlowSageMakerIntegration` class from `src.pipeline.mlflow_integration`
- **Experiments**: Create experiments with `create_experiment` method
- **Runs**: Use context manager with `start_run` method
- **Parameters**: Log parameters with `log_parameters` method
- **Metrics**: Log metrics with `log_metrics` method
- **Models**: Log models with `log_model` method
- **SageMaker Jobs**: Log SageMaker jobs with `log_sagemaker_job` method

### Data Processing
- **S3 Access**: Use `S3DataAccess` class from `src.data.s3_utils`
- **Validation**: Use `YOLOv11Validator` for format compliance
- **Profiling**: Use `DroneImageryProfiler` for dataset analysis