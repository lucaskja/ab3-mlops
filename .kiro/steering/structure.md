# Project Structure

## Directory Organization

```
├── configs/                 # Configuration files and infrastructure
│   ├── project_config.py    # Centralized project configuration
│   ├── environment_config.py # Environment-specific configuration
│   ├── iam-roles-cloudformation.yaml  # IAM infrastructure
│   ├── iam_roles_cdk.py     # CDK implementation of IAM roles
│   ├── training_config.yaml # Training configuration
│   ├── sagemaker_training_config.json # SageMaker training configuration
│   ├── s3_training_config.json # S3 training data configuration
│   ├── example_training_config.json # Example training configuration
│   ├── IAM_SETUP_README.md  # IAM setup documentation
│   ├── cdk/                 # AWS CDK infrastructure code
│   │   ├── lib/             # CDK stack definitions
│   │   │   ├── endpoint-stack.ts # SageMaker endpoint stack
│   │   │   └── iam-stack.ts # IAM roles stack
│   │   ├── bin/             # CDK entry points
│   │   ├── package.json     # Node.js dependencies
│   │   └── tsconfig.json    # TypeScript configuration
│   └── environments/        # Environment-specific configurations
├── docs/                    # Project documentation
│   ├── architecture/        # Architecture documentation
│   │   └── mlops_architecture.md # MLOps architecture overview
│   └── user-guides/         # User guides for different roles
│       ├── data_scientist_guide.md # Guide for Data Scientists
│       └── ml_engineer_guide.md # Guide for ML Engineers
├── .kiro/                   # Kiro AI assistant configuration
│   ├── hooks/               # Kiro hooks for automation
│   ├── settings/            # Kiro settings
│   ├── specs/               # Kiro specifications
│   │   ├── mlops-sagemaker-demo/ # MLOps SageMaker demo spec
│   │   │   ├── design.md    # Design document
│   │   │   ├── requirements.md # Requirements document
│   │   │   └── tasks.md     # Implementation tasks
│   │   └── sagemaker-core-setup/ # Core SageMaker setup spec
│   │       ├── design.md    # Design document for core setup
│   │       ├── requirements.md # Requirements for core setup
│   │       └── tasks.md     # Implementation tasks for core setup
│   └── steering/            # Kiro steering documents
│       ├── structure.md     # Project structure documentation
│       ├── tech.md          # Technology stack documentation
│       ├── product.md       # Product overview
│       ├── ground_truth_labeling.md # Ground Truth labeling guidance
│       ├── sagemaker_reference.md # SageMaker reference
│       ├── aws_mlops_reference.md # AWS MLOps reference
│       └── aws_sagemaker_mlops_reference.md # AWS SageMaker MLOps reference
├── notebooks/               # Jupyter notebooks for development
│   ├── README.md            # Notebooks documentation
│   ├── README_data-scientist.md # Generated Data Scientist notebook guide
│   ├── README_ml-engineer.md # Generated ML Engineer notebook guide
│   ├── data-scientist-core.ipynb # Core notebook for Data Scientists
│   ├── data-scientist-core-enhanced.ipynb # Enhanced core notebook with MLFlow integration
│   ├── ml-engineer-core.ipynb # Core notebook for ML Engineers
│   ├── ml-engineer-core-enhanced.ipynb # Enhanced core notebook with MLFlow and Model Registry
│   ├── data-exploration/    # Data analysis and profiling notebooks
│   │   ├── data-profiling.ipynb # Data profiling notebook
│   │   └── dataset-analysis.ipynb # Dataset analysis notebook
│   ├── data-labeling/       # Ground Truth labeling job notebooks
│   │   ├── create_labeling_job.ipynb           # Basic labeling job creation
│   │   ├── create_labeling_job_enhanced.ipynb  # Enhanced labeling workflow
│   │   └── create_labeling_job_interactive.ipynb # Interactive UI with dashboard for labeling
│   ├── demo/                # Demo notebooks
│   │   ├── cleanup_resources.ipynb # Resource cleanup notebook
│   │   ├── cost_monitoring.ipynb # Cost monitoring and analysis notebook
│   │   ├── data_scientist_workflow.ipynb # Data Scientist workflow demo with data exploration, labeling, and model development
│   │   ├── end_to_end_mlops_workflow.ipynb # End-to-end MLOps workflow demo
│   │   └── ml_engineer_workflow.ipynb # ML Engineer workflow demo
│   ├── model-development/   # Model training and experimentation
│   └── pipeline-development/ # Pipeline development notebooks
├── examples/                # Example code and usage patterns
│   ├── data-labeling/       # Ground Truth labeling examples
│   │   └── ground_truth_example.py # Ground Truth example script
│   ├── model-training/      # Model training examples
│   └── pipeline/            # Pipeline orchestration examples
├── scripts/                 # Utility and setup scripts
│   ├── demo/                # Demo setup and teardown scripts
│   │   ├── README.md        # Demo scripts documentation
│   │   ├── setup_demo.py    # Demo environment setup script
│   │   └── teardown_demo.py # Demo environment cleanup script
│   ├── monitoring/          # Model monitoring scripts
│   │   └── setup_model_monitoring.py # Model monitoring setup script
│   ├── preprocessing/       # Data preprocessing scripts
│   │   └── preprocess_yolo_data.py # YOLO data preprocessing script
│   ├── setup/               # Environment and AWS setup scripts
│   │   ├── README.md        # Setup scripts documentation
│   │   ├── cleanup_core_sagemaker.sh # Core SageMaker cleanup script
│   │   ├── cleanup_resources.sh # Resource cleanup script
│   │   ├── configure_aws.sh # AWS configuration script
│   │   ├── deploy_cdk.sh    # CDK deployment script
│   │   ├── deploy_complete_infrastructure.sh # Complete infrastructure deployment script
│   │   ├── deploy_complete_infrastructure_modified.sh # Modified infrastructure deployment script
│   │   ├── deploy_core_sagemaker.sh # Core SageMaker deployment script
│   │   ├── deploy_iam_roles.sh # IAM roles deployment script
│   │   ├── deploy_iam_roles_temp.sh # Temporary IAM roles deployment script
│   │   ├── deploy_notebooks_to_studio.py # Deploy core notebooks to SageMaker Studio
│   │   ├── reorganize_s3_data.py # S3 data reorganization script
│   │   ├── setup_mlflow_sagemaker.py # MLFlow SageMaker setup script
│   │   ├── setup_sagemaker_project.py # SageMaker project setup script
│   │   ├── validate_cdk_deployment.sh # CDK deployment validation script
│   │   ├── validate_cdk_security.sh # CDK security validation script
│   │   ├── validate_cleanup.py # Cleanup validation script
│   │   ├── validate_core_sagemaker.py # Core SageMaker validation script
│   │   ├── validate_deployment.py # Deployment validation script
│   │   └── validate_iam_roles.py # IAM roles validation script
│   └── training/            # Training execution scripts
│       ├── analyze_model_monitoring.py # Model monitoring analysis script
│       ├── automated_retraining_trigger.py # Automated model retraining based on drift detection
│       ├── complete_pipeline_orchestration.py # Complete pipeline orchestration script
│       ├── create_sagemaker_pipeline.py # SageMaker pipeline creation script
│       ├── deploy_endpoint_lambda.py # Endpoint deployment Lambda script
│       ├── evaluate_model.py # Model evaluation script
│       ├── setup_clarify_explainability.py # SageMaker Clarify setup script
│       ├── setup_model_monitoring.py # Model monitoring setup script
│       ├── submit_sagemaker_job.py # SageMaker job submission script
│       ├── test_mlflow_integration.py # MLFlow integration test script
│       ├── train_sagemaker.py # SageMaker training job execution script
│       ├── train_sagemaker_observability.py # Training with observability features script
│       └── train_yolov11.py # YOLOv11 training script
├── src/                    # Source code modules
│   ├── README.md           # Source code documentation
│   ├── __init__.py         # Package initialization
│   ├── data/               # Data processing and validation
│   │   ├── __init__.py     # Package initialization
│   │   ├── README_ground_truth.md # Ground Truth documentation
│   │   ├── data_profiler.py # Data profiling utilities
│   │   ├── data_validator.py # Data validation utilities
│   │   ├── ground_truth_utils.py # Ground Truth labeling utilities
│   │   ├── s3_utils.py     # S3 access utilities
│   │   └── yolo_preprocessor.py # YOLO data preprocessing
│   ├── lambda_functions/   # AWS Lambda function implementations
│   │   ├── drift_detection/ # Drift detection Lambda function
│   │   │   └── lambda_function.py # Lambda handler for drift detection
│   │   ├── model_approval/ # Model approval workflow Lambda function
│   │   │   └── lambda_function.py # Lambda handler for model approval
│   │   ├── scheduled_evaluation/ # Scheduled model evaluation Lambda function
│   │   │   └── lambda_function.py # Lambda handler for scheduled evaluation
│   │   └── utils.py        # Shared utilities for Lambda functions
│   ├── models/             # Model implementation modules
│   │   ├── __init__.py     # Package initialization
│   │   └── yolov11_trainer.py # YOLOv11 training implementation
│   ├── monitoring/         # Model monitoring modules
│   │   ├── cost_tracking.py # Cost monitoring and tracking utilities
│   │   └── drift_detection.py # Drift detection utilities
│   └── pipeline/           # Pipeline orchestration modules
│       ├── __init__.py     # Package initialization
│       ├── clarify_integration.py # SageMaker Clarify integration
│       ├── error_recovery.py # Pipeline error handling
│       ├── event_bridge_integration.py # EventBridge integration
│       ├── mlflow_integration.py # MLFlow experiment tracking
│       ├── mlflow_visualization.py # MLFlow visualization utilities
│       ├── model_monitor.py # SageMaker Model Monitor
│       ├── model_monitor_integration.py # Model monitoring integration
│       ├── sagemaker_pipeline.py # SageMaker Pipeline implementation
│       ├── sagemaker_pipeline_factory.py # SageMaker Pipeline factory
│       ├── sagemaker_training.py # SageMaker training job management
│       └── script_templates.py # Script templates for pipeline steps
├── tests/                  # Unit and integration tests
│   ├── __init__.py         # Package initialization
│   ├── ground_truth/       # Tests for Ground Truth utilities
│   │   └── test_ground_truth_utils.py # Tests for Ground Truth utilities
│   ├── test_automated_retraining_trigger.py # Tests for automated retraining functionality
│   ├── test_automated_retraining_trigger_new.py # New tests for automated retraining
│   ├── test_data_profiler.py # Tests for data profiler
│   ├── test_drift_detection.py # Tests for drift detection
│   ├── test_endpoint_performance.py # Tests for endpoint performance
│   ├── test_error_recovery.py # Tests for error recovery functionality
│   ├── test_event_bridge_integration.py # Tests for EventBridge integration
│   ├── test_mlflow_integration.py # Tests for MLFlow integration
│   ├── test_model_monitor.py # Tests for model monitor
│   ├── test_model_monitor_integration.py # Tests for model monitor integration
│   ├── test_monitoring_alerting_integration.py # Tests for monitoring alerting
│   ├── test_pipeline_integration.py # Tests for pipeline integration
│   ├── test_role_based_access.py # Tests for role-based access
│   ├── test_script_templates.py # Tests for script templates
│   ├── test_yolo_preprocessor.py # Tests for YOLO preprocessing
│   └── test_yolov11_trainer.py # Tests for YOLOv11 trainer
├── mlruns/                 # Local MLFlow tracking data
├── logs/                   # Application logs
├── runs/                   # Training run outputs
│   └── detect/             # Detection model runs
│       └── yolov11_training/ # YOLOv11 training runs
├── README.md               # Project README
├── DEPLOYMENT_SUMMARY.md   # Deployment summary and status
├── SAGEMAKER_STUDIO_INSTRUCTIONS.md # Generated SageMaker Studio access instructions
├── cdk-fixes-summary.md    # Summary of CDK fixes
└── requirements.txt        # Python dependencies
```

## Code Organization Patterns

### Configuration Management
- **Central Config**: All project settings in `configs/project_config.py`
- **Environment Config**: Environment-specific settings in `configs/environment_config.py`
- **Environment Variables**: AWS profile and region configuration
- **Constants**: Project names, bucket names, and resource identifiers
- **Training Configs**: Multiple formats supported (YAML, JSON) in the configs directory

### Module Structure
- **Data Layer** (`src/data/`): S3 utilities, validation, profiling, and Ground Truth integration
- **Model Layer** (`src/models/`): Model implementations and utilities, focusing on YOLOv11
- **Pipeline Layer** (`src/pipeline/`): Orchestration and workflow management with SageMaker Pipelines
- **Monitoring Layer** (`src/monitoring/`): Drift detection, cost tracking, and model monitoring
- **Lambda Functions Layer** (`src/lambda_functions/`): AWS Lambda functions for drift detection, model evaluation, and approval workflows
- **Automated Retraining** (`scripts/training/automated_retraining_trigger.py`): Setup script for automated model retraining infrastructure, including Lambda functions and EventBridge rules
- **Testing Layer** (`tests/`): Unit and integration tests for all components
- **Documentation Layer** (`docs/`): Architecture documentation and user guides
- **Demo Layer** (`notebooks/demo/` and `scripts/demo/`): End-to-end workflow demonstrations for different user roles
- **Core Setup Layer** (`scripts/setup/`): Core SageMaker infrastructure deployment and management

### Naming Conventions
- **Files**: snake_case for Python files
- **Classes**: PascalCase (e.g., `DroneImageryProfiler`, `YOLOv11Validator`, `MLFlowSageMakerIntegration`)
- **Functions**: snake_case with descriptive names
- **Constants**: UPPER_SNAKE_CASE in configuration files
- **AWS Resources**: Prefixed with project name (`mlops-sagemaker-demo-` or `sagemaker-core-setup-`)
- **Test Files**: Prefixed with `test_` followed by the module name being tested
- **Notebooks**: Descriptive names with hyphens for directories and underscores for files
- **Core Notebooks**: Root-level notebooks with role-specific names (e.g., `data-scientist-core.ipynb`, `ml-engineer-core.ipynb`)

### Import Patterns
- **Relative Imports**: Use relative imports within modules
- **Configuration**: Import from `configs.project_config`
- **Logging**: Configure at module level with `logging.getLogger(__name__)`
- **AWS SDK**: Import boto3 and sagemaker modules with explicit session management
- **Project Modules**: Import from src modules using absolute imports
- **Demo Notebooks**: Import project modules using sys.path.append('..') to access src modules
- **Core Notebooks**: Direct imports of standard libraries and boto3 with minimal dependencies

## File Conventions

### Python Files
- **Docstrings**: Comprehensive module and function documentation
- **Type Hints**: Use typing module for function signatures
- **Error Handling**: Comprehensive exception handling with logging
- **AWS Integration**: Use boto3 with profile-based authentication
- **Shebang**: Include `#!/usr/bin/env python3` for executable scripts

### Test Files
- **Test Classes**: Inherit from `unittest.TestCase`
- **Test Methods**: Prefixed with `test_` followed by the functionality being tested
- **Mocking**: Use `unittest.mock` for external dependencies
- **Setup/Teardown**: Use `setUp` and `tearDown` methods for test environment management
- **Assertions**: Use appropriate assertion methods for different validation types
- **Lambda Testing**: Mock AWS clients and test Lambda function creation and configuration
- **File Operations**: Mock file operations for Lambda code loading and ZIP file creation
- **AWS Resource Testing**: Test AWS resource creation and configuration with mocked responses

### Notebooks
- **Purpose-Driven**: Each notebook has a specific analytical purpose
- **Documentation**: Clear markdown cells explaining analysis steps
- **Modular**: Import utilities from `src/` modules rather than duplicating code
- **Interactive Interfaces**: Use ipywidgets for interactive user interfaces in notebooks
- **Visualization**: Include data and result visualization with matplotlib and other libraries
- **Demo Notebooks**: Comprehensive end-to-end workflows in the `notebooks/demo/` directory
- **Role-Specific Notebooks**: Separate notebooks for Data Scientist and ML Engineer workflows
- **Core Notebooks**: Simplified notebooks at the root level for essential functionality:
  - `data-scientist-core.ipynb`: Data exploration and preparation for YOLOv11 training
  - `data-scientist-core-enhanced.ipynb`: Enhanced data exploration with MLFlow experiment tracking and Ground Truth integration
  - `ml-engineer-core.ipynb`: Pipeline execution and management for YOLOv11 models
  - `ml-engineer-core-enhanced.ipynb`: Enhanced pipeline management with MLFlow and Model Registry integration
- **Generated Documentation**: Role-specific README files created during deployment:
  - `README_data-scientist.md`: Generated guide for Data Scientists with notebook usage instructions
  - `README_ml-engineer.md`: Generated guide for ML Engineers with notebook usage instructions
- **Interactive Dashboards**: Advanced notebooks like `create_labeling_job_interactive.ipynb` implement complete dashboard interfaces with tabs, progress tracking, and real-time updates

### Scripts
- **Executable**: All scripts have proper shebang and execute permissions
- **Error Handling**: Use `set -e` for bash scripts
- **Logging**: Colored output for status, warnings, and errors
- **Validation**: Check prerequisites before execution
- **AWS Profile**: Always use the "ab" profile for AWS operations
- **Demo Scripts**: Setup and teardown scripts in `scripts/demo/` for demonstration environments
- **Core Setup Scripts**: Simplified deployment scripts in `scripts/setup/` for core SageMaker infrastructure:
  - `deploy_core_sagemaker.sh`: Deploys essential SageMaker components
  - `deploy_notebooks_to_studio.py`: Deploys core notebooks to SageMaker Studio user profiles
  - `copy_notebooks_to_studio.sh`: Simple script for copying notebooks to SageMaker Studio
  - `deploy_studio_notebooks.sh`: Generated deployment script for SageMaker Studio notebooks
  - `validate_core_sagemaker.py`: Validates core SageMaker deployment
  - `cleanup_core_sagemaker.sh`: Removes core SageMaker resources
- **Documentation Files**: Generated documentation for deployment and access:
  - `DEPLOYMENT_SUMMARY.md`: Comprehensive deployment status and instructions
  - `SAGEMAKER_STUDIO_INSTRUCTIONS.md`: Step-by-step SageMaker Studio access instructions

### Infrastructure as Code
- **CDK**: TypeScript-based AWS CDK for infrastructure definition
- **CloudFormation**: Legacy CloudFormation templates for IAM roles
- **Validation**: Scripts to validate infrastructure deployments
- **Security**: CDK Nag for security validation of infrastructure

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
- **Security**: Validate security with CDK Nag and custom validation scripts
- **Cleanup**: Use cleanup scripts to remove resources when no longer needed

### Core SageMaker Setup Workflow
1. **Deploy Core Infrastructure**: Use `deploy_core_sagemaker.sh` to set up essential components
2. **Deploy Notebooks**: Use `deploy_notebooks_to_studio.py` to deploy core notebooks to SageMaker Studio user profiles
3. **Validate Deployment**: Run `validate_core_sagemaker.py` to verify resources
4. **Access SageMaker Studio**: Log in with appropriate user profile (Data Scientist or ML Engineer)
5. **Use Core Notebooks**: Choose between basic or enhanced notebooks based on requirements:
   - **Basic**: `data-scientist-core.ipynb` or `ml-engineer-core.ipynb` for essential functionality
   - **Enhanced**: `data-scientist-core-enhanced.ipynb` or `ml-engineer-core-enhanced.ipynb` for MLFlow integration
6. **Clean Up Resources**: Use `cleanup_core_sagemaker.sh` when no longer needed

### Notebook Template Development Workflow
1. **Template Creation**: Create notebook templates with role-specific functionality
2. **Content Development**: Implement core workflows for each user role (Data Scientist, ML Engineer)
3. **MLFlow Integration**: Add comprehensive experiment tracking and model registry integration
4. **Documentation Generation**: Generate role-specific README files during deployment
5. **Deployment Integration**: Upload notebooks to S3 and create deployment scripts for SageMaker Studio
6. **User Instructions**: Generate comprehensive access instructions in `SAGEMAKER_STUDIO_INSTRUCTIONS.md`

### Enhanced Notebooks Features
- **MLFlow Integration**: Enhanced notebooks include comprehensive MLFlow experiment tracking
- **Model Registry**: ML Engineer enhanced notebook includes SageMaker Model Registry integration
- **Ground Truth Integration**: Data Scientist enhanced notebook includes complete Ground Truth labeling workflow
- **Artifact Management**: Automatic logging of visualizations, metrics, and model artifacts
- **Experiment Comparison**: Built-in functionality to compare experiments and model versions
- **Production Readiness**: Enhanced notebooks prepare models for production deployment workflows
- **Template-Based Development**: Notebooks serve as templates for creating role-specific workflows
- **Deployment Automation**: Notebooks are automatically deployed to SageMaker Studio with proper organization

### Automated Retraining Workflow
- **Drift Detection**: Configure Model Monitor with `DriftDetector` from `src.monitoring.drift_detection`
- **EventBridge Rules**: Set up rules using `EventBridgeIntegration` from `src.pipeline.event_bridge_integration`
- **Lambda Functions**: Use Lambda functions in `src.lambda_functions` directory:
  - `drift_detection/lambda_function.py`: Detects data drift and triggers retraining
  - `scheduled_evaluation/lambda_function.py`: Performs scheduled model evaluation
  - `model_approval/lambda_function.py`: Handles model approval workflows
- **Lambda Utilities**: Use `utils.py` for shared Lambda functionality like loading code and replacing placeholders
- **Approval Workflow**: Implement approval workflows with SNS notifications
- **Retraining Triggers**: Configure automated pipeline execution on drift detection

### MLOps Workflow
- **Data Preparation**: Use notebooks in `notebooks/data-exploration/` and `notebooks/data-labeling/`
- **Model Development**: Experiment in notebooks, then implement in `src/models/`
- **Pipeline Creation**: Define pipelines in `src/pipeline/` and execute with scripts in `scripts/training/`
- **Deployment**: Use CDK stacks in `configs/cdk/` for infrastructure deployment
- **Monitoring**: Set up monitoring with scripts in `scripts/monitoring/` and modules in `src/monitoring/`
- **Demo**: Use notebooks in `notebooks/demo/` for end-to-end demonstrations
- **Role-Based Workflows**: Separate workflows for Data Scientists and ML Engineers

### Cost Management Workflow
- **Cost Tracking**: Use `src/monitoring/cost_tracking.py` to monitor AWS resource costs
- **Budget Alerts**: Configure budget thresholds and alerts
- **Resource Optimization**: Use cost-optimized instance types and spot instances
- **Cleanup**: Ensure proper resource cleanup to minimize ongoing costs

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
- **Visualization**: Use functions from `mlflow_visualization.py` for experiment visualization

### Data Processing
- **S3 Access**: Use `S3DataAccess` class from `src.data.s3_utils`
- **Validation**: Use `YOLOv11Validator` for format compliance
- **Profiling**: Use `DroneImageryProfiler` for dataset analysis
- **Ground Truth**: Use functions from `ground_truth_utils.py` for labeling job management

### Ground Truth Operations
- **Job Creation**: Use `create_labeling_job_config` to configure labeling jobs
- **Job Monitoring**: Use `monitor_labeling_job` and `get_labeling_job_metrics` to track progress
- **Format Conversion**: Use `convert_ground_truth_to_yolo` to prepare data for training
- **Visualization**: Use `visualize_annotations` to inspect labeled data
- **Cost Management**: Use `estimate_labeling_cost` to control labeling expenses
- **Interactive Notebooks**: Use notebooks in `notebooks/data-labeling/` for interactive job creation

### Core SageMaker Operations
- **Deployment**: Use `scripts/setup/deploy_core_sagemaker.sh` with appropriate options
- **Notebook Deployment**: Use `scripts/setup/deploy_notebooks_to_studio.py` to deploy core notebooks to SageMaker Studio
- **Validation**: Use `scripts/setup/validate_core_sagemaker.py` to verify deployment
- **Cleanup**: Use `scripts/setup/cleanup_core_sagemaker.sh` to remove resources
- **Data Exploration**: Use `notebooks/data-scientist-core.ipynb` or `notebooks/data-scientist-core-enhanced.ipynb` for data analysis
- **Pipeline Management**: Use `notebooks/ml-engineer-core.ipynb` or `notebooks/ml-engineer-core-enhanced.ipynb` for pipeline execution

### Notebook Template System
- **Template Development**: Core notebooks serve as templates for role-specific workflows
- **Automated Deployment**: Notebooks are automatically uploaded to S3 with timestamped organization
- **Role-Based Access**: Separate deployment paths for Data Scientists and ML Engineers
- **Documentation Generation**: Automatic generation of role-specific README files and access instructions
- **SageMaker Studio Integration**: Direct deployment to SageMaker Studio user profiles with proper organization
- **Version Control**: Timestamped deployment ensures version tracking and rollback capabilities

### Role-Based Workflows
- **Data Scientists**: Use core notebooks for essential functionality or demo notebooks for comprehensive workflows:
  - **Core**: `notebooks/data-scientist-core.ipynb` for basic data exploration and preparation
  - **Enhanced**: `notebooks/data-scientist-core-enhanced.ipynb` for data exploration with MLFlow experiment tracking and Ground Truth labeling
  - **Comprehensive**: `notebooks/demo/data_scientist_workflow.ipynb` for complete workflow demonstration
  - **Documentation**: `notebooks/README_data-scientist.md` for role-specific usage instructions
  - Features: Data exploration and profiling with interactive widgets, Ground Truth labeling job creation and management, Model development with MLFlow experiment tracking, Model evaluation on test datasets
- **ML Engineers**: Use core notebooks for essential functionality or demo notebooks for comprehensive workflows:
  - **Core**: `notebooks/ml-engineer-core.ipynb` for basic pipeline execution and management
  - **Enhanced**: `notebooks/ml-engineer-core-enhanced.ipynb` for pipeline management with MLFlow and Model Registry integration
  - **Comprehensive**: `notebooks/demo/ml_engineer_workflow.ipynb` for complete workflow demonstration
  - **Documentation**: `notebooks/README_ml-engineer.md` for role-specific usage instructions
- **End-to-End Demo**: Use `notebooks/demo/end_to_end_mlops_workflow.ipynb` for complete workflow demonstration
- **Deployment Access**: Follow instructions in `SAGEMAKER_STUDIO_INSTRUCTIONS.md` for accessing notebooks in SageMaker Studio