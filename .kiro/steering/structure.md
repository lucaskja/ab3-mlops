# Project Structure

## Directory Organization

```
├── configs/                 # Configuration files and infrastructure
│   ├── project_config.py   # Centralized project configuration
│   ├── iam-roles-cloudformation.yaml  # IAM infrastructure
│   └── iam_roles_cdk.py    # CDK alternative for IAM setup
├── notebooks/              # Jupyter notebooks for development
│   ├── data-exploration/   # Data analysis and profiling notebooks
│   ├── model-development/  # Model training and experimentation
│   └── pipeline-development/ # Pipeline development notebooks
├── scripts/                # Utility and setup scripts
│   ├── setup/             # Environment and AWS setup scripts
│   └── training/          # Training execution scripts
└── src/                   # Source code modules
    ├── data/              # Data processing and validation
    ├── models/            # Model implementation modules
    └── pipeline/          # Pipeline orchestration modules
```

## Code Organization Patterns

### Configuration Management
- **Central Config**: All project settings in `configs/project_config.py`
- **Environment Variables**: AWS profile and region configuration
- **Constants**: Project names, bucket names, and resource identifiers

### Module Structure
- **Data Layer** (`src/data/`): S3 utilities, validation, and profiling
- **Model Layer** (`src/models/`): Model implementations and utilities
- **Pipeline Layer** (`src/pipeline/`): Orchestration and workflow management

### Naming Conventions
- **Files**: snake_case for Python files
- **Classes**: PascalCase (e.g., `DroneImageryProfiler`, `YOLOv11Validator`)
- **Functions**: snake_case with descriptive names
- **Constants**: UPPER_SNAKE_CASE in configuration files
- **AWS Resources**: Prefixed with project name (`mlops-sagemaker-demo-`)

### Import Patterns
- **Relative Imports**: Use relative imports within modules
- **Configuration**: Import from `configs.project_config`
- **Logging**: Configure at module level with `logging.getLogger(__name__)`

## File Conventions

### Python Files
- **Docstrings**: Comprehensive module and function documentation
- **Type Hints**: Use typing module for function signatures
- **Error Handling**: Comprehensive exception handling with logging
- **AWS Integration**: Use boto3 with profile-based authentication

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
3. Create notebook for experimentation in relevant `notebooks/` subdirectory
4. Add setup/deployment scripts in `scripts/` if needed

### AWS Resource Management
- **Infrastructure**: Define in CloudFormation templates
- **Deployment**: Use provided shell scripts with `ab` profile
- **Validation**: Run validation scripts after deployment

### Data Processing
- **S3 Access**: Use `S3DataAccess` class from `src.data.s3_utils`
- **Validation**: Use `YOLOv11Validator` for format compliance
- **Profiling**: Use `DroneImageryProfiler` for dataset analysis