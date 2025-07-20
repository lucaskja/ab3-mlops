# MLOps SageMaker Demo

A comprehensive MLOps demonstration using AWS SageMaker for YOLOv11 object detection on drone imagery, with governance, monitoring, and pipeline orchestration. This project follows the [AWS SageMaker MLOps reference implementation](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) with enhancements for computer vision workloads.

## Architecture Overview

This project implements a complete machine learning lifecycle from data ingestion to model deployment, with clear role separation between Data Scientists and ML Engineers, following AWS best practices for MLOps.

```mermaid
graph TB
    subgraph "Data Layer"
        S3[S3 Bucket: lucaskle-ab3-project-pv]
        S3_Artifacts[S3: Model Artifacts]
    end
    
    subgraph "Development Environment"
        Studio[SageMaker Studio]
        Notebooks[Jupyter Notebooks]
        MLFlow[MLFlow Tracking]
        GroundTruth[Ground Truth Labeling]
    end
    
    subgraph "Pipeline Orchestration"
        Pipeline[SageMaker Pipelines]
        Processing[Data Processing]
        Training[Model Training]
        Evaluation[Model Evaluation]
    end
    
    subgraph "Model Management"
        Registry[Model Registry]
        Endpoints[SageMaker Endpoints]
        Monitor[Model Monitor]
        Clarify[SageMaker Clarify]
    end
    
    subgraph "Governance & Security"
        IAM[IAM Roles & Policies]
        EventBridge[EventBridge]
        CloudWatch[CloudWatch Logs]
    end
    
    subgraph "CI/CD"
        Projects[SageMaker Projects]
        CodePipeline[AWS CodePipeline]
        CodeBuild[AWS CodeBuild]
        CodeCommit[AWS CodeCommit]
    end
    
    S3 --> Studio
    Studio --> Notebooks
    Notebooks --> MLFlow
    Notebooks --> GroundTruth
    GroundTruth --> S3
    Studio --> Pipeline
    Pipeline --> Processing
    Processing --> Training
    Training --> Evaluation
    Evaluation --> Registry
    Registry --> Endpoints
    Endpoints --> Monitor
    Monitor --> Clarify
    IAM --> Studio
    IAM --> Pipeline
    IAM --> GroundTruth
    EventBridge --> Monitor
    CloudWatch --> Monitor
    Projects --> CodePipeline
    CodePipeline --> CodeBuild
    CodePipeline --> CodeCommit
    Registry --> CodePipeline
    CodePipeline --> Endpoints
```

## Key Features

- **Data Management**: S3-based data access with validation and comprehensive image profiling
- **Data Profiling**: Advanced drone imagery analysis with quality metrics, recommendations, and visualization
- **Data Labeling**: Ground Truth integration for efficient dataset creation with automated YOLO format conversion
- **Model Development**: YOLOv11 implementation for drone detection
- **Pipeline Orchestration**: SageMaker Pipelines for automated workflows
- **Experiment Tracking**: MLFlow integration for model versioning
- **Monitoring**: Model performance and data drift monitoring
- **Governance**: IAM role-based access control for different team roles
- **Cost Optimization**: Resource tagging, spot instances, and auto-scaling
- **CI/CD Integration**: SageMaker Projects with CodePipeline for automated model building and deployment

## MLOps Workflow

This project follows the six-step process outlined in the [AWS SageMaker MLOps reference implementation](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production):

1. **Experiment in a notebook** - Initial development and experimentation with YOLOv11 for drone imagery
2. **Scale with SageMaker processing jobs and Python SDK** - Moving computation to SageMaker for data preprocessing
3. **Operationalize with ML pipeline and model registry** - Building automation with SageMaker Pipelines
4. **Add a model building CI/CD pipeline** - Automating the model building process
5. **Add a model deployment pipeline** - Automating model deployment with conditional steps
6. **Add model and data monitoring** - Ensuring ongoing quality with Model Monitor and EventBridge alerts

### Infrastructure Deployment

The project includes a comprehensive infrastructure deployment script (`scripts/setup/deploy_complete_infrastructure.sh`) that automates the setup of all required AWS resources. This script follows a modular approach, allowing you to:

- Deploy the complete infrastructure stack with a single command
- Skip specific components (monitoring, MLFlow, EventBridge, etc.) as needed
- Configure email notifications for alerts and monitoring
- Validate the deployment with automated checks
- Set up cost monitoring and budgets

The script handles the deployment of:
- IAM roles and policies for governance
- CDK stacks for SageMaker endpoints and other resources
- MLFlow tracking server on SageMaker
- SageMaker Projects for CI/CD
- Model deployment to SageMaker endpoints
- Model monitoring with SageMaker Model Monitor
- Explainability with SageMaker Clarify
- EventBridge rules for alerts and notifications
- Cost monitoring and budgets

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant GT as Ground Truth
    participant S3 as S3 Bucket
    participant Pipeline as SageMaker Pipeline
    participant Registry as Model Registry
    participant CI as CI/CD Pipeline
    participant Endpoint as SageMaker Endpoint
    participant Monitor as Model Monitor
    
    DS->>S3: Upload raw images
    DS->>GT: Create labeling job
    GT->>S3: Store labeled data
    S3->>Pipeline: Trigger pipeline execution
    Pipeline->>Pipeline: Preprocess data
    Pipeline->>Pipeline: Train YOLOv11 model
    Pipeline->>Pipeline: Evaluate model
    Pipeline->>Registry: Register model
    Registry->>CI: Trigger deployment pipeline
    CI->>CI: Deploy to staging
    CI->>CI: Run tests
    CI->>CI: Manual approval
    CI->>Endpoint: Deploy to production
    Endpoint->>Monitor: Configure monitoring
    Monitor->>Monitor: Detect drift
```

## Project Structure

```
├── configs/                 # Configuration files and infrastructure
│   ├── project_config.py    # Centralized project configuration
│   ├── environment_config.py # Environment-specific configuration
│   ├── cdk/                 # AWS CDK infrastructure code
│   ├── sagemaker_projects/  # SageMaker Projects templates and seed code
│   │   ├── templates/       # CloudFormation templates for SageMaker Projects
│   │   └── seed_code/       # Seed code for model building and deployment
│   └── environments/        # Environment-specific configurations
├── docs/                    # Comprehensive documentation
│   ├── architecture/        # Architecture diagrams and descriptions
│   ├── user-guides/         # Role-specific user guides
│   └── workflows/           # MLOps workflow documentation
├── examples/                # Example code and usage patterns
│   ├── data-labeling/       # Ground Truth labeling examples
│   ├── model-training/      # YOLOv11 training examples
│   └── pipeline/            # Pipeline orchestration examples
├── notebooks/               # Jupyter notebooks for development
│   ├── data-exploration/    # Data analysis and profiling notebooks
│   ├── data-labeling/       # Ground Truth labeling notebooks
│   ├── model-development/   # Model training and experimentation
│   └── pipeline-development/ # Pipeline development notebooks
├── scripts/                 # Utility and setup scripts
│   ├── setup/               # Environment and AWS setup scripts
│   ├── preprocessing/       # Data preprocessing scripts
│   ├── training/            # Training execution scripts
│   │   ├── prepare_model_registration.py # Model registration preparation
│   │   ├── train_yolov11.py # YOLOv11 training script
│   │   ├── evaluate_model.py # Model evaluation script
│   │   └── ...              # Other training scripts
│   ├── deployment/          # Model deployment scripts
│   └── monitoring/          # Monitoring and alerting scripts
├── src/                     # Source code modules
│   ├── data/                # Data processing and validation
│   ├── models/              # Model implementation modules
│   ├── pipeline/            # Pipeline orchestration modules
│   │   ├── components/      # Pipeline component classes
│   │   ├── templates/       # Script templates for pipeline steps
│   │   ├── script_templates.py # Template manager for generating scripts
│   │   └── ...              # Other pipeline modules
│   └── monitoring/          # Monitoring and observability modules
├── tests/                   # Unit and integration tests
├── mlruns/                  # Local MLFlow tracking data
└── logs/                    # Application logs
```

## Setup Instructions

### Prerequisites

- AWS account with appropriate permissions
- AWS CLI configured with "ab" profile
- Python 3.10+
- Docker (for local development)

### Dependencies

The project uses a simplified set of core dependencies:
- boto3 (≥1.28.0): AWS SDK for Python
- sagemaker (≥2.190.0): Amazon SageMaker Python SDK
- pandas (≥2.0.0): Data manipulation and analysis
- matplotlib (≥3.7.0) & seaborn (≥0.12.0): Data visualization
- numpy (≥1.24.0): Numerical computing
- PyYAML (≥6.0) & xmltodict (≥0.13.0): Configuration and data parsing

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-sagemaker-demo.git
cd mlops-sagemaker-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** The setup and deployment scripts automatically check for a virtual environment named `venv` in the project root directory. The scripts will use the Python interpreter from this environment if available, otherwise they will fall back to using the system's Python3.

> **Note:** The setup and deployment scripts now require a virtual environment named `venv` in the project root directory. The scripts will automatically check for this environment and use the Python interpreter from it if available, otherwise they will fall back to using the system's Python3.

#### Option 1: Complete Infrastructure Deployment (Recommended)

Use the complete infrastructure deployment script to set up all required AWS resources in one go:

```bash
./scripts/setup/deploy_complete_infrastructure.sh --email your.email@example.com
```

This script will automatically check for and use the Python interpreter from the `venv` virtual environment if available, otherwise it will fall back to using the system's Python3.

This script will:
- Configure AWS CLI with the "ab" profile
- Deploy IAM roles and policies with robust error handling
- Deploy CDK stacks for endpoints and other resources with fallback options (like using default inline Lambda code)
- Set up MLFlow on SageMaker
- Set up SageMaker Projects for CI/CD
- Deploy a model to a SageMaker endpoint
- Configure model monitoring and SageMaker Clarify
- Set up EventBridge rules and SNS notifications
- Configure cost monitoring and budgets
- Validate the deployment with graceful handling of errors and missing validation scripts

You can customize the deployment with various options:

```bash
./scripts/setup/deploy_complete_infrastructure.sh --help
```

The script supports additional options for more flexible deployments:

- `--skip-aws-config`: Skip the AWS CLI configuration prompt (useful for automated deployments)
- `--skip-stack-wait`: Skip waiting for CloudFormation stack updates (speeds up deployment significantly)
- `--skip-validation`: Skip validation steps for faster deployment
- `--skip-monitoring`: Skip model monitoring setup
- `--skip-mlflow`: Skip MLFlow setup
- `--skip-eventbridge`: Skip EventBridge setup
- `--skip-cost-monitoring`: Skip cost monitoring setup

For example, to deploy quickly without waiting for stack updates or AWS configuration prompts:

```bash
./scripts/setup/deploy_complete_infrastructure.sh --profile ab --skip-aws-config --skip-stack-wait
```

This will significantly speed up the deployment process by not waiting for CloudFormation stack updates to complete before proceeding to the next steps.

#### Option 2: Step-by-Step Manual Setup

If you prefer to set up components individually:

1. Configure AWS CLI with "ab" profile:
```bash
./scripts/setup/configure_aws.sh
```

2. Deploy IAM roles and policies:
```bash
./scripts/setup/deploy_iam_roles.sh
```

   The script includes error handling that allows the process to continue even when non-critical errors occur, such as when no updates are needed for an existing stack. The script now supports the `--skip-wait` flag to skip waiting for CloudFormation stack updates, which can speed up deployment.

3. Validate the IAM setup (if the validation script exists):
```bash
venv/bin/python scripts/setup/validate_iam_roles.py --profile ab
```

   The script will display the command being executed and then run the validation.

4. Deploy CDK infrastructure:
```bash
./scripts/setup/deploy_cdk.sh --profile ab
```

This script will:
- Load project configuration using the virtual environment Python if available
- Install CDK dependencies if needed
- Check for Lambda code and use default inline code if not found (improved error handling)
- Synthesize the CDK stack to check for errors before deployment
- Deploy the stack with the necessary parameters
- Update the project configuration with the Lambda function ARN

   The script now includes improved error handling that allows deployment to continue even when Lambda code is not found at the specified path, using default inline Lambda code instead.

   You can also use the `--skip-wait` option to skip waiting for CloudFormation stack updates, which can speed up the deployment process significantly.

5. Set up SageMaker Projects for CI/CD:
```bash
venv/bin/python scripts/setup/setup_sagemaker_project.py --profile ab
```

## Governance and Role-Based Access Control

This project implements strict role separation through IAM roles and policies:

### Data Scientist Role

- Read-only access to raw data in S3
- Full access to SageMaker Studio notebooks
- Access to MLFlow for experiment tracking
- Permissions to create Ground Truth labeling jobs
- No access to production resources

```mermaid
graph TB
    subgraph "Data Scientist Role"
        DS_S3[Read-only S3 Access]
        DS_Studio[SageMaker Studio Access]
        DS_MLFlow[MLFlow Access]
        DS_GT[Ground Truth Access]
    end
    
    subgraph "Resources"
        S3[S3 Buckets]
        Studio[SageMaker Studio]
    end
    
    DS_S3 --> S3
    DS_Studio --> Studio
    DS_MLFlow --> Studio
    DS_GT --> Studio
```

### ML Engineer Role

- Full access to SageMaker Pipelines
- Access to Model Registry and deployment resources
- Permission to create and manage endpoints
- Access to monitoring and production resources

```mermaid
graph TB
    subgraph "ML Engineer Role"
        ML_S3[Read/Write S3 Access]
        ML_Pipeline[Pipeline Access]
        ML_Registry[Model Registry Access]
        ML_Endpoints[Endpoint Management]
        ML_Monitor[Monitoring Access]
    end
    
    subgraph "Resources"
        S3[S3 Buckets]
        Pipeline[SageMaker Pipelines]
        Registry[Model Registry]
        Endpoints[SageMaker Endpoints]
        Monitor[Model Monitor]
    end
    
    ML_S3 --> S3
    ML_Pipeline --> Pipeline
    ML_Registry --> Registry
    ML_Endpoints --> Endpoints
    ML_Monitor --> Monitor
```

## Usage Guides

### Data Scientists

1. **Data Exploration**: Use notebooks in `notebooks/data-exploration/` to analyze the drone imagery dataset.
   - Utilize the `DroneImageryProfiler` to analyze image characteristics, quality metrics, and get recommendations
   - Generate comprehensive profile reports with resolution, brightness, contrast, sharpness, and color diversity metrics
2. **Data Labeling**: Create and manage Ground Truth labeling jobs using notebooks in `notebooks/data-labeling/`.
3. **Model Development**: Experiment with YOLOv11 models using notebooks in `notebooks/model-development/`.

For detailed instructions, see the [Data Scientist Guide](docs/user-guides/data_scientist_guide.md).

### ML Engineers

1. **Pipeline Development**: Create and modify SageMaker Pipelines using code in `src/pipeline/`.
2. **Model Deployment**: Deploy models to endpoints using the deployment scripts.
3. **Monitoring Setup**: Configure model monitoring using the monitoring modules.
4. **CI/CD Management**: Manage SageMaker Projects and CI/CD pipelines for automated model building and deployment.

For detailed instructions, see the [ML Engineer Guide](docs/user-guides/ml_engineer_guide.md).

## Data Profiling and Analysis

The project includes a comprehensive `DroneImageryProfiler` for analyzing drone imagery datasets:

```python
from src.data.data_profiler import DroneImageryProfiler
from src.data.s3_utils import S3DataAccess

# Initialize S3 access and profiler
s3_access = S3DataAccess(bucket_name="lucaskle-ab3-project-pv", profile_name="ab")
profiler = DroneImageryProfiler(s3_access)

# Profile images (with optional sampling)
image_keys = s3_access.list_objects(prefix="raw-images/")
profile_data = profiler.profile_images(image_keys, sample_size=100)

# Generate human-readable report
report = profiler.generate_profile_report(profile_data)
print(report)

# Get data quality recommendations
recommendations = profiler.get_recommendations(profile_data)
for rec in recommendations:
    print(f"- {rec}")
```

The profiler analyzes:
- Image dimensions and aspect ratios
- File sizes and formats
- Color modes and diversity
- Brightness, contrast, and sharpness
- Quality metrics and error rates

Based on the analysis, it provides actionable recommendations for data preprocessing and model training optimization.

## Ground Truth Labeling Workflow

The project includes a comprehensive workflow for creating and managing SageMaker Ground Truth labeling jobs:

```mermaid
graph LR
    A[Upload Images to S3] --> B[Create Labeling Job]
    B --> C[Configure Annotation Task]
    C --> D[Set Worker Type]
    D --> E[Monitor Job Progress]
    E --> F[Process Completed Labels]
    F --> G[Convert to YOLOv11 Format]
```

To create a labeling job:

```bash
# Run the Ground Truth example script
venv/bin/python examples/data-labeling/ground_truth_example.py --bucket lucaskle-ab3-project-pv --prefix raw-images
```

For more detailed control, use the interactive Jupyter notebook in `notebooks/data-labeling/create_labeling_job_interactive.ipynb`.

## SageMaker Pipeline Implementation

The project implements a comprehensive SageMaker Pipeline for YOLOv11 training and deployment:

```mermaid
graph LR
    A[Data Preprocessing] --> B[Data Validation]
    B --> C[Model Training]
    C --> D[Model Evaluation]
    D --> E[Model Registration]
    E --> F[Conditional Deployment]
    F --> G[Endpoint Configuration]
    G --> H[Model Monitoring Setup]
```

### Model Registration Process

The model registration process prepares models for the SageMaker Model Registry with evaluation metrics and approval status:

```python
# Example of model registration preparation
from scripts.training.prepare_model_registration import prepare_model_registration

# Prepare model for registration
model_package_info = prepare_model_registration({
    'profile': 'ab',
    'model_info': 'model_info.json',
    'evaluation_results': 'evaluation_results.json',
    'output': 'model_package_info.json'
})

# The model_package_info contains:
# - model_name: Name of the trained model
# - model_artifact_path: S3 path to model artifacts
# - inference_image: Docker image for inference
# - model_metrics: Performance metrics (mAP, precision, recall)
# - approval_status: Set to 'PendingManualApproval' by default
# - created_at: Timestamp of creation
```

This process ensures that all models are registered with consistent metadata and performance metrics, enabling proper governance and approval workflows.

### Template-Based Script Generation

The pipeline uses a file-based template system for generating processing scripts. Templates are stored in the `src/pipeline/templates/` directory and loaded dynamically by the `ScriptTemplateManager`:

```python
from src.pipeline.script_templates import script_template_manager

# Generate a preprocessing script with custom logic
preprocessing_script = script_template_manager.generate_preprocessing_script(
    preprocessing_logic="""
    # Custom preprocessing logic for YOLOv11 format
    import cv2
    from glob import glob
    
    # Process images and annotations
    image_files = glob(os.path.join(input_path, "*.jpg"))
    for image_file in image_files:
        # Process each image
        img = cv2.imread(image_file)
        # Apply preprocessing transformations
        # ...
    """,
    additional_args='parser.add_argument("--image-size", type=int, default=640)',
    kwargs_extraction='"image_size": args.image_size'
)
```

This approach allows for flexible script generation while maintaining consistent structure and error handling.

To execute the pipeline:

```python
from src.pipeline.sagemaker_pipeline_factory import PipelineFactory

# Create pipeline factory
factory = PipelineFactory(aws_profile="ab", region="us-east-1")

# Create complete pipeline
pipeline = factory.create_complete_pipeline(
    pipeline_name="yolov11-training-pipeline",
    preprocessing_script="scripts/preprocessing/preprocess_yolo_data.py",
    training_script="scripts/training/train_yolov11.py",
    evaluation_script="scripts/training/evaluate_model.py",
    input_data="s3://lucaskle-ab3-project-pv/raw-data/",
    model_name="yolov11-drone-detection",
    instance_type_training="ml.g4dn.xlarge"
)

# Execute pipeline
execution = factory.execute_pipeline(pipeline)
```

## CI/CD with SageMaker Projects

The project implements CI/CD using SageMaker Projects, which provides automated model building and deployment pipelines:

```mermaid
graph TB
    subgraph "Model Building Pipeline"
        Source[Source Code Repository]
        Build[Build Model]
        Test[Test Model]
        Register[Register Model]
    end
    
    subgraph "Model Deployment Pipeline"
        ModelSource[Model Source]
        StagingDeploy[Deploy to Staging]
        StagingTest[Test in Staging]
        Approval[Manual Approval]
        ProdDeploy[Deploy to Production]
    end
    
    Source --> Build
    Build --> Test
    Test --> Register
    Register --> ModelSource
    ModelSource --> StagingDeploy
    StagingDeploy --> StagingTest
    StagingTest --> Approval
    Approval --> ProdDeploy
```

### Setting Up SageMaker Projects

To set up SageMaker Projects for CI/CD:

```bash
# Set up SageMaker Project
venv/bin/python scripts/setup/setup_sagemaker_project.py --profile ab
```

This script:
1. Creates seed code zip files for model building and deployment
2. Uploads templates and seed code to S3
3. Creates Service Catalog products for model building and deployment
4. Creates a SageMaker Project that uses these products

### Model Building Pipeline

The model building pipeline:
1. Pulls code from a CodeCommit repository
2. Builds the model using SageMaker Pipelines
3. Tests the model against quality thresholds
4. Prepares model registration information with evaluation metrics
5. Registers the model in the SageMaker Model Registry with approval status

### Model Deployment Pipeline

The model deployment pipeline:
1. Pulls the latest approved model from the Model Registry
2. Deploys the model to a staging endpoint
3. Tests the model in the staging environment
4. Requires manual approval for production deployment
5. Deploys the model to the production endpoint

### CI/CD Workflow

The CI/CD workflow is triggered by:
1. Code changes in the model building repository
2. New approved models in the Model Registry

This ensures that both code changes and model updates flow through the CI/CD pipeline, maintaining quality and governance.

## Model Monitoring and Drift Detection

The project implements comprehensive model monitoring using SageMaker Model Monitor:

- **Data Quality Monitoring**: Detect drift in input data distributions
- **Model Quality Monitoring**: Track model performance metrics over time
- **Bias Monitoring**: Detect bias in model predictions using SageMaker Clarify
- **Feature Attribution Drift**: Monitor changes in feature importance

To set up monitoring:

```python
from src.pipeline.model_monitor import setup_model_monitoring

# Set up model monitoring
setup_model_monitoring(
    endpoint_name="yolov11-endpoint",
    baseline_dataset="s3://lucaskle-ab3-project-pv/baseline-data/",
    monitoring_schedule_name="yolov11-monitoring-schedule"
)
```

## Cost Optimization

This project implements several cost optimization strategies:

1. **Spot Instances**: Use spot instances for training jobs to reduce costs by up to 90%
2. **Auto-scaling**: Configure auto-scaling for inference endpoints to match demand
3. **Resource Scheduling**: Automatically shut down development resources when not in use
4. **Cost Monitoring**: Track costs using the "ab" AWS CLI profile and Cost Explorer
5. **Budget Alerts**: Set up budget thresholds and email notifications

```python
# Example of spot instance configuration for training
from sagemaker.estimator import EstimatorBase

estimator = EstimatorBase(
    role="SageMakerRole",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    use_spot_instances=True,  # Enable spot instances
    max_wait=36000,  # Maximum time to wait for spot instances
    max_run=3600,    # Maximum training time
)
```

For automated cost monitoring setup, use the `deploy_complete_infrastructure.sh` script with the email notification option:

```bash
./scripts/setup/deploy_complete_infrastructure.sh --email your.email@example.com
```

This will set up a cost monitoring dashboard and create a monthly budget with alerts when you reach 80% of your budget threshold.

For detailed cost monitoring, use the cost tracking functions in `src/monitoring/cost_tracking.py`.

## Testing and Validation

The project includes comprehensive unit and integration tests for all components:

```bash
# Run all tests
venv/bin/python -m unittest discover tests

# Run specific test file
venv/bin/python -m unittest tests/test_data_profiler.py
```

Key test modules include:
- `test_data_profiler.py`: Tests for the DroneImageryProfiler class
- `test_yolo_preprocessor.py`: Tests for YOLO data preprocessing
- `test_ground_truth_utils.py`: Tests for Ground Truth integration
- `test_model_monitor.py`: Tests for model monitoring functionality
- `test_drift_detection.py`: Tests for data drift detection
- `test_pipeline_integration.py`: End-to-end tests for SageMaker Pipeline functionality
- `test_endpoint_performance.py`: Tests for endpoint performance and latency

### Pipeline Integration Testing

The project includes comprehensive integration tests for the SageMaker Pipeline implementation in `tests/test_pipeline_integration.py`. These tests validate:

1. **End-to-End Pipeline Execution**: Tests the complete pipeline workflow from creation to execution
2. **Component Integration**: Verifies proper integration between preprocessing, training, and evaluation steps
3. **Custom Pipeline Components**: Tests the ability to create pipelines with custom component configurations
4. **Error Handling**: Validates pipeline error detection and reporting
5. **MLFlow Integration**: Tests integration between SageMaker Pipelines and MLFlow tracking

The tests use mocking to simulate AWS services, allowing for comprehensive testing without actual AWS resource creation:

```python
# Example of pipeline integration test
def test_pipeline_component_integration(self):
    """Test integration between pipeline components"""
    # Create pipeline factory
    factory = SageMakerPipelineFactory(
        aws_profile="ab",
        region="us-east-1",
        config=self.config
    )
    
    # Create preprocessing step
    preprocessing_step = factory.create_preprocessing_step(
        input_data="s3://test-bucket/input-data"
    )
    
    # Create training step
    training_step = factory.create_training_step(
        preprocessed_data=preprocessing_step.get_output()
    )
    
    # Create evaluation step
    evaluation_step = factory.create_evaluation_step(
        model=training_step.get_model(),
        test_data=preprocessing_step.get_test_data_output()
    )
    
    # Verify step dependencies
    self.assertEqual(training_step.get_dependencies()[0], preprocessing_step.get_step())
    self.assertEqual(evaluation_step.get_dependencies()[0], training_step.get_step())
```

These tests ensure that the pipeline components work together correctly and that the pipeline can handle various configurations and error conditions.

## Deployment Optimization

To speed up the deployment process, especially during development and testing, you can use these flags:

### IAM Roles Deployment
```bash
# Skip waiting for CloudFormation stack updates
./scripts/setup/deploy_iam_roles.sh --skip-wait
```

### Complete Infrastructure Deployment
```bash
# Skip waiting for all CloudFormation stack updates
./scripts/setup/deploy_complete_infrastructure.sh --skip-stack-wait
```

These flags significantly speed up deployment by not waiting for CloudFormation operations to complete before proceeding to the next steps. This is particularly useful during development when you're making frequent changes to the infrastructure.

## Cleanup Procedures

To avoid ongoing costs, run the cleanup script when you're done:

```bash
./scripts/setup/cleanup_resources.sh --profile ab
```

This script will:

1. Delete all SageMaker endpoints
2. Terminate all SageMaker notebook instances
3. Stop all SageMaker training jobs
4. Delete all SageMaker models
5. Delete all CloudWatch alarms
6. Delete all EventBridge rules
7. Delete all CloudFormation stacks
8. Delete all CodePipeline pipelines
9. Delete all CodeBuild projects
10. Delete all CodeCommit repositories

You can customize the cleanup with various options:

```bash
./scripts/setup/cleanup_resources.sh --help
```

Before running the cleanup script, verify that you want to delete all resources by running:

```bash
venv/bin/python scripts/setup/validate_cleanup.py --profile ab
```

## AWS Documentation References

For more information on the AWS services used in this project, refer to the following documentation:

- [Implement MLOps with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/mlops.html)
- [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-mlops.html)
- [SageMaker Ground Truth](https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html)
- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker MLOps Project Templates](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-templates.html)
- [SageMaker Projects](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)
- [AWS CodePipeline](https://docs.aws.amazon.com/codepipeline/latest/userguide/welcome.html)
- [AWS CodeBuild](https://docs.aws.amazon.com/codebuild/latest/userguide/welcome.html)
- [AWS CodeCommit](https://docs.aws.amazon.com/codecommit/latest/userguide/welcome.html)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AWS SageMaker team for their [MLOps reference implementation](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production)
- Ultralytics for the YOLOv11 object detection framework