# MLOps SageMaker Demo

A comprehensive demonstration project showcasing MLOps practices using Amazon SageMaker with YOLOv11 for drone detection. This project demonstrates end-to-end machine learning operations with governance, role-based access control, automated pipelines, and comprehensive monitoring.

## 🎯 Project Overview

This MLOps demonstration showcases:
- **Role-based governance** with separate Data Scientist and ML Engineer access
- **YOLOv11 object detection** for drone imagery analysis
- **Automated ML pipelines** using SageMaker Pipelines
- **Experiment tracking** with MLFlow integration
- **Model monitoring** and drift detection
- **Cost optimization** with AWS resource management

## 🏗️ Architecture

The project implements a complete MLOps architecture with clear separation of concerns:

- **Data Scientists**: Read-only access to datasets, model development in notebooks
- **ML Engineers**: Full pipeline development, deployment, and monitoring capabilities
- **Automated Governance**: IAM-based access control and audit logging

## 📁 Project Structure

```
├── configs/                    # Configuration and infrastructure
│   ├── project_config.py      # Centralized project configuration
│   ├── iam-roles-cloudformation.yaml  # IAM infrastructure as code
│   └── iam_roles_cdk.py       # CDK alternative for IAM setup
├── notebooks/                 # Jupyter notebooks for development
│   ├── data-exploration/      # Dataset analysis and profiling
│   │   ├── data-profiling.ipynb
│   │   └── dataset-analysis.ipynb
│   ├── model-development/     # Model training and experimentation
│   └── pipeline-development/  # Pipeline development notebooks
├── scripts/                   # Automation and setup scripts
│   ├── setup/                # Environment and AWS configuration
│   │   ├── configure_aws.sh   # AWS CLI profile setup
│   │   ├── deploy_iam_roles.sh # IAM roles deployment
│   │   └── validate_iam_roles.py # Role validation
│   └── training/             # Training execution scripts
├── src/                      # Source code modules
│   ├── data/                 # Data processing and validation
│   │   ├── s3_utils.py       # S3 data access utilities
│   │   ├── data_profiler.py  # Dataset profiling and analysis
│   │   ├── data_validator.py # YOLOv11 format validation
│   │   └── yolo_preprocessor.py # YOLOv11 preprocessing pipeline
│   ├── models/               # Model implementation modules
│   │   └── yolov11_trainer.py # YOLOv11 training implementation
│   └── pipeline/             # Pipeline orchestration modules
└── tests/                    # Unit tests
    ├── test_yolo_preprocessor.py # Comprehensive preprocessing tests
    └── test_yolov11_trainer.py   # YOLOv11 trainer unit tests
```

## 🚀 Quick Start

### Prerequisites

- AWS CLI installed and configured
- Python 3.10+ with pip
- Access to AWS account with appropriate permissions

### 1. Environment Setup

Configure AWS CLI with the 'ab' profile for cost tracking:
```bash
./scripts/setup/configure_aws.sh
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Deploy Infrastructure

Deploy IAM roles and policies:
```bash
./scripts/setup/deploy_iam_roles.sh
```

Validate role-based access:
```bash
python3 scripts/setup/validate_iam_roles.py --profile ab
```

### 3. Data Exploration

Start with data exploration notebooks:
```bash
jupyter notebook notebooks/data-exploration/data-profiling.ipynb
```

## 🔧 Configuration

### Central Configuration
All project settings are managed in `configs/project_config.py`:

- **AWS Profile**: `ab` (for cost allocation)
- **AWS Region**: `us-east-1`
- **Data Bucket**: `lucaskle-ab3-project-pv`
- **Model Framework**: YOLOv11 with PyTorch
- **Instance Types**: Optimized for cost and performance

### Key Configuration Parameters

```python
# Model Configuration
MODEL_NAME = "yolov11-drone-detection"
TRAINING_INSTANCE_TYPE = "ml.g4dn.xlarge"
INFERENCE_INSTANCE_TYPE = "ml.m5.large"

# MLFlow Configuration
MLFLOW_EXPERIMENT_NAME = "yolov11-drone-detection"
MLFLOW_ARTIFACT_BUCKET = "mlops-sagemaker-demo-mlflow-artifacts"

# Monitoring Configuration
DATA_CAPTURE_PERCENTAGE = 100
MONITORING_SCHEDULE_CRON = "cron(0 */6 * * ? *)"  # Every 6 hours
```

## 🛠️ Core Components

### Data Processing Pipeline
- **S3 Data Access**: Secure, role-based data access with comprehensive error handling
- **Data Profiling**: Automated analysis of drone imagery characteristics
- **Format Validation**: YOLOv11 format compliance checking
- **Preprocessing**: Image augmentation and format conversion

### YOLOv11 Implementation
- **Multi-format Support**: COCO, PASCAL VOC, YOLO, and custom JSON formats
- **Data Augmentation**: Comprehensive augmentation pipeline with Albumentations
- **Quality Validation**: Automated annotation quality assessment
- **Format Conversion**: Seamless conversion to YOLOv11 format

### Governance and Security
- **IAM Roles**: Separate roles for Data Scientists and ML Engineers
- **Access Control**: Fine-grained permissions with least privilege principle
- **Audit Logging**: Comprehensive activity logging for compliance
- **Cost Management**: Resource tagging and cost optimization

## 📊 Data Management

### Supported Data Formats
- **Images**: JPEG, PNG, BMP, TIFF
- **Annotations**: COCO JSON, PASCAL VOC XML, YOLO TXT, Custom JSON
- **Storage**: S3 with versioning and lifecycle management

### Data Validation Features
- Image quality assessment (brightness, contrast, sharpness)
- Annotation format validation
- Bounding box coordinate validation
- Dataset consistency analysis

## 🧪 Testing

Run comprehensive unit tests:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_yolo_preprocessor.py -v
```

### Test Coverage
- Data preprocessing functions
- Format conversion accuracy
- Augmentation pipeline functionality
- Validation logic
- Error handling scenarios

## 🔍 Usage Examples

### Data Profiling
```python
from src.data.s3_utils import S3DataAccess
from src.data.data_profiler import DroneImageryProfiler

# Initialize S3 access
s3_access = S3DataAccess("lucaskle-ab3-project-pv", aws_profile="ab")

# Profile dataset
profiler = DroneImageryProfiler(s3_access)
image_keys = s3_access.filter_objects_by_extension(['jpg', 'png'])
profile_data = profiler.profile_images(image_keys, sample_size=100)

# Generate report
report = profiler.generate_profile_report(profile_data)
print(report)
```

### YOLOv11 Preprocessing
```python
from src.data.yolo_preprocessor import YOLOv11Preprocessor, AnnotationFormat

# Initialize preprocessor
preprocessor = YOLOv11Preprocessor(
    s3_access=s3_access,
    target_size=640,
    class_names=["vehicle", "person", "building"]
)

# Convert annotations
annotations = preprocessor.convert_annotations_to_yolo(
    annotation_data, AnnotationFormat.COCO
)

# Validate quality
validation_results = preprocessor.validate_annotation_quality(annotations)
```

### Data Validation
```python
from src.data.data_validator import YOLOv11Validator

# Initialize validator
validator = YOLOv11Validator(s3_access)

# Validate dataset structure
image_files = s3_access.filter_objects_by_extension(['jpg', 'png'])
annotation_files = s3_access.filter_objects_by_extension(['txt'])

validation_results = validator.validate_dataset_structure(
    image_files, annotation_files
)
```

## 💰 Cost Optimization

### Resource Management
- **Spot Instances**: Used for training jobs where appropriate
- **Auto Scaling**: Inference endpoints scale based on demand
- **Scheduled Shutdown**: Development resources automatically stopped
- **Cost Monitoring**: Real-time cost tracking with "ab" profile tagging

### Cost Allocation Tags
```python
COST_ALLOCATION_TAGS = {
    "Project": "mlops-sagemaker-demo",
    "Environment": "Development",
    "Owner": "MLOps-Team",
    "CostCenter": "R&D"
}
```

## 🔐 Security and Governance

### IAM Role Structure
- **Data Scientist Role**: Read-only data access, notebook development
- **ML Engineer Role**: Full pipeline and deployment access
- **SageMaker Execution Role**: Service role for training and inference

### Security Features
- Encryption at rest and in transit
- VPC isolation for sensitive workloads
- IAM least privilege access
- Comprehensive audit logging

## 📈 Monitoring and Observability

### Model Monitoring
- Data drift detection with SageMaker Model Monitor
- Performance degradation alerts
- Automated baseline calculation
- Custom monitoring schedules

### Logging and Metrics
- CloudWatch integration for all components
- Custom metrics for business KPIs
- Distributed tracing for pipeline execution
- Real-time dashboards for stakeholders

## 🚧 Current Implementation Status

### ✅ Completed Components
- [x] Project structure and configuration
- [x] IAM roles and governance setup
- [x] S3 data access utilities
- [x] Data profiling and analysis tools
- [x] YOLOv11 preprocessing pipeline
- [x] Data validation framework
- [x] Comprehensive unit tests
- [x] Setup and deployment scripts

### 🔄 In Progress
- [ ] YOLOv11 training implementation
- [ ] MLFlow experiment tracking
- [ ] SageMaker Pipeline orchestration
- [ ] Model monitoring setup
- [ ] End-to-end integration

### 📋 Upcoming Features
- [ ] Model deployment automation
- [ ] Advanced monitoring dashboards
- [ ] Cost optimization automation
- [ ] Performance benchmarking
- [ ] Documentation and demos

## 🤝 Contributing

1. Follow the established project structure
2. Add comprehensive unit tests for new features
3. Update documentation for any changes
4. Use the configured AWS profile for all operations
5. Maintain security and governance standards

## 📚 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project Configuration Reference](configs/project_config.py)

## 🆘 Troubleshooting

### Common Issues

**AWS Profile Configuration**:
```bash
# Verify profile setup
aws sts get-caller-identity --profile ab

# Reconfigure if needed
aws configure --profile ab
```

**S3 Access Issues**:
```bash
# Test bucket access
aws s3 ls s3://lucaskle-ab3-project-pv --profile ab

# Check IAM permissions
python3 scripts/setup/validate_iam_roles.py --profile ab
```

**Python Dependencies**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check YOLOv11 installation
python -c "import ultralytics; print('YOLOv11 installed successfully')"
```

For additional support, refer to the comprehensive logging and error handling implemented throughout the codebase.