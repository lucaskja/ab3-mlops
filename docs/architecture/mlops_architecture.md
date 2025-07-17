# MLOps Architecture

This document describes the architecture of the MLOps SageMaker Demo project, focusing on the key components and their interactions.

## High-Level Architecture

The MLOps architecture follows the AWS reference implementation for MLOps with SageMaker, with enhancements for YOLOv11 object detection on drone imagery.

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
```

## Component Details

### 1. Data Layer

The data layer consists of S3 buckets for storing raw data, processed data, and model artifacts.

**S3 Bucket Structure:**
```
lucaskle-ab3-project-pv/
├── raw-images/                # Raw drone imagery
├── labeled-data/              # Ground Truth labeled data
├── processed-data/            # Preprocessed data for training
├── model-artifacts/           # Model artifacts from training
├── monitoring/                # Model monitoring data
└── ground-truth-jobs/         # Ground Truth job artifacts
```

### 2. Development Environment

The development environment is based on SageMaker Studio, providing Jupyter notebooks for data scientists and ML engineers.

**SageMaker Studio Components:**
- User profiles for Data Scientists and ML Engineers
- Jupyter notebooks for experimentation and development
- MLFlow integration for experiment tracking
- Ground Truth integration for data labeling

### 3. Pipeline Orchestration

SageMaker Pipelines are used to orchestrate the ML workflow, from data preprocessing to model deployment.

**Pipeline Steps:**
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

### 4. Model Management

Model management includes model registry, deployment, and monitoring.

**Model Registry:**
- Version tracking for models
- Approval workflow for production deployment
- Lineage tracking for model artifacts

**Model Deployment:**
- Auto-scaling endpoints for inference
- Blue/green deployment for zero-downtime updates
- A/B testing for model comparison

**Model Monitoring:**
- Data quality monitoring for drift detection
- Model quality monitoring for performance degradation
- Automated alerts for drift detection

### 5. Governance & Security

Governance and security are implemented through IAM roles, policies, and audit logging.

**Role-Based Access Control:**
```mermaid
graph TB
    subgraph "Data Scientist Role"
        DS_S3[Read-only S3 Access]
        DS_Studio[SageMaker Studio Access]
        DS_MLFlow[MLFlow Access]
        DS_GT[Ground Truth Access]
    end
    
    subgraph "ML Engineer Role"
        ML_S3[Read/Write S3 Access]
        ML_Pipeline[Pipeline Access]
        ML_Registry[Model Registry Access]
        ML_Endpoints[Endpoint Management]
        ML_Monitor[Monitoring Access]
    end
    
    subgraph "Resources"
        S3[S3 Buckets]
        Studio[SageMaker Studio]
        Pipeline[SageMaker Pipelines]
        Registry[Model Registry]
        Endpoints[SageMaker Endpoints]
        Monitor[Model Monitor]
    end
    
    DS_S3 --> S3
    DS_Studio --> Studio
    DS_MLFlow --> Studio
    DS_GT --> Studio
    
    ML_S3 --> S3
    ML_Pipeline --> Pipeline
    ML_Registry --> Registry
    ML_Endpoints --> Endpoints
    ML_Monitor --> Monitor
```

## Data Flow

The data flow through the MLOps pipeline follows these steps:

1. **Data Ingestion**: Raw drone imagery is stored in the S3 bucket
2. **Data Labeling**: Ground Truth is used to label the images
3. **Data Preprocessing**: Images are preprocessed for YOLOv11 training
4. **Model Training**: YOLOv11 models are trained with experiment tracking
5. **Model Evaluation**: Models are evaluated against validation data
6. **Model Registration**: Approved models are registered in the Model Registry
7. **Model Deployment**: Models are deployed to SageMaker endpoints
8. **Model Monitoring**: Deployed models are monitored for drift

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant GT as Ground Truth
    participant S3 as S3 Bucket
    participant Pipeline as SageMaker Pipeline
    participant Registry as Model Registry
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
    Registry->>Endpoint: Deploy model
    Endpoint->>Monitor: Configure monitoring
    Monitor->>Monitor: Detect drift
```

## Infrastructure as Code

The infrastructure is defined using AWS CDK with TypeScript, enabling version-controlled infrastructure deployment.

**CDK Stack Structure:**
```
configs/cdk/
├── bin/
│   └── app.ts                 # CDK app entry point
└── lib/
    ├── iam-stack.ts           # IAM roles and policies
    └── endpoint-stack.ts      # SageMaker endpoint configuration
```

## Monitoring and Observability

Monitoring and observability are implemented using SageMaker Model Monitor, CloudWatch, and EventBridge.

**Monitoring Components:**
- Data quality monitoring for input drift detection
- Model quality monitoring for performance degradation
- CloudWatch dashboards for visualization
- EventBridge rules for alerting

## Ground Truth Integration

Ground Truth is integrated for efficient data labeling, with custom workflows for drone imagery annotation.

**Labeling Workflow:**
```mermaid
graph LR
    A[Upload Images to S3] --> B[Create Labeling Job]
    B --> C[Configure Annotation Task]
    C --> D[Set Worker Type]
    D --> E[Monitor Job Progress]
    E --> F[Process Completed Labels]
    F --> G[Convert to YOLOv11 Format]
```

## Deployment Strategy

The deployment strategy follows a CI/CD approach with automated testing and validation.

**Deployment Pipeline:**
```mermaid
graph LR
    A[Code Commit] --> B[Build]
    B --> C[Test]
    C --> D[Deploy to Staging]
    D --> E[Validate]
    E --> F[Deploy to Production]
    F --> G[Monitor]
```

## Cost Optimization

Cost optimization is implemented through resource tagging, spot instances, and auto-scaling.

**Cost Optimization Strategies:**
- Use spot instances for training jobs
- Implement auto-scaling for inference endpoints
- Schedule shutdown of development resources
- Monitor costs with the "ab" profile tagging