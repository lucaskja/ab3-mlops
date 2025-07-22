# ML Engineer Notebooks - Enhanced Workflow Guide

Core notebooks for ML Engineers with comprehensive MLflow integration, automated validation, deployment automation, and advanced performance analytics.

## 🚀 Quick Start

### Prerequisites Checklist
- ✅ AWS CLI configured with "ab" profile
- ✅ Access to S3 bucket: `lucaskle-ab3-project-pv`
- ✅ MLEngineerRole IAM permissions
- ✅ SageMaker managed MLflow tracking server access
- ✅ YOLOv11 training container in ECR (optional - will show instructions if missing)

### Opening the Enhanced Notebook
1. **Navigate to**: `/home/sagemaker-user/ml-engineer-notebooks/`
2. **Open**: `ml-engineer-core-enhanced.ipynb`
3. **Select Kernel**: Python 3 (Data Science)
4. **Run cells sequentially** from top to bottom

## 📋 What the ML Engineer Creates

When you run the enhanced notebook, you'll create and manage:

### 1. Training Infrastructure
- **SageMaker Training Jobs**: YOLOv11 model training with spot instances
- **MLflow Experiments**: Comprehensive experiment tracking
- **Model Artifacts**: Stored in S3 with proper versioning

### 2. Model Registry Assets
- **Model Package Groups**: Organized model version collections
- **Model Versions**: Each training run creates a new model version
- **Approval Workflows**: Manual approval gates for production deployment

### 3. Automated Validation Pipeline
- **Performance Thresholds**: Automated quality gates (mAP > 0.5, loss < 2.0)
- **Model Evaluation**: Comprehensive validation before registration
- **Quality Reports**: Detailed validation results and recommendations

### 4. Deployment Infrastructure (New!)
- **SageMaker Endpoints**: Auto-scaling inference endpoints for approved models
- **Blue/Green Deployments**: Zero-downtime model updates
- **Endpoint Configurations**: Optimized for YOLOv11 inference

### 5. Performance Analytics (New!)
- **Model Comparison Reports**: Multi-dimensional performance analysis
- **Statistical Analysis**: Trend analysis and hyperparameter correlation
- **Automated Recommendations**: Data-driven model selection guidance

## 🔍 Where to Monitor Progress

### 1. SageMaker Console Monitoring

#### Training Jobs
- **Location**: SageMaker Console → Training → Training jobs
- **What to Check**: 
  - Job status (InProgress/Completed/Failed)
  - Training metrics and logs
  - Resource utilization and costs
- **Job Naming**: `yolov11-training-YYYY-MM-DD-HH-MM-SS`

#### Model Registry
- **Location**: SageMaker Console → Inference → Model registry
- **What to Check**:
  - Model package groups: `yolov11-drone-detection-models`
  - Model versions and approval status
  - Model metadata and lineage
- **Key Metrics**: Creation time, approval status, associated training job

#### Endpoints (After Deployment)
- **Location**: SageMaker Console → Inference → Endpoints
- **What to Check**:
  - Endpoint status and health
  - Auto-scaling configuration
  - Invocation metrics and latency
- **Endpoint Naming**: `yolov11-endpoint-YYYY-MM-DD-HH-MM-SS`

### 2. MLflow Tracking UI

#### Accessing MLflow
1. **In SageMaker Studio**: Go to "Experiments and trials" → Click "MLflow"
2. **Direct Access**: Use tracking server ARN in notebook
3. **Experiment Name**: `yolov11-drone-detection`

#### What to Monitor
- **Experiments**: All YOLOv11 training experiments
- **Runs**: Individual training runs with parameters and metrics
- **Artifacts**: Model files, training logs, and visualizations
- **Model Registry**: Registered models with versions and stages

#### Key Metrics to Track
- `final_map50`: mAP@0.5 performance metric
- `final_map50_95`: mAP@0.5:0.95 performance metric
- `training_duration_minutes`: Training efficiency
- `final_val_loss`: Validation loss

### 3. CloudWatch Monitoring

#### Training Metrics
- **Location**: CloudWatch Console → Metrics → SageMaker
- **Metrics to Watch**:
  - `train:loss`: Training loss over time
  - `val:loss`: Validation loss over time
  - `val:mAP50`: Validation mAP@0.5 over time
  - `val:mAP50-95`: Validation mAP@0.5:0.95 over time

#### Endpoint Metrics (After Deployment)
- **Invocations**: Number of inference requests
- **ModelLatency`: Inference response time
- **OverheadLatency`: Endpoint overhead time
- **Errors`: Failed inference requests

### 4. S3 Storage Monitoring

#### Model Artifacts
- **Location**: `s3://lucaskle-ab3-project-pv/model-artifacts/`
- **Contents**: 
  - Trained model files (`model.tar.gz`)
  - Training outputs and logs
  - Model evaluation results

#### Dataset Access
- **Location**: `s3://lucaskle-ab3-project-pv/datasets/`
- **Usage**: Training data input for pipeline execution

## 🔄 Complete Workflow Execution

### Phase 1: Setup and Configuration (Cells 1-2)
**What Happens:**
- MLflow connection to SageMaker managed server
- Model Registry setup and package group creation
- Dataset discovery and validation

**Monitor At:**
- Notebook output for connection status
- SageMaker Console → Model Registry for package group creation

### Phase 2: Training Execution (Cells 3-4)
**What Happens:**
- SageMaker training job submission with MLflow tracking
- Real-time job monitoring and status updates
- Automatic parameter and metadata logging

**Monitor At:**
- SageMaker Console → Training jobs for job status
- MLflow UI for experiment tracking
- CloudWatch for training metrics

### Phase 3: Automated Validation (Cell 5)
**What Happens:**
- Automated model performance validation
- Quality gate evaluation (mAP thresholds, loss limits)
- Validation report generation

**Monitor At:**
- Notebook output for validation results
- MLflow UI for validation metrics and tags

### Phase 4: Model Registration (Cell 6)
**What Happens:**
- Automatic model registration in SageMaker Model Registry
- Metadata association and tagging
- Approval workflow initiation

**Monitor At:**
- SageMaker Console → Model Registry for new model versions
- MLflow UI for model registration confirmation

### Phase 5: Automated Deployment (Cell 7)
**What Happens:**
- Automatic endpoint creation for approved models
- Blue/green deployment configuration
- Auto-scaling setup and health checks

**Monitor At:**
- SageMaker Console → Endpoints for deployment status
- CloudWatch for endpoint metrics
- Notebook output for deployment confirmation

### Phase 6: Performance Analysis (Cells 8-9)
**What Happens:**
- Multi-model performance comparison
- Statistical analysis and trend identification
- Automated recommendations generation

**Monitor At:**
- Notebook visualizations and analysis output
- MLflow UI for comparative metrics
- Generated performance reports

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### 1. MLflow Connection Failed
**Symptoms**: "Could not connect to SageMaker managed MLflow"
**Solutions**:
- Verify IAM role has `sagemaker-mlflow:*` permissions
- Check tracking server ARN is correct
- Fallback to local MLflow will activate automatically

#### 2. Training Job Failed
**Symptoms**: Job status shows "Failed" in SageMaker Console
**Check**:
- CloudWatch logs for detailed error messages
- ECR repository for YOLOv11 training container
- S3 dataset structure and permissions
**Common Fixes**:
- Ensure dataset follows YOLOv11 format
- Verify sufficient IAM permissions
- Check instance type availability in region

#### 3. Model Registration Failed
**Symptoms**: "Error registering model" in notebook output
**Solutions**:
- Ensure training job completed successfully
- Verify Model Registry permissions
- Check inference container exists in ECR

#### 4. Deployment Failed
**Symptoms**: Endpoint creation fails or shows "Failed" status
**Solutions**:
- Verify model is approved in Model Registry
- Check inference container and model artifacts
- Ensure sufficient service limits for endpoints

#### 5. No Performance Data
**Symptoms**: Empty comparison tables or "No data available"
**Solutions**:
- Ensure training jobs completed successfully
- Check CloudWatch metrics are being published
- Verify MLflow tracking is working correctly

## 📊 Success Indicators

### Training Success
- ✅ Training job status: "Completed"
- ✅ Final mAP@0.5 > 0.5 (configurable threshold)
- ✅ Final validation loss < 2.0 (configurable threshold)
- ✅ MLflow run logged with all parameters and metrics

### Validation Success
- ✅ Automated validation passes all quality gates
- ✅ Model performance meets or exceeds baseline
- ✅ Validation report shows "PASSED" status

### Registration Success
- ✅ Model appears in SageMaker Model Registry
- ✅ Model status: "PendingManualApproval" or "Approved"
- ✅ All metadata and tags properly associated

### Deployment Success
- ✅ Endpoint status: "InService"
- ✅ Health checks passing
- ✅ Auto-scaling configuration active
- ✅ Sample inference requests successful

### Analytics Success
- ✅ Performance comparison charts generated
- ✅ Statistical analysis completed
- ✅ Model recommendations provided
- ✅ Trend analysis shows improvement over time

## 🎯 Best Practices

### 1. Resource Management
- Use spot instances for training to reduce costs
- Monitor training duration and set appropriate timeouts
- Clean up failed or unnecessary endpoints

### 2. Model Governance
- Always review model validation results before approval
- Document approval decisions in Model Registry
- Maintain clear model versioning strategy

### 3. Performance Monitoring
- Regularly run performance comparison analysis
- Track model performance trends over time
- Set up alerts for performance degradation

### 4. Cost Optimization
- Use appropriate instance types for workload
- Monitor billable training time
- Implement automatic endpoint scaling

## 📞 Support and Resources

### Getting Help
1. **Notebook Issues**: Check cell outputs and error messages
2. **AWS Service Issues**: Check AWS Service Health Dashboard
3. **MLflow Issues**: Verify tracking server connectivity
4. **Performance Issues**: Review CloudWatch metrics and logs

### Additional Resources
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Project Repository](../README.md)

---

**Last Updated**: 2025-07-22  
**Notebook Version**: Enhanced with Automated Validation, Deployment, and Performance Analytics  
**MLflow Integration**: SageMaker Managed Tracking Server  
**Deployment Target**: Production-Ready with Blue/Green Strategy
