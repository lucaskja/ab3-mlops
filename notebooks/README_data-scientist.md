# Data Scientist Notebooks - Enhanced Workflow Guide

Core notebooks for Data Scientists with comprehensive MLflow integration, dataset preparation, and seamless ML Engineer workflow integration.

## 🚀 Quick Start

### Prerequisites Checklist
- ✅ AWS CLI configured with "ab" profile
- ✅ Access to S3 bucket: `lucaskle-ab3-project-pv`
- ✅ DataScientistRole IAM permissions (read-only for production resources)
- ✅ SageMaker managed MLflow tracking server access
- ✅ Ground Truth labeling permissions (optional)

### Opening the Enhanced Notebook
1. **Navigate to**: `/home/sagemaker-user/data-scientist-notebooks/`
2. **Open**: `data-scientist-core-enhanced.ipynb`
3. **Select Kernel**: Python 3 (Data Science)
4. **Run cells sequentially** from top to bottom

## 📋 What the Data Scientist Creates

When you run the enhanced notebook, you'll create and manage:

### 1. Dataset Exploration and Analysis
- **MLflow Experiments**: Comprehensive data exploration tracking
- **Data Quality Reports**: Statistical analysis and validation results
- **Visualization Artifacts**: Charts, plots, and analysis visualizations
- **Dataset Profiling**: Image characteristics, quality metrics, and recommendations

### 2. YOLOv11-Ready Datasets (Key Output for ML Engineers!)
- **Structured Datasets**: Properly formatted for YOLOv11 training
- **Dataset Location**: `s3://lucaskle-ab3-project-pv/datasets/yolov11_dataset_TIMESTAMP/`
- **Required Structure**:
  ```
  s3://lucaskle-ab3-project-pv/datasets/yolov11_dataset_TIMESTAMP/
  ├── train/
  │   ├── images/     ✅ Training images
  │   └── labels/     ✅ Training labels (.txt files)
  ├── val/
  │   ├── images/     ✅ Validation images  
  │   └── labels/     ✅ Validation labels (.txt files)
  ├── data.yaml       ✅ Dataset configuration
  └── dataset_info.json ✅ Complete metadata
  ```

### 3. Ground Truth Labeling Jobs (Optional)
- **Labeling Jobs**: SageMaker Ground Truth object detection tasks
- **Annotation Workflows**: Efficient drone imagery labeling
- **Quality Control**: Automated validation and consistency checks
- **Cost Management**: Budget controls and progress monitoring

### 4. Data Preparation Artifacts
- **Image Preprocessing**: Optimized images for training
- **Label Conversion**: Annotations in YOLOv11 format
- **Data Splits**: Proper train/validation separation
- **Metadata Files**: Complete dataset documentation

## 🔍 Where to Monitor Progress

### 1. MLflow Tracking UI

#### Accessing MLflow
1. **In SageMaker Studio**: Go to "Experiments and trials" → Click "MLflow"
2. **Experiment Name**: `drone-imagery-data-exploration`
3. **What to Monitor**:
   - Data exploration runs with parameters and metrics
   - Dataset preparation progress and results
   - Image analysis artifacts and visualizations
   - Data quality metrics and validation results

### 2. S3 Storage Monitoring

#### Dataset Creation Progress
- **Location**: `s3://lucaskle-ab3-project-pv/datasets/`
- **What to Check**:
  - New dataset directories with timestamps
  - Complete folder structure (train/, val/, files)
  - Image and label file counts
  - Metadata files (data.yaml, dataset_info.json)

#### Raw Data Access
- **Location**: `s3://lucaskle-ab3-project-pv/raw-data/`
- **Usage**: Source images for exploration and preparation

### 3. Ground Truth Console (If Using Labeling)

#### Labeling Jobs
- **Location**: SageMaker Console → Ground Truth → Labeling jobs
- **What to Monitor**:
  - Job status and progress percentage
  - Worker performance and quality metrics
  - Cost tracking and budget utilization
  - Annotation completion rates

### 4. SageMaker Studio Notebook Output

#### Real-time Feedback
- **Cell Outputs**: Progress bars, statistics, and validation results
- **Visualizations**: Data distribution charts and sample images
- **Error Messages**: Data quality issues and recommendations
- **Success Confirmations**: Dataset creation and validation status

## 🔄 Complete Workflow Execution

### Phase 1: Data Discovery and Exploration (Cells 1-3)
**What Happens:**
- MLflow connection to SageMaker managed server
- S3 data discovery and initial exploration
- Image analysis and quality assessment

**Monitor At:**
- Notebook output for data statistics
- MLflow UI for exploration experiment tracking
- S3 console for data access validation

### Phase 2: Advanced Data Analysis (Cells 4-6)
**What Happens:**
- Comprehensive image profiling and quality metrics
- Statistical analysis of dataset characteristics
- Visualization generation and artifact logging

**Monitor At:**
- MLflow UI for analysis artifacts and metrics
- Notebook visualizations for data insights
- Generated charts and statistical reports

### Phase 3: Dataset Preparation for YOLOv11 (Cells 7-9)
**What Happens:**
- YOLOv11 format conversion and validation
- Train/validation split creation
- Metadata file generation (data.yaml, dataset_info.json)

**Monitor At:**
- S3 console for new dataset directory creation
- Notebook output for validation results
- MLflow UI for preparation metrics and artifacts

### Phase 4: Ground Truth Integration (Optional, Cells 10-12)
**What Happens:**
- Ground Truth labeling job creation and configuration
- Annotation workflow setup and monitoring
- Label quality validation and conversion

**Monitor At:**
- SageMaker Console → Ground Truth for job status
- Cost monitoring for labeling expenses
- Notebook output for job progress updates

### Phase 5: Dataset Validation and Handoff (Cells 13-14)
**What Happens:**
- Final dataset structure validation
- Metadata completion and documentation
- ML Engineer handoff preparation

**Monitor At:**
- S3 console for complete dataset structure
- MLflow UI for final validation metrics
- Generated dataset documentation

## 🤝 Integration with ML Engineer Workflow

### Critical Handoff Points

#### 1. Dataset Structure Compliance
**Your Responsibility:**
- ✅ Create datasets in `s3://lucaskle-ab3-project-pv/datasets/`
- ✅ Follow exact YOLOv11 structure requirements
- ✅ Include all required files (data.yaml, dataset_info.json)
- ✅ Validate image/label count matching

**ML Engineer Dependency:**
- ML Engineers will auto-discover your datasets
- Their notebook validates structure before training
- Invalid datasets will be rejected with clear error messages

#### 2. Dataset Naming Convention
**Best Practice:**
- Use descriptive names: `yolov11_drone_detection_2025_01_15`
- Include timestamps for version tracking
- Avoid special characters and spaces
- Keep names under 50 characters

#### 3. Quality Standards
**Minimum Requirements:**
- At least 100 training images (recommended: 1000+)
- At least 50 validation images (recommended: 200+)
- Image/label count must match exactly
- All labels must be in proper YOLOv11 format
- Images should be consistent in quality and resolution

### Communication Protocol

#### Dataset Ready Notification
When your dataset is complete:
1. ✅ Verify all validation checks pass
2. ✅ Document dataset characteristics in MLflow
3. ✅ Notify ML Engineer team with dataset name and location
4. ✅ Provide any special training considerations or recommendations

#### Feedback Loop
- ML Engineers will provide training results and model performance
- Use feedback to improve future dataset preparation
- Collaborate on data quality improvements and labeling strategies

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### 1. MLflow Connection Failed
**Symptoms**: "Could not connect to SageMaker managed MLflow"
**Solutions**:
- Verify IAM role has `sagemaker-mlflow:*` permissions
- Check tracking server ARN is correct
- Fallback to local MLflow will activate automatically

#### 2. S3 Access Denied
**Symptoms**: "Access Denied" when reading from S3
**Solutions**:
- Verify DataScientistRole has read permissions for bucket
- Check AWS profile "ab" is configured correctly
- Ensure bucket `lucaskle-ab3-project-pv` exists and is accessible

#### 3. Dataset Structure Validation Failed
**Symptoms**: "Invalid dataset structure" or missing components
**Solutions**:
- Check all required directories exist: train/images/, train/labels/, val/images/, val/labels/
- Verify data.yaml and dataset_info.json files are present
- Ensure image and label counts match exactly
- Validate file permissions and accessibility

#### 4. Ground Truth Job Failed
**Symptoms**: Labeling job shows "Failed" status
**Solutions**:
- Check IAM permissions for Ground Truth operations
- Verify input manifest format is correct
- Ensure sufficient budget allocation
- Review job configuration for errors

#### 5. Image Processing Errors
**Symptoms**: "Cannot process image" or format errors
**Solutions**:
- Verify image formats are supported (JPG, PNG)
- Check for corrupted or incomplete image files
- Ensure consistent image dimensions and quality
- Validate file naming conventions

## 📊 Success Indicators

### Data Exploration Success
- ✅ MLflow experiment created with comprehensive tracking
- ✅ Data quality metrics logged and visualized
- ✅ Image analysis completed with statistical insights
- ✅ Visualization artifacts generated and stored

### Dataset Preparation Success
- ✅ YOLOv11 dataset structure created and validated
- ✅ All required files present (data.yaml, dataset_info.json)
- ✅ Image/label counts match exactly
- ✅ Train/validation split properly configured
- ✅ Dataset accessible to ML Engineers

### Ground Truth Success (If Used)
- ✅ Labeling job completed successfully
- ✅ Annotations converted to YOLOv11 format
- ✅ Quality validation passed
- ✅ Cost stayed within budget limits

### Integration Success
- ✅ Dataset discoverable by ML Engineer notebook
- ✅ Structure validation passes all checks
- ✅ Metadata complete and accurate
- ✅ Ready for training pipeline execution

## 🎯 Best Practices

### 1. Data Quality Management
- Always validate image quality before dataset creation
- Maintain consistent image resolution and format
- Document any data preprocessing steps in MLflow
- Include representative samples from all target classes

### 2. Dataset Organization
- Use clear, descriptive dataset names with timestamps
- Include comprehensive metadata in dataset_info.json
- Document data sources and collection methods
- Maintain version history for dataset iterations

### 3. Collaboration Efficiency
- Communicate dataset readiness clearly to ML Engineers
- Provide training recommendations based on data analysis
- Document any known data quality issues or limitations
- Maintain feedback loop for continuous improvement

### 4. Resource Management
- Monitor S3 storage usage and costs
- Clean up intermediate processing files
- Use appropriate instance types for data processing
- Optimize Ground Truth labeling costs

## 📞 Support and Resources

### Getting Help
1. **Notebook Issues**: Check cell outputs and error messages
2. **S3 Access Issues**: Verify IAM permissions and bucket access
3. **MLflow Issues**: Check tracking server connectivity
4. **Ground Truth Issues**: Review labeling job configuration and logs

### Additional Resources
- [SageMaker Ground Truth Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html)
- [YOLOv11 Data Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project Repository](../README.md)

### Integration Resources
- [ML Engineer Workflow Guide](README_ml-engineer.md)
- [Dataset Structure Validation](../src/data/data_validator.py)
- [Ground Truth Utilities](../src/data/ground_truth_utils.py)

---

**Last Updated**: 2025-07-22  
**Notebook Version**: Enhanced with YOLOv11 Dataset Preparation and ML Engineer Integration  
**MLflow Integration**: SageMaker Managed Tracking Server  
**Primary Output**: Production-Ready YOLOv11 Datasets for Training Pipeline
