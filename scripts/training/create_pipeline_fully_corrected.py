#!/usr/bin/env python3
"""
Fully Corrected SageMaker Pipeline Creation Script for YOLOv11
Fixes all parameter access inconsistencies based on AWS documentation
"""

import boto3
import logging
from datetime import datetime
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat
)
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput, Processor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.serverless import ServerlessInferenceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - you'll need to define these
REGION = "us-east-1"
BUCKET_NAME = "lucaskle-ab3-project-pv"
ROLE_ARN = "arn:aws:iam::339712742264:role/SageMakerExecutionRole"
MODEL_PACKAGE_GROUP_NAME = "yolov11-drone-detection"
PIPELINE_NAME = f"yolov11-training-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Get account ID
sts_client = boto3.client('sts', region_name=REGION)
account_id = sts_client.get_caller_identity()["Account"]

# Pipeline session
pipeline_session = PipelineSession()

def extract_config_values():
    """Extract configuration values - you'll need to implement this based on your config system"""
    # This is a placeholder - replace with your actual config extraction logic
    return {
        'dataset_path': f"s3://{BUCKET_NAME}/data/training",
        'dataset_name': "drone-detection",
        'model_variant': "yolov11n",
        'image_size': 640,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 0.001,
        'instance_type': "ml.g4dn.xlarge",
        'use_spot': False,
        'performance_threshold': 0.5,
        'endpoint_name': f"yolov11-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    }

def create_pipeline_parameters():
    """Create SageMaker Pipeline parameters with consistent naming"""
    
    # Extract current configuration
    config = extract_config_values()
    if not config:
        print("❌ No configuration available")
        return None, None
    
    # Define pipeline parameters with current values as defaults
    parameters = [
        # Dataset parameters
        ParameterString(
            name="DatasetPath",
            default_value=config['dataset_path']
        ),
        ParameterString(
            name="DatasetName", 
            default_value=config['dataset_name']
        ),
        
        # Model parameters
        ParameterString(
            name="ModelVariant",
            default_value=config['model_variant']
        ),
        ParameterInteger(
            name="ImageSize",
            default_value=config['image_size']
        ),
        
        # Training parameters
        ParameterInteger(
            name="BatchSize",
            default_value=config['batch_size']
        ),
        ParameterInteger(
            name="Epochs",
            default_value=config['epochs']
        ),
        ParameterFloat(
            name="LearningRate",
            default_value=config['learning_rate']
        ),
        
        # Infrastructure parameters
        ParameterString(
            name="InstanceType",
            default_value=config['instance_type']
        ),
        ParameterString(
            name="UseSpot",
            default_value="true" if config['use_spot'] else "false"
        ),
        
        # Performance threshold
        ParameterFloat(
            name="PerformanceThreshold",
            default_value=config['performance_threshold']
        ),
        
        # Deployment parameters
        ParameterString(
            name="EndpointName",
            default_value=config['endpoint_name']
        ),
        
        # Output paths
        ParameterString(
            name="ModelOutputPath",
            default_value=f"s3://{BUCKET_NAME}/pipeline-artifacts/models"
        ),
        ParameterString(
            name="EvaluationOutputPath",
            default_value=f"s3://{BUCKET_NAME}/pipeline-artifacts/evaluation"
        )
    ]
    
    return parameters, config

def create_data_validation_step(param_dict):
    """Create data validation processing step - CORRECTED parameter access"""
    
    validation_processor = ScriptProcessor(
        command=["python3"],
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-preprocessing:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=pipeline_session
    )
    
    validation_step = ProcessingStep(
        name="DataValidation",
        processor=validation_processor,
        inputs=[
            ProcessingInput(
                source=param_dict['DatasetPath'],  # CORRECT: Use parameter name
                destination="/opt/ml/processing/input",
                input_name="dataset"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="validation_report",
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET_NAME}/pipeline-artifacts/validation"
            )
        ],
        code="scripts/validate_dataset.py",
        job_arguments=[
            "--dataset-name", param_dict['DatasetName'],  # CORRECT: Use parameter name
            "--model-variant", param_dict['ModelVariant']  # CORRECT: Use parameter name
        ]
    )
    
    return validation_step

def create_training_step(param_dict, validation_step):
    """Create YOLOv11 training step - CORRECTED parameter access"""
    
    estimator = Estimator(
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-training:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type=param_dict['InstanceType'],  # CORRECT: Use parameter name
        output_path=param_dict['ModelOutputPath'],  # CORRECT: Use parameter name
        sagemaker_session=pipeline_session,
        use_spot_instances=(param_dict['UseSpot'] == "true"),  # CORRECT: Use parameter name
        max_wait=3600 if param_dict['UseSpot'] == "true" else None,
        max_run=3600,
        hyperparameters={
            "model_variant": param_dict['ModelVariant'],    # CORRECT: Use parameter name
            "image_size": param_dict['ImageSize'],          # CORRECT: Use parameter name
            "batch_size": param_dict['BatchSize'],          # CORRECT: Use parameter name
            "epochs": param_dict['Epochs'],                 # CORRECT: Use parameter name
            "learning_rate": param_dict['LearningRate'],    # CORRECT: Use parameter name
            "dataset_name": param_dict['DatasetName']       # CORRECT: Use parameter name
        }
    )
    
    training_step = TrainingStep(
        name="YOLOv11Training",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=param_dict['DatasetPath'],  # CORRECT: Use parameter name
                content_type="application/x-image"
            )
        },
        depends_on=[validation_step.name]
    )
    
    return training_step

def create_evaluation_step(param_dict, training_step):
    """Create model evaluation processing step - FULLY CORRECTED"""
    
    # Use ScriptProcessor for proper script execution
    evaluation_processor = ScriptProcessor(
        command=["python3"],  # This handles script execution properly
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-evaluation:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",  # GPU for faster evaluation
        sagemaker_session=pipeline_session
    )
    
    # Property file for evaluation metrics
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    # Evaluation step - CORRECTED parameter access
    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model"
            ),
            ProcessingInput(
                # CORRECTED: Use param_dict with correct parameter name
                source=param_dict['DatasetPath'],  # Was: parameters['dataset_path']
                destination="/opt/ml/processing/test",
                input_name="test_data"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                # CORRECTED: Use param_dict with correct parameter name
                destination=param_dict['EvaluationOutputPath']  # Was: parameters['evaluation_output_path']
            )
        ],
        property_files=[evaluation_report],
        code="scripts/evaluate_model.py",  # ScriptProcessor handles this correctly
        job_arguments=[
            # CORRECTED: Use param_dict with correct parameter names
            "--model-variant", param_dict['ModelVariant'],  # Was: parameters['model_variant']
            "--dataset-name", param_dict['DatasetName']     # Was: parameters['dataset_name']
        ]
    )
    
    return evaluation_step, evaluation_report

def create_performance_condition(evaluation_step, evaluation_report, param_dict):
    """Create condition step for performance threshold checking - CORRECTED"""
    
    # Condition: mAP@0.5 >= threshold
    performance_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.mAP_50"
        ),
        # CORRECTED: Use param_dict with correct parameter name
        right=param_dict['PerformanceThreshold']  # Was: parameters['performance_threshold']
    )
    
    return performance_condition

def create_model_registration_step(param_dict, training_step, evaluation_step, evaluation_report):
    """Create model registration step for Model Registry - CORRECTED"""
    
    # Model registration
    register_model_step = RegisterModel(
        name="RegisterYOLOv11Model",
        estimator=training_step.estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json", "image/jpeg", "image/png"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large", "ml.g4dn.xlarge"],
        transform_instances=["ml.m5.large", "ml.g4dn.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        model_metrics=[
            {
                "Name": "mAP@0.5",
                "Value": JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,  # Use the PropertyFile object directly
                    json_path="metrics.mAP_50"
                )
            },
            {
                "Name": "mAP@0.5:0.95",
                "Value": JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,  # Use the PropertyFile object directly
                    json_path="metrics.mAP_50_95"
                )
            }
        ]
    )
    
    return register_model_step

def create_model_creation_step(param_dict, register_model_step):
    """Create model creation step for deployment - CORRECTED"""
    
    # Model creation for deployment using model package
    model = Model(
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-inference:latest",
        # Use model package ARN for registered models
        model_data=register_model_step.properties.ModelPackageArn,
        role=ROLE_ARN,
        sagemaker_session=pipeline_session
    )
    
    create_model_step = CreateModelStep(
        name="CreateYOLOv11Model",
        model=model
    )
    
    return create_model_step

def create_serverless_endpoint_step(param_dict, create_model_step):
    """Create serverless endpoint deployment step - CORRECTED"""
    
    # Serverless inference configuration
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=4096,
        max_concurrency=20,
        provisioned_concurrency=1  # Keep warm for faster response
    )
    
    # Endpoint deployment processor
    deployment_processor = Processor(
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/sagemaker-deployment:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.t3.medium",
        sagemaker_session=pipeline_session
    )
    
    # Deployment step
    deployment_step = ProcessingStep(
        name="DeployServerlessEndpoint",
        processor=deployment_processor,
        inputs=[
            ProcessingInput(
                source=create_model_step.properties.ModelName,
                destination="/opt/ml/processing/model",
                input_name="model_name"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="deployment_status",
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET_NAME}/pipeline-artifacts/deployment"
            )
        ],
        code="scripts/deploy_serverless_endpoint.py",
        job_arguments=[
            # CORRECTED: Use param_dict with correct parameter name
            "--endpoint-name", param_dict['EndpointName'],  # Was: parameters['endpoint_name']
            "--memory-size", "4096",
            "--max-concurrency", "20",
            "--provisioned-concurrency", "1"
        ]
    )
    
    return deployment_step

def create_complete_pipeline(parameters):
    """Create the complete SageMaker Pipeline - FULLY CORRECTED parameter access"""
    
    print("🔧 Creating complete pipeline with corrected parameter handling...")
    
    # Convert parameter list to dictionary for easier access
    param_dict = {p.name: p for p in parameters}
    
    print(f"📋 Available parameter names: {list(param_dict.keys())}")
    
    # Step 1: Data Validation
    validation_step = create_data_validation_step(param_dict)
    print("   ✅ Data validation step created")
    
    # Step 2: Model Training
    training_step = create_training_step(param_dict, validation_step)
    print("   ✅ Training step created")
    
    # Step 3: Model Evaluation - CORRECTED to use param_dict
    evaluation_step, evaluation_report = create_evaluation_step(param_dict, training_step)
    print("   ✅ Evaluation step created")
    
    # Step 4: Performance Condition Check - CORRECTED to use param_dict
    performance_condition = create_performance_condition(evaluation_step, evaluation_report, param_dict)
    print("   ✅ Performance condition created")
    
    # Step 5: Model Registration - CORRECTED to use param_dict
    register_model_step = create_model_registration_step(param_dict, training_step, evaluation_step, evaluation_report)
    print("   ✅ Model registration step created")
    
    # Create conditional step for performance-based registration
    performance_condition_step = ConditionStep(
        name="CheckPerformanceThreshold",
        conditions=[performance_condition],
        if_steps=[register_model_step],
        else_steps=[]
    )
    
    # Assemble pipeline steps
    pipeline_steps = [
        validation_step,
        training_step,
        evaluation_step,
        performance_condition_step
    ]
    
    # Create the pipeline - parameters is already a list
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=parameters,  # CORRECT: Use parameters directly (it's already a list)
        steps=pipeline_steps,
        sagemaker_session=pipeline_session
    )
    
    print("   ✅ Complete pipeline assembly completed")
    
    return pipeline

def main():
    """Main execution function"""
    
    # Create pipeline parameters
    pipeline_parameters, current_config = create_pipeline_parameters()

    if pipeline_parameters:
        print("✅ Pipeline parameters created successfully!")
        
        print("🚀 Creating Complete YOLOv11 SageMaker Pipeline (FULLY CORRECTED)")
        print("=" * 65)
        
        try:
            # Create corrected complete pipeline
            corrected_pipeline = create_complete_pipeline(pipeline_parameters)
            
            print(f"\n✅ Corrected Pipeline '{PIPELINE_NAME}' created successfully!")
            print(f"\n📋 Pipeline Summary:")
            print(f"   Name: {corrected_pipeline.name}")
            print(f"   Steps: {len(corrected_pipeline.steps)}")
            print(f"   Parameters: {len(corrected_pipeline.parameters)}")
            
            # Test pipeline creation with better error handling
            print(f"\n📝 Testing corrected pipeline definition creation...")
            
            try:
                corrected_pipeline.upsert(role_arn=ROLE_ARN)
                print("✅ Corrected pipeline definition created/updated successfully!")
                
                print(f"\n🎯 Corrected complete pipeline is ready for execution!")
                
                # Display pipeline structure
                print(f"\n🔗 Pipeline Flow:")
                print("   1. DataValidation (ProcessingStep)")
                print("   2. YOLOv11Training (TrainingStep)")
                print("   3. ModelEvaluation (ProcessingStep)")
                print("   4. CheckPerformanceThreshold (ConditionStep)")
                print("      ├─ IF mAP@0.5 >= threshold: RegisterYOLOv11Model")
                print("      └─ ELSE: Skip registration")
                
                return corrected_pipeline
                
            except Exception as create_error:
                print(f"❌ Pipeline upsert() failed: {create_error}")
                print(f"Error type: {type(create_error).__name__}")
                return None
                
        except Exception as e:
            print(f"❌ Pipeline object creation failed: {e}")
            print(f"Error type: {type(e).__name__}")
            return None
            
    else:
        print("❌ Cannot create pipeline - parameters not available")
        return None

if __name__ == "__main__":
    pipeline = main()
    if pipeline:
        print("\n🎉 Pipeline creation completed successfully!")
    else:
        print("\n❌ Pipeline creation failed!")
