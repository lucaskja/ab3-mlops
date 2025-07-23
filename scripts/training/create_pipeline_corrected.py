#!/usr/bin/env python3
"""
Corrected SageMaker Pipeline Creation Script for YOLOv11
Based on AWS documentation for proper parameter handling
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
    TrainingStep
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REGION = "us-east-1"
BUCKET_NAME = "lucaskle-ab3-project-pv"
ROLE_ARN = "arn:aws:iam::339712742264:role/SageMakerExecutionRole"
MODEL_PACKAGE_GROUP_NAME = "yolov11-drone-detection"

# Get account ID
sts_client = boto3.client('sts', region_name=REGION)
account_id = sts_client.get_caller_identity()["Account"]

# Pipeline session
pipeline_session = PipelineSession()

def create_pipeline_parameters():
    """Create SageMaker Pipeline parameters as a list (AWS documentation pattern)"""
    
    # Create parameters as individual objects (not dictionary)
    dataset_path = ParameterString(
        name="DatasetPath",
        default_value=f"s3://{BUCKET_NAME}/data/training"
    )
    
    dataset_name = ParameterString(
        name="DatasetName", 
        default_value="drone-detection"
    )
    
    model_variant = ParameterString(
        name="ModelVariant",
        default_value="yolov11n"
    )
    
    instance_type = ParameterString(
        name="InstanceType",
        default_value="ml.g4dn.xlarge"
    )
    
    image_size = ParameterInteger(
        name="ImageSize",
        default_value=640
    )
    
    batch_size = ParameterInteger(
        name="BatchSize",
        default_value=16
    )
    
    epochs = ParameterInteger(
        name="Epochs",
        default_value=10
    )
    
    learning_rate = ParameterFloat(
        name="LearningRate",
        default_value=0.001
    )
    
    use_spot = ParameterString(
        name="UseSpot",
        default_value="false"
    )
    
    performance_threshold = ParameterFloat(
        name="PerformanceThreshold",
        default_value=0.5
    )
    
    model_output_path = ParameterString(
        name="ModelOutputPath",
        default_value=f"s3://{BUCKET_NAME}/pipeline-artifacts/models"
    )
    
    evaluation_output_path = ParameterString(
        name="EvaluationOutputPath",
        default_value=f"s3://{BUCKET_NAME}/pipeline-artifacts/evaluation"
    )
    
    # Return as list (required by Pipeline constructor)
    parameters = [
        dataset_path,
        dataset_name,
        model_variant,
        instance_type,
        image_size,
        batch_size,
        epochs,
        learning_rate,
        use_spot,
        performance_threshold,
        model_output_path,
        evaluation_output_path
    ]
    
    # Also return as dictionary for easier access in step creation
    param_dict = {p.name: p for p in parameters}
    
    return parameters, param_dict

def create_data_validation_step(param_dict):
    """Create data validation processing step with correct parameter access"""
    
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
                source=param_dict['DatasetPath'],  # Correct: Use parameter name from AWS docs
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
            "--dataset-name", param_dict['DatasetName'],
            "--model-variant", param_dict['ModelVariant']
        ]
    )
    
    return validation_step

def create_training_step(param_dict, validation_step):
    """Create YOLOv11 training step with correct parameter access"""
    
    estimator = Estimator(
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-training:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type=param_dict['InstanceType'],
        output_path=param_dict['ModelOutputPath'],
        sagemaker_session=pipeline_session,
        hyperparameters={
            "model_variant": param_dict['ModelVariant'],
            "image_size": param_dict['ImageSize'],
            "batch_size": param_dict['BatchSize'],
            "epochs": param_dict['Epochs'],
            "learning_rate": param_dict['LearningRate'],
            "dataset_name": param_dict['DatasetName']
        }
    )
    
    training_step = TrainingStep(
        name="YOLOv11Training",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=param_dict['DatasetPath'],
                content_type="application/x-image"
            )
        },
        depends_on=[validation_step.name]
    )
    
    return training_step

def create_evaluation_step(param_dict, training_step):
    """Create model evaluation processing step - CORRECTED parameter access"""
    
    evaluation_processor = ScriptProcessor(
        command=["python3"],
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-evaluation:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        sagemaker_session=pipeline_session
    )
    
    # Property file for evaluation metrics
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
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
                # CORRECTED: Use correct parameter name from param_dict
                source=param_dict['DatasetPath'],  # Was: parameters['dataset_path']
                destination="/opt/ml/processing/test",
                input_name="test_data"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                # CORRECTED: Use correct parameter name from param_dict
                destination=param_dict['EvaluationOutputPath']  # Was: parameters['evaluation_output_path']
            )
        ],
        property_files=[evaluation_report],
        code="scripts/evaluate_model.py",
        job_arguments=[
            # CORRECTED: Use correct parameter names from param_dict
            "--model-variant", param_dict['ModelVariant'],  # Was: parameters['model_variant']
            "--dataset-name", param_dict['DatasetName']     # Was: parameters['dataset_name']
        ]
    )
    
    return evaluation_step, evaluation_report

def create_model_registration_step(param_dict, training_step, evaluation_step, evaluation_report):
    """Create model registration step for Model Registry"""
    
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
                    property_file=evaluation_report,
                    json_path="metrics.mAP_50"
                )
            },
            {
                "Name": "mAP@0.5:0.95", 
                "Value": JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="metrics.mAP_50_95"
                )
            }
        ]
    )
    
    return register_model_step

def create_performance_condition(evaluation_step, evaluation_report, param_dict):
    """Create condition step for performance threshold checking"""
    
    performance_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.mAP_50"
        ),
        # CORRECTED: Use correct parameter name from param_dict
        right=param_dict['PerformanceThreshold']  # Was: parameters['performance_threshold']
    )
    
    return performance_condition

def create_yolov11_pipeline():
    """Create the complete YOLOv11 SageMaker Pipeline with corrected parameter handling"""
    
    logger.info("🚀 Creating YOLOv11 SageMaker Pipeline with corrected parameter handling...")
    
    try:
        # Create parameters (returns list for Pipeline and dict for step access)
        logger.info("📋 Creating pipeline parameters...")
        parameters_list, param_dict = create_pipeline_parameters()
        
        logger.info(f"✅ Created {len(parameters_list)} parameters")
        logger.info("📋 Parameter names:")
        for param in parameters_list:
            logger.info(f"   - {param.name}: {param.default_value}")
        
        # Create steps using param_dict for easy access
        logger.info("🔧 Creating pipeline steps...")
        
        # 1. Data validation step
        validation_step = create_data_validation_step(param_dict)
        logger.info("   ✅ Data validation step created")
        
        # 2. Training step
        training_step = create_training_step(param_dict, validation_step)
        logger.info("   ✅ Training step created")
        
        # 3. Evaluation step
        evaluation_step, evaluation_report = create_evaluation_step(param_dict, training_step)
        logger.info("   ✅ Evaluation step created")
        
        # 4. Model registration step
        register_model_step = create_model_registration_step(
            param_dict, training_step, evaluation_step, evaluation_report
        )
        logger.info("   ✅ Model registration step created")
        
        # 5. Performance condition
        performance_condition = create_performance_condition(
            evaluation_step, evaluation_report, param_dict
        )
        
        # 6. Conditional step for model registration
        condition_step = ConditionStep(
            name="CheckModelPerformance",
            conditions=[performance_condition],
            if_steps=[register_model_step],
            else_steps=[]
        )
        logger.info("   ✅ Conditional step created")
        
        # Create pipeline with parameters_list (as required by AWS docs)
        pipeline_name = f"yolov11-training-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=parameters_list,  # Use list as per AWS documentation
            steps=[
                validation_step,
                training_step,
                evaluation_step,
                condition_step
            ],
            sagemaker_session=pipeline_session
        )
        
        logger.info("✅ Pipeline object created successfully!")
        
        # Create/update pipeline
        logger.info("📤 Upserting pipeline...")
        pipeline.upsert(role_arn=ROLE_ARN)
        logger.info(f"✅ Pipeline '{pipeline_name}' created/updated successfully!")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Pipeline creation failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Debug information
        if 'param_dict' in locals():
            logger.error("🔍 Available parameter names in param_dict:")
            for name in param_dict.keys():
                logger.error(f"   - {name}")
        
        raise

def main():
    """Main execution function"""
    
    try:
        # Create pipeline
        pipeline = create_yolov11_pipeline()
        
        print("\n🎉 Pipeline Creation Summary:")
        print(f"   Pipeline Name: {pipeline.name}")
        print(f"   Steps: {len(pipeline.steps)}")
        print(f"   Parameters: {len(pipeline.parameters)}")
        
        print("\n📋 Pipeline Steps:")
        for i, step in enumerate(pipeline.steps, 1):
            print(f"   {i}. {step.name} ({type(step).__name__})")
        
        print("\n📋 Pipeline Parameters:")
        for param in pipeline.parameters:
            print(f"   - {param.name}: {param.default_value}")
        
        print("\n📋 Next Steps:")
        print("   1. Check pipeline in SageMaker Studio")
        print("   2. Execute pipeline with: pipeline.start()")
        print("   3. Monitor execution in SageMaker Pipelines console")
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ Pipeline creation failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None

if __name__ == "__main__":
    main()
