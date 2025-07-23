#!/usr/bin/env python3
"""
Fixed SageMaker Pipeline Creation Script for YOLOv11
Addresses parameter access issues and ensures proper step configuration
"""

import boto3
import logging
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
    ParameterBoolean
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
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from datetime import datetime

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
    """Create pipeline parameters with consistent naming"""
    
    parameters = {
        # Data parameters - using consistent snake_case naming
        'dataset_path': ParameterString(
            name="DatasetPath",
            default_value=f"s3://{BUCKET_NAME}/data/training"
        ),
        'dataset_name': ParameterString(
            name="DatasetName", 
            default_value="drone-detection"
        ),
        'model_output_path': ParameterString(
            name="ModelOutputPath",
            default_value=f"s3://{BUCKET_NAME}/models"
        ),
        'evaluation_output_path': ParameterString(
            name="EvaluationOutputPath",
            default_value=f"s3://{BUCKET_NAME}/evaluation"
        ),
        
        # Training parameters
        'model_variant': ParameterString(
            name="ModelVariant",
            default_value="yolov11n"
        ),
        'instance_type': ParameterString(
            name="InstanceType",
            default_value="ml.g4dn.xlarge"
        ),
        'image_size': ParameterInteger(
            name="ImageSize",
            default_value=640
        ),
        'batch_size': ParameterInteger(
            name="BatchSize",
            default_value=16
        ),
        'epochs': ParameterInteger(
            name="Epochs",
            default_value=10
        ),
        'learning_rate': ParameterFloat(
            name="LearningRate",
            default_value=0.001
        ),
        'use_spot': ParameterString(
            name="UseSpot",
            default_value="false"
        ),
        
        # Evaluation parameters
        'performance_threshold': ParameterFloat(
            name="PerformanceThreshold",
            default_value=0.5
        ),
        
        # Deployment parameters
        'endpoint_name': ParameterString(
            name="EndpointName",
            default_value=f"yolov11-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    }
    
    return parameters

def create_data_validation_step(parameters):
    """Create data validation processing step with proper parameter access"""
    
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
                source=parameters['dataset_path'],  # Use parameter object directly
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
            "--dataset-name", parameters['dataset_name'],
            "--model-variant", parameters['model_variant']
        ]
    )
    
    return validation_step

def create_training_step(parameters, validation_step):
    """Create YOLOv11 training step with proper parameter access"""
    
    estimator = Estimator(
        image_uri=f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/yolov11-training:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type=parameters['instance_type'],
        output_path=parameters['model_output_path'],
        sagemaker_session=pipeline_session,
        hyperparameters={
            "model_variant": parameters['model_variant'],
            "image_size": parameters['image_size'],
            "batch_size": parameters['batch_size'],
            "epochs": parameters['epochs'],
            "learning_rate": parameters['learning_rate'],
            "dataset_name": parameters['dataset_name']
        }
    )
    
    training_step = TrainingStep(
        name="YOLOv11Training",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=parameters['dataset_path'],
                content_type="application/x-image"
            )
        },
        depends_on=[validation_step.name]
    )
    
    return training_step

def create_evaluation_step(parameters, training_step):
    """Create model evaluation processing step"""
    
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
                source=parameters['dataset_path'],
                destination="/opt/ml/processing/test",
                input_name="test_data"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=parameters['evaluation_output_path']
            )
        ],
        property_files=[evaluation_report],
        code="scripts/evaluate_model.py",
        job_arguments=[
            "--model-variant", parameters['model_variant'],
            "--dataset-name", parameters['dataset_name']
        ]
    )
    
    return evaluation_step, evaluation_report

def create_model_registration_step(parameters, training_step, evaluation_step, evaluation_report):
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

def create_performance_condition(evaluation_step, evaluation_report, parameters):
    """Create condition step for performance threshold checking"""
    
    performance_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.mAP_50"
        ),
        right=parameters['performance_threshold']
    )
    
    return performance_condition

def create_yolov11_pipeline():
    """Create the complete YOLOv11 SageMaker Pipeline"""
    
    logger.info("🚀 Creating YOLOv11 SageMaker Pipeline...")
    
    try:
        # Create parameters
        logger.info("📋 Creating pipeline parameters...")
        parameters = create_pipeline_parameters()
        
        # Create steps
        logger.info("🔧 Creating pipeline steps...")
        
        # 1. Data validation step
        validation_step = create_data_validation_step(parameters)
        logger.info("   ✅ Data validation step created")
        
        # 2. Training step
        training_step = create_training_step(parameters, validation_step)
        logger.info("   ✅ Training step created")
        
        # 3. Evaluation step
        evaluation_step, evaluation_report = create_evaluation_step(parameters, training_step)
        logger.info("   ✅ Evaluation step created")
        
        # 4. Model registration step
        register_model_step = create_model_registration_step(
            parameters, training_step, evaluation_step, evaluation_report
        )
        logger.info("   ✅ Model registration step created")
        
        # 5. Performance condition
        performance_condition = create_performance_condition(
            evaluation_step, evaluation_report, parameters
        )
        
        # 6. Conditional step for model registration
        condition_step = ConditionStep(
            name="CheckModelPerformance",
            conditions=[performance_condition],
            if_steps=[register_model_step],
            else_steps=[]
        )
        logger.info("   ✅ Conditional step created")
        
        # Create pipeline
        pipeline_name = f"yolov11-training-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=list(parameters.values()),  # Convert to list of parameter objects
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
        
        print("\n📋 Next Steps:")
        print("   1. Check pipeline in SageMaker Studio")
        print("   2. Execute pipeline with custom parameters if needed")
        print("   3. Monitor execution in SageMaker Pipelines console")
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ Pipeline creation failed: {str(e)}")
        return None

if __name__ == "__main__":
    main()
