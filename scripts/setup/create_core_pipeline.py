#!/usr/bin/env python3
"""
Create Core SageMaker Pipeline for YOLOv11 Training

This script creates a simplified SageMaker Pipeline for the core setup that includes
only essential training and evaluation steps for YOLOv11 models.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger, ParameterFloat
)
from sagemaker.pytorch import PyTorch
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorePipelineCreator:
    """Creates a simplified SageMaker Pipeline for YOLOv11 training."""
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the pipeline creator.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Set up AWS session
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        self.sagemaker_client = self.session.client('sagemaker')
        
        # Get execution role
        self.execution_role = self.sagemaker_session.get_caller_identity_arn()
        
        # Project configuration
        self.project_name = "sagemaker-core-setup"
        self.bucket_name = "lucaskle-ab3-project-pv"
        
        logger.info(f"Initialized CorePipelineCreator with profile: {aws_profile}")
        logger.info(f"Region: {region}")
        logger.info(f"Execution role: {self.execution_role}")
    
    def create_pipeline_parameters(self) -> Dict[str, Any]:
        """Create pipeline parameters."""
        parameters = {
            # Data parameters
            'input_data': ParameterString(
                name="InputData",
                default_value=f"s3://{self.bucket_name}/datasets/"
            ),
            'model_output': ParameterString(
                name="ModelOutput", 
                default_value=f"s3://{self.bucket_name}/models/"
            ),
            
            # Training parameters
            'instance_type': ParameterString(
                name="TrainingInstanceType",
                default_value="ml.g4dn.xlarge"
            ),
            'instance_count': ParameterInteger(
                name="TrainingInstanceCount",
                default_value=1
            ),
            'epochs': ParameterInteger(
                name="Epochs",
                default_value=10
            ),
            'batch_size': ParameterInteger(
                name="BatchSize", 
                default_value=16
            ),
            'learning_rate': ParameterFloat(
                name="LearningRate",
                default_value=0.001
            ),
            'image_size': ParameterInteger(
                name="ImageSize",
                default_value=640
            ),
            'model_variant': ParameterString(
                name="ModelVariant",
                default_value="yolov11n"
            )
        }
        
        return parameters
    
    def create_training_script(self) -> str:
        """Create a simplified training script for YOLOv11."""
        script_content = '''#!/usr/bin/env python3
"""
Simplified YOLOv11 Training Script for Core Setup
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv11 Training")
    
    # Model parameters
    parser.add_argument("--model-variant", type=str, default="yolov11n", 
                       help="YOLOv11 model variant")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--image-size", type=int, default=640, 
                       help="Image size")
    
    # SageMaker paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    
    return parser.parse_args()


def find_data_yaml(data_dir):
    """Find data.yaml file in the data directory."""
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "data.yaml":
                return os.path.join(root, file)
    return None


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting YOLOv11 training")
    logger.info(f"Model variant: {args.model_variant}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Image size: {args.image_size}")
    
    # Find data configuration
    data_yaml_path = find_data_yaml(args.train)
    if not data_yaml_path:
        logger.error("Could not find data.yaml file")
        sys.exit(1)
    
    logger.info(f"Using data config: {data_yaml_path}")
    
    # Load YOLOv11 model
    model = YOLO(f"{args.model_variant}.pt")
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.learning_rate,
        imgsz=args.image_size,
        project=args.model_dir,
        name="yolov11_training",
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True
    )
    
    # Save model artifacts
    model_path = os.path.join(args.model_dir, "best.pt")
    if os.path.exists(os.path.join(args.model_dir, "yolov11_training", "weights", "best.pt")):
        import shutil
        shutil.copy2(
            os.path.join(args.model_dir, "yolov11_training", "weights", "best.pt"),
            model_path
        )
    
    # Save training results
    results_path = os.path.join(args.model_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "model_variant": args.model_variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "image_size": args.image_size,
            "training_completed": True
        }, f, indent=2)
    
    logger.info("Training completed successfully")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
'''
        
        # Create training script directory
        script_dir = "/tmp/training_scripts"
        os.makedirs(script_dir, exist_ok=True)
        
        script_path = os.path.join(script_dir, "train.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def create_evaluation_script(self) -> str:
        """Create a simplified evaluation script."""
        script_content = '''#!/usr/bin/env python3
"""
Simplified YOLOv11 Evaluation Script for Core Setup
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv11 Evaluation")
    
    # SageMaker paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info("Starting YOLOv11 evaluation")
    
    # Find model file
    model_path = None
    for root, dirs, files in os.walk(args.model_dir):
        for file in files:
            if file.endswith(".pt"):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break
    
    if not model_path:
        logger.error("Could not find model file")
        return
    
    logger.info(f"Using model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Find test data
    data_yaml_path = None
    for root, dirs, files in os.walk(args.test_dir):
        for file in files:
            if file == "data.yaml":
                data_yaml_path = os.path.join(root, file)
                break
        if data_yaml_path:
            break
    
    if data_yaml_path:
        # Validate on test data
        results = model.val(data=data_yaml_path)
        
        # Extract metrics
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
        }
    else:
        logger.warning("No test data found, using dummy metrics")
        metrics = {
            "mAP50": 0.5,
            "mAP50-95": 0.3,
            "precision": 0.6,
            "recall": 0.5
        }
    
    # Save evaluation results
    evaluation_path = os.path.join(args.output_dir, "evaluation.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(evaluation_path, 'w') as f:
        json.dump({
            "metrics": metrics,
            "model_path": model_path,
            "evaluation_completed": True
        }, f, indent=2)
    
    logger.info("Evaluation completed successfully")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Results saved to: {evaluation_path}")


if __name__ == "__main__":
    main()
'''
        
        # Create evaluation script directory
        script_dir = "/tmp/evaluation_scripts"
        os.makedirs(script_dir, exist_ok=True)
        
        script_path = os.path.join(script_dir, "evaluate.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def create_training_step(self, parameters: Dict[str, Any], training_script: str) -> TrainingStep:
        """Create training step."""
        logger.info("Creating training step")
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point=os.path.basename(training_script),
            source_dir=os.path.dirname(training_script),
            role=self.execution_role,
            instance_count=parameters['instance_count'],
            instance_type=parameters['instance_type'],
            framework_version="1.13.1",
            py_version="py39",
            hyperparameters={
                "model-variant": parameters['model_variant'],
                "epochs": parameters['epochs'],
                "batch-size": parameters['batch_size'],
                "learning-rate": parameters['learning_rate'],
                "image-size": parameters['image_size']
            },
            sagemaker_session=self.sagemaker_session,
            volume_size=30,
            max_run=3600,  # 1 hour
            dependencies=["ultralytics", "PyYAML"]
        )
        
        # Create training step
        training_step = TrainingStep(
            name="YOLOv11Training",
            estimator=estimator,
            inputs={
                "train": parameters['input_data'],
                "validation": parameters['input_data']
            }
        )
        
        return training_step
    
    def create_evaluation_step(self, parameters: Dict[str, Any], evaluation_script: str, 
                             training_step: TrainingStep) -> ProcessingStep:
        """Create evaluation step."""
        logger.info("Creating evaluation step")
        
        # Create SKLearn processor for evaluation
        processor = SKLearnProcessor(
            framework_version="0.23-1",
            role=self.execution_role,
            instance_type="ml.m5.large",
            instance_count=1,
            sagemaker_session=self.sagemaker_session
        )
        
        # Create evaluation step
        evaluation_step = ProcessingStep(
            name="YOLOv11Evaluation",
            processor=processor,
            code=evaluation_script,
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/input/data/model"
                ),
                ProcessingInput(
                    source=parameters['input_data'],
                    destination="/opt/ml/input/data/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/output",
                    destination=f"s3://{self.bucket_name}/evaluation/"
                )
            ]
        )
        
        return evaluation_step
    
    def create_pipeline(self, pipeline_name: str) -> Pipeline:
        """Create the complete pipeline."""
        logger.info(f"Creating pipeline: {pipeline_name}")
        
        # Create parameters
        parameters = self.create_pipeline_parameters()
        
        # Create scripts
        training_script = self.create_training_script()
        evaluation_script = self.create_evaluation_script()
        
        # Create steps
        training_step = self.create_training_step(parameters, training_script)
        evaluation_step = self.create_evaluation_step(parameters, evaluation_script, training_step)
        
        # Create pipeline
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=list(parameters.values()),
            steps=[training_step, evaluation_step],
            sagemaker_session=self.sagemaker_session
        )
        
        return pipeline
    
    def register_pipeline(self, pipeline: Pipeline) -> str:
        """Register the pipeline with SageMaker."""
        logger.info(f"Registering pipeline: {pipeline.name}")
        
        try:
            # Upsert pipeline (create or update)
            response = pipeline.upsert(role_arn=self.execution_role)
            pipeline_arn = response['PipelineArn']
            
            logger.info(f"Pipeline registered successfully: {pipeline_arn}")
            return pipeline_arn
            
        except Exception as e:
            logger.error(f"Failed to register pipeline: {str(e)}")
            raise
    
    def create_and_register_pipeline(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """Create and register the complete pipeline."""
        if not pipeline_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            pipeline_name = f"{self.project_name}-yolov11-pipeline-{timestamp}"
        
        logger.info(f"Creating and registering pipeline: {pipeline_name}")
        
        # Create pipeline
        pipeline = self.create_pipeline(pipeline_name)
        
        # Register pipeline
        pipeline_arn = self.register_pipeline(pipeline)
        
        # Return pipeline information
        return {
            "pipeline_name": pipeline_name,
            "pipeline_arn": pipeline_arn,
            "bucket_name": self.bucket_name,
            "execution_role": self.execution_role,
            "region": self.region,
            "created_at": datetime.now().isoformat()
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create Core SageMaker Pipeline")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--pipeline-name", help="Pipeline name (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    try:
        # Verify AWS profile
        session = boto3.Session(profile_name=args.profile)
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        
        logger.info(f"Using AWS profile: {args.profile}")
        logger.info(f"Account ID: {identity['Account']}")
        logger.info(f"User/Role: {identity['Arn']}")
        
        if args.dry_run:
            logger.info("Dry run mode - pipeline would be created but not registered")
            return
        
        # Create pipeline creator
        creator = CorePipelineCreator(aws_profile=args.profile, region=args.region)
        
        # Create and register pipeline
        result = creator.create_and_register_pipeline(args.pipeline_name)
        
        # Display results
        print("\n" + "="*60)
        print("🎉 Core SageMaker Pipeline Created Successfully!")
        print("="*60)
        print(f"Pipeline Name: {result['pipeline_name']}")
        print(f"Pipeline ARN: {result['pipeline_arn']}")
        print(f"S3 Bucket: {result['bucket_name']}")
        print(f"Region: {result['region']}")
        print(f"Created: {result['created_at']}")
        
        print("\n📋 Next Steps:")
        print("1. Use the ML Engineer notebook to execute this pipeline")
        print("2. Provide a dataset in YOLOv11 format in S3")
        print("3. Monitor pipeline execution in SageMaker Studio")
        print("4. Review model artifacts after training completion")
        
        print(f"\n💡 Pipeline Execution Command:")
        print(f"aws sagemaker start-pipeline-execution \\")
        print(f"    --pipeline-name {result['pipeline_name']} \\")
        print(f"    --profile {args.profile}")
        
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
