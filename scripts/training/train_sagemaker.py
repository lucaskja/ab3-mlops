#!/usr/bin/env python3
"""
SageMaker Training Entry Point

This script serves as the entry point for SageMaker training jobs.
It handles SageMaker-specific environment setup and integrates with
the YOLOv11 training pipeline.

Usage:
    This script is called by SageMaker with the following environment:
    - /opt/ml/input/data/training/ - Training data
    - /opt/ml/input/data/validation/ - Validation data (optional)
    - /opt/ml/model/ - Model output directory
    - /opt/ml/output/ - Output directory for artifacts
    - /opt/ml/input/config/hyperparameters.json - Hyperparameters
"""

import os
import sys
import json
import logging
import argparse
import shutil
from pathlib import Path

# Add source directory to Python path
sys.path.append('/opt/ml/code/src')
sys.path.append('/opt/ml/code')

from models.yolov11_trainer import YOLOv11Trainer, TrainingConfig
from data.s3_utils import S3DataAccess

# Configure logging for SageMaker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SageMakerEnvironment:
    """
    Handles SageMaker environment setup and data management.
    """
    
    def __init__(self):
        """Initialize SageMaker environment."""
        # SageMaker paths
        self.input_path = Path('/opt/ml/input')
        self.output_path = Path('/opt/ml/output')
        self.model_path = Path('/opt/ml/model')
        self.code_path = Path('/opt/ml/code')
        
        # Data paths
        self.data_path = self.input_path / 'data'
        self.config_path = self.input_path / 'config'
        
        # Create output directories
        self.output_path.mkdir(exist_ok=True)
        self.model_path.mkdir(exist_ok=True)
        
        logger.info("SageMaker environment initialized")
        self._log_environment_info()
    
    def _log_environment_info(self):
        """Log SageMaker environment information."""
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Code path: {self.code_path}")
        
        # Log available data channels
        if self.data_path.exists():
            data_channels = [d.name for d in self.data_path.iterdir() if d.is_dir()]
            logger.info(f"Available data channels: {data_channels}")
        
        # Log environment variables
        sagemaker_env_vars = {k: v for k, v in os.environ.items() if k.startswith('SM_')}
        logger.info(f"SageMaker environment variables: {sagemaker_env_vars}")
    
    def load_hyperparameters(self) -> dict:
        """
        Load hyperparameters from SageMaker configuration.
        
        Returns:
            Dictionary of hyperparameters
        """
        hyperparams_file = self.config_path / 'hyperparameters.json'
        
        if hyperparams_file.exists():
            with open(hyperparams_file, 'r') as f:
                hyperparams = json.load(f)
            logger.info(f"Loaded hyperparameters: {hyperparams}")
            return hyperparams
        else:
            logger.warning("No hyperparameters file found, using defaults")
            return {}
    
    def prepare_training_data(self) -> str:
        """
        Prepare training data from SageMaker input channels.
        
        Returns:
            Path to prepared training data
        """
        logger.info("Preparing training data...")
        
        # Check for training data channel
        training_channel = self.data_path / 'training'
        validation_channel = self.data_path / 'validation'
        
        if training_channel.exists():
            logger.info(f"Found training data channel: {training_channel}")
            return str(training_channel)
        
        # If no specific channels, look for data in any available channel
        available_channels = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        if available_channels:
            data_channel = available_channels[0]
            logger.info(f"Using data channel: {data_channel}")
            return str(data_channel)
        
        raise ValueError("No training data found in SageMaker input channels")
    
    def save_model_artifacts(self, model_path: str, training_results: dict):
        """
        Save model artifacts to SageMaker model directory.
        
        Args:
            model_path: Path to trained model
            training_results: Training results dictionary
        """
        logger.info("Saving model artifacts...")
        
        try:
            # Copy model file
            if os.path.exists(model_path):
                model_filename = os.path.basename(model_path)
                target_model_path = self.model_path / model_filename
                shutil.copy2(model_path, target_model_path)
                logger.info(f"Model saved to: {target_model_path}")
            
            # Save training results
            results_file = self.model_path / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            logger.info(f"Training results saved to: {results_file}")
            
            # Save model configuration
            config_file = self.model_path / 'model_config.json'
            model_config = {
                'framework': 'pytorch',
                'model_type': 'yolov11',
                'input_shape': [3, 640, 640],
                'num_classes': training_results.get('training_config', {}).get('num_classes', 39),
                'class_names': training_results.get('training_config', {}).get('class_names', [])
            }
            
            with open(config_file, 'w') as f:
                json.dump(model_config, f, indent=2)
            logger.info(f"Model configuration saved to: {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {str(e)}")
            raise


def parse_sagemaker_arguments():
    """Parse command line arguments for SageMaker training."""
    parser = argparse.ArgumentParser(description='SageMaker YOLOv11 Training')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolov11n',
                       help='YOLOv11 model variant')
    parser.add_argument('--pretrained', type=bool, default=True,
                       help='Use pretrained weights')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    
    # Hardware parameters
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # S3 parameters
    parser.add_argument('--s3-bucket', type=str,
                       help='S3 bucket for additional data access')
    parser.add_argument('--aws-profile', type=str, default='default',
                       help='AWS profile (usually default in SageMaker)')
    
    return parser.parse_args()


def create_training_config_from_sagemaker(args, hyperparams: dict) -> TrainingConfig:
    """
    Create training configuration from SageMaker arguments and hyperparameters.
    
    Args:
        args: Parsed command line arguments
        hyperparams: Hyperparameters from SageMaker
        
    Returns:
        Training configuration
    """
    # Merge command line args with hyperparameters (hyperparams take precedence)
    config_dict = {
        'model_variant': hyperparams.get('model', args.model),
        'pretrained': hyperparams.get('pretrained', args.pretrained),
        'epochs': int(hyperparams.get('epochs', args.epochs)),
        'batch_size': int(hyperparams.get('batch-size', args.batch_size)),
        'learning_rate': float(hyperparams.get('learning-rate', args.learning_rate)),
        'image_size': int(hyperparams.get('img-size', args.img_size)),
        'workers': int(hyperparams.get('workers', args.workers)),
        'device': 'auto',
        'project_path': '/opt/ml/output',
        'name': 'sagemaker_training',
        's3_bucket': hyperparams.get('s3-bucket', args.s3_bucket),
        'aws_profile': args.aws_profile,
        'experiment_name': 'sagemaker-yolov11-training'
    }
    
    # Determine number of classes and class names from hyperparameters or defaults
    num_classes = int(hyperparams.get('num-classes', 39))  # Default for plant disease dataset
    class_names = hyperparams.get('class-names')
    
    if isinstance(class_names, str):
        # If class names are provided as a string, parse them
        class_names = [name.strip() for name in class_names.split(',')]
    elif not class_names:
        # Generate default class names
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    config_dict.update({
        'num_classes': num_classes,
        'class_names': class_names
    })
    
    return TrainingConfig(**config_dict)


def setup_distributed_training():
    """Setup distributed training if running on multiple instances."""
    world_size = int(os.environ.get('SM_NUM_GPUS', 1))
    rank = int(os.environ.get('SM_CURRENT_HOST_RANK', 0))
    
    if world_size > 1:
        logger.info(f"Setting up distributed training: rank={rank}, world_size={world_size}")
        
        # Set environment variables for PyTorch distributed training
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        
        # Set master address and port
        master_addr = os.environ.get('SM_MASTER_ADDR', 'localhost')
        master_port = os.environ.get('SM_MASTER_PORT', '12345')
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        logger.info(f"Distributed training setup: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    
    return world_size > 1


def log_training_metrics(metrics: dict):
    """
    Log training metrics in SageMaker format for CloudWatch.
    
    Args:
        metrics: Dictionary of metrics to log
    """
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            # Format for SageMaker metrics parsing
            print(f"{metric_name}: {metric_value}")


def main():
    """Main training function for SageMaker."""
    logger.info("Starting SageMaker YOLOv11 training...")
    
    try:
        # Parse arguments
        args = parse_sagemaker_arguments()
        
        # Initialize SageMaker environment
        sm_env = SageMakerEnvironment()
        
        # Load hyperparameters
        hyperparams = sm_env.load_hyperparameters()
        
        # Create training configuration
        config = create_training_config_from_sagemaker(args, hyperparams)
        
        logger.info(f"Training configuration: {config}")
        
        # Setup distributed training if needed
        is_distributed = setup_distributed_training()
        
        # Prepare training data
        data_path = sm_env.prepare_training_data()
        
        # Initialize trainer
        trainer = YOLOv11Trainer(config)
        
        # Start training
        logger.info("Starting model training...")
        
        if config.s3_bucket:
            # Train with S3 data
            results = trainer.train_with_s3_data(
                s3_prefix=config.s3_data_prefix,
                local_data_dir=data_path,
                cleanup_after=False  # Don't cleanup in SageMaker
            )
        else:
            # Train with local data
            results = trainer.train(data_path=data_path)
        
        logger.info("Training completed successfully!")
        
        # Log final metrics for SageMaker
        final_metrics = results.get('final_metrics', {})
        log_training_metrics(final_metrics)
        
        # Save model artifacts
        model_path = results.get('model_path', '')
        sm_env.save_model_artifacts(model_path, results)
        
        logger.info("SageMaker training job completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"SageMaker training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)