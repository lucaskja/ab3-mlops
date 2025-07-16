#!/usr/bin/env python3
"""
YOLOv11 Training Execution Script

This script provides a command-line interface for training YOLOv11 models
with comprehensive configuration management and logging.

Usage:
    python scripts/training/train_yolov11.py --config configs/training_config.yaml
    python scripts/training/train_yolov11.py --data /path/to/dataset --epochs 100
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.yolov11_trainer import YOLOv11Trainer, TrainingConfig, load_training_config, save_training_config
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv11 model for drone detection')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    
    # Data parameters
    parser.add_argument('--data', type=str, help='Path to dataset directory')
    parser.add_argument('--classes', nargs='+', default=['vehicle', 'person', 'building'],
                       help='Class names for the dataset')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolov11n',
                       choices=['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
                       help='YOLOv11 model variant')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='auto', help='Training device')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='yolov11_training', help='Experiment name')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # MLFlow parameters
    parser.add_argument('--experiment', type=str, default='yolov11-drone-detection',
                       help='MLFlow experiment name')
    parser.add_argument('--run-name', type=str, help='MLFlow run name')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    parser.add_argument('--eval-split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split for evaluation')
    
    # Save configuration
    parser.add_argument('--save-config', type=str, help='Save configuration to file')
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from command line arguments"""
    
    config = TrainingConfig(
        model_variant=args.model,
        pretrained=args.pretrained,
        num_classes=len(args.classes),
        class_names=args.classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.img_size,
        device=args.device,
        workers=args.workers,
        data_path=args.data or "",
        project_path=args.project,
        name=args.name,
        experiment_name=args.experiment,
        run_name=args.run_name
    )
    
    return config


def validate_dataset(data_path: str) -> bool:
    """Validate dataset structure"""
    if not os.path.exists(data_path):
        logger.error(f"Dataset path does not exist: {data_path}")
        return False
    
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    
    for req_dir in required_dirs:
        full_path = os.path.join(data_path, req_dir)
        if not os.path.exists(full_path):
            logger.error(f"Required directory missing: {full_path}")
            return False
    
    # Check if dataset.yaml exists
    dataset_yaml = os.path.join(data_path, 'dataset.yaml')
    if not os.path.exists(dataset_yaml):
        logger.warning(f"Dataset configuration not found: {dataset_yaml}")
        logger.info("Configuration will be created automatically")
    
    logger.info("Dataset structure validation passed")
    return True


def setup_environment():
    """Setup training environment"""
    # Set AWS profile if configured
    project_config = get_config()
    aws_profile = project_config.get('aws', {}).get('profile')
    
    if aws_profile:
        os.environ['AWS_PROFILE'] = aws_profile
        logger.info(f"AWS profile set to: {aws_profile}")
    
    # Create necessary directories
    os.makedirs('runs/detect', exist_ok=True)
    os.makedirs('logs', exist_ok=True)


def main():
    """Main training execution function"""
    args = parse_arguments()
    
    logger.info("Starting YOLOv11 training script")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup environment
    setup_environment()
    
    try:
        # Load or create configuration
        if args.config and os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            config = load_training_config(args.config)
            
            # Override with command line arguments if provided
            if args.data:
                config.data_path = args.data
            if args.epochs != 100:  # Default value
                config.epochs = args.epochs
            if args.batch_size != 16:  # Default value
                config.batch_size = args.batch_size
                
        else:
            logger.info("Creating configuration from command line arguments")
            config = create_config_from_args(args)
        
        # Save configuration if requested
        if args.save_config:
            save_training_config(config, args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
        
        # Validate dataset
        if config.data_path:
            if not validate_dataset(config.data_path):
                logger.error("Dataset validation failed")
                return 1
        else:
            logger.error("No dataset path provided")
            return 1
        
        # Initialize trainer
        logger.info("Initializing YOLOv11 trainer")
        trainer = YOLOv11Trainer(config)
        
        # Start training
        logger.info("Starting model training")
        results = trainer.train(
            data_path=config.data_path,
            resume_from=args.resume
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        # Run evaluation if requested
        if args.evaluate and results.get('model_path'):
            logger.info(f"Running evaluation on {args.eval_split} set")
            eval_metrics = trainer.evaluate_model(
                model_path=results['model_path'],
                data_path=config.data_path,
                split=args.eval_split
            )
            logger.info(f"Evaluation metrics: {eval_metrics}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)