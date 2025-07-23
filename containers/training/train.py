#!/usr/bin/env python3
"""
YOLOv11 Training Script for SageMaker Pipeline

This script trains a YOLOv11 model using hyperparameters from SageMaker.
"""

import os
import argparse
import logging
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments from SageMaker."""
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables and paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    
    # Hyperparameters from SageMaker
    parser.add_argument("--model_variant", type=str, default="n")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dataset_name", type=str, default="yolov11_dataset")
    
    # Additional training parameters
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=10)
    
    return parser.parse_args()

def setup_dataset_config(training_data_path):
    """Setup dataset configuration for training."""
    training_path = Path(training_data_path)
    
    # Look for existing data.yaml
    data_yaml_path = training_path / "data.yaml"
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Found existing data.yaml: {config}")
        return str(data_yaml_path), config
    
    # Look for dataset structure
    train_images = training_path / "train" / "images"
    train_labels = training_path / "train" / "labels"
    
    if not (train_images.exists() and train_labels.exists()):
        raise FileNotFoundError(f"Training data structure not found in {training_path}")
    
    # Create basic configuration
    config = {
        'path': str(training_path),
        'train': 'train/images',
        'val': 'train/images',  # Use same for validation if no separate val set
        'names': {0: 'object'},  # Default single class
        'nc': 1
    }
    
    # Try to infer classes from a label file
    label_files = list(train_labels.glob("*.txt"))
    if label_files:
        try:
            with open(label_files[0], 'r') as f:
                lines = f.readlines()
            if lines:
                # Get unique class IDs
                class_ids = set()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))
                
                if class_ids:
                    max_class = max(class_ids)
                    config['nc'] = max_class + 1
                    config['names'] = {i: f'class_{i}' for i in range(max_class + 1)}
        except Exception as e:
            logger.warning(f"Could not infer classes from labels: {e}")
    
    # Save configuration
    with open(data_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml: {config}")
    return str(data_yaml_path), config

def train_model(args):
    """Train YOLOv11 model."""
    logger.info("Starting YOLOv11 training")
    logger.info(f"Training arguments: {vars(args)}")
    
    # Setup dataset configuration
    data_config_path, dataset_config = setup_dataset_config(args.training)
    
    # Initialize YOLOv11 model
    model_name = f"yolo11{args.model_variant}.pt"
    model = YOLO(model_name)
    logger.info(f"Initialized YOLOv11 model: {model_name}")
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    logger.info("Starting model training...")
    results = model.train(
        data=data_config_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        lr0=args.learning_rate,
        patience=args.patience,
        save_period=args.save_period,
        project=str(model_dir),
        name='yolov11_training',
        exist_ok=True,
        verbose=True,
        device='cpu'  # SageMaker will handle GPU allocation
    )
    
    # Save the best model
    best_model_path = model_dir / "yolov11_training" / "weights" / "best.pt"
    final_model_path = model_dir / "model.tar.gz"
    
    if best_model_path.exists():
        # Create tar.gz for SageMaker
        import tarfile
        with tarfile.open(final_model_path, "w:gz") as tar:
            tar.add(best_model_path, arcname="model.pt")
            # Add config file
            tar.add(data_config_path, arcname="data.yaml")
        
        logger.info(f"Model saved to {final_model_path}")
    else:
        logger.warning("Best model not found, saving current model")
        model.save(str(model_dir / "model.pt"))
    
    # Save training results
    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics if available
    metrics = {}
    if hasattr(results, 'results_dict'):
        results_dict = results.results_dict
        metrics = {
            "final_mAP_0.5": results_dict.get('metrics/mAP50(B)', 0.0),
            "final_mAP_0.5:0.95": results_dict.get('metrics/mAP50-95(B)', 0.0),
            "final_precision": results_dict.get('metrics/precision(B)', 0.0),
            "final_recall": results_dict.get('metrics/recall(B)', 0.0),
            "final_loss": results_dict.get('train/box_loss', 0.0)
        }
    
    training_summary = {
        "model_variant": args.model_variant,
        "dataset_name": args.dataset_name,
        "epochs_completed": args.epochs,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience
        },
        "final_metrics": metrics,
        "dataset_config": dataset_config,
        "model_path": str(final_model_path)
    }
    
    # Save training summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info(f"Training completed. Summary saved to {summary_path}")
    
    return training_summary

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("YOLOv11 Training Container Started")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Training data: {args.training}")
    logger.info(f"Output directory: {args.output_data_dir}")
    
    # Check training data exists
    training_path = Path(args.training)
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_path}")
    
    # Train the model
    training_summary = train_model(args)
    
    logger.info("YOLOv11 Training Container Completed Successfully")
    logger.info(f"Final mAP@0.5: {training_summary['final_metrics'].get('final_mAP_0.5', 'N/A')}")

if __name__ == "__main__":
    main()
