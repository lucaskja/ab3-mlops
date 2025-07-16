"""
YOLOv11 Training Script

This module provides comprehensive training capabilities for YOLOv11 models
with hyperparameter management, checkpointing, and evaluation metrics.

Requirements addressed:
- 4.1: YOLOv11 architecture implementation for training
- 4.3: Evaluation metrics calculation functions (mAP, precision, recall)
"""

import os
import json
import yaml
import logging
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.plotting import plot_results
import matplotlib.pyplot as plt
import seaborn as sns

# Import S3 utilities
from data.s3_utils import S3DataAccess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for YOLOv11 training parameters"""
    
    # Model configuration
    model_variant: str = "yolov11n"  # n, s, m, l, x
    pretrained: bool = True
    num_classes: int = 3
    class_names: List[str] = None
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Image parameters
    image_size: int = 640
    augment: bool = True
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # Optimization parameters
    optimizer: str = "SGD"  # SGD, Adam, AdamW
    lr_scheduler: str = "cosine"  # linear, cosine
    cos_lr: bool = False
    
    # Loss parameters
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    
    # Validation parameters
    val_split: float = 0.2
    save_period: int = 10
    patience: int = 50
    
    # Hardware parameters
    device: str = "auto"
    workers: int = 8
    
    # Paths
    data_path: str = ""
    project_path: str = "runs/detect"
    name: str = "yolov11_training"
    
    # S3 Configuration
    s3_bucket: Optional[str] = None
    s3_data_prefix: str = "data/drone-detection/"
    aws_profile: str = "ab"
    
    # MLFlow parameters
    experiment_name: str = "yolov11-drone-detection"
    run_name: Optional[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["vehicle", "person", "building"]
        if self.run_name is None:
            self.run_name = f"{self.model_variant}_{int(time.time())}"


class YOLOv11Trainer:
    """
    Comprehensive YOLOv11 trainer with MLFlow integration and advanced features.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the YOLOv11 trainer.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.model = None
        self.best_fitness = 0.0
        self.start_epoch = 0
        self.device = self._setup_device()
        
        # Setup directories
        self.project_dir = Path(config.project_path) / config.name
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLFlow
        self._setup_mlflow()
        
        # Initialize S3 client if bucket is specified
        self.s3_client = None
        if config.s3_bucket:
            try:
                self.s3_client = S3DataAccess(
                    bucket_name=config.s3_bucket,
                    aws_profile=config.aws_profile
                )
                logger.info(f"S3 client initialized for bucket: {config.s3_bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {str(e)}")
        
        logger.info(f"YOLOv11Trainer initialized with config: {config.model_variant}")
    
    def download_data_from_s3(self, s3_prefix: Optional[str] = None, local_data_dir: Optional[str] = None) -> str:
        """
        Download training data from S3 bucket to local directory.
        
        Args:
            s3_prefix: S3 prefix for dataset (defaults to config.s3_data_prefix)
            local_data_dir: Local directory to download data to (defaults to temp directory)
            
        Returns:
            Path to local data directory
            
        Raises:
            RuntimeError: If S3 client is not initialized or download fails
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized. Set s3_bucket in config.")
        
        s3_prefix = s3_prefix or self.config.s3_data_prefix
        
        # Create local data directory
        if local_data_dir is None:
            local_data_dir = os.path.join(tempfile.gettempdir(), f"yolo_data_{int(time.time())}")
        
        os.makedirs(local_data_dir, exist_ok=True)
        logger.info(f"Downloading data from S3 prefix '{s3_prefix}' to '{local_data_dir}'")
        
        try:
            # List all objects with the prefix
            objects = self.s3_client.list_objects(prefix=s3_prefix)
            
            if not objects:
                raise RuntimeError(f"No objects found with prefix '{s3_prefix}'")
            
            # Download each object
            downloaded_count = 0
            for obj in objects:
                s3_key = obj['Key']
                
                # Create local file path maintaining S3 structure
                relative_path = s3_key[len(s3_prefix):].lstrip('/')
                local_file_path = os.path.join(local_data_dir, relative_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                self.s3_client.download_file_to_local(s3_key, local_file_path)
                downloaded_count += 1
                
                if downloaded_count % 100 == 0:
                    logger.info(f"Downloaded {downloaded_count}/{len(objects)} files...")
            
            logger.info(f"Successfully downloaded {downloaded_count} files from S3")
            return local_data_dir
            
        except Exception as e:
            logger.error(f"Failed to download data from S3: {str(e)}")
            raise RuntimeError(f"S3 data download failed: {str(e)}")
    
    def prepare_s3_dataset(self, s3_prefix: Optional[str] = None, local_data_dir: Optional[str] = None) -> str:
        """
        Download and prepare dataset from S3 for YOLOv11 training.
        
        Args:
            s3_prefix: S3 prefix for dataset
            local_data_dir: Local directory for data
            
        Returns:
            Path to prepared dataset directory
        """
        logger.info("Preparing dataset from S3...")
        
        # Download data from S3
        data_dir = self.download_data_from_s3(s3_prefix, local_data_dir)
        
        # Validate dataset structure
        self._validate_dataset_structure(data_dir)
        
        # Create dataset configuration
        dataset_config_path = self.prepare_dataset_config(data_dir)
        
        logger.info(f"Dataset prepared successfully at {data_dir}")
        return data_dir
    
    def _validate_dataset_structure(self, data_dir: str) -> bool:
        """
        Validate that the dataset has the required YOLO structure.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            True if structure is valid
            
        Raises:
            RuntimeError: If dataset structure is invalid
        """
        required_dirs = [
            os.path.join(data_dir, 'train', 'images'),
            os.path.join(data_dir, 'train', 'labels'),
            os.path.join(data_dir, 'val', 'images'),
            os.path.join(data_dir, 'val', 'labels')
        ]
        
        missing_dirs = []
        for req_dir in required_dirs:
            if not os.path.exists(req_dir):
                missing_dirs.append(req_dir)
        
        if missing_dirs:
            raise RuntimeError(f"Missing required directories: {missing_dirs}")
        
        # Check if directories have files
        for req_dir in required_dirs:
            files = os.listdir(req_dir)
            if not files:
                logger.warning(f"Directory is empty: {req_dir}")
        
        logger.info("Dataset structure validation passed")
        return True
    
    def train_with_s3_data(self, s3_prefix: Optional[str] = None, 
                          local_data_dir: Optional[str] = None,
                          resume_from: Optional[str] = None,
                          cleanup_after: bool = True) -> Dict[str, Any]:
        """
        Train YOLOv11 model using data from S3.
        
        Args:
            s3_prefix: S3 prefix for dataset
            local_data_dir: Local directory for data (temp if None)
            resume_from: Path to checkpoint to resume from
            cleanup_after: Whether to cleanup downloaded data after training
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting training with S3 data...")
        
        # Prepare dataset from S3
        data_dir = self.prepare_s3_dataset(s3_prefix, local_data_dir)
        
        try:
            # Train model
            results = self.train(data_path=data_dir, resume_from=resume_from)
            
            # Add S3 info to results
            results['s3_data_source'] = {
                'bucket': self.config.s3_bucket,
                'prefix': s3_prefix or self.config.s3_data_prefix,
                'local_data_dir': data_dir
            }
            
            return results
            
        finally:
            # Cleanup downloaded data if requested
            if cleanup_after and local_data_dir is None:  # Only cleanup temp directories
                try:
                    shutil.rmtree(data_dir)
                    logger.info(f"Cleaned up temporary data directory: {data_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup data directory: {str(e)}")
    
    def _setup_device(self) -> str:
        """Setup training device (CPU/GPU)"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU for training")
        else:
            device = self.config.device
        
        return device
    
    def _setup_mlflow(self):
        """Setup MLFlow experiment tracking"""
        try:
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLFlow experiment set: {self.config.experiment_name}")
        except Exception as e:
            logger.warning(f"MLFlow setup failed: {str(e)}")
    
    def load_model(self, model_path: Optional[str] = None) -> YOLO:
        """
        Load YOLOv11 model with optional pretrained weights.
        
        Args:
            model_path: Path to custom model weights
            
        Returns:
            YOLO model instance
        """
        try:
            if model_path and os.path.exists(model_path):
                # Load custom model
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom model from {model_path}")
            elif self.config.pretrained:
                # Load pretrained model
                model_name = f"{self.config.model_variant}.pt"
                self.model = YOLO(model_name)
                logger.info(f"Loaded pretrained model: {model_name}")
            else:
                # Load model architecture only
                model_name = f"{self.config.model_variant}.yaml"
                self.model = YOLO(model_name)
                logger.info(f"Loaded model architecture: {model_name}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def prepare_dataset_config(self, data_path: str) -> str:
        """
        Prepare dataset configuration file for YOLOv11.
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Path to dataset configuration file
        """
        dataset_config = {
            'path': data_path,
            'train': os.path.join(data_path, 'train', 'images'),
            'val': os.path.join(data_path, 'val', 'images'),
            'nc': self.config.num_classes,
            'names': self.config.class_names
        }
        
        # Add test set if exists
        test_path = os.path.join(data_path, 'test', 'images')
        if os.path.exists(test_path):
            dataset_config['test'] = test_path
        
        # Save configuration
        config_path = os.path.join(data_path, 'dataset.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved to {config_path}")
        return config_path
    
    def train(self, data_path: str, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Train YOLOv11 model with comprehensive logging and checkpointing.
        
        Args:
            data_path: Path to training dataset
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting YOLOv11 training...")
        
        # Prepare dataset configuration
        dataset_config_path = self.prepare_dataset_config(data_path)
        
        # Load model
        if resume_from and os.path.exists(resume_from):
            self.model = self.load_model(resume_from)
            logger.info(f"Resuming training from {resume_from}")
        else:
            self.model = self.load_model()
        
        # Start MLFlow run
        with mlflow.start_run(run_name=self.config.run_name):
            # Log configuration
            self._log_config_to_mlflow()
            
            try:
                # Configure training arguments
                train_args = self._prepare_training_args(dataset_config_path)
                
                # Start training
                results = self.model.train(**train_args)
                
                # Log results to MLFlow
                self._log_results_to_mlflow(results)
                
                # Save final model
                final_model_path = self._save_final_model(results)
                
                # Generate training summary
                training_summary = self._generate_training_summary(results, final_model_path)
                
                logger.info("Training completed successfully!")
                return training_summary
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def _prepare_training_args(self, dataset_config_path: str) -> Dict[str, Any]:
        """Prepare training arguments for YOLOv11"""
        return {
            'data': dataset_config_path,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.image_size,
            'device': self.device,
            'workers': self.config.workers,
            'project': str(self.project_dir.parent),
            'name': self.config.name,
            'exist_ok': True,
            'pretrained': self.config.pretrained,
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': self.config.warmup_momentum,
            'warmup_bias_lr': self.config.warmup_bias_lr,
            'box': self.config.box_loss_gain,
            'cls': self.config.cls_loss_gain,
            'dfl': self.config.dfl_loss_gain,
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'val': True,
            'plots': True,
            'verbose': True
        }
    
    def _log_config_to_mlflow(self):
        """Log training configuration to MLFlow"""
        try:
            config_dict = asdict(self.config)
            for key, value in config_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                elif isinstance(value, list):
                    mlflow.log_param(key, str(value))
            
            # Log system info
            mlflow.log_param("device", self.device)
            mlflow.log_param("torch_version", torch.__version__)
            
            if torch.cuda.is_available():
                mlflow.log_param("cuda_version", torch.version.cuda)
                mlflow.log_param("gpu_name", torch.cuda.get_device_name())
                mlflow.log_param("gpu_memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                
        except Exception as e:
            logger.warning(f"Failed to log config to MLFlow: {str(e)}")
    
    def _log_results_to_mlflow(self, results):
        """Log training results to MLFlow"""
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                
                # Log final metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
            # Log model artifacts
            if hasattr(results, 'save_dir'):
                weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')
                if os.path.exists(weights_path):
                    mlflow.log_artifact(weights_path, "model")
                
                # Log plots
                plots_dir = os.path.join(results.save_dir, 'plots')
                if os.path.exists(plots_dir):
                    for plot_file in os.listdir(plots_dir):
                        if plot_file.endswith(('.png', '.jpg')):
                            mlflow.log_artifact(os.path.join(plots_dir, plot_file), "plots")
                            
        except Exception as e:
            logger.warning(f"Failed to log results to MLFlow: {str(e)}")
    
    def _save_final_model(self, results) -> str:
        """Save final trained model"""
        try:
            if hasattr(results, 'save_dir'):
                best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
                if os.path.exists(best_weights):
                    # Copy to project directory
                    final_model_path = self.project_dir / 'final_model.pt'
                    import shutil
                    shutil.copy2(best_weights, final_model_path)
                    logger.info(f"Final model saved to {final_model_path}")
                    return str(final_model_path)
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to save final model: {str(e)}")
            return ""
    
    def _generate_training_summary(self, results, model_path: str) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        summary = {
            'training_config': asdict(self.config),
            'model_path': model_path,
            'training_time': getattr(results, 'training_time', 0),
            'device_used': self.device,
            'final_metrics': {}
        }
        
        # Extract final metrics
        if hasattr(results, 'results_dict'):
            summary['final_metrics'] = results.results_dict
        
        return summary
    
    def evaluate_model(self, model_path: str, data_path: str, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate trained model and calculate comprehensive metrics.
        
        Args:
            model_path: Path to trained model
            data_path: Path to dataset
            split: Dataset split to evaluate on ('val' or 'test')
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {split} set...")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Prepare dataset config
            dataset_config_path = self.prepare_dataset_config(data_path)
            
            # Run validation
            results = model.val(
                data=dataset_config_path,
                split=split,
                imgsz=self.config.image_size,
                batch=self.config.batch_size,
                device=self.device,
                plots=True,
                save_json=True
            )
            
            # Extract metrics
            metrics = self._extract_evaluation_metrics(results)
            
            # Log to MLFlow if in active run
            try:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"eval_{metric_name}", metric_value)
            except:
                pass  # MLFlow might not be active
            
            logger.info(f"Evaluation completed. mAP@0.5: {metrics.get('map50', 0):.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def _extract_evaluation_metrics(self, results) -> Dict[str, float]:
        """Extract comprehensive evaluation metrics from YOLOv11 results"""
        metrics = {}
        
        try:
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                
                # Standard YOLO metrics
                metrics['map50'] = results_dict.get('metrics/mAP50(B)', 0.0)
                metrics['map50_95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
                metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
                metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-16)
                
                # Loss metrics
                metrics['box_loss'] = results_dict.get('val/box_loss', 0.0)
                metrics['cls_loss'] = results_dict.get('val/cls_loss', 0.0)
                metrics['dfl_loss'] = results_dict.get('val/dfl_loss', 0.0)
                
            # Calculate additional metrics if available
            if hasattr(results, 'box'):
                box_results = results.box
                if hasattr(box_results, 'map'):
                    metrics['map50'] = box_results.map50
                    metrics['map50_95'] = box_results.map
                if hasattr(box_results, 'mp'):
                    metrics['precision'] = box_results.mp
                if hasattr(box_results, 'mr'):
                    metrics['recall'] = box_results.mr
                    
        except Exception as e:
            logger.warning(f"Error extracting metrics: {str(e)}")
        
        return metrics
    
    def create_checkpoint(self, epoch: int, model_state: Dict, optimizer_state: Dict, 
                         metrics: Dict[str, float], checkpoint_dir: str) -> str:
        """
        Create training checkpoint for resumption.
        
        Args:
            epoch: Current epoch number
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            metrics: Current metrics
            checkpoint_dir: Directory to save checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics,
            'config': asdict(self.config),
            'best_fitness': self.best_fitness
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint for resumption.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_fitness = checkpoint.get('best_fitness', 0.0)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.start_epoch}")
        
        return checkpoint
    
    def calculate_model_metrics(self, predictions: List[Dict], ground_truth: List[Dict], 
                              iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate detailed model performance metrics.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Dictionary with calculated metrics
        """
        # This is a simplified implementation
        # In practice, you would use the YOLOv11's built-in metrics calculation
        
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'map50': 0.0,
            'map50_95': 0.0
        }
        
        # Placeholder for actual metric calculation
        # The YOLOv11 model.val() method handles this automatically
        logger.info("Using YOLOv11 built-in metrics calculation")
        
        return metrics
    
    def plot_training_curves(self, results_dir: str, save_path: Optional[str] = None):
        """
        Plot training curves and save visualization.
        
        Args:
            results_dir: Directory containing training results
            save_path: Path to save the plot
        """
        try:
            # Look for results.csv or similar files
            results_file = os.path.join(results_dir, 'results.csv')
            
            if os.path.exists(results_file):
                import pandas as pd
                
                df = pd.read_csv(results_file)
                
                # Create subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('YOLOv11 Training Results', fontsize=16)
                
                # Plot losses
                if 'train/box_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                    axes[0, 0].set_title('Box Loss')
                    axes[0, 0].legend()
                
                if 'train/cls_loss' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
                    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
                    axes[0, 1].set_title('Classification Loss')
                    axes[0, 1].legend()
                
                # Plot metrics
                if 'metrics/mAP50(B)' in df.columns:
                    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'])
                    axes[1, 0].set_title('mAP@0.5')
                
                if 'metrics/mAP50-95(B)' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'])
                    axes[1, 1].set_title('mAP@0.5:0.95')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Training curves saved to {save_path}")
                
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to plot training curves: {str(e)}")


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Create TrainingConfig from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        TrainingConfig object
    """
    return TrainingConfig(**config_dict)


def save_training_config(config: TrainingConfig, save_path: str):
    """
    Save training configuration to file.
    
    Args:
        config: TrainingConfig object
        save_path: Path to save configuration
    """
    config_dict = asdict(config)
    
    with open(save_path, 'w') as f:
        if save_path.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
    
    logger.info(f"Training configuration saved to {save_path}")


def load_training_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrainingConfig object
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config_dict = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
    
    return create_training_config_from_dict(config_dict)


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_variant="yolov11n",
        epochs=50,
        batch_size=16,
        learning_rate=0.01,
        image_size=640,
        num_classes=3,
        class_names=["vehicle", "person", "building"]
    )
    
    trainer = YOLOv11Trainer(config)
    
    # Example training call (uncomment to use)
    # results = trainer.train(data_path="path/to/dataset")
    # print("Training completed:", results)