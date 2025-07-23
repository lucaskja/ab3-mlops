#!/usr/bin/env python3
"""
YOLOv11 Model Evaluation Script for SageMaker Pipeline

This script evaluates a trained YOLOv11 model and generates performance metrics.
"""

import os
import argparse
import logging
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import mlflow
import boto3
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # SageMaker paths
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/input/model")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/input/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    
    # Evaluation parameters
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--performance-threshold", type=float, default=0.7)
    
    # MLflow tracking
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment-name", type=str, default="yolov11-evaluation")
    
    return parser.parse_args()

def setup_mlflow(tracking_uri, experiment_name):
    """Setup MLflow tracking."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except Exception as e:
        logger.warning(f"Could not set MLflow experiment: {e}")

def load_model(model_path):
    """Load the trained YOLOv11 model."""
    model_file = Path(model_path) / "model.pt"
    if not model_file.exists():
        # Try alternative paths
        alternative_paths = [
            Path(model_path) / "best.pt",
            Path(model_path) / "last.pt",
            Path(model_path)  # If model_path is the file itself
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists() and alt_path.is_file():
                model_file = alt_path
                break
        else:
            raise FileNotFoundError(f"Model file not found in {model_path}")
    
    logger.info(f"Loading model from: {model_file}")
    model = YOLO(str(model_file))
    return model

def load_dataset_config(data_path):
    """Load dataset configuration from data.yaml file."""
    config_path = Path(data_path) / "data.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded dataset config: {config}")
        return config
    else:
        logger.warning(f"No data.yaml found at {config_path}")
        return {}

def create_evaluation_config(test_path, dataset_config):
    """Create evaluation configuration for YOLOv11."""
    # Create a temporary data.yaml for evaluation
    eval_config = {
        'path': test_path,
        'val': 'test/images',  # Point to test images
        'names': dataset_config.get('names', {}),
        'nc': dataset_config.get('nc', len(dataset_config.get('names', {})))
    }
    
    config_path = Path(test_path) / "eval_data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    
    logger.info(f"Created evaluation config at {config_path}")
    return str(config_path)

def evaluate_model(args):
    """Evaluate YOLOv11 model."""
    logger.info("Starting YOLOv11 model evaluation")
    
    # Load the trained model
    model = load_model(args.model_path)
    
    # Load dataset configuration
    dataset_config = load_dataset_config(args.test_data)
    
    # Create evaluation configuration
    eval_config_path = create_evaluation_config(args.test_data, dataset_config)
    
    # Setup MLflow
    setup_mlflow(args.mlflow_tracking_uri, args.mlflow_experiment_name)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "img_size": args.img_size,
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "performance_threshold": args.performance_threshold,
            "num_classes": dataset_config.get('nc', 0),
            "class_names": list(dataset_config.get('names', {}).values())
        })
        
        # Run validation/evaluation
        logger.info("Running model validation...")
        results = model.val(
            data=eval_config_path,
            imgsz=args.img_size,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            verbose=True
        )
        
        # Extract metrics
        metrics = {}
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            metrics = {
                "mAP_0.5": results_dict.get('metrics/mAP50(B)', 0.0),
                "mAP_0.5:0.95": results_dict.get('metrics/mAP50-95(B)', 0.0),
                "precision": results_dict.get('metrics/precision(B)', 0.0),
                "recall": results_dict.get('metrics/recall(B)', 0.0),
                "f1_score": 2 * (results_dict.get('metrics/precision(B)', 0.0) * results_dict.get('metrics/recall(B)', 0.0)) / 
                           (results_dict.get('metrics/precision(B)', 0.0) + results_dict.get('metrics/recall(B)', 0.0) + 1e-8)
            }
        else:
            # Fallback metrics extraction
            if hasattr(results, 'box'):
                box_results = results.box
                metrics = {
                    "mAP_0.5": float(box_results.map50) if hasattr(box_results, 'map50') else 0.0,
                    "mAP_0.5:0.95": float(box_results.map) if hasattr(box_results, 'map') else 0.0,
                    "precision": float(box_results.mp) if hasattr(box_results, 'mp') else 0.0,
                    "recall": float(box_results.mr) if hasattr(box_results, 'mr') else 0.0,
                    "f1_score": 0.0  # Will calculate below
                }
                # Calculate F1 score
                if metrics["precision"] > 0 and metrics["recall"] > 0:
                    metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Determine if model meets performance threshold
        primary_metric = metrics.get("mAP_0.5", 0.0)
        meets_threshold = primary_metric >= args.performance_threshold
        
        # Create evaluation report
        evaluation_report = {
            "model_performance": metrics,
            "performance_threshold": args.performance_threshold,
            "meets_threshold": meets_threshold,
            "primary_metric": "mAP_0.5",
            "primary_metric_value": primary_metric,
            "evaluation_config": {
                "img_size": args.img_size,
                "conf_threshold": args.conf_threshold,
                "iou_threshold": args.iou_threshold
            },
            "dataset_info": dataset_config,
            "recommendation": "APPROVE" if meets_threshold else "REJECT"
        }
        
        # Save evaluation report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        # Save metrics for pipeline conditions
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save threshold result for pipeline
        threshold_path = output_dir / "threshold_result.json"
        threshold_result = {
            "meets_threshold": meets_threshold,
            "threshold_value": args.performance_threshold,
            "actual_value": primary_metric,
            "metric_name": "mAP_0.5"
        }
        with open(threshold_path, 'w') as f:
            json.dump(threshold_result, f, indent=2)
        
        logger.info(f"Evaluation completed. Report saved to {report_path}")
        logger.info(f"Model performance - mAP@0.5: {primary_metric:.4f}")
        logger.info(f"Meets threshold ({args.performance_threshold}): {meets_threshold}")
        
        # Log artifacts to MLflow
        mlflow.log_artifact(str(report_path), "evaluation")
        mlflow.log_artifact(str(metrics_path), "evaluation")
        mlflow.log_artifact(str(threshold_path), "evaluation")
        
        return evaluation_report

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("YOLOv11 Evaluation Container Started")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check input data
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    # Evaluate the model
    evaluation_report = evaluate_model(args)
    
    logger.info("YOLOv11 Evaluation Container Completed")
    logger.info(f"Final recommendation: {evaluation_report['recommendation']}")

if __name__ == "__main__":
    main()
