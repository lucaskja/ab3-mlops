#!/usr/bin/env python3
"""
YOLOv11 Model Evaluation Script for SageMaker Pipeline

This script evaluates a trained YOLOv11 model and generates evaluation.json for pipeline conditions.
"""

import os
import argparse
import logging
import json
import yaml
import tarfile
from pathlib import Path
from ultralytics import YOLO
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # SageMaker processing paths
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    
    # Pipeline parameters
    parser.add_argument("--model-variant", type=str, default="n")
    parser.add_argument("--dataset-name", type=str, required=True)
    
    # Evaluation parameters
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    
    return parser.parse_args()

def extract_model(model_path):
    """Extract model from tar.gz if needed."""
    model_path = Path(model_path)
    
    # Look for model files
    model_files = list(model_path.glob("*.pt")) + list(model_path.glob("model.tar.gz"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_path}")
    
    model_file = model_files[0]
    
    # If it's a tar.gz file, extract it
    if model_file.suffix == '.gz':
        extract_dir = model_path / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        with tarfile.open(model_file, "r:gz") as tar:
            tar.extractall(extract_dir)
        
        # Find the extracted model
        extracted_models = list(extract_dir.glob("*.pt"))
        if extracted_models:
            model_file = extracted_models[0]
        else:
            raise FileNotFoundError("No .pt file found in extracted model")
    
    logger.info(f"Using model file: {model_file}")
    return str(model_file)

def setup_test_dataset(test_data_path):
    """Setup test dataset configuration."""
    test_path = Path(test_data_path)
    
    # Look for existing data.yaml
    data_yaml_path = test_path / "data.yaml"
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Found existing test data.yaml: {config}")
        return str(data_yaml_path), config
    
    # Look for test structure
    test_images = test_path / "test" / "images"
    test_labels = test_path / "test" / "labels"
    
    if not test_images.exists():
        # Try alternative structures
        alt_paths = [
            test_path / "images",
            test_path / "validation" / "images",
            test_path / "val" / "images"
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                test_images = alt_path
                test_labels = alt_path.parent / "labels"
                break
        else:
            raise FileNotFoundError(f"Test images not found in {test_path}")
    
    # Create test configuration
    config = {
        'path': str(test_path),
        'val': str(test_images.relative_to(test_path)),
        'names': {0: 'object'},  # Default single class
        'nc': 1
    }
    
    # Try to infer classes from labels if available
    if test_labels.exists():
        label_files = list(test_labels.glob("*.txt"))
        if label_files:
            try:
                class_ids = set()
                for label_file in label_files[:10]:  # Sample first 10
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                
                if class_ids:
                    max_class = max(class_ids)
                    config['nc'] = max_class + 1
                    config['names'] = {i: f'class_{i}' for i in range(max_class + 1)}
            except Exception as e:
                logger.warning(f"Could not infer classes from test labels: {e}")
    
    # Save test configuration
    with open(data_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created test data.yaml: {config}")
    return str(data_yaml_path), config

def evaluate_model(args):
    """Evaluate YOLOv11 model."""
    logger.info("Starting YOLOv11 model evaluation")
    
    # Extract and load model
    model_file = extract_model(args.model_path)
    model = YOLO(model_file)
    logger.info(f"Loaded model from: {model_file}")
    
    # Setup test dataset
    test_config_path, dataset_config = setup_test_dataset(args.test_data)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    logger.info("Running model validation...")
    try:
        results = model.val(
            data=test_config_path,
            imgsz=args.img_size,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            verbose=True,
            device='cpu'
        )
        
        # Extract metrics
        metrics = {}
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            metrics = {
                "mAP_50": float(results_dict.get('metrics/mAP50(B)', 0.0)),
                "mAP_50_95": float(results_dict.get('metrics/mAP50-95(B)', 0.0)),
                "precision": float(results_dict.get('metrics/precision(B)', 0.0)),
                "recall": float(results_dict.get('metrics/recall(B)', 0.0))
            }
        elif hasattr(results, 'box'):
            # Alternative metrics extraction
            box_results = results.box
            metrics = {
                "mAP_50": float(box_results.map50) if hasattr(box_results, 'map50') else 0.0,
                "mAP_50_95": float(box_results.map) if hasattr(box_results, 'map') else 0.0,
                "precision": float(box_results.mp) if hasattr(box_results, 'mp') else 0.0,
                "recall": float(box_results.mr) if hasattr(box_results, 'mr') else 0.0
            }
        else:
            # Fallback metrics
            logger.warning("Could not extract detailed metrics, using defaults")
            metrics = {
                "mAP_50": 0.0,
                "mAP_50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Calculate F1 score
        if metrics["precision"] > 0 and metrics["recall"] > 0:
            metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1_score"] = 0.0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Provide default metrics for pipeline to continue
        metrics = {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # Create evaluation report for pipeline
    evaluation_report = {
        "metrics": metrics,
        "model_variant": args.model_variant,
        "dataset_name": args.dataset_name,
        "evaluation_config": {
            "img_size": args.img_size,
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold
        },
        "dataset_info": dataset_config
    }
    
    # Save evaluation.json (required by pipeline PropertyFile)
    evaluation_path = output_dir / "evaluation.json"
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Also save detailed report
    detailed_report_path = output_dir / "detailed_evaluation_report.json"
    detailed_report = {
        **evaluation_report,
        "evaluation_timestamp": str(pd.Timestamp.now()),
        "model_file": model_file,
        "test_config": test_config_path
    }
    
    with open(detailed_report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    logger.info(f"Evaluation completed. Report saved to {evaluation_path}")
    logger.info(f"Model performance - mAP@0.5: {metrics['mAP_50']:.4f}")
    
    return evaluation_report

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("YOLOv11 Evaluation Container Started")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info(f"Model variant: {args.model_variant}")
    
    # Check inputs exist
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    if not Path(args.test_data).exists():
        raise FileNotFoundError(f"Test data not found: {args.test_data}")
    
    # Evaluate the model
    evaluation_report = evaluate_model(args)
    
    logger.info("YOLOv11 Evaluation Container Completed Successfully")
    logger.info(f"Final mAP@0.5: {evaluation_report['metrics']['mAP_50']:.4f}")

if __name__ == "__main__":
    import pandas as pd
    main()
