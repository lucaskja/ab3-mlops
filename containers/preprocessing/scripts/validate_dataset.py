#!/usr/bin/env python3
"""
Dataset Validation Script for SageMaker Pipeline

This script validates YOLOv11 dataset format and generates a validation report.
"""

import os
import argparse
import logging
import json
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # SageMaker processing paths (these match your pipeline configuration)
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    
    # Pipeline parameters (passed via job_arguments)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--model-variant", type=str, default="n")
    
    return parser.parse_args()

def find_dataset_structure(input_path):
    """Find the actual dataset structure in the input path."""
    input_path = Path(input_path)
    logger.info(f"Searching for dataset in: {input_path}")
    
    # List contents to understand structure
    if input_path.exists():
        contents = list(input_path.iterdir())
        logger.info(f"Input directory contents: {[item.name for item in contents]}")
        
        # Look for common dataset patterns
        possible_paths = [
            input_path,  # Direct path
            input_path / "dataset",  # Dataset subdirectory
            input_path / "yolov11_dataset_20250723_031301",  # Specific dataset name
        ]
        
        # Also check for any subdirectories that might contain the dataset
        for item in contents:
            if item.is_dir():
                possible_paths.append(item)
        
        # Find the path that contains data.yaml or train directory
        for path in possible_paths:
            if path.exists():
                if (path / "data.yaml").exists() or (path / "train").exists():
                    logger.info(f"Found dataset at: {path}")
                    return path
    
    logger.warning(f"No clear dataset structure found in {input_path}")
    return input_path

def validate_yolo_structure(dataset_path):
    """Validate YOLO dataset structure."""
    dataset_path = Path(dataset_path)
    validation_results = {
        "structure_valid": True,
        "issues": [],
        "statistics": {},
        "dataset_path": str(dataset_path)
    }
    
    logger.info(f"Validating YOLO structure at: {dataset_path}")
    
    # Check if dataset path exists
    if not dataset_path.exists():
        validation_results["structure_valid"] = False
        validation_results["issues"].append(f"Dataset path does not exist: {dataset_path}")
        return validation_results
    
    # Check for data.yaml
    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        validation_results["issues"].append("Missing data.yaml configuration file")
        # Don't mark as invalid yet, try to continue validation
    else:
        # Load data.yaml
        try:
            with open(data_yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            validation_results["config"] = config
            logger.info(f"Loaded data.yaml: {config}")
            
            # Check required fields in config
            required_fields = ['names', 'nc']
            for field in required_fields:
                if field not in config:
                    validation_results["issues"].append(f"Missing required field in data.yaml: {field}")
            
            # Validate class consistency
            if 'names' in config and 'nc' in config:
                if len(config['names']) != config['nc']:
                    validation_results["issues"].append("Number of classes (nc) doesn't match names list length")
                    
        except Exception as e:
            validation_results["issues"].append(f"Invalid data.yaml format: {e}")
    
    # Check for train directory
    train_dir = dataset_path / "train"
    if train_dir.exists():
        train_images = train_dir / "images"
        train_labels = train_dir / "labels"
        
        if train_images.exists():
            # Count images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(train_images.glob(f"*{ext}")))
                image_files.extend(list(train_images.glob(f"*{ext.upper()}")))
            
            validation_results["statistics"]["train_images"] = len(image_files)
            logger.info(f"Found {len(image_files)} training images")
            
            if train_labels.exists():
                label_files = list(train_labels.glob("*.txt"))
                validation_results["statistics"]["train_labels"] = len(label_files)
                logger.info(f"Found {len(label_files)} training labels")
                
                # Check image-label pairing (sample first 10)
                unpaired_images = 0
                sample_size = min(10, len(image_files))
                for img_file in image_files[:sample_size]:
                    label_file = train_labels / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        unpaired_images += 1
                
                if unpaired_images > 0:
                    validation_results["issues"].append(f"Found {unpaired_images} unpaired images in sample of {sample_size}")
            else:
                validation_results["issues"].append("Missing train/labels directory")
        else:
            validation_results["issues"].append("Missing train/images directory")
    else:
        validation_results["issues"].append("Missing train directory")
    
    # Determine overall validity
    critical_issues = [issue for issue in validation_results["issues"] 
                      if "Missing train" in issue or "does not exist" in issue]
    validation_results["structure_valid"] = len(critical_issues) == 0
    
    return validation_results

def validate_image_quality(dataset_path, sample_size=5):
    """Validate image quality and characteristics."""
    dataset_path = Path(dataset_path)
    quality_results = {
        "images_checked": 0,
        "corrupted_images": 0,
        "size_statistics": {},
        "format_distribution": {}
    }
    
    # Find image files
    train_images = dataset_path / "train" / "images"
    if not train_images.exists():
        logger.warning(f"Train images directory not found: {train_images}")
        return quality_results
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(train_images.glob(f"*{ext}")))
        image_files.extend(list(train_images.glob(f"*{ext.upper()}")))
    
    if not image_files:
        logger.warning("No image files found")
        return quality_results
    
    sample_files = image_files[:sample_size] if len(image_files) > sample_size else image_files
    logger.info(f"Checking {len(sample_files)} sample images for quality")
    
    widths, heights = [], []
    formats = {}
    
    for img_file in sample_files:
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                
                format_name = img.format or "Unknown"
                formats[format_name] = formats.get(format_name, 0) + 1
                
                quality_results["images_checked"] += 1
                
        except Exception as e:
            logger.warning(f"Corrupted image {img_file}: {e}")
            quality_results["corrupted_images"] += 1
    
    if widths and heights:
        quality_results["size_statistics"] = {
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)),
            "height_mean": float(np.mean(heights)),
            "height_std": float(np.std(heights)),
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights))
        }
    
    quality_results["format_distribution"] = formats
    
    return quality_results

def main():
    """Main validation function."""
    args = parse_args()
    
    logger.info("=== YOLOv11 Dataset Validation Started ===")
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info(f"Model variant: {args.model_variant}")
    logger.info(f"Input path: {args.input_data}")
    logger.info(f"Output path: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the actual dataset location
    dataset_path = find_dataset_structure(args.input_data)
    
    # Validate dataset structure
    logger.info("Validating dataset structure...")
    structure_validation = validate_yolo_structure(dataset_path)
    
    # Validate image quality
    logger.info("Validating image quality...")
    quality_validation = validate_image_quality(dataset_path)
    
    # Determine overall status
    overall_status = "PASSED"
    if not structure_validation["structure_valid"]:
        overall_status = "FAILED"
    elif quality_validation["corrupted_images"] > 0:
        overall_status = "WARNING"
    
    # Create comprehensive validation report
    validation_report = {
        "dataset_name": args.dataset_name,
        "model_variant": args.model_variant,
        "validation_timestamp": str(pd.Timestamp.now()),
        "overall_status": overall_status,
        "dataset_path": str(dataset_path),
        "structure_validation": structure_validation,
        "quality_validation": quality_validation,
        "recommendations": []
    }
    
    # Add recommendations
    if not structure_validation["structure_valid"]:
        validation_report["recommendations"].append("Fix dataset structure issues before training")
    
    if quality_validation["corrupted_images"] > 0:
        validation_report["recommendations"].append("Remove or fix corrupted images")
    
    if len(structure_validation["issues"]) == 0 and quality_validation["corrupted_images"] == 0:
        validation_report["recommendations"].append("Dataset is ready for training")
    
    # Save validation report
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"=== Validation Completed ===")
    logger.info(f"Status: {validation_report['overall_status']}")
    logger.info(f"Issues found: {len(structure_validation['issues'])}")
    logger.info(f"Images checked: {quality_validation['images_checked']}")
    logger.info(f"Corrupted images: {quality_validation['corrupted_images']}")
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary for logs
    print(f"Dataset Validation Summary:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Status: {validation_report['overall_status']}")
    print(f"  Issues: {len(structure_validation['issues'])}")
    print(f"  Images checked: {quality_validation['images_checked']}")
    print(f"  Corrupted images: {quality_validation['corrupted_images']}")
    
    # Exit with appropriate code
    if overall_status == "FAILED":
        logger.error("Validation failed - critical issues found")
        exit(1)
    elif overall_status == "WARNING":
        logger.warning("Validation completed with warnings")
        exit(0)
    else:
        logger.info("Validation passed successfully")
        exit(0)

if __name__ == "__main__":
    main()
