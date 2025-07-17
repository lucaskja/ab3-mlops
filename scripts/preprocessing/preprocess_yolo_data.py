#!/usr/bin/env python3
"""
SageMaker Pipeline Preprocessing Script for YOLOv11 Data

This script preprocesses drone imagery data for YOLOv11 training,
performs data validation, and generates a data quality report.

It is designed to be run as a SageMaker Processing job within a pipeline.
"""

import os
import sys
import json
import argparse
import logging
import shutil
import time
from pathlib import Path
import random
import numpy as np
from typing import Dict, Any, List, Tuple
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess data for YOLOv11 training")
    
    parser.add_argument("--input-data", type=str, required=True,
                       help="Input data directory")
    parser.add_argument("--output-train", type=str, required=True,
                       help="Output directory for training data")
    parser.add_argument("--output-validation", type=str, required=True,
                       help="Output directory for validation data")
    parser.add_argument("--output-test", type=str, required=True,
                       help="Output directory for test data")
    parser.add_argument("--output-metadata", type=str, required=True,
                       help="Output directory for metadata")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Percentage of data for training")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Percentage of data for validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def setup_directories(args):
    """
    Setup input and output directories.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of directory paths
    """
    # Create output directories
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_validation, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)
    os.makedirs(args.output_metadata, exist_ok=True)
    
    # Create subdirectories for images and labels
    for output_dir in [args.output_train, args.output_validation, args.output_test]:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    return {
        "input": args.input_data,
        "train": args.output_train,
        "validation": args.output_validation,
        "test": args.output_test,
        "metadata": args.output_metadata
    }


def discover_dataset(input_dir: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Discover images and labels in the input directory.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        Tuple of (image_paths, label_paths, class_counts)
    """
    logger.info(f"Discovering dataset in: {input_dir}")
    
    # Find images directory
    images_dir = None
    labels_dir = None
    
    # Check common directory structures
    potential_img_dirs = [
        os.path.join(input_dir, "images"),
        os.path.join(input_dir, "img"),
        input_dir
    ]
    
    potential_label_dirs = [
        os.path.join(input_dir, "labels"),
        os.path.join(input_dir, "annotations"),
        input_dir
    ]
    
    # Find images directory
    for img_dir in potential_img_dirs:
        if os.path.isdir(img_dir) and any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(img_dir)):
            images_dir = img_dir
            break
    
    # Find labels directory
    for lbl_dir in potential_label_dirs:
        if os.path.isdir(lbl_dir) and any(f.endswith('.txt') for f in os.listdir(lbl_dir)):
            labels_dir = lbl_dir
            break
    
    if not images_dir:
        raise ValueError(f"Could not find images directory in: {input_dir}")
    
    if not labels_dir:
        raise ValueError(f"Could not find labels directory in: {input_dir}")
    
    logger.info(f"Found images directory: {images_dir}")
    logger.info(f"Found labels directory: {labels_dir}")
    
    # Get image and label files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Get corresponding label files
    image_basenames = [os.path.splitext(f)[0] for f in image_files]
    label_files = []
    class_counts = {}
    
    for basename in image_basenames:
        label_path = os.path.join(labels_dir, f"{basename}.txt")
        
        if os.path.exists(label_path):
            label_files.append(f"{basename}.txt")
            
            # Count classes in label file
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        else:
            logger.warning(f"No label file found for image: {basename}")
    
    # Create full paths
    image_paths = [os.path.join(images_dir, f) for f in image_files if os.path.splitext(f)[0] in image_basenames]
    label_paths = [os.path.join(labels_dir, f) for f in label_files]
    
    logger.info(f"Found {len(image_paths)} images and {len(label_paths)} labels")
    logger.info(f"Class distribution: {class_counts}")
    
    return image_paths, label_paths, class_counts


def split_dataset(
    image_paths: List[str],
    label_paths: List[str],
    train_split: float,
    validation_split: float,
    seed: int
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        image_paths: List of image paths
        label_paths: List of label paths
        train_split: Percentage for training
        validation_split: Percentage for validation
        seed: Random seed
        
    Returns:
        Dictionary with splits
    """
    logger.info("Splitting dataset into train, validation, and test sets")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create image-label pairs
    dataset = list(zip(image_paths, label_paths))
    random.shuffle(dataset)
    
    # Calculate split indices
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * validation_split)
    
    # Split dataset
    train_set = dataset[:n_train]
    val_set = dataset[n_train:n_train + n_val]
    test_set = dataset[n_train + n_val:]
    
    # Unzip pairs
    train_images, train_labels = zip(*train_set) if train_set else ([], [])
    val_images, val_labels = zip(*val_set) if val_set else ([], [])
    test_images, test_labels = zip(*test_set) if test_set else ([], [])
    
    logger.info(f"Split dataset: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
    
    return {
        "train": (list(train_images), list(train_labels)),
        "validation": (list(val_images), list(val_labels)),
        "test": (list(test_images), list(test_labels))
    }


def copy_dataset_split(
    split_name: str,
    images: List[str],
    labels: List[str],
    output_dir: str
) -> Dict[str, int]:
    """
    Copy dataset split to output directory.
    
    Args:
        split_name: Name of the split (train, validation, test)
        images: List of image paths
        labels: List of label paths
        output_dir: Output directory
        
    Returns:
        Statistics about the copied data
    """
    logger.info(f"Copying {split_name} split to: {output_dir}")
    
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Copy images
    for img_path in images:
        img_filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(images_dir, img_filename))
    
    # Copy labels
    class_counts = {}
    for lbl_path in labels:
        lbl_filename = os.path.basename(lbl_path)
        dest_path = os.path.join(labels_dir, lbl_filename)
        shutil.copy2(lbl_path, dest_path)
        
        # Count classes
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    stats = {
        "num_images": len(images),
        "num_labels": len(labels),
        "class_counts": class_counts
    }
    
    logger.info(f"Copied {len(images)} images and {len(labels)} labels for {split_name} split")
    return stats


def validate_yolo_format(label_paths: List[str]) -> Dict[str, Any]:
    """
    Validate YOLO format labels.
    
    Args:
        label_paths: List of label paths
        
    Returns:
        Validation results
    """
    logger.info("Validating YOLO format labels")
    
    validation_results = {
        "valid_files": 0,
        "invalid_files": 0,
        "errors": [],
        "class_distribution": {},
        "avg_objects_per_image": 0,
        "total_objects": 0
    }
    
    total_objects = 0
    
    for label_path in label_paths:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            file_valid = True
            file_objects = 0
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                
                # Check format: class_id x_center y_center width height
                if len(parts) != 5:
                    validation_results["errors"].append(
                        f"Invalid format in {label_path}, line {i+1}: expected 5 values, got {len(parts)}"
                    )
                    file_valid = False
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Check value ranges
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        validation_results["errors"].append(
                            f"Invalid values in {label_path}, line {i+1}: values must be between 0 and 1"
                        )
                        file_valid = False
                    
                    # Update class distribution
                    validation_results["class_distribution"][class_id] = \
                        validation_results["class_distribution"].get(class_id, 0) + 1
                    
                    file_objects += 1
                    total_objects += 1
                    
                except ValueError:
                    validation_results["errors"].append(
                        f"Invalid values in {label_path}, line {i+1}: could not convert to numbers"
                    )
                    file_valid = False
            
            if file_valid:
                validation_results["valid_files"] += 1
            else:
                validation_results["invalid_files"] += 1
            
        except Exception as e:
            validation_results["errors"].append(f"Error processing {label_path}: {str(e)}")
            validation_results["invalid_files"] += 1
    
    # Calculate average objects per image
    if len(label_paths) > 0:
        validation_results["avg_objects_per_image"] = total_objects / len(label_paths)
    
    validation_results["total_objects"] = total_objects
    
    logger.info(f"Validation results: {validation_results['valid_files']} valid files, "
               f"{validation_results['invalid_files']} invalid files")
    
    return validation_results


def generate_data_quality_report(
    dataset_stats: Dict[str, Dict[str, Any]],
    validation_results: Dict[str, Any],
    output_path: str
) -> Dict[str, Any]:
    """
    Generate data quality report.
    
    Args:
        dataset_stats: Statistics for each dataset split
        validation_results: Validation results
        output_path: Path to save the report
        
    Returns:
        Data quality report
    """
    logger.info("Generating data quality report")
    
    # Create report
    report = {
        "dataset_summary": {
            "total_images": sum(stats["num_images"] for stats in dataset_stats.values()),
            "total_labels": sum(stats["num_labels"] for stats in dataset_stats.values()),
            "splits": {
                split: {
                    "num_images": stats["num_images"],
                    "num_labels": stats["num_labels"]
                } for split, stats in dataset_stats.items()
            }
        },
        "validation_results": {
            "valid_files": validation_results["valid_files"],
            "invalid_files": validation_results["invalid_files"],
            "total_objects": validation_results["total_objects"],
            "avg_objects_per_image": validation_results["avg_objects_per_image"]
        },
        "class_distribution": validation_results["class_distribution"],
        "data_quality_score": calculate_data_quality_score(validation_results, dataset_stats),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Data quality report saved to: {output_path}")
    return report


def calculate_data_quality_score(
    validation_results: Dict[str, Any],
    dataset_stats: Dict[str, Dict[str, Any]]
) -> float:
    """
    Calculate data quality score based on validation results.
    
    Args:
        validation_results: Validation results
        dataset_stats: Dataset statistics
        
    Returns:
        Data quality score (0-100)
    """
    # Calculate score components
    total_files = validation_results["valid_files"] + validation_results["invalid_files"]
    
    if total_files == 0:
        return 0
    
    # Validity score (0-40 points)
    validity_score = (validation_results["valid_files"] / total_files) * 40
    
    # Class balance score (0-30 points)
    class_counts = validation_results["class_distribution"]
    if not class_counts:
        class_balance_score = 0
    else:
        counts = list(class_counts.values())
        mean_count = sum(counts) / len(counts)
        std_dev = np.std(counts) if len(counts) > 1 else 0
        
        # Lower coefficient of variation is better
        cv = std_dev / mean_count if mean_count > 0 else 1
        class_balance_score = 30 * max(0, 1 - min(cv, 1))
    
    # Split balance score (0-30 points)
    train_count = dataset_stats.get("train", {}).get("num_images", 0)
    val_count = dataset_stats.get("validation", {}).get("num_images", 0)
    test_count = dataset_stats.get("test", {}).get("num_images", 0)
    
    total_count = train_count + val_count + test_count
    
    if total_count == 0:
        split_balance_score = 0
    else:
        # Ideal split: 70-80% train, 10-15% validation, 10-15% test
        train_ratio = train_count / total_count
        val_ratio = val_count / total_count
        test_ratio = test_count / total_count
        
        train_score = 1 - min(1, abs(train_ratio - 0.75) / 0.25)
        val_score = 1 - min(1, abs(val_ratio - 0.125) / 0.125)
        test_score = 1 - min(1, abs(test_ratio - 0.125) / 0.125)
        
        split_balance_score = 30 * (train_score * 0.6 + val_score * 0.2 + test_score * 0.2)
    
    # Calculate final score
    final_score = validity_score + class_balance_score + split_balance_score
    
    logger.info(f"Data quality score: {final_score:.2f}/100 "
               f"(validity: {validity_score:.2f}, class balance: {class_balance_score:.2f}, "
               f"split balance: {split_balance_score:.2f})")
    
    return final_score


def create_dataset_yaml(
    class_distribution: Dict[int, int],
    output_dir: str,
    dataset_name: str = "drone_imagery"
) -> str:
    """
    Create YAML configuration file for YOLOv11 training.
    
    Args:
        class_distribution: Class distribution dictionary
        output_dir: Output directory
        dataset_name: Name of the dataset
        
    Returns:
        Path to the created YAML file
    """
    logger.info("Creating dataset YAML configuration")
    
    # Sort classes by ID
    class_ids = sorted(class_distribution.keys())
    
    # Create class names (placeholder)
    class_names = [f"class_{class_id}" for class_id in class_ids]
    
    # Create YAML content
    yaml_content = f"""# YOLOv11 dataset configuration
# Generated by SageMaker preprocessing script

# Dataset name
name: {dataset_name}

# Paths
path: /opt/ml/processing/output  # Will be replaced during training
train: train/images
val: validation/images
test: test/images

# Classes
nc: {len(class_ids)}  # Number of classes
names: {class_names}  # Class names
"""
    
    # Save YAML file
    yaml_path = os.path.join(output_dir, f"{dataset_name}.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Dataset YAML configuration saved to: {yaml_path}")
    return yaml_path


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting data preprocessing for YOLOv11 training")
    
    try:
        # Setup directories
        dirs = setup_directories(args)
        
        # Discover dataset
        image_paths, label_paths, class_counts = discover_dataset(args.input_data)
        
        # Validate YOLO format
        validation_results = validate_yolo_format(label_paths)
        
        # Split dataset
        splits = split_dataset(
            image_paths,
            label_paths,
            args.train_split,
            args.validation_split,
            args.seed
        )
        
        # Copy splits to output directories
        dataset_stats = {}
        for split_name, (split_images, split_labels) in splits.items():
            stats = copy_dataset_split(
                split_name,
                split_images,
                split_labels,
                dirs[split_name]
            )
            dataset_stats[split_name] = stats
        
        # Create dataset YAML configuration
        yaml_path = create_dataset_yaml(
            validation_results["class_distribution"],
            args.output_metadata,
            "drone_imagery"
        )
        
        # Generate data quality report
        report_path = os.path.join(args.output_metadata, "data_quality_report.json")
        report = generate_data_quality_report(
            dataset_stats,
            validation_results,
            report_path
        )
        
        logger.info("Data preprocessing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())