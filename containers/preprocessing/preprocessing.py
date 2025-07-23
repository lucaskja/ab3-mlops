#!/usr/bin/env python3
"""
YOLOv11 Data Preprocessing Script for SageMaker Pipeline

This script preprocesses YOLOv11 format data for training, validation, and test splits.
"""

import os
import argparse
import logging
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-train", type=str, default="/opt/ml/processing/output/train")
    parser.add_argument("--output-validation", type=str, default="/opt/ml/processing/output/validation")
    parser.add_argument("--output-test", type=str, default="/opt/ml/processing/output/test")
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def load_dataset_config(input_path: str) -> Dict:
    """Load dataset configuration from data.yaml file."""
    config_path = Path(input_path) / "data.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded dataset config: {config}")
        return config
    else:
        logger.warning(f"No data.yaml found at {config_path}")
        return {}

def get_image_label_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Get pairs of image and label files."""
    pairs = []
    
    # Look for images in train/images directory
    images_dir = data_dir / "train" / "images"
    labels_dir = data_dir / "train" / "labels"
    
    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return pairs
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            # Find corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                logger.warning(f"No label file found for {img_file}")
    
    logger.info(f"Found {len(pairs)} image-label pairs")
    return pairs

def copy_files_to_split(pairs: List[Tuple[Path, Path]], output_dir: Path, split_name: str):
    """Copy image-label pairs to the appropriate split directory."""
    images_dir = output_dir / split_name / "images"
    labels_dir = output_dir / split_name / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file, label_file in pairs:
        # Copy image
        shutil.copy2(img_file, images_dir / img_file.name)
        # Copy label
        shutil.copy2(label_file, labels_dir / label_file.name)
    
    logger.info(f"Copied {len(pairs)} pairs to {split_name} split")

def create_data_yaml(output_dir: Path, config: Dict, split_name: str):
    """Create data.yaml file for the split."""
    data_yaml = {
        'path': str(output_dir / split_name),
        'train': 'images',
        'val': 'images',  # For single split, use same directory
        'names': config.get('names', {}),
        'nc': config.get('nc', len(config.get('names', {})))
    }
    
    yaml_path = output_dir / split_name / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml for {split_name} split")

def preprocess_data(input_path: str, output_train_path: str, output_validation_path: str, 
                   output_test_path: str, train_split: float, val_split: float, 
                   test_split: float, seed: int):
    """
    Preprocess YOLOv11 data and split into train, validation, and test sets.
    
    Args:
        input_path: Path to input data
        output_train_path: Path to output training data
        output_validation_path: Path to output validation data
        output_test_path: Path to output test data
        train_split: Fraction for training split
        val_split: Fraction for validation split
        test_split: Fraction for test split
        seed: Random seed for reproducibility
    """
    logger.info("Starting YOLOv11 data preprocessing")
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    output_train = Path(output_train_path)
    output_val = Path(output_validation_path)
    output_test = Path(output_test_path)
    
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)
    
    # Load dataset configuration
    input_dir = Path(input_path)
    config = load_dataset_config(input_path)
    
    # Get image-label pairs
    pairs = get_image_label_pairs(input_dir)
    
    if not pairs:
        logger.error("No image-label pairs found!")
        return
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Calculate split indices
    total_pairs = len(pairs)
    train_end = int(total_pairs * train_split)
    val_end = train_end + int(total_pairs * val_split)
    
    # Split the data
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    logger.info(f"Data split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Copy files to respective splits
    copy_files_to_split(train_pairs, output_train, "train")
    copy_files_to_split(val_pairs, output_val, "validation")
    copy_files_to_split(test_pairs, output_test, "test")
    
    # Create data.yaml files for each split
    create_data_yaml(output_train, config, "train")
    create_data_yaml(output_val, config, "validation")
    create_data_yaml(output_test, config, "test")
    
    # Create summary information
    summary = {
        "total_samples": total_pairs,
        "train_samples": len(train_pairs),
        "validation_samples": len(val_pairs),
        "test_samples": len(test_pairs),
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "classes": config.get('names', {}),
        "num_classes": config.get('nc', 0)
    }
    
    # Save summary to each output directory
    for output_dir in [output_train, output_val, output_test]:
        summary_path = output_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    logger.info("YOLOv11 data preprocessing completed successfully")

def main():
    """Main function."""
    args = parse_args()
    
    # Extract arguments
    input_path = args.input_data
    output_train_path = args.output_train
    output_validation_path = args.output_validation
    output_test_path = args.output_test
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split
    seed = args.seed
    
    # Validate splits sum to 1.0
    if abs(train_split + val_split + test_split - 1.0) > 0.001:
        logger.error(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
        return
    
    # Preprocess data
    preprocess_data(
        input_path=input_path,
        output_train_path=output_train_path,
        output_validation_path=output_validation_path,
        output_test_path=output_test_path,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed
    )

if __name__ == "__main__":
    main()
