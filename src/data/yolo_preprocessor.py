"""
YOLOv11 Data Preprocessing Pipeline

This module provides comprehensive data preprocessing capabilities for converting
drone imagery to YOLOv11 format, including data augmentation and validation.

Requirements addressed:
- 4.1: Data preprocessing functions to convert drone imagery to YOLOv11 format
- 4.2: Data augmentation pipeline for training data enhancement
- Data validation functions to ensure annotation quality
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import yaml
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import random
import math
import os
from dataclasses import dataclass
from enum import Enum
import albumentations as A
# Make pytorch import optional for testing
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    ToTensorV2 = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationFormat(Enum):
    """Supported annotation formats for conversion"""
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    YOLO = "yolo"
    CUSTOM_JSON = "custom_json"


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x_center: float  # Normalized center x (0-1)
    y_center: float  # Normalized center y (0-1)
    width: float     # Normalized width (0-1)
    height: float    # Normalized height (0-1)
    class_id: int    # Class identifier
    confidence: float = 1.0  # Confidence score


@dataclass
class ImageAnnotation:
    """Image with annotations"""
    image_path: str
    image_width: int
    image_height: int
    bboxes: List[BoundingBox]
    metadata: Dict[str, Any] = None


class YOLOv11Preprocessor:
    """
    Comprehensive preprocessor for YOLOv11 format conversion and augmentation.
    
    Handles conversion from various annotation formats to YOLOv11 format,
    applies data augmentation, and validates data quality.
    """
    
    def __init__(self, s3_access, target_size: int = 640, class_names: Optional[List[str]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            s3_access: S3DataAccess instance for data retrieval
            target_size: Target image size for YOLOv11 (default: 640)
            class_names: List of class names for the dataset
        """
        self.s3_access = s3_access
        self.target_size = target_size
        self.class_names = class_names or []
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _create_augmentation_pipeline(self) -> A.Compose:
        """
        Create Albumentations augmentation pipeline for YOLOv11.
        
        Returns:
            Albumentations composition for augmentation
        """
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.Affine(
                translate_percent=0.1,
                scale=(0.8, 1.2),
                rotate=15,
                p=0.5
            ),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise and blur
            A.GaussNoise(noise_scale_factor=0.1, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.1),
            
            # Weather effects
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.RandomSunFlare(p=0.05),
            
            # Cutout and mixup-like effects
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                p=0.2
            ),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0.01,
            min_visibility=0.3
        ))
    
    def convert_annotations_to_yolo(self, 
                                  annotation_data: Dict[str, Any], 
                                  format_type: AnnotationFormat) -> List[ImageAnnotation]:
        """
        Convert annotations from various formats to YOLOv11 format.
        
        Args:
            annotation_data: Annotation data in source format
            format_type: Source annotation format
            
        Returns:
            List of ImageAnnotation objects in YOLOv11 format
        """
        logger.info(f"Converting annotations from {format_type.value} to YOLOv11 format")
        
        if format_type == AnnotationFormat.COCO:
            return self._convert_from_coco(annotation_data)
        elif format_type == AnnotationFormat.PASCAL_VOC:
            return self._convert_from_pascal_voc(annotation_data)
        elif format_type == AnnotationFormat.YOLO:
            return self._convert_from_yolo(annotation_data)
        elif format_type == AnnotationFormat.CUSTOM_JSON:
            return self._convert_from_custom_json(annotation_data)
        else:
            raise ValueError(f"Unsupported annotation format: {format_type}")
    
    def _convert_from_coco(self, coco_data: Dict[str, Any]) -> List[ImageAnnotation]:
        """Convert COCO format annotations to YOLOv11 format"""
        annotations = []
        
        # Create image ID to info mapping
        image_info = {img['id']: img for img in coco_data.get('images', [])}
        
        # Create category ID to name mapping
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Convert each image
        for image_id, img_info in image_info.items():
            bboxes = []
            
            if image_id in image_annotations:
                for ann in image_annotations[image_id]:
                    # COCO bbox format: [x, y, width, height] (absolute coordinates)
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLOv11 format (normalized center coordinates)
                    x_center = (x + w / 2) / img_info['width']
                    y_center = (y + h / 2) / img_info['height']
                    norm_width = w / img_info['width']
                    norm_height = h / img_info['height']
                    
                    # Get class ID
                    category_name = categories.get(ann['category_id'], 'unknown')
                    class_id = self.class_to_id.get(category_name, 0)
                    
                    bbox = BoundingBox(
                        x_center=x_center,
                        y_center=y_center,
                        width=norm_width,
                        height=norm_height,
                        class_id=class_id
                    )
                    bboxes.append(bbox)
            
            annotation = ImageAnnotation(
                image_path=img_info.get('file_name', f"image_{image_id}"),
                image_width=img_info['width'],
                image_height=img_info['height'],
                bboxes=bboxes,
                metadata={'source_format': 'coco', 'image_id': image_id}
            )
            annotations.append(annotation)
        
        logger.info(f"Converted {len(annotations)} images from COCO format")
        return annotations
    
    def _convert_from_pascal_voc(self, voc_data: List[str]) -> List[ImageAnnotation]:
        """Convert PASCAL VOC XML annotations to YOLOv11 format"""
        annotations = []
        
        for xml_content in voc_data:
            try:
                root = ET.fromstring(xml_content)
                
                # Extract image information
                filename = root.find('filename').text if root.find('filename') is not None else 'unknown'
                size_elem = root.find('size')
                
                if size_elem is not None:
                    width = int(size_elem.find('width').text)
                    height = int(size_elem.find('height').text)
                else:
                    logger.warning(f"No size information found for {filename}")
                    continue
                
                # Extract bounding boxes
                bboxes = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = self.class_to_id.get(class_name, 0)
                    
                    bbox_elem = obj.find('bndbox')
                    if bbox_elem is not None:
                        xmin = float(bbox_elem.find('xmin').text)
                        ymin = float(bbox_elem.find('ymin').text)
                        xmax = float(bbox_elem.find('xmax').text)
                        ymax = float(bbox_elem.find('ymax').text)
                        
                        # Convert to YOLOv11 format
                        x_center = (xmin + xmax) / 2 / width
                        y_center = (ymin + ymax) / 2 / height
                        bbox_width = (xmax - xmin) / width
                        bbox_height = (ymax - ymin) / height
                        
                        bbox = BoundingBox(
                            x_center=x_center,
                            y_center=y_center,
                            width=bbox_width,
                            height=bbox_height,
                            class_id=class_id
                        )
                        bboxes.append(bbox)
                
                annotation = ImageAnnotation(
                    image_path=filename,
                    image_width=width,
                    image_height=height,
                    bboxes=bboxes,
                    metadata={'source_format': 'pascal_voc'}
                )
                annotations.append(annotation)
                
            except Exception as e:
                logger.error(f"Error parsing VOC XML: {str(e)}")
                continue
        
        logger.info(f"Converted {len(annotations)} images from PASCAL VOC format")
        return annotations
    
    def _convert_from_yolo(self, yolo_data: Dict[str, str]) -> List[ImageAnnotation]:
        """Convert existing YOLO format annotations (validation/normalization)"""
        annotations = []
        
        for image_path, annotation_content in yolo_data.items():
            bboxes = []
            
            # Parse YOLO annotation lines
            for line in annotation_content.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Validate coordinates are normalized
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                logger.warning(f"Invalid normalized coordinates in {image_path}: {line}")
                                continue
                            
                            bbox = BoundingBox(
                                x_center=x_center,
                                y_center=y_center,
                                width=width,
                                height=height,
                                class_id=class_id
                            )
                            bboxes.append(bbox)
                            
                        except ValueError as e:
                            logger.warning(f"Error parsing YOLO annotation line '{line}': {str(e)}")
                            continue
            
            # Assume standard image dimensions if not provided
            annotation = ImageAnnotation(
                image_path=image_path,
                image_width=self.target_size,
                image_height=self.target_size,
                bboxes=bboxes,
                metadata={'source_format': 'yolo'}
            )
            annotations.append(annotation)
        
        logger.info(f"Validated {len(annotations)} images from YOLO format")
        return annotations
    
    def _convert_from_custom_json(self, json_data: Dict[str, Any]) -> List[ImageAnnotation]:
        """Convert custom JSON format to YOLOv11 format"""
        annotations = []
        
        # This is a flexible converter for custom JSON formats
        # Adapt based on your specific JSON structure
        
        if 'images' in json_data:
            for img_data in json_data['images']:
                bboxes = []
                
                for ann in img_data.get('annotations', []):
                    # Assume bbox format can vary - handle multiple possibilities
                    if 'bbox' in ann:
                        bbox_data = ann['bbox']
                        
                        # Handle different bbox formats
                        if len(bbox_data) == 4:
                            if 'format' in ann and ann['format'] == 'yolo':
                                # Already in YOLO format
                                x_center, y_center, width, height = bbox_data
                            else:
                                # Assume [x, y, width, height] format
                                x, y, w, h = bbox_data
                                x_center = (x + w / 2) / img_data['width']
                                y_center = (y + h / 2) / img_data['height']
                                width = w / img_data['width']
                                height = h / img_data['height']
                        else:
                            logger.warning(f"Unsupported bbox format: {bbox_data}")
                            continue
                        
                        class_name = ann.get('class', ann.get('category', 'unknown'))
                        class_id = self.class_to_id.get(class_name, 0)
                        
                        bbox = BoundingBox(
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height,
                            class_id=class_id
                        )
                        bboxes.append(bbox)
                
                annotation = ImageAnnotation(
                    image_path=img_data.get('file_name', img_data.get('path', 'unknown')),
                    image_width=img_data.get('width', self.target_size),
                    image_height=img_data.get('height', self.target_size),
                    bboxes=bboxes,
                    metadata={'source_format': 'custom_json'}
                )
                annotations.append(annotation)
        
        logger.info(f"Converted {len(annotations)} images from custom JSON format")
        return annotations
    
    def preprocess_image(self, 
                        image_data: Union[np.ndarray, Image.Image], 
                        target_size: Optional[int] = None) -> np.ndarray:
        """
        Preprocess image for YOLOv11 training.
        
        Args:
            image_data: Input image as numpy array or PIL Image
            target_size: Target size for resizing (defaults to self.target_size)
            
        Returns:
            Preprocessed image as numpy array
        """
        target_size = target_size or self.target_size
        
        # Convert PIL to numpy if needed
        if isinstance(image_data, Image.Image):
            image = np.array(image_data)
        else:
            image = image_data.copy()
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize with aspect ratio preservation
        image = self._resize_with_padding(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio using padding.
        
        Args:
            image: Input image
            target_size: Target square size
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_size / w, target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Calculate padding offsets
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    def apply_augmentation(self, 
                          image: np.ndarray, 
                          bboxes: List[BoundingBox],
                          augmentation_probability: float = 0.8) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Apply data augmentation to image and bounding boxes.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            augmentation_probability: Probability of applying augmentation
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes)
        """
        if random.random() > augmentation_probability:
            return image, bboxes
        
        # Convert bboxes to albumentations format
        bbox_list = []
        class_labels = []
        
        for bbox in bboxes:
            bbox_list.append([bbox.x_center, bbox.y_center, bbox.width, bbox.height])
            class_labels.append(bbox.class_id)
        
        try:
            # Apply augmentation
            augmented = self.augmentation_pipeline(
                image=image,
                bboxes=bbox_list,
                class_labels=class_labels
            )
            
            # Convert back to BoundingBox objects
            augmented_bboxes = []
            for bbox_coords, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                augmented_bbox = BoundingBox(
                    x_center=bbox_coords[0],
                    y_center=bbox_coords[1],
                    width=bbox_coords[2],
                    height=bbox_coords[3],
                    class_id=class_id
                )
                augmented_bboxes.append(augmented_bbox)
            
            return augmented['image'], augmented_bboxes
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {str(e)}, returning original")
            return image, bboxes
    
    def validate_annotation_quality(self, annotations: List[ImageAnnotation]) -> Dict[str, Any]:
        """
        Validate annotation quality for YOLOv11 training.
        
        Args:
            annotations: List of ImageAnnotation objects
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating annotation quality for {len(annotations)} images")
        
        validation_results = {
            'total_images': len(annotations),
            'valid_images': 0,
            'invalid_images': 0,
            'total_bboxes': 0,
            'valid_bboxes': 0,
            'invalid_bboxes': 0,
            'issues': [],
            'class_distribution': {},
            'bbox_size_stats': {
                'min_area': float('inf'),
                'max_area': 0,
                'avg_area': 0,
                'areas': []
            }
        }
        
        for idx, annotation in enumerate(annotations):
            image_valid = True
            
            # Validate image dimensions
            if annotation.image_width <= 0 or annotation.image_height <= 0:
                validation_results['issues'].append(f"Invalid image dimensions for {annotation.image_path}")
                image_valid = False
            
            # Validate bounding boxes
            for bbox_idx, bbox in enumerate(annotation.bboxes):
                bbox_valid = True
                validation_results['total_bboxes'] += 1
                
                # Check coordinate bounds
                if not (0 <= bbox.x_center <= 1 and 0 <= bbox.y_center <= 1):
                    validation_results['issues'].append(
                        f"Invalid center coordinates in {annotation.image_path}, bbox {bbox_idx}"
                    )
                    bbox_valid = False
                
                if not (0 < bbox.width <= 1 and 0 < bbox.height <= 1):
                    validation_results['issues'].append(
                        f"Invalid dimensions in {annotation.image_path}, bbox {bbox_idx}"
                    )
                    bbox_valid = False
                
                # Check if bbox extends beyond image bounds
                x_min = bbox.x_center - bbox.width / 2
                x_max = bbox.x_center + bbox.width / 2
                y_min = bbox.y_center - bbox.height / 2
                y_max = bbox.y_center + bbox.height / 2
                
                if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                    validation_results['issues'].append(
                        f"Bounding box extends beyond image bounds in {annotation.image_path}, bbox {bbox_idx}"
                    )
                    bbox_valid = False
                
                # Check minimum size
                area = bbox.width * bbox.height
                if area < 0.0001:  # Very small objects might be noise
                    validation_results['issues'].append(
                        f"Very small bounding box in {annotation.image_path}, bbox {bbox_idx}"
                    )
                
                # Update statistics
                if bbox_valid:
                    validation_results['valid_bboxes'] += 1
                    validation_results['bbox_size_stats']['areas'].append(area)
                    validation_results['bbox_size_stats']['min_area'] = min(
                        validation_results['bbox_size_stats']['min_area'], area
                    )
                    validation_results['bbox_size_stats']['max_area'] = max(
                        validation_results['bbox_size_stats']['max_area'], area
                    )
                    
                    # Update class distribution
                    class_id = bbox.class_id
                    validation_results['class_distribution'][class_id] = \
                        validation_results['class_distribution'].get(class_id, 0) + 1
                else:
                    validation_results['invalid_bboxes'] += 1
                    image_valid = False
            
            if image_valid:
                validation_results['valid_images'] += 1
            else:
                validation_results['invalid_images'] += 1
        
        # Calculate average area
        if validation_results['bbox_size_stats']['areas']:
            validation_results['bbox_size_stats']['avg_area'] = np.mean(
                validation_results['bbox_size_stats']['areas']
            )
        
        # Generate summary
        validation_results['summary'] = {
            'image_validity_rate': validation_results['valid_images'] / max(validation_results['total_images'], 1),
            'bbox_validity_rate': validation_results['valid_bboxes'] / max(validation_results['total_bboxes'], 1),
            'avg_bboxes_per_image': validation_results['total_bboxes'] / max(validation_results['total_images'], 1),
            'num_classes': len(validation_results['class_distribution'])
        }
        
        logger.info(f"Validation completed: {validation_results['valid_images']}/{validation_results['total_images']} valid images")
        
        return validation_results
    
    def save_yolo_annotations(self, 
                             annotations: List[ImageAnnotation], 
                             output_dir: str,
                             split_ratios: Dict[str, float] = None) -> Dict[str, List[str]]:
        """
        Save annotations in YOLOv11 format and create dataset splits.
        
        Args:
            annotations: List of ImageAnnotation objects
            output_dir: Output directory for saving annotations
            split_ratios: Dictionary with train/val/test split ratios
            
        Returns:
            Dictionary with file paths for each split
        """
        split_ratios = split_ratios or {'train': 0.8, 'val': 0.2}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for split in split_ratios.keys():
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
        
        # Shuffle annotations for random splits
        shuffled_annotations = annotations.copy()
        random.shuffle(shuffled_annotations)
        
        # Calculate split indices
        total = len(shuffled_annotations)
        splits = {}
        start_idx = 0
        
        for split_name, ratio in split_ratios.items():
            end_idx = start_idx + int(total * ratio)
            if split_name == list(split_ratios.keys())[-1]:  # Last split gets remaining
                end_idx = total
            splits[split_name] = shuffled_annotations[start_idx:end_idx]
            start_idx = end_idx
        
        # Save annotations for each split
        saved_files = {}
        
        for split_name, split_annotations in splits.items():
            split_files = []
            
            for annotation in split_annotations:
                # Generate filename
                base_name = Path(annotation.image_path).stem
                label_file = os.path.join(output_dir, split_name, 'labels', f"{base_name}.txt")
                
                # Write YOLO format annotation
                with open(label_file, 'w') as f:
                    for bbox in annotation.bboxes:
                        f.write(f"{bbox.class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} "
                               f"{bbox.width:.6f} {bbox.height:.6f}\n")
                
                split_files.append(label_file)
            
            saved_files[split_name] = split_files
            logger.info(f"Saved {len(split_files)} {split_name} annotations")
        
        # Create dataset configuration file
        dataset_config = {
            'path': output_dir,
            'train': os.path.join(output_dir, 'train', 'images'),
            'val': os.path.join(output_dir, 'val', 'images'),
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        if 'test' in splits:
            dataset_config['test'] = os.path.join(output_dir, 'test', 'images')
        
        config_file = os.path.join(output_dir, 'dataset.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved to {config_file}")
        
        return saved_files
    
    def create_dataset_summary(self, annotations: List[ImageAnnotation]) -> Dict[str, Any]:
        """
        Create comprehensive dataset summary for YOLOv11 training.
        
        Args:
            annotations: List of ImageAnnotation objects
            
        Returns:
            Dictionary with dataset summary statistics
        """
        summary = {
            'dataset_info': {
                'total_images': len(annotations),
                'total_annotations': sum(len(ann.bboxes) for ann in annotations),
                'num_classes': len(self.class_names),
                'class_names': self.class_names
            },
            'image_statistics': {
                'resolutions': [],
                'aspect_ratios': [],
                'avg_annotations_per_image': 0
            },
            'annotation_statistics': {
                'class_distribution': {},
                'bbox_areas': [],
                'bbox_aspect_ratios': []
            },
            'quality_metrics': {}
        }
        
        # Collect statistics
        for annotation in annotations:
            # Image statistics
            summary['image_statistics']['resolutions'].append(
                (annotation.image_width, annotation.image_height)
            )
            summary['image_statistics']['aspect_ratios'].append(
                annotation.image_width / annotation.image_height
            )
            
            # Annotation statistics
            for bbox in annotation.bboxes:
                class_id = bbox.class_id
                summary['annotation_statistics']['class_distribution'][class_id] = \
                    summary['annotation_statistics']['class_distribution'].get(class_id, 0) + 1
                
                area = bbox.width * bbox.height
                summary['annotation_statistics']['bbox_areas'].append(area)
                summary['annotation_statistics']['bbox_aspect_ratios'].append(
                    bbox.width / bbox.height
                )
        
        # Calculate derived statistics
        if summary['dataset_info']['total_images'] > 0:
            summary['image_statistics']['avg_annotations_per_image'] = \
                summary['dataset_info']['total_annotations'] / summary['dataset_info']['total_images']
        
        if summary['annotation_statistics']['bbox_areas']:
            areas = summary['annotation_statistics']['bbox_areas']
            summary['annotation_statistics']['area_stats'] = {
                'min': min(areas),
                'max': max(areas),
                'mean': np.mean(areas),
                'std': np.std(areas)
            }
        
        # Quality assessment
        validation_results = self.validate_annotation_quality(annotations)
        summary['quality_metrics'] = validation_results['summary']
        
        return summary


def create_yolo_preprocessor(s3_access, 
                           target_size: int = 640, 
                           class_names: List[str] = None) -> YOLOv11Preprocessor:
    """
    Factory function to create YOLOv11Preprocessor instance.
    
    Args:
        s3_access: S3DataAccess instance
        target_size: Target image size for YOLOv11
        class_names: List of class names
        
    Returns:
        Configured YOLOv11Preprocessor instance
    """
    return YOLOv11Preprocessor(
        s3_access=s3_access,
        target_size=target_size,
        class_names=class_names
    )