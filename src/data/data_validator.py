"""
Data Validation Utilities for YOLOv11 Format Requirements

This module provides validation functions to ensure data compatibility
with YOLOv11 format requirements and ML pipeline standards.

Requirements addressed:
- Data quality validation functions for YOLOv11 format requirements
- Comprehensive dataset structure validation
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import json
import yaml
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv11Validator:
    """
    Validator for YOLOv11 format requirements and data quality standards.
    
    Ensures dataset compatibility with YOLOv11 training requirements
    and provides quality assessment metrics.
    """
    
    # YOLOv11 standard input sizes
    YOLO_STANDARD_SIZES = [320, 416, 512, 608, 640, 736, 832, 896, 960, 1024, 1280]
    
    # Minimum requirements for YOLOv11
    MIN_IMAGE_SIZE = 32
    MAX_IMAGE_SIZE = 4096
    SUPPORTED_FORMATS = ['JPEG', 'PNG', 'BMP', 'TIFF']
    SUPPORTED_COLOR_MODES = ['RGB', 'L', 'RGBA']
    
    def __init__(self, s3_access):
        """
        Initialize validator with S3 access.
        
        Args:
            s3_access: S3DataAccess instance for data retrieval
        """
        self.s3_access = s3_access
        
    def validate_dataset_structure(self, image_files: List[str], annotation_files: List[str]) -> Dict[str, Any]:
        """
        Validate overall dataset structure for YOLOv11 compatibility.
        
        Args:
            image_files: List of image file keys
            annotation_files: List of annotation file keys
            
        Returns:
            Dictionary with validation results and recommendations
        """
        logger.info("Validating dataset structure for YOLOv11 compatibility...")
        
        results = {
            'valid_images': 0,
            'images_need_preprocessing': 0,
            'annotation_files_found': len(annotation_files),
            'recommended_input_size': None,
            'issues': [],
            'recommendations': [],
            'format_distribution': {},
            'size_distribution': {},
            'quality_summary': {}
        }
        
        # Analyze image files
        if image_files:
            image_analysis = self._analyze_image_compatibility(image_files[:50])  # Sample for performance
            results.update(image_analysis)
        else:
            results['issues'].append("No image files found in dataset")
        
        # Analyze annotation files
        if annotation_files:
            annotation_analysis = self._analyze_annotation_files(annotation_files[:20])  # Sample for performance
            results['annotation_analysis'] = annotation_analysis
        else:
            results['issues'].append("No annotation files found - dataset appears to be unlabeled")
            results['recommendations'].append("Consider creating annotations for supervised learning")
        
        # Generate overall recommendations
        overall_recommendations = self._generate_dataset_recommendations(results)
        results['recommendations'].extend(overall_recommendations)
        
        logger.info(f"Dataset validation completed: {results['valid_images']} valid images, {len(results['issues'])} issues found")
        
        return results
    
    def _analyze_image_compatibility(self, image_files: List[str]) -> Dict[str, Any]:
        """
        Analyze image files for YOLOv11 compatibility.
        
        Args:
            image_files: List of image file keys to analyze
            
        Returns:
            Dictionary with compatibility analysis results
        """
        valid_count = 0
        preprocessing_needed = 0
        format_counts = {}
        size_counts = {}
        resolution_data = []
        
        for img_key in image_files:
            try:
                # Download and analyze image
                img_data = self.s3_access.download_file_to_memory(img_key)
                
                with Image.open(img_data) as img:
                    width, height = img.size
                    format_name = img.format or 'Unknown'
                    color_mode = img.mode
                    
                    # Count formats
                    format_counts[format_name] = format_counts.get(format_name, 0) + 1
                    
                    # Analyze size compatibility
                    min_dim = min(width, height)
                    max_dim = max(width, height)
                    
                    resolution_data.append((width, height))
                    
                    # Check if image meets YOLOv11 requirements
                    is_valid = self._check_image_validity(img, width, height, format_name, color_mode)
                    
                    if is_valid:
                        valid_count += 1
                    else:
                        preprocessing_needed += 1
                        
                    # Categorize by size
                    size_category = self._categorize_image_size(min_dim)
                    size_counts[size_category] = size_counts.get(size_category, 0) + 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing image {img_key}: {str(e)}")
                preprocessing_needed += 1
        
        # Determine recommended input size
        recommended_size = self._recommend_input_size(resolution_data)
        
        return {
            'valid_images': valid_count,
            'images_need_preprocessing': preprocessing_needed,
            'format_distribution': format_counts,
            'size_distribution': size_counts,
            'recommended_input_size': recommended_size,
            'resolution_data': resolution_data
        }
    
    def _check_image_validity(self, img: Image.Image, width: int, height: int, 
                            format_name: str, color_mode: str) -> bool:
        """
        Check if an individual image meets YOLOv11 requirements.
        
        Args:
            img: PIL Image object
            width: Image width
            height: Image height
            format_name: Image format
            color_mode: Image color mode
            
        Returns:
            True if image is valid for YOLOv11, False otherwise
        """
        # Check minimum size requirements
        if min(width, height) < self.MIN_IMAGE_SIZE:
            return False
        
        # Check maximum size requirements
        if max(width, height) > self.MAX_IMAGE_SIZE:
            return False
        
        # Check format compatibility
        if format_name not in self.SUPPORTED_FORMATS:
            return False
        
        # Check color mode compatibility
        if color_mode not in self.SUPPORTED_COLOR_MODES:
            return False
        
        # Check if image is corrupted
        try:
            img.verify()
            return True
        except Exception:
            return False
    
    def _categorize_image_size(self, min_dimension: int) -> str:
        """
        Categorize image by size for analysis.
        
        Args:
            min_dimension: Minimum dimension of the image
            
        Returns:
            Size category string
        """
        if min_dimension < 320:
            return "small (<320px)"
        elif min_dimension < 640:
            return "medium (320-640px)"
        elif min_dimension < 1024:
            return "large (640-1024px)"
        else:
            return "extra_large (>1024px)"
    
    def _recommend_input_size(self, resolution_data: List[Tuple[int, int]]) -> int:
        """
        Recommend optimal input size for YOLOv11 based on dataset characteristics.
        
        Args:
            resolution_data: List of (width, height) tuples
            
        Returns:
            Recommended input size
        """
        if not resolution_data:
            return 640  # Default YOLOv11 size
        
        # Calculate average minimum dimension
        min_dimensions = [min(w, h) for w, h in resolution_data]
        avg_min_dim = np.mean(min_dimensions)
        
        # Find closest standard YOLO size
        closest_size = min(self.YOLO_STANDARD_SIZES, key=lambda x: abs(x - avg_min_dim))
        
        # Ensure it's not too small for the dataset
        if closest_size < np.percentile(min_dimensions, 25):  # 25th percentile
            # Find next larger size
            larger_sizes = [s for s in self.YOLO_STANDARD_SIZES if s > closest_size]
            if larger_sizes:
                closest_size = larger_sizes[0]
        
        return closest_size
    
    def _analyze_annotation_files(self, annotation_files: List[str]) -> Dict[str, Any]:
        """
        Analyze annotation files to understand labeling format and quality.
        
        Args:
            annotation_files: List of annotation file keys
            
        Returns:
            Dictionary with annotation analysis results
        """
        analysis = {
            'total_files': len(annotation_files),
            'formats_detected': {},
            'sample_analysis': [],
            'issues': [],
            'yolo_format_compatible': 0
        }
        
        # Sample a few annotation files for analysis
        sample_files = annotation_files[:10] if len(annotation_files) > 10 else annotation_files
        
        for ann_file in sample_files:
            try:
                # Download annotation file
                file_data = self.s3_access.download_file_to_memory(ann_file)
                content = file_data.read().decode('utf-8')
                
                # Detect format and analyze
                format_info = self._detect_annotation_format(ann_file, content)
                analysis['sample_analysis'].append({
                    'file': ann_file,
                    'format': format_info['format'],
                    'valid': format_info['valid'],
                    'details': format_info['details']
                })
                
                # Count formats
                format_name = format_info['format']
                analysis['formats_detected'][format_name] = analysis['formats_detected'].get(format_name, 0) + 1
                
                # Check YOLO compatibility
                if format_info['yolo_compatible']:
                    analysis['yolo_format_compatible'] += 1
                    
            except Exception as e:
                analysis['issues'].append(f"Error reading {ann_file}: {str(e)}")
        
        return analysis
    
    def _detect_annotation_format(self, filename: str, content: str) -> Dict[str, Any]:
        """
        Detect annotation file format and assess compatibility.
        
        Args:
            filename: Name of the annotation file
            content: File content as string
            
        Returns:
            Dictionary with format detection results
        """
        result = {
            'format': 'unknown',
            'valid': False,
            'yolo_compatible': False,
            'details': {}
        }
        
        file_ext = Path(filename).suffix.lower()
        
        try:
            # JSON format detection
            if file_ext == '.json' or content.strip().startswith('{'):
                json_data = json.loads(content)
                result['format'] = 'json'
                result['valid'] = True
                result['details'] = self._analyze_json_annotations(json_data)
                
            # YAML format detection
            elif file_ext in ['.yaml', '.yml'] or 'classes:' in content:
                yaml_data = yaml.safe_load(content)
                result['format'] = 'yaml'
                result['valid'] = True
                result['details'] = self._analyze_yaml_annotations(yaml_data)
                
            # XML format detection (PASCAL VOC, etc.)
            elif file_ext == '.xml' or content.strip().startswith('<'):
                root = ET.fromstring(content)
                result['format'] = 'xml'
                result['valid'] = True
                result['details'] = self._analyze_xml_annotations(root)
                
            # YOLO format detection (plain text)
            elif file_ext == '.txt':
                result['format'] = 'yolo_txt'
                result['valid'] = True
                result['yolo_compatible'] = True
                result['details'] = self._analyze_yolo_txt_annotations(content)
                
            else:
                result['details']['error'] = f"Unrecognized format for file: {filename}"
                
        except Exception as e:
            result['details']['error'] = f"Error parsing file: {str(e)}"
        
        return result
    
    def _analyze_json_annotations(self, json_data: Dict) -> Dict[str, Any]:
        """Analyze JSON annotation format (COCO, custom, etc.)"""
        details = {}
        
        # Check for COCO format
        if 'images' in json_data and 'annotations' in json_data and 'categories' in json_data:
            details['format_type'] = 'coco'
            details['num_images'] = len(json_data.get('images', []))
            details['num_annotations'] = len(json_data.get('annotations', []))
            details['num_categories'] = len(json_data.get('categories', []))
        else:
            details['format_type'] = 'custom_json'
            details['keys'] = list(json_data.keys())
        
        return details
    
    def _analyze_yaml_annotations(self, yaml_data: Dict) -> Dict[str, Any]:
        """Analyze YAML annotation format (YOLOv5/v8 dataset config, etc.)"""
        details = {}
        
        if 'train' in yaml_data and 'val' in yaml_data and 'names' in yaml_data:
            details['format_type'] = 'yolo_dataset_config'
            details['num_classes'] = len(yaml_data.get('names', []))
            details['train_path'] = yaml_data.get('train')
            details['val_path'] = yaml_data.get('val')
        else:
            details['format_type'] = 'custom_yaml'
            details['keys'] = list(yaml_data.keys())
        
        return details
    
    def _analyze_xml_annotations(self, root: ET.Element) -> Dict[str, Any]:
        """Analyze XML annotation format (PASCAL VOC, etc.)"""
        details = {}
        
        if root.tag == 'annotation':
            details['format_type'] = 'pascal_voc'
            objects = root.findall('object')
            details['num_objects'] = len(objects)
            details['classes'] = list(set(obj.find('name').text for obj in objects if obj.find('name') is not None))
        else:
            details['format_type'] = 'custom_xml'
            details['root_tag'] = root.tag
        
        return details
    
    def _analyze_yolo_txt_annotations(self, content: str) -> Dict[str, Any]:
        """Analyze YOLO text format annotations"""
        details = {}
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        details['num_annotations'] = len(lines)
        
        # Validate YOLO format (class_id x_center y_center width height)
        valid_lines = 0
        class_ids = set()
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    
                    # Check if coordinates are normalized (0-1 range)
                    if all(0 <= coord <= 1 for coord in coords):
                        valid_lines += 1
                        class_ids.add(class_id)
                except ValueError:
                    pass
        
        details['valid_annotations'] = valid_lines
        details['unique_classes'] = len(class_ids)
        details['class_ids'] = sorted(list(class_ids))
        details['format_valid'] = valid_lines == len(lines)
        
        return details
    
    def assess_data_quality(self, image_files: List[str]) -> Dict[str, Any]:
        """
        Assess overall data quality for ML training.
        
        Args:
            image_files: List of image file keys to assess
            
        Returns:
            Dictionary with quality assessment results
        """
        logger.info(f"Assessing data quality for {len(image_files)} images...")
        
        quality_scores = []
        high_quality_count = 0
        medium_quality_count = 0
        low_quality_count = 0
        
        metrics = {
            'brightness_variance': [],
            'contrast_scores': [],
            'sharpness_scores': [],
            'size_consistency': [],
            'format_consistency': []
        }
        
        formats = []
        sizes = []
        
        for img_key in image_files:
            try:
                img_data = self.s3_access.download_file_to_memory(img_key)
                
                with Image.open(img_data) as img:
                    # Calculate individual quality score
                    quality_score = self._calculate_image_quality_score(img)
                    quality_scores.append(quality_score)
                    
                    # Categorize quality
                    if quality_score >= 7.0:
                        high_quality_count += 1
                    elif quality_score >= 4.0:
                        medium_quality_count += 1
                    else:
                        low_quality_count += 1
                    
                    # Collect metrics for consistency analysis
                    formats.append(img.format)
                    sizes.append(img.size)
                    
            except Exception as e:
                logger.warning(f"Error assessing quality for {img_key}: {str(e)}")
                quality_scores.append(0.0)  # Assign lowest score for errors
                low_quality_count += 1
        
        # Calculate overall metrics
        overall_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # Assess consistency
        format_consistency = len(set(formats)) / len(formats) if formats else 0
        size_consistency = self._calculate_size_consistency(sizes)
        
        return {
            'overall_score': overall_score,
            'quality_scores': quality_scores,
            'high_quality_count': high_quality_count,
            'medium_quality_count': medium_quality_count,
            'low_quality_count': low_quality_count,
            'metrics': {
                'format_consistency': 1.0 - format_consistency,  # Higher is better
                'size_consistency': size_consistency,
                'average_quality': overall_score
            }
        }
    
    def _calculate_image_quality_score(self, img: Image.Image) -> float:
        """
        Calculate a quality score for an individual image.
        
        Args:
            img: PIL Image object
            
        Returns:
            Quality score from 0-10
        """
        score = 10.0  # Start with perfect score
        
        # Size penalty
        width, height = img.size
        min_dim = min(width, height)
        
        if min_dim < 224:
            score -= 3.0  # Significant penalty for very small images
        elif min_dim < 416:
            score -= 1.5  # Moderate penalty for small images
        
        # Aspect ratio penalty
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.0:
            score -= 2.0  # Penalty for extreme aspect ratios
        elif aspect_ratio > 2.0:
            score -= 1.0
        
        # Format penalty
        if img.format not in self.SUPPORTED_FORMATS:
            score -= 1.0
        
        # Color mode penalty
        if img.mode not in self.SUPPORTED_COLOR_MODES:
            score -= 1.0
        
        # Ensure score is within bounds
        return max(0.0, min(10.0, score))
    
    def _calculate_size_consistency(self, sizes: List[Tuple[int, int]]) -> float:
        """
        Calculate size consistency score.
        
        Args:
            sizes: List of (width, height) tuples
            
        Returns:
            Consistency score from 0-1 (higher is more consistent)
        """
        if not sizes:
            return 0.0
        
        # Calculate coefficient of variation for both dimensions
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        width_cv = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1.0
        height_cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 1.0
        
        # Convert to consistency score (lower CV = higher consistency)
        avg_cv = (width_cv + height_cv) / 2
        consistency = max(0.0, 1.0 - avg_cv)
        
        return consistency
    
    def _generate_dataset_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Args:
            validation_results: Results from dataset validation
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Image preprocessing recommendations
        if validation_results['images_need_preprocessing'] > 0:
            recommendations.append(f"Preprocess {validation_results['images_need_preprocessing']} images for YOLOv11 compatibility")
        
        # Input size recommendation
        if validation_results['recommended_input_size']:
            recommendations.append(f"Use input size {validation_results['recommended_input_size']}x{validation_results['recommended_input_size']} for optimal performance")
        
        # Annotation recommendations
        if validation_results['annotation_files_found'] == 0:
            recommendations.append("Create annotations for supervised learning - consider using annotation tools like LabelImg or CVAT")
        
        # Format standardization
        format_dist = validation_results.get('format_distribution', {})
        if len(format_dist) > 2:
            recommendations.append("Consider standardizing image formats to JPEG or PNG for consistency")
        
        return recommendations