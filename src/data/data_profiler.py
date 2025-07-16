"""
Data Profiling Utilities for Drone Imagery Analysis

This module provides comprehensive data profiling capabilities for analyzing
drone imagery characteristics and generating insights for ML pipeline development.

Requirements addressed:
- 1.2: Data profiling and visualization capabilities
- Analysis of drone imagery characteristics
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from typing import Dict, List, Tuple, Any, Optional
import io
import logging
from collections import Counter
import statistics
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneImageryProfiler:
    """
    Comprehensive profiler for drone imagery datasets.
    
    Analyzes image characteristics, quality metrics, and provides insights
    for ML pipeline optimization.
    """
    
    def __init__(self, s3_access):
        """
        Initialize the profiler with S3 access.
        
        Args:
            s3_access: S3DataAccess instance for data retrieval
        """
        self.s3_access = s3_access
        self.profile_cache = {}
        
    def profile_images(self, image_keys: List[str], sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Profile a collection of images and extract comprehensive characteristics.
        
        Args:
            image_keys: List of S3 object keys for images
            sample_size: Optional limit on number of images to analyze
            
        Returns:
            Dictionary containing comprehensive image profile data
        """
        if sample_size and len(image_keys) > sample_size:
            image_keys = np.random.choice(image_keys, sample_size, replace=False).tolist()
        
        logger.info(f"Profiling {len(image_keys)} images...")
        
        profile_data = {
            'widths': [],
            'heights': [],
            'file_sizes': [],
            'color_modes': [],
            'formats': [],
            'aspect_ratios': [],
            'brightness_scores': [],
            'contrast_scores': [],
            'sharpness_scores': [],
            'color_diversity_scores': [],
            'errors': []
        }
        
        processed_count = 0
        
        for idx, key in enumerate(image_keys):
            try:
                # Download image to memory
                img_data = self.s3_access.download_file_to_memory(key)
                
                # Get file size
                file_size = len(img_data.getvalue())
                profile_data['file_sizes'].append(file_size)
                
                # Open and analyze image
                img_data.seek(0)
                with Image.open(img_data) as img:
                    # Basic properties
                    width, height = img.size
                    profile_data['widths'].append(width)
                    profile_data['heights'].append(height)
                    profile_data['aspect_ratios'].append(width / height)
                    profile_data['color_modes'].append(img.mode)
                    profile_data['formats'].append(img.format or 'Unknown')
                    
                    # Quality metrics
                    quality_metrics = self._analyze_image_quality(img)
                    profile_data['brightness_scores'].append(quality_metrics['brightness'])
                    profile_data['contrast_scores'].append(quality_metrics['contrast'])
                    profile_data['sharpness_scores'].append(quality_metrics['sharpness'])
                    profile_data['color_diversity_scores'].append(quality_metrics['color_diversity'])
                
                processed_count += 1
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(image_keys)} images")
                    
            except Exception as e:
                error_msg = f"Error processing {key}: {str(e)}"
                profile_data['errors'].append(error_msg)
                logger.warning(error_msg)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(profile_data)
        
        logger.info(f"Successfully profiled {processed_count} images with {len(profile_data['errors'])} errors")
        
        return {**profile_data, **summary}
    
    def _analyze_image_quality(self, img: Image.Image) -> Dict[str, float]:
        """
        Analyze image quality metrics.
        
        Args:
            img: PIL Image object
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to RGB if necessary for consistent analysis
        if img.mode != 'RGB':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        
        # Calculate brightness (average luminance)
        stat = ImageStat.Stat(img_rgb)
        brightness = sum(stat.mean) / len(stat.mean) / 255.0
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = sum(stat.stddev) / len(stat.stddev) / 255.0
        
        # Estimate sharpness using edge detection approximation
        sharpness = self._estimate_sharpness(img_rgb)
        
        # Calculate color diversity (histogram spread)
        color_diversity = self._calculate_color_diversity(img_rgb)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'color_diversity': color_diversity
        }
    
    def _estimate_sharpness(self, img: Image.Image) -> float:
        """
        Estimate image sharpness using Laplacian variance method.
        
        Args:
            img: PIL Image object
            
        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            # Convert to grayscale for edge detection
            gray = img.convert('L')
            
            # Resize for faster processing if image is very large
            if gray.size[0] * gray.size[1] > 1000000:  # 1MP threshold
                gray = gray.resize((800, 600), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(gray)
            
            # Apply Laplacian kernel for edge detection
            laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            
            # Convolve with kernel (simplified edge detection)
            edges = np.abs(np.convolve(img_array.flatten(), laplacian_kernel.flatten(), mode='valid'))
            
            # Return variance of edge responses (normalized)
            sharpness = np.var(edges) / 10000.0  # Normalize to 0-1 range approximately
            
            return min(sharpness, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating sharpness: {str(e)}")
            return 0.5  # Default moderate sharpness
    
    def _calculate_color_diversity(self, img: Image.Image) -> float:
        """
        Calculate color diversity using histogram analysis.
        
        Args:
            img: PIL Image object
            
        Returns:
            Color diversity score (0-1, higher = more diverse)
        """
        try:
            # Get histogram for each channel
            histograms = []
            for channel in range(3):  # RGB channels
                hist = img.histogram()[channel * 256:(channel + 1) * 256]
                histograms.append(hist)
            
            # Calculate entropy-like measure for color diversity
            total_pixels = img.size[0] * img.size[1]
            diversity_scores = []
            
            for hist in histograms:
                # Normalize histogram
                normalized_hist = [count / total_pixels for count in hist if count > 0]
                
                # Calculate entropy
                entropy = -sum(p * np.log2(p) for p in normalized_hist if p > 0)
                diversity_scores.append(entropy)
            
            # Average entropy across channels, normalized to 0-1
            avg_entropy = np.mean(diversity_scores)
            normalized_diversity = min(avg_entropy / 8.0, 1.0)  # 8 is theoretical max for 8-bit
            
            return normalized_diversity
            
        except Exception as e:
            logger.warning(f"Error calculating color diversity: {str(e)}")
            return 0.5  # Default moderate diversity
    
    def _calculate_summary_statistics(self, profile_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Calculate summary statistics from profile data.
        
        Args:
            profile_data: Dictionary containing profile data lists
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Image dimensions
        if profile_data['widths']:
            summary.update({
                'min_width': min(profile_data['widths']),
                'max_width': max(profile_data['widths']),
                'avg_width': statistics.mean(profile_data['widths']),
                'median_width': statistics.median(profile_data['widths']),
                'min_height': min(profile_data['heights']),
                'max_height': max(profile_data['heights']),
                'avg_height': statistics.mean(profile_data['heights']),
                'median_height': statistics.median(profile_data['heights'])
            })
        
        # File sizes
        if profile_data['file_sizes']:
            avg_size_mb = statistics.mean(profile_data['file_sizes']) / (1024 * 1024)
            summary.update({
                'avg_file_size': avg_size_mb,
                'min_file_size_mb': min(profile_data['file_sizes']) / (1024 * 1024),
                'max_file_size_mb': max(profile_data['file_sizes']) / (1024 * 1024)
            })
        
        # Aspect ratios
        if profile_data['aspect_ratios']:
            summary.update({
                'avg_aspect_ratio': statistics.mean(profile_data['aspect_ratios']),
                'aspect_ratio_std': statistics.stdev(profile_data['aspect_ratios']) if len(profile_data['aspect_ratios']) > 1 else 0
            })
        
        # Quality metrics
        quality_metrics = ['brightness_scores', 'contrast_scores', 'sharpness_scores', 'color_diversity_scores']
        for metric in quality_metrics:
            if profile_data[metric]:
                metric_name = metric.replace('_scores', '')
                summary.update({
                    f'avg_{metric_name}': statistics.mean(profile_data[metric]),
                    f'min_{metric_name}': min(profile_data[metric]),
                    f'max_{metric_name}': max(profile_data[metric])
                })
        
        # Categorical data
        summary['color_modes'] = list(set(profile_data['color_modes']))
        summary['formats'] = list(set(profile_data['formats']))
        
        # Processing statistics
        summary.update({
            'total_images_processed': len(profile_data['widths']),
            'processing_errors': len(profile_data['errors']),
            'success_rate': len(profile_data['widths']) / (len(profile_data['widths']) + len(profile_data['errors'])) if (len(profile_data['widths']) + len(profile_data['errors'])) > 0 else 0
        })
        
        return summary
    
    def generate_profile_report(self, profile_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable profile report.
        
        Args:
            profile_data: Profile data from profile_images method
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("DRONE IMAGERY DATASET PROFILE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset overview
        report.append("📊 DATASET OVERVIEW")
        report.append("-" * 30)
        report.append(f"Total images processed: {profile_data.get('total_images_processed', 0)}")
        report.append(f"Processing errors: {profile_data.get('processing_errors', 0)}")
        report.append(f"Success rate: {profile_data.get('success_rate', 0):.1%}")
        report.append("")
        
        # Image characteristics
        report.append("🖼️ IMAGE CHARACTERISTICS")
        report.append("-" * 30)
        if 'avg_width' in profile_data:
            report.append(f"Resolution range: {profile_data['min_width']}x{profile_data['min_height']} to {profile_data['max_width']}x{profile_data['max_height']}")
            report.append(f"Average resolution: {profile_data['avg_width']:.0f}x{profile_data['avg_height']:.0f}")
            report.append(f"Average aspect ratio: {profile_data.get('avg_aspect_ratio', 0):.2f}")
        
        if 'avg_file_size' in profile_data:
            report.append(f"Average file size: {profile_data['avg_file_size']:.1f} MB")
            report.append(f"File size range: {profile_data['min_file_size_mb']:.1f} - {profile_data['max_file_size_mb']:.1f} MB")
        
        report.append(f"Color modes: {', '.join(profile_data.get('color_modes', []))}")
        report.append(f"File formats: {', '.join(profile_data.get('formats', []))}")
        report.append("")
        
        # Quality metrics
        report.append("✅ QUALITY METRICS")
        report.append("-" * 30)
        quality_metrics = [
            ('brightness', 'Brightness'),
            ('contrast', 'Contrast'),
            ('sharpness', 'Sharpness'),
            ('color_diversity', 'Color Diversity')
        ]
        
        for metric, label in quality_metrics:
            avg_key = f'avg_{metric}'
            if avg_key in profile_data:
                report.append(f"{label}: {profile_data[avg_key]:.3f} (range: {profile_data[f'min_{metric}']:.3f} - {profile_data[f'max_{metric}']:.3f})")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_recommendations(self, profile_data: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on profile analysis.
        
        Args:
            profile_data: Profile data from profile_images method
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Resolution recommendations
        if 'avg_width' in profile_data:
            avg_width = profile_data['avg_width']
            avg_height = profile_data['avg_height']
            
            if avg_width < 640 or avg_height < 640:
                recommendations.append("Consider upscaling images to at least 640x640 for optimal YOLOv11 performance")
            
            if profile_data.get('aspect_ratio_std', 0) > 0.5:
                recommendations.append("High aspect ratio variation detected - consider standardizing image dimensions")
        
        # Quality recommendations
        if 'avg_brightness' in profile_data:
            if profile_data['avg_brightness'] < 0.3:
                recommendations.append("Images appear dark - consider brightness enhancement preprocessing")
            elif profile_data['avg_brightness'] > 0.8:
                recommendations.append("Images appear overexposed - consider brightness normalization")
        
        if 'avg_contrast' in profile_data:
            if profile_data['avg_contrast'] < 0.2:
                recommendations.append("Low contrast detected - consider contrast enhancement techniques")
        
        if 'avg_sharpness' in profile_data:
            if profile_data['avg_sharpness'] < 0.3:
                recommendations.append("Images may be blurry - consider sharpening filters or quality filtering")
        
        # File format recommendations
        formats = profile_data.get('formats', [])
        if 'JPEG' not in formats and 'PNG' not in formats:
            recommendations.append("Consider converting images to JPEG or PNG format for better compatibility")
        
        # Processing recommendations
        error_rate = profile_data.get('processing_errors', 0) / max(profile_data.get('total_images_processed', 1), 1)
        if error_rate > 0.1:
            recommendations.append("High error rate detected - review data quality and file integrity")
        
        if not recommendations:
            recommendations.append("Dataset appears to be in good condition for ML training")
        
        return recommendations