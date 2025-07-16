"""
Unit Tests for YOLOv11 Data Preprocessing Pipeline

This module contains comprehensive unit tests for the YOLOv11 preprocessing
functionality including format conversion, augmentation, and validation.

Requirements addressed:
- Unit tests for all preprocessing functions
- Validation of data format conversion accuracy
- Testing of augmentation pipeline functionality
"""

import unittest
import numpy as np
from PIL import Image
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import io

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.yolo_preprocessor import (
    YOLOv11Preprocessor,
    BoundingBox,
    ImageAnnotation,
    AnnotationFormat,
    create_yolo_preprocessor
)


class TestBoundingBox(unittest.TestCase):
    """Test BoundingBox dataclass functionality"""
    
    def test_bounding_box_creation(self):
        """Test BoundingBox object creation"""
        bbox = BoundingBox(
            x_center=0.5,
            y_center=0.5,
            width=0.2,
            height=0.3,
            class_id=1,
            confidence=0.95
        )
        
        self.assertEqual(bbox.x_center, 0.5)
        self.assertEqual(bbox.y_center, 0.5)
        self.assertEqual(bbox.width, 0.2)
        self.assertEqual(bbox.height, 0.3)
        self.assertEqual(bbox.class_id, 1)
        self.assertEqual(bbox.confidence, 0.95)
    
    def test_bounding_box_default_confidence(self):
        """Test BoundingBox with default confidence"""
        bbox = BoundingBox(
            x_center=0.5,
            y_center=0.5,
            width=0.2,
            height=0.3,
            class_id=1
        )
        
        self.assertEqual(bbox.confidence, 1.0)


class TestImageAnnotation(unittest.TestCase):
    """Test ImageAnnotation dataclass functionality"""
    
    def test_image_annotation_creation(self):
        """Test ImageAnnotation object creation"""
        bbox1 = BoundingBox(0.3, 0.3, 0.2, 0.2, 0)
        bbox2 = BoundingBox(0.7, 0.7, 0.1, 0.1, 1)
        
        annotation = ImageAnnotation(
            image_path="test_image.jpg",
            image_width=640,
            image_height=480,
            bboxes=[bbox1, bbox2],
            metadata={"source": "test"}
        )
        
        self.assertEqual(annotation.image_path, "test_image.jpg")
        self.assertEqual(annotation.image_width, 640)
        self.assertEqual(annotation.image_height, 480)
        self.assertEqual(len(annotation.bboxes), 2)
        self.assertEqual(annotation.metadata["source"], "test")


class TestYOLOv11Preprocessor(unittest.TestCase):
    """Test YOLOv11Preprocessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_s3_access = Mock()
        self.class_names = ["vehicle", "person", "building"]
        self.preprocessor = YOLOv11Preprocessor(
            s3_access=self.mock_s3_access,
            target_size=640,
            class_names=self.class_names
        )
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        self.assertEqual(self.preprocessor.target_size, 640)
        self.assertEqual(self.preprocessor.class_names, self.class_names)
        self.assertEqual(len(self.preprocessor.class_to_id), 3)
        self.assertEqual(self.preprocessor.class_to_id["vehicle"], 0)
        self.assertEqual(self.preprocessor.class_to_id["person"], 1)
        self.assertEqual(self.preprocessor.class_to_id["building"], 2)
    
    def test_create_augmentation_pipeline(self):
        """Test augmentation pipeline creation"""
        pipeline = self.preprocessor._create_augmentation_pipeline()
        self.assertIsNotNone(pipeline)
        # Check that pipeline has transforms
        self.assertTrue(hasattr(pipeline, 'transforms'))
    
    def test_preprocess_image_numpy(self):
        """Test image preprocessing with numpy array input"""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess_image(test_image)
        
        # Check output properties
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertTrue(processed.dtype == np.float32)
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))
    
    def test_preprocess_image_pil(self):
        """Test image preprocessing with PIL Image input"""
        # Create test PIL image
        test_image = Image.new('RGB', (640, 480), color='red')
        
        processed = self.preprocessor.preprocess_image(test_image)
        
        # Check output properties
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertTrue(processed.dtype == np.float32)
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))
    
    def test_resize_with_padding(self):
        """Test image resizing with padding"""
        # Create rectangular test image
        test_image = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        
        resized = self.preprocessor._resize_with_padding(test_image, 640)
        
        # Check output dimensions
        self.assertEqual(resized.shape, (640, 640, 3))
        self.assertTrue(resized.dtype == np.uint8)
    
    def test_convert_from_coco_format(self):
        """Test COCO format conversion"""
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "test1.jpg",
                    "width": 640,
                    "height": 480
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150]  # [x, y, width, height]
                }
            ],
            "categories": [
                {"id": 1, "name": "vehicle"}
            ]
        }
        
        annotations = self.preprocessor._convert_from_coco(coco_data)
        
        self.assertEqual(len(annotations), 1)
        annotation = annotations[0]
        self.assertEqual(annotation.image_path, "test1.jpg")
        self.assertEqual(annotation.image_width, 640)
        self.assertEqual(annotation.image_height, 480)
        self.assertEqual(len(annotation.bboxes), 1)
        
        bbox = annotation.bboxes[0]
        # Check YOLO format conversion
        expected_x_center = (100 + 200/2) / 640  # (x + w/2) / width
        expected_y_center = (100 + 150/2) / 480  # (y + h/2) / height
        expected_width = 200 / 640  # w / width
        expected_height = 150 / 480  # h / height
        
        self.assertAlmostEqual(bbox.x_center, expected_x_center, places=5)
        self.assertAlmostEqual(bbox.y_center, expected_y_center, places=5)
        self.assertAlmostEqual(bbox.width, expected_width, places=5)
        self.assertAlmostEqual(bbox.height, expected_height, places=5)
        self.assertEqual(bbox.class_id, 0)  # "vehicle" maps to class 0
    
    def test_convert_from_yolo_format(self):
        """Test YOLO format validation/conversion"""
        yolo_data = {
            "image1.jpg": "0 0.5 0.5 0.2 0.3\n1 0.8 0.2 0.1 0.1"
        }
        
        annotations = self.preprocessor._convert_from_yolo(yolo_data)
        
        self.assertEqual(len(annotations), 1)
        annotation = annotations[0]
        self.assertEqual(annotation.image_path, "image1.jpg")
        self.assertEqual(len(annotation.bboxes), 2)
        
        # Check first bbox
        bbox1 = annotation.bboxes[0]
        self.assertEqual(bbox1.class_id, 0)
        self.assertEqual(bbox1.x_center, 0.5)
        self.assertEqual(bbox1.y_center, 0.5)
        self.assertEqual(bbox1.width, 0.2)
        self.assertEqual(bbox1.height, 0.3)
        
        # Check second bbox
        bbox2 = annotation.bboxes[1]
        self.assertEqual(bbox2.class_id, 1)
        self.assertEqual(bbox2.x_center, 0.8)
        self.assertEqual(bbox2.y_center, 0.2)
        self.assertEqual(bbox2.width, 0.1)
        self.assertEqual(bbox2.height, 0.1)
    
    def test_validate_annotation_quality(self):
        """Test annotation quality validation"""
        # Create test annotations with various quality issues
        good_bbox = BoundingBox(0.5, 0.5, 0.2, 0.3, 0)
        bad_bbox_coords = BoundingBox(1.5, 0.5, 0.2, 0.3, 0)  # Invalid coordinates
        tiny_bbox = BoundingBox(0.5, 0.5, 0.001, 0.001, 0)  # Very small
        
        annotations = [
            ImageAnnotation("good.jpg", 640, 480, [good_bbox]),
            ImageAnnotation("bad.jpg", 640, 480, [bad_bbox_coords]),
            ImageAnnotation("tiny.jpg", 640, 480, [tiny_bbox]),
            ImageAnnotation("empty.jpg", 640, 480, []),  # No annotations
        ]
        
        results = self.preprocessor.validate_annotation_quality(annotations)
        
        self.assertEqual(results['total_images'], 4)
        self.assertEqual(results['total_bboxes'], 3)
        self.assertGreater(len(results['issues']), 0)  # Should have validation issues
        self.assertIn('summary', results)
        self.assertTrue(0 <= results['summary']['image_validity_rate'] <= 1)
        self.assertTrue(0 <= results['summary']['bbox_validity_rate'] <= 1)
    
    def test_apply_augmentation(self):
        """Test data augmentation application"""
        # Create test image and bboxes
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_bboxes = [
            BoundingBox(0.5, 0.5, 0.2, 0.3, 0),
            BoundingBox(0.8, 0.2, 0.1, 0.1, 1)
        ]
        
        # Test with high probability to ensure augmentation is applied
        aug_image, aug_bboxes = self.preprocessor.apply_augmentation(
            test_image, test_bboxes, augmentation_probability=1.0
        )
        
        # Check that we get valid outputs
        self.assertEqual(aug_image.shape, test_image.shape)
        self.assertIsInstance(aug_bboxes, list)
        # Note: Due to augmentation, bboxes might be filtered out, so we check <= original count
        self.assertLessEqual(len(aug_bboxes), len(test_bboxes))
    
    def test_create_dataset_summary(self):
        """Test dataset summary creation"""
        # Create test annotations
        annotations = [
            ImageAnnotation("img1.jpg", 640, 480, [
                BoundingBox(0.3, 0.3, 0.2, 0.2, 0),
                BoundingBox(0.7, 0.7, 0.1, 0.1, 1)
            ]),
            ImageAnnotation("img2.jpg", 800, 600, [
                BoundingBox(0.5, 0.5, 0.3, 0.4, 0)
            ])
        ]
        
        summary = self.preprocessor.create_dataset_summary(annotations)
        
        # Check summary structure
        self.assertIn('dataset_info', summary)
        self.assertIn('image_statistics', summary)
        self.assertIn('annotation_statistics', summary)
        self.assertIn('quality_metrics', summary)
        
        # Check dataset info
        self.assertEqual(summary['dataset_info']['total_images'], 2)
        self.assertEqual(summary['dataset_info']['total_annotations'], 3)
        self.assertEqual(summary['dataset_info']['num_classes'], 3)
        
        # Check statistics
        self.assertEqual(len(summary['image_statistics']['resolutions']), 2)
        self.assertEqual(len(summary['annotation_statistics']['bbox_areas']), 3)
    
    def test_save_yolo_annotations(self):
        """Test saving annotations in YOLO format"""
        # Create test annotations
        annotations = [
            ImageAnnotation("img1.jpg", 640, 480, [
                BoundingBox(0.5, 0.5, 0.2, 0.3, 0)
            ]),
            ImageAnnotation("img2.jpg", 640, 480, [
                BoundingBox(0.3, 0.7, 0.1, 0.2, 1)
            ])
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = self.preprocessor.save_yolo_annotations(
                annotations, temp_dir, {'train': 0.8, 'val': 0.2}
            )
            
            # Check that files were created
            self.assertIn('train', saved_files)
            self.assertIn('val', saved_files)
            
            # Check dataset config file
            config_file = os.path.join(temp_dir, 'dataset.yaml')
            self.assertTrue(os.path.exists(config_file))
            
            # Check that at least one annotation file was created
            total_files = len(saved_files['train']) + len(saved_files['val'])
            self.assertEqual(total_files, 2)  # Should have 2 annotation files total


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_yolo_preprocessor(self):
        """Test factory function for creating preprocessor"""
        mock_s3_access = Mock()
        class_names = ["car", "truck"]
        
        preprocessor = create_yolo_preprocessor(
            s3_access=mock_s3_access,
            target_size=416,
            class_names=class_names
        )
        
        self.assertIsInstance(preprocessor, YOLOv11Preprocessor)
        self.assertEqual(preprocessor.target_size, 416)
        self.assertEqual(preprocessor.class_names, class_names)


class TestAnnotationFormatConversions(unittest.TestCase):
    """Test various annotation format conversions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_s3_access = Mock()
        self.preprocessor = YOLOv11Preprocessor(
            s3_access=self.mock_s3_access,
            class_names=["vehicle", "person"]
        )
    
    def test_pascal_voc_conversion(self):
        """Test PASCAL VOC XML format conversion"""
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <filename>test.jpg</filename>
            <size>
                <width>640</width>
                <height>480</height>
            </size>
            <object>
                <name>vehicle</name>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>100</ymin>
                    <xmax>300</xmax>
                    <ymax>250</ymax>
                </bndbox>
            </object>
        </annotation>"""
        
        annotations = self.preprocessor._convert_from_pascal_voc([xml_content])
        
        self.assertEqual(len(annotations), 1)
        annotation = annotations[0]
        self.assertEqual(annotation.image_path, "test.jpg")
        self.assertEqual(annotation.image_width, 640)
        self.assertEqual(annotation.image_height, 480)
        self.assertEqual(len(annotation.bboxes), 1)
        
        bbox = annotation.bboxes[0]
        # Check conversion: center = (xmin + xmax) / 2 / width
        expected_x_center = (100 + 300) / 2 / 640
        expected_y_center = (100 + 250) / 2 / 480
        expected_width = (300 - 100) / 640
        expected_height = (250 - 100) / 480
        
        self.assertAlmostEqual(bbox.x_center, expected_x_center, places=5)
        self.assertAlmostEqual(bbox.y_center, expected_y_center, places=5)
        self.assertAlmostEqual(bbox.width, expected_width, places=5)
        self.assertAlmostEqual(bbox.height, expected_height, places=5)
        self.assertEqual(bbox.class_id, 0)  # "vehicle" maps to class 0
    
    def test_custom_json_conversion(self):
        """Test custom JSON format conversion"""
        json_data = {
            "images": [
                {
                    "file_name": "custom.jpg",
                    "width": 800,
                    "height": 600,
                    "annotations": [
                        {
                            "bbox": [0.5, 0.5, 0.2, 0.3],
                            "format": "yolo",
                            "class": "person"
                        }
                    ]
                }
            ]
        }
        
        annotations = self.preprocessor._convert_from_custom_json(json_data)
        
        self.assertEqual(len(annotations), 1)
        annotation = annotations[0]
        self.assertEqual(annotation.image_path, "custom.jpg")
        self.assertEqual(len(annotation.bboxes), 1)
        
        bbox = annotation.bboxes[0]
        self.assertEqual(bbox.x_center, 0.5)
        self.assertEqual(bbox.y_center, 0.5)
        self.assertEqual(bbox.width, 0.2)
        self.assertEqual(bbox.height, 0.3)
        self.assertEqual(bbox.class_id, 1)  # "person" maps to class 1
    
    def test_invalid_yolo_format_handling(self):
        """Test handling of invalid YOLO format data"""
        invalid_yolo_data = {
            "bad_image.jpg": "invalid line format\n2 1.5 0.5 0.2 0.3"  # Invalid coordinates
        }
        
        annotations = self.preprocessor._convert_from_yolo(invalid_yolo_data)
        
        self.assertEqual(len(annotations), 1)
        annotation = annotations[0]
        # Should have filtered out invalid annotations
        self.assertEqual(len(annotation.bboxes), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_s3_access = Mock()
        self.preprocessor = YOLOv11Preprocessor(
            s3_access=self.mock_s3_access,
            class_names=["test_class"]
        )
    
    def test_empty_annotations_list(self):
        """Test handling of empty annotations list"""
        results = self.preprocessor.validate_annotation_quality([])
        
        self.assertEqual(results['total_images'], 0)
        self.assertEqual(results['total_bboxes'], 0)
        self.assertEqual(results['valid_images'], 0)
    
    def test_image_with_no_bboxes(self):
        """Test handling of images with no bounding boxes"""
        annotation = ImageAnnotation("empty.jpg", 640, 480, [])
        
        results = self.preprocessor.validate_annotation_quality([annotation])
        
        self.assertEqual(results['total_images'], 1)
        self.assertEqual(results['total_bboxes'], 0)
        self.assertEqual(results['valid_images'], 1)  # Image itself is valid
    
    def test_augmentation_failure_handling(self):
        """Test handling of augmentation failures"""
        # Create invalid image that might cause augmentation to fail
        invalid_image = np.array([])  # Empty array
        test_bboxes = [BoundingBox(0.5, 0.5, 0.2, 0.3, 0)]
        
        # Should return original data on failure
        result_image, result_bboxes = self.preprocessor.apply_augmentation(
            invalid_image, test_bboxes, augmentation_probability=1.0
        )
        
        # Should return original inputs when augmentation fails
        np.testing.assert_array_equal(result_image, invalid_image)
        self.assertEqual(result_bboxes, test_bboxes)
    
    def test_grayscale_image_preprocessing(self):
        """Test preprocessing of grayscale images"""
        # Create grayscale test image
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess_image(gray_image)
        
        # Should be converted to RGB
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertTrue(processed.dtype == np.float32)
    
    def test_rgba_image_preprocessing(self):
        """Test preprocessing of RGBA images"""
        # Create RGBA test image
        rgba_image = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess_image(rgba_image)
        
        # Should be converted to RGB
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertTrue(processed.dtype == np.float32)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)