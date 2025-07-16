#!/usr/bin/env python3
"""
Unit tests for YOLOv11 Trainer

Tests the training functionality, configuration management, and evaluation metrics.
"""

import unittest
import tempfile
import shutil
import os
import json
import yaml
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.yolov11_trainer import (
    YOLOv11Trainer, 
    TrainingConfig, 
    create_training_config_from_dict,
    save_training_config,
    load_training_config
)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig class"""
    
    def test_default_config_creation(self):
        """Test creating config with default values"""
        config = TrainingConfig()
        
        self.assertEqual(config.model_variant, "yolov11n")
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.image_size, 640)
        self.assertEqual(config.num_classes, 3)
        self.assertEqual(config.class_names, ["vehicle", "person", "building"])
    
    def test_custom_config_creation(self):
        """Test creating config with custom values"""
        config = TrainingConfig(
            model_variant="yolov11s",
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            class_names=["car", "truck", "bus"]
        )
        
        self.assertEqual(config.model_variant, "yolov11s")
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.class_names, ["car", "truck", "bus"])
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "model_variant": "yolov11m",
            "epochs": 200,
            "batch_size": 8,
            "class_names": ["object1", "object2"]
        }
        
        config = create_training_config_from_dict(config_dict)
        
        self.assertEqual(config.model_variant, "yolov11m")
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.class_names, ["object1", "object2"])


class TestConfigSaveLoad(unittest.TestCase):
    """Test configuration save and load functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_variant="yolov11s",
            epochs=50,
            batch_size=32,
            class_names=["test1", "test2", "test3"]
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_json_config(self):
        """Test saving and loading JSON configuration"""
        config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Save config
        save_training_config(self.config, config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Load config
        loaded_config = load_training_config(config_path)
        
        self.assertEqual(loaded_config.model_variant, self.config.model_variant)
        self.assertEqual(loaded_config.epochs, self.config.epochs)
        self.assertEqual(loaded_config.batch_size, self.config.batch_size)
        self.assertEqual(loaded_config.class_names, self.config.class_names)
    
    def test_save_load_yaml_config(self):
        """Test saving and loading YAML configuration"""
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Save config
        save_training_config(self.config, config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Load config
        loaded_config = load_training_config(config_path)
        
        self.assertEqual(loaded_config.model_variant, self.config.model_variant)
        self.assertEqual(loaded_config.epochs, self.config.epochs)
        self.assertEqual(loaded_config.batch_size, self.config.batch_size)
        self.assertEqual(loaded_config.class_names, self.config.class_names)
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats"""
        config_path = os.path.join(self.temp_dir, "test_config.txt")
        
        with self.assertRaises(ValueError):
            save_training_config(self.config, config_path)


class TestYOLOv11Trainer(unittest.TestCase):
    """Test YOLOv11Trainer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_variant="yolov11n",
            epochs=1,  # Small number for testing
            batch_size=2,
            project_path=self.temp_dir,
            name="test_training"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = YOLOv11Trainer(self.config)
        
        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.best_fitness, 0.0)
        self.assertEqual(trainer.start_epoch, 0)
        self.assertIsNotNone(trainer.device)
    
    @patch('torch.cuda.is_available')
    def test_device_setup_gpu(self, mock_cuda_available):
        """Test device setup with GPU available"""
        mock_cuda_available.return_value = True
        
        trainer = YOLOv11Trainer(self.config)
        
        self.assertEqual(trainer.device, "cuda")
    
    @patch('torch.cuda.is_available')
    def test_device_setup_cpu(self, mock_cuda_available):
        """Test device setup with CPU only"""
        mock_cuda_available.return_value = False
        
        trainer = YOLOv11Trainer(self.config)
        
        self.assertEqual(trainer.device, "cpu")
    
    def test_prepare_dataset_config(self):
        """Test dataset configuration preparation"""
        trainer = YOLOv11Trainer(self.config)
        
        # Create mock dataset structure
        dataset_dir = os.path.join(self.temp_dir, "dataset")
        os.makedirs(os.path.join(dataset_dir, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "val", "images"), exist_ok=True)
        
        config_path = trainer.prepare_dataset_config(dataset_dir)
        
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify config
        with open(config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        self.assertEqual(dataset_config['path'], dataset_dir)
        self.assertEqual(dataset_config['nc'], self.config.num_classes)
        self.assertEqual(dataset_config['names'], self.config.class_names)
    
    @patch('models.yolov11_trainer.YOLO')
    def test_load_model_pretrained(self, mock_yolo):
        """Test loading pretrained model"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        trainer = YOLOv11Trainer(self.config)
        model = trainer.load_model()
        
        self.assertEqual(model, mock_model)
        mock_yolo.assert_called_once_with("yolov11n.pt")
    
    @patch('models.yolov11_trainer.YOLO')
    def test_load_model_custom(self, mock_yolo):
        """Test loading custom model"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Create a mock model file
        model_path = os.path.join(self.temp_dir, "custom_model.pt")
        with open(model_path, 'w') as f:
            f.write("mock model")
        
        trainer = YOLOv11Trainer(self.config)
        model = trainer.load_model(model_path)
        
        self.assertEqual(model, mock_model)
        mock_yolo.assert_called_once_with(model_path)
    
    def test_training_args_preparation(self):
        """Test training arguments preparation"""
        trainer = YOLOv11Trainer(self.config)
        dataset_config_path = "test_dataset.yaml"
        
        args = trainer._prepare_training_args(dataset_config_path)
        
        self.assertEqual(args['data'], dataset_config_path)
        self.assertEqual(args['epochs'], self.config.epochs)
        self.assertEqual(args['batch'], self.config.batch_size)
        self.assertEqual(args['imgsz'], self.config.image_size)
        self.assertEqual(args['lr0'], self.config.learning_rate)
        self.assertEqual(args['optimizer'], self.config.optimizer)
    
    def test_checkpoint_creation_and_loading(self):
        """Test checkpoint creation and loading"""
        trainer = YOLOv11Trainer(self.config)
        
        # Mock data for checkpoint
        epoch = 10
        model_state = {"layer1": "weights"}
        optimizer_state = {"lr": 0.01}
        metrics = {"map50": 0.85, "precision": 0.9}
        
        # Create checkpoint
        checkpoint_path = trainer.create_checkpoint(
            epoch, model_state, optimizer_state, metrics, self.temp_dir
        )
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Load checkpoint
        loaded_checkpoint = trainer.load_checkpoint(checkpoint_path)
        
        self.assertEqual(loaded_checkpoint['epoch'], epoch)
        self.assertEqual(loaded_checkpoint['model_state_dict'], model_state)
        self.assertEqual(loaded_checkpoint['optimizer_state_dict'], optimizer_state)
        self.assertEqual(loaded_checkpoint['metrics'], metrics)
        self.assertEqual(trainer.start_epoch, epoch + 1)
    
    def test_extract_evaluation_metrics(self):
        """Test evaluation metrics extraction"""
        trainer = YOLOv11Trainer(self.config)
        
        # Mock results object
        mock_results = Mock()
        mock_results.results_dict = {
            'metrics/mAP50(B)': 0.85,
            'metrics/mAP50-95(B)': 0.72,
            'metrics/precision(B)': 0.88,
            'metrics/recall(B)': 0.82,
            'val/box_loss': 0.15,
            'val/cls_loss': 0.08,
            'val/dfl_loss': 0.12
        }
        
        metrics = trainer._extract_evaluation_metrics(mock_results)
        
        self.assertEqual(metrics['map50'], 0.85)
        self.assertEqual(metrics['map50_95'], 0.72)
        self.assertEqual(metrics['precision'], 0.88)
        self.assertEqual(metrics['recall'], 0.82)
        self.assertEqual(metrics['box_loss'], 0.15)
        self.assertEqual(metrics['cls_loss'], 0.08)
        self.assertEqual(metrics['dfl_loss'], 0.12)
        
        # Check F1 score calculation
        expected_f1 = 2 * (0.88 * 0.82) / (0.88 + 0.82)
        self.assertAlmostEqual(metrics['f1_score'], expected_f1, places=4)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        
        # Create mock dataset structure
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.dataset_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, split, 'labels'), exist_ok=True)
            
            # Create mock image and label files
            for i in range(2):
                # Mock image file
                img_path = os.path.join(self.dataset_dir, split, 'images', f'image_{i}.jpg')
                with open(img_path, 'w') as f:
                    f.write("mock image")
                
                # Mock label file
                label_path = os.path.join(self.dataset_dir, split, 'labels', f'image_{i}.txt')
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.2 0.2\n")  # Mock YOLO annotation
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('models.yolov11_trainer.mlflow')
    @patch('models.yolov11_trainer.YOLO')
    def test_training_workflow(self, mock_yolo, mock_mlflow):
        """Test complete training workflow"""
        # Mock YOLO model
        mock_model = Mock()
        mock_results = Mock()
        mock_results.save_dir = os.path.join(self.temp_dir, "runs")
        mock_results.results_dict = {'metrics/mAP50(B)': 0.85}
        mock_model.train.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        # Mock MLFlow
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Create trainer config
        config = TrainingConfig(
            model_variant="yolov11n",
            epochs=1,
            batch_size=2,
            project_path=self.temp_dir,
            name="test_training"
        )
        
        trainer = YOLOv11Trainer(config)
        
        # Run training
        results = trainer.train(self.dataset_dir)
        
        # Verify training was called
        mock_model.train.assert_called_once()
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('training_config', results)
        self.assertIn('device_used', results)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestTrainingConfig))
    test_suite.addTest(unittest.makeSuite(TestConfigSaveLoad))
    test_suite.addTest(unittest.makeSuite(TestYOLOv11Trainer))
    test_suite.addTest(unittest.makeSuite(TestTrainingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)