"""
Unit tests for YOLO integration utilities.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from duckietown_utils.yolo_utils import (
    YOLOModelLoader,
    YOLOInferenceWrapper,
    create_yolo_inference_system
)


class TestYOLOModelLoader(unittest.TestCase):
    """Test cases for YOLOModelLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_model_path = "test_model.pt"
        self.loader = YOLOModelLoader(
            model_path=self.test_model_path,
            device='cpu',
            confidence_threshold=0.5
        )
    
    def test_init(self):
        """Test YOLOModelLoader initialization."""
        self.assertEqual(self.loader.model_path, self.test_model_path)
        self.assertEqual(self.loader.device, 'cpu')
        self.assertEqual(self.loader.confidence_threshold, 0.5)
        self.assertIsNone(self.loader.model)
        self.assertFalse(self.loader._model_loaded)
    
    def test_determine_device_auto_with_cuda(self):
        """Test device determination when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = self.loader._determine_device('auto')
            self.assertEqual(device, 'cuda')
    
    def test_determine_device_auto_without_cuda(self):
        """Test device determination when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            device = self.loader._determine_device('auto')
            self.assertEqual(device, 'cpu')
    
    def test_determine_device_explicit(self):
        """Test explicit device specification."""
        device = self.loader._determine_device('cpu')
        self.assertEqual(device, 'cpu')
    
    def test_load_model_file_not_found(self):
        """Test model loading when file doesn't exist."""
        result = self.loader.load_model()
        self.assertFalse(result)
        self.assertFalse(self.loader.is_loaded())
    
    @patch('duckietown_utils.yolo_utils.YOLO_AVAILABLE', False)
    def test_load_model_yolo_not_available(self):
        """Test model loading when YOLO is not available."""
        result = self.loader.load_model()
        self.assertFalse(result)
        self.assertFalse(self.loader.is_loaded())
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with patch('duckietown_utils.yolo_utils.YOLO_AVAILABLE', True), \
             patch('os.path.exists', return_value=True), \
             patch('duckietown_utils.yolo_utils.YOLO') as mock_yolo_class:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.to = Mock()
            mock_yolo_class.return_value = mock_model
            
            # Mock successful inference test
            mock_model.return_value = [Mock()]
            
            result = self.loader.load_model()
            
            self.assertTrue(result)
            self.assertTrue(self.loader.is_loaded())
            self.assertIsNotNone(self.loader.model)
    
    def test_load_model_inference_failure(self):
        """Test model loading with inference failure."""
        with patch('duckietown_utils.yolo_utils.YOLO_AVAILABLE', True), \
             patch('os.path.exists', return_value=True), \
             patch('duckietown_utils.yolo_utils.YOLO') as mock_yolo_class:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.to = Mock()
            mock_yolo_class.return_value = mock_model
            
            # Mock inference failure
            mock_model.side_effect = Exception("Inference failed")
            
            result = self.loader.load_model()
            
            self.assertFalse(result)
            self.assertFalse(self.loader.is_loaded())
    
    def test_get_model_not_loaded(self):
        """Test getting model when not loaded."""
        with self.assertRaises(RuntimeError):
            self.loader.get_model()
    
    def test_reload_model(self):
        """Test model reloading."""
        with patch('duckietown_utils.yolo_utils.YOLO_AVAILABLE', True), \
             patch('os.path.exists', return_value=True), \
             patch('duckietown_utils.yolo_utils.YOLO') as mock_yolo_class:
            
            # Setup mocks for successful loading
            mock_model = Mock()
            mock_model.to = Mock()
            mock_yolo_class.return_value = mock_model
            mock_model.return_value = [Mock()]
            
            # First load
            self.assertTrue(self.loader.load_model())
            self.assertTrue(self.loader.is_loaded())
            
            # Reload
            self.assertTrue(self.loader.reload_model())
            self.assertTrue(self.loader.is_loaded())


class TestYOLOInferenceWrapper(unittest.TestCase):
    """Test cases for YOLOInferenceWrapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = Mock(spec=YOLOModelLoader)
        self.wrapper = YOLOInferenceWrapper(
            model_loader=self.mock_loader,
            max_detections=5
        )
    
    def test_init(self):
        """Test YOLOInferenceWrapper initialization."""
        self.assertEqual(self.wrapper.model_loader, self.mock_loader)
        self.assertEqual(self.wrapper.max_detections, 5)
        self.assertEqual(self.wrapper._inference_count, 0)
        self.assertEqual(self.wrapper._total_inference_time, 0.0)
    
    def test_detect_objects_model_not_loaded(self):
        """Test object detection when model is not loaded."""
        self.mock_loader.is_loaded.return_value = False
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.wrapper.detect_objects(image)
        
        expected = {
            'detections': [],
            'detection_count': 0,
            'inference_time': 0.0,
            'frame_shape': None,
            'safety_critical': False
        }
        self.assertEqual(result, expected)
    
    def test_validate_image_none(self):
        """Test image validation with None input."""
        self.assertFalse(self.wrapper._validate_image(None))
    
    def test_validate_image_wrong_dimensions(self):
        """Test image validation with wrong dimensions."""
        # 2D image
        image_2d = np.zeros((480, 640), dtype=np.uint8)
        self.assertFalse(self.wrapper._validate_image(image_2d))
        
        # 4D image
        image_4d = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        self.assertFalse(self.wrapper._validate_image(image_4d))
    
    def test_validate_image_wrong_channels(self):
        """Test image validation with wrong number of channels."""
        # Grayscale image
        image_gray = np.zeros((480, 640, 1), dtype=np.uint8)
        self.assertFalse(self.wrapper._validate_image(image_gray))
        
        # RGBA image
        image_rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        self.assertFalse(self.wrapper._validate_image(image_rgba))
    
    def test_validate_image_valid(self):
        """Test image validation with valid input."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertTrue(self.wrapper._validate_image(image))
    
    def test_process_results_empty(self):
        """Test processing empty results."""
        detections = self.wrapper._process_results([], (480, 640, 3))
        self.assertEqual(detections, [])
    
    def test_process_results_no_boxes(self):
        """Test processing results with no boxes."""
        mock_result = Mock()
        mock_result.boxes = None
        results = [mock_result]
        
        detections = self.wrapper._process_results(results, (480, 640, 3))
        self.assertEqual(detections, [])
    
    def test_process_results_with_detections(self):
        """Test processing results with actual detections."""
        # Create mock result with boxes
        mock_result = Mock()
        mock_boxes = Mock()
        
        # Mock box data
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [100, 100, 200, 200],  # x1, y1, x2, y2
            [300, 150, 400, 250]
        ])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8, 0.6])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 1])
        
        mock_result.boxes = mock_boxes
        mock_result.names = {0: 'person', 1: 'car'}
        
        results = [mock_result]
        detections = self.wrapper._process_results(results, (480, 640, 3))
        
        self.assertEqual(len(detections), 2)
        
        # Check first detection
        det1 = detections[0]
        self.assertEqual(det1['class'], 'person')
        self.assertEqual(det1['confidence'], 0.8)
        self.assertEqual(det1['bbox'], [100, 100, 200, 200])
        self.assertAlmostEqual(det1['center'][0], 150.0)
        self.assertAlmostEqual(det1['center'][1], 150.0)
    
    def test_check_safety_critical_safe(self):
        """Test safety check with safe detections."""
        detections = [
            {'distance': 2.0, 'relative_position': [0.0, 0.5]},
            {'distance': 3.0, 'relative_position': [0.2, 0.3]}
        ]
        self.assertFalse(self.wrapper._check_safety_critical(detections))
    
    def test_check_safety_critical_unsafe_distance(self):
        """Test safety check with unsafe distance."""
        detections = [
            {'distance': 0.5, 'relative_position': [0.0, 0.1]},  # Close and in front
        ]
        self.assertTrue(self.wrapper._check_safety_critical(detections))
    
    def test_check_safety_critical_behind_robot(self):
        """Test safety check with object behind robot."""
        detections = [
            {'distance': 0.5, 'relative_position': [0.0, -0.5]},  # Close but behind
        ]
        self.assertFalse(self.wrapper._check_safety_critical(detections))
    
    def test_performance_stats_no_inferences(self):
        """Test performance stats with no inferences."""
        stats = self.wrapper.get_performance_stats()
        expected = {'avg_inference_time': 0.0, 'total_inferences': 0}
        self.assertEqual(stats, expected)
    
    def test_performance_stats_with_inferences(self):
        """Test performance stats with inferences."""
        # Simulate some inferences
        self.wrapper._update_metrics(0.1)
        self.wrapper._update_metrics(0.2)
        
        stats = self.wrapper.get_performance_stats()
        self.assertEqual(stats['total_inferences'], 2)
        self.assertAlmostEqual(stats['avg_inference_time'], 0.15)
        self.assertAlmostEqual(stats['total_time'], 0.3)
    
    def test_reset_stats(self):
        """Test resetting performance statistics."""
        # Add some stats
        self.wrapper._update_metrics(0.1)
        self.wrapper._update_metrics(0.2)
        
        # Reset
        self.wrapper.reset_stats()
        
        stats = self.wrapper.get_performance_stats()
        expected = {'avg_inference_time': 0.0, 'total_inferences': 0}
        self.assertEqual(stats, expected)


class TestCreateYOLOInferenceSystem(unittest.TestCase):
    """Test cases for create_yolo_inference_system factory function."""
    
    @patch('duckietown_utils.yolo_utils.YOLOModelLoader')
    @patch('duckietown_utils.yolo_utils.YOLOInferenceWrapper')
    def test_create_system_success(self, mock_wrapper_class, mock_loader_class):
        """Test successful system creation."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.load_model.return_value = True
        mock_loader_class.return_value = mock_loader
        
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper
        
        # Create system
        result = create_yolo_inference_system(
            model_path="test.pt",
            device="cpu",
            confidence_threshold=0.6,
            max_detections=8
        )
        
        # Verify calls
        mock_loader_class.assert_called_once_with(
            model_path="test.pt",
            device="cpu",
            confidence_threshold=0.6
        )
        mock_loader.load_model.assert_called_once()
        mock_wrapper_class.assert_called_once_with(
            model_loader=mock_loader,
            max_detections=8
        )
        
        self.assertEqual(result, mock_wrapper)
    
    @patch('duckietown_utils.yolo_utils.YOLOModelLoader')
    def test_create_system_load_failure(self, mock_loader_class):
        """Test system creation with model load failure."""
        # Setup mock to fail loading
        mock_loader = Mock()
        mock_loader.load_model.return_value = False
        mock_loader_class.return_value = mock_loader
        
        # Create system
        result = create_yolo_inference_system(
            model_path="test.pt",
            device="cpu"
        )
        
        self.assertIsNone(result)
    
    @patch('duckietown_utils.yolo_utils.YOLOModelLoader')
    def test_create_system_exception(self, mock_loader_class):
        """Test system creation with exception."""
        # Setup mock to raise exception
        mock_loader_class.side_effect = Exception("Test exception")
        
        # Create system
        result = create_yolo_inference_system(
            model_path="test.pt",
            device="cpu"
        )
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()