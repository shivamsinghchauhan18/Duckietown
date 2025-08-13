"""
Unit tests for YOLO Object Detection Wrapper.

This module contains comprehensive tests for the YOLOObjectDetectionWrapper class,
including functionality tests, error handling, and integration scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gym
from gym import spaces

from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper


class MockEnvironment(gym.Env):
    """Mock environment for testing."""
    
    def __init__(self):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
    
    def reset(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def step(self, action):
        obs = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs, 0.0, False, {}
    
    def render(self, mode='human'):
        pass


class TestYOLOObjectDetectionWrapper(unittest.TestCase):
    """Test cases for YOLOObjectDetectionWrapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()
        
        # Mock YOLO system creation to avoid actual model loading
        self.yolo_system_patcher = patch(
            'duckietown_utils.wrappers.yolo_detection_wrapper.create_yolo_inference_system'
        )
        self.mock_create_yolo = self.yolo_system_patcher.start()
        
        # Create mock YOLO system
        self.mock_yolo_system = Mock()
        self.mock_create_yolo.return_value = self.mock_yolo_system
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.yolo_system_patcher.stop()
    
    def test_init_default_parameters(self):
        """Test wrapper initialization with default parameters."""
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        self.assertEqual(wrapper.model_path, "yolov5s.pt")
        self.assertEqual(wrapper.confidence_threshold, 0.5)
        self.assertEqual(wrapper.device, 'auto')
        self.assertEqual(wrapper.max_detections, 10)
        self.assertEqual(wrapper.safety_distance_threshold, 1.0)
        self.assertTrue(wrapper.include_image_in_obs)
        self.assertFalse(wrapper.flatten_detections)
        self.assertEqual(wrapper.detection_timeout, 0.1)
    
    def test_init_custom_parameters(self):
        """Test wrapper initialization with custom parameters."""
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            model_path="custom_model.pt",
            confidence_threshold=0.7,
            device='cpu',
            max_detections=5,
            safety_distance_threshold=0.8,
            include_image_in_obs=False,
            flatten_detections=True,
            detection_timeout=0.2
        )
        
        self.assertEqual(wrapper.model_path, "custom_model.pt")
        self.assertEqual(wrapper.confidence_threshold, 0.7)
        self.assertEqual(wrapper.device, 'cpu')
        self.assertEqual(wrapper.max_detections, 5)
        self.assertEqual(wrapper.safety_distance_threshold, 0.8)
        self.assertFalse(wrapper.include_image_in_obs)
        self.assertTrue(wrapper.flatten_detections)
        self.assertEqual(wrapper.detection_timeout, 0.2)
    
    def test_yolo_system_initialization_success(self):
        """Test successful YOLO system initialization."""
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        self.mock_create_yolo.assert_called_once_with(
            model_path="yolov5s.pt",
            device='auto',
            confidence_threshold=0.5,
            max_detections=10
        )
        self.assertTrue(wrapper.is_detection_enabled())
        self.assertEqual(wrapper.yolo_system, self.mock_yolo_system)
    
    def test_yolo_system_initialization_failure(self):
        """Test YOLO system initialization failure."""
        self.mock_create_yolo.return_value = None
        
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        self.assertFalse(wrapper.is_detection_enabled())
        self.assertIsNone(wrapper.yolo_system)
    
    def test_yolo_system_initialization_exception(self):
        """Test YOLO system initialization with exception."""
        self.mock_create_yolo.side_effect = Exception("Test exception")
        
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        self.assertFalse(wrapper.is_detection_enabled())
        self.assertIsNone(wrapper.yolo_system)
    
    def test_dict_observation_space_setup(self):
        """Test dictionary observation space setup."""
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=False,
            include_image_in_obs=True
        )
        
        self.assertIsInstance(wrapper.observation_space, spaces.Dict)
        self.assertIn('image', wrapper.observation_space.spaces)
        self.assertIn('detections', wrapper.observation_space.spaces)
        self.assertIn('detection_count', wrapper.observation_space.spaces)
        self.assertIn('safety_critical', wrapper.observation_space.spaces)
        self.assertIn('inference_time', wrapper.observation_space.spaces)
        
        # Check detection array shape
        detection_space = wrapper.observation_space.spaces['detections']
        self.assertEqual(detection_space.shape, (10, 9))  # max_detections=10, 9 features
    
    def test_dict_observation_space_no_image(self):
        """Test dictionary observation space without image."""
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=False,
            include_image_in_obs=False
        )
        
        self.assertIsInstance(wrapper.observation_space, spaces.Dict)
        self.assertNotIn('image', wrapper.observation_space.spaces)
        self.assertIn('detections', wrapper.observation_space.spaces)
    
    def test_flattened_observation_space_setup(self):
        """Test flattened observation space setup."""
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=True,
            include_image_in_obs=True,
            max_detections=5
        )
        
        self.assertIsInstance(wrapper.observation_space, spaces.Box)
        
        # Calculate expected size
        image_size = np.prod(self.mock_env.observation_space.shape)  # 480*640*3
        detection_size = 5 * 9  # max_detections * features_per_detection
        metadata_size = 3  # detection_count, safety_critical, inference_time
        expected_size = image_size + detection_size + metadata_size
        
        self.assertEqual(wrapper.observation_space.shape, (expected_size,))
    
    def test_flattened_observation_space_no_image(self):
        """Test flattened observation space without image."""
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=True,
            include_image_in_obs=False,
            max_detections=5
        )
        
        # Calculate expected size without image
        detection_size = 5 * 9
        metadata_size = 3
        expected_size = detection_size + metadata_size
        
        self.assertEqual(wrapper.observation_space.shape, (expected_size,))
    
    def test_observation_with_detections(self):
        """Test observation processing with successful detections."""
        # Setup mock detection result
        mock_detections = [
            {
                'class': 'person',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'relative_position': [0.1, 0.2],
                'distance': 1.5
            },
            {
                'class': 'car',
                'confidence': 0.6,
                'bbox': [300, 150, 400, 250],
                'relative_position': [-0.1, 0.3],
                'distance': 2.0
            }
        ]
        
        mock_result = {
            'detections': mock_detections,
            'detection_count': 2,
            'inference_time': 0.05,
            'frame_shape': (480, 640, 3),
            'safety_critical': False
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=False
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Verify structure
        self.assertIsInstance(result_obs, dict)
        self.assertIn('image', result_obs)
        self.assertIn('detections', result_obs)
        self.assertIn('detection_count', result_obs)
        self.assertIn('safety_critical', result_obs)
        self.assertIn('inference_time', result_obs)
        
        # Verify content
        np.testing.assert_array_equal(result_obs['image'], input_obs)
        self.assertEqual(result_obs['detection_count'][0], 2)
        self.assertEqual(result_obs['safety_critical'][0], 0)  # False -> 0
        self.assertAlmostEqual(result_obs['inference_time'][0], 0.05)
        
        # Verify detection array
        detections_array = result_obs['detections']
        self.assertEqual(detections_array.shape, (10, 9))  # max_detections=10, 9 features
        
        # Check first detection
        first_detection = detections_array[0]
        self.assertEqual(first_detection[1], 0.8)  # confidence
        self.assertEqual(first_detection[2], 100)  # x1
        self.assertEqual(first_detection[3], 100)  # y1
        self.assertEqual(first_detection[4], 200)  # x2
        self.assertEqual(first_detection[5], 200)  # y2
        self.assertEqual(first_detection[6], 0.1)  # rel_x
        self.assertEqual(first_detection[7], 0.2)  # rel_y
        self.assertEqual(first_detection[8], 1.5)  # distance
    
    def test_observation_no_detections(self):
        """Test observation processing with no detections."""
        mock_result = {
            'detections': [],
            'detection_count': 0,
            'inference_time': 0.03,
            'frame_shape': (480, 640, 3),
            'safety_critical': False
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=False
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        self.assertEqual(result_obs['detection_count'][0], 0)
        self.assertEqual(result_obs['safety_critical'][0], 0)
        
        # Verify detection array is zeros
        detections_array = result_obs['detections']
        np.testing.assert_array_equal(detections_array, np.zeros((10, 9)))
    
    def test_observation_detection_disabled(self):
        """Test observation processing when detection is disabled."""
        self.mock_create_yolo.return_value = None
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=False
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Should return empty detection result
        self.assertEqual(result_obs['detection_count'][0], 0)
        self.assertEqual(result_obs['safety_critical'][0], 0)
        self.assertEqual(result_obs['inference_time'][0], 0.0)
    
    def test_observation_detection_exception(self):
        """Test observation processing with detection exception."""
        self.mock_yolo_system.detect_objects.side_effect = Exception("Detection failed")
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=False
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Should return empty detection result
        self.assertEqual(result_obs['detection_count'][0], 0)
        self.assertEqual(result_obs['safety_critical'][0], 0)
        self.assertEqual(result_obs['inference_time'][0], 0.0)
        
        # Check that failed detection was recorded
        stats = wrapper.get_detection_stats()
        self.assertEqual(stats['failed_detections'], 1)
    
    def test_flattened_observation_output(self):
        """Test flattened observation output format."""
        mock_detections = [
            {
                'class': 'person',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'relative_position': [0.1, 0.2],
                'distance': 1.5
            }
        ]
        
        mock_result = {
            'detections': mock_detections,
            'detection_count': 1,
            'inference_time': 0.05,
            'frame_shape': (480, 640, 3),
            'safety_critical': False
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            flatten_detections=True,
            max_detections=2
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Should be a 1D numpy array
        self.assertIsInstance(result_obs, np.ndarray)
        self.assertEqual(len(result_obs.shape), 1)
        
        # Calculate expected size
        image_size = np.prod(input_obs.shape)
        detection_size = 2 * 9  # max_detections * features
        metadata_size = 3
        expected_size = image_size + detection_size + metadata_size
        
        self.assertEqual(result_obs.shape[0], expected_size)
    
    def test_safety_critical_detection(self):
        """Test safety critical detection identification."""
        # Create detection that should be safety critical (close and in front)
        mock_detections = [
            {
                'class': 'person',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'relative_position': [0.0, 0.1],  # In front
                'distance': 0.5  # Close (< safety_distance_threshold=1.0)
            }
        ]
        
        mock_result = {
            'detections': mock_detections,
            'detection_count': 1,
            'inference_time': 0.05,
            'frame_shape': (480, 640, 3),
            'safety_critical': False  # Will be overridden by wrapper
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            safety_distance_threshold=1.0
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Should be marked as safety critical
        self.assertEqual(result_obs['safety_critical'][0], 1)
        
        # Check statistics
        stats = wrapper.get_detection_stats()
        self.assertEqual(stats['safety_critical_detections'], 1)
    
    def test_safety_not_critical_behind_robot(self):
        """Test that objects behind robot are not safety critical."""
        mock_detections = [
            {
                'class': 'person',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'relative_position': [0.0, -0.5],  # Behind robot
                'distance': 0.3  # Very close but behind
            }
        ]
        
        mock_result = {
            'detections': mock_detections,
            'detection_count': 1,
            'inference_time': 0.05,
            'frame_shape': (480, 640, 3),
            'safety_critical': False
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Should NOT be marked as safety critical
        self.assertEqual(result_obs['safety_critical'][0], 0)
    
    def test_detection_timeout_handling(self):
        """Test detection timeout handling."""
        # Mock slow detection that exceeds timeout
        def slow_detection(image):
            import time
            time.sleep(0.2)  # Longer than default timeout of 0.1s
            return {'detections': [], 'detection_count': 0, 'inference_time': 0.2}
        
        self.mock_yolo_system.detect_objects.side_effect = slow_detection
        
        wrapper = YOLOObjectDetectionWrapper(
            self.mock_env,
            detection_timeout=0.1
        )
        
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_obs = wrapper.observation(input_obs)
        
        # Should return empty result due to timeout
        self.assertEqual(result_obs['detection_count'][0], 0)
        
        # Check timeout was recorded
        stats = wrapper.get_detection_stats()
        self.assertEqual(stats['timeout_detections'], 1)
    
    def test_detection_statistics(self):
        """Test detection statistics tracking."""
        mock_result = {
            'detections': [],
            'detection_count': 0,
            'inference_time': 0.05,
            'frame_shape': (480, 640, 3),
            'safety_critical': False
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        self.mock_yolo_system.get_performance_stats.return_value = {
            'avg_inference_time': 0.04,
            'total_inferences': 5
        }
        
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        # Perform several detections
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            wrapper.observation(input_obs)
        
        stats = wrapper.get_detection_stats()
        
        self.assertEqual(stats['total_detections'], 3)
        self.assertEqual(stats['failed_detections'], 0)
        self.assertEqual(stats['timeout_detections'], 0)
        self.assertEqual(stats['safety_critical_detections'], 0)
        self.assertEqual(stats['success_rate'], 1.0)
        self.assertEqual(stats['safety_critical_rate'], 0.0)
        self.assertEqual(stats['avg_inference_time'], 0.04)
        self.assertEqual(stats['total_inferences'], 5)
    
    def test_reset_detection_statistics(self):
        """Test resetting detection statistics."""
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        # Add some stats
        wrapper._detection_stats['total_detections'] = 5
        wrapper._detection_stats['failed_detections'] = 1
        
        # Reset
        wrapper.reset_detection_stats()
        
        stats = wrapper.get_detection_stats()
        self.assertEqual(stats['total_detections'], 0)
        self.assertEqual(stats['failed_detections'], 0)
        
        # Verify YOLO system reset was called
        self.mock_yolo_system.reset_stats.assert_called_once()
    
    def test_get_last_detection_result(self):
        """Test getting last detection result."""
        mock_result = {
            'detections': [{'class': 'test', 'confidence': 0.9}],
            'detection_count': 1,
            'inference_time': 0.05
        }
        
        self.mock_yolo_system.detect_objects.return_value = mock_result
        
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        # Initially should be None
        self.assertIsNone(wrapper.get_last_detection_result())
        
        # After detection, should return result
        input_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        wrapper.observation(input_obs)
        
        last_result = wrapper.get_last_detection_result()
        self.assertEqual(last_result['detection_count'], 1)
        self.assertEqual(last_result['detections'][0]['class'], 'test')
    
    def test_reload_yolo_model(self):
        """Test YOLO model reloading."""
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        # Mock successful reload
        with patch.object(wrapper, '_initialize_yolo_system', return_value=True) as mock_init:
            result = wrapper.reload_yolo_model()
            
            self.assertTrue(result)
            mock_init.assert_called_once()
    
    def test_reset_environment(self):
        """Test environment reset functionality."""
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        # Set some detection history
        wrapper._last_detection_result = {'test': 'data'}
        
        # Reset environment
        obs = wrapper.reset()
        
        # Detection history should be cleared
        self.assertIsNone(wrapper._last_detection_result)
        
        # Should return processed observation
        self.assertIsInstance(obs, dict)  # Default is dict observation


class TestYOLODetectionWrapperIntegration(unittest.TestCase):
    """Integration tests for YOLO Detection Wrapper."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_env = MockEnvironment()
    
    @patch('duckietown_utils.wrappers.yolo_detection_wrapper.create_yolo_inference_system')
    def test_end_to_end_episode(self, mock_create_yolo):
        """Test complete episode with wrapper."""
        # Setup mock YOLO system
        mock_yolo_system = Mock()
        mock_create_yolo.return_value = mock_yolo_system
        
        mock_result = {
            'detections': [],
            'detection_count': 0,
            'inference_time': 0.05,
            'frame_shape': (480, 640, 3),
            'safety_critical': False
        }
        mock_yolo_system.detect_objects.return_value = mock_result
        mock_yolo_system.get_performance_stats.return_value = {
            'avg_inference_time': 0.05,
            'total_inferences': 0
        }
        
        # Create wrapper
        wrapper = YOLOObjectDetectionWrapper(self.mock_env)
        
        # Run episode
        obs = wrapper.reset()
        self.assertIsInstance(obs, dict)
        
        for _ in range(5):
            action = wrapper.action_space.sample()
            obs, reward, done, info = wrapper.step(action)
            self.assertIsInstance(obs, dict)
            
            if done:
                break
        
        # Check statistics
        stats = wrapper.get_detection_stats()
        self.assertGreaterEqual(stats['total_detections'], 5)  # At least 5 detections (reset + steps)


if __name__ == '__main__':
    unittest.main()