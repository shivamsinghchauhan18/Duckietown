"""
Unit tests for Enhanced Observation Wrapper.

This module contains comprehensive tests for the EnhancedObservationWrapper class,
including feature extraction, normalization, observation space compatibility,
and integration with YOLO detection results.
"""

import unittest
import numpy as np
import gym
from gym import spaces
from unittest.mock import Mock, patch, MagicMock

from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper


class MockEnvironment(gym.Env):
    """Mock environment for testing."""
    
    def __init__(self, observation_space_type='image'):
        if observation_space_type == 'image':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(120, 160, 3), dtype=np.uint8
            )
        elif observation_space_type == 'dict':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
                'detections': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 9), dtype=np.float32),
                'detection_count': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
                'safety_critical': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
                'inference_time': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def reset(self):
        if isinstance(self.observation_space, spaces.Dict):
            return {
                'image': np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
                'detections': np.random.randn(10, 9).astype(np.float32),
                'detection_count': np.array([3]),
                'safety_critical': np.array([0]),
                'inference_time': np.array([0.05])
            }
        else:
            return np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    
    def step(self, action):
        obs = self.reset()
        return obs, 0.0, False, {}


class TestEnhancedObservationWrapper(unittest.TestCase):
    """Test cases for EnhancedObservationWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env_image = MockEnvironment('image')
        self.mock_env_dict = MockEnvironment('dict')
    
    def test_initialization_flattened_mode(self):
        """Test wrapper initialization in flattened mode."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_image,
            output_mode='flattened',
            include_detection_features=True,
            include_image_features=True
        )
        
        # Check observation space is Box type
        self.assertIsInstance(wrapper.observation_space, spaces.Box)
        self.assertEqual(len(wrapper.observation_space.shape), 1)  # Should be 1D
        
        # Check configuration
        self.assertEqual(wrapper.output_mode, 'flattened')
        self.assertTrue(wrapper.include_detection_features)
        self.assertTrue(wrapper.include_image_features)
    
    def test_initialization_dict_mode(self):
        """Test wrapper initialization in dictionary mode."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            output_mode='dict',
            include_detection_features=True,
            include_image_features=True
        )
        
        # Check observation space is Dict type
        self.assertIsInstance(wrapper.observation_space, spaces.Dict)
        self.assertIn('safety_features', wrapper.observation_space.spaces)
        
        if wrapper.include_detection_features:
            self.assertIn('detection_features', wrapper.observation_space.spaces)
        
        if wrapper.include_image_features:
            self.assertIn('image', wrapper.observation_space.spaces)
    
    def test_flattened_observation_processing(self):
        """Test observation processing in flattened mode."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            output_mode='flattened',
            include_detection_features=True,
            include_image_features=True,
            max_detections=5
        )
        
        # Create test observation
        test_obs = {
            'image': np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
            'detections': np.random.randn(5, 9).astype(np.float32),
            'detection_count': np.array([3]),
            'safety_critical': np.array([1]),
            'inference_time': np.array([0.05])
        }
        
        # Process observation
        processed_obs = wrapper.observation(test_obs)
        
        # Check output format
        self.assertIsInstance(processed_obs, np.ndarray)
        self.assertEqual(len(processed_obs.shape), 1)  # Should be 1D
        self.assertEqual(processed_obs.dtype, np.float32)
        
        # Check size matches observation space
        self.assertEqual(processed_obs.shape[0], wrapper.observation_space.shape[0])
    
    def test_dict_observation_processing(self):
        """Test observation processing in dictionary mode."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            output_mode='dict',
            include_detection_features=True,
            include_image_features=True,
            max_detections=5
        )
        
        # Create test observation
        test_obs = {
            'image': np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
            'detections': np.random.randn(5, 9).astype(np.float32),
            'detection_count': np.array([3]),
            'safety_critical': np.array([1]),
            'inference_time': np.array([0.05])
        }
        
        # Process observation
        processed_obs = wrapper.observation(test_obs)
        
        # Check output format
        self.assertIsInstance(processed_obs, dict)
        self.assertIn('safety_features', processed_obs)
        
        if wrapper.include_detection_features:
            self.assertIn('detection_features', processed_obs)
            self.assertEqual(processed_obs['detection_features'].shape[0], 5 * 9)  # max_detections * feature_size
        
        if wrapper.include_image_features:
            self.assertIn('image', processed_obs)
    
    def test_image_only_processing(self):
        """Test processing with only image features (no YOLO wrapper)."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_image,
            output_mode='flattened',
            include_detection_features=True,
            include_image_features=True
        )
        
        # Create raw image observation
        test_obs = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        
        # Process observation
        processed_obs = wrapper.observation(test_obs)
        
        # Check output format
        self.assertIsInstance(processed_obs, np.ndarray)
        self.assertEqual(len(processed_obs.shape), 1)
        self.assertEqual(processed_obs.dtype, np.float32)
    
    def test_detection_feature_extraction(self):
        """Test detection feature extraction and processing."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            max_detections=3,
            detection_feature_size=9,
            normalize_features=False
        )
        
        # Create test detections
        test_detections = np.array([
            [1, 0.9, 100, 50, 200, 150, 0.5, 1.0, 2.5],  # Detection 1
            [2, 0.8, 150, 75, 250, 175, -0.3, 0.8, 3.2],  # Detection 2
            [0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0]  # Empty detection
        ], dtype=np.float32)
        
        # Extract features
        features = wrapper._extract_detection_features(test_detections)
        
        # Check feature size
        expected_size = 3 * 9  # max_detections * detection_feature_size
        self.assertEqual(features.shape[0], expected_size)
        self.assertEqual(features.dtype, np.float32)
        
        # Check that non-zero detections are preserved
        self.assertAlmostEqual(features[0], 1.0)  # First class ID
        self.assertAlmostEqual(features[1], 0.9)  # First confidence
    
    def test_safety_feature_extraction(self):
        """Test safety and metadata feature extraction."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            max_detections=5,
            safety_feature_weight=2.0,
            distance_normalization_factor=10.0
        )
        
        # Create test detections with distances
        test_detections = np.array([
            [1, 0.9, 100, 50, 200, 150, 0.5, 1.0, 2.5],  # Distance 2.5m
            [2, 0.8, 150, 75, 250, 175, -0.3, 0.8, 1.2],  # Distance 1.2m
            [3, 0.7, 200, 100, 300, 200, 0.0, 0.5, 4.0],  # Distance 4.0m
        ], dtype=np.float32)
        
        # Extract safety features
        safety_features = wrapper._extract_safety_features(
            test_detections, 3, 1, 0.05
        )
        
        # Check feature size and content
        self.assertEqual(safety_features.shape[0], 5)
        self.assertEqual(safety_features.dtype, np.float32)
        
        # Check normalized detection count
        self.assertAlmostEqual(safety_features[0], 3.0 / 5.0)  # 3 detections / max_detections
        
        # Check safety critical flag (weighted)
        self.assertAlmostEqual(safety_features[1], 1.0 * 2.0)  # safety_critical * weight
        
        # Check inference time (clipped)
        self.assertAlmostEqual(safety_features[2], 0.05)
        
        # Check average distance (normalized)
        expected_avg_distance = (2.5 + 1.2 + 4.0) / 3.0 / 10.0
        self.assertAlmostEqual(safety_features[3], expected_avg_distance, places=3)
        
        # Check closest distance (normalized)
        expected_closest_distance = 1.2 / 10.0
        self.assertAlmostEqual(safety_features[4], expected_closest_distance, places=3)
    
    def test_image_feature_extraction_flatten(self):
        """Test image feature extraction with flatten method."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_image,
            image_feature_method='flatten',
            normalize_features=False
        )
        
        # Create test image
        test_image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        
        # Extract features
        features = wrapper._extract_image_features(test_image)
        
        # Check feature size and type
        expected_size = 120 * 160 * 3
        self.assertEqual(features.shape[0], expected_size)
        self.assertEqual(features.dtype, np.float32)
    
    def test_image_feature_extraction_encode(self):
        """Test image feature extraction with encode method."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_image,
            image_feature_method='encode',
            normalize_features=False
        )
        
        # Create test image
        test_image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        
        # Extract features
        features = wrapper._extract_image_features(test_image)
        
        # Check feature size (should be 512 for encoded features)
        self.assertEqual(features.shape[0], 512)
        self.assertEqual(features.dtype, np.float32)
    
    def test_feature_normalization(self):
        """Test feature normalization methods."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            normalize_features=True,
            feature_scaling_method='minmax'
        )
        
        # Test detection feature normalization
        test_features = np.array([100.0, 0.5, 200.0, 150.0], dtype=np.float32)
        
        # Set up mock statistics
        wrapper._feature_stats['detection_min'] = np.array([0.0, 0.0, 0.0, 0.0])
        wrapper._feature_stats['detection_max'] = np.array([1000.0, 1.0, 640.0, 480.0])
        
        normalized = wrapper._normalize_detection_features(test_features)
        
        # Check normalization
        self.assertAlmostEqual(normalized[0], 100.0 / 1000.0)  # class_id normalization
        self.assertAlmostEqual(normalized[1], 0.5 / 1.0)  # confidence normalization
        self.assertAlmostEqual(normalized[2], 200.0 / 640.0)  # bbox normalization
        self.assertAlmostEqual(normalized[3], 150.0 / 480.0)  # bbox normalization
    
    def test_observation_space_compatibility(self):
        """Test observation space compatibility with PPO requirements."""
        # Test flattened mode (most compatible with PPO)
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            output_mode='flattened'
        )
        
        # Check that observation space is Box type (required for most PPO implementations)
        self.assertIsInstance(wrapper.observation_space, spaces.Box)
        
        # Check that observation space has finite bounds or is properly shaped
        self.assertTrue(hasattr(wrapper.observation_space, 'shape'))
        self.assertEqual(len(wrapper.observation_space.shape), 1)  # Should be 1D for PPO
        
        # Test that actual observations match the space
        test_obs = wrapper.reset()
        self.assertTrue(wrapper.observation_space.contains(test_obs))
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        wrapper = EnhancedObservationWrapper(self.mock_env_dict)
        
        # Reset and get observation
        obs = wrapper.reset()
        
        # Check that observation is processed correctly
        if wrapper.output_mode == 'flattened':
            self.assertIsInstance(obs, np.ndarray)
        else:
            self.assertIsInstance(obs, dict)
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        wrapper = EnhancedObservationWrapper(self.mock_env_dict)
        
        # Process some observations
        for _ in range(5):
            test_obs = self.mock_env_dict.reset()
            wrapper.observation(test_obs)
        
        # Check statistics
        stats = wrapper.get_feature_stats()
        self.assertEqual(stats['total_observations'], 5)
        self.assertGreaterEqual(stats['feature_extraction_time'], 0.0)
        
        # Reset statistics
        wrapper.reset_stats()
        stats = wrapper.get_feature_stats()
        self.assertEqual(stats['total_observations'], 0)
    
    def test_configuration_options(self):
        """Test various configuration options."""
        # Test with detection features only
        wrapper1 = EnhancedObservationWrapper(
            self.mock_env_dict,
            include_detection_features=True,
            include_image_features=False
        )
        
        obs1 = wrapper1.observation(self.mock_env_dict.reset())
        
        # Test with image features only
        wrapper2 = EnhancedObservationWrapper(
            self.mock_env_dict,
            include_detection_features=False,
            include_image_features=True
        )
        
        obs2 = wrapper2.observation(self.mock_env_dict.reset())
        
        # Both should work without errors
        self.assertIsNotNone(obs1)
        self.assertIsNotNone(obs2)
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        wrapper = EnhancedObservationWrapper(self.mock_env_dict)
        
        # Test with None observation
        try:
            result = wrapper.observation(None)
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            # Should be a meaningful error
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
        
        # Test with malformed observation
        malformed_obs = {'image': None, 'detections': None}
        try:
            result = wrapper.observation(malformed_obs)
            # Should handle gracefully
            self.assertIsNotNone(result)
        except Exception as e:
            # Should be a meaningful error
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
    
    def test_observation_info(self):
        """Test observation info functionality."""
        wrapper = EnhancedObservationWrapper(
            self.mock_env_dict,
            output_mode='flattened',
            max_detections=8,
            normalize_features=True
        )
        
        info = wrapper.get_observation_info()
        
        # Check that info contains expected keys
        expected_keys = [
            'output_mode', 'include_detection_features', 'include_image_features',
            'max_detections', 'normalize_features', 'feature_scaling_method'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check specific values
        self.assertEqual(info['output_mode'], 'flattened')
        self.assertEqual(info['max_detections'], 8)
        self.assertTrue(info['normalize_features'])


if __name__ == '__main__':
    unittest.main()