"""
Unit tests for ObjectAvoidanceActionWrapper.

This module provides comprehensive tests for the object avoidance action wrapper,
including avoidance calculations, safety constraints, and integration scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gym
from gym import spaces

from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper


class MockEnvironment(gym.Env):
    """Mock environment for testing the object avoidance wrapper."""
    
    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'detections': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 9), dtype=np.float32),
            'detection_count': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'safety_critical': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'inference_time': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self._last_observation = None
    
    def step(self, action):
        return self._last_observation, 0.0, False, {}
    
    def reset(self):
        self._last_observation = {
            'image': np.zeros((64, 64, 3), dtype=np.uint8),
            'detections': np.zeros((10, 9), dtype=np.float32),
            'detection_count': np.array([0], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32),
            'inference_time': np.array([0.0], dtype=np.float32)
        }
        return self._last_observation


class TestObjectAvoidanceActionWrapper(unittest.TestCase):
    """Test cases for ObjectAvoidanceActionWrapper."""
    
    def setUp(self):
        """Set up test environment and wrapper."""
        self.mock_env = MockEnvironment()
        self.wrapper = ObjectAvoidanceActionWrapper(
            self.mock_env,
            safety_distance=0.5,
            min_clearance=0.2,
            avoidance_strength=1.0,
            debug_logging=False
        )
    
    def test_initialization(self):
        """Test wrapper initialization with valid parameters."""
        self.assertEqual(self.wrapper.safety_distance, 0.5)
        self.assertEqual(self.wrapper.min_clearance, 0.2)
        self.assertEqual(self.wrapper.avoidance_strength, 1.0)
        self.assertFalse(self.wrapper.avoidance_active)
        self.assertFalse(self.wrapper.emergency_brake_active)
    
    def test_initialization_invalid_parameters(self):
        """Test wrapper initialization with invalid parameters."""
        # Test invalid avoidance strength
        with self.assertRaises(ValueError):
            ObjectAvoidanceActionWrapper(self.mock_env, avoidance_strength=3.0)
        
        # Test invalid max avoidance action
        with self.assertRaises(ValueError):
            ObjectAvoidanceActionWrapper(self.mock_env, max_avoidance_action=1.5)
        
        # Test invalid smoothing factor
        with self.assertRaises(ValueError):
            ObjectAvoidanceActionWrapper(self.mock_env, smoothing_factor=1.5)
        
        # Test min_clearance >= safety_distance
        with self.assertRaises(ValueError):
            ObjectAvoidanceActionWrapper(
                self.mock_env, 
                safety_distance=0.3, 
                min_clearance=0.4
            )
    
    def test_action_no_detections(self):
        """Test action modification with no detections."""
        # Set up environment with no detections
        self.mock_env._last_observation = {
            'detections': np.zeros((10, 9), dtype=np.float32),
            'detection_count': np.array([0], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = self.wrapper.action(original_action)
        
        # Action should be unchanged (with minimal smoothing effect)
        np.testing.assert_allclose(modified_action, original_action, atol=0.1)
        self.assertFalse(self.wrapper.is_avoidance_active())
        self.assertFalse(self.wrapper.is_emergency_brake_active())
    
    def test_action_with_distant_object(self):
        """Test action modification with distant object (outside safety distance)."""
        # Create detection outside safety distance
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.8, 10, 10, 20, 20, 0.0, 1.0, 0.8]  # distance = 0.8m > 0.5m
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = self.wrapper.action(original_action)
        
        # Action should be unchanged since object is outside safety distance
        np.testing.assert_allclose(modified_action, original_action, atol=0.1)
        self.assertFalse(self.wrapper.is_avoidance_active())
    
    def test_action_with_close_object_right(self):
        """Test action modification with close object on the right."""
        # Create detection on the right side within safety distance
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.9, 10, 10, 20, 20, 0.3, 0.5, 0.3]  # right side, distance = 0.3m
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = self.wrapper.action(original_action)
        
        # Should steer left (increase left wheel, decrease right wheel)
        self.assertGreater(modified_action[0], modified_action[1])
        self.assertTrue(self.wrapper.is_avoidance_active())
    
    def test_action_with_close_object_left(self):
        """Test action modification with close object on the left."""
        # Create detection on the left side within safety distance
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.9, 10, 10, 20, 20, -0.3, 0.5, 0.3]  # left side, distance = 0.3m
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = self.wrapper.action(original_action)
        
        # Should steer right (decrease left wheel, increase right wheel)
        self.assertLess(modified_action[0], modified_action[1])
        self.assertTrue(self.wrapper.is_avoidance_active())
    
    def test_emergency_brake_activation(self):
        """Test emergency brake activation with very close object."""
        # Create detection within emergency brake distance
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.9, 10, 10, 20, 20, 0.0, 0.5, 0.1]  # front center, distance = 0.1m
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = self.wrapper.action(original_action)
        
        # Should apply emergency brake (stop)
        np.testing.assert_allclose(modified_action, [0.0, 0.0], atol=1e-6)
        self.assertTrue(self.wrapper.is_emergency_brake_active())
    
    def test_emergency_brake_disabled(self):
        """Test behavior when emergency brake is disabled."""
        wrapper = ObjectAvoidanceActionWrapper(
            self.mock_env,
            enable_emergency_brake=False
        )
        
        # Create detection within emergency brake distance
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.9, 10, 10, 20, 20, 0.0, 0.5, 0.1]  # front center, distance = 0.1m
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = wrapper.action(original_action)
        
        # Should not apply emergency brake
        self.assertFalse(wrapper.is_emergency_brake_active())
        # But should still apply avoidance
        self.assertTrue(wrapper.is_avoidance_active())
    
    def test_multiple_objects_priority(self):
        """Test avoidance with multiple objects - closer objects should have higher priority."""
        # Create two detections: one close, one far
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.8, 10, 10, 20, 20, 0.2, 0.5, 0.25]   # closer object on right
        detections[1] = [1, 0.7, 30, 30, 40, 40, -0.2, 0.5, 0.4]   # farther object on left
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([2], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        original_action = np.array([0.8, 0.8])
        modified_action = self.wrapper.action(original_action)
        
        # Should primarily avoid the closer object (steer left away from right object)
        self.assertGreater(modified_action[0], modified_action[1])
        self.assertTrue(self.wrapper.is_avoidance_active())
    
    def test_action_smoothing(self):
        """Test action smoothing functionality."""
        wrapper = ObjectAvoidanceActionWrapper(
            self.mock_env,
            smoothing_factor=0.5  # 50% smoothing
        )
        
        # No detections for first action
        self.mock_env._last_observation = {
            'detections': np.zeros((10, 9), dtype=np.float32),
            'detection_count': np.array([0], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32)
        }
        
        action1 = np.array([0.8, 0.8])
        result1 = wrapper.action(action1)
        
        # Second action should be smoothed with first
        action2 = np.array([0.2, 0.2])
        result2 = wrapper.action(action2)
        
        # Result should be between action1 and action2 due to smoothing
        expected = 0.5 * result1 + 0.5 * action2
        np.testing.assert_allclose(result2, expected, atol=0.1)
    
    def test_action_clipping(self):
        """Test that actions are properly clipped to valid range."""
        # Create strong avoidance scenario
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 1.0, 10, 10, 20, 20, 0.4, 0.5, 0.2]  # strong avoidance needed
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        # High avoidance strength to test clipping
        wrapper = ObjectAvoidanceActionWrapper(
            self.mock_env,
            avoidance_strength=2.0,
            max_avoidance_action=0.8
        )
        
        original_action = np.array([0.9, 0.9])
        modified_action = wrapper.action(original_action)
        
        # Actions should be clipped to [0, 1] range
        self.assertTrue(np.all(modified_action >= 0.0))
        self.assertTrue(np.all(modified_action <= 1.0))
    
    def test_detection_filtering(self):
        """Test filtering of irrelevant detections."""
        # Create detections: one relevant, one too far laterally, one behind
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.8, 10, 10, 20, 20, 0.2, 0.5, 0.3]    # relevant
        detections[1] = [1, 0.8, 30, 30, 40, 40, 1.5, 0.5, 0.3]    # too far right
        detections[2] = [1, 0.8, 50, 50, 60, 60, 0.0, -0.5, 0.3]   # behind robot
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([3], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        # Extract and filter detections
        obs = self.wrapper._get_current_observation()
        all_detections = self.wrapper._extract_detections(obs)
        relevant_detections = self.wrapper._filter_relevant_detections(all_detections)
        
        # Should only have one relevant detection
        self.assertEqual(len(relevant_detections), 1)
        self.assertAlmostEqual(relevant_detections[0]['relative_position'][0], 0.2)
    
    def test_statistics_tracking(self):
        """Test avoidance statistics tracking."""
        # Reset stats
        self.wrapper.reset_avoidance_stats()
        
        # Perform actions with and without avoidance
        self.mock_env._last_observation = {
            'detections': np.zeros((10, 9), dtype=np.float32),
            'detection_count': np.array([0], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32)
        }
        
        # Action without avoidance
        self.wrapper.action(np.array([0.8, 0.8]))
        
        # Action with avoidance
        detections = np.zeros((10, 9), dtype=np.float32)
        detections[0] = [1, 0.8, 10, 10, 20, 20, 0.2, 0.5, 0.3]
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        self.wrapper.action(np.array([0.8, 0.8]))
        
        # Check statistics
        stats = self.wrapper.get_avoidance_stats()
        self.assertEqual(stats['total_steps'], 2)
        self.assertEqual(stats['avoidance_activations'], 1)
        self.assertAlmostEqual(stats['avoidance_rate'], 0.5)
    
    def test_configuration_update(self):
        """Test configuration parameter updates."""
        original_safety_distance = self.wrapper.safety_distance
        
        # Update configuration
        self.wrapper.update_configuration(safety_distance=0.8)
        
        self.assertEqual(self.wrapper.safety_distance, 0.8)
        self.assertNotEqual(self.wrapper.safety_distance, original_safety_distance)
    
    def test_reset_functionality(self):
        """Test wrapper reset functionality."""
        # Set some state
        self.wrapper.last_action = np.array([0.5, 0.5])
        self.wrapper.avoidance_active = True
        self.wrapper.emergency_brake_active = True
        
        # Reset wrapper
        self.wrapper.reset()
        
        # State should be reset
        np.testing.assert_allclose(self.wrapper.last_action, [0.0, 0.0])
        self.assertFalse(self.wrapper.avoidance_active)
        self.assertFalse(self.wrapper.emergency_brake_active)
    
    def test_force_calculation_edge_cases(self):
        """Test edge cases in force calculation."""
        # Test with zero distance (should not crash)
        detection = {
            'relative_position': [0.0, 0.0],
            'distance': 0.0,
            'confidence': 1.0
        }
        
        force = self.wrapper._calculate_object_repulsive_force(detection)
        
        # Should return valid force vector
        self.assertEqual(len(force), 2)
        self.assertTrue(np.isfinite(force).all())
    
    def test_observation_extraction_error_handling(self):
        """Test error handling in observation extraction."""
        # Test with invalid observation format
        invalid_obs = "invalid_observation"
        
        detections = self.wrapper._extract_detections(invalid_obs)
        
        # Should return empty list without crashing
        self.assertEqual(detections, [])
    
    def test_action_input_formats(self):
        """Test different action input formats."""
        self.mock_env._last_observation = {
            'detections': np.zeros((10, 9), dtype=np.float32),
            'detection_count': np.array([0], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32)
        }
        
        # Test numpy array
        action_array = np.array([0.8, 0.8])
        result_array = self.wrapper.action(action_array)
        self.assertIsInstance(result_array, np.ndarray)
        
        # Test list
        action_list = [0.8, 0.8]
        result_list = self.wrapper.action(action_list)
        self.assertIsInstance(result_list, np.ndarray)
        
        # Test tuple
        action_tuple = (0.8, 0.8)
        result_tuple = self.wrapper.action(action_tuple)
        self.assertIsInstance(result_tuple, np.ndarray)


class TestObjectAvoidanceIntegration(unittest.TestCase):
    """Integration tests for ObjectAvoidanceActionWrapper."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.mock_env = MockEnvironment()
        self.wrapper = ObjectAvoidanceActionWrapper(
            self.mock_env,
            safety_distance=0.5,
            min_clearance=0.2,
            avoidance_strength=1.0,
            debug_logging=True
        )
    
    def test_full_avoidance_scenario(self):
        """Test complete avoidance scenario from detection to action modification."""
        # Scenario: Object approaching from right side
        detections = np.zeros((10, 9), dtype=np.float32)
        
        # Start with distant object
        detections[0] = [1, 0.8, 10, 10, 20, 20, 0.3, 1.0, 0.6]  # distance = 0.6m
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([1], dtype=np.int32),
            'safety_critical': np.array([0], dtype=np.int32)
        }
        
        # Should not trigger avoidance yet
        action1 = self.wrapper.action(np.array([0.8, 0.8]))
        self.assertFalse(self.wrapper.is_avoidance_active())
        
        # Object gets closer
        detections[0] = [1, 0.8, 10, 10, 20, 20, 0.3, 0.8, 0.4]  # distance = 0.4m
        self.mock_env._last_observation['detections'] = detections
        self.mock_env._last_observation['safety_critical'] = np.array([1], dtype=np.int32)
        
        # Should trigger avoidance
        action2 = self.wrapper.action(np.array([0.8, 0.8]))
        self.assertTrue(self.wrapper.is_avoidance_active())
        
        # Object gets very close
        detections[0] = [1, 0.9, 10, 10, 20, 20, 0.1, 0.5, 0.1]  # distance = 0.1m
        self.mock_env._last_observation['detections'] = detections
        
        # Should trigger emergency brake
        action3 = self.wrapper.action(np.array([0.8, 0.8]))
        self.assertTrue(self.wrapper.is_emergency_brake_active())
        np.testing.assert_allclose(action3, [0.0, 0.0])
    
    def test_performance_requirements(self):
        """Test that wrapper meets performance requirements."""
        import time
        
        # Create complex scenario with multiple objects
        detections = np.zeros((10, 9), dtype=np.float32)
        for i in range(5):
            detections[i] = [1, 0.8, 10*i, 10*i, 20*i, 20*i, 0.1*i, 0.5, 0.3+0.1*i]
        
        self.mock_env._last_observation = {
            'detections': detections,
            'detection_count': np.array([5], dtype=np.int32),
            'safety_critical': np.array([1], dtype=np.int32)
        }
        
        # Measure processing time
        start_time = time.time()
        for _ in range(100):
            self.wrapper.action(np.array([0.8, 0.8]))
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Should process in less than 10ms per action (requirement from design)
        self.assertLess(avg_time, 0.01, f"Average processing time {avg_time:.4f}s exceeds 10ms requirement")


if __name__ == '__main__':
    unittest.main()