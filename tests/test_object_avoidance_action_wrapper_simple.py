"""
Simple unit tests for ObjectAvoidanceActionWrapper core logic.

This module provides tests for the core avoidance algorithms without requiring
gym dependencies, focusing on the mathematical calculations and safety constraints.
"""

import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockActionSpace:
    """Mock action space for testing."""
    def __init__(self, shape):
        self.shape = shape


class MockEnv:
    """Minimal mock environment for testing core functionality."""
    def __init__(self):
        self.action_space = MockActionSpace((2,))
        self._last_observation = {}


class TestObjectAvoidanceCore(unittest.TestCase):
    """Test core avoidance algorithms without gym dependencies."""
    
    def setUp(self):
        """Set up test environment."""
        # Import here to avoid gym dependency at module level
        try:
            from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
            self.ObjectAvoidanceActionWrapper = ObjectAvoidanceActionWrapper
        except ImportError as e:
            self.skipTest(f"Cannot import ObjectAvoidanceActionWrapper: {e}")
        
        self.mock_env = MockEnv()
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test valid configuration
        try:
            wrapper = self.ObjectAvoidanceActionWrapper(
                self.mock_env,
                safety_distance=0.5,
                min_clearance=0.2,
                avoidance_strength=1.0,
                max_avoidance_action=0.8,
                smoothing_factor=0.7
            )
            self.assertIsNotNone(wrapper)
        except Exception as e:
            self.fail(f"Valid configuration should not raise exception: {e}")
        
        # Test invalid avoidance strength
        with self.assertRaises(ValueError):
            self.ObjectAvoidanceActionWrapper(
                self.mock_env,
                avoidance_strength=3.0  # > 2.0
            )
        
        # Test invalid max avoidance action
        with self.assertRaises(ValueError):
            self.ObjectAvoidanceActionWrapper(
                self.mock_env,
                max_avoidance_action=1.5  # > 1.0
            )
        
        # Test invalid clearance relationship
        with self.assertRaises(ValueError):
            self.ObjectAvoidanceActionWrapper(
                self.mock_env,
                safety_distance=0.3,
                min_clearance=0.4  # >= safety_distance
            )
    
    def test_repulsive_force_calculation(self):
        """Test object repulsive force calculation."""
        wrapper = self.ObjectAvoidanceActionWrapper(self.mock_env)
        
        # Test force calculation for object on right
        detection_right = {
            'relative_position': [0.3, 0.5],  # Right side, in front
            'distance': 0.3,
            'confidence': 0.9
        }
        
        force_right = wrapper._calculate_object_repulsive_force(detection_right)
        
        # Force should point left (negative x direction)
        self.assertLess(force_right[0], 0, "Force should point left away from right object")
        self.assertEqual(len(force_right), 2, "Force should be 2D vector")
        
        # Test force calculation for object on left
        detection_left = {
            'relative_position': [-0.3, 0.5],  # Left side, in front
            'distance': 0.3,
            'confidence': 0.9
        }
        
        force_left = wrapper._calculate_object_repulsive_force(detection_left)
        
        # Force should point right (positive x direction)
        self.assertGreater(force_left[0], 0, "Force should point right away from left object")
    
    def test_force_magnitude_by_distance(self):
        """Test that force magnitude decreases with distance."""
        wrapper = self.ObjectAvoidanceActionWrapper(
            self.mock_env,
            safety_distance=0.5,
            min_clearance=0.2
        )
        
        # Close object
        detection_close = {
            'relative_position': [0.2, 0.5],
            'distance': 0.25,  # Between min_clearance and safety_distance
            'confidence': 1.0
        }
        
        # Far object
        detection_far = {
            'relative_position': [0.2, 0.5],
            'distance': 0.45,  # Near safety_distance
            'confidence': 1.0
        }
        
        force_close = wrapper._calculate_object_repulsive_force(detection_close)
        force_far = wrapper._calculate_object_repulsive_force(detection_far)
        
        force_close_mag = np.linalg.norm(force_close)
        force_far_mag = np.linalg.norm(force_far)
        
        # Closer object should generate stronger force
        self.assertGreater(force_close_mag, force_far_mag, 
                          "Closer objects should generate stronger repulsive forces")
    
    def test_detection_filtering(self):
        """Test filtering of relevant detections."""
        wrapper = self.ObjectAvoidanceActionWrapper(
            self.mock_env,
            safety_distance=0.5,
            detection_field_width=1.0
        )
        
        detections = [
            # Relevant detection
            {
                'relative_position': [0.2, 0.5],
                'distance': 0.3,
                'confidence': 0.8
            },
            # Too far laterally
            {
                'relative_position': [0.8, 0.5],
                'distance': 0.3,
                'confidence': 0.8
            },
            # Too far in distance
            {
                'relative_position': [0.2, 0.5],
                'distance': 0.8,
                'confidence': 0.8
            },
            # Behind robot
            {
                'relative_position': [0.2, -0.5],
                'distance': 0.3,
                'confidence': 0.8
            }
        ]
        
        relevant = wrapper._filter_relevant_detections(detections)
        
        # Should only have one relevant detection
        self.assertEqual(len(relevant), 1, "Should filter to only relevant detections")
        self.assertEqual(relevant[0]['relative_position'], [0.2, 0.5])
    
    def test_action_modification_application(self):
        """Test application of avoidance force to actions."""
        wrapper = self.ObjectAvoidanceActionWrapper(self.mock_env)
        
        original_action = np.array([0.8, 0.8])
        
        # Force pointing left (avoid right object)
        avoidance_force = np.array([-0.3, 0.0])  # Strong left force
        
        modified_action = wrapper._apply_avoidance_force(original_action, avoidance_force)
        
        # Left wheel should increase, right wheel should decrease
        self.assertGreater(modified_action[0], modified_action[1], 
                          "Should steer left when avoiding right object")
        
        # Actions should remain in valid range
        self.assertTrue(np.all(modified_action >= 0.0), "Actions should be >= 0")
        self.assertTrue(np.all(modified_action <= 1.0), "Actions should be <= 1")
    
    def test_action_smoothing(self):
        """Test action smoothing functionality."""
        wrapper = self.ObjectAvoidanceActionWrapper(
            self.mock_env,
            smoothing_factor=0.5
        )
        
        # Set previous action
        wrapper.last_action = np.array([0.2, 0.2])
        
        # New action
        new_action = np.array([0.8, 0.8])
        
        smoothed = wrapper._apply_action_smoothing(new_action)
        
        # Should be between previous and new action
        expected = 0.5 * wrapper.last_action + 0.5 * new_action
        np.testing.assert_allclose(smoothed, expected, atol=1e-6)
    
    def test_emergency_brake_logic(self):
        """Test emergency brake decision logic."""
        wrapper = self.ObjectAvoidanceActionWrapper(
            self.mock_env,
            emergency_brake_distance=0.15,
            enable_emergency_brake=True
        )
        
        original_action = np.array([0.8, 0.8])
        
        # Object within emergency brake distance
        close_detections = [{
            'relative_position': [0.0, 0.2],  # Directly in front
            'distance': 0.1,  # Very close
            'confidence': 0.9
        }]
        
        emergency_action = wrapper._check_emergency_brake(close_detections, original_action)
        
        # Should return emergency stop
        self.assertIsNotNone(emergency_action, "Should trigger emergency brake")
        np.testing.assert_allclose(emergency_action, [0.0, 0.0], 
                                  "Emergency action should be stop")
        
        # Object outside emergency brake distance
        far_detections = [{
            'relative_position': [0.0, 0.2],
            'distance': 0.3,  # Outside emergency distance
            'confidence': 0.9
        }]
        
        no_emergency = wrapper._check_emergency_brake(far_detections, original_action)
        
        # Should not trigger emergency brake
        self.assertIsNone(no_emergency, "Should not trigger emergency brake for distant objects")
    
    def test_zero_distance_edge_case(self):
        """Test handling of zero distance edge case."""
        wrapper = self.ObjectAvoidanceActionWrapper(self.mock_env)
        
        # Detection with zero distance
        detection = {
            'relative_position': [0.0, 0.0],
            'distance': 0.0,
            'confidence': 1.0
        }
        
        # Should not crash
        force = wrapper._calculate_object_repulsive_force(detection)
        
        # Should return valid force
        self.assertEqual(len(force), 2)
        self.assertTrue(np.isfinite(force).all())
    
    def test_configuration_getters_setters(self):
        """Test configuration management."""
        wrapper = self.ObjectAvoidanceActionWrapper(
            self.mock_env,
            safety_distance=0.5
        )
        
        # Test getter
        config = wrapper.get_configuration()
        self.assertEqual(config['safety_distance'], 0.5)
        
        # Test setter
        wrapper.update_configuration(safety_distance=0.8)
        self.assertEqual(wrapper.safety_distance, 0.8)
        
        updated_config = wrapper.get_configuration()
        self.assertEqual(updated_config['safety_distance'], 0.8)
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        wrapper = self.ObjectAvoidanceActionWrapper(self.mock_env)
        
        # Reset stats
        wrapper.reset_avoidance_stats()
        
        # Update stats
        wrapper._update_stats(avoidance_active=True)
        wrapper._update_stats(avoidance_active=False)
        wrapper._update_stats(emergency_brake=True)
        
        stats = wrapper.get_avoidance_stats()
        
        self.assertEqual(stats['total_steps'], 3)
        self.assertEqual(stats['avoidance_activations'], 1)
        self.assertEqual(stats['emergency_brakes'], 1)
        self.assertAlmostEqual(stats['avoidance_rate'], 1/3)


if __name__ == '__main__':
    unittest.main()