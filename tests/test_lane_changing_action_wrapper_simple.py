"""
Simple unit tests for Lane Changing Action Wrapper.

This module contains basic tests for the LaneChangingActionWrapper class
to verify core functionality and integration.
"""

import unittest
import numpy as np
import gym
from gym import spaces

# Import the wrapper to test
from duckietown_utils.wrappers.lane_changing_action_wrapper import (
    LaneChangingActionWrapper,
    LaneChangePhase
)


class SimpleMockEnvironment(gym.Env):
    """Simple mock environment for basic testing."""
    
    def __init__(self):
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        self._last_observation = {}
    
    def step(self, action):
        return np.zeros((120, 160, 3)), 0.0, False, {}
    
    def reset(self):
        return np.zeros((120, 160, 3))


class TestLaneChangingActionWrapperSimple(unittest.TestCase):
    """Simple test cases for LaneChangingActionWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = SimpleMockEnvironment()
        self.wrapper = LaneChangingActionWrapper(
            self.mock_env,
            debug_logging=False  # Reduce log noise in tests
        )
    
    def test_basic_initialization(self):
        """Test basic wrapper initialization."""
        self.assertIsInstance(self.wrapper, LaneChangingActionWrapper)
        self.assertEqual(self.wrapper.current_lane, 0)
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
    
    def test_action_passthrough_basic(self):
        """Test basic action passthrough functionality."""
        # Set up empty observation (no obstacles)
        self.mock_env._last_observation = {}
        
        original_action = np.array([0.5, 0.7])
        result_action = self.wrapper.action(original_action)
        
        # Should pass through unchanged when no lane change is needed
        np.testing.assert_array_equal(result_action, original_action)
    
    def test_action_with_list_input(self):
        """Test action method with list input."""
        self.mock_env._last_observation = {}
        
        original_action = [0.4, 0.6]  # List instead of numpy array
        result_action = self.wrapper.action(original_action)
        
        # Should handle list input and return numpy array
        self.assertIsInstance(result_action, np.ndarray)
        np.testing.assert_array_equal(result_action, np.array(original_action))
    
    def test_disabled_lane_changing(self):
        """Test wrapper behavior when lane changing is disabled."""
        wrapper = LaneChangingActionWrapper(
            self.mock_env,
            enable_lane_changing=False
        )
        
        original_action = np.array([0.3, 0.8])
        result_action = wrapper.action(original_action)
        
        np.testing.assert_array_equal(result_action, original_action)
    
    def test_basic_statistics(self):
        """Test basic statistics functionality."""
        stats = self.wrapper.get_lane_change_stats()
        
        # Check that statistics dictionary has expected keys
        expected_keys = [
            'total_steps', 'lane_change_attempts', 'successful_lane_changes',
            'aborted_lane_changes', 'emergency_aborts', 'success_rate',
            'current_phase', 'current_lane'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Initial values should be zero/default
        self.assertEqual(stats['total_steps'], 0)
        self.assertEqual(stats['lane_change_attempts'], 0)
        self.assertEqual(stats['current_lane'], 0)
    
    def test_configuration_getter(self):
        """Test configuration getter functionality."""
        config = self.wrapper.get_configuration()
        
        # Check that configuration has expected parameters
        expected_params = [
            'lane_change_threshold', 'safety_margin', 'max_lane_change_time',
            'num_lanes', 'enable_lane_changing'
        ]
        
        for param in expected_params:
            self.assertIn(param, config)
    
    def test_utility_methods_basic(self):
        """Test basic utility methods."""
        # Test current lane
        self.assertEqual(self.wrapper.get_current_lane(), 0)
        
        # Test lane changing status
        self.assertFalse(self.wrapper.is_lane_changing())
        
        # Test phase getter
        self.assertEqual(self.wrapper.get_lane_change_phase(), LaneChangePhase.LANE_FOLLOWING)
    
    def test_reset_basic(self):
        """Test basic reset functionality."""
        # Modify some state
        self.wrapper.current_lane = 1
        
        # Reset
        result = self.wrapper.reset()
        
        # Check that state is reset
        self.assertEqual(self.wrapper.current_lane, 0)
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
    
    def test_force_lane_change_basic(self):
        """Test basic forced lane change functionality."""
        # Test valid lane change
        result = self.wrapper.force_lane_change(1)
        self.assertTrue(result)
        
        # Test invalid lane (out of range)
        result = self.wrapper.force_lane_change(10)
        self.assertFalse(result)
        
        # Reset for next test
        self.wrapper.reset()
        
        # Test same lane
        result = self.wrapper.force_lane_change(0)
        self.assertFalse(result)
    
    def test_action_clipping(self):
        """Test that actions are properly clipped to valid range."""
        # This test verifies that even if internal calculations produce
        # out-of-range values, the final action is clipped
        
        self.mock_env._last_observation = {}
        
        # Test with extreme input values
        extreme_action = np.array([2.0, -1.0])  # Out of valid range [0, 1]
        result_action = self.wrapper.action(extreme_action)
        
        # Result should be clipped to valid range
        self.assertTrue(np.all(result_action >= 0.0))
        self.assertTrue(np.all(result_action <= 1.0))
    
    def test_multiple_action_calls(self):
        """Test multiple consecutive action calls."""
        self.mock_env._last_observation = {}
        
        actions = [
            np.array([0.2, 0.3]),
            np.array([0.5, 0.5]),
            np.array([0.8, 0.7])
        ]
        
        # Process multiple actions
        for action in actions:
            result = self.wrapper.action(action)
            
            # Each should be processed without error
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 2)
            
            # Update statistics
            stats = self.wrapper.get_lane_change_stats()
            self.assertGreaterEqual(stats['total_steps'], 1)


class TestLaneChangingConfigurationValidation(unittest.TestCase):
    """Test configuration validation for LaneChangingActionWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = SimpleMockEnvironment()
    
    def test_valid_configurations(self):
        """Test that valid configurations are accepted."""
        # Test minimal valid configuration
        wrapper1 = LaneChangingActionWrapper(self.mock_env)
        self.assertIsInstance(wrapper1, LaneChangingActionWrapper)
        
        # Test custom valid configuration
        wrapper2 = LaneChangingActionWrapper(
            self.mock_env,
            lane_change_threshold=0.5,
            safety_margin=1.5,
            max_lane_change_time=4.0,
            min_lane_change_time=0.5,
            num_lanes=3
        )
        self.assertIsInstance(wrapper2, LaneChangingActionWrapper)
    
    def test_invalid_threshold(self):
        """Test invalid lane change threshold values."""
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, lane_change_threshold=-0.1)
        
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, lane_change_threshold=1.1)
    
    def test_invalid_safety_margin(self):
        """Test invalid safety margin values."""
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, safety_margin=0.0)
        
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, safety_margin=-1.0)
    
    def test_invalid_time_constraints(self):
        """Test invalid time constraint values."""
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(
                self.mock_env,
                max_lane_change_time=1.0,
                min_lane_change_time=2.0  # min > max
            )
    
    def test_invalid_num_lanes(self):
        """Test invalid number of lanes."""
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, num_lanes=0)
        
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, num_lanes=-1)


if __name__ == '__main__':
    unittest.main()