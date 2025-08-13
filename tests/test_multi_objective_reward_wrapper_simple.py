"""
Simple unit tests for MultiObjectiveRewardWrapper.

Basic functionality tests that can run without complex dependencies.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper


class SimpleTestMultiObjectiveRewardWrapper(unittest.TestCase):
    """Simple test cases for MultiObjectiveRewardWrapper."""
    
    def setUp(self):
        """Set up simple mock environment."""
        self.mock_env = Mock()
        self.mock_env.unwrapped = Mock()
        
        # Mock basic environment properties
        self.mock_env.unwrapped.cur_pos = np.array([0.0, 0.0, 0.0])
        self.mock_env.unwrapped.cur_angle = 0.0
        self.mock_env.unwrapped.wheelVels = np.array([0.5, 0.5])
        
        # Mock methods
        self.mock_env.step = Mock(return_value=(
            np.random.rand(84, 84, 3),  # observation
            1.0,  # reward
            False,  # done
            {}  # info
        ))
        self.mock_env.reset = Mock(return_value=np.random.rand(84, 84, 3))
    
    def test_initialization_with_defaults(self):
        """Test basic initialization with default weights."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Check that wrapper initializes correctly
        self.assertIsNotNone(wrapper.reward_weights)
        self.assertIn('lane_following', wrapper.reward_weights)
        self.assertIn('object_avoidance', wrapper.reward_weights)
        self.assertIn('lane_changing', wrapper.reward_weights)
        self.assertIn('efficiency', wrapper.reward_weights)
        self.assertIn('safety_penalty', wrapper.reward_weights)
        
        # Check default values
        self.assertEqual(wrapper.reward_weights['lane_following'], 1.0)
        self.assertEqual(wrapper.reward_weights['safety_penalty'], -2.0)
    
    def test_initialization_with_custom_weights(self):
        """Test initialization with custom weights."""
        custom_weights = {
            'lane_following': 2.0,
            'object_avoidance': 1.5,
            'lane_changing': 0.8,
            'efficiency': 0.4,
            'safety_penalty': -3.0
        }
        
        wrapper = MultiObjectiveRewardWrapper(self.mock_env, custom_weights)
        
        # Check that custom weights are set
        for key, value in custom_weights.items():
            self.assertEqual(wrapper.reward_weights[key], value)
    
    def test_weight_validation_error(self):
        """Test that invalid weights raise appropriate errors."""
        # Test missing keys
        incomplete_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5
        }
        
        with self.assertRaises(ValueError):
            MultiObjectiveRewardWrapper(self.mock_env, incomplete_weights)
        
        # Test invalid type
        invalid_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5,
            'lane_changing': 'invalid',
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
        
        with self.assertRaises(ValueError):
            MultiObjectiveRewardWrapper(self.mock_env, invalid_weights)
    
    def test_reward_components_structure(self):
        """Test that reward components have correct structure."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        components = wrapper.get_reward_components()
        
        # Check that all expected components exist
        expected_components = [
            'lane_following',
            'object_avoidance', 
            'lane_changing',
            'efficiency',
            'safety_penalty',
            'total'
        ]
        
        for component in expected_components:
            self.assertIn(component, components)
            self.assertIsInstance(components[component], (int, float))
    
    def test_step_integration(self):
        """Test basic step function integration."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock the reward calculation to return a simple value
        wrapper.reward = Mock(return_value=1.5)
        
        action = np.array([0.5, 0.3])
        obs, reward, done, info = wrapper.step(action)
        
        # Check that environment step was called
        self.mock_env.step.assert_called_once_with(action)
        
        # Check that reward was calculated
        wrapper.reward.assert_called_once()
        
        # Check that info contains custom rewards
        self.assertIn('custom_rewards', info)
        self.assertIn('reward_weights', info)
        
        # Check return types
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_reset_integration(self):
        """Test basic reset function integration."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Set some state to verify reset
        wrapper.episode_step = 10
        wrapper.prev_pos = np.array([1.0, 1.0, 1.0])
        
        obs = wrapper.reset()
        
        # Check that environment reset was called
        self.mock_env.reset.assert_called_once()
        
        # Check that state was reset
        self.assertEqual(wrapper.episode_step, 0)
        self.assertIsNone(wrapper.prev_pos)
        
        # Check return type
        self.assertIsInstance(obs, np.ndarray)
    
    def test_update_weights_functionality(self):
        """Test weight update functionality."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        original_lane_weight = wrapper.reward_weights['lane_following']
        
        # Update some weights
        new_weights = {
            'lane_following': 2.5,
            'efficiency': 0.8
        }
        
        wrapper.update_weights(new_weights)
        
        # Check that specified weights were updated
        self.assertEqual(wrapper.reward_weights['lane_following'], 2.5)
        self.assertEqual(wrapper.reward_weights['efficiency'], 0.8)
        
        # Check that other weights remain unchanged
        self.assertEqual(wrapper.reward_weights['object_avoidance'], 0.5)  # Default value
    
    def test_get_methods(self):
        """Test getter methods return correct data."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Test get_reward_weights
        weights = wrapper.get_reward_weights()
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 5)  # Should have 5 weight components
        
        # Test get_reward_components
        components = wrapper.get_reward_components()
        self.assertIsInstance(components, dict)
        self.assertEqual(len(components), 6)  # Should have 6 components (including total)
        
        # Test that returned dicts are copies (not references)
        weights['lane_following'] = 999
        self.assertNotEqual(wrapper.reward_weights['lane_following'], 999)
        
        components['total'] = 999
        self.assertNotEqual(wrapper.reward_components['total'], 999)
    
    def test_reward_calculation_basic(self):
        """Test basic reward calculation without complex mocking."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock individual reward methods to return known values
        wrapper._calculate_lane_following_reward = Mock(return_value=0.8)
        wrapper._calculate_object_avoidance_reward = Mock(return_value=0.2)
        wrapper._calculate_lane_changing_reward = Mock(return_value=0.1)
        wrapper._calculate_efficiency_reward = Mock(return_value=0.3)
        wrapper._calculate_safety_penalty = Mock(return_value=-0.1)
        
        reward = wrapper.reward(1.0)  # Original reward is ignored
        
        # Calculate expected weighted sum
        expected = (
            1.0 * 0.8 +    # lane_following: 1.0 * 0.8 = 0.8
            0.5 * 0.2 +    # object_avoidance: 0.5 * 0.2 = 0.1
            0.3 * 0.1 +    # lane_changing: 0.3 * 0.1 = 0.03
            0.2 * 0.3 +    # efficiency: 0.2 * 0.3 = 0.06
            -2.0 * -0.1    # safety_penalty: -2.0 * -0.1 = 0.2
        )  # Total: 0.8 + 0.1 + 0.03 + 0.06 + 0.2 = 1.19
        
        self.assertAlmostEqual(reward, expected, places=3)
        
        # Check that components were stored
        self.assertEqual(wrapper.reward_components['lane_following'], 0.8)
        self.assertEqual(wrapper.reward_components['total'], expected)


if __name__ == '__main__':
    unittest.main()