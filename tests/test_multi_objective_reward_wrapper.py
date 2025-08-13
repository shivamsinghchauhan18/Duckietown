"""
Unit tests for MultiObjectiveRewardWrapper.

Tests the multi-objective reward calculation, weight configuration,
and integration with the Duckietown environment.
"""

import unittest
import numpy as np
import gym
from unittest.mock import Mock, MagicMock, patch
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
from gym_duckietown.simulator import NotInLane


class MockDuckietownEnv:
    """Mock Duckietown environment for testing."""
    
    def __init__(self):
        self.cur_pos = np.array([0.0, 0.0, 0.0])
        self.cur_angle = 0.0
        self.wheelVels = np.array([0.5, 0.5])
        self.collision_penalty = 0.0
        self.last_observation = None
        self.last_action = None
        self.lane_change_state = {}
        
    def get_lane_pos2(self, pos, angle):
        """Mock lane position calculation."""
        mock_lp = Mock()
        mock_lp.dist = 0.0  # Distance from lane center
        mock_lp.angle_deg = 0.0  # Angle deviation in degrees
        return mock_lp
    
    def closest_curve_point(self, pos, angle):
        """Mock closest curve point calculation."""
        curve_point = np.array([0.1, 0.0, 0.0])
        tangent = np.array([1.0, 0.0, 0.0])
        return curve_point, tangent
    
    def proximity_penalty2(self, pos, angle):
        """Mock proximity penalty calculation."""
        return 0.0
    
    def step(self, action):
        """Mock environment step."""
        obs = np.random.rand(84, 84, 3)
        reward = 1.0
        done = False
        info = {}
        return obs, reward, done, info
    
    def reset(self):
        """Mock environment reset."""
        return np.random.rand(84, 84, 3)


class TestMultiObjectiveRewardWrapper(unittest.TestCase):
    """Test cases for MultiObjectiveRewardWrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_env = MockDuckietownEnv()
        self.mock_env.unwrapped = self.mock_env
        
    def test_initialization_default_weights(self):
        """Test wrapper initialization with default weights."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        expected_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5,
            'lane_changing': 0.3,
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
        
        self.assertEqual(wrapper.reward_weights, expected_weights)
        self.assertEqual(wrapper.episode_step, 0)
        self.assertIsNone(wrapper.prev_pos)
    
    def test_initialization_custom_weights(self):
        """Test wrapper initialization with custom weights."""
        custom_weights = {
            'lane_following': 2.0,
            'object_avoidance': 1.0,
            'lane_changing': 0.5,
            'efficiency': 0.1,
            'safety_penalty': -3.0
        }
        
        wrapper = MultiObjectiveRewardWrapper(self.mock_env, custom_weights)
        self.assertEqual(wrapper.reward_weights, custom_weights)
    
    def test_weight_validation_missing_keys(self):
        """Test that missing weight keys raise ValueError."""
        incomplete_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5
            # Missing other required keys
        }
        
        with self.assertRaises(ValueError) as context:
            MultiObjectiveRewardWrapper(self.mock_env, incomplete_weights)
        
        self.assertIn("Missing reward weight keys", str(context.exception))
    
    def test_weight_validation_invalid_type(self):
        """Test that non-numeric weights raise ValueError."""
        invalid_weights = {
            'lane_following': 1.0,
            'object_avoidance': 'invalid',  # Non-numeric
            'lane_changing': 0.3,
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
        
        with self.assertRaises(ValueError) as context:
            MultiObjectiveRewardWrapper(self.mock_env, invalid_weights)
        
        self.assertIn("must be numeric", str(context.exception))
    
    def test_lane_following_reward_calculation(self):
        """Test lane following reward calculation."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Test perfect lane following (center of lane, correct orientation)
        self.mock_env.cur_pos = np.array([0.0, 0.0, 0.0])
        self.mock_env.cur_angle = 0.0
        
        mock_lp = Mock()
        mock_lp.dist = 0.0  # Perfect center
        mock_lp.angle_deg = 0.0  # Perfect orientation
        self.mock_env.get_lane_pos2 = Mock(return_value=mock_lp)
        
        reward = wrapper._calculate_lane_following_reward()
        self.assertEqual(reward, 1.0)  # Perfect score
    
    def test_lane_following_reward_off_center(self):
        """Test lane following reward when off center."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Test off-center position
        mock_lp = Mock()
        mock_lp.dist = 0.025  # Half of max acceptable distance (0.05)
        mock_lp.angle_deg = 15.0  # Half of max acceptable angle (30.0)
        self.mock_env.get_lane_pos2 = Mock(return_value=mock_lp)
        
        reward = wrapper._calculate_lane_following_reward()
        expected = 0.6 * 0.5 + 0.4 * 0.5  # 50% distance + 50% angle
        self.assertAlmostEqual(reward, expected, places=3)
    
    def test_lane_following_reward_not_in_lane(self):
        """Test lane following reward when not in lane."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Simulate NotInLane exception
        self.mock_env.get_lane_pos2 = Mock(side_effect=NotInLane())
        
        reward = wrapper._calculate_lane_following_reward()
        self.assertEqual(reward, -1.0)  # Heavy penalty
    
    def test_object_avoidance_reward_safe_distance(self):
        """Test object avoidance reward with safe distance."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock observation with safe object distance
        self.mock_env.last_observation = {
            'detections': [
                {'distance': 1.0, 'class': 'duckie'},  # Safe distance
                {'distance': 0.8, 'class': 'cone'}
            ]
        }
        
        reward = wrapper._calculate_object_avoidance_reward()
        self.assertEqual(reward, 0.1)  # Small positive reward for safe distance
    
    def test_object_avoidance_reward_unsafe_distance(self):
        """Test object avoidance reward with unsafe distance."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock observation with unsafe object distance
        self.mock_env.last_observation = {
            'detections': [
                {'distance': 0.3, 'class': 'duckie'}  # Unsafe distance (< 0.5)
            ]
        }
        
        reward = wrapper._calculate_object_avoidance_reward()
        expected_penalty = -(0.5 - 0.3) / 0.5  # Proximity penalty
        self.assertAlmostEqual(reward, expected_penalty, places=3)
    
    def test_object_avoidance_reward_no_objects(self):
        """Test object avoidance reward with no objects detected."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock observation with no detections
        self.mock_env.last_observation = {'detections': []}
        
        reward = wrapper._calculate_object_avoidance_reward()
        self.assertEqual(reward, 0.1)  # Small positive reward for no objects
    
    def test_lane_changing_reward_executing(self):
        """Test lane changing reward during execution."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock lane change state during execution
        self.mock_env.lane_change_state = {
            'current_phase': 'executing',
            'progress': 0.7
        }
        
        reward = wrapper._calculate_lane_changing_reward()
        expected = 0.5 * 0.7  # Progress-based reward
        self.assertAlmostEqual(reward, expected, places=3)
    
    def test_lane_changing_reward_completed(self):
        """Test lane changing reward when completed."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock completed lane change
        self.mock_env.lane_change_state = {
            'current_phase': 'following',
            'lane_change_completed': True
        }
        
        reward = wrapper._calculate_lane_changing_reward()
        self.assertEqual(reward, 1.0)  # Completion bonus
    
    def test_efficiency_reward_forward_progress(self):
        """Test efficiency reward for forward progress."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Set previous position
        wrapper.prev_pos = np.array([0.0, 0.0, 0.0])
        self.mock_env.cur_pos = np.array([0.1, 0.0, 0.0])  # Moved forward
        
        # Mock curve point and tangent
        curve_point = np.array([0.05, 0.0, 0.0])
        tangent = np.array([1.0, 0.0, 0.0])  # Forward direction
        self.mock_env.closest_curve_point = Mock(return_value=(curve_point, tangent))
        
        reward = wrapper._calculate_efficiency_reward()
        self.assertGreater(reward, 0)  # Should be positive for forward movement
    
    def test_efficiency_reward_backward_movement(self):
        """Test efficiency reward for backward movement."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Set previous position
        wrapper.prev_pos = np.array([0.1, 0.0, 0.0])
        self.mock_env.cur_pos = np.array([0.0, 0.0, 0.0])  # Moved backward
        
        # Mock curve point and tangent
        curve_point = np.array([0.05, 0.0, 0.0])
        tangent = np.array([1.0, 0.0, 0.0])  # Forward direction
        self.mock_env.closest_curve_point = Mock(return_value=(curve_point, tangent))
        
        reward = wrapper._calculate_efficiency_reward()
        self.assertEqual(reward, -0.5)  # Penalty for backward movement
    
    def test_safety_penalty_collision(self):
        """Test safety penalty for collisions."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock collision
        self.mock_env.collision_penalty = 2.0
        
        penalty = wrapper._calculate_safety_penalty()
        self.assertEqual(penalty, -2.0)  # Negative penalty
    
    def test_safety_penalty_off_lane(self):
        """Test safety penalty for being off lane."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock being far from lane center
        mock_lp = Mock()
        mock_lp.dist = 0.15  # Far from center (> 0.1 threshold)
        mock_lp.angle_deg = 0.0
        self.mock_env.get_lane_pos2 = Mock(return_value=mock_lp)
        
        penalty = wrapper._calculate_safety_penalty()
        expected = -(0.15 * 2.0)  # Distance penalty
        self.assertAlmostEqual(penalty, expected, places=3)
    
    def test_safety_penalty_not_in_lane(self):
        """Test safety penalty for being completely off road."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock NotInLane exception
        self.mock_env.get_lane_pos2 = Mock(side_effect=NotInLane())
        
        penalty = wrapper._calculate_safety_penalty()
        self.assertEqual(penalty, -5.0)  # Heavy penalty for off-road
    
    def test_multi_objective_reward_calculation(self):
        """Test complete multi-objective reward calculation."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock all reward components
        wrapper._calculate_lane_following_reward = Mock(return_value=0.8)
        wrapper._calculate_object_avoidance_reward = Mock(return_value=0.1)
        wrapper._calculate_lane_changing_reward = Mock(return_value=0.5)
        wrapper._calculate_efficiency_reward = Mock(return_value=0.3)
        wrapper._calculate_safety_penalty = Mock(return_value=-0.2)
        
        reward = wrapper.reward(1.0)  # Original reward ignored
        
        # Calculate expected weighted sum
        expected = (
            1.0 * 0.8 +    # lane_following
            0.5 * 0.1 +    # object_avoidance
            0.3 * 0.5 +    # lane_changing
            0.2 * 0.3 +    # efficiency
            -2.0 * -0.2    # safety_penalty
        )
        
        self.assertAlmostEqual(reward, expected, places=3)
    
    def test_step_function(self):
        """Test the step function integration."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Mock reward calculation
        wrapper.reward = Mock(return_value=1.5)
        
        action = np.array([0.5, 0.3])
        obs, reward, done, info = wrapper.step(action)
        
        # Check that reward was calculated
        wrapper.reward.assert_called_once()
        
        # Check that custom rewards are in info
        self.assertIn('custom_rewards', info)
        self.assertIn('reward_weights', info)
        
        # Check that episode step was incremented
        self.assertEqual(wrapper.episode_step, 1)
    
    def test_reset_function(self):
        """Test the reset function."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Set some state
        wrapper.prev_pos = np.array([1.0, 1.0, 1.0])
        wrapper.episode_step = 10
        wrapper.reward_components['lane_following'] = 0.5
        
        obs = wrapper.reset()
        
        # Check that state was reset
        self.assertIsNone(wrapper.prev_pos)
        self.assertEqual(wrapper.episode_step, 0)
        self.assertEqual(wrapper.reward_components['lane_following'], 0.0)
    
    def test_update_weights(self):
        """Test updating reward weights."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        new_weights = {
            'lane_following': 2.0,
            'object_avoidance': 1.0
        }
        
        wrapper.update_weights(new_weights)
        
        # Check that weights were updated
        self.assertEqual(wrapper.reward_weights['lane_following'], 2.0)
        self.assertEqual(wrapper.reward_weights['object_avoidance'], 1.0)
        
        # Check that other weights remain unchanged
        self.assertEqual(wrapper.reward_weights['lane_changing'], 0.3)
    
    def test_update_weights_invalid(self):
        """Test updating weights with invalid values."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        original_weights = wrapper.reward_weights.copy()
        
        invalid_weights = {
            'lane_following': 'invalid'  # Non-numeric
        }
        
        with self.assertRaises(ValueError):
            wrapper.update_weights(invalid_weights)
        
        # Check that original weights are preserved
        self.assertEqual(wrapper.reward_weights, original_weights)
    
    def test_get_reward_components(self):
        """Test getting reward components."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        # Set some components
        wrapper.reward_components['lane_following'] = 0.8
        wrapper.reward_components['total'] = 1.2
        
        components = wrapper.get_reward_components()
        
        # Check that components are returned correctly
        self.assertEqual(components['lane_following'], 0.8)
        self.assertEqual(components['total'], 1.2)
        
        # Check that it's a copy (modifying returned dict doesn't affect original)
        components['lane_following'] = 0.5
        self.assertEqual(wrapper.reward_components['lane_following'], 0.8)
    
    def test_get_reward_weights(self):
        """Test getting reward weights."""
        wrapper = MultiObjectiveRewardWrapper(self.mock_env)
        
        weights = wrapper.get_reward_weights()
        
        # Check that weights are returned correctly
        self.assertEqual(weights['lane_following'], 1.0)
        self.assertEqual(weights['safety_penalty'], -2.0)
        
        # Check that it's a copy
        weights['lane_following'] = 2.0
        self.assertEqual(wrapper.reward_weights['lane_following'], 1.0)


if __name__ == '__main__':
    unittest.main()