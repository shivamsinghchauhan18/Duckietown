"""
Unit tests for Lane Changing Action Wrapper.

This module contains comprehensive tests for the LaneChangingActionWrapper class,
including state machine transitions, safety checks, trajectory planning, and
error handling scenarios.
"""

import unittest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gym
from gym import spaces

# Import the wrapper to test
from duckietown_utils.wrappers.lane_changing_action_wrapper import (
    LaneChangingActionWrapper,
    LaneChangePhase,
    LaneChangeState,
    LaneInfo
)


class MockEnvironment(gym.Env):
    """Mock environment for testing."""
    
    def __init__(self):
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        self._last_observation = {}
    
    def step(self, action):
        return np.zeros((120, 160, 3)), 0.0, False, {}
    
    def reset(self):
        return np.zeros((120, 160, 3))
    
    def render(self, mode='human'):
        pass


class TestLaneChangingActionWrapper(unittest.TestCase):
    """Test cases for LaneChangingActionWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()
        self.wrapper = LaneChangingActionWrapper(
            self.mock_env,
            lane_change_threshold=0.3,
            safety_margin=2.0,
            max_lane_change_time=3.0,
            min_lane_change_time=1.0,
            lane_width=0.6,
            num_lanes=2,
            debug_logging=True
        )
    
    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.num_lanes, 2)
        self.assertEqual(self.wrapper.current_lane, 0)
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        self.assertIsInstance(self.wrapper.action_space, spaces.Box)
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test invalid lane_change_threshold
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, lane_change_threshold=1.5)
        
        # Test invalid safety_margin
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, safety_margin=-1.0)
        
        # Test invalid time constraints
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(
                self.mock_env, 
                max_lane_change_time=1.0, 
                min_lane_change_time=2.0
            )
        
        # Test invalid trajectory_smoothness
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, trajectory_smoothness=1.5)
        
        # Test invalid lane_change_speed_factor
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, lane_change_speed_factor=0.0)
        
        # Test invalid num_lanes
        with self.assertRaises(ValueError):
            LaneChangingActionWrapper(self.mock_env, num_lanes=0)
    
    def test_action_passthrough_when_disabled(self):
        """Test that actions pass through unchanged when lane changing is disabled."""
        wrapper = LaneChangingActionWrapper(self.mock_env, enable_lane_changing=False)
        
        original_action = np.array([0.5, 0.7])
        result_action = wrapper.action(original_action)
        
        np.testing.assert_array_equal(result_action, original_action)
    
    def test_action_passthrough_during_lane_following(self):
        """Test that actions pass through unchanged during normal lane following."""
        # Set up observation with no obstacles
        self.mock_env._last_observation = {
            'detections': np.zeros((10, 9)),
            'detection_count': [0]
        }
        
        original_action = np.array([0.5, 0.7])
        result_action = self.wrapper.action(original_action)
        
        np.testing.assert_array_equal(result_action, original_action)
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
    
    def test_detection_extraction(self):
        """Test extraction of detection information from observations."""
        # Create mock observation with detections
        detection_data = np.zeros((10, 9))
        detection_data[0] = [1, 0.8, 100, 50, 150, 100, 0.2, 1.5, 1.0]  # Object ahead
        detection_data[1] = [2, 0.6, 200, 60, 250, 110, -0.3, 2.0, 1.8]  # Object to left
        
        self.mock_env._last_observation = {
            'detections': detection_data,
            'detection_count': [2]
        }
        
        detections = self.wrapper._extract_detections(self.mock_env._last_observation)
        
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]['class_id'], 1)
        self.assertEqual(detections[0]['confidence'], 0.8)
        self.assertEqual(detections[0]['distance'], 1.0)
        self.assertEqual(detections[1]['class_id'], 2)
        self.assertEqual(detections[1]['confidence'], 0.6)
    
    def test_obstacle_detection_in_current_lane(self):
        """Test detection of obstacles in current lane."""
        detections = [
            {
                'class_id': 1,
                'confidence': 0.8,
                'relative_position': [0.1, 1.5],  # Slightly right, ahead
                'distance': 1.0
            },
            {
                'class_id': 2,
                'confidence': 0.6,
                'relative_position': [0.8, 2.0],  # Far right, ahead (different lane)
                'distance': 1.8
            }
        ]
        
        obstacles_in_lane = self.wrapper._get_obstacles_in_current_lane(detections)
        
        # Only first obstacle should be in current lane (within lane_width/2 = 0.3)
        self.assertEqual(len(obstacles_in_lane), 1)
        self.assertEqual(obstacles_in_lane[0]['class_id'], 1)
    
    def test_lane_change_consideration(self):
        """Test logic for considering lane changes."""
        # Test with no obstacles - should not consider lane change
        self.assertFalse(self.wrapper._should_consider_lane_change({}))
        
        # Test with distant obstacle - should not consider lane change
        detections = [
            {
                'class_id': 1,
                'confidence': 0.8,
                'relative_position': [0.1, 1.5],
                'distance': 5.0  # Far away
            }
        ]
        
        with patch.object(self.wrapper, '_extract_detections', return_value=detections):
            self.assertFalse(self.wrapper._should_consider_lane_change({}))
        
        # Test with close obstacle - should consider lane change
        detections[0]['distance'] = 3.0  # Within 2x safety margin (4.0)
        
        with patch.object(self.wrapper, '_extract_detections', return_value=detections):
            self.assertTrue(self.wrapper._should_consider_lane_change({}))
    
    def test_lane_analysis(self):
        """Test analysis of individual lanes."""
        detections = [
            {
                'class_id': 1,
                'confidence': 0.8,
                'relative_position': [0.1, 1.5],  # In current lane (lane 0)
                'distance': 1.0
            },
            {
                'class_id': 2,
                'confidence': 0.6,
                'relative_position': [0.7, 2.0],  # In lane 1 (0.6m offset)
                'distance': 1.8
            }
        ]
        
        # Analyze current lane (should have obstacle)
        lane_0_info = self.wrapper._analyze_lane(0, detections)
        self.assertFalse(lane_0_info.is_available)
        self.assertLess(lane_0_info.safe_distance_ahead, 2.0)
        
        # Analyze lane 1 (should also have obstacle but might be available depending on distance)
        lane_1_info = self.wrapper._analyze_lane(1, detections)
        # Lane 1 has obstacle at 1.8m distance, which is less than safety_margin (2.0)
        self.assertFalse(lane_1_info.is_available)
    
    def test_lane_selection(self):
        """Test selection of best lane for lane change."""
        # Create lane options
        lane_options = [
            LaneInfo(lane_id=0, occupancy=0.5, safe_distance_ahead=1.0, is_available=False),
            LaneInfo(lane_id=1, occupancy=0.1, safe_distance_ahead=5.0, is_available=True)
        ]
        
        best_lane = self.wrapper._select_best_lane(lane_options)
        self.assertEqual(best_lane, 1)
        
        # Test with no available lanes
        for lane in lane_options:
            lane.is_available = False
        
        best_lane = self.wrapper._select_best_lane(lane_options)
        self.assertIsNone(best_lane)
    
    def test_safety_checks(self):
        """Test safety check functionality."""
        # Mock target lane analysis
        target_lane_info = LaneInfo(
            lane_id=1,
            occupancy=0.1,
            safe_distance_ahead=3.0,
            safe_distance_behind=3.0,
            is_available=True
        )
        
        self.wrapper.lane_change_state.target_lane = 1
        
        with patch.object(self.wrapper, '_analyze_lane', return_value=target_lane_info):
            with patch.object(self.wrapper, '_check_emergency_obstacles', return_value=True):
                result = self.wrapper._perform_final_safety_checks({})
                self.assertTrue(result)
        
        # Test with unsafe target lane
        target_lane_info.is_available = False
        
        with patch.object(self.wrapper, '_analyze_lane', return_value=target_lane_info):
            result = self.wrapper._perform_final_safety_checks({})
            self.assertFalse(result)
    
    def test_emergency_obstacle_detection(self):
        """Test emergency obstacle detection."""
        # Test with safe detections
        safe_detections = [
            {
                'distance': 1.0,
                'relative_position': [0.1, 1.5]
            }
        ]
        
        result = self.wrapper._check_emergency_obstacles(safe_detections)
        self.assertTrue(result)
        
        # Test with emergency obstacle (too close)
        emergency_detections = [
            {
                'distance': 0.2,  # Less than emergency_abort_distance (0.3)
                'relative_position': [0.1, 1.5]
            }
        ]
        
        result = self.wrapper._check_emergency_obstacles(emergency_detections)
        self.assertFalse(result)
    
    def test_trajectory_planning(self):
        """Test lane change trajectory planning."""
        self.wrapper.lane_change_state.target_lane = 1
        self.wrapper.current_lane = 0
        
        self.wrapper._plan_lane_change_trajectory()
        
        # Check that trajectory was created
        self.assertGreater(len(self.wrapper.lane_change_trajectory), 0)
        
        # Check trajectory endpoints
        self.assertAlmostEqual(self.wrapper.lane_change_trajectory[0], 0.0, places=2)  # Start at current lane
        expected_end = (1 - 0) * self.wrapper.lane_width  # End at target lane
        self.assertAlmostEqual(self.wrapper.lane_change_trajectory[-1], expected_end, places=2)
    
    def test_trajectory_smoothing(self):
        """Test trajectory smoothing function."""
        # Test with different smoothness values
        progress_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for smoothness in [0.0, 0.5, 1.0]:
            self.wrapper.trajectory_smoothness = smoothness
            
            smoothed_values = [self.wrapper._smooth_trajectory_progress(p) for p in progress_values]
            
            # Check endpoints are preserved
            self.assertAlmostEqual(smoothed_values[0], 0.0, places=6)
            self.assertAlmostEqual(smoothed_values[-1], 1.0, places=6)
            
            # Check monotonicity
            for i in range(len(smoothed_values) - 1):
                self.assertLessEqual(smoothed_values[i], smoothed_values[i + 1])
    
    def test_action_modification_during_lane_change(self):
        """Test action modification during lane change execution."""
        # Set up lane change state
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        self.wrapper.lane_change_state.target_lane = 1
        self.wrapper.lane_change_state.progress = 0.5
        self.wrapper._plan_lane_change_trajectory()
        
        original_action = np.array([0.6, 0.6])
        modified_action = self.wrapper._modify_action_for_lane_change(original_action)
        
        # Action should be modified (different from original)
        self.assertFalse(np.array_equal(modified_action, original_action))
        
        # Actions should be in valid range
        self.assertTrue(np.all(modified_action >= 0.0))
        self.assertTrue(np.all(modified_action <= 1.0))
        
        # Speed should be reduced
        original_speed = np.mean(original_action)
        modified_speed = np.mean(modified_action)
        self.assertLess(modified_speed, original_speed)
    
    def test_state_machine_transitions(self):
        """Test state machine phase transitions."""
        # Start in lane following
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        
        # Mock obstacle detection to trigger evaluation
        with patch.object(self.wrapper, '_should_consider_lane_change', return_value=True):
            self.wrapper._update_state_machine({})
        
        # Should transition to evaluating
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.EVALUATING_CHANGE)
        
        # Mock evaluation completion with valid lane
        self.wrapper.evaluation_start_time = time.time() - 1.0  # Simulate elapsed time
        
        with patch.object(self.wrapper, '_evaluate_lane_options') as mock_evaluate:
            with patch.object(self.wrapper, '_select_best_lane', return_value=1):
                mock_evaluate.return_value = [
                    LaneInfo(lane_id=0, is_available=False),
                    LaneInfo(lane_id=1, is_available=True)
                ]
                
                self.wrapper._update_state_machine({})
        
        # Should transition to initiating
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.INITIATING_CHANGE)
        self.assertEqual(self.wrapper.lane_change_state.target_lane, 1)
    
    def test_lane_change_completion(self):
        """Test successful lane change completion."""
        # Set up executing state
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        self.wrapper.lane_change_state.target_lane = 1
        self.wrapper.lane_change_state.progress = 1.0
        self.wrapper.lane_change_state.start_time = time.time() - 2.0  # Sufficient time elapsed
        
        initial_lane = self.wrapper.current_lane
        initial_successful_changes = self.wrapper.lane_change_stats['successful_lane_changes']
        
        self.wrapper._complete_lane_change()
        
        # Check state updates
        self.assertEqual(self.wrapper.current_lane, 1)
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        self.assertEqual(
            self.wrapper.lane_change_stats['successful_lane_changes'],
            initial_successful_changes + 1
        )
    
    def test_lane_change_abort(self):
        """Test lane change abort scenarios."""
        # Set up executing state
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        self.wrapper.lane_change_state.target_lane = 1
        
        initial_aborts = self.wrapper.lane_change_stats['aborted_lane_changes']
        
        self.wrapper._abort_lane_change()
        
        # Check state reset
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        self.assertEqual(
            self.wrapper.lane_change_stats['aborted_lane_changes'],
            initial_aborts + 1
        )
    
    def test_emergency_abort(self):
        """Test emergency abort functionality."""
        # Set up executing state
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        
        initial_emergency_aborts = self.wrapper.lane_change_stats['emergency_aborts']
        
        self.wrapper._emergency_abort_lane_change()
        
        # Check state reset and statistics
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        self.assertEqual(
            self.wrapper.lane_change_stats['emergency_aborts'],
            initial_emergency_aborts + 1
        )
    
    def test_timeout_handling(self):
        """Test lane change timeout handling."""
        # Set up executing state with timeout
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        self.wrapper.lane_change_state.start_time = time.time() - 5.0  # Exceed max_lane_change_time
        
        with patch.object(self.wrapper, '_should_emergency_abort', return_value=False):
            self.wrapper._update_state_machine({})
        
        # Should abort due to timeout
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
    
    def test_force_lane_change(self):
        """Test forced lane change functionality."""
        # Test valid forced lane change
        result = self.wrapper.force_lane_change(1)
        self.assertTrue(result)
        self.assertEqual(self.wrapper.lane_change_state.target_lane, 1)
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.INITIATING_CHANGE)
        
        # Test invalid lane number
        result = self.wrapper.force_lane_change(5)
        self.assertFalse(result)
        
        # Test same lane
        self.wrapper.current_lane = 1
        result = self.wrapper.force_lane_change(1)
        self.assertFalse(result)
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        initial_stats = self.wrapper.get_lane_change_stats()
        
        # Simulate some lane change attempts
        self.wrapper.lane_change_stats['lane_change_attempts'] = 10
        self.wrapper.lane_change_stats['successful_lane_changes'] = 7
        self.wrapper.lane_change_stats['aborted_lane_changes'] = 2
        self.wrapper.lane_change_stats['emergency_aborts'] = 1
        
        stats = self.wrapper.get_lane_change_stats()
        
        self.assertEqual(stats['success_rate'], 0.7)
        self.assertEqual(stats['abort_rate'], 0.2)
        self.assertEqual(stats['emergency_abort_rate'], 0.1)
        self.assertEqual(stats['current_lane'], self.wrapper.current_lane)
        self.assertEqual(stats['current_phase'], self.wrapper.lane_change_state.current_phase.value)
    
    def test_configuration_updates(self):
        """Test configuration parameter updates."""
        original_threshold = self.wrapper.lane_change_threshold
        
        # Update configuration
        self.wrapper.update_configuration(lane_change_threshold=0.5)
        
        self.assertEqual(self.wrapper.lane_change_threshold, 0.5)
        self.assertNotEqual(self.wrapper.lane_change_threshold, original_threshold)
        
        # Test invalid configuration update
        with self.assertRaises(ValueError):
            self.wrapper.update_configuration(lane_change_threshold=1.5)
    
    def test_utility_methods(self):
        """Test utility methods."""
        # Test lane changing status
        self.assertFalse(self.wrapper.is_lane_changing())
        
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        self.assertTrue(self.wrapper.is_lane_changing())
        
        # Test current lane getter
        self.assertEqual(self.wrapper.get_current_lane(), 0)
        
        # Test phase getter
        self.assertEqual(self.wrapper.get_lane_change_phase(), LaneChangePhase.EXECUTING_CHANGE)
    
    def test_reset_functionality(self):
        """Test wrapper reset functionality."""
        # Modify state
        self.wrapper.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
        self.wrapper.current_lane = 1
        self.wrapper.lane_change_trajectory = [0.1, 0.2, 0.3]
        
        # Reset
        self.wrapper.reset()
        
        # Check state reset
        self.assertEqual(self.wrapper.lane_change_state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        self.assertEqual(self.wrapper.current_lane, 0)
        self.assertEqual(len(self.wrapper.lane_change_trajectory), 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty detections
        result = self.wrapper._extract_detections({})
        self.assertEqual(len(result), 0)
        
        # Test with malformed observation
        result = self.wrapper._extract_detections("invalid")
        self.assertEqual(len(result), 0)
        
        # Test trajectory planning with same lane
        self.wrapper.lane_change_state.target_lane = 0  # Same as current
        self.wrapper._plan_lane_change_trajectory()
        
        # Should still create trajectory (even if minimal)
        self.assertGreaterEqual(len(self.wrapper.lane_change_trajectory), 0)
        
        # Test action modification with empty trajectory
        self.wrapper.lane_change_trajectory = []
        original_action = np.array([0.5, 0.5])
        result = self.wrapper._modify_action_for_lane_change(original_action)
        np.testing.assert_array_equal(result, original_action)


class TestLaneChangeDataStructures(unittest.TestCase):
    """Test lane change data structures."""
    
    def test_lane_change_state(self):
        """Test LaneChangeState dataclass."""
        state = LaneChangeState()
        
        self.assertEqual(state.current_phase, LaneChangePhase.LANE_FOLLOWING)
        self.assertIsNone(state.target_lane)
        self.assertEqual(state.progress, 0.0)
        self.assertIsInstance(state.safety_checks, dict)
    
    def test_lane_info(self):
        """Test LaneInfo dataclass."""
        lane_info = LaneInfo(lane_id=1)
        
        self.assertEqual(lane_info.lane_id, 1)
        self.assertEqual(lane_info.occupancy, 0.0)
        self.assertEqual(lane_info.safe_distance_ahead, float('inf'))
        self.assertEqual(lane_info.safe_distance_behind, float('inf'))
        self.assertTrue(lane_info.is_available)
    
    def test_lane_change_phase_enum(self):
        """Test LaneChangePhase enum."""
        self.assertEqual(LaneChangePhase.LANE_FOLLOWING.value, "following")
        self.assertEqual(LaneChangePhase.EVALUATING_CHANGE.value, "evaluating")
        self.assertEqual(LaneChangePhase.INITIATING_CHANGE.value, "initiating")
        self.assertEqual(LaneChangePhase.EXECUTING_CHANGE.value, "executing")


if __name__ == '__main__':
    unittest.main()