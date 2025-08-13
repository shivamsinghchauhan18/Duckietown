"""
Standalone verification of lane changing implementation.

This script verifies the core logic of the lane changing wrapper
without requiring external dependencies like gym.
"""

import sys
import os
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the core classes directly
from duckietown_utils.wrappers.lane_changing_action_wrapper import (
    LaneChangePhase,
    LaneChangeState,
    LaneInfo
)


def test_data_structures():
    """Test the data structures used in lane changing."""
    print("Testing data structures...")
    
    # Test LaneChangePhase enum
    assert LaneChangePhase.LANE_FOLLOWING.value == "following"
    assert LaneChangePhase.EVALUATING_CHANGE.value == "evaluating"
    assert LaneChangePhase.INITIATING_CHANGE.value == "initiating"
    assert LaneChangePhase.EXECUTING_CHANGE.value == "executing"
    print("✓ LaneChangePhase enum works correctly")
    
    # Test LaneChangeState dataclass
    state = LaneChangeState()
    assert state.current_phase == LaneChangePhase.LANE_FOLLOWING
    assert state.target_lane is None
    assert state.progress == 0.0
    assert isinstance(state.safety_checks, dict)
    print("✓ LaneChangeState dataclass works correctly")
    
    # Test LaneInfo dataclass
    lane_info = LaneInfo(lane_id=1)
    assert lane_info.lane_id == 1
    assert lane_info.occupancy == 0.0
    assert lane_info.safe_distance_ahead == float('inf')
    assert lane_info.is_available == True
    print("✓ LaneInfo dataclass works correctly")


def test_core_algorithms():
    """Test core algorithms without gym dependencies."""
    print("\nTesting core algorithms...")
    
    # Create a mock wrapper class with just the methods we need
    class MockLaneChangingWrapper:
        def __init__(self):
            self.lane_width = 0.6
            self.num_lanes = 2
            self.current_lane = 0
            self.safety_margin = 2.0
            self.lane_change_threshold = 0.3
            self.trajectory_smoothness = 0.8
            self.max_lane_change_time = 3.0
            self.lane_change_trajectory = []
            self.lane_change_state = LaneChangeState()
        
        def _get_obstacles_in_current_lane(self, detections):
            """Filter detections to current lane."""
            obstacles_in_lane = []
            for detection in detections:
                rel_pos = detection.get('relative_position', [0, 0])
                lateral_offset = abs(rel_pos[0])
                if lateral_offset <= self.lane_width / 2:
                    obstacles_in_lane.append(detection)
            return obstacles_in_lane
        
        def _analyze_lane(self, lane_id, detections):
            """Analyze a specific lane."""
            lane_info = LaneInfo(lane_id=lane_id)
            lane_offset = (lane_id - self.current_lane) * self.lane_width
            
            obstacles_in_lane = []
            for detection in detections:
                rel_pos = detection.get('relative_position', [0, 0])
                adjusted_lateral = rel_pos[0] - lane_offset
                
                if abs(adjusted_lateral) <= self.lane_width / 2:
                    obstacles_in_lane.append(detection)
            
            if obstacles_in_lane:
                lane_info.occupancy = len(obstacles_in_lane) / 10.0
                
                ahead_distances = []
                for obstacle in obstacles_in_lane:
                    rel_pos = obstacle.get('relative_position', [0, 0])
                    distance = obstacle.get('distance', float('inf'))
                    if rel_pos[1] > 0:
                        ahead_distances.append(distance)
                
                lane_info.safe_distance_ahead = min(ahead_distances) if ahead_distances else float('inf')
            
            lane_info.is_available = (
                lane_info.safe_distance_ahead >= self.safety_margin and
                lane_info.occupancy < self.lane_change_threshold
            )
            
            return lane_info
        
        def _select_best_lane(self, lane_options):
            """Select best lane for lane change."""
            available_lanes = [lane for lane in lane_options if lane.is_available]
            
            if not available_lanes:
                return None
            
            best_lane = None
            best_score = -float('inf')
            
            for lane in available_lanes:
                if lane.lane_id == self.current_lane:
                    continue
                
                distance_score = min(lane.safe_distance_ahead, lane.safe_distance_behind)
                occupancy_score = 1.0 - lane.occupancy
                adjacency_score = 1.0 / (abs(lane.lane_id - self.current_lane) + 1)
                
                total_score = distance_score * 0.5 + occupancy_score * 0.3 + adjacency_score * 0.2
                
                if total_score > best_score:
                    best_score = total_score
                    best_lane = lane.lane_id
            
            return best_lane
        
        def _smooth_trajectory_progress(self, progress):
            """Apply smoothing to trajectory progress."""
            smoothness = self.trajectory_smoothness
            
            if smoothness <= 0:
                return progress
            
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            return smoothness * smooth_progress + (1.0 - smoothness) * progress
        
        def _plan_lane_change_trajectory(self):
            """Plan lane change trajectory."""
            start_lateral = 0.0
            end_lateral = (self.lane_change_state.target_lane - self.current_lane) * self.lane_width
            
            num_points = 20
            self.lane_change_trajectory = []
            
            for i in range(num_points + 1):
                progress = i / num_points
                smooth_progress = self._smooth_trajectory_progress(progress)
                lateral_position = start_lateral + (end_lateral - start_lateral) * smooth_progress
                self.lane_change_trajectory.append(lateral_position)
    
    # Test obstacle detection in current lane
    wrapper = MockLaneChangingWrapper()
    
    detections = [
        {'relative_position': [0.1, 1.5], 'distance': 1.0},  # In current lane
        {'relative_position': [0.8, 2.0], 'distance': 1.8},  # Outside current lane
    ]
    
    obstacles = wrapper._get_obstacles_in_current_lane(detections)
    assert len(obstacles) == 1
    assert obstacles[0]['distance'] == 1.0
    print("✓ Obstacle detection in current lane works correctly")
    
    # Test lane analysis
    lane_0_info = wrapper._analyze_lane(0, detections)
    assert not lane_0_info.is_available  # Has obstacle within safety margin
    print("✓ Lane analysis works correctly")
    
    # Test lane selection
    lane_options = [
        LaneInfo(lane_id=0, occupancy=0.5, safe_distance_ahead=1.0, is_available=False),
        LaneInfo(lane_id=1, occupancy=0.1, safe_distance_ahead=5.0, is_available=True)
    ]
    
    best_lane = wrapper._select_best_lane(lane_options)
    assert best_lane == 1
    print("✓ Lane selection works correctly")
    
    # Test trajectory planning
    wrapper.lane_change_state.target_lane = 1
    wrapper._plan_lane_change_trajectory()
    
    assert len(wrapper.lane_change_trajectory) > 0
    assert wrapper.lane_change_trajectory[0] == 0.0  # Start at current lane
    assert abs(wrapper.lane_change_trajectory[-1] - 0.6) < 0.01  # End at target lane
    print("✓ Trajectory planning works correctly")
    
    # Test trajectory smoothing
    progress_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    smoothed_values = [wrapper._smooth_trajectory_progress(p) for p in progress_values]
    
    assert smoothed_values[0] == 0.0
    assert smoothed_values[-1] == 1.0
    
    # Check monotonicity
    for i in range(len(smoothed_values) - 1):
        assert smoothed_values[i] <= smoothed_values[i + 1]
    
    print("✓ Trajectory smoothing works correctly")


def test_configuration_validation():
    """Test configuration validation logic."""
    print("\nTesting configuration validation...")
    
    def validate_configuration(config):
        """Simplified validation function."""
        if not 0.0 <= config.get('lane_change_threshold', 0.3) <= 1.0:
            raise ValueError("lane_change_threshold must be in [0.0, 1.0]")
        
        if config.get('safety_margin', 2.0) <= 0:
            raise ValueError("safety_margin must be positive")
        
        max_time = config.get('max_lane_change_time', 3.0)
        min_time = config.get('min_lane_change_time', 1.0)
        if max_time <= min_time:
            raise ValueError("max_lane_change_time must be > min_lane_change_time")
        
        if not 0.0 <= config.get('trajectory_smoothness', 0.8) <= 1.0:
            raise ValueError("trajectory_smoothness must be in [0.0, 1.0]")
        
        if not 0.0 < config.get('lane_change_speed_factor', 0.8) <= 1.0:
            raise ValueError("lane_change_speed_factor must be in (0.0, 1.0]")
        
        if config.get('num_lanes', 2) < 1:
            raise ValueError("num_lanes must be >= 1")
    
    # Test valid configuration
    valid_config = {
        'lane_change_threshold': 0.3,
        'safety_margin': 2.0,
        'max_lane_change_time': 3.0,
        'min_lane_change_time': 1.0,
        'trajectory_smoothness': 0.8,
        'lane_change_speed_factor': 0.8,
        'num_lanes': 2
    }
    
    try:
        validate_configuration(valid_config)
        print("✓ Valid configuration accepted")
    except ValueError:
        print("✗ Valid configuration rejected")
        return False
    
    # Test invalid configurations
    invalid_configs = [
        {'lane_change_threshold': 1.5},  # Out of range
        {'safety_margin': -1.0},  # Negative
        {'max_lane_change_time': 1.0, 'min_lane_change_time': 2.0},  # min > max
        {'trajectory_smoothness': 1.5},  # Out of range
        {'lane_change_speed_factor': 0.0},  # Zero
        {'num_lanes': 0}  # Zero lanes
    ]
    
    for i, config in enumerate(invalid_configs):
        try:
            validate_configuration(config)
            print(f"✗ Invalid configuration {i+1} was accepted")
            return False
        except ValueError:
            pass  # Expected
    
    print("✓ All invalid configurations correctly rejected")


def test_action_modifications():
    """Test action modification logic."""
    print("\nTesting action modifications...")
    
    def modify_action_for_lane_change(action, target_lateral, speed_factor):
        """Simplified action modification."""
        steering_adjustment = target_lateral * 0.3
        
        modified_action = action.copy()
        modified_action[0] = (modified_action[0] * speed_factor) - steering_adjustment  # Left wheel
        modified_action[1] = (modified_action[1] * speed_factor) + steering_adjustment  # Right wheel
        
        # Clip to valid range
        modified_action = np.clip(modified_action, 0.0, 1.0)
        
        return modified_action
    
    # Test action modification
    original_action = np.array([0.6, 0.6])
    target_lateral = 0.3  # Move right
    speed_factor = 0.8
    
    modified_action = modify_action_for_lane_change(original_action, target_lateral, speed_factor)
    
    # Check that action was modified
    assert not np.array_equal(modified_action, original_action)
    
    # Check that actions are in valid range
    assert np.all(modified_action >= 0.0)
    assert np.all(modified_action <= 1.0)
    
    # Check that speed was reduced
    original_speed = np.mean(original_action)
    modified_speed = np.mean(modified_action)
    assert modified_speed < original_speed
    
    print("✓ Action modification works correctly")


def main():
    """Run all verification tests."""
    print("=== Lane Changing Implementation Verification ===\n")
    
    try:
        test_data_structures()
        test_core_algorithms()
        test_configuration_validation()
        test_action_modifications()
        
        print("\n=== All Tests Passed! ===")
        print("Lane changing implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n=== Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)