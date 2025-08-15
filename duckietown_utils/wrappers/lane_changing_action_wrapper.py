"""
Lane Changing Action Wrapper for Duckietown RL Environment.

This module implements a gym action wrapper that enables dynamic lane changing decisions
based on obstacles and traffic conditions. It uses a state machine approach with
comprehensive safety checks and trajectory planning.
"""

import logging
import math
import time
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

import gym
import numpy as np
from gym import spaces

from ..error_handling import (
    ErrorHandlingMixin, ErrorContext, ErrorSeverity, RecoveryStrategy,
    SafetyViolationError, ActionValidationError, SafetyOverrideSystem
)

logger = logging.getLogger(__name__)


class LaneChangePhase(Enum):
    """Lane change state machine phases."""
    LANE_FOLLOWING = "following"
    EVALUATING_CHANGE = "evaluating"
    INITIATING_CHANGE = "initiating"
    EXECUTING_CHANGE = "executing"


@dataclass
class LaneChangeState:
    """State information for lane changing process."""
    current_phase: LaneChangePhase = LaneChangePhase.LANE_FOLLOWING
    target_lane: Optional[int] = None
    progress: float = 0.0  # 0.0 to 1.0
    start_time: float = 0.0
    safety_checks: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.safety_checks is None:
            self.safety_checks = {}


@dataclass
class LaneInfo:
    """Information about a lane."""
    lane_id: int
    occupancy: float = 0.0  # 0.0 to 1.0
    safe_distance_ahead: float = float('inf')
    safe_distance_behind: float = float('inf')
    is_available: bool = True


class LaneChangingActionWrapper(ErrorHandlingMixin, gym.ActionWrapper):
    """
    Gym action wrapper that enables dynamic lane changing decisions.
    
    This wrapper implements a state machine-based approach to lane changing with
    comprehensive safety checks, trajectory planning, and fallback mechanisms.
    It evaluates lane occupancy, plans safe trajectories, and executes lane changes
    with proper timing constraints.
    """
    
    def __init__(
        self,
        env: gym.Env,
        lane_change_threshold: float = 0.3,
        safety_margin: float = 2.0,
        max_lane_change_time: float = 3.0,
        min_lane_change_time: float = 1.0,
        lane_width: float = 0.6,
        num_lanes: int = 2,
        evaluation_time: float = 0.5,
        trajectory_smoothness: float = 0.8,
        safety_check_frequency: float = 0.1,
        emergency_abort_distance: float = 0.3,
        lane_change_speed_factor: float = 0.8,
        enable_lane_changing: bool = True,
        debug_logging: bool = False
    ):
        """
        Initialize Lane Changing Action Wrapper.
        
        Args:
            env: Base gym environment to wrap
            lane_change_threshold: Threshold for initiating lane change (0.0-1.0)
            safety_margin: Required clear distance in target lane (meters)
            max_lane_change_time: Maximum time allowed for lane change (seconds)
            min_lane_change_time: Minimum time for lane change execution (seconds)
            lane_width: Width of each lane (meters)
            num_lanes: Number of available lanes
            evaluation_time: Time to spend evaluating lane change options (seconds)
            trajectory_smoothness: Smoothness factor for trajectory planning (0.0-1.0)
            safety_check_frequency: Frequency of safety checks during execution (seconds)
            emergency_abort_distance: Distance threshold for emergency abort (meters)
            lane_change_speed_factor: Speed reduction factor during lane change (0.0-1.0)
            enable_lane_changing: Whether lane changing is enabled
            debug_logging: Enable detailed debug logging
        """
        super().__init__(env)
        
        # Store configuration parameters
        self.lane_change_threshold = lane_change_threshold
        self.safety_margin = safety_margin
        self.max_lane_change_time = max_lane_change_time
        self.min_lane_change_time = min_lane_change_time
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.evaluation_time = evaluation_time
        self.trajectory_smoothness = trajectory_smoothness
        self.safety_check_frequency = safety_check_frequency
        self.emergency_abort_distance = emergency_abort_distance
        self.lane_change_speed_factor = lane_change_speed_factor
        self.enable_lane_changing = enable_lane_changing
        self.debug_logging = debug_logging
        
        # State tracking
        self.lane_change_state = LaneChangeState()
        self.current_lane = 0  # Assume starting in lane 0
        self.last_safety_check_time = 0.0
        self.evaluation_start_time = 0.0
        self.lane_change_trajectory = []
        self.blocked_lanes = set()
        
        # Statistics
        self.lane_change_stats = {
            'total_steps': 0,
            'lane_change_attempts': 0,
            'successful_lane_changes': 0,
            'aborted_lane_changes': 0,
            'emergency_aborts': 0,
            'avg_lane_change_time': 0.0,
            'safety_violations': 0
        }
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"LaneChangingActionWrapper initialized with {num_lanes} lanes")
    
    def _validate_configuration(self):
        """Validate wrapper configuration parameters."""
        if not 0.0 <= self.lane_change_threshold <= 1.0:
            raise ValueError(f"lane_change_threshold must be in [0.0, 1.0], got {self.lane_change_threshold}")
        
        if self.safety_margin <= 0:
            raise ValueError(f"safety_margin must be positive, got {self.safety_margin}")
        
        if self.max_lane_change_time <= self.min_lane_change_time:
            raise ValueError(f"max_lane_change_time must be > min_lane_change_time")
        
        if not 0.0 <= self.trajectory_smoothness <= 1.0:
            raise ValueError(f"trajectory_smoothness must be in [0.0, 1.0], got {self.trajectory_smoothness}")
        
        if not 0.0 < self.lane_change_speed_factor <= 1.0:
            raise ValueError(f"lane_change_speed_factor must be in (0.0, 1.0], got {self.lane_change_speed_factor}")
        
        if self.num_lanes < 1:
            raise ValueError(f"num_lanes must be >= 1, got {self.num_lanes}")
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Process action through lane changing state machine.
        
        Args:
            action: Original action from RL agent [left_wheel_vel, right_wheel_vel]
            
        Returns:
            Modified action with lane changing behavior
        """
        # Ensure action is numpy array
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        
        if not self.enable_lane_changing:
            return action
        
        # Get current observation for detection and lane data
        current_obs = self._get_current_observation()
        
        # Update lane change state machine
        self._update_state_machine(current_obs)
        
        # Apply lane changing modifications based on current phase
        modified_action = self._apply_lane_change_action(action, current_obs)
        
        # Update statistics
        self._update_stats()
        
        return modified_action
    
    def _get_current_observation(self) -> Union[Dict[str, Any], np.ndarray]:
        """
        Get current observation from environment.
        
        Note: This is a simplified approach. In practice, the observation should be
        passed through the action method or stored from the last step.
        """
        if hasattr(self.env, '_last_observation'):
            return self.env._last_observation
        else:
            return {}
    
    def _update_state_machine(self, observation: Union[Dict[str, Any], np.ndarray]):
        """
        Update the lane changing state machine based on current conditions.
        
        Args:
            observation: Current environment observation
        """
        current_time = time.time()
        
        if self.lane_change_state.current_phase == LaneChangePhase.LANE_FOLLOWING:
            self._handle_lane_following_phase(observation, current_time)
        
        elif self.lane_change_state.current_phase == LaneChangePhase.EVALUATING_CHANGE:
            self._handle_evaluating_phase(observation, current_time)
        
        elif self.lane_change_state.current_phase == LaneChangePhase.INITIATING_CHANGE:
            self._handle_initiating_phase(observation, current_time)
        
        elif self.lane_change_state.current_phase == LaneChangePhase.EXECUTING_CHANGE:
            self._handle_executing_phase(observation, current_time)
    
    def _handle_lane_following_phase(self, observation: Dict[str, Any], current_time: float):
        """Handle lane following phase logic."""
        # Check if lane change is needed due to obstacles
        if self._should_consider_lane_change(observation):
            if self.debug_logging:
                logger.debug("Obstacle detected, evaluating lane change options")
            
            self.lane_change_state.current_phase = LaneChangePhase.EVALUATING_CHANGE
            self.evaluation_start_time = current_time
            self.lane_change_stats['lane_change_attempts'] += 1
    
    def _handle_evaluating_phase(self, observation: Dict[str, Any], current_time: float):
        """Handle evaluation phase logic."""
        # Spend some time evaluating options
        if current_time - self.evaluation_start_time < self.evaluation_time:
            return
        
        # Evaluate available lanes
        lane_options = self._evaluate_lane_options(observation)
        
        # Select best lane option
        best_lane = self._select_best_lane(lane_options)
        
        if best_lane is not None and best_lane != self.current_lane:
            # Initiate lane change
            self.lane_change_state.target_lane = best_lane
            self.lane_change_state.current_phase = LaneChangePhase.INITIATING_CHANGE
            self.lane_change_state.start_time = current_time
            
            if self.debug_logging:
                logger.debug(f"Initiating lane change from lane {self.current_lane} to lane {best_lane}")
        else:
            # No suitable lane found, return to following
            if self.debug_logging:
                logger.debug("No suitable lane change option found, returning to lane following")
            
            self.lane_change_state.current_phase = LaneChangePhase.LANE_FOLLOWING
            self.lane_change_stats['aborted_lane_changes'] += 1
    
    def _handle_initiating_phase(self, observation: Dict[str, Any], current_time: float):
        """Handle initiation phase logic."""
        # Perform final safety checks
        if self._perform_final_safety_checks(observation):
            # Start executing lane change
            self.lane_change_state.current_phase = LaneChangePhase.EXECUTING_CHANGE
            self.lane_change_state.progress = 0.0
            self._plan_lane_change_trajectory()
            
            if self.debug_logging:
                logger.debug(f"Starting lane change execution to lane {self.lane_change_state.target_lane}")
        else:
            # Safety check failed, abort
            if self.debug_logging:
                logger.warning("Final safety check failed, aborting lane change")
            
            self._abort_lane_change()
    
    def _handle_executing_phase(self, observation: Dict[str, Any], current_time: float):
        """Handle execution phase logic."""
        elapsed_time = current_time - self.lane_change_state.start_time
        
        # Check for emergency abort conditions
        if self._should_emergency_abort(observation):
            if self.debug_logging:
                logger.warning("Emergency abort triggered during lane change")
            
            self._emergency_abort_lane_change()
            return
        
        # Check for timeout
        if elapsed_time > self.max_lane_change_time:
            if self.debug_logging:
                logger.warning("Lane change timeout, aborting")
            
            self._abort_lane_change()
            return
        
        # Update progress
        min_progress = elapsed_time / self.max_lane_change_time
        trajectory_progress = self._calculate_trajectory_progress()
        self.lane_change_state.progress = max(min_progress, trajectory_progress)
        
        # Check if lane change is complete
        if (self.lane_change_state.progress >= 1.0 and 
            elapsed_time >= self.min_lane_change_time):
            
            self._complete_lane_change()
    
    def _should_consider_lane_change(self, observation: Dict[str, Any]) -> bool:
        """
        Determine if a lane change should be considered based on current conditions.
        
        Args:
            observation: Current environment observation
            
        Returns:
            True if lane change should be considered
        """
        # Extract detection information
        detections = self._extract_detections(observation)
        
        # Check for obstacles in current lane
        obstacles_in_lane = self._get_obstacles_in_current_lane(detections)
        
        if not obstacles_in_lane:
            return False
        
        # Check if obstacles are close enough to warrant lane change
        for obstacle in obstacles_in_lane:
            distance = obstacle.get('distance', float('inf'))
            if distance <= self.safety_margin * 2:  # Consider lane change at 2x safety margin
                return True
        
        return False
    
    def _extract_detections(self, observation: Union[Dict[str, Any], np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract detection information from observation.
        
        Args:
            observation: Environment observation containing detection data
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            if isinstance(observation, dict):
                if 'detections' in observation:
                    detection_array = observation['detections']
                    detection_count = observation.get('detection_count', [0])[0]
                    
                    # Convert array format back to detection dictionaries
                    for i in range(min(detection_count, len(detection_array))):
                        det_data = detection_array[i]
                        if det_data[1] > 0:  # confidence > 0
                            detection = {
                                'class_id': int(det_data[0]),
                                'confidence': float(det_data[1]),
                                'bbox': [det_data[2], det_data[3], det_data[4], det_data[5]],
                                'relative_position': [det_data[6], det_data[7]],
                                'distance': float(det_data[8])
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Failed to extract detections: {str(e)}")
            return []
    
    def _get_obstacles_in_current_lane(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections to get obstacles in current lane.
        
        Args:
            detections: List of all detections
            
        Returns:
            List of obstacles in current lane
        """
        obstacles_in_lane = []
        
        for detection in detections:
            rel_pos = detection.get('relative_position', [0, 0])
            
            # Check if obstacle is in current lane (within lane width)
            lateral_offset = abs(rel_pos[0])
            if lateral_offset <= self.lane_width / 2:
                obstacles_in_lane.append(detection)
        
        return obstacles_in_lane
    
    def _evaluate_lane_options(self, observation: Dict[str, Any]) -> List[LaneInfo]:
        """
        Evaluate all available lane options.
        
        Args:
            observation: Current environment observation
            
        Returns:
            List of lane information for all lanes
        """
        lane_options = []
        detections = self._extract_detections(observation)
        
        for lane_id in range(self.num_lanes):
            lane_info = self._analyze_lane(lane_id, detections)
            lane_options.append(lane_info)
        
        return lane_options
    
    def _analyze_lane(self, lane_id: int, detections: List[Dict[str, Any]]) -> LaneInfo:
        """
        Analyze a specific lane for occupancy and safety.
        
        Args:
            lane_id: ID of the lane to analyze
            detections: List of all detections
            
        Returns:
            Lane information
        """
        lane_info = LaneInfo(lane_id=lane_id)
        
        # Calculate lane center position relative to current position
        lane_offset = (lane_id - self.current_lane) * self.lane_width
        
        # Check for obstacles in this lane
        obstacles_in_lane = []
        for detection in detections:
            rel_pos = detection.get('relative_position', [0, 0])
            
            # Adjust relative position for target lane
            adjusted_lateral = rel_pos[0] - lane_offset
            
            # Check if obstacle would be in target lane
            if abs(adjusted_lateral) <= self.lane_width / 2:
                obstacles_in_lane.append(detection)
        
        # Calculate occupancy and safety distances
        if obstacles_in_lane:
            lane_info.occupancy = len(obstacles_in_lane) / 10.0  # Normalize by max expected obstacles
            
            # Find closest obstacles ahead and behind
            ahead_distances = []
            behind_distances = []
            
            for obstacle in obstacles_in_lane:
                rel_pos = obstacle.get('relative_position', [0, 0])
                distance = obstacle.get('distance', float('inf'))
                
                if rel_pos[1] > 0:  # Ahead
                    ahead_distances.append(distance)
                else:  # Behind
                    behind_distances.append(distance)
            
            lane_info.safe_distance_ahead = min(ahead_distances) if ahead_distances else float('inf')
            lane_info.safe_distance_behind = min(behind_distances) if behind_distances else float('inf')
        
        # Determine if lane is available for lane change
        lane_info.is_available = (
            lane_info.safe_distance_ahead >= self.safety_margin and
            lane_info.safe_distance_behind >= self.safety_margin and
            lane_info.occupancy < self.lane_change_threshold
        )
        
        return lane_info
    
    def _select_best_lane(self, lane_options: List[LaneInfo]) -> Optional[int]:
        """
        Select the best lane option for lane change.
        
        Args:
            lane_options: List of analyzed lane options
            
        Returns:
            Best lane ID or None if no suitable option
        """
        # Filter to available lanes only
        available_lanes = [lane for lane in lane_options if lane.is_available]
        
        if not available_lanes:
            return None
        
        # Score lanes based on multiple criteria
        best_lane = None
        best_score = -float('inf')
        
        for lane in available_lanes:
            # Skip current lane
            if lane.lane_id == self.current_lane:
                continue
            
            # Calculate score based on:
            # 1. Distance to obstacles (higher is better)
            # 2. Lower occupancy (lower is better)
            # 3. Preference for adjacent lanes (closer is better)
            
            distance_score = min(lane.safe_distance_ahead, lane.safe_distance_behind)
            occupancy_score = 1.0 - lane.occupancy
            adjacency_score = 1.0 / (abs(lane.lane_id - self.current_lane) + 1)
            
            total_score = distance_score * 0.5 + occupancy_score * 0.3 + adjacency_score * 0.2
            
            if total_score > best_score:
                best_score = total_score
                best_lane = lane.lane_id
        
        return best_lane
    
    def _perform_final_safety_checks(self, observation: Dict[str, Any]) -> bool:
        """
        Perform final safety checks before executing lane change.
        
        Args:
            observation: Current environment observation
            
        Returns:
            True if safe to proceed with lane change
        """
        # Re-evaluate target lane
        detections = self._extract_detections(observation)
        target_lane_info = self._analyze_lane(self.lane_change_state.target_lane, detections)
        
        # Check if target lane is still available
        if not target_lane_info.is_available:
            return False
        
        # Additional safety checks
        safety_checks = {
            'target_lane_clear': target_lane_info.safe_distance_ahead >= self.safety_margin,
            'sufficient_space_behind': target_lane_info.safe_distance_behind >= self.safety_margin,
            'low_occupancy': target_lane_info.occupancy < self.lane_change_threshold,
            'no_emergency_obstacles': self._check_emergency_obstacles(detections)
        }
        
        self.lane_change_state.safety_checks = safety_checks
        
        # All checks must pass
        return all(safety_checks.values())
    
    def _check_emergency_obstacles(self, detections: List[Dict[str, Any]]) -> bool:
        """
        Check for emergency obstacles that would prevent lane change.
        
        Args:
            detections: List of all detections
            
        Returns:
            True if no emergency obstacles detected
        """
        for detection in detections:
            distance = detection.get('distance', float('inf'))
            rel_pos = detection.get('relative_position', [0, 0])
            
            # Check for very close obstacles
            if distance <= self.emergency_abort_distance:
                return False
            
            # Check for fast-approaching obstacles (simplified)
            if distance <= self.safety_margin / 2 and rel_pos[1] > 0:
                return False
        
        return True
    
    def _plan_lane_change_trajectory(self):
        """Plan the trajectory for lane change execution."""
        # Simple trajectory planning - linear interpolation between lanes
        start_lateral = 0.0  # Current lane center
        end_lateral = (self.lane_change_state.target_lane - self.current_lane) * self.lane_width
        
        # Create trajectory points
        num_points = 20
        self.lane_change_trajectory = []
        
        for i in range(num_points + 1):
            progress = i / num_points
            
            # Apply smoothing function (sigmoid-like)
            smooth_progress = self._smooth_trajectory_progress(progress)
            
            lateral_position = start_lateral + (end_lateral - start_lateral) * smooth_progress
            self.lane_change_trajectory.append(lateral_position)
    
    def _smooth_trajectory_progress(self, progress: float) -> float:
        """
        Apply smoothing to trajectory progress.
        
        Args:
            progress: Raw progress (0.0 to 1.0)
            
        Returns:
            Smoothed progress
        """
        # Use smoothstep function for smooth acceleration/deceleration
        smoothness = self.trajectory_smoothness
        
        if smoothness <= 0:
            return progress
        
        # Smoothstep: 3t² - 2t³
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        
        # Blend with linear progress based on smoothness factor
        return smoothness * smooth_progress + (1.0 - smoothness) * progress
    
    def _calculate_trajectory_progress(self) -> float:
        """
        Calculate current progress along planned trajectory.
        
        Returns:
            Progress value (0.0 to 1.0)
        """
        # Simplified progress calculation based on time
        elapsed_time = time.time() - self.lane_change_state.start_time
        time_progress = elapsed_time / self.max_lane_change_time
        
        return min(time_progress, 1.0)
    
    def _should_emergency_abort(self, observation: Dict[str, Any]) -> bool:
        """
        Check if emergency abort is needed during lane change execution.
        
        Args:
            observation: Current environment observation
            
        Returns:
            True if emergency abort is needed
        """
        detections = self._extract_detections(observation)
        
        # Check for emergency obstacles
        if not self._check_emergency_obstacles(detections):
            return True
        
        # Check if target lane became unavailable
        target_lane_info = self._analyze_lane(self.lane_change_state.target_lane, detections)
        if not target_lane_info.is_available:
            return True
        
        return False
    
    def _apply_lane_change_action(
        self, 
        action: np.ndarray, 
        observation: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply lane changing modifications to the action.
        
        Args:
            action: Original action from agent
            observation: Current environment observation
            
        Returns:
            Modified action with lane changing behavior
        """
        if self.lane_change_state.current_phase == LaneChangePhase.EXECUTING_CHANGE:
            return self._modify_action_for_lane_change(action)
        else:
            return action
    
    def _modify_action_for_lane_change(self, action: np.ndarray) -> np.ndarray:
        """
        Modify action during lane change execution.
        
        Args:
            action: Original action
            
        Returns:
            Modified action for lane change
        """
        # Get current trajectory target
        if not self.lane_change_trajectory:
            return action
        
        progress = self.lane_change_state.progress
        trajectory_index = int(progress * (len(self.lane_change_trajectory) - 1))
        trajectory_index = min(trajectory_index, len(self.lane_change_trajectory) - 1)
        
        target_lateral = self.lane_change_trajectory[trajectory_index]
        
        # Calculate steering adjustment needed
        # Positive target_lateral means move right (reduce left wheel, increase right wheel)
        steering_adjustment = target_lateral * 0.3  # Scale factor for steering sensitivity
        
        # Apply speed reduction during lane change
        speed_factor = self.lane_change_speed_factor
        
        # Modify action
        modified_action = action.copy()
        modified_action[0] = (modified_action[0] * speed_factor) - steering_adjustment  # Left wheel
        modified_action[1] = (modified_action[1] * speed_factor) + steering_adjustment  # Right wheel
        
        # Ensure actions remain in valid range
        modified_action = np.clip(modified_action, 0.0, 1.0)
        
        return modified_action
    
    def _complete_lane_change(self):
        """Complete the lane change and update state."""
        if self.debug_logging:
            logger.info(f"Lane change completed: {self.current_lane} -> {self.lane_change_state.target_lane}")
        
        # Update current lane
        self.current_lane = self.lane_change_state.target_lane
        
        # Update statistics
        self.lane_change_stats['successful_lane_changes'] += 1
        
        elapsed_time = time.time() - self.lane_change_state.start_time
        total_changes = self.lane_change_stats['successful_lane_changes']
        current_avg = self.lane_change_stats['avg_lane_change_time']
        self.lane_change_stats['avg_lane_change_time'] = (
            (current_avg * (total_changes - 1) + elapsed_time) / total_changes
        )
        
        # Reset state
        self._reset_lane_change_state()
    
    def _abort_lane_change(self):
        """Abort the current lane change attempt."""
        if self.debug_logging:
            logger.warning("Lane change aborted")
        
        self.lane_change_stats['aborted_lane_changes'] += 1
        self._reset_lane_change_state()
    
    def _emergency_abort_lane_change(self):
        """Emergency abort the current lane change."""
        if self.debug_logging:
            logger.error("Emergency abort of lane change")
        
        self.lane_change_stats['emergency_aborts'] += 1
        self._reset_lane_change_state()
    
    def _reset_lane_change_state(self):
        """Reset lane change state to lane following."""
        self.lane_change_state = LaneChangeState()
        self.lane_change_trajectory = []
    
    def _update_stats(self):
        """Update lane changing statistics."""
        self.lane_change_stats['total_steps'] += 1
    
    def get_lane_change_stats(self) -> Dict[str, Any]:
        """
        Get lane changing performance statistics.
        
        Returns:
            Dictionary with lane changing statistics
        """
        stats = self.lane_change_stats.copy()
        
        # Calculate derived statistics
        total_attempts = stats['lane_change_attempts']
        if total_attempts > 0:
            stats['success_rate'] = stats['successful_lane_changes'] / total_attempts
            stats['abort_rate'] = stats['aborted_lane_changes'] / total_attempts
            stats['emergency_abort_rate'] = stats['emergency_aborts'] / total_attempts
        else:
            stats['success_rate'] = 0.0
            stats['abort_rate'] = 0.0
            stats['emergency_abort_rate'] = 0.0
        
        # Add current state information
        stats['current_phase'] = self.lane_change_state.current_phase.value
        stats['current_lane'] = self.current_lane
        stats['target_lane'] = self.lane_change_state.target_lane
        stats['lane_change_progress'] = self.lane_change_state.progress
        
        return stats
    
    def get_current_lane(self) -> int:
        """Get current lane number."""
        return self.current_lane
    
    def get_lane_change_phase(self) -> LaneChangePhase:
        """Get current lane change phase."""
        return self.lane_change_state.current_phase
    
    def is_lane_changing(self) -> bool:
        """Check if currently performing a lane change."""
        return self.lane_change_state.current_phase in [
            LaneChangePhase.INITIATING_CHANGE,
            LaneChangePhase.EXECUTING_CHANGE
        ]
    
    def force_lane_change(self, target_lane: int) -> bool:
        """
        Force a lane change to the specified lane (for testing/debugging).
        
        Args:
            target_lane: Target lane number
            
        Returns:
            True if lane change was initiated
        """
        if not 0 <= target_lane < self.num_lanes:
            return False
        
        if target_lane == self.current_lane:
            return False
        
        if self.is_lane_changing():
            return False
        
        # Force initiate lane change
        self.lane_change_state.target_lane = target_lane
        self.lane_change_state.current_phase = LaneChangePhase.INITIATING_CHANGE
        self.lane_change_state.start_time = time.time()
        
        logger.info(f"Forced lane change initiated: {self.current_lane} -> {target_lane}")
        return True
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current wrapper configuration."""
        return {
            'lane_change_threshold': self.lane_change_threshold,
            'safety_margin': self.safety_margin,
            'max_lane_change_time': self.max_lane_change_time,
            'min_lane_change_time': self.min_lane_change_time,
            'lane_width': self.lane_width,
            'num_lanes': self.num_lanes,
            'evaluation_time': self.evaluation_time,
            'trajectory_smoothness': self.trajectory_smoothness,
            'safety_check_frequency': self.safety_check_frequency,
            'emergency_abort_distance': self.emergency_abort_distance,
            'lane_change_speed_factor': self.lane_change_speed_factor,
            'enable_lane_changing': self.enable_lane_changing,
            'debug_logging': self.debug_logging
        }
    
    def update_configuration(self, **kwargs):
        """
        Update wrapper configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        # Re-validate configuration
        self._validate_configuration()
    
    def reset(self, **kwargs):
        """Reset wrapper state for new episode."""
        # Reset lane change state
        self._reset_lane_change_state()
        self.current_lane = 0  # Reset to starting lane
        self.last_safety_check_time = 0.0
        self.evaluation_start_time = 0.0
        self.blocked_lanes = set()
        
        # Reset base environment
        return self.env.reset(**kwargs)