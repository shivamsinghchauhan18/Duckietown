"""
Dynamic Lane Changing Wrapper for Duckietown

This wrapper implements dynamic lane changing logic that can be triggered by
object detection or navigation requirements. It works with the existing
lane following system to provide smooth lane transitions.

Authors: Generated for Dynamic Lane Changing and Object Avoidance
License: MIT
"""

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    print("Warning: gym not available, using minimal gym interface")
    GYM_AVAILABLE = False
    
    # Import minimal gym interface from yolo_detection_wrapper
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    try:
        from yolo_detection_wrapper import gym
    except ImportError:
        # Create minimal interface if not available
        class ActionWrapper:
            def __init__(self, env):
                self.env = env
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            
            def reset(self, **kwargs):
                return self.env.reset(**kwargs)
            
            def action(self, action):
                return action
            
            def step(self, action):
                return self.env.step(self.action(action))
        
        class gym:
            ActionWrapper = ActionWrapper

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LaneChangeState(Enum):
    """States for the lane changing state machine"""
    LANE_FOLLOWING = "lane_following"
    LANE_CHANGE_INITIATED = "lane_change_initiated"
    LANE_CHANGING = "lane_changing"
    LANE_CHANGE_COMPLETING = "lane_change_completing"
    LANE_CHANGE_COMPLETE = "lane_change_complete"


class LaneChangeDirection(Enum):
    """Direction for lane changes"""
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class DynamicLaneChangingWrapper(gym.ActionWrapper):
    """
    Wrapper that implements dynamic lane changing capabilities
    
    This wrapper:
    - Monitors for lane change triggers (from object detection or manual)
    - Implements a state machine for smooth lane transitions
    - Modifies actions to execute lane changes
    - Provides lane change status and progress information
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        """
        Initialize Dynamic Lane Changing Wrapper
        
        Args:
            env: Gym environment to wrap
            config: Configuration dictionary with lane changing settings
        """
        super().__init__(env)
        
        # Configuration with defaults
        self.config = config or {}
        self.lane_change_duration = self.config.get('lane_change_duration', 60)  # steps
        self.lane_change_intensity = self.config.get('lane_change_intensity', 0.7)  # steering intensity
        self.safety_check_enabled = self.config.get('safety_check_enabled', True)
        self.auto_trigger_enabled = self.config.get('auto_trigger_enabled', True)
        self.min_distance_for_lane_change = self.config.get('min_distance_for_lane_change', 0.5)
        
        # State management
        self.current_state = LaneChangeState.LANE_FOLLOWING
        self.target_direction = LaneChangeDirection.NONE
        self.lane_change_progress = 0
        self.lane_change_step_count = 0
        
        # Trigger management
        self.lane_change_requested = False
        self.requested_direction = LaneChangeDirection.NONE
        self.trigger_source = None  # 'manual', 'object_avoidance', etc.
        
        # Lane tracking
        self.current_lane_offset = 0.0  # Offset from lane center (-1 to 1)
        self.target_lane_offset = 0.0
        self.lane_change_start_offset = 0.0
        
        # Safety and performance tracking
        self.lane_changes_completed = 0
        self.lane_changes_aborted = 0
        self.last_lane_change_step = 0
        
        logger.info("DynamicLaneChangingWrapper initialized")
    
    def reset(self, **kwargs):
        """Reset environment and lane changing state"""
        obs = self.env.reset(**kwargs)
        self._reset_lane_change_state()
        return obs
    
    def action(self, action):
        """
        Modify action to implement lane changing behavior
        
        Args:
            action: Original action from the policy
            
        Returns:
            Modified action that includes lane changing maneuvers
        """
        # Update state machine
        self._update_state_machine()
        
        # Modify action based on current state
        modified_action = self._modify_action_for_lane_change(action)
        
        return modified_action
    
    def step(self, action):
        """Step environment with lane changing logic"""
        obs, reward, done, info = self.env.step(action)
        
        # Update lane tracking based on environment state
        self._update_lane_tracking(info)
        
        # Check for automatic lane change triggers
        if self.auto_trigger_enabled:
            self._check_auto_triggers(info)
        
        # Add lane changing information to info
        info['lane_changing'] = self._get_lane_change_info()
        
        return obs, reward, done, info
    
    def _reset_lane_change_state(self):
        """Reset all lane changing state variables"""
        self.current_state = LaneChangeState.LANE_FOLLOWING
        self.target_direction = LaneChangeDirection.NONE
        self.lane_change_progress = 0
        self.lane_change_step_count = 0
        self.lane_change_requested = False
        self.requested_direction = LaneChangeDirection.NONE
        self.trigger_source = None
        self.current_lane_offset = 0.0
        self.target_lane_offset = 0.0
        self.lane_change_start_offset = 0.0
    
    def _update_state_machine(self):
        """Update the lane changing state machine"""
        if self.current_state == LaneChangeState.LANE_FOLLOWING:
            if self.lane_change_requested:
                self._initiate_lane_change()
        
        elif self.current_state == LaneChangeState.LANE_CHANGE_INITIATED:
            if self._safety_check_passed():
                self.current_state = LaneChangeState.LANE_CHANGING
                self.lane_change_step_count = 0
                logger.info(f"Lane change started: {self.target_direction.value}")
            else:
                # Safety check failed, abort
                self._abort_lane_change()
        
        elif self.current_state == LaneChangeState.LANE_CHANGING:
            self.lane_change_step_count += 1
            self.lane_change_progress = min(1.0, self.lane_change_step_count / self.lane_change_duration)
            
            if self.lane_change_progress >= 0.8:  # Start completing phase at 80%
                self.current_state = LaneChangeState.LANE_CHANGE_COMPLETING
        
        elif self.current_state == LaneChangeState.LANE_CHANGE_COMPLETING:
            self.lane_change_step_count += 1
            self.lane_change_progress = min(1.0, self.lane_change_step_count / self.lane_change_duration)
            
            if self.lane_change_progress >= 1.0:
                self._complete_lane_change()
    
    def _initiate_lane_change(self):
        """Initiate a lane change maneuver"""
        self.current_state = LaneChangeState.LANE_CHANGE_INITIATED
        self.target_direction = self.requested_direction
        self.lane_change_start_offset = self.current_lane_offset
        
        # Set target offset based on direction
        if self.target_direction == LaneChangeDirection.LEFT:
            self.target_lane_offset = self.current_lane_offset - 1.0  # Move one lane left
        elif self.target_direction == LaneChangeDirection.RIGHT:
            self.target_lane_offset = self.current_lane_offset + 1.0  # Move one lane right
        
        # Clear the request
        self.lane_change_requested = False
        
        logger.info(f"Lane change initiated: {self.target_direction.value} from offset {self.current_lane_offset}")
    
    def _complete_lane_change(self):
        """Complete the lane change and return to lane following"""
        self.current_state = LaneChangeState.LANE_FOLLOWING
        self.current_lane_offset = self.target_lane_offset
        self.lane_change_progress = 0
        self.lane_change_step_count = 0
        self.target_direction = LaneChangeDirection.NONE
        self.lane_changes_completed += 1
        self.last_lane_change_step = 0
        
        logger.info(f"Lane change completed. New lane offset: {self.current_lane_offset}")
    
    def _abort_lane_change(self):
        """Abort the current lane change attempt"""
        self.current_state = LaneChangeState.LANE_FOLLOWING
        self.target_direction = LaneChangeDirection.NONE
        self.lane_change_requested = False
        self.lane_changes_aborted += 1
        
        logger.warning("Lane change aborted due to safety check failure")
    
    def _safety_check_passed(self) -> bool:
        """
        Perform safety checks before executing lane change
        
        Returns:
            True if it's safe to change lanes, False otherwise
        """
        if not self.safety_check_enabled:
            return True
        
        # Basic safety checks - can be extended with more sophisticated logic
        
        # Check if enough time has passed since last lane change
        if self.last_lane_change_step < 30:  # Minimum 30 steps between lane changes
            return False
        
        # Check if we're not already at the edge of the road
        if self.target_direction == LaneChangeDirection.LEFT and self.current_lane_offset <= -1.5:
            return False
        if self.target_direction == LaneChangeDirection.RIGHT and self.current_lane_offset >= 1.5:
            return False
        
        # Additional safety checks can be added here
        # - Check for obstacles in target lane
        # - Check speed and stability
        # - Check road geometry
        
        return True
    
    def _modify_action_for_lane_change(self, action):
        """
        Modify the action to implement lane changing maneuver
        
        Args:
            action: Original action [steering, speed] or similar format
            
        Returns:
            Modified action with lane changing steering input
        """
        if self.current_state == LaneChangeState.LANE_FOLLOWING:
            return action
        
        # Calculate lane change steering adjustment
        steering_adjustment = 0.0
        
        if self.current_state in [LaneChangeState.LANE_CHANGING, LaneChangeState.LANE_CHANGE_COMPLETING]:
            # Calculate smooth steering profile for lane change
            progress = self.lane_change_progress
            
            # Use a smooth S-curve for steering (sine wave)
            steering_profile = np.sin(progress * np.pi)  # 0 to 1 and back to 0
            
            # Determine steering direction
            if self.target_direction == LaneChangeDirection.LEFT:
                steering_adjustment = -self.lane_change_intensity * steering_profile
            elif self.target_direction == LaneChangeDirection.RIGHT:
                steering_adjustment = self.lane_change_intensity * steering_profile
        
        # Apply steering adjustment to action
        modified_action = action.copy() if hasattr(action, 'copy') else np.array(action)
        
        # Assume first element is steering (common in Duckietown)
        if len(modified_action) >= 1:
            modified_action[0] = np.clip(modified_action[0] + steering_adjustment, -1.0, 1.0)
        
        return modified_action
    
    def _update_lane_tracking(self, info: Dict):
        """
        Update lane tracking information based on environment feedback
        
        Args:
            info: Info dictionary from environment step
        """
        # This is a placeholder - in a real implementation, you would
        # extract lane position information from the environment or
        # computer vision analysis
        
        # For now, we'll estimate based on our lane change progress
        if self.current_state in [LaneChangeState.LANE_CHANGING, LaneChangeState.LANE_CHANGE_COMPLETING]:
            # Interpolate between start and target offset
            self.current_lane_offset = (
                self.lane_change_start_offset + 
                (self.target_lane_offset - self.lane_change_start_offset) * self.lane_change_progress
            )
        
        # Update step counter for safety checks
        self.last_lane_change_step += 1
    
    def _check_auto_triggers(self, info: Dict):
        """
        Check for automatic lane change triggers from object detection or other sources
        
        Args:
            info: Info dictionary from environment step
        """
        # Check for object avoidance trigger
        if 'object_avoidance' in info:
            avoidance_info = info['object_avoidance']
            if avoidance_info.get('lane_change_recommended', False):
                # Determine best direction based on object positions
                direction = self._determine_best_lane_change_direction(avoidance_info)
                if direction != LaneChangeDirection.NONE:
                    self.trigger_lane_change(direction, 'object_avoidance')
        
        # Add other trigger conditions here
        # - Navigation waypoints requiring lane changes
        # - Traffic flow optimization
        # - Road signs or lane markings
    
    def _determine_best_lane_change_direction(self, avoidance_info: Dict) -> LaneChangeDirection:
        """
        Determine the best direction for lane change based on object avoidance info
        
        Args:
            avoidance_info: Object avoidance information
            
        Returns:
            Recommended lane change direction
        """
        critical_objects = avoidance_info.get('critical_objects', [])
        
        if not critical_objects:
            return LaneChangeDirection.NONE
        
        # Count objects by position
        left_objects = sum(1 for obj in critical_objects if obj['position'] == 'left')
        right_objects = sum(1 for obj in critical_objects if obj['position'] == 'right')
        center_objects = sum(1 for obj in critical_objects if obj['position'] == 'center')
        
        # If there are center objects, we need to change lanes
        if center_objects > 0:
            # Choose direction with fewer objects
            if left_objects < right_objects:
                return LaneChangeDirection.LEFT
            elif right_objects < left_objects:
                return LaneChangeDirection.RIGHT
            else:
                # Equal objects or no preference - choose based on current lane position
                if self.current_lane_offset > 0:  # Currently right of center
                    return LaneChangeDirection.LEFT
                else:  # Currently left of center or centered
                    return LaneChangeDirection.RIGHT
        
        return LaneChangeDirection.NONE
    
    def trigger_lane_change(self, direction: LaneChangeDirection, source: str = 'manual'):
        """
        Trigger a lane change manually
        
        Args:
            direction: Direction to change lanes
            source: Source of the trigger ('manual', 'object_avoidance', etc.)
        """
        if self.current_state == LaneChangeState.LANE_FOLLOWING and not self.lane_change_requested:
            self.lane_change_requested = True
            self.requested_direction = direction
            self.trigger_source = source
            logger.info(f"Lane change triggered: {direction.value} (source: {source})")
        else:
            logger.warning(f"Lane change request ignored - current state: {self.current_state}")
    
    def _get_lane_change_info(self) -> Dict:
        """Get current lane changing status information"""
        return {
            'state': self.current_state.value,
            'target_direction': self.target_direction.value,
            'progress': self.lane_change_progress,
            'current_lane_offset': self.current_lane_offset,
            'target_lane_offset': self.target_lane_offset,
            'lane_changes_completed': self.lane_changes_completed,
            'lane_changes_aborted': self.lane_changes_aborted,
            'trigger_source': self.trigger_source,
            'is_changing_lanes': self.current_state in [
                LaneChangeState.LANE_CHANGING, 
                LaneChangeState.LANE_CHANGE_COMPLETING
            ]
        }
    
    def get_lane_change_statistics(self) -> Dict:
        """Get lane changing performance statistics"""
        total_attempts = self.lane_changes_completed + self.lane_changes_aborted
        success_rate = self.lane_changes_completed / total_attempts if total_attempts > 0 else 0
        
        return {
            'total_lane_changes_completed': self.lane_changes_completed,
            'total_lane_changes_aborted': self.lane_changes_aborted,
            'total_attempts': total_attempts,
            'success_rate': success_rate,
            'current_state': self.current_state.value,
            'current_lane_offset': self.current_lane_offset
        }