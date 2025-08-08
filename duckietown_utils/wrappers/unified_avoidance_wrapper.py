"""
Unified Dynamic Lane Changing and Object Avoidance Wrapper for Duckietown

This wrapper combines YOLO v5 object detection with dynamic lane changing
to create an integrated object avoidance system for Duckietown.

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
        class Wrapper:
            def __init__(self, env):
                self.env = env
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            
            def reset(self, **kwargs):
                return self.env.reset(**kwargs)
            
            def step(self, action):
                return self.env.step(action)
        
        class gym:
            Wrapper = Wrapper

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

try:
    from .yolo_detection_wrapper import YOLOObjectAvoidanceWrapper
    from .lane_changing_wrapper import DynamicLaneChangingWrapper, LaneChangeDirection
except ImportError:
    # Fallback for direct import
    try:
        from yolo_detection_wrapper import YOLOObjectAvoidanceWrapper
        from lane_changing_wrapper import DynamicLaneChangingWrapper, LaneChangeDirection
    except ImportError:
        print("Warning: Could not import other wrappers for UnifiedAvoidanceWrapper")
        # Create dummy classes for graceful degradation
        class DummyWrapper:
            def __init__(self, env, config=None):
                self.env = env
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            
            def reset(self, **kwargs):
                return self.env.reset(**kwargs)
            
            def step(self, action):
                return self.env.step(action)
        
        YOLOObjectAvoidanceWrapper = DummyWrapper
        DynamicLaneChangingWrapper = DummyWrapper
        
        class LaneChangeDirection:
            LEFT = "left"
            RIGHT = "right"
            NONE = "none"

logger = logging.getLogger(__name__)


class UnifiedAvoidanceWrapper(gym.Wrapper):
    """
    Unified wrapper that combines YOLO object detection with dynamic lane changing
    for comprehensive object avoidance in Duckietown.
    
    This wrapper:
    - Performs real-time object detection using YOLO v5
    - Analyzes objects for collision risk
    - Triggers appropriate avoidance maneuvers (lane changes, speed adjustments)
    - Coordinates between detection and lane changing systems
    - Provides comprehensive avoidance status and metrics
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        """
        Initialize Unified Avoidance Wrapper
        
        Args:
            env: Gym environment to wrap
            config: Configuration dictionary with all system settings
        """
        # Configuration with defaults
        self.config = config or {}
        
        # Split config for sub-components
        yolo_config = self.config.get('yolo', {})
        lane_change_config = self.config.get('lane_changing', {})
        
        # Initialize YOLO object detection wrapper
        env_with_yolo = YOLOObjectAvoidanceWrapper(env, yolo_config)
        
        # Initialize lane changing wrapper  
        env_with_lane_changing = DynamicLaneChangingWrapper(env_with_yolo, lane_change_config)
        
        super().__init__(env_with_lane_changing)
        
        # Unified system parameters
        self.emergency_brake_threshold = self.config.get('emergency_brake_threshold', 0.2)
        self.speed_reduction_factor = self.config.get('speed_reduction_factor', 0.5)
        self.avoidance_priority = self.config.get('avoidance_priority', ['lane_change', 'speed_reduction', 'emergency_brake'])
        self.reaction_time_steps = self.config.get('reaction_time_steps', 3)
        
        # System state
        self.avoidance_mode = 'normal'  # 'normal', 'cautious', 'emergency'
        self.consecutive_detections = 0
        self.avoidance_history = []
        self.performance_metrics = {
            'total_avoidances': 0,
            'successful_lane_changes': 0,
            'emergency_brakes': 0,
            'near_misses': 0
        }
        
        logger.info("UnifiedAvoidanceWrapper initialized")
    
    def reset(self, **kwargs):
        """Reset environment and all avoidance systems"""
        obs = self.env.reset(**kwargs)
        self._reset_avoidance_state()
        return obs
    
    def step(self, action):
        """Step with unified object avoidance system"""
        # Get action from the lane changing wrapper (which includes YOLO detection)
        obs, reward, done, info = self.env.step(action)
        
        # Perform unified avoidance analysis
        avoidance_decision = self._analyze_unified_avoidance(info)
        
        # Apply avoidance modifications
        modified_action = self._apply_avoidance_decision(action, avoidance_decision)
        
        # Update system state and metrics
        self._update_avoidance_state(avoidance_decision, info)
        
        # Add unified avoidance info
        info['unified_avoidance'] = {
            'decision': avoidance_decision,
            'mode': self.avoidance_mode,
            'consecutive_detections': self.consecutive_detections,
            'performance_metrics': self.performance_metrics.copy(),
            'action_modified': not np.array_equal(action, modified_action)
        }
        
        return obs, reward, done, info
    
    def _reset_avoidance_state(self):
        """Reset all avoidance system state"""
        self.avoidance_mode = 'normal'
        self.consecutive_detections = 0
        self.avoidance_history = []
    
    def _analyze_unified_avoidance(self, info: Dict) -> Dict:
        """
        Analyze all available information to make unified avoidance decisions
        
        Args:
            info: Environment info dictionary with detection and lane changing data
            
        Returns:
            Unified avoidance decision dictionary
        """
        yolo_info = info.get('yolo_detections', {})
        object_avoidance_info = info.get('object_avoidance', {})
        lane_changing_info = info.get('lane_changing', {})
        
        detections = yolo_info.get('detections', [])
        critical_objects = object_avoidance_info.get('critical_objects', [])
        closest_distance = object_avoidance_info.get('closest_object_distance', float('inf'))
        
        # Determine threat level
        threat_level = self._assess_threat_level(critical_objects, closest_distance)
        
        # Make avoidance decision based on threat level and system capabilities
        avoidance_actions = []
        
        if threat_level == 'emergency':
            avoidance_actions.append('emergency_brake')
            self.avoidance_mode = 'emergency'
            
        elif threat_level == 'high':
            self.avoidance_mode = 'cautious'
            
            # Check if lane change is possible and safe
            if self._can_perform_lane_change(lane_changing_info, critical_objects):
                avoidance_actions.append('lane_change')
            else:
                avoidance_actions.append('speed_reduction')
                
        elif threat_level == 'medium':
            self.avoidance_mode = 'cautious'
            avoidance_actions.append('speed_reduction')
            
        else:
            self.avoidance_mode = 'normal'
        
        # Determine specific parameters for each action
        decision = {
            'threat_level': threat_level,
            'actions': avoidance_actions,
            'lane_change_direction': self._determine_optimal_lane_change(critical_objects),
            'speed_factor': self._calculate_speed_factor(threat_level, closest_distance),
            'emergency_brake': 'emergency_brake' in avoidance_actions,
            'reasoning': self._generate_decision_reasoning(threat_level, critical_objects, lane_changing_info)
        }
        
        return decision
    
    def _assess_threat_level(self, critical_objects: List[Dict], closest_distance: float) -> str:
        """
        Assess the threat level based on detected objects
        
        Args:
            critical_objects: List of critical object detections
            closest_distance: Distance to closest object
            
        Returns:
            Threat level: 'none', 'low', 'medium', 'high', 'emergency'
        """
        if not critical_objects:
            return 'none'
        
        # Count consecutive detections for temporal consistency
        if critical_objects:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0
        
        # Assess based on distance and object characteristics
        if closest_distance < self.emergency_brake_threshold:
            return 'emergency'
        elif closest_distance < 0.4 and self.consecutive_detections >= self.reaction_time_steps:
            return 'high'
        elif closest_distance < 0.6 and self.consecutive_detections >= 2:
            return 'medium'
        elif closest_distance < 0.8:
            return 'low'
        else:
            return 'none'
    
    def _can_perform_lane_change(self, lane_changing_info: Dict, critical_objects: List[Dict]) -> bool:
        """
        Determine if a lane change is possible and safe
        
        Args:
            lane_changing_info: Current lane changing status
            critical_objects: List of critical objects
            
        Returns:
            True if lane change is safe and possible
        """
        # Check if lane changing system is ready
        if lane_changing_info.get('state') != 'lane_following':
            return False
        
        # Check if we have room to change lanes (simplified check)
        left_clear = True
        right_clear = True
        
        for obj in critical_objects:
            if obj['position'] == 'left':
                left_clear = False
            elif obj['position'] == 'right':
                right_clear = False
        
        # At least one direction should be clear
        return left_clear or right_clear
    
    def _determine_optimal_lane_change(self, critical_objects: List[Dict]) -> LaneChangeDirection:
        """
        Determine the optimal lane change direction based on object positions
        
        Args:
            critical_objects: List of critical objects
            
        Returns:
            Optimal lane change direction
        """
        if not critical_objects:
            return LaneChangeDirection.NONE
        
        # Count objects in each direction
        left_count = sum(1 for obj in critical_objects if obj['position'] == 'left')
        right_count = sum(1 for obj in critical_objects if obj['position'] == 'right')
        center_count = sum(1 for obj in critical_objects if obj['position'] == 'center')
        
        # If center is blocked, choose the clearer side
        if center_count > 0:
            if left_count < right_count:
                return LaneChangeDirection.LEFT
            elif right_count < left_count:
                return LaneChangeDirection.RIGHT
            else:
                # Default to right in case of tie
                return LaneChangeDirection.RIGHT
        
        return LaneChangeDirection.NONE
    
    def _calculate_speed_factor(self, threat_level: str, closest_distance: float) -> float:
        """
        Calculate speed reduction factor based on threat level
        
        Args:
            threat_level: Current threat level
            closest_distance: Distance to closest object
            
        Returns:
            Speed factor (0.0 to 1.0)
        """
        if threat_level == 'emergency':
            return 0.0  # Full stop
        elif threat_level == 'high':
            return max(0.3, closest_distance)  # Significant reduction
        elif threat_level == 'medium':
            return max(0.6, closest_distance * 1.2)  # Moderate reduction
        elif threat_level == 'low':
            return max(0.8, closest_distance * 1.5)  # Light reduction
        else:
            return 1.0  # No reduction
    
    def _generate_decision_reasoning(self, threat_level: str, critical_objects: List[Dict], 
                                   lane_changing_info: Dict) -> str:
        """Generate human-readable reasoning for the avoidance decision"""
        if threat_level == 'none':
            return "No threats detected, continuing normal operation"
        
        obj_count = len(critical_objects)
        lane_state = lane_changing_info.get('state', 'unknown')
        
        reasoning = f"Threat level: {threat_level}. "
        reasoning += f"Detected {obj_count} critical object(s). "
        reasoning += f"Lane changing system state: {lane_state}. "
        
        if threat_level == 'emergency':
            reasoning += "Emergency braking required due to imminent collision risk."
        elif threat_level == 'high':
            reasoning += "High risk situation requiring immediate avoidance action."
        elif threat_level == 'medium':
            reasoning += "Moderate risk requiring speed reduction and monitoring."
        else:
            reasoning += "Low risk situation with preventive measures."
        
        return reasoning
    
    def _apply_avoidance_decision(self, original_action, decision: Dict):
        """
        Apply avoidance decision to modify the action
        
        Args:
            original_action: Original action from the policy
            decision: Avoidance decision dictionary
            
        Returns:
            Modified action with avoidance behaviors
        """
        modified_action = np.array(original_action, copy=True)
        
        # Apply speed modifications
        if len(modified_action) >= 2:  # Assume [steering, speed] format
            modified_action[1] *= decision['speed_factor']
        
        # Apply emergency brake
        if decision['emergency_brake']:
            if len(modified_action) >= 2:
                modified_action[1] = 0.0  # Full stop
        
        # Lane change trigger (handled by lane changing wrapper)
        if 'lane_change' in decision['actions']:
            direction = decision['lane_change_direction']
            if hasattr(self.env, 'trigger_lane_change'):
                self.env.trigger_lane_change(direction, 'unified_avoidance')
        
        return modified_action
    
    def _update_avoidance_state(self, decision: Dict, info: Dict):
        """Update avoidance system state and metrics"""
        # Update performance metrics
        if 'lane_change' in decision['actions']:
            self.performance_metrics['total_avoidances'] += 1
            
        if decision['emergency_brake']:
            self.performance_metrics['emergency_brakes'] += 1
            
        # Track lane change success (from lane changing wrapper)
        lane_info = info.get('lane_changing', {})
        if lane_info.get('state') == 'lane_change_complete':
            self.performance_metrics['successful_lane_changes'] += 1
        
        # Add to history
        self.avoidance_history.append({
            'step': len(self.avoidance_history),
            'decision': decision.copy(),
            'threat_level': decision['threat_level'],
            'actions_taken': decision['actions'].copy()
        })
        
        # Limit history size
        if len(self.avoidance_history) > 1000:
            self.avoidance_history.pop(0)
    
    def get_avoidance_statistics(self) -> Dict:
        """Get comprehensive avoidance system statistics"""
        total_decisions = len(self.avoidance_history)
        
        if total_decisions == 0:
            return {'total_decisions': 0}
        
        # Count decisions by threat level
        threat_counts = {}
        action_counts = {}
        
        for entry in self.avoidance_history:
            threat_level = entry['threat_level']
            threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
            
            for action in entry['actions_taken']:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'threat_level_distribution': threat_counts,
            'action_distribution': action_counts,
            'performance_metrics': self.performance_metrics.copy(),
            'current_mode': self.avoidance_mode,
            'consecutive_detections': self.consecutive_detections
        }