"""
Object Avoidance Action Wrapper for Duckietown RL Environment.

This module implements a gym action wrapper that modifies actions to avoid detected objects
using a potential field-based algorithm while maintaining lane following behavior.
The wrapper provides smooth action modifications and priority-based avoidance for multiple objects.
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

import gym
import numpy as np
from gym import spaces

logger = logging.getLogger(__name__)


class ObjectAvoidanceActionWrapper(gym.ActionWrapper):
    """
    Gym action wrapper that modifies actions to avoid detected objects.
    
    This wrapper implements a potential field-based avoidance algorithm that smoothly
    modifies the robot's actions to avoid obstacles while maintaining lane following
    behavior. It supports configurable safety parameters and priority-based avoidance
    for multiple detected objects.
    
    The wrapper expects the environment to provide detection information in the observation
    space, typically from a YOLOObjectDetectionWrapper or similar detection system.
    """
    
    def __init__(
        self,
        env: gym.Env,
        safety_distance: float = 0.5,
        min_clearance: float = 0.2,
        avoidance_strength: float = 1.0,
        max_avoidance_action: float = 0.8,
        smoothing_factor: float = 0.7,
        lane_attraction_strength: float = 0.3,
        emergency_brake_distance: float = 0.15,
        detection_field_width: float = 1.0,
        enable_emergency_brake: bool = True,
        debug_logging: bool = False
    ):
        """
        Initialize Object Avoidance Action Wrapper.
        
        Args:
            env: Base gym environment to wrap (should provide detection data)
            safety_distance: Distance threshold for initiating avoidance (meters)
            min_clearance: Minimum required clearance from objects (meters)
            avoidance_strength: Strength of avoidance force (0.0-2.0)
            max_avoidance_action: Maximum action modification magnitude (0.0-1.0)
            smoothing_factor: Action smoothing factor for preventing jerky movements (0.0-1.0)
            lane_attraction_strength: Strength of lane center attraction (0.0-1.0)
            emergency_brake_distance: Distance for emergency braking (meters)
            detection_field_width: Width of detection field for relevance filtering (meters)
            enable_emergency_brake: Whether to enable emergency braking
            debug_logging: Enable detailed debug logging
        """
        super().__init__(env)
        
        # Store configuration parameters
        self.safety_distance = safety_distance
        self.min_clearance = min_clearance
        self.avoidance_strength = avoidance_strength
        self.max_avoidance_action = max_avoidance_action
        self.smoothing_factor = smoothing_factor
        self.lane_attraction_strength = lane_attraction_strength
        self.emergency_brake_distance = emergency_brake_distance
        self.detection_field_width = detection_field_width
        self.enable_emergency_brake = enable_emergency_brake
        self.debug_logging = debug_logging
        
        # State tracking
        self.last_action = np.zeros(self.action_space.shape)
        self.last_avoidance_force = np.array([0.0, 0.0])  # [lateral, longitudinal]
        self.emergency_brake_active = False
        self.avoidance_active = False
        
        # Statistics
        self.avoidance_stats = {
            'total_steps': 0,
            'avoidance_activations': 0,
            'emergency_brakes': 0,
            'max_avoidance_force': 0.0,
            'avg_avoidance_force': 0.0
        }
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"ObjectAvoidanceActionWrapper initialized with safety_distance={safety_distance}m")
    
    def _validate_configuration(self):
        """Validate wrapper configuration parameters."""
        if not 0.0 <= self.avoidance_strength <= 2.0:
            raise ValueError(f"avoidance_strength must be in [0.0, 2.0], got {self.avoidance_strength}")
        
        if not 0.0 <= self.max_avoidance_action <= 1.0:
            raise ValueError(f"max_avoidance_action must be in [0.0, 1.0], got {self.max_avoidance_action}")
        
        if not 0.0 <= self.smoothing_factor <= 1.0:
            raise ValueError(f"smoothing_factor must be in [0.0, 1.0], got {self.smoothing_factor}")
        
        if self.min_clearance >= self.safety_distance:
            raise ValueError(f"min_clearance ({self.min_clearance}) must be < safety_distance ({self.safety_distance})")
        
        if self.emergency_brake_distance >= self.min_clearance:
            raise ValueError(f"emergency_brake_distance ({self.emergency_brake_distance}) must be < min_clearance ({self.min_clearance})")
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Modify action to avoid detected objects using potential field algorithm.
        
        Args:
            action: Original action from RL agent [left_wheel_vel, right_wheel_vel]
            
        Returns:
            Modified action with object avoidance
        """
        # Ensure action is numpy array
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        
        # Get current observation for detection data
        current_obs = self._get_current_observation()
        
        # Extract detection information
        detections = self._extract_detections(current_obs)
        
        # Calculate avoidance forces
        avoidance_force = self._calculate_avoidance_force(detections)
        
        # Apply emergency braking if needed
        emergency_action = self._check_emergency_brake(detections, action)
        if emergency_action is not None:
            self._update_stats(emergency_brake=True)
            return emergency_action
        
        # Apply avoidance modifications
        modified_action = self._apply_avoidance_force(action, avoidance_force)
        
        # Apply action smoothing
        smoothed_action = self._apply_action_smoothing(modified_action)
        
        # Update state and statistics
        self._update_state(smoothed_action, avoidance_force)
        self._update_stats(avoidance_active=np.linalg.norm(avoidance_force) > 0.01)
        
        return smoothed_action
    
    def _get_current_observation(self) -> Union[Dict[str, Any], np.ndarray]:
        """
        Get current observation from environment.
        
        Note: This is a simplified approach. In practice, the observation should be
        passed through the action method or stored from the last step.
        """
        # This is a placeholder - in real implementation, observation should be
        # provided through a different mechanism or stored from last step
        if hasattr(self.env, '_last_observation'):
            return self.env._last_observation
        else:
            # Return empty observation if not available
            return {}
    
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
                # Dictionary observation format (from YOLOObjectDetectionWrapper)
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
                
                # Check for safety critical flag
                safety_critical = observation.get('safety_critical', [False])[0]
                if safety_critical and self.debug_logging:
                    logger.debug(f"Safety critical detection active with {len(detections)} objects")
            
            # Filter detections by relevance (within detection field)
            relevant_detections = self._filter_relevant_detections(detections)
            
            if self.debug_logging and relevant_detections:
                logger.debug(f"Found {len(relevant_detections)} relevant detections for avoidance")
            
            return relevant_detections
            
        except Exception as e:
            logger.error(f"Failed to extract detections: {str(e)}")
            return []
    
    def _filter_relevant_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections to only include those relevant for avoidance.
        
        Args:
            detections: List of all detections
            
        Returns:
            List of relevant detections for avoidance
        """
        relevant = []
        
        for detection in detections:
            rel_pos = detection.get('relative_position', [0, 0])
            distance = detection.get('distance', float('inf'))
            
            # Filter by distance and lateral position
            if (distance <= self.safety_distance and 
                abs(rel_pos[0]) <= self.detection_field_width / 2 and
                rel_pos[1] > -0.3):  # Object not too far behind
                relevant.append(detection)
        
        return relevant
    
    def _calculate_avoidance_force(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate avoidance force using potential field algorithm.
        
        Args:
            detections: List of relevant detections
            
        Returns:
            Avoidance force vector [lateral_force, longitudinal_force]
        """
        if not detections:
            self.avoidance_active = False
            return np.array([0.0, 0.0])
        
        total_force = np.array([0.0, 0.0])
        
        # Calculate repulsive forces from each object
        for detection in detections:
            force = self._calculate_object_repulsive_force(detection)
            total_force += force
        
        # Add lane center attraction force
        lane_force = self._calculate_lane_attraction_force()
        total_force += lane_force
        
        # Normalize and scale force
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > 0:
            # Scale by avoidance strength
            total_force = total_force * self.avoidance_strength
            
            # Limit maximum force magnitude
            max_force = self.max_avoidance_action
            if force_magnitude > max_force:
                total_force = total_force * (max_force / force_magnitude)
        
        self.avoidance_active = force_magnitude > 0.01
        
        if self.debug_logging and self.avoidance_active:
            logger.debug(f"Avoidance force: lateral={total_force[0]:.3f}, longitudinal={total_force[1]:.3f}")
        
        return total_force
    
    def _calculate_object_repulsive_force(self, detection: Dict[str, Any]) -> np.ndarray:
        """
        Calculate repulsive force from a single object.
        
        Args:
            detection: Object detection dictionary
            
        Returns:
            Repulsive force vector [lateral_force, longitudinal_force]
        """
        rel_pos = detection.get('relative_position', [0, 0])
        distance = detection.get('distance', float('inf'))
        confidence = detection.get('confidence', 0.0)
        
        # Calculate force direction (away from object)
        if distance < 1e-6:  # Avoid division by zero
            force_direction = np.array([1.0, 0.0])  # Default to right
        else:
            # Force direction is away from object position
            force_direction = np.array([-rel_pos[0], -rel_pos[1]])
            force_magnitude = np.linalg.norm(force_direction)
            if force_magnitude > 0:
                force_direction = force_direction / force_magnitude
            else:
                force_direction = np.array([1.0, 0.0])
        
        # Calculate force magnitude based on distance and confidence
        if distance <= self.min_clearance:
            # Very close - maximum repulsion
            force_magnitude = 1.0
        elif distance >= self.safety_distance:
            # Far enough - no repulsion
            force_magnitude = 0.0
        else:
            # Inverse square law with linear falloff
            normalized_distance = (distance - self.min_clearance) / (self.safety_distance - self.min_clearance)
            force_magnitude = (1.0 - normalized_distance) ** 2
        
        # Scale by object confidence
        force_magnitude *= confidence
        
        # Apply priority weighting (closer objects have higher priority)
        priority_weight = 1.0 / max(distance, 0.1)
        force_magnitude *= priority_weight
        
        return force_direction * force_magnitude
    
    def _calculate_lane_attraction_force(self) -> np.ndarray:
        """
        Calculate attraction force toward lane center.
        
        Returns:
            Lane attraction force vector [lateral_force, longitudinal_force]
        """
        # Simple lane center attraction (assumes robot should stay centered)
        # In a more sophisticated implementation, this would use lane detection
        lateral_attraction = -0.0  # Assume we're already centered
        longitudinal_attraction = 0.0  # No longitudinal bias
        
        return np.array([lateral_attraction, longitudinal_attraction]) * self.lane_attraction_strength
    
    def _check_emergency_brake(
        self, 
        detections: List[Dict[str, Any]], 
        original_action: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Check if emergency braking is needed and return emergency action.
        
        Args:
            detections: List of relevant detections
            original_action: Original action from agent
            
        Returns:
            Emergency brake action if needed, None otherwise
        """
        if not self.enable_emergency_brake:
            return None
        
        # Check for objects within emergency brake distance
        for detection in detections:
            distance = detection.get('distance', float('inf'))
            rel_pos = detection.get('relative_position', [0, 0])
            
            # Emergency brake if object is very close and in front
            if (distance <= self.emergency_brake_distance and 
                abs(rel_pos[0]) <= 0.3 and  # Directly in front
                rel_pos[1] > 0):  # In front of robot
                
                self.emergency_brake_active = True
                
                if self.debug_logging:
                    logger.warning(f"Emergency brake activated! Object at {distance:.3f}m")
                
                # Return emergency stop action
                return np.array([0.0, 0.0])
        
        self.emergency_brake_active = False
        return None
    
    def _apply_avoidance_force(self, action: np.ndarray, avoidance_force: np.ndarray) -> np.ndarray:
        """
        Apply avoidance force to modify the original action.
        
        Args:
            action: Original action [left_wheel_vel, right_wheel_vel]
            avoidance_force: Avoidance force [lateral_force, longitudinal_force]
            
        Returns:
            Modified action with avoidance applied
        """
        if np.linalg.norm(avoidance_force) < 1e-6:
            return action.copy()
        
        lateral_force = avoidance_force[0]
        longitudinal_force = avoidance_force[1]
        
        # Convert forces to wheel velocity modifications
        # Lateral force affects differential wheel speeds
        # Positive lateral force means turn right (reduce right wheel, increase left wheel)
        lateral_modification = lateral_force * 0.5
        
        # Longitudinal force affects both wheels equally
        # Negative longitudinal force means slow down
        longitudinal_modification = longitudinal_force
        
        # Apply modifications
        modified_action = action.copy()
        modified_action[0] += lateral_modification - longitudinal_modification  # Left wheel
        modified_action[1] -= lateral_modification - longitudinal_modification  # Right wheel
        
        # Ensure actions remain in valid range [0, 1]
        modified_action = np.clip(modified_action, 0.0, 1.0)
        
        return modified_action
    
    def _apply_action_smoothing(self, action: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to prevent jerky movements.
        
        Args:
            action: Current action to smooth
            
        Returns:
            Smoothed action
        """
        if self.smoothing_factor <= 0:
            return action
        
        # Exponential moving average smoothing
        smoothed_action = (
            (1.0 - self.smoothing_factor) * self.last_action + 
            self.smoothing_factor * action
        )
        
        return smoothed_action
    
    def _update_state(self, action: np.ndarray, avoidance_force: np.ndarray):
        """Update internal state tracking."""
        self.last_action = action.copy()
        self.last_avoidance_force = avoidance_force.copy()
    
    def _update_stats(self, avoidance_active: bool = False, emergency_brake: bool = False):
        """Update avoidance statistics."""
        self.avoidance_stats['total_steps'] += 1
        
        if avoidance_active:
            self.avoidance_stats['avoidance_activations'] += 1
            
            force_magnitude = np.linalg.norm(self.last_avoidance_force)
            self.avoidance_stats['max_avoidance_force'] = max(
                self.avoidance_stats['max_avoidance_force'], 
                force_magnitude
            )
            
            # Update running average
            total_activations = self.avoidance_stats['avoidance_activations']
            current_avg = self.avoidance_stats['avg_avoidance_force']
            self.avoidance_stats['avg_avoidance_force'] = (
                (current_avg * (total_activations - 1) + force_magnitude) / total_activations
            )
        
        if emergency_brake:
            self.avoidance_stats['emergency_brakes'] += 1
    
    def get_avoidance_stats(self) -> Dict[str, Any]:
        """
        Get avoidance performance statistics.
        
        Returns:
            Dictionary with avoidance statistics
        """
        stats = self.avoidance_stats.copy()
        
        # Calculate derived statistics
        total_steps = stats['total_steps']
        if total_steps > 0:
            stats['avoidance_rate'] = stats['avoidance_activations'] / total_steps
            stats['emergency_brake_rate'] = stats['emergency_brakes'] / total_steps
        else:
            stats['avoidance_rate'] = 0.0
            stats['emergency_brake_rate'] = 0.0
        
        # Add current state
        stats['current_avoidance_active'] = self.avoidance_active
        stats['current_emergency_brake'] = self.emergency_brake_active
        stats['last_avoidance_force'] = self.last_avoidance_force.tolist()
        
        return stats
    
    def reset_avoidance_stats(self):
        """Reset avoidance statistics."""
        self.avoidance_stats = {
            'total_steps': 0,
            'avoidance_activations': 0,
            'emergency_brakes': 0,
            'max_avoidance_force': 0.0,
            'avg_avoidance_force': 0.0
        }
    
    def is_avoidance_active(self) -> bool:
        """Check if avoidance is currently active."""
        return self.avoidance_active
    
    def is_emergency_brake_active(self) -> bool:
        """Check if emergency brake is currently active."""
        return self.emergency_brake_active
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current wrapper configuration."""
        return {
            'safety_distance': self.safety_distance,
            'min_clearance': self.min_clearance,
            'avoidance_strength': self.avoidance_strength,
            'max_avoidance_action': self.max_avoidance_action,
            'smoothing_factor': self.smoothing_factor,
            'lane_attraction_strength': self.lane_attraction_strength,
            'emergency_brake_distance': self.emergency_brake_distance,
            'detection_field_width': self.detection_field_width,
            'enable_emergency_brake': self.enable_emergency_brake,
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
        # Reset state tracking
        self.last_action = np.zeros(self.action_space.shape)
        self.last_avoidance_force = np.array([0.0, 0.0])
        self.emergency_brake_active = False
        self.avoidance_active = False
        
        # Reset base environment
        return self.env.reset(**kwargs)