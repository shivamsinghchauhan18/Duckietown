"""
Multi-Objective Reward Wrapper for Enhanced Duckietown RL

This wrapper combines multiple reward components (lane following, object avoidance, 
lane changing, efficiency, and safety) into a single reward signal with configurable weights.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

try:
    import gym
    from gym_duckietown.simulator import NotInLane
    GYM_AVAILABLE = True
except ImportError:
    # For testing without gym dependencies
    GYM_AVAILABLE = False
    
    # Mock gym.RewardWrapper for testing
    class RewardWrapper:
        def __init__(self, env):
            self.env = env
        
        def step(self, action):
            return self.env.step(action)
        
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)
        
        def reward(self, reward):
            return reward
    
    # Mock NotInLane exception
    class NotInLane(Exception):
        pass
    
    gym = type('gym', (), {'RewardWrapper': RewardWrapper})()

logger = logging.getLogger(__name__)


class MultiObjectiveRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that combines multiple objective rewards with configurable weights.
    
    This wrapper calculates rewards for:
    - Lane following (based on position and orientation)
    - Object avoidance (maintaining safe distances)
    - Lane changing (successful lane changes when needed)
    - Efficiency (forward progress)
    - Safety penalties (collisions and unsafe maneuvers)
    """
    
    def __init__(self, env, reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the multi-objective reward wrapper.
        
        Args:
            env: The environment to wrap
            reward_weights: Dictionary of reward component weights. Default weights are used if None.
        """
        super(MultiObjectiveRewardWrapper, self).__init__(env)
        
        # Default reward weights
        self.default_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5,
            'lane_changing': 0.3,
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
        
        # Use provided weights or defaults
        self.reward_weights = reward_weights if reward_weights is not None else self.default_weights.copy()
        
        # Validate weights
        self._validate_weights()
        
        # Initialize reward component tracking
        self.reward_components = {
            'lane_following': 0.0,
            'object_avoidance': 0.0,
            'lane_changing': 0.0,
            'efficiency': 0.0,
            'safety_penalty': 0.0,
            'total': 0.0
        }
        
        # State tracking for reward calculations
        self.prev_pos = None
        self.prev_lane_pos = None
        self.lane_change_start_time = None
        self.lane_change_target = None
        self.episode_step = 0
        
        logger.info(f"MultiObjectiveRewardWrapper initialized with weights: {self.reward_weights}")
    
    def _validate_weights(self):
        """Validate that all required weight keys are present and values are numeric."""
        required_keys = set(self.default_weights.keys())
        provided_keys = set(self.reward_weights.keys())
        
        if not required_keys.issubset(provided_keys):
            missing_keys = required_keys - provided_keys
            raise ValueError(f"Missing reward weight keys: {missing_keys}")
        
        for key, weight in self.reward_weights.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Reward weight '{key}' must be numeric, got {type(weight)}")
    
    def reward(self, reward: float) -> float:
        """
        Calculate the multi-objective reward.
        
        Args:
            reward: Original reward from the environment
            
        Returns:
            Combined multi-objective reward
        """
        # Calculate individual reward components
        lane_following_reward = self._calculate_lane_following_reward()
        object_avoidance_reward = self._calculate_object_avoidance_reward()
        lane_changing_reward = self._calculate_lane_changing_reward()
        efficiency_reward = self._calculate_efficiency_reward()
        safety_penalty = self._calculate_safety_penalty()
        
        # Store components for logging
        self.reward_components.update({
            'lane_following': lane_following_reward,
            'object_avoidance': object_avoidance_reward,
            'lane_changing': lane_changing_reward,
            'efficiency': efficiency_reward,
            'safety_penalty': safety_penalty
        })
        
        # Calculate weighted total reward
        total_reward = (
            self.reward_weights['lane_following'] * lane_following_reward +
            self.reward_weights['object_avoidance'] * object_avoidance_reward +
            self.reward_weights['lane_changing'] * lane_changing_reward +
            self.reward_weights['efficiency'] * efficiency_reward +
            self.reward_weights['safety_penalty'] * safety_penalty
        )
        
        self.reward_components['total'] = total_reward
        
        # Log reward components for debugging
        logger.debug(f"Reward components: {self.reward_components}")
        
        return total_reward
    
    def _calculate_lane_following_reward(self) -> float:
        """
        Calculate reward for lane following behavior.
        
        Returns:
            Lane following reward based on position and orientation
        """
        try:
            pos = self.unwrapped.cur_pos
            angle = self.unwrapped.cur_angle
            lp = self.unwrapped.get_lane_pos2(pos, angle)
            
            # Reward based on distance from lane center (closer is better)
            # lp.dist is negative on left side, positive on right side of lane center
            max_lane_dist = 0.05  # Maximum acceptable distance from lane center
            dist_reward = max(0, 1.0 - abs(lp.dist) / max_lane_dist)
            
            # Reward based on orientation alignment with lane direction
            max_angle_dev = 30.0  # Maximum acceptable angle deviation in degrees
            angle_reward = max(0, 1.0 - abs(lp.angle_deg) / max_angle_dev)
            
            # Combine distance and angle rewards
            lane_reward = 0.6 * dist_reward + 0.4 * angle_reward
            
            return lane_reward
            
        except (NotInLane, AttributeError):
            # Heavy penalty for being off the road
            return -1.0
    
    def _calculate_object_avoidance_reward(self) -> float:
        """
        Calculate reward for object avoidance behavior.
        
        Returns:
            Object avoidance reward based on maintaining safe distances
        """
        try:
            # Check if we have object detection information in the observation
            if hasattr(self.env, 'last_observation') and isinstance(self.env.last_observation, dict):
                obs = self.env.last_observation
                if 'detections' in obs and obs['detections']:
                    # Calculate reward based on closest object distance
                    min_distance = min(det.get('distance', float('inf')) for det in obs['detections'])
                    
                    safety_distance = 0.5  # Minimum safe distance
                    if min_distance < safety_distance:
                        # Penalty for being too close to objects
                        proximity_penalty = (safety_distance - min_distance) / safety_distance
                        return -proximity_penalty
                    else:
                        # Small positive reward for maintaining safe distance
                        return 0.1
            
            # Use simulator's proximity penalty if available
            if hasattr(self.unwrapped, 'proximity_penalty2'):
                pos = self.unwrapped.cur_pos
                angle = self.unwrapped.cur_angle
                proximity_penalty = self.unwrapped.proximity_penalty2(pos, angle)
                # Convert penalty to reward (less penalty = more reward)
                return max(0, 1.0 - proximity_penalty)
            
            # No objects detected or no proximity information
            return 0.1
            
        except (AttributeError, KeyError):
            return 0.0
    
    def _calculate_lane_changing_reward(self) -> float:
        """
        Calculate reward for lane changing behavior.
        
        Returns:
            Lane changing reward based on successful lane changes when needed
        """
        try:
            # Check if we have lane changing information
            if hasattr(self.env, 'lane_change_state'):
                lane_state = self.env.lane_change_state
                
                if lane_state.get('current_phase') == 'executing':
                    # Reward for successful lane change execution
                    progress = lane_state.get('progress', 0.0)
                    return 0.5 * progress
                elif lane_state.get('current_phase') == 'following' and lane_state.get('lane_change_completed', False):
                    # Bonus for completing a lane change
                    return 1.0
                elif lane_state.get('current_phase') == 'evaluating':
                    # Small reward for evaluating lane change options
                    return 0.1
            
            # Check for lane change action in the action space
            if hasattr(self.env, 'last_action') and isinstance(self.env.last_action, dict):
                if self.env.last_action.get('lane_change_initiated', False):
                    return 0.3
            
            return 0.0
            
        except (AttributeError, KeyError):
            return 0.0
    
    def _calculate_efficiency_reward(self) -> float:
        """
        Calculate reward for forward progress efficiency.
        
        Returns:
            Efficiency reward based on forward movement
        """
        try:
            pos = self.unwrapped.cur_pos
            
            if self.prev_pos is not None:
                # Calculate distance traveled
                distance_traveled = np.linalg.norm(pos - self.prev_pos)
                
                # Get lane direction to ensure forward progress
                angle = self.unwrapped.cur_angle
                curve_point, tangent = self.unwrapped.closest_curve_point(pos, angle)
                
                if curve_point is not None and tangent is not None:
                    # Check if movement is in the correct direction
                    movement_vector = pos - self.prev_pos
                    forward_progress = np.dot(movement_vector, tangent)
                    
                    if forward_progress > 0:
                        # Reward proportional to forward progress
                        return min(1.0, forward_progress * 10.0)  # Scale factor
                    else:
                        # Penalty for backward movement
                        return -0.5
                else:
                    # Fallback: reward any movement
                    return min(0.5, distance_traveled * 5.0)
            
            return 0.0
            
        except (AttributeError, KeyError):
            return 0.0
    
    def _calculate_safety_penalty(self) -> float:
        """
        Calculate safety penalties for collisions and unsafe maneuvers.
        
        Returns:
            Safety penalty (negative value for unsafe behavior)
        """
        penalty = 0.0
        
        try:
            # Check for collision with objects
            if hasattr(self.unwrapped, 'collision_penalty'):
                collision_penalty = self.unwrapped.collision_penalty
                penalty += collision_penalty
            
            # Check if robot is off the road
            try:
                pos = self.unwrapped.cur_pos
                angle = self.unwrapped.cur_angle
                lp = self.unwrapped.get_lane_pos2(pos, angle)
                
                # Penalty for being too far from lane center
                if abs(lp.dist) > 0.1:  # 10cm from lane center
                    penalty += abs(lp.dist) * 2.0
                    
            except NotInLane:
                # Heavy penalty for being completely off the road
                penalty += 5.0
            
            # Check for unsafe lane changes
            if hasattr(self.env, 'lane_change_state'):
                lane_state = self.env.lane_change_state
                if lane_state.get('unsafe_lane_change', False):
                    penalty += 2.0
            
            # Check for excessive speed in turns
            if hasattr(self.unwrapped, 'wheelVels'):
                max_wheel_vel = np.max(np.abs(self.unwrapped.wheelVels))
                if max_wheel_vel > 1.5:  # Threshold for excessive speed
                    try:
                        pos = self.unwrapped.cur_pos
                        angle = self.unwrapped.cur_angle
                        lp = self.unwrapped.get_lane_pos2(pos, angle)
                        if abs(lp.angle_deg) > 20:  # In a turn
                            penalty += (max_wheel_vel - 1.5) * 0.5
                    except NotInLane:
                        pass
            
            return -penalty  # Return negative value as penalty
            
        except (AttributeError, KeyError):
            return 0.0
    
    def step(self, action):
        """
        Step the environment and calculate multi-objective reward.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Store current position for next step calculations
        if hasattr(self.unwrapped, 'cur_pos'):
            self.prev_pos = self.unwrapped.cur_pos.copy()
        
        # Step the environment
        observation, reward, done, info = self.env.step(action)
        
        # Store observation and action for reward calculations
        if hasattr(self.env, 'last_observation'):
            self.env.last_observation = observation
        if hasattr(self.env, 'last_action'):
            self.env.last_action = action
        
        # Calculate multi-objective reward
        multi_reward = self.reward(reward)
        
        # Add reward components to info for logging
        if 'custom_rewards' not in info:
            info['custom_rewards'] = {}
        
        info['custom_rewards'].update(self.reward_components)
        info['reward_weights'] = self.reward_weights.copy()
        
        self.episode_step += 1
        
        return observation, multi_reward, done, info
    
    def reset(self, **kwargs):
        """
        Reset the environment and reward tracking.
        
        Args:
            **kwargs: Arguments to pass to environment reset
            
        Returns:
            Initial observation
        """
        # Reset tracking variables
        self.prev_pos = None
        self.prev_lane_pos = None
        self.lane_change_start_time = None
        self.lane_change_target = None
        self.episode_step = 0
        
        # Reset reward components
        for key in self.reward_components:
            self.reward_components[key] = 0.0
        
        return self.env.reset(**kwargs)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update reward weights during runtime.
        
        Args:
            new_weights: Dictionary of new reward weights
        """
        # Validate new weights
        old_weights = self.reward_weights.copy()
        self.reward_weights.update(new_weights)
        
        try:
            self._validate_weights()
            logger.info(f"Updated reward weights: {self.reward_weights}")
        except ValueError as e:
            # Restore old weights if validation fails
            self.reward_weights = old_weights
            logger.error(f"Failed to update weights: {e}")
            raise
    
    def get_reward_components(self) -> Dict[str, float]:
        """
        Get the current reward components.
        
        Returns:
            Dictionary of current reward components
        """
        return self.reward_components.copy()
    
    def get_reward_weights(self) -> Dict[str, float]:
        """
        Get the current reward weights.
        
        Returns:
            Dictionary of current reward weights
        """
        return self.reward_weights.copy()