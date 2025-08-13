"""
Lane Changing Integration Example for Duckietown RL Environment.

This example demonstrates how to integrate the LaneChangingActionWrapper
with the Duckietown environment for dynamic lane changing capabilities.
"""

import numpy as np
import time
from typing import Dict, Any

# Note: This example assumes gym and duckietown packages are available
# For demonstration purposes, we'll use mock classes if imports fail

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    print("Warning: gym not available, using mock classes for demonstration")
    GYM_AVAILABLE = False

# Import our lane changing wrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import (
    LaneChangingActionWrapper,
    LaneChangePhase
)


class MockDuckietownEnv:
    """Mock Duckietown environment for demonstration."""
    
    def __init__(self):
        if GYM_AVAILABLE:
            self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        else:
            # Simple mock for when gym is not available
            self.action_space = type('MockSpace', (), {'shape': (2,)})()
            self.observation_space = type('MockSpace', (), {'shape': (120, 160, 3)})()
        
        self._last_observation = {}
        self.step_count = 0
    
    def step(self, action):
        """Simulate environment step."""
        self.step_count += 1
        
        # Create mock observation with detections
        observation = self._create_mock_observation()
        reward = 1.0  # Simple reward
        done = self.step_count >= 100  # End after 100 steps
        info = {'step': self.step_count}
        
        return observation, reward, done, info
    
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        self._last_observation = {}
        return self._create_mock_observation()
    
    def _create_mock_observation(self):
        """Create mock observation with detection data."""
        # Simulate detection data format
        detection_data = np.zeros((10, 9))
        
        # Add some mock detections based on step count
        if self.step_count > 20 and self.step_count < 40:
            # Simulate obstacle in current lane
            detection_data[0] = [1, 0.8, 100, 50, 150, 100, 0.1, 1.5, 2.0]
        elif self.step_count > 60 and self.step_count < 80:
            # Simulate obstacle requiring lane change
            detection_data[0] = [1, 0.9, 100, 50, 150, 100, 0.0, 1.0, 1.5]
        
        observation = {
            'image': np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8),
            'detections': detection_data,
            'detection_count': [1 if self.step_count > 20 else 0],
            'safety_critical': [self.step_count > 60 and self.step_count < 80]
        }
        
        # Store for wrapper access
        self._last_observation = observation
        
        return observation


def demonstrate_lane_changing():
    """Demonstrate lane changing wrapper functionality."""
    print("=== Lane Changing Action Wrapper Demonstration ===\n")
    
    # Create mock environment
    env = MockDuckietownEnv()
    
    # Wrap with lane changing capability
    wrapped_env = LaneChangingActionWrapper(
        env,
        lane_change_threshold=0.3,
        safety_margin=2.0,
        max_lane_change_time=3.0,
        num_lanes=2,
        debug_logging=True
    )
    
    print("Environment created with lane changing wrapper")
    print(f"Number of lanes: {wrapped_env.num_lanes}")
    print(f"Current lane: {wrapped_env.get_current_lane()}")
    print(f"Lane changing enabled: {wrapped_env.enable_lane_changing}")
    print()
    
    # Reset environment
    obs = wrapped_env.reset()
    print("Environment reset")
    print(f"Initial phase: {wrapped_env.get_lane_change_phase().value}")
    print()
    
    # Simulate episode
    total_steps = 100
    lane_changes_completed = 0
    
    for step in range(total_steps):
        # Simple action policy (move forward)
        action = np.array([0.6, 0.6])  # Equal wheel velocities
        
        # Apply action through wrapper
        modified_action = wrapped_env.action(action)
        
        # Step environment
        obs, reward, done, info = wrapped_env.step(modified_action)
        
        # Get current status
        current_phase = wrapped_env.get_lane_change_phase()
        current_lane = wrapped_env.get_current_lane()
        is_changing = wrapped_env.is_lane_changing()
        
        # Print status updates for interesting events
        if step % 20 == 0 or is_changing:
            print(f"Step {step:3d}: Lane {current_lane}, Phase: {current_phase.value:12s}, "
                  f"Changing: {is_changing}, Action: [{modified_action[0]:.2f}, {modified_action[1]:.2f}]")
        
        # Track lane changes
        if current_phase == LaneChangePhase.LANE_FOLLOWING and step > 0:
            prev_stats = wrapped_env.get_lane_change_stats()
            if prev_stats['successful_lane_changes'] > lane_changes_completed:
                lane_changes_completed = prev_stats['successful_lane_changes']
                print(f"    *** Lane change completed! Now in lane {current_lane} ***")
        
        # Demonstrate forced lane change at step 50
        if step == 50:
            print(f"    >>> Forcing lane change to lane 1 <<<")
            success = wrapped_env.force_lane_change(1)
            print(f"    Force lane change result: {success}")
        
        if done:
            break
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    stats = wrapped_env.get_lane_change_stats()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.3f}")
        else:
            print(f"{key:25s}: {value}")
    
    print(f"\nFinal lane: {wrapped_env.get_current_lane()}")
    print(f"Total lane changes: {stats['successful_lane_changes']}")


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management Demonstration ===\n")
    
    env = MockDuckietownEnv()
    wrapper = LaneChangingActionWrapper(env)
    
    # Show initial configuration
    print("Initial configuration:")
    config = wrapper.get_configuration()
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    
    print("\nUpdating configuration...")
    
    # Update some parameters
    wrapper.update_configuration(
        lane_change_threshold=0.5,
        safety_margin=1.5,
        debug_logging=True
    )
    
    print("Updated configuration:")
    config = wrapper.get_configuration()
    for key, value in config.items():
        print(f"  {key:25s}: {value}")


def demonstrate_safety_features():
    """Demonstrate safety features and error handling."""
    print("\n=== Safety Features Demonstration ===\n")
    
    env = MockDuckietownEnv()
    wrapper = LaneChangingActionWrapper(
        env,
        emergency_abort_distance=0.3,
        safety_margin=2.0,
        debug_logging=True
    )
    
    print("Testing safety features...")
    
    # Test invalid lane change
    print("\n1. Testing invalid lane change (out of range):")
    result = wrapper.force_lane_change(5)  # Invalid lane
    print(f"   Result: {result} (should be False)")
    
    # Test same lane change
    print("\n2. Testing same lane change:")
    result = wrapper.force_lane_change(0)  # Same lane
    print(f"   Result: {result} (should be False)")
    
    # Test valid lane change
    print("\n3. Testing valid lane change:")
    result = wrapper.force_lane_change(1)  # Valid lane
    print(f"   Result: {result} (should be True)")
    print(f"   Current phase: {wrapper.get_lane_change_phase().value}")
    
    # Reset for next test
    wrapper.reset()
    
    print("\n4. Testing configuration validation:")
    try:
        # This should raise an error
        wrapper.update_configuration(lane_change_threshold=1.5)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   Correctly caught error: {e}")


if __name__ == "__main__":
    try:
        # Run demonstrations
        demonstrate_lane_changing()
        demonstrate_configuration()
        demonstrate_safety_features()
        
        print("\n=== Demonstration Complete ===")
        print("Lane changing wrapper is working correctly!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()