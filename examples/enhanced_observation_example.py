"""
Enhanced Observation Wrapper Integration Example.

This example demonstrates how to use the EnhancedObservationWrapper
with the YOLO Object Detection Wrapper to create a unified observation
space suitable for PPO training.
"""

import sys
import os
import numpy as np
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    logger.warning("gym not available, using mock environment")
    GYM_AVAILABLE = False

try:
    from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
    from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
    WRAPPERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Wrappers not available: {e}")
    WRAPPERS_AVAILABLE = False


class MockDuckietownEnv:
    """Mock Duckietown environment for demonstration."""
    
    def __init__(self):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 160, 3), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.step_count = 0
    
    def reset(self):
        """Reset environment and return initial observation."""
        self.step_count = 0
        return self._generate_observation()
    
    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1
        obs = self._generate_observation()
        reward = 0.1  # Small positive reward
        done = self.step_count >= 100  # Episode ends after 100 steps
        info = {'step': self.step_count}
        return obs, reward, done, info
    
    def _generate_observation(self):
        """Generate a mock camera observation."""
        # Create a simple synthetic image with some patterns
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        
        # Add some background
        image[:, :, 0] = 50  # Red channel
        image[:, :, 1] = 100  # Green channel
        image[:, :, 2] = 150  # Blue channel
        
        # Add some "objects" (colored rectangles)
        if self.step_count % 20 < 10:  # Object appears periodically
            # Add a "duckiebot" (yellow rectangle)
            image[40:80, 60:100, 0] = 255  # Red
            image[40:80, 60:100, 1] = 255  # Green
            image[40:80, 60:100, 2] = 0    # Blue (yellow = red + green)
        
        if self.step_count % 30 < 15:  # Another object
            # Add a "cone" (orange rectangle)
            image[60:90, 100:130, 0] = 255  # Red
            image[60:90, 100:130, 1] = 165  # Green
            image[60:90, 100:130, 2] = 0    # Blue (orange)
        
        return image


class MockYOLODetectionWrapper:
    """Mock YOLO detection wrapper for demonstration."""
    
    def __init__(self, env, **kwargs):
        self.env = env
        self.observation_space = spaces.Dict({
            'image': env.observation_space,
            'detections': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 9), dtype=np.float32),
            'detection_count': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'safety_critical': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'inference_time': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = env.action_space
    
    def reset(self):
        """Reset environment and return enhanced observation."""
        obs = self.env.reset()
        return self.observation(obs)
    
    def step(self, action):
        """Take a step and return enhanced observation."""
        obs, reward, done, info = self.env.step(action)
        enhanced_obs = self.observation(obs)
        return enhanced_obs, reward, done, info
    
    def observation(self, obs):
        """Convert image observation to detection format."""
        # Simulate object detection results
        detections = np.zeros((10, 9), dtype=np.float32)
        detection_count = 0
        safety_critical = False
        
        # Simple "detection" based on image content
        # Check for yellow pixels (simulated duckiebot)
        yellow_pixels = np.sum((obs[:, :, 0] > 200) & (obs[:, :, 1] > 200) & (obs[:, :, 2] < 50))
        if yellow_pixels > 100:  # Threshold for detection
            # Add duckiebot detection
            detections[detection_count] = [
                1,      # class_id (duckiebot)
                0.85,   # confidence
                60, 40, 100, 80,  # bbox (x1, y1, x2, y2)
                0.0, 1.5,  # relative_position (x, y)
                2.0     # distance
            ]
            detection_count += 1
            if detections[0, 8] < 1.0:  # Distance < 1m
                safety_critical = True
        
        # Check for orange pixels (simulated cone)
        orange_pixels = np.sum((obs[:, :, 0] > 200) & (obs[:, :, 1] > 100) & (obs[:, :, 1] < 200) & (obs[:, :, 2] < 50))
        if orange_pixels > 50:
            # Add cone detection
            detections[detection_count] = [
                2,      # class_id (cone)
                0.75,   # confidence
                100, 60, 130, 90,  # bbox
                0.5, 1.0,  # relative_position
                3.5     # distance
            ]
            detection_count += 1
        
        return {
            'image': obs,
            'detections': detections,
            'detection_count': np.array([detection_count]),
            'safety_critical': np.array([int(safety_critical)]),
            'inference_time': np.array([0.03])  # 30ms inference time
        }


def demonstrate_enhanced_observation_wrapper():
    """Demonstrate the Enhanced Observation Wrapper functionality."""
    print("Enhanced Observation Wrapper Demonstration")
    print("=" * 50)
    
    if not WRAPPERS_AVAILABLE:
        print("Wrappers not available, cannot run demonstration")
        return False
    
    # Create base environment
    print("1. Creating base environment...")
    base_env = MockDuckietownEnv()
    print(f"   Base observation space: {base_env.observation_space}")
    
    # Add YOLO detection wrapper (or mock)
    print("\n2. Adding YOLO detection wrapper...")
    if GYM_AVAILABLE:
        try:
            yolo_env = YOLOObjectDetectionWrapper(
                base_env,
                model_path="yolov5s.pt",  # This will likely fail without proper setup
                confidence_threshold=0.5,
                flatten_detections=False
            )
            print("   Using real YOLO wrapper")
        except Exception as e:
            print(f"   Real YOLO wrapper failed ({e}), using mock")
            yolo_env = MockYOLODetectionWrapper(base_env)
    else:
        yolo_env = MockYOLODetectionWrapper(base_env)
    
    print(f"   YOLO observation space: {yolo_env.observation_space}")
    
    # Add Enhanced Observation Wrapper
    print("\n3. Adding Enhanced Observation Wrapper...")
    
    # Test flattened mode (PPO compatible)
    enhanced_env_flat = EnhancedObservationWrapper(
        yolo_env,
        output_mode='flattened',
        include_detection_features=True,
        include_image_features=True,
        max_detections=5,
        normalize_features=True,
        feature_scaling_method='minmax'
    )
    
    print(f"   Enhanced observation space (flattened): {enhanced_env_flat.observation_space}")
    print(f"   Observation space shape: {enhanced_env_flat.observation_space.shape}")
    
    # Test dictionary mode
    enhanced_env_dict = EnhancedObservationWrapper(
        yolo_env,
        output_mode='dict',
        include_detection_features=True,
        include_image_features=True,
        max_detections=5,
        normalize_features=True
    )
    
    print(f"   Enhanced observation space (dict): {enhanced_env_dict.observation_space}")
    
    # Run a few steps to demonstrate functionality
    print("\n4. Running demonstration episodes...")
    
    # Test flattened mode
    print("\n   Testing flattened mode:")
    obs_flat = enhanced_env_flat.reset()
    print(f"   Initial observation shape: {obs_flat.shape}")
    print(f"   Initial observation dtype: {obs_flat.dtype}")
    print(f"   Observation range: [{np.min(obs_flat):.3f}, {np.max(obs_flat):.3f}]")
    
    for step in range(5):
        action = enhanced_env_flat.action_space.sample()  # Random action
        obs_flat, reward, done, info = enhanced_env_flat.step(action)
        print(f"   Step {step+1}: obs_shape={obs_flat.shape}, reward={reward:.3f}")
        
        if done:
            obs_flat = enhanced_env_flat.reset()
            print("   Episode ended, reset environment")
    
    # Test dictionary mode
    print("\n   Testing dictionary mode:")
    obs_dict = enhanced_env_dict.reset()
    print(f"   Initial observation keys: {list(obs_dict.keys())}")
    
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: {type(value)}")
    
    for step in range(3):
        action = enhanced_env_dict.action_space.sample()
        obs_dict, reward, done, info = enhanced_env_dict.step(action)
        
        # Check safety features
        if 'safety_features' in obs_dict:
            safety = obs_dict['safety_features']
            print(f"   Step {step+1}: safety_features={safety}")
        
        if done:
            obs_dict = enhanced_env_dict.reset()
    
    # Display statistics
    print("\n5. Performance statistics:")
    
    flat_stats = enhanced_env_flat.get_feature_stats()
    print("   Flattened mode stats:")
    for key, value in flat_stats.items():
        print(f"     {key}: {value}")
    
    dict_stats = enhanced_env_dict.get_feature_stats()
    print("   Dictionary mode stats:")
    for key, value in dict_stats.items():
        print(f"     {key}: {value}")
    
    # Display configuration info
    print("\n6. Configuration information:")
    
    flat_info = enhanced_env_flat.get_observation_info()
    print("   Flattened mode config:")
    for key, value in flat_info.items():
        print(f"     {key}: {value}")
    
    print("\n✓ Enhanced Observation Wrapper demonstration completed successfully!")
    return True


def demonstrate_ppo_compatibility():
    """Demonstrate PPO compatibility features."""
    print("\nPPO Compatibility Demonstration")
    print("=" * 40)
    
    if not WRAPPERS_AVAILABLE:
        print("Wrappers not available")
        return False
    
    # Create environment with PPO-optimized settings
    base_env = MockDuckietownEnv()
    yolo_env = MockYOLODetectionWrapper(base_env)
    
    ppo_env = EnhancedObservationWrapper(
        yolo_env,
        output_mode='flattened',  # PPO works best with flattened observations
        include_detection_features=True,
        include_image_features=True,
        image_feature_method='encode',  # Use encoded features for efficiency
        normalize_features=True,
        feature_scaling_method='minmax',  # Stable normalization for PPO
        max_detections=5  # Reasonable number for real-time processing
    )
    
    print(f"PPO-optimized observation space: {ppo_env.observation_space}")
    print(f"Observation dimension: {ppo_env.observation_space.shape[0]}")
    
    # Test observation consistency
    obs1 = ppo_env.reset()
    obs2 = ppo_env.reset()
    
    print(f"Observation consistency check:")
    print(f"  Shape consistency: {obs1.shape == obs2.shape}")
    print(f"  Dtype consistency: {obs1.dtype == obs2.dtype}")
    print(f"  Value range: [{np.min(obs1):.3f}, {np.max(obs1):.3f}]")
    
    # Test action space compatibility
    print(f"Action space: {ppo_env.action_space}")
    print(f"Action space compatible with continuous control: {isinstance(ppo_env.action_space, spaces.Box)}")
    
    # Simulate a training-like loop
    print("\nSimulating training loop:")
    episode_rewards = []
    
    for episode in range(3):
        obs = ppo_env.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < 20:  # Short episodes for demo
            # Simulate PPO action selection (random for demo)
            action = ppo_env.action_space.sample()
            
            obs, reward, done, info = ppo_env.step(action)
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: {step_count} steps, reward={episode_reward:.3f}")
    
    print(f"Average episode reward: {np.mean(episode_rewards):.3f}")
    
    print("✓ PPO compatibility demonstration completed!")
    return True


def main():
    """Main demonstration function."""
    print("Enhanced Observation Wrapper Examples")
    print("=" * 60)
    
    success = True
    
    try:
        # Run main demonstration
        if not demonstrate_enhanced_observation_wrapper():
            success = False
        
        # Run PPO compatibility demonstration
        if not demonstrate_ppo_compatibility():
            success = False
            
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All demonstrations completed successfully!")
        print("\nThe Enhanced Observation Wrapper is ready for use with:")
        print("  - YOLO Object Detection integration")
        print("  - PPO training compatibility")
        print("  - Flexible observation space configuration")
        print("  - Real-time feature extraction and normalization")
    else:
        print("\n" + "=" * 60)
        print("✗ Some demonstrations failed")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)