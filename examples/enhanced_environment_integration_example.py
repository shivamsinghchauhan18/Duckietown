#!/usr/bin/env python3
"""
Enhanced Environment Integration Example

This example demonstrates how to use the enhanced Duckietown RL environment
with object detection, avoidance, and lane changing capabilities.

Usage:
    python examples/enhanced_environment_integration_example.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from duckietown_utils.env import launch_and_wrap_enhanced_env, get_enhanced_wrappers
from config.enhanced_config import EnhancedRLConfig, load_enhanced_config
from config.config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_enhanced_config():
    """Create a sample enhanced configuration."""
    config = EnhancedRLConfig(
        enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
        debug_mode=True
    )
    
    # Customize YOLO settings
    config.yolo.confidence_threshold = 0.6
    config.yolo.device = 'auto'  # Auto-detect GPU/CPU
    
    # Customize object avoidance settings
    config.object_avoidance.safety_distance = 0.6
    config.object_avoidance.avoidance_strength = 1.2
    
    # Customize lane changing settings
    config.lane_changing.lane_change_threshold = 0.4
    config.lane_changing.safety_margin = 2.5
    
    # Customize reward weights
    config.reward.lane_following_weight = 1.0
    config.reward.object_avoidance_weight = 0.8
    config.reward.lane_change_weight = 0.5
    config.reward.efficiency_weight = 0.3
    
    # Enable comprehensive logging
    config.logging.log_detections = True
    config.logging.log_actions = True
    config.logging.log_rewards = True
    config.logging.log_performance = True
    
    return config


def create_sample_env_config():
    """Create a sample environment configuration."""
    return {
        'training_map': 'small_loop',
        'episode_max_steps': 500,
        'domain_rand': False,
        'dynamics_rand': False,
        'camera_rand': False,
        'accepted_start_angle_deg': 60,
        'distortion': False,
        'simulation_framerate': 30,
        'frame_skip': 1,
        'robot_speed': 0.3,
        'mode': 'train',
        'aido_wrapper': False,
        'crop_image_top': True,
        'top_crop_divider': 3,
        'grayscale_image': False,
        'resized_input_shape': (84, 84, 3),
        'frame_stacking': False,
        'motion_blur': False,
        'action_type': 'continuous',
        'reward_function': 'Posangle'
    }


def example_basic_usage():
    """Example of basic enhanced environment usage."""
    print("\n=== Basic Enhanced Environment Usage ===")
    
    # Create configurations
    env_config = create_sample_env_config()
    enhanced_config = create_sample_enhanced_config()
    
    try:
        # Create enhanced environment
        print("Creating enhanced environment...")
        env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
        
        # Get wrapper information
        obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)
        
        print(f"Environment created successfully!")
        print(f"Enhanced wrappers applied: {len(enhanced_wrappers)}")
        print(f"Total observation wrappers: {len(obs_wrappers)}")
        print(f"Total action wrappers: {len(action_wrappers)}")
        print(f"Total reward wrappers: {len(reward_wrappers)}")
        
        # Print wrapper details
        print("\nEnhanced wrappers:")
        for wrapper in enhanced_wrappers:
            print(f"  - {wrapper.__class__.__name__}")
        
        # Simulate environment interaction
        print("\nSimulating environment interaction...")
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {step + 1}: reward={reward:.3f}, done={done}")
            
            if done:
                obs = env.reset()
                print("Episode finished, environment reset")
        
        print("Basic usage example completed successfully!")
        
    except ImportError as e:
        print(f"Skipping basic usage example due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error in basic usage example: {e}")


def example_selective_features():
    """Example of using selective enhanced features."""
    print("\n=== Selective Features Usage ===")
    
    # Create configuration with only specific features
    enhanced_config = EnhancedRLConfig(
        enabled_features=['object_avoidance'],  # Only object avoidance
        debug_mode=False
    )
    
    env_config = create_sample_env_config()
    
    try:
        print("Creating environment with only object avoidance...")
        env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
        
        obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)
        
        print(f"Enhanced wrappers applied: {len(enhanced_wrappers)}")
        print("Enhanced wrappers:")
        for wrapper in enhanced_wrappers:
            print(f"  - {wrapper.__class__.__name__}")
        
        print("Selective features example completed successfully!")
        
    except ImportError as e:
        print(f"Skipping selective features example due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error in selective features example: {e}")


def example_config_from_file():
    """Example of loading configuration from file."""
    print("\n=== Configuration from File Usage ===")
    
    # Create a sample configuration file
    config_path = Path("temp_enhanced_config.yml")
    
    try:
        enhanced_config = create_sample_enhanced_config()
        enhanced_config.to_yaml(config_path)
        print(f"Configuration saved to {config_path}")
        
        # Load configuration from file
        loaded_config = load_enhanced_config(config_path)
        print(f"Configuration loaded from file")
        print(f"Enabled features: {loaded_config.enabled_features}")
        print(f"YOLO confidence threshold: {loaded_config.yolo.confidence_threshold}")
        
        # Use loaded configuration
        env_config = create_sample_env_config()
        env = launch_and_wrap_enhanced_env(env_config, loaded_config)
        
        print("Configuration from file example completed successfully!")
        
    except ImportError as e:
        print(f"Skipping config from file example due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error in config from file example: {e}")
    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()
            print(f"Temporary config file {config_path} removed")


def example_backward_compatibility():
    """Example of backward compatibility with standard environment."""
    print("\n=== Backward Compatibility Usage ===")
    
    # Create configuration with no enhanced features
    enhanced_config = EnhancedRLConfig(
        enabled_features=[],  # No enhanced features
        debug_mode=False
    )
    
    env_config = create_sample_env_config()
    
    try:
        print("Creating environment with no enhanced features (backward compatibility)...")
        env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
        
        obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)
        
        print(f"Enhanced wrappers applied: {len(enhanced_wrappers)} (should be 0)")
        print(f"Standard wrappers still work: {len(obs_wrappers + action_wrappers + reward_wrappers)} total")
        
        print("Backward compatibility example completed successfully!")
        
    except ImportError as e:
        print(f"Skipping backward compatibility example due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error in backward compatibility example: {e}")


def example_error_handling():
    """Example of error handling and graceful degradation."""
    print("\n=== Error Handling Usage ===")
    
    # Create configuration that might fail (e.g., invalid YOLO model path)
    enhanced_config = EnhancedRLConfig(
        enabled_features=['yolo', 'object_avoidance'],
        debug_mode=False  # Non-debug mode for graceful degradation
    )
    enhanced_config.yolo.model_path = "nonexistent_model.pt"
    
    env_config = create_sample_env_config()
    
    try:
        print("Creating environment with potentially failing configuration...")
        env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
        
        obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)
        
        print(f"Environment created despite potential failures")
        print(f"Enhanced wrappers applied: {len(enhanced_wrappers)}")
        print("Error handling demonstrates graceful degradation")
        
        print("Error handling example completed successfully!")
        
    except ImportError as e:
        print(f"Skipping error handling example due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error in error handling example: {e}")


def main():
    """Run all examples."""
    print("Enhanced Duckietown RL Environment Integration Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Selective Features", example_selective_features),
        ("Configuration from File", example_config_from_file),
        ("Backward Compatibility", example_backward_compatibility),
        ("Error Handling", example_error_handling),
    ]
    
    for example_name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example '{example_name}' failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nTo use the enhanced environment in your own code:")
    print("1. Import: from duckietown_utils.env import launch_and_wrap_enhanced_env")
    print("2. Create config: from config.enhanced_config import EnhancedRLConfig")
    print("3. Launch: env = launch_and_wrap_enhanced_env(env_config, enhanced_config)")


if __name__ == "__main__":
    main()