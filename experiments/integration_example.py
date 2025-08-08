"""
Integration example for Dynamic Lane Changing and Object Avoidance with existing Duckietown-RL

This script shows how to integrate the new avoidance system with the existing
training and evaluation infrastructure.

Authors: Generated for Dynamic Lane Changing and Object Avoidance
License: MIT
"""

import os
import sys
import logging
import yaml
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config, update_config
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.wrappers.unified_avoidance_wrapper import UnifiedAvoidanceWrapper

logger = logging.getLogger(__name__)


def load_avoidance_config(config_path: str = None) -> Dict[str, Any]:
    """Load avoidance system configuration"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/avoidance_config.yml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Failed to load avoidance config: {e}, using defaults")
        return {}


def create_avoidance_enabled_env(base_config: Dict[str, Any], avoidance_config: Dict[str, Any] = None):
    """
    Create environment with avoidance system integrated
    
    Args:
        base_config: Base Duckietown configuration
        avoidance_config: Avoidance system configuration
        
    Returns:
        Environment with avoidance capabilities
    """
    if avoidance_config is None:
        avoidance_config = load_avoidance_config()
    
    # Create base environment using existing infrastructure
    base_env = launch_and_wrap_env(base_config)
    
    # Extract avoidance configuration
    unified_config = {
        'yolo': avoidance_config.get('yolo_config', {}),
        'lane_changing': avoidance_config.get('lane_changing_config', {}),
        **avoidance_config.get('unified_avoidance_config', {})
    }
    
    # Wrap with unified avoidance system
    avoidance_env = UnifiedAvoidanceWrapper(base_env, unified_config)
    
    return avoidance_env


def update_config_for_avoidance(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update existing Duckietown config to work optimally with avoidance system
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Updated configuration
    """
    # Enable obstacles for testing object avoidance
    config['env_config']['spawn_obstacles'] = True
    config['env_config']['obstacles'] = {
        'duckie': {'density': 0.2, 'static': False},
        'duckiebot': {'density': 0.1, 'static': False}
    }
    
    # Use a map that supports lane changing
    if config['env_config'].get('training_map') == 'small_loop':
        config['env_config']['training_map'] = 'loop_empty'  # Better for lane changing
    
    # Adjust episode length for more complex scenarios
    if config['env_config'].get('episode_max_steps', 500) < 1000:
        config['env_config']['episode_max_steps'] = 1000
    
    # Enable higher resolution for better object detection
    current_shape = config['env_config'].get('resized_input_shape', '(84, 84)')
    if '84' in current_shape:
        config['env_config']['resized_input_shape'] = '(128, 128)'
    
    return config


def train_with_avoidance():
    """Example of training an agent with avoidance capabilities"""
    print("Training agent with dynamic lane changing and object avoidance...")
    
    # Load base configuration
    base_config = load_config('./config/config.yml')
    
    # Update for avoidance
    base_config = update_config_for_avoidance(base_config)
    
    # Load avoidance configuration
    avoidance_config = load_avoidance_config()
    
    # Update base config for training mode
    update_config(base_config, {
        'env_config': {
            'mode': 'train',
            'training_map': 'multimap1'  # Use multiple maps for robustness
        }
    })
    
    try:
        # This would integrate with the existing training pipeline
        # For now, just create the environment to show integration
        env = create_avoidance_enabled_env(base_config['env_config'], avoidance_config)
        
        print("Environment created successfully with avoidance capabilities")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test a few steps
        obs = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Print avoidance system status
            if 'unified_avoidance' in info:
                avoidance_info = info['unified_avoidance']
                mode = avoidance_info.get('mode', 'normal')
                if mode != 'normal':
                    print(f"Step {i}: Avoidance mode: {mode}")
        
        print("Training integration test successful!")
        
    except Exception as e:
        logger.error(f"Training integration failed: {e}")
        raise


def evaluate_with_avoidance(model_path: str = None):
    """Example of evaluating an agent with avoidance capabilities"""
    print("Evaluating agent with dynamic lane changing and object avoidance...")
    
    # Load base configuration  
    base_config = load_config('./config/config.yml')
    
    # Update for evaluation mode
    update_config(base_config, {
        'env_config': {
            'mode': 'inference',
            'training_map': 'loop_empty',
            'spawn_obstacles': True,  # Test with obstacles
            'domain_rand': False
        }
    })
    
    # Load avoidance configuration
    avoidance_config = load_avoidance_config()
    
    try:
        # Create environment with avoidance
        env = create_avoidance_enabled_env(base_config['env_config'], avoidance_config)
        
        print("Evaluation environment created with avoidance capabilities")
        
        # Run evaluation episode
        obs = env.reset()
        total_reward = 0
        step_count = 0
        avoidance_activations = 0
        lane_changes = 0
        
        done = False
        while not done and step_count < 1000:
            # Use random policy for demonstration
            # In real evaluation, you would load a trained model
            action = env.action_space.sample()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Track avoidance system usage
            if 'unified_avoidance' in info:
                avoidance_info = info['unified_avoidance']
                if avoidance_info.get('mode') != 'normal':
                    avoidance_activations += 1
            
            if 'lane_changing' in info:
                lane_info = info['lane_changing']
                if lane_info.get('state') == 'lane_change_complete':
                    lane_changes += 1
        
        # Print evaluation results
        print(f"\nEvaluation Results:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward: {total_reward/step_count:.4f}")
        print(f"  Avoidance activations: {avoidance_activations}")
        print(f"  Successful lane changes: {lane_changes}")
        
        # Get detailed avoidance statistics
        if hasattr(env, 'get_avoidance_statistics'):
            stats = env.get_avoidance_statistics()
            print(f"  Avoidance statistics: {stats}")
        
        print("Evaluation with avoidance completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation with avoidance failed: {e}")
        raise


def create_deployment_package():
    """Create a deployment package for real Duckiebot"""
    print("Creating deployment package for real Duckiebot...")
    
    deployment_config = {
        'yolo': {
            'yolo_model': 'yolov5s',  # Optimized for real-time
            'confidence_threshold': 0.6,
            'device': 'cpu',
            'enable_visualization': False  # Disable for performance
        },
        'lane_changing': {
            'lane_change_duration': 90,  # Slower for safety
            'lane_change_intensity': 0.5,
            'safety_check_enabled': True,
            'auto_trigger_enabled': True
        },
        'emergency_brake_threshold': 0.3,  # Conservative
        'reaction_time_steps': 5  # More reaction time
    }
    
    # Save deployment configuration
    deployment_dir = './deployment'
    os.makedirs(deployment_dir, exist_ok=True)
    
    with open(os.path.join(deployment_dir, 'robot_config.yml'), 'w') as f:
        yaml.dump(deployment_config, f, default_flow_style=False)
    
    # Create deployment script
    deployment_script = '''#!/usr/bin/env python3
"""
Deployment script for Duckiebot with avoidance capabilities
"""

import yaml
import sys
import os

# Add your Duckiebot paths here
sys.path.append('/path/to/duckietown-rl')

from duckietown_utils.wrappers.unified_avoidance_wrapper import UnifiedAvoidanceWrapper

def load_config():
    with open('./robot_config.yml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Create your Duckiebot environment here
    # base_env = create_duckiebot_env()
    
    # Wrap with avoidance system
    # avoidance_env = UnifiedAvoidanceWrapper(base_env, config)
    
    print("Duckiebot avoidance system ready!")
    
    # Add your control loop here
    
if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(deployment_dir, 'deploy_avoidance.py'), 'w') as f:
        f.write(deployment_script)
    
    print(f"Deployment package created in: {deployment_dir}")
    print("Files created:")
    print("  - robot_config.yml: Configuration for real robot")
    print("  - deploy_avoidance.py: Deployment script template")


def main():
    """Main function demonstrating integration"""
    print("=== Duckietown Dynamic Lane Changing and Object Avoidance Integration ===")
    print("This script demonstrates integration with existing Duckietown-RL infrastructure")
    
    try:
        # Test integration with training
        print("\n1. Testing training integration...")
        train_with_avoidance()
        
        # Test integration with evaluation
        print("\n2. Testing evaluation integration...")
        evaluate_with_avoidance()
        
        # Create deployment package
        print("\n3. Creating deployment package...")
        create_deployment_package()
        
        print("\n=== Integration Complete ===")
        print("The avoidance system is successfully integrated and ready for:")
        print("  - Training with obstacles and lane changing scenarios")
        print("  - Evaluation with comprehensive avoidance metrics")
        print("  - Deployment to real Duckiebot hardware")
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())