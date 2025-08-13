#!/usr/bin/env python3
"""
Integration example for MultiObjectiveRewardWrapper.

This example demonstrates how to use the MultiObjectiveRewardWrapper
with the enhanced Duckietown RL environment.
"""

import numpy as np
from typing import Dict, Any

# Note: This example shows the integration pattern.
# Actual execution requires gym and gym-duckietown to be installed.

def create_enhanced_environment_with_multi_objective_reward():
    """
    Create an enhanced Duckietown environment with multi-objective reward.
    
    This function demonstrates the proper wrapper composition order
    and configuration for the multi-objective reward system.
    """
    
    # Import statements (would work with proper environment setup)
    try:
        import gym
        import gym_duckietown
        from duckietown_utils.wrappers import (
            YOLOObjectDetectionWrapper,
            EnhancedObservationWrapper,
            ObjectAvoidanceActionWrapper,
            LaneChangingActionWrapper,
            MultiObjectiveRewardWrapper
        )
        
        # Create base environment
        env = gym.make('Duckietown-udem1-v0')
        
        # Apply observation wrappers first
        env = YOLOObjectDetectionWrapper(
            env,
            model_path='models/yolov5s.pt',
            confidence_threshold=0.5
        )
        
        env = EnhancedObservationWrapper(
            env,
            include_detection_features=True
        )
        
        # Apply action wrappers
        env = ObjectAvoidanceActionWrapper(
            env,
            safety_distance=0.5,
            avoidance_strength=1.0
        )
        
        env = LaneChangingActionWrapper(
            env,
            lane_change_threshold=0.3,
            safety_margin=2.0
        )
        
        # Apply multi-objective reward wrapper (should be last)
        reward_weights = {
            'lane_following': 1.0,      # Primary objective
            'object_avoidance': 0.8,    # High priority for safety
            'lane_changing': 0.4,       # Medium priority for efficiency
            'efficiency': 0.3,          # Encourage forward progress
            'safety_penalty': -3.0      # Strong penalty for unsafe behavior
        }
        
        env = MultiObjectiveRewardWrapper(env, reward_weights)
        
        return env
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("This example requires gym and gym-duckietown to be installed.")
        return None


def demonstrate_reward_configuration():
    """Demonstrate different reward weight configurations for various scenarios."""
    
    # Configuration 1: Conservative driving (prioritize safety)
    conservative_weights = {
        'lane_following': 1.0,
        'object_avoidance': 1.5,    # Higher weight for safety
        'lane_changing': 0.2,       # Lower weight for lane changes
        'efficiency': 0.1,          # Lower priority for speed
        'safety_penalty': -5.0      # Very high penalty for unsafe behavior
    }
    
    # Configuration 2: Aggressive driving (prioritize efficiency)
    aggressive_weights = {
        'lane_following': 0.8,
        'object_avoidance': 0.6,
        'lane_changing': 0.8,       # Higher weight for lane changes
        'efficiency': 0.7,          # Higher priority for speed
        'safety_penalty': -2.0      # Lower penalty (still negative)
    }
    
    # Configuration 3: Balanced driving (default configuration)
    balanced_weights = {
        'lane_following': 1.0,
        'object_avoidance': 0.5,
        'lane_changing': 0.3,
        'efficiency': 0.2,
        'safety_penalty': -2.0
    }
    
    configurations = {
        'conservative': conservative_weights,
        'aggressive': aggressive_weights,
        'balanced': balanced_weights
    }
    
    print("Reward Weight Configurations:")
    print("=" * 50)
    
    for name, weights in configurations.items():
        print(f"\n{name.upper()} Configuration:")
        for component, weight in weights.items():
            print(f"  {component}: {weight}")
    
    return configurations


def simulate_reward_calculation():
    """Simulate reward calculation with different scenarios."""
    
    print("\nReward Calculation Simulation:")
    print("=" * 50)
    
    # Simulate different driving scenarios
    scenarios = {
        'perfect_lane_following': {
            'lane_following': 1.0,
            'object_avoidance': 0.1,
            'lane_changing': 0.0,
            'efficiency': 0.8,
            'safety_penalty': 0.0
        },
        'obstacle_avoidance': {
            'lane_following': 0.6,
            'object_avoidance': 0.8,
            'lane_changing': 0.0,
            'efficiency': 0.4,
            'safety_penalty': 0.0
        },
        'lane_change_maneuver': {
            'lane_following': 0.7,
            'object_avoidance': 0.2,
            'lane_changing': 1.0,
            'efficiency': 0.6,
            'safety_penalty': 0.0
        },
        'unsafe_driving': {
            'lane_following': 0.3,
            'object_avoidance': -0.5,
            'lane_changing': 0.0,
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
    }
    
    # Default weights
    weights = {
        'lane_following': 1.0,
        'object_avoidance': 0.5,
        'lane_changing': 0.3,
        'efficiency': 0.2,
        'safety_penalty': -2.0
    }
    
    for scenario_name, components in scenarios.items():
        total_reward = sum(weights[comp] * value for comp, value in components.items())
        
        print(f"\n{scenario_name.replace('_', ' ').title()}:")
        print(f"  Components: {components}")
        print(f"  Total Reward: {total_reward:.3f}")


def training_integration_example():
    """Example of integrating with PPO training."""
    
    training_code = '''
# Example training integration with RLlib PPO

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

def train_with_multi_objective_reward():
    """Train PPO agent with multi-objective reward."""
    
    # Initialize Ray
    ray.init()
    
    # Environment configuration
    env_config = {
        'reward_weights': {
            'lane_following': 1.0,
            'object_avoidance': 0.8,
            'lane_changing': 0.4,
            'efficiency': 0.3,
            'safety_penalty': -3.0
        }
    }
    
    # PPO configuration
    config = {
        'env': 'enhanced_duckietown_env',
        'env_config': env_config,
        'num_workers': 4,
        'train_batch_size': 4000,
        'sgd_minibatch_size': 128,
        'num_sgd_iter': 10,
        'lr': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_param': 0.2,
        'framework': 'torch'
    }
    
    # Create and train the agent
    agent = PPO(config=config)
    
    for i in range(1000):
        result = agent.train()
        
        # Log reward components
        if 'custom_rewards' in result.get('info', {}):
            rewards = result['info']['custom_rewards']
            print(f"Episode {i}:")
            print(f"  Lane Following: {rewards.get('lane_following', 0):.3f}")
            print(f"  Object Avoidance: {rewards.get('object_avoidance', 0):.3f}")
            print(f"  Lane Changing: {rewards.get('lane_changing', 0):.3f}")
            print(f"  Efficiency: {rewards.get('efficiency', 0):.3f}")
            print(f"  Safety Penalty: {rewards.get('safety_penalty', 0):.3f}")
            print(f"  Total: {rewards.get('total', 0):.3f}")
        
        # Save checkpoint every 100 episodes
        if i % 100 == 0:
            checkpoint = agent.save()
            print(f"Checkpoint saved: {checkpoint}")
    
    ray.shutdown()

# Usage
if __name__ == "__main__":
    train_with_multi_objective_reward()
'''
    
    print("\nTraining Integration Example:")
    print("=" * 50)
    print(training_code)


def main():
    """Main demonstration function."""
    print("MultiObjectiveRewardWrapper Integration Example")
    print("=" * 60)
    
    # Demonstrate environment creation
    print("\n1. Environment Creation:")
    env = create_enhanced_environment_with_multi_objective_reward()
    if env:
        print("✓ Enhanced environment created successfully")
    else:
        print("ℹ Environment creation skipped (dependencies not available)")
    
    # Demonstrate reward configurations
    print("\n2. Reward Configurations:")
    configurations = demonstrate_reward_configuration()
    
    # Simulate reward calculations
    print("\n3. Reward Calculation Examples:")
    simulate_reward_calculation()
    
    # Show training integration
    print("\n4. Training Integration:")
    training_integration_example()
    
    print("\n" + "=" * 60)
    print("Integration example complete!")
    print("See the code above for implementation details.")


if __name__ == "__main__":
    main()