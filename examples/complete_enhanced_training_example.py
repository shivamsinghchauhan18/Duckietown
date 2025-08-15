#!/usr/bin/env python3
"""
Complete Enhanced Duckietown RL Training Example

This script demonstrates a complete training pipeline using all enhanced features:
- YOLO object detection
- Object avoidance
- Lane changing
- Multi-objective rewards
- Enhanced logging and visualization

Usage:
    python examples/complete_enhanced_training_example.py --config config/enhanced_config.yml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import ray
import torch
import numpy as np
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.env import launch_and_wrap_enhanced_env
from duckietown_utils.enhanced_logger import EnhancedLogger
from duckietown_utils.visualization_manager import VisualizationManager
from duckietown_utils.training_utils import EnhancedTrainingCallbacks


def create_enhanced_env(env_config):
    """
    Environment factory function for RLLib.
    
    Args:
        env_config (dict): Environment configuration from RLLib
        
    Returns:
        gym.Env: Enhanced Duckietown environment
    """
    # Load configuration
    config_path = env_config.get('config_path', 'config/enhanced_config.yml')
    config = EnhancedRLConfig.from_yaml(config_path)
    
    # Override config with env_config parameters
    if 'map_name' in env_config:
        map_name = env_config['map_name']
    else:
        map_name = 'loop_obstacles'
    
    # Create enhanced environment
    env = launch_and_wrap_enhanced_env(
        map_name=map_name,
        config=config,
        seed=env_config.get('seed', None)
    )
    
    return env


def setup_training_config(config_path: str, experiment_name: str):
    """
    Set up PPO training configuration for enhanced environment.
    
    Args:
        config_path (str): Path to enhanced RL configuration file
        experiment_name (str): Name for the training experiment
        
    Returns:
        dict: PPO training configuration
    """
    # Load enhanced config to get observation/action space info
    enhanced_config = EnhancedRLConfig.from_yaml(config_path)
    
    training_config = {
        # Environment configuration
        "env": "enhanced_duckietown",
        "env_config": {
            "config_path": config_path,
            "map_name": "loop_obstacles"
        },
        
        # Framework and parallelization
        "framework": "torch",
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        
        # Training parameters
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "lr_schedule": [
            [0, 3e-4],
            [1000000, 1e-4],
            [2000000, 5e-5]
        ],
        
        # PPO-specific parameters
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        
        # Neural network configuration
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "use_lstm": False,
            "max_seq_len": 20,
        },
        
        # Exploration configuration
        "exploration_config": {
            "type": "StochasticSampling",
        },
        
        # Evaluation configuration
        "evaluation_interval": 50,
        "evaluation_num_episodes": 10,
        "evaluation_config": {
            "env_config": {
                "config_path": config_path,
                "map_name": "loop_obstacles"
            },
            "explore": False,
        },
        
        # Callbacks for enhanced logging
        "callbacks": EnhancedTrainingCallbacks,
        
        # Checkpointing
        "checkpoint_freq": 100,
        "keep_checkpoints_num": 5,
        
        # Logging
        "log_level": "INFO",
        "metrics_smoothing_episodes": 100,
    }
    
    return training_config


def run_training(config_path: str, experiment_name: str, num_iterations: int = 1000):
    """
    Run enhanced Duckietown RL training.
    
    Args:
        config_path (str): Path to configuration file
        experiment_name (str): Name for the experiment
        num_iterations (int): Number of training iterations
    """
    print(f"Starting enhanced Duckietown RL training: {experiment_name}")
    print(f"Configuration: {config_path}")
    print(f"Iterations: {num_iterations}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=8,
            num_gpus=1 if torch.cuda.is_available() else 0,
            object_store_memory=2000000000,  # 2GB
        )
    
    # Register environment
    register_env("enhanced_duckietown", create_enhanced_env)
    
    # Set up training configuration
    training_config = setup_training_config(config_path, experiment_name)
    
    # Create trainer
    trainer = PPOTrainer(config=training_config)
    
    # Set up logging
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = EnhancedLogger(
        log_level='INFO',
        log_dir=log_dir,
        enable_structured_logging=True
    )
    
    # Training loop
    best_reward = float('-inf')
    training_start_time = time.time()
    
    try:
        for iteration in range(num_iterations):
            iteration_start_time = time.time()
            
            # Train one iteration
            result = trainer.train()
            
            iteration_time = time.time() - iteration_start_time
            
            # Extract key metrics
            episode_reward_mean = result.get('episode_reward_mean', 0)
            episode_len_mean = result.get('episode_len_mean', 0)
            episodes_this_iter = result.get('episodes_this_iter', 0)
            
            # Log progress
            if iteration % 10 == 0:
                print(f"\nIteration {iteration}")
                print(f"  Episode Reward Mean: {episode_reward_mean:.3f}")
                print(f"  Episode Length Mean: {episode_len_mean:.1f}")
                print(f"  Episodes This Iter: {episodes_this_iter}")
                print(f"  Iteration Time: {iteration_time:.1f}s")
                
                # Log to enhanced logger
                logger.log_training_metrics({
                    'iteration': iteration,
                    'episode_reward_mean': episode_reward_mean,
                    'episode_len_mean': episode_len_mean,
                    'episodes_this_iter': episodes_this_iter,
                    'iteration_time': iteration_time
                })
            
            # Save best model
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                checkpoint_path = trainer.save(f"{log_dir}/best_model")
                print(f"  New best model saved: {checkpoint_path}")
                logger.log_event('best_model_saved', {
                    'reward': best_reward,
                    'iteration': iteration,
                    'checkpoint_path': checkpoint_path
                })
            
            # Regular checkpointing
            if iteration % 100 == 0 and iteration > 0:
                checkpoint_path = trainer.save(f"{log_dir}/checkpoint_{iteration}")
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Evaluation logging
            if 'evaluation' in result:
                eval_reward = result['evaluation']['episode_reward_mean']
                print(f"  Evaluation Reward: {eval_reward:.3f}")
                logger.log_evaluation_metrics({
                    'iteration': iteration,
                    'evaluation_reward': eval_reward,
                    'evaluation_episodes': result['evaluation'].get('episodes_this_iter', 0)
                })
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        logger.log_error('training_failed', str(e))
        raise
    
    finally:
        # Final checkpoint
        final_checkpoint = trainer.save(f"{log_dir}/final_model")
        print(f"\nFinal model saved: {final_checkpoint}")
        
        # Training summary
        total_time = time.time() - training_start_time
        print(f"\nTraining Summary:")
        print(f"  Total Time: {total_time/3600:.2f} hours")
        print(f"  Best Reward: {best_reward:.3f}")
        print(f"  Final Checkpoint: {final_checkpoint}")
        
        logger.log_training_summary({
            'total_time': total_time,
            'best_reward': best_reward,
            'final_checkpoint': final_checkpoint,
            'total_iterations': iteration + 1
        })
        
        # Clean up
        trainer.stop()
        ray.shutdown()


def run_evaluation(checkpoint_path: str, config_path: str, num_episodes: int = 10):
    """
    Evaluate a trained model with visualization.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        config_path (str): Path to configuration file
        num_episodes (int): Number of episodes to evaluate
    """
    print(f"Evaluating model: {checkpoint_path}")
    print(f"Configuration: {config_path}")
    print(f"Episodes: {num_episodes}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(local_mode=True)
    
    # Register environment
    register_env("enhanced_duckietown", create_enhanced_env)
    
    # Set up configuration
    training_config = setup_training_config(config_path, "evaluation")
    training_config["num_workers"] = 0  # No parallel workers for evaluation
    
    # Create trainer and load model
    trainer = PPOTrainer(config=training_config)
    trainer.restore(checkpoint_path)
    
    # Create evaluation environment
    env = create_enhanced_env({
        "config_path": config_path,
        "map_name": "loop_obstacles"
    })
    
    # Set up visualization
    viz_manager = VisualizationManager(
        enable_detection_viz=True,
        enable_action_viz=True,
        enable_reward_viz=True,
        save_visualizations=True,
        output_dir="evaluation_viz"
    )
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from trained model
            action = trainer.compute_action(obs, explore=False)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Update visualization
            viz_manager.update(obs, action, reward, info)
            
            # Print step info occasionally
            if episode_length % 50 == 0:
                print(f"  Step {episode_length}, Reward: {episode_reward:.3f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {episode + 1} completed:")
        print(f"    Total Reward: {episode_reward:.3f}")
        print(f"    Episode Length: {episode_length}")
    
    # Evaluation summary
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Summary:")
    print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  Mean Episode Length: {mean_length:.1f}")
    print(f"  Success Rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100:.1f}%")
    
    # Clean up
    env.close()
    trainer.stop()
    ray.shutdown()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Enhanced Duckietown RL Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/enhanced_config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="enhanced_training",
        help="Name for the training experiment"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate"],
        default="train",
        help="Mode: train or evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for evaluation"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )
    
    args = parser.parse_args()
    
    # Validate configuration file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if args.mode == "train":
        run_training(args.config, args.experiment_name, args.iterations)
    
    elif args.mode == "evaluate":
        if not args.checkpoint:
            print("Error: Checkpoint path required for evaluation mode")
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        run_evaluation(args.checkpoint, args.config, args.eval_episodes)


if __name__ == "__main__":
    main()