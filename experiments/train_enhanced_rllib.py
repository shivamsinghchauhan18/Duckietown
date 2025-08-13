"""
Enhanced PPO Training Script for Duckietown RL with Multi-Objective Rewards
Supports object detection, avoidance, lane changing, and curriculum learning.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import CSVLogger, TBXLogger

from config.paths import ArtifactPaths
from config.config import load_config, print_config, dump_config, update_config, find_and_load_config_by_seed
from config.enhanced_config import load_enhanced_config, EnhancedRLConfig

from duckietown_utils.env import launch_and_wrap_enhanced_env
from duckietown_utils.utils import seed
from duckietown_utils.enhanced_rllib_callbacks import (
    EnhancedRLLibCallbacks, 
    CurriculumLearningCallback,
    ModelCheckpointCallback
)
from duckietown_utils.enhanced_rllib_loggers import (
    EnhancedTensorboardLogger, 
    EnhancedWeightsAndBiasesLogger
)
from duckietown_utils.enhanced_logger import EnhancedLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EnhancedPPOTrainer:
    """Enhanced PPO Trainer with multi-objective rewards and curriculum learning."""
    
    def __init__(self, config_path: str = './config/config.yml', 
                 enhanced_config_path: str = './config/enhanced_config.yml',
                 config_updates: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced PPO Trainer.
        
        Args:
            config_path: Path to standard configuration file
            enhanced_config_path: Path to enhanced configuration file
            config_updates: Dictionary of configuration updates
        """
        self.config = load_config(config_path, config_updates=config_updates or {})
        self.enhanced_config = load_enhanced_config(enhanced_config_path)
        self.enhanced_logger = EnhancedLogger(self.enhanced_config.logging)
        
        # Set random seed
        seed(self.config.get('seed', 1234))
        
        # Initialize paths
        self.paths = ArtifactPaths(
            self.config['experiment_name'], 
            self.config['seed'], 
            algo_name=self.config['algo']
        )
        
        # Initialize curriculum learning
        self.curriculum_callback = CurriculumLearningCallback(self.enhanced_config)
        self.checkpoint_callback = ModelCheckpointCallback(self.paths, self.enhanced_config)
        
        logger.info("Enhanced PPO Trainer initialized")
    
    def setup_training_config(self) -> Dict[str, Any]:
        """
        Setup training configuration with enhanced callbacks and loggers.
        
        Returns:
            Complete training configuration dictionary
        """
        # Create enhanced callbacks
        enhanced_callbacks = EnhancedRLLibCallbacks(
            enhanced_config=self.enhanced_config,
            enhanced_logger=self.enhanced_logger,
            curriculum_callback=self.curriculum_callback,
            checkpoint_callback=self.checkpoint_callback
        )
        
        # Update RLLib configuration with enhanced callbacks
        rllib_config = self.config["rllib_config"].copy()
        rllib_config.update({
            'env': 'EnhancedDuckietown',
            'callbacks': enhanced_callbacks,
            'env_config': {
                **self.config["env_config"],
                'enhanced_config': self.enhanced_config
            },
            # Multi-objective reward configuration
            'multiagent': {
                'policies': {
                    'default_policy': (None, None, None, {
                        'reward_weights': {
                            'lane_following': self.enhanced_config.reward.lane_following_weight,
                            'object_avoidance': self.enhanced_config.reward.object_avoidance_weight,
                            'lane_change': self.enhanced_config.reward.lane_change_weight,
                            'efficiency': self.enhanced_config.reward.efficiency_weight,
                            'safety_penalty': self.enhanced_config.reward.safety_penalty_weight
                        }
                    })
                }
            }
        })
        
        # Add curriculum learning configuration
        if hasattr(self.enhanced_config, 'curriculum_learning'):
            rllib_config['curriculum_learning'] = {
                'enabled': True,
                'stages': self.enhanced_config.curriculum_learning.stages,
                'transition_criteria': self.enhanced_config.curriculum_learning.transition_criteria
            }
        
        return rllib_config
    
    def register_enhanced_environment(self):
        """Register enhanced Duckietown environment with Ray."""
        def create_enhanced_env(env_config):
            return launch_and_wrap_enhanced_env(
                env_config, 
                enhanced_config=env_config.get('enhanced_config', self.enhanced_config)
            )
        
        register_env('EnhancedDuckietown', create_enhanced_env)
        logger.info("Enhanced Duckietown environment registered")
    
    def setup_restoration(self) -> Optional[str]:
        """
        Setup model restoration from checkpoint if specified.
        
        Returns:
            Path to checkpoint file or None
        """
        if self.config.get('restore_seed', -1) >= 0:
            pretrained_config, checkpoint_path = find_and_load_config_by_seed(
                self.config['restore_seed'],
                preselected_experiment_idx=self.config.get('restore_experiment_idx', 0),
                preselected_checkpoint_idx=self.config.get('restore_checkpoint_idx', 0)
            )
            logger.warning(f"Restoring from checkpoint: {checkpoint_path}")
            
            # Merge configurations while preserving enhanced settings
            merged_config = pretrained_config.copy()
            merged_config.update(self.config)
            self.config = merged_config
            
            return checkpoint_path
        return None
    
    def backup_code(self):
        """Backup source code for reproducibility."""
        backup_dirs = ['./duckietown_utils', './experiments', './config']
        for dir_path in backup_dirs:
            if os.path.exists(dir_path):
                os.system(f'cp -ar {dir_path} {self.paths.code_backup_path}/')
        logger.info(f"Code backed up to {self.paths.code_backup_path}")
    
    def train(self, timesteps_total: Optional[int] = None) -> Dict[str, Any]:
        """
        Run enhanced PPO training.
        
        Args:
            timesteps_total: Total timesteps to train (overrides config)
            
        Returns:
            Training results dictionary
        """
        # Setup Ray
        ray.init(**self.config["ray_init_config"])
        
        # Register enhanced environment
        self.register_enhanced_environment()
        
        # Setup training configuration
        rllib_config = self.setup_training_config()
        
        # Setup restoration
        checkpoint_path = self.setup_restoration()
        
        # Backup code
        self.backup_code()
        
        # Save configuration
        dump_config(self.config, self.paths.experiment_base_path)
        self.enhanced_config.save(self.paths.experiment_base_path / 'enhanced_config.yml')
        
        # Print configuration
        print_config(self.config)
        logger.info(f"Enhanced configuration: {self.enhanced_config}")
        
        # Setup loggers
        loggers = [
            CSVLogger, 
            TBXLogger, 
            EnhancedTensorboardLogger, 
            EnhancedWeightsAndBiasesLogger
        ]
        
        # Run training
        total_timesteps = timesteps_total or self.config.get("timesteps_total", 1000000)
        
        logger.info(f"Starting enhanced PPO training for {total_timesteps} timesteps")
        
        results = tune.run(
            PPOTrainer,
            stop={'timesteps_total': total_timesteps},
            config=rllib_config,
            local_dir="./artifacts",
            checkpoint_at_end=True,
            trial_name_creator=lambda trial: f"EnhancedPPO_{trial.trainable_name}",
            name=self.paths.experiment_folder,
            keep_checkpoints_num=3,  # Keep more checkpoints for enhanced training
            checkpoint_score_attr="episode_reward_mean",
            checkpoint_freq=5,  # More frequent checkpointing
            restore=checkpoint_path,
            loggers=loggers,
            # Enhanced training configuration
            resources_per_trial={
                "cpu": self.config["ray_init_config"].get("num_cpus", 1),
                "gpu": rllib_config.get("num_gpus", 0)
            },
            # Progress reporting
            progress_reporter=tune.CLIReporter(
                metric_columns=[
                    "episode_reward_mean", 
                    "episode_len_mean",
                    "custom_metrics/lane_following_reward_mean",
                    "custom_metrics/object_avoidance_reward_mean", 
                    "custom_metrics/lane_change_reward_mean",
                    "custom_metrics/safety_violations_mean",
                    "custom_metrics/curriculum_stage"
                ]
            )
        )
        
        logger.info("Enhanced PPO training completed")
        return results
    
    def evaluate_model(self, checkpoint_path: str, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate trained model performance.
        
        Args:
            checkpoint_path: Path to model checkpoint
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        # Setup evaluation configuration
        eval_config = self.config.copy()
        eval_config['env_config']['mode'] = 'inference'
        eval_config['rllib_config'].update({
            'explore': False,
            'num_workers': 0,
            'num_gpus': 0
        })
        
        # Create trainer and restore model
        trainer = PPOTrainer(config=eval_config['rllib_config'])
        trainer.restore(checkpoint_path)
        
        # Run evaluation episodes
        evaluation_results = {
            'episodes': [],
            'metrics': {
                'mean_reward': 0.0,
                'mean_episode_length': 0.0,
                'success_rate': 0.0,
                'collision_rate': 0.0,
                'lane_following_score': 0.0,
                'object_avoidance_score': 0.0,
                'lane_change_score': 0.0
            }
        }
        
        # Create evaluation environment
        eval_env = launch_and_wrap_enhanced_env(
            eval_config['env_config'], 
            self.enhanced_config
        )
        
        total_reward = 0.0
        total_length = 0.0
        successes = 0
        collisions = 0
        
        for episode in range(num_episodes):
            obs = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            episode_data = {
                'episode': episode,
                'reward': 0.0,
                'length': 0,
                'success': False,
                'collision': False,
                'metrics': {}
            }
            
            while not done:
                action = trainer.compute_action(obs)
                obs, reward, done, info = eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track episode metrics
                if info and 'custom_metrics' in info:
                    for key, value in info['custom_metrics'].items():
                        if key not in episode_data['metrics']:
                            episode_data['metrics'][key] = []
                        episode_data['metrics'][key].append(value)
            
            # Process episode results
            episode_data['reward'] = episode_reward
            episode_data['length'] = episode_length
            episode_data['success'] = episode_reward > 0  # Simple success criterion
            episode_data['collision'] = info.get('collision', False) if info else False
            
            evaluation_results['episodes'].append(episode_data)
            
            total_reward += episode_reward
            total_length += episode_length
            if episode_data['success']:
                successes += 1
            if episode_data['collision']:
                collisions += 1
        
        # Calculate aggregate metrics
        evaluation_results['metrics'].update({
            'mean_reward': total_reward / num_episodes,
            'mean_episode_length': total_length / num_episodes,
            'success_rate': successes / num_episodes,
            'collision_rate': collisions / num_episodes
        })
        
        # Calculate component-specific scores
        for metric_name in ['lane_following_score', 'object_avoidance_score', 'lane_change_score']:
            scores = []
            for episode_data in evaluation_results['episodes']:
                if metric_name in episode_data['metrics']:
                    scores.extend(episode_data['metrics'][metric_name])
            if scores:
                evaluation_results['metrics'][metric_name] = sum(scores) / len(scores)
        
        logger.info(f"Evaluation completed: {evaluation_results['metrics']}")
        return evaluation_results


def main():
    """Main training function with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced PPO Training for Duckietown RL')
    parser.add_argument('--config', type=str, default='./config/config.yml',
                       help='Path to standard configuration file')
    parser.add_argument('--enhanced-config', type=str, default='./config/enhanced_config.yml',
                       help='Path to enhanced configuration file')
    parser.add_argument('--experiment-name', type=str, default='EnhancedPPO',
                       help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total timesteps to train')
    parser.add_argument('--evaluate', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    # Setup configuration updates
    config_updates = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "env_config": {"mode": "train"}
    }
    
    # Create trainer
    trainer = EnhancedPPOTrainer(
        config_path=args.config,
        enhanced_config_path=args.enhanced_config,
        config_updates=config_updates
    )
    
    if args.evaluate:
        # Run evaluation
        results = trainer.evaluate_model(args.evaluate, args.eval_episodes)
        print(f"Evaluation Results: {results['metrics']}")
    else:
        # Run training
        results = trainer.train(timesteps_total=args.timesteps)
        print(f"Training completed: {results}")


if __name__ == "__main__":
    main()