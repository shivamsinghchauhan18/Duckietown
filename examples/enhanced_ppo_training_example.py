"""
Example script demonstrating enhanced PPO training with multi-objective rewards.
This example shows how to use the enhanced training system with object detection,
avoidance, lane changing, and curriculum learning.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
import argparse
from pathlib import Path

from experiments.train_enhanced_rllib import EnhancedPPOTrainer
from duckietown_utils.training_utils import ModelEvaluator, compare_models
from config.enhanced_config import load_enhanced_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_enhanced_training_example():
    """Run enhanced PPO training example."""
    logger.info("Starting Enhanced PPO Training Example")
    
    # Configuration
    config_updates = {
        "experiment_name": "enhanced_ppo_example",
        "seed": 42,
        "env_config": {"mode": "train"}
    }
    
    # Create enhanced trainer
    trainer = EnhancedPPOTrainer(
        config_path='./config/enhanced_ppo_config.yml',
        enhanced_config_path='./config/enhanced_config.yml',
        config_updates=config_updates
    )
    
    logger.info("Enhanced PPO Trainer initialized")
    logger.info(f"Enabled features: {trainer.enhanced_config.enabled_features}")
    
    # Run training
    logger.info("Starting training...")
    try:
        results = trainer.train(timesteps_total=50000)  # Short training for example
        logger.info(f"Training completed successfully!")
        logger.info(f"Final results: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_model_evaluation_example():
    """Run model evaluation example."""
    logger.info("Starting Model Evaluation Example")
    
    # Load enhanced configuration
    enhanced_config = load_enhanced_config('./config/enhanced_config.yml')
    
    # Create evaluator
    evaluator = ModelEvaluator(enhanced_config)
    
    # Example checkpoint path (would be real path in practice)
    checkpoint_path = "./artifacts/enhanced_ppo_example/checkpoint_001"
    
    # Environment configuration for evaluation
    env_config = {
        'mode': 'inference',
        'episode_max_steps': 500,
        'resized_input_shape': '(84, 84)',
        'crop_image_top': True,
        'top_crop_divider': 3,
        'grayscale_image': False,
        'frame_stacking': True,
        'frame_stacking_depth': 4,
        'action_type': 'heading',
        'reward_function': 'posangle',
        'distortion': True,
        'training_map': 'small_loop',
        'domain_rand': False,
        'spawn_obstacles': False
    }
    
    try:
        # Note: This would fail without a real checkpoint
        # results = evaluator.evaluate_checkpoint(
        #     checkpoint_path=checkpoint_path,
        #     env_config=env_config,
        #     num_episodes=10,
        #     render=False
        # )
        # logger.info(f"Evaluation results: {results['summary']}")
        
        logger.info("Model evaluation example setup completed")
        logger.info("(Actual evaluation requires trained checkpoint)")
        
    except Exception as e:
        logger.warning(f"Evaluation example failed (expected without checkpoint): {e}")


def run_curriculum_learning_example():
    """Demonstrate curriculum learning configuration."""
    logger.info("Starting Curriculum Learning Example")
    
    # Load enhanced configuration
    enhanced_config = load_enhanced_config('./config/enhanced_config.yml')
    
    # Example curriculum stages
    curriculum_stages = [
        {
            'name': 'basic_lane_following',
            'min_timesteps': 10000,
            'criteria': {'episode_reward_mean': 0.3},
            'env_config': {
                'spawn_obstacles': False,
                'domain_rand': False,
                'training_map': 'small_loop'
            }
        },
        {
            'name': 'static_obstacles',
            'min_timesteps': 15000,
            'criteria': {'episode_reward_mean': 0.5, 'safety_score': 0.7},
            'env_config': {
                'spawn_obstacles': True,
                'obstacles': {'duckie': {'density': 0.2, 'static': True}},
                'training_map': 'small_loop'
            }
        },
        {
            'name': 'dynamic_obstacles',
            'min_timesteps': 20000,
            'criteria': {'episode_reward_mean': 0.7, 'safety_score': 0.8},
            'env_config': {
                'spawn_obstacles': True,
                'obstacles': {'duckie': {'density': 0.3, 'static': False}},
                'training_map': 'multimap1'
            }
        }
    ]
    
    logger.info("Curriculum Learning Stages:")
    for i, stage in enumerate(curriculum_stages):
        logger.info(f"  Stage {i+1}: {stage['name']}")
        logger.info(f"    Min timesteps: {stage['min_timesteps']}")
        logger.info(f"    Criteria: {stage['criteria']}")
        logger.info(f"    Environment: {stage['env_config']}")
    
    logger.info("Curriculum learning example completed")


def run_multi_objective_reward_example():
    """Demonstrate multi-objective reward configuration."""
    logger.info("Starting Multi-Objective Reward Example")
    
    # Load enhanced configuration
    enhanced_config = load_enhanced_config('./config/enhanced_config.yml')
    
    # Display reward configuration
    reward_config = enhanced_config.reward
    logger.info("Multi-Objective Reward Configuration:")
    logger.info(f"  Lane Following Weight: {reward_config.lane_following_weight}")
    logger.info(f"  Object Avoidance Weight: {reward_config.object_avoidance_weight}")
    logger.info(f"  Lane Change Weight: {reward_config.lane_change_weight}")
    logger.info(f"  Efficiency Weight: {reward_config.efficiency_weight}")
    logger.info(f"  Safety Penalty Weight: {reward_config.safety_penalty_weight}")
    logger.info(f"  Collision Penalty: {reward_config.collision_penalty}")
    
    # Example reward calculation
    example_components = {
        'lane_following': 0.8,
        'object_avoidance': 0.6,
        'lane_change': 0.4,
        'efficiency': 0.7,
        'safety_penalty': -0.1
    }
    
    total_reward = (
        example_components['lane_following'] * reward_config.lane_following_weight +
        example_components['object_avoidance'] * reward_config.object_avoidance_weight +
        example_components['lane_change'] * reward_config.lane_change_weight +
        example_components['efficiency'] * reward_config.efficiency_weight +
        example_components['safety_penalty'] * reward_config.safety_penalty_weight
    )
    
    logger.info(f"\nExample Reward Calculation:")
    logger.info(f"  Components: {example_components}")
    logger.info(f"  Total Weighted Reward: {total_reward:.3f}")
    
    logger.info("Multi-objective reward example completed")


def run_performance_monitoring_example():
    """Demonstrate performance monitoring and logging."""
    logger.info("Starting Performance Monitoring Example")
    
    # Load enhanced configuration
    enhanced_config = load_enhanced_config('./config/enhanced_config.yml')
    
    # Display logging configuration
    logging_config = enhanced_config.logging
    logger.info("Enhanced Logging Configuration:")
    logger.info(f"  Log Level: {logging_config.log_level}")
    logger.info(f"  Log Detections: {logging_config.log_detections}")
    logger.info(f"  Log Actions: {logging_config.log_actions}")
    logger.info(f"  Log Rewards: {logging_config.log_rewards}")
    logger.info(f"  Log Performance: {logging_config.log_performance}")
    logger.info(f"  Console Logging: {logging_config.console_logging}")
    
    # Example metrics that would be logged
    example_metrics = {
        'episode_reward_mean': 0.75,
        'lane_following_reward_mean': 0.8,
        'object_avoidance_reward_mean': 0.6,
        'lane_change_reward_mean': 0.4,
        'safety_score': 0.85,
        'efficiency_score': 0.7,
        'objects_detected_total': 12,
        'avg_detection_confidence': 0.82,
        'avoidance_actions_total': 8,
        'lane_change_success_rate': 0.75,
        'safety_violations_total': 2,
        'processing_time_detection_mean': 0.025,
        'processing_time_action_mean': 0.008
    }
    
    logger.info(f"\nExample Training Metrics:")
    for metric, value in example_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info("Performance monitoring example completed")


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced PPO Training Examples')
    parser.add_argument('--example', type=str, default='all',
                       choices=['all', 'training', 'evaluation', 'curriculum', 'rewards', 'monitoring'],
                       help='Which example to run')
    parser.add_argument('--train', action='store_true',
                       help='Run actual training (requires proper setup)')
    
    args = parser.parse_args()
    
    logger.info("Enhanced PPO Training Examples")
    logger.info("=" * 50)
    
    try:
        if args.example in ['all', 'curriculum']:
            run_curriculum_learning_example()
            logger.info("")
        
        if args.example in ['all', 'rewards']:
            run_multi_objective_reward_example()
            logger.info("")
        
        if args.example in ['all', 'monitoring']:
            run_performance_monitoring_example()
            logger.info("")
        
        if args.example in ['all', 'evaluation']:
            run_model_evaluation_example()
            logger.info("")
        
        if args.example in ['all', 'training'] or args.train:
            if args.train:
                run_enhanced_training_example()
            else:
                logger.info("Training Example Setup:")
                logger.info("  To run actual training, use --train flag")
                logger.info("  This requires proper environment setup and dependencies")
                logger.info("  See SETUP_GUIDE.md for installation instructions")
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()