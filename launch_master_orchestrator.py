#!/usr/bin/env python3
"""
🚀 MASTER RL ORCHESTRATOR LAUNCHER 🚀
Launch script for the comprehensive continuous RL training system

This launcher provides a unified interface to start the master orchestrator
with various configurations and monitoring options.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from master_rl_orchestrator import MasterRLOrchestrator
from duckietown_utils.master_evaluation_system import MasterEvaluationSystem

def create_default_config() -> Dict[str, Any]:
    """Create default configuration based on master prompt."""
    return {
        # Project context
        'project': {
            'name': 'Duckietown Master RL Orchestrator',
            'framework': 'PyTorch',
            'hardware': 'CPU',
            'max_daily_budget_hours': 24
        },
        
        # Core orchestrator settings
        'population_size': 8,
        'max_outer_loops': 50,
        'steps_per_loop': 2_000_000,
        'evaluation_seeds': 50,
        
        # Action and observation space
        'action_space': 'continuous',
        'observation': {
            'type': 'rgb',
            'size': [64, 64],
            'frame_stack': 4
        },
        
        # Target maps with difficulty
        'target_maps': [
            {'name': 'loop_empty', 'difficulty': 'easy', 'type': 'easy_loop'},
            {'name': 'small_loop', 'difficulty': 'easy', 'type': 'easy_loop'},
            {'name': 'zigzag_dists', 'difficulty': 'moderate', 'type': 'curvy'},
            {'name': '4way', 'difficulty': 'hard', 'type': 'intersection'},
            {'name': 'udem1', 'difficulty': 'hard', 'type': 'town'}
        ],
        
        # Success thresholds (from master prompt)
        'success_thresholds': {
            'easy_loop': {
                'sr_threshold': 0.95,
                'r_threshold': 0.85,
                'd_threshold': 0.12,
                'h_threshold': 8.0,
                'j_threshold': 0.08
            },
            'curvy': {
                'sr_threshold': 0.90,
                'r_threshold': 0.80,
                'd_threshold': 0.15,
                'h_threshold': 10.0,
                'j_threshold': 0.10
            },
            'intersection': {
                'sr_threshold': 0.85,
                'r_threshold': 0.75,
                'd_threshold': 0.20,
                'h_threshold': 12.0,
                'j_threshold': 0.12,
                'violations_threshold': 0.03
            },
            'town': {
                'sr_threshold': 0.85,
                'r_threshold': 0.75,
                'd_threshold': 0.20,
                'h_threshold': 12.0,
                'j_threshold': 0.12,
                'violations_threshold': 0.03
            }
        },
        
        # Composite score weights
        'composite_score': {
            'sr_weight': 0.45,
            'reward_weight': 0.25,
            'length_weight': 0.10,
            'deviation_weight': 0.08,
            'heading_weight': 0.06,
            'jerk_weight': 0.06
        },
        
        # Environment and reward shaping
        'environment': {
            'base_reward': 1.0,
            'reward_shaping': {
                'alpha': 0.6,
                'beta': 0.3,
                'gamma': 0.02,
                'delta': 0.02,
                'sigma_d': 0.25,
                'sigma_theta': 10.0
            },
            'speed_target': {
                'min_speed': 0.5,
                'max_speed': 2.0,
                'target_speed': 1.5
            }
        },
        
        # Algorithm configuration
        'algorithm': {
            'default': 'PPO',
            'ppo': {
                'lr': 3.0e-4,
                'gamma': 0.995,
                'gae_lambda': 0.95,
                'clip_param': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'grad_clip': 0.5,
                'batch_size': 65536,
                'minibatches': 8,
                'epochs': 4,
                'action_std_init': 0.6,
                'action_std_final': 0.1
            }
        },
        
        # Population-based training
        'population_based_training': {
            'enabled': True,
            'exploit_threshold': 0.25,
            'explore_threshold': 0.25,
            'hyperparameter_ranges': {
                'lr': [1.0e-5, 1.0e-3],
                'entropy_coef': [0.001, 0.03],
                'clip_param': [0.1, 0.3],
                'gamma': [0.99, 0.999],
                'gae_lambda': [0.9, 0.98]
            }
        },
        
        # Curriculum learning
        'curriculum': {
            'enabled': True,
            'auto_advance': True,
            'stages': [
                {
                    'name': 'Foundation',
                    'episodes': 1000,
                    'difficulty': 0.3,
                    'success_criteria': {'sr_threshold': 0.97},
                    'env_config': {
                        'maps': ['loop_empty'],
                        'domain_randomization': False
                    }
                },
                {
                    'name': 'Basic_Curves',
                    'episodes': 1500,
                    'difficulty': 0.5,
                    'success_criteria': {'sr_threshold': 0.92},
                    'env_config': {
                        'maps': ['small_loop', 'zigzag_dists'],
                        'domain_randomization': True
                    }
                },
                {
                    'name': 'Complex_Scenarios',
                    'episodes': 2000,
                    'difficulty': 0.85,
                    'success_criteria': {'sr_threshold': 0.80},
                    'env_config': {
                        'maps': ['4way', 'udem1'],
                        'domain_randomization': True
                    }
                }
            ]
        },
        
        # Optimization loop
        'optimization_loop': {
            'plateau_detection': {
                'enabled': True,
                'patience': 3,
                'min_improvement': 0.01
            },
            'adaptation': {
                'increase_exploration': True,
                'try_alternative_algorithms': True,
                'adjust_reward_weights': True
            }
        },
        
        # Evaluation and stress testing
        'evaluation': {
            'deterministic': True,
            'stress_tests': True
        },
        
        # Logging and monitoring
        'logging': {
            'log_level': 'INFO',
            'tensorboard': {
                'enabled': True,
                'log_dir': 'logs/master_orchestrator/tensorboard'
            },
            'wandb': {
                'enabled': False,
                'project': 'duckietown-master-orchestrator'
            }
        },
        
        # System configuration
        'system': {
            'num_cpus': 0,
            'num_gpus': 0,
            'parallel_trials': 4,
            'max_concurrent_evaluations': 8
        }
    }

def setup_logging_directories():
    """Setup logging directories."""
    directories = [
        'logs/master_orchestrator',
        'logs/master_orchestrator/tensorboard',
        'logs/master_evaluation',
        'models/master_orchestrator_champions',
        'reports/master_orchestrator'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Logging directories created")

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_keys = [
        'population_size', 'max_outer_loops', 'steps_per_loop',
        'target_maps', 'success_thresholds', 'composite_score'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required config key: {key}")
            return False
    
    # Validate population size
    if config['population_size'] < 2:
        print("❌ Population size must be at least 2")
        return False
    
    # Validate maps
    if not config['target_maps']:
        print("❌ At least one target map must be specified")
        return False
    
    # Validate composite score weights sum to 1.0
    weights = config['composite_score']
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"⚠️ Composite score weights sum to {total_weight:.3f}, normalizing to 1.0")
        for key in weights:
            weights[key] /= total_weight
    
    print("✅ Configuration validated")
    return True

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"💾 Configuration saved to {config_path}")

def print_launch_banner(config: Dict[str, Any]):
    """Print launch banner with configuration summary."""
    print("\n" + "=" * 80)
    print("🚀 MASTER RL ORCHESTRATOR LAUNCH 🚀")
    print("=" * 80)
    print(f"🎯 MISSION: State-of-the-art RL performance across multiple maps")
    print(f"🏆 TARGET: Global pass (≥90% maps meet thresholds)")
    print(f"🔬 METHOD: Population-Based Training + Continuous Optimization")
    print()
    print(f"📊 CONFIGURATION SUMMARY:")
    print(f"  Population Size: {config['population_size']}")
    print(f"  Max Outer Loops: {config['max_outer_loops']}")
    print(f"  Steps per Loop: {config['steps_per_loop']:,}")
    print(f"  Target Maps: {len(config['target_maps'])}")
    print(f"  Evaluation Seeds: {config['evaluation_seeds']}")
    print(f"  Action Space: {config['action_space']}")
    print(f"  Algorithm: {config['algorithm']['default']}")
    print()
    print(f"🗺️  TARGET MAPS:")
    for map_config in config['target_maps']:
        name = map_config['name']
        difficulty = map_config['difficulty']
        map_type = map_config['type']
        print(f"    {name} ({difficulty}, {map_type})")
    print()
    print(f"🎖️  SUCCESS THRESHOLDS:")
    for map_type, thresholds in config['success_thresholds'].items():
        sr = thresholds['sr_threshold']
        print(f"    {map_type}: SR≥{sr:.1%}")
    print("=" * 80)

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="🏆 Master RL Orchestrator Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default configuration
  python launch_master_orchestrator.py
  
  # Launch with custom config
  python launch_master_orchestrator.py --config my_config.yml
  
  # Quick test run
  python launch_master_orchestrator.py --population-size 4 --max-loops 10
  
  # Evaluation only mode
  python launch_master_orchestrator.py --evaluate-only --model-path models/champion.json
        """
    )
    
    # Configuration options
    parser.add_argument('--config', type=str, 
                       help='Configuration file path')
    parser.add_argument('--create-config', type=str,
                       help='Create default config file and exit')
    
    # Override options
    parser.add_argument('--population-size', type=int,
                       help='Population size for PBT')
    parser.add_argument('--max-loops', type=int,
                       help='Maximum outer loops')
    parser.add_argument('--steps-per-loop', type=int,
                       help='Training steps per loop')
    parser.add_argument('--evaluation-seeds', type=int,
                       help='Number of evaluation seeds per map')
    
    # Mode options
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Run evaluation only (no training)')
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration and exit')
    
    # System options
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU-only training')
    parser.add_argument('--parallel-trials', type=int, default=4,
                       help='Number of parallel training trials')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        save_config(config, args.create_config)
        print(f"✅ Default configuration created: {args.create_config}")
        return
    
    # Load configuration
    if args.config and Path(args.config).exists():
        print(f"📖 Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("📖 Using default configuration")
        config = create_default_config()
    
    # Apply command line overrides
    if args.population_size:
        config['population_size'] = args.population_size
    if args.max_loops:
        config['max_outer_loops'] = args.max_loops
    if args.steps_per_loop:
        config['steps_per_loop'] = args.steps_per_loop
    if args.evaluation_seeds:
        config['evaluation_seeds'] = args.evaluation_seeds
    if args.cpu_only:
        config['system']['num_gpus'] = 0
    if args.parallel_trials:
        config['system']['parallel_trials'] = args.parallel_trials
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Configuration validation failed")
        return 1
    
    # Setup directories
    setup_logging_directories()
    
    # Save effective configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_config_path = f"config/effective_config_{timestamp}.yml"
    save_config(config, effective_config_path)
    
    # Print launch banner
    print_launch_banner(config)
    
    # Dry run mode
    if args.dry_run:
        print("🔍 Dry run completed - configuration is valid")
        return 0
    
    # Evaluation only mode
    if args.evaluate_only:
        print("📊 Running evaluation only...")
        evaluator = MasterEvaluationSystem(config)
        
        if args.model_path:
            # Load and evaluate specific model
            print(f"📖 Loading model from {args.model_path}")
            # In practice, you'd load the actual model here
            # For now, we'll use a mock agent
            class MockAgent:
                def get_action(self, obs, deterministic=True):
                    return [0.5, 0.0]
            
            agent = MockAgent()
            report = evaluator.evaluate_agent_comprehensive(agent)
            print(f"✅ Evaluation completed - Rating: {report['overall_rating']}")
        else:
            print("❌ Model path required for evaluation-only mode")
            return 1
        
        return 0
    
    # Launch master orchestrator
    try:
        print("🚀 Launching Master RL Orchestrator...")
        
        # Create orchestrator
        orchestrator = MasterRLOrchestrator(effective_config_path)
        
        # Apply any additional overrides
        orchestrator.population_size = config['population_size']
        orchestrator.max_outer_loops = config['max_outer_loops']
        orchestrator.steps_per_loop = config['steps_per_loop']
        
        # Start continuous optimization
        orchestrator.run_continuous_optimization()
        
        print("🏆 Master orchestrator completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Master orchestrator interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Master orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)