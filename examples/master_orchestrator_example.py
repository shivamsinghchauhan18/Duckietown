#!/usr/bin/env python3
"""
🏆 MASTER ORCHESTRATOR EXAMPLE 🏆
Example usage of the Master RL Training Orchestrator

This example demonstrates how to use the enhanced training orchestration
system with various configurations and modes.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from master_rl_orchestrator import MasterRLOrchestrator
from duckietown_utils.master_evaluation_system import MasterEvaluationSystem
from launch_master_orchestrator import create_default_config, save_config

def example_quick_test():
    """Example: Quick test run with minimal configuration."""
    print("🚀 EXAMPLE 1: Quick Test Run")
    print("=" * 50)
    
    # Create minimal config for testing
    config = {
        'population_size': 4,
        'max_outer_loops': 5,
        'steps_per_loop': 100_000,  # Reduced for quick test
        'evaluation_seeds': 10,     # Reduced for quick test
        'target_maps': [
            {'name': 'loop_empty', 'difficulty': 'easy', 'type': 'easy_loop'},
            {'name': 'zigzag_dists', 'difficulty': 'moderate', 'type': 'curvy'}
        ],
        'success_thresholds': {
            'easy_loop': {
                'sr_threshold': 0.90,  # Slightly relaxed for quick test
                'r_threshold': 0.80,
                'd_threshold': 0.15,
                'h_threshold': 10.0,
                'j_threshold': 0.10
            },
            'curvy': {
                'sr_threshold': 0.85,
                'r_threshold': 0.75,
                'd_threshold': 0.20,
                'h_threshold': 12.0,
                'j_threshold': 0.12
            }
        },
        'composite_score': {
            'sr_weight': 0.45,
            'reward_weight': 0.25,
            'length_weight': 0.10,
            'deviation_weight': 0.08,
            'heading_weight': 0.06,
            'jerk_weight': 0.06
        }
    }
    
    # Save config
    config_path = "config/quick_test_config.yml"
    save_config(config, config_path)
    
    try:
        # Create and run orchestrator
        orchestrator = MasterRLOrchestrator(config_path)
        print(f"✅ Quick test orchestrator created")
        print(f"📊 Population: {orchestrator.population_size}")
        print(f"🔄 Max loops: {orchestrator.max_outer_loops}")
        print(f"⚡ Steps per loop: {orchestrator.steps_per_loop:,}")
        
        # Note: In a real scenario, you would call:
        # orchestrator.run_continuous_optimization()
        # For this example, we'll just demonstrate the setup
        
        print("🎯 Quick test setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def example_evaluation_only():
    """Example: Evaluation-only mode."""
    print("\n🔍 EXAMPLE 2: Evaluation Only Mode")
    print("=" * 50)
    
    # Create evaluation config
    config = create_default_config()
    config['evaluation_seeds'] = 20  # Reduced for example
    
    # Create evaluation system
    evaluator = MasterEvaluationSystem(config)
    
    # Mock agent for demonstration
    class MockAgent:
        """Mock agent that performs reasonably well."""
        def __init__(self):
            self.performance_level = 0.8  # 80% performance
        
        def get_action(self, obs, deterministic=True):
            # Simple forward action with slight steering
            import random
            steering = random.uniform(-0.1, 0.1) if not deterministic else 0.0
            return [0.6, steering]  # [throttle, steering]
        
        def predict(self, obs, deterministic=True):
            return self.get_action(obs, deterministic), None
    
    # Create mock agent
    agent = MockAgent()
    
    try:
        print("📊 Starting comprehensive evaluation...")
        
        # Run evaluation (this would take time in real scenario)
        # For demo, we'll simulate the process
        print("  🗺️ Evaluating on loop_empty...")
        print("  🗺️ Evaluating on zigzag_dists...")
        print("  🗺️ Evaluating on 4way...")
        print("  🔥 Running stress tests...")
        
        # In real usage:
        # report = evaluator.evaluate_agent_comprehensive(agent)
        
        # Simulate report
        mock_report = {
            'overall_rating': '🥉 ADVANCED',
            'global_metrics': {
                'global_composite_score': 78.5,
                'global_success_rate': 0.82,
                'global_pass_achieved': False,
                'maps_passed': 2,
                'total_maps': 3
            }
        }
        
        print(f"✅ Evaluation completed!")
        print(f"🏆 Overall Rating: {mock_report['overall_rating']}")
        print(f"📊 Global Score: {mock_report['global_metrics']['global_composite_score']:.1f}")
        print(f"🎯 Maps Passed: {mock_report['global_metrics']['maps_passed']}/{mock_report['global_metrics']['total_maps']}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def example_custom_configuration():
    """Example: Custom configuration for specific research goals."""
    print("\n⚙️ EXAMPLE 3: Custom Configuration")
    print("=" * 50)
    
    # Create custom config focused on safety
    safety_focused_config = create_default_config()
    
    # Modify for safety focus
    safety_focused_config.update({
        'population_size': 6,
        'max_outer_loops': 30,
        
        # Safety-focused composite scoring
        'composite_score': {
            'sr_weight': 0.60,      # Higher weight on success rate
            'reward_weight': 0.15,  # Lower weight on reward
            'length_weight': 0.05,
            'deviation_weight': 0.10,  # Higher weight on precision
            'heading_weight': 0.05,
            'jerk_weight': 0.05
        },
        
        # Stricter safety thresholds
        'success_thresholds': {
            'easy_loop': {
                'sr_threshold': 0.98,  # Very high success rate required
                'r_threshold': 0.80,   # Moderate reward requirement
                'd_threshold': 0.10,   # Tighter deviation tolerance
                'h_threshold': 6.0,    # Tighter heading tolerance
                'j_threshold': 0.06    # Smoother driving required
            }
        },
        
        # Conservative algorithm settings
        'algorithm': {
            'default': 'PPO',
            'ppo': {
                'lr': 1.0e-4,          # Lower learning rate
                'entropy_coef': 0.005, # Lower exploration
                'clip_param': 0.15,    # Tighter clipping
                'gamma': 0.999,        # Higher discount factor
                'value_coef': 1.0      # Higher value function weight
            }
        }
    })
    
    # Save custom config
    custom_config_path = "config/safety_focused_config.yml"
    save_config(safety_focused_config, custom_config_path)
    
    print("✅ Safety-focused configuration created")
    print(f"📁 Saved to: {custom_config_path}")
    print(f"🛡️ Success rate weight: {safety_focused_config['composite_score']['sr_weight']:.1%}")
    print(f"🎯 Target success rate: {safety_focused_config['success_thresholds']['easy_loop']['sr_threshold']:.1%}")
    print(f"📏 Max deviation: {safety_focused_config['success_thresholds']['easy_loop']['d_threshold']:.2f}m")

def example_performance_focused():
    """Example: Performance-focused configuration for speed and efficiency."""
    print("\n⚡ EXAMPLE 4: Performance-Focused Configuration")
    print("=" * 50)
    
    # Create performance-focused config
    performance_config = create_default_config()
    
    # Modify for performance focus
    performance_config.update({
        'population_size': 12,  # Larger population for more exploration
        'max_outer_loops': 100, # More loops for optimization
        
        # Performance-focused composite scoring
        'composite_score': {
            'sr_weight': 0.35,      # Moderate success rate weight
            'reward_weight': 0.35,  # High reward weight
            'length_weight': 0.15,  # Higher weight on episode length (speed)
            'deviation_weight': 0.05,
            'heading_weight': 0.05,
            'jerk_weight': 0.05
        },
        
        # Aggressive algorithm settings
        'algorithm': {
            'default': 'PPO',
            'ppo': {
                'lr': 5.0e-4,          # Higher learning rate
                'entropy_coef': 0.02,  # Higher exploration
                'clip_param': 0.25,    # Looser clipping
                'batch_size': 131072,  # Larger batch size
                'epochs': 6            # More epochs per update
            }
        },
        
        # Aggressive PBT settings
        'population_based_training': {
            'enabled': True,
            'exploit_threshold': 0.30,  # Kill bottom 30%
            'explore_threshold': 0.20,  # Clone top 20%
            'hyperparameter_ranges': {
                'lr': [1.0e-4, 2.0e-3],     # Wider learning rate range
                'entropy_coef': [0.005, 0.05], # Wider exploration range
                'clip_param': [0.1, 0.4]     # Wider clipping range
            }
        }
    })
    
    # Save performance config
    perf_config_path = "config/performance_focused_config.yml"
    save_config(performance_config, perf_config_path)
    
    print("✅ Performance-focused configuration created")
    print(f"📁 Saved to: {perf_config_path}")
    print(f"🧬 Population size: {performance_config['population_size']}")
    print(f"🔄 Max loops: {performance_config['max_outer_loops']}")
    print(f"🏆 Reward weight: {performance_config['composite_score']['reward_weight']:.1%}")
    print(f"⚡ Speed weight: {performance_config['composite_score']['length_weight']:.1%}")

def example_curriculum_design():
    """Example: Custom curriculum learning design."""
    print("\n📚 EXAMPLE 5: Custom Curriculum Design")
    print("=" * 50)
    
    # Create curriculum-focused config
    curriculum_config = create_default_config()
    
    # Design progressive curriculum
    custom_curriculum = {
        'enabled': True,
        'auto_advance': True,
        'stages': [
            {
                'name': 'Basic_Lane_Following',
                'episodes': 500,
                'difficulty': 0.2,
                'success_criteria': {'sr_threshold': 0.95},
                'description': 'Perfect basic lane following on simple tracks',
                'env_config': {
                    'maps': ['loop_empty'],
                    'domain_randomization': False,
                    'weather': False,
                    'obstacles': False,
                    'max_speed': 1.0
                }
            },
            {
                'name': 'Curve_Mastery',
                'episodes': 800,
                'difficulty': 0.4,
                'success_criteria': {'sr_threshold': 0.90},
                'description': 'Master curved tracks with precision',
                'env_config': {
                    'maps': ['small_loop', 'zigzag_dists'],
                    'domain_randomization': True,
                    'weather': False,
                    'obstacles': False,
                    'max_speed': 1.5
                }
            },
            {
                'name': 'Weather_Adaptation',
                'episodes': 1000,
                'difficulty': 0.6,
                'success_criteria': {'sr_threshold': 0.85},
                'description': 'Adapt to weather and lighting changes',
                'env_config': {
                    'maps': ['small_loop', 'zigzag_dists'],
                    'domain_randomization': True,
                    'weather': True,
                    'obstacles': False,
                    'max_speed': 1.8
                }
            },
            {
                'name': 'Intersection_Navigation',
                'episodes': 1200,
                'difficulty': 0.8,
                'success_criteria': {
                    'sr_threshold': 0.80,
                    'violations_rate': 0.02
                },
                'description': 'Navigate complex intersections safely',
                'env_config': {
                    'maps': ['4way'],
                    'domain_randomization': True,
                    'weather': True,
                    'obstacles': True,
                    'traffic': True,
                    'max_speed': 2.0
                }
            },
            {
                'name': 'Urban_Mastery',
                'episodes': 1500,
                'difficulty': 1.0,
                'success_criteria': {
                    'sr_threshold': 0.75,
                    'violations_rate': 0.01
                },
                'description': 'Master complex urban environments',
                'env_config': {
                    'maps': ['udem1'],
                    'domain_randomization': True,
                    'weather': True,
                    'obstacles': True,
                    'traffic': True,
                    'pedestrians': True,
                    'max_speed': 2.0
                }
            }
        ]
    }
    
    curriculum_config['curriculum'] = custom_curriculum
    
    # Save curriculum config
    curriculum_config_path = "config/custom_curriculum_config.yml"
    save_config(curriculum_config, curriculum_config_path)
    
    print("✅ Custom curriculum configuration created")
    print(f"📁 Saved to: {curriculum_config_path}")
    print(f"📚 Curriculum stages: {len(custom_curriculum['stages'])}")
    
    for i, stage in enumerate(custom_curriculum['stages']):
        print(f"  {i+1}. {stage['name']} ({stage['episodes']} episodes, difficulty {stage['difficulty']})")

def main():
    """Run all examples."""
    print("🏆 MASTER ORCHESTRATOR EXAMPLES")
    print("=" * 80)
    print("This script demonstrates various usage patterns of the Master RL Orchestrator")
    print("=" * 80)
    
    # Ensure config directory exists
    Path("config").mkdir(exist_ok=True)
    
    # Run examples
    example_quick_test()
    example_evaluation_only()
    example_custom_configuration()
    example_performance_focused()
    example_curriculum_design()
    
    print("\n" + "=" * 80)
    print("🎯 ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("📁 Configuration files created in config/ directory:")
    print("  - quick_test_config.yml")
    print("  - safety_focused_config.yml") 
    print("  - performance_focused_config.yml")
    print("  - custom_curriculum_config.yml")
    print()
    print("🚀 To run the orchestrator with any config:")
    print("  python launch_master_orchestrator.py --config config/your_config.yml")
    print()
    print("📊 To run evaluation only:")
    print("  python launch_master_orchestrator.py --evaluate-only --model-path models/your_model.json")
    print()
    print("🏆 Ready to achieve legendary performance!")

if __name__ == "__main__":
    main()