#!/usr/bin/env python3
"""
📊 METRICS CALCULATOR EXAMPLE 📊
Comprehensive example demonstrating the MetricsCalculator integration

This example shows how to use the MetricsCalculator with the evaluation
orchestrator and suite manager for comprehensive model evaluation.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.metrics_calculator import (
    MetricsCalculator, CompositeScoreConfig, NormalizationScope
)
from duckietown_utils.suite_manager import (
    SuiteManager, EpisodeResult, SuiteResults, SuiteType
)
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator


def create_sample_episode_results(num_episodes: int = 50, 
                                 success_rate: float = 0.8,
                                 map_names: List[str] = None) -> List[EpisodeResult]:
    """Create sample episode results for demonstration."""
    if map_names is None:
        map_names = ['LF-norm-loop', 'LF-norm-zigzag', 'huge_loop']
    
    episodes = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_episodes):
        # Determine success based on success rate
        success = np.random.random() < success_rate
        
        # Generate realistic metrics based on success
        if success:
            reward = np.random.normal(0.8, 0.1)
            episode_length = int(np.random.normal(500, 50))
            lateral_deviation = np.random.exponential(0.08)
            heading_error = np.random.exponential(5.0)
            jerk = np.random.exponential(0.04)
            stability = np.random.beta(8, 2)  # Skewed towards high values
            lap_time = np.random.normal(35.0, 5.0)
        else:
            reward = np.random.normal(0.3, 0.15)
            episode_length = int(np.random.normal(300, 100))
            lateral_deviation = np.random.exponential(0.25)
            heading_error = np.random.exponential(15.0)
            jerk = np.random.exponential(0.12)
            stability = np.random.beta(2, 5)  # Skewed towards low values
            lap_time = None
        
        # Clamp values to reasonable ranges
        reward = np.clip(reward, 0.0, 1.0)
        episode_length = max(episode_length, 50)
        lateral_deviation = max(lateral_deviation, 0.01)
        heading_error = max(heading_error, 0.5)
        jerk = max(jerk, 0.005)
        stability = np.clip(stability, 0.0, 1.0)
        
        # Safety events
        collision = not success and np.random.random() < 0.6
        off_lane = not success and not collision and np.random.random() < 0.8
        
        # Traffic violations
        violations = {}
        if np.random.random() < 0.2:  # 20% chance of violations
            violations = {
                'speed': np.random.randint(0, 3),
                'stop_sign': np.random.randint(0, 2),
                'right_of_way': np.random.randint(0, 1)
            }
        
        episode = EpisodeResult(
            episode_id=f"episode_{i:04d}",
            map_name=map_names[i % len(map_names)],
            seed=i,
            success=success,
            reward=reward,
            episode_length=episode_length,
            lateral_deviation=lateral_deviation,
            heading_error=heading_error,
            jerk=jerk,
            stability=stability,
            collision=collision,
            off_lane=off_lane,
            violations=violations,
            lap_time=lap_time if success else None
        )
        episodes.append(episode)
    
    return episodes


def demonstrate_basic_metrics_calculation():
    """Demonstrate basic metrics calculation functionality."""
    print("🔍 BASIC METRICS CALCULATION DEMO")
    print("=" * 50)
    
    # Initialize metrics calculator
    calculator = MetricsCalculator({
        'confidence_level': 0.95,
        'bootstrap_samples': 5000
    })
    
    # Create sample episodes
    episodes = create_sample_episode_results(30, success_rate=0.75)
    
    print(f"📊 Analyzing {len(episodes)} episodes...")
    
    # Calculate aggregated metrics
    metrics = calculator.aggregate_episode_metrics(episodes)
    
    print("\n📈 PRIMARY METRICS:")
    primary_metrics = ['success_rate', 'mean_reward', 'episode_length', 
                      'lateral_deviation', 'heading_error', 'smoothness']
    
    for metric_name in primary_metrics:
        if metric_name in metrics:
            metric = metrics[metric_name]
            ci = metric.confidence_interval
            print(f"  {metric_name:20}: {metric.value:.4f} "
                  f"[{ci.lower:.4f}, {ci.upper:.4f}] (n={metric.sample_size})")
    
    print("\n🛡️ SAFETY METRICS:")
    safety_metrics = ['collision_rate', 'off_lane_rate', 'violation_rate']
    
    for metric_name in safety_metrics:
        if metric_name in metrics:
            metric = metrics[metric_name]
            ci = metric.confidence_interval
            print(f"  {metric_name:20}: {metric.value:.4f} "
                  f"[{ci.lower:.4f}, {ci.upper:.4f}] (n={metric.sample_size})")
    
    return metrics


def demonstrate_per_map_analysis():
    """Demonstrate per-map metrics analysis."""
    print("\n\n🗺️ PER-MAP ANALYSIS DEMO")
    print("=" * 50)
    
    calculator = MetricsCalculator()
    
    # Create episodes with different performance per map
    map_performance = {
        'LF-norm-loop': 0.85,      # Easy map
        'LF-norm-zigzag': 0.70,    # Medium map
        'huge_loop': 0.55          # Hard map
    }
    
    all_episodes = []
    for map_name, success_rate in map_performance.items():
        episodes = create_sample_episode_results(20, success_rate, [map_name])
        all_episodes.extend(episodes)
    
    # Calculate per-map metrics
    per_map_metrics = calculator.calculate_per_map_metrics(all_episodes)
    
    print("📍 Performance by Map:")
    for map_name, map_metrics in per_map_metrics.items():
        success_rate = map_metrics['success_rate'].value
        mean_reward = map_metrics['mean_reward'].value
        lateral_dev = map_metrics['lateral_deviation'].value
        
        print(f"\n  {map_name}:")
        print(f"    Success Rate: {success_rate:.3f}")
        print(f"    Mean Reward:  {mean_reward:.3f}")
        print(f"    Lateral Dev:  {lateral_dev:.4f}")
    
    return per_map_metrics


def demonstrate_normalization_and_composite_scoring():
    """Demonstrate normalization and composite scoring."""
    print("\n\n🎯 NORMALIZATION & COMPOSITE SCORING DEMO")
    print("=" * 50)
    
    # Configure composite scoring
    composite_config = {
        'composite_weights': {
            'success_rate': 0.40,
            'mean_reward': 0.25,
            'episode_length': 0.10,
            'lateral_deviation': 0.10,
            'heading_error': 0.08,
            'smoothness': 0.07
        },
        'include_safety_penalty': True,
        'safety_penalty_weight': 0.15
    }
    
    calculator = MetricsCalculator(composite_config)
    
    # Create multiple models with different performance levels
    models_data = [
        {'name': 'Champion Model', 'success_rate': 0.90, 'episodes': 40},
        {'name': 'Good Model', 'success_rate': 0.75, 'episodes': 40},
        {'name': 'Average Model', 'success_rate': 0.60, 'episodes': 40},
        {'name': 'Poor Model', 'success_rate': 0.40, 'episodes': 40}
    ]
    
    all_model_metrics = []
    model_results = {}
    
    # Calculate metrics for each model
    for model_data in models_data:
        episodes = create_sample_episode_results(
            model_data['episodes'], 
            model_data['success_rate']
        )
        
        metrics = calculator.aggregate_episode_metrics(episodes)
        all_model_metrics.append(metrics)
        model_results[model_data['name']] = metrics
    
    # Compute normalization statistics across all models
    norm_stats = calculator.compute_normalization_stats(all_model_metrics)
    
    print("📏 Normalization Statistics:")
    for metric_name, stats in norm_stats.items():
        if metric_name in ['success_rate', 'mean_reward', 'lateral_deviation']:
            print(f"  {metric_name:20}: min={stats['min']:.4f}, "
                  f"max={stats['max']:.4f}, mean={stats['mean']:.4f}")
    
    # Normalize metrics and calculate composite scores
    print("\n🏆 MODEL COMPARISON WITH COMPOSITE SCORES:")
    
    model_scores = []
    for model_name, metrics in model_results.items():
        # Normalize metrics
        normalized_metrics = calculator.normalize_metrics(metrics, norm_stats)
        
        # Calculate composite score
        composite_score = calculator.calculate_composite_score(normalized_metrics)
        
        model_scores.append({
            'name': model_name,
            'composite_score': composite_score.value,
            'success_rate': metrics['success_rate'].value,
            'mean_reward': metrics['mean_reward'].value
        })
    
    # Sort by composite score
    model_scores.sort(key=lambda x: x['composite_score'], reverse=True)
    
    for i, model in enumerate(model_scores, 1):
        print(f"  {i}. {model['name']:15} - "
              f"Composite: {model['composite_score']:.4f}, "
              f"Success: {model['success_rate']:.3f}, "
              f"Reward: {model['mean_reward']:.3f}")
    
    return model_scores


def demonstrate_suite_integration():
    """Demonstrate integration with suite manager."""
    print("\n\n🧪 SUITE INTEGRATION DEMO")
    print("=" * 50)
    
    calculator = MetricsCalculator()
    
    # Create suite results for different suite types
    suite_configs = [
        {'name': 'base', 'type': SuiteType.BASE, 'success_rate': 0.85},
        {'name': 'hard_randomization', 'type': SuiteType.HARD_RANDOMIZATION, 'success_rate': 0.65},
        {'name': 'out_of_distribution', 'type': SuiteType.OUT_OF_DISTRIBUTION, 'success_rate': 0.45}
    ]
    
    suite_results_list = []
    
    for suite_config in suite_configs:
        episodes = create_sample_episode_results(30, suite_config['success_rate'])
        
        suite_results = SuiteResults(
            suite_name=suite_config['name'],
            suite_type=suite_config['type'],
            model_id="demo_model",
            policy_mode="deterministic",
            total_episodes=len(episodes),
            successful_episodes=sum(1 for ep in episodes if ep.success),
            episode_results=episodes
        )
        suite_results_list.append(suite_results)
    
    # Calculate comprehensive model metrics
    model_metrics = calculator.calculate_model_metrics("demo_model", suite_results_list)
    
    print("📊 COMPREHENSIVE MODEL METRICS:")
    print(f"  Model ID: {model_metrics.model_id}")
    print(f"  Total Episodes: {model_metrics.metadata['total_episodes']}")
    print(f"  Total Suites: {model_metrics.metadata['total_suites']}")
    
    print("\n🎯 PRIMARY METRICS:")
    for metric_name, metric_result in model_metrics.primary_metrics.items():
        print(f"  {metric_name:20}: {metric_result.value:.4f}")
    
    print("\n🛡️ SAFETY METRICS:")
    for metric_name, metric_result in model_metrics.safety_metrics.items():
        print(f"  {metric_name:20}: {metric_result.value:.4f}")
    
    print("\n📍 PER-SUITE PERFORMANCE:")
    for suite_name, suite_metrics in model_metrics.per_suite_metrics.items():
        success_rate = suite_metrics['success_rate'].value
        mean_reward = suite_metrics['mean_reward'].value
        print(f"  {suite_name:20}: Success={success_rate:.3f}, Reward={mean_reward:.3f}")
    
    return model_metrics


def demonstrate_confidence_intervals():
    """Demonstrate confidence interval calculations."""
    print("\n\n📊 CONFIDENCE INTERVALS DEMO")
    print("=" * 50)
    
    calculator = MetricsCalculator({'confidence_level': 0.95})
    
    # Test different sample sizes
    sample_sizes = [10, 30, 100]
    
    for n in sample_sizes:
        episodes = create_sample_episode_results(n, success_rate=0.75)
        metrics = calculator.aggregate_episode_metrics(episodes)
        
        success_rate = metrics['success_rate']
        mean_reward = metrics['mean_reward']
        
        print(f"\n📈 Sample Size: {n}")
        print(f"  Success Rate: {success_rate.value:.4f} "
              f"[{success_rate.confidence_interval.lower:.4f}, "
              f"{success_rate.confidence_interval.upper:.4f}] "
              f"({success_rate.confidence_interval.method})")
        
        print(f"  Mean Reward:  {mean_reward.value:.4f} "
              f"[{mean_reward.confidence_interval.lower:.4f}, "
              f"{mean_reward.confidence_interval.upper:.4f}] "
              f"({mean_reward.confidence_interval.method})")


def save_example_results():
    """Save example results to JSON for inspection."""
    print("\n\n💾 SAVING EXAMPLE RESULTS")
    print("=" * 50)
    
    calculator = MetricsCalculator()
    episodes = create_sample_episode_results(50, success_rate=0.80)
    metrics = calculator.aggregate_episode_metrics(episodes)
    
    # Convert to serializable format
    results = {}
    for metric_name, metric_result in metrics.items():
        results[metric_name] = {
            'value': metric_result.value,
            'sample_size': metric_result.sample_size,
            'confidence_interval': {
                'lower': metric_result.confidence_interval.lower,
                'upper': metric_result.confidence_interval.upper,
                'method': metric_result.confidence_interval.method
            } if metric_result.confidence_interval else None,
            'metadata': metric_result.metadata
        }
    
    # Save to file
    output_file = Path('logs') / 'metrics_calculator_example_results.json'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Metrics calculated: {len(results)}")


def main():
    """Run all demonstration examples."""
    print("🚀 METRICS CALCULATOR COMPREHENSIVE DEMO")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_basic_metrics_calculation()
        demonstrate_per_map_analysis()
        demonstrate_normalization_and_composite_scoring()
        demonstrate_suite_integration()
        demonstrate_confidence_intervals()
        save_example_results()
        
        print("\n\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()