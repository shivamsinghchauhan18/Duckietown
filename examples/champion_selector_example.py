#!/usr/bin/env python3
"""
🏆 CHAMPION SELECTOR EXAMPLE 🏆
Example usage of the ChampionSelector for automated model ranking and selection

This example demonstrates:
- Multi-criteria ranking algorithm
- Pareto front analysis for trade-off visualization
- Regression detection and champion validation
- Statistical significance validation for champion updates
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.champion_selector import ChampionSelector, ValidationStatus
from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer

def create_sample_model_metrics(model_id: str, base_performance: float, 
                               variation: float = 0.0) -> ModelMetrics:
    """Create sample model metrics for demonstration.
    
    Args:
        model_id: Unique identifier for the model
        base_performance: Base performance level (0.0 to 1.0)
        variation: Random variation to add to metrics
        
    Returns:
        ModelMetrics: Sample model metrics
    """
    # Add some randomness for realistic variation
    np.random.seed(hash(model_id) % 2**32)
    noise = np.random.normal(0, variation, 10)
    
    # Primary metrics
    success_rate = np.clip(base_performance + noise[0], 0.0, 1.0)
    mean_reward = np.clip(base_performance * 0.8 + noise[1] * 0.1, 0.0, 1.0)
    episode_length = max(300, 600 - base_performance * 200 + noise[2] * 50)
    lateral_deviation = max(0.01, 0.2 - base_performance * 0.15 + abs(noise[3]) * 0.05)
    heading_error = max(1.0, 15.0 - base_performance * 10 + abs(noise[4]) * 3)
    smoothness = max(0.01, 0.15 - base_performance * 0.1 + abs(noise[5]) * 0.03)
    
    primary_metrics = {
        'success_rate': MetricResult(name='success_rate', value=success_rate),
        'mean_reward': MetricResult(name='mean_reward', value=mean_reward),
        'episode_length': MetricResult(name='episode_length', value=episode_length),
        'lateral_deviation': MetricResult(name='lateral_deviation', value=lateral_deviation),
        'heading_error': MetricResult(name='heading_error', value=heading_error),
        'smoothness': MetricResult(name='smoothness', value=smoothness)
    }
    
    # Secondary metrics
    stability = np.clip(base_performance + noise[6] * 0.1, 0.0, 1.0)
    lap_time = max(20.0, 60.0 - base_performance * 30 + noise[7] * 10)
    
    secondary_metrics = {
        'stability': MetricResult(name='stability', value=stability),
        'lap_time': MetricResult(name='lap_time', value=lap_time),
        'completion_rate': MetricResult(name='completion_rate', value=success_rate)
    }
    
    # Safety metrics
    collision_rate = max(0.0, 0.2 - base_performance * 0.15 + abs(noise[8]) * 0.05)
    off_lane_rate = max(0.0, 0.15 - base_performance * 0.1 + abs(noise[9]) * 0.03)
    
    safety_metrics = {
        'collision_rate': MetricResult(name='collision_rate', value=collision_rate),
        'off_lane_rate': MetricResult(name='off_lane_rate', value=off_lane_rate),
        'violation_rate': MetricResult(name='violation_rate', value=collision_rate * 0.5)
    }
    
    # Composite score (weighted combination)
    composite_score = (
        0.45 * success_rate +
        0.25 * mean_reward +
        0.10 * (1.0 - min(episode_length / 1000.0, 1.0)) +
        0.08 * (1.0 - min(lateral_deviation / 0.5, 1.0)) +
        0.06 * (1.0 - min(heading_error / 30.0, 1.0)) +
        0.06 * (1.0 - min(smoothness / 0.3, 1.0))
    )
    
    composite_score_result = MetricResult(name='composite_score', value=composite_score)
    
    # Per-map metrics (simulate 5 different maps)
    per_map_metrics = {}
    map_names = ['loop_empty', 'zigzag', 'techtrack', 'multi_track', 'huge_loop']
    
    for i, map_name in enumerate(map_names):
        # Add map-specific variation
        map_success_rate = np.clip(success_rate + np.random.normal(0, 0.05), 0.0, 1.0)
        per_map_metrics[map_name] = {
            'success_rate': MetricResult(name='success_rate', value=map_success_rate),
            'lateral_deviation': MetricResult(name='lateral_deviation', 
                                            value=lateral_deviation + np.random.normal(0, 0.02))
        }
    
    # Per-suite metrics
    per_suite_metrics = {
        'base': {
            'success_rate': MetricResult(name='success_rate', value=success_rate),
            'mean_reward': MetricResult(name='mean_reward', value=mean_reward)
        },
        'hard': {
            'success_rate': MetricResult(name='success_rate', value=success_rate * 0.85),
            'mean_reward': MetricResult(name='mean_reward', value=mean_reward * 0.9)
        },
        'ood': {
            'success_rate': MetricResult(name='success_rate', value=success_rate * 0.75),
            'mean_reward': MetricResult(name='mean_reward', value=mean_reward * 0.8)
        },
        'stress': {
            'success_rate': MetricResult(name='success_rate', value=success_rate * 0.6),
            'mean_reward': MetricResult(name='mean_reward', value=mean_reward * 0.7)
        }
    }
    
    return ModelMetrics(
        model_id=model_id,
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        safety_metrics=safety_metrics,
        composite_score=composite_score_result,
        per_map_metrics=per_map_metrics,
        per_suite_metrics=per_suite_metrics,
        metadata={
            'training_episodes': 1000000,
            'evaluation_episodes': 500,
            'model_architecture': 'PPO',
            'creation_timestamp': '2025-01-15T10:30:00Z'
        }
    )

def demonstrate_champion_selection():
    """Demonstrate the champion selection process."""
    print("🏆 Champion Selector Example")
    print("=" * 50)
    
    # Create ChampionSelector with custom configuration
    config = {
        'min_maps_threshold': 0.8,  # 80% of maps must meet threshold
        'min_success_rate_threshold': 0.75,  # 75% minimum success rate
        'success_rate_regression_threshold': 0.05,  # 5% regression threshold
        'smoothness_regression_threshold': 0.20,  # 20% smoothness regression
        'require_statistical_significance': True,
        'significance_alpha': 0.05,
        'pareto_axes': [
            ['success_rate', 'lateral_deviation', 'smoothness'],
            ['success_rate', 'episode_length'],
            ['composite_score', 'stability']
        ]
    }
    
    selector = ChampionSelector(config)
    print(f"✅ Initialized ChampionSelector with {len(selector.ranking_criteria)} ranking criteria")
    
    # Create sample model candidates with different performance levels
    print("\n📊 Creating sample model candidates...")
    
    model_candidates = [
        create_sample_model_metrics("elite_model_v3", 0.92, 0.02),      # Excellent
        create_sample_model_metrics("advanced_model_v2", 0.88, 0.03),   # Very good
        create_sample_model_metrics("standard_model_v4", 0.85, 0.04),   # Good
        create_sample_model_metrics("baseline_model_v1", 0.80, 0.05),   # Decent
        create_sample_model_metrics("experimental_v1", 0.75, 0.08),     # Experimental
        create_sample_model_metrics("legacy_model", 0.70, 0.06),        # Legacy
    ]
    
    print(f"✅ Created {len(model_candidates)} model candidates")
    
    # Display candidate summary
    print("\n📋 Model Candidate Summary:")
    print("-" * 80)
    print(f"{'Model ID':<20} {'Composite':<10} {'Success Rate':<12} {'Smoothness':<10} {'Stability':<10}")
    print("-" * 80)
    
    for model in model_candidates:
        composite = model.composite_score.value if model.composite_score else 0.0
        success_rate = model.primary_metrics['success_rate'].value
        smoothness = model.primary_metrics['smoothness'].value
        stability = model.secondary_metrics['stability'].value
        
        print(f"{model.model_id:<20} {composite:<10.3f} {success_rate:<12.3f} "
              f"{smoothness:<10.3f} {stability:<10.3f}")
    
    # Perform champion selection (no current champion)
    print("\n🏆 Performing initial champion selection...")
    selection_result = selector.select_champion(model_candidates)
    
    print(f"✅ Selected champion: {selection_result.new_champion_id}")
    
    # Display detailed results
    print("\n📊 Detailed Selection Results:")
    print("-" * 100)
    print(f"{'Rank':<5} {'Model ID':<20} {'Composite':<10} {'Validation':<15} {'Pareto Rank':<12}")
    print("-" * 100)
    
    for ranking in selection_result.rankings:
        validation_status = ranking.validation.status.value if ranking.validation else "unknown"
        pareto_rank = ranking.pareto_rank if ranking.pareto_rank else "N/A"
        
        print(f"{ranking.rank:<5} {ranking.model_id:<20} {ranking.global_composite_score:<10.3f} "
              f"{validation_status:<15} {pareto_rank:<12}")
    
    # Display Pareto front analysis
    print(f"\n🎯 Pareto Front Analysis ({len(selection_result.pareto_fronts)} fronts analyzed):")
    for i, front in enumerate(selection_result.pareto_fronts):
        print(f"\nFront {i+1}: {' vs '.join(front.axes)}")
        print(f"  Non-dominated models: {', '.join(front.non_dominated_models)}")
        print(f"  Dominated models: {', '.join(front.dominated_models)}")
        
        if 'extreme_points' in front.trade_off_analysis:
            print("  Extreme points:")
            for key, point in front.trade_off_analysis['extreme_points'].items():
                print(f"    {key}: {point['model_id']} ({point['value']:.3f})")
    
    # Demonstrate champion update scenario
    print("\n🔄 Demonstrating champion update scenario...")
    current_champion = selection_result.new_champion_id
    
    # Create new candidate models (some better, some worse)
    new_candidates = model_candidates + [
        create_sample_model_metrics("super_elite_v1", 0.95, 0.01),      # Better than current
        create_sample_model_metrics("regression_model", 0.82, 0.06),    # Potential regression
    ]
    
    print(f"📈 Added 2 new candidates, total: {len(new_candidates)}")
    
    # Perform champion selection with current champion
    update_result = selector.select_champion(new_candidates, current_champion)
    
    print(f"🏆 Champion update result:")
    print(f"  Previous champion: {update_result.previous_champion_id}")
    print(f"  New champion: {update_result.new_champion_id}")
    print(f"  Champion changed: {update_result.new_champion_id != update_result.previous_champion_id}")
    
    # Display regression analysis
    print("\n🔍 Regression Analysis:")
    for ranking in update_result.rankings:
        if ranking.regression_analysis and ranking.regression_analysis.is_regression:
            analysis = ranking.regression_analysis
            print(f"  {ranking.model_id}: REGRESSION DETECTED")
            for reason in analysis.regression_reasons:
                print(f"    - {reason}")
        elif ranking.regression_analysis:
            analysis = ranking.regression_analysis
            print(f"  {ranking.model_id}: No regression detected")
            if analysis.success_rate_change:
                print(f"    Success rate change: {analysis.success_rate_change:+.3f}")
    
    # Generate champion summary
    summary = selector.get_champion_summary(update_result)
    
    print("\n📋 Champion Selection Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save results to file
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "champion_selector_example_results.json"
    
    # Convert results to JSON-serializable format
    results_data = {
        'champion_selection': {
            'new_champion_id': update_result.new_champion_id,
            'previous_champion_id': update_result.previous_champion_id,
            'total_candidates': len(new_candidates),
            'selection_metadata': update_result.selection_metadata
        },
        'rankings': [
            {
                'model_id': r.model_id,
                'rank': r.rank,
                'global_composite_score': r.global_composite_score,
                'validation_status': r.validation.status.value if r.validation else None,
                'pareto_rank': r.pareto_rank,
                'tie_breaker_used': r.tie_breaker_used.value if r.tie_breaker_used else None
            }
            for r in update_result.rankings
        ],
        'pareto_fronts': [
            {
                'axes': front.axes,
                'non_dominated_models': front.non_dominated_models,
                'dominated_models': front.dominated_models,
                'trade_off_analysis': front.trade_off_analysis
            }
            for front in update_result.pareto_fronts
        ],
        'summary': summary
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Champion Selector example completed successfully!")

def demonstrate_validation_scenarios():
    """Demonstrate different validation scenarios."""
    print("\n🔍 Validation Scenarios Demonstration")
    print("=" * 50)
    
    selector = ChampionSelector()
    
    # Scenario 1: Valid champion
    print("\n✅ Scenario 1: Valid Champion")
    valid_model = create_sample_model_metrics("valid_champion", 0.90, 0.01)
    validation = selector._validate_champion_candidate(valid_model)
    print(f"Status: {validation.status.value}")
    print(f"Maps meeting threshold: {validation.maps_meeting_threshold}/{validation.total_maps}")
    
    # Scenario 2: Low success rate
    print("\n❌ Scenario 2: Low Success Rate")
    low_sr_model = create_sample_model_metrics("low_success_model", 0.60, 0.02)
    validation = selector._validate_champion_candidate(low_sr_model)
    print(f"Status: {validation.status.value}")
    print(f"Maps below threshold: {validation.maps_below_success_threshold}")
    
    # Scenario 3: Insufficient maps meeting threshold
    print("\n⚠️  Scenario 3: Insufficient Maps")
    # Create model with mixed performance across maps
    insufficient_model = ModelMetrics(
        model_id="insufficient_maps",
        primary_metrics={'success_rate': MetricResult(name='success_rate', value=0.75)},
        secondary_metrics={},
        safety_metrics={},
        composite_score=MetricResult(name='composite_score', value=0.75),
        per_map_metrics={
            'map_1': {'success_rate': MetricResult(name='success_rate', value=0.80)},  # Good
            'map_2': {'success_rate': MetricResult(name='success_rate', value=0.70)},  # Poor
            'map_3': {'success_rate': MetricResult(name='success_rate', value=0.65)},  # Poor
            'map_4': {'success_rate': MetricResult(name='success_rate', value=0.72)},  # Poor
            'map_5': {'success_rate': MetricResult(name='success_rate', value=0.78)},  # Good
        },
        per_suite_metrics={}
    )
    
    validation = selector._validate_champion_candidate(insufficient_model)
    print(f"Status: {validation.status.value}")
    print(f"Maps meeting threshold: {validation.maps_meeting_threshold}/{validation.total_maps}")
    print(f"Threshold percentage: {validation.validation_details['maps_threshold_percentage']:.1%}")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demonstrate_champion_selection()
        demonstrate_validation_scenarios()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)