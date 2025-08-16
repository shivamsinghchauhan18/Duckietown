#!/usr/bin/env python3
"""
🔬 ROBUSTNESS ANALYZER EXAMPLE 🔬
Comprehensive example demonstrating the RobustnessAnalyzer functionality

This example shows how to:
1. Configure parameter sweeps for environmental robustness testing
2. Analyze robustness curves and calculate AUC metrics
3. Detect sensitivity thresholds and operating ranges
4. Compare robustness across multiple models
5. Generate robustness reports and visualizations
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.robustness_analyzer import (
    RobustnessAnalyzer, ParameterSweepConfig, ParameterType,
    ParameterPoint, RobustnessCurve, RobustnessAnalysisResult
)
from duckietown_utils.suite_manager import EpisodeResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_episodes(param_value: float, model_performance: float = 0.8, 
                          num_episodes: int = 50) -> List[EpisodeResult]:
    """Create sample episode results for a given parameter value.
    
    Args:
        param_value: Environmental parameter value
        model_performance: Base model performance (0-1)
        num_episodes: Number of episodes to generate
        
    Returns:
        List[EpisodeResult]: Generated episode results
    """
    episodes = []
    
    # Performance degrades as parameter deviates from optimal (1.0)
    performance_factor = 1.0 - 0.3 * abs(param_value - 1.0)
    actual_performance = model_performance * max(0.1, performance_factor)
    
    for i in range(num_episodes):
        # Add some randomness
        success_prob = actual_performance + np.random.normal(0, 0.1)
        success_prob = np.clip(success_prob, 0.0, 1.0)
        
        success = np.random.random() < success_prob
        collision = not success and np.random.random() < 0.6
        off_lane = not success and not collision
        
        # Generate realistic metrics
        reward_base = 0.8 * actual_performance
        reward = reward_base + np.random.normal(0, 0.1)
        reward = np.clip(reward, 0.0, 1.0)
        
        episode = EpisodeResult(
            episode_id=f"param_{param_value:.2f}_ep_{i:03d}",
            map_name="loop_empty",
            seed=i,
            success=success,
            reward=reward,
            episode_length=1000 + np.random.randint(-100, 100),
            lateral_deviation=0.05 + 0.1 * (1 - actual_performance) + np.random.normal(0, 0.02),
            heading_error=1.0 + 2 * (1 - actual_performance) + np.random.normal(0, 0.5),
            jerk=0.3 + 0.2 * (1 - actual_performance) + np.random.normal(0, 0.05),
            stability=2.0 * actual_performance + np.random.normal(0, 0.2),
            collision=collision,
            off_lane=off_lane,
            violations={},
            lap_time=30.0 + np.random.normal(0, 5),
            metadata={
                "model_id": "sample_model",
                "mode": "deterministic",
                "suite": "robustness_test"
            },
            timestamp="2024-01-01T00:00:00"
        )
        episodes.append(episode)
    
    return episodes

def demonstrate_parameter_sweep_configuration():
    """Demonstrate different parameter sweep configurations."""
    logger.info("🔧 Demonstrating Parameter Sweep Configurations")
    
    # Linear sweep configuration
    linear_config = ParameterSweepConfig(
        parameter_type=ParameterType.LIGHTING_INTENSITY,
        parameter_name="lighting_intensity",
        min_value=0.5,
        max_value=2.0,
        num_points=10,
        sweep_method="linear",
        baseline_value=1.0,
        description="Linear sweep of lighting intensity from 0.5x to 2.0x normal"
    )
    
    # Logarithmic sweep configuration
    log_config = ParameterSweepConfig(
        parameter_type=ParameterType.FRICTION_COEFFICIENT,
        parameter_name="friction_coefficient",
        min_value=0.1,
        max_value=2.0,
        num_points=8,
        sweep_method="log",
        baseline_value=1.0,
        description="Logarithmic sweep of friction coefficient"
    )
    
    # Custom sweep configuration
    custom_config = ParameterSweepConfig(
        parameter_type=ParameterType.CAMERA_PITCH,
        parameter_name="camera_pitch_degrees",
        min_value=-10,
        max_value=10,
        sweep_method="custom",
        custom_values=[-10, -5, -2, 0, 2, 5, 10],
        baseline_value=0,
        description="Custom sweep of camera pitch angles"
    )
    
    # Initialize analyzer
    analyzer = RobustnessAnalyzer({
        'confidence_level': 0.95,
        'sensitivity_threshold': 0.15,
        'min_operating_performance': 0.75
    })
    
    # Generate sweep values
    linear_values = analyzer.generate_parameter_sweep_values(linear_config)
    log_values = analyzer.generate_parameter_sweep_values(log_config)
    custom_values = analyzer.generate_parameter_sweep_values(custom_config)
    
    logger.info(f"📊 Linear sweep values: {linear_values}")
    logger.info(f"📊 Log sweep values: {log_values}")
    logger.info(f"📊 Custom sweep values: {custom_values}")
    
    return {
        'linear': (linear_config, linear_values),
        'log': (log_config, log_values),
        'custom': (custom_config, custom_values)
    }

def demonstrate_single_parameter_analysis():
    """Demonstrate analysis of a single parameter sweep."""
    logger.info("🔬 Demonstrating Single Parameter Analysis")
    
    # Create analyzer
    analyzer = RobustnessAnalyzer({
        'confidence_level': 0.95,
        'sensitivity_threshold': 0.1,  # 10% performance degradation
        'min_operating_performance': 0.7,  # 70% success rate minimum
        'auc_normalization': True,
        'robustness_weights': {
            'success_rate_auc': 0.5,
            'reward_auc': 0.3,
            'stability_auc': 0.2
        }
    })
    
    # Create sweep configuration
    config = ParameterSweepConfig(
        parameter_type=ParameterType.LIGHTING_INTENSITY,
        parameter_name="lighting_intensity",
        min_value=0.3,
        max_value=2.5,
        num_points=12,
        sweep_method="linear",
        baseline_value=1.0,
        description="Lighting intensity robustness test"
    )
    
    # Generate parameter values
    param_values = analyzer.generate_parameter_sweep_values(config)
    
    # Create episode results for each parameter value
    parameter_results = {}
    for param_val in param_values:
        episodes = create_sample_episodes(param_val, model_performance=0.85, num_episodes=30)
        parameter_results[param_val] = episodes
        logger.info(f"📊 Generated {len(episodes)} episodes for parameter value {param_val:.2f}")
    
    # Analyze parameter sweep
    curve = analyzer.analyze_parameter_sweep("demo_model_v1", parameter_results, config)
    
    # Display results
    logger.info(f"✅ Parameter Sweep Analysis Complete")
    logger.info(f"🎯 Model: {curve.model_id}")
    logger.info(f"📊 Parameter: {curve.parameter_name}")
    logger.info(f"📈 AUC Success Rate: {curve.auc_success_rate:.3f}")
    logger.info(f"📈 AUC Reward: {curve.auc_reward:.3f}")
    logger.info(f"📈 AUC Stability: {curve.auc_stability:.3f}")
    logger.info(f"⚠️  Sensitivity Threshold: {curve.sensitivity_threshold}")
    logger.info(f"📏 Operating Range: {curve.operating_range}")
    logger.info(f"🔍 Degradation Points: {len(curve.degradation_points)}")
    
    # Display sweep points
    logger.info("📊 Sweep Points Summary:")
    for point in curve.sweep_points:
        logger.info(f"  Parameter: {point.parameter_value:.2f}, "
                   f"Success Rate: {point.success_rate:.3f}, "
                   f"Reward: {point.mean_reward:.3f}, "
                   f"Stability: {point.stability:.3f}")
    
    return curve

def demonstrate_multi_parameter_analysis():
    """Demonstrate analysis across multiple parameters."""
    logger.info("🔬 Demonstrating Multi-Parameter Analysis")
    
    # Create analyzer
    analyzer = RobustnessAnalyzer({
        'confidence_level': 0.95,
        'sensitivity_threshold': 0.12,
        'min_operating_performance': 0.75
    })
    
    # Define multiple parameter sweeps
    sweep_configs = {
        'lighting_intensity': ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.4,
            max_value=2.2,
            num_points=8,
            sweep_method="linear",
            baseline_value=1.0
        ),
        'friction_coefficient': ParameterSweepConfig(
            parameter_type=ParameterType.FRICTION_COEFFICIENT,
            parameter_name="friction_coefficient",
            min_value=0.3,
            max_value=1.8,
            num_points=8,
            sweep_method="linear",
            baseline_value=1.0
        ),
        'camera_pitch': ParameterSweepConfig(
            parameter_type=ParameterType.CAMERA_PITCH,
            parameter_name="camera_pitch_degrees",
            min_value=-8,
            max_value=8,
            num_points=9,
            sweep_method="linear",
            baseline_value=0
        )
    }
    
    # Generate results for each parameter
    parameter_sweep_results = {}
    
    for param_name, config in sweep_configs.items():
        param_values = analyzer.generate_parameter_sweep_values(config)
        param_results = {}
        
        for param_val in param_values:
            # Different parameters have different impact on performance
            if param_name == 'lighting_intensity':
                base_performance = 0.88
            elif param_name == 'friction_coefficient':
                base_performance = 0.82
            else:  # camera_pitch
                base_performance = 0.85
            
            episodes = create_sample_episodes(param_val, base_performance, num_episodes=25)
            param_results[param_val] = episodes
        
        parameter_sweep_results[param_name] = param_results
        logger.info(f"📊 Generated results for {param_name} with {len(param_values)} parameter values")
    
    # Analyze overall robustness
    analysis_result = analyzer.analyze_model_robustness(
        "multi_param_model",
        parameter_sweep_results,
        sweep_configs
    )
    
    # Display results
    logger.info(f"✅ Multi-Parameter Analysis Complete")
    logger.info(f"🎯 Model: {analysis_result.model_id}")
    logger.info(f"🏆 Overall Robustness Score: {analysis_result.overall_robustness_score:.3f}")
    logger.info(f"📊 Parameters Tested: {len(analysis_result.parameter_curves)}")
    
    # Display per-parameter results
    for param_name, curve in analysis_result.parameter_curves.items():
        logger.info(f"📈 {param_name}:")
        logger.info(f"  AUC Success Rate: {curve.auc_success_rate:.3f}")
        logger.info(f"  Sensitivity Threshold: {curve.sensitivity_threshold}")
        logger.info(f"  Operating Range: {curve.operating_range}")
    
    # Display sensitivity summary
    logger.info("⚠️  Sensitivity Summary:")
    for param_name, threshold in analysis_result.sensitivity_summary.items():
        logger.info(f"  {param_name}: {threshold}")
    
    # Display operating ranges
    logger.info("📏 Operating Ranges:")
    for param_name, op_range in analysis_result.operating_ranges.items():
        if op_range:
            logger.info(f"  {param_name}: [{op_range[0]:.2f}, {op_range[1]:.2f}]")
        else:
            logger.info(f"  {param_name}: No safe operating range found")
    
    # Display recommendations
    logger.info("💡 Recommendations:")
    for i, recommendation in enumerate(analysis_result.recommendations, 1):
        logger.info(f"  {i}. {recommendation}")
    
    return analysis_result

def demonstrate_multi_model_comparison():
    """Demonstrate comparison of robustness across multiple models."""
    logger.info("🔬 Demonstrating Multi-Model Comparison")
    
    # Create analyzer
    analyzer = RobustnessAnalyzer({
        'confidence_level': 0.95,
        'sensitivity_threshold': 0.1,
        'min_operating_performance': 0.7
    })
    
    # Define parameter sweep
    config = ParameterSweepConfig(
        parameter_type=ParameterType.LIGHTING_INTENSITY,
        parameter_name="lighting_intensity",
        min_value=0.5,
        max_value=2.0,
        num_points=8,
        sweep_method="linear",
        baseline_value=1.0
    )
    
    param_values = analyzer.generate_parameter_sweep_values(config)
    
    # Create results for multiple models with different robustness characteristics
    models = {
        'robust_model': 0.9,      # High base performance
        'standard_model': 0.8,    # Medium base performance
        'fragile_model': 0.75     # Lower base performance
    }
    
    model_results = {}
    
    for model_id, base_performance in models.items():
        # Generate parameter sweep results
        parameter_results = {}
        for param_val in param_values:
            episodes = create_sample_episodes(param_val, base_performance, num_episodes=20)
            parameter_results[param_val] = episodes
        
        # Analyze robustness for this model
        sweep_configs = {'lighting_intensity': config}
        parameter_sweep_results = {'lighting_intensity': parameter_results}
        
        analysis_result = analyzer.analyze_model_robustness(
            model_id,
            parameter_sweep_results,
            sweep_configs
        )
        
        model_results[model_id] = analysis_result
        logger.info(f"📊 Analyzed robustness for {model_id}")
    
    # Compare models
    comparison = analyzer.compare_model_robustness(model_results)
    
    # Display comparison results
    logger.info(f"✅ Multi-Model Comparison Complete")
    logger.info(f"🏆 Models Compared: {len(comparison.model_results)}")
    
    # Display overall rankings
    logger.info("🏆 Overall Robustness Rankings:")
    for i, (model_id, score) in enumerate(comparison.robustness_rankings, 1):
        logger.info(f"  {i}. {model_id}: {score:.3f}")
    
    # Display parameter-specific rankings
    logger.info("📊 Parameter-Specific Rankings:")
    for param_name, rankings in comparison.parameter_rankings.items():
        logger.info(f"  {param_name}:")
        for i, (model_id, auc_score) in enumerate(rankings, 1):
            logger.info(f"    {i}. {model_id}: AUC = {auc_score:.3f}")
    
    # Display sensitivity comparison
    logger.info("⚠️  Sensitivity Comparison:")
    for param_name, sensitivities in comparison.sensitivity_comparison.items():
        logger.info(f"  {param_name}:")
        for model_id, sensitivity in sensitivities.items():
            if sensitivity == float('inf'):
                logger.info(f"    {model_id}: No sensitivity threshold detected")
            else:
                logger.info(f"    {model_id}: {sensitivity:.3f}")
    
    # Display best operating ranges
    logger.info("📏 Best Operating Ranges (Union of all models):")
    for param_name, op_range in comparison.best_operating_ranges.items():
        logger.info(f"  {param_name}: [{op_range[0]:.2f}, {op_range[1]:.2f}]")
    
    return comparison

def demonstrate_visualization_and_export():
    """Demonstrate visualization and export functionality."""
    logger.info("📊 Demonstrating Visualization and Export")
    
    # Create analyzer with plotting configuration
    analyzer = RobustnessAnalyzer({
        'plot_config': {
            'figsize': (12, 10),
            'dpi': 150,
            'style': 'seaborn-v0_8',
            'save_plots': True,
            'plot_format': 'png'
        }
    })
    
    # Generate a robustness curve for demonstration
    config = ParameterSweepConfig(
        parameter_type=ParameterType.LIGHTING_INTENSITY,
        parameter_name="lighting_intensity",
        min_value=0.4,
        max_value=2.2,
        num_points=10,
        sweep_method="linear",
        baseline_value=1.0
    )
    
    param_values = analyzer.generate_parameter_sweep_values(config)
    parameter_results = {}
    
    for param_val in param_values:
        episodes = create_sample_episodes(param_val, model_performance=0.85, num_episodes=25)
        parameter_results[param_val] = episodes
    
    # Analyze parameter sweep
    curve = analyzer.analyze_parameter_sweep("visualization_model", parameter_results, config)
    
    # Create output directory
    output_dir = Path("logs/robustness_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot robustness curve
    try:
        plot_path = output_dir / "robustness_curve_example.png"
        fig = analyzer.plot_robustness_curve(curve, str(plot_path))
        logger.info(f"📊 Robustness curve plot saved to {plot_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not create plot: {e}")
    
    # Create analysis result
    sweep_configs = {'lighting_intensity': config}
    parameter_sweep_results = {'lighting_intensity': parameter_results}
    
    analysis_result = analyzer.analyze_model_robustness(
        "visualization_model",
        parameter_sweep_results,
        sweep_configs
    )
    
    # Export results in different formats
    try:
        # Export as JSON
        json_path = output_dir / "robustness_analysis_example.json"
        analyzer.export_robustness_results(analysis_result, str(json_path), format='json')
        logger.info(f"📁 Results exported to JSON: {json_path}")
        
        # Export as CSV
        csv_path = output_dir / "robustness_analysis_example.csv"
        analyzer.export_robustness_results(analysis_result, str(csv_path), format='csv')
        logger.info(f"📁 Results exported to CSV: {csv_path}")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not export results: {e}")
    
    return analysis_result

def main():
    """Run all robustness analyzer examples."""
    logger.info("🚀 Starting Robustness Analyzer Examples")
    
    try:
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Demonstrate different aspects of the robustness analyzer
        logger.info("\n" + "="*60)
        sweep_configs = demonstrate_parameter_sweep_configuration()
        
        logger.info("\n" + "="*60)
        single_curve = demonstrate_single_parameter_analysis()
        
        logger.info("\n" + "="*60)
        multi_param_result = demonstrate_multi_parameter_analysis()
        
        logger.info("\n" + "="*60)
        comparison_result = demonstrate_multi_model_comparison()
        
        logger.info("\n" + "="*60)
        visualization_result = demonstrate_visualization_and_export()
        
        logger.info("\n" + "="*60)
        logger.info("✅ All Robustness Analyzer Examples Completed Successfully!")
        
        # Summary statistics
        logger.info("\n📊 Example Summary:")
        logger.info(f"🔧 Parameter sweep configurations: {len(sweep_configs)}")
        logger.info(f"📈 Single parameter AUC: {single_curve.auc_success_rate:.3f}")
        logger.info(f"🎯 Multi-parameter score: {multi_param_result.overall_robustness_score:.3f}")
        logger.info(f"🏆 Best model in comparison: {comparison_result.robustness_rankings[0][0]}")
        logger.info(f"📊 Visualization model score: {visualization_result.overall_robustness_score:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in robustness analyzer examples: {e}")
        raise

if __name__ == "__main__":
    main()