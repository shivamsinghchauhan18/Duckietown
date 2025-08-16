#!/usr/bin/env python3
"""
📊 REPORT GENERATOR EXAMPLE 📊
Example usage of the ReportGenerator for comprehensive evaluation reports

This example demonstrates:
- Comprehensive evaluation report generation
- Leaderboard generation with confidence intervals
- Per-map performance tables and statistical comparison matrices
- Pareto plots and robustness curve visualizations
- Executive summary generation with recommendations
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.report_generator import ReportGenerator, ReportConfig
from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult, ConfidenceInterval
from duckietown_utils.champion_selector import ChampionSelector, ChampionSelectionResult
from duckietown_utils.robustness_analyzer import RobustnessAnalyzer, RobustnessAnalysisResult, RobustnessCurve, ParameterPoint, ParameterType
# Note: failure_analyzer returns Dict[str, Any] rather than a specific result class

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
    
    # Primary metrics with confidence intervals
    success_rate = np.clip(base_performance + noise[0], 0.0, 1.0)
    success_rate_ci = ConfidenceInterval(
        lower=max(0.0, success_rate - 0.05),
        upper=min(1.0, success_rate + 0.05),
        confidence_level=0.95,
        method="wilson"
    )
    
    mean_reward = np.clip(base_performance * 0.8 + noise[1] * 0.1, 0.0, 1.0)
    reward_ci = ConfidenceInterval(
        lower=max(0.0, mean_reward - 0.03),
        upper=min(1.0, mean_reward + 0.03),
        confidence_level=0.95,
        method="bootstrap"
    )
    
    episode_length = max(300, 600 - base_performance * 200 + noise[2] * 50)
    lateral_deviation = max(0.01, 0.2 - base_performance * 0.15 + abs(noise[3]) * 0.05)
    heading_error = max(1.0, 15.0 - base_performance * 10 + abs(noise[4]) * 3)
    smoothness = max(0.01, 0.15 - base_performance * 0.1 + abs(noise[5]) * 0.03)
    
    primary_metrics = {
        'success_rate': MetricResult(
            name='success_rate', 
            value=success_rate,
            confidence_interval=success_rate_ci,
            sample_size=100
        ),
        'mean_reward': MetricResult(
            name='mean_reward', 
            value=mean_reward,
            confidence_interval=reward_ci,
            sample_size=100
        ),
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
    
    composite_score_ci = ConfidenceInterval(
        lower=max(0.0, composite_score - 0.02),
        upper=min(1.0, composite_score + 0.02),
        confidence_level=0.95,
        method="bootstrap"
    )
    
    composite_score_result = MetricResult(
        name='composite_score', 
        value=composite_score,
        confidence_interval=composite_score_ci,
        sample_size=100
    )
    
    # Per-map metrics (simulate 5 different maps)
    per_map_metrics = {}
    map_names = ['loop_empty', 'zigzag', 'techtrack', 'multi_track', 'huge_loop']
    
    for i, map_name in enumerate(map_names):
        # Add map-specific variation
        map_success_rate = np.clip(success_rate + np.random.normal(0, 0.05), 0.0, 1.0)
        map_lateral_dev = lateral_deviation + np.random.normal(0, 0.02)
        
        per_map_metrics[map_name] = {
            'success_rate': MetricResult(
                name='success_rate', 
                value=map_success_rate,
                confidence_interval=ConfidenceInterval(
                    lower=max(0.0, map_success_rate - 0.08),
                    upper=min(1.0, map_success_rate + 0.08),
                    confidence_level=0.95,
                    method="wilson"
                ),
                sample_size=20
            ),
            'lateral_deviation': MetricResult(name='lateral_deviation', value=map_lateral_dev)
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

def create_sample_robustness_results(model_metrics_list: List[ModelMetrics]) -> Dict[str, RobustnessAnalysisResult]:
    """Create sample robustness analysis results.
    
    Args:
        model_metrics_list: List of model metrics
        
    Returns:
        Dict[str, RobustnessAnalysisResult]: Robustness results by model
    """
    robustness_results = {}
    
    for model_metrics in model_metrics_list:
        base_performance = model_metrics.primary_metrics['success_rate'].value
        
        # Create sample parameter curves
        parameter_curves = {}
        
        # Lighting intensity curve
        lighting_points = []
        for param_val in np.linspace(0.5, 2.0, 8):
            # Performance degrades as we move away from 1.0
            performance_factor = 1.0 - 0.3 * abs(param_val - 1.0)
            success_rate = base_performance * max(0.3, performance_factor)
            
            point = ParameterPoint(
                parameter_value=param_val,
                success_rate=success_rate,
                mean_reward=success_rate * 0.8,
                stability=success_rate * 1.2,
                sample_size=50
            )
            lighting_points.append(point)
        
        lighting_curve = RobustnessCurve(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            model_id=model_metrics.model_id,
            sweep_points=lighting_points,
            auc_success_rate=np.trapz([p.success_rate for p in lighting_points], 
                                    [p.parameter_value for p in lighting_points]) / 1.5,
            auc_reward=np.trapz([p.mean_reward for p in lighting_points], 
                              [p.parameter_value for p in lighting_points]) / 1.5,
            auc_stability=np.trapz([p.stability for p in lighting_points], 
                                 [p.parameter_value for p in lighting_points]) / 1.5,
            sensitivity_threshold=1.5 if base_performance > 0.8 else float('inf'),
            operating_range=(0.7, 1.4) if base_performance > 0.8 else None,
            degradation_points=[p for p in lighting_points if p.success_rate < base_performance * 0.9]
        )
        
        parameter_curves['lighting_intensity'] = lighting_curve
        
        # Overall robustness score
        overall_score = lighting_curve.auc_success_rate
        
        robustness_result = RobustnessAnalysisResult(
            model_id=model_metrics.model_id,
            parameter_curves=parameter_curves,
            overall_robustness_score=overall_score,
            sensitivity_summary={'lighting_intensity': lighting_curve.sensitivity_threshold},
            operating_ranges={'lighting_intensity': lighting_curve.operating_range},
            recommendations=[
                f"Model shows {'good' if overall_score > 0.7 else 'limited'} robustness to lighting changes",
                "Consider additional training with varied lighting conditions" if overall_score < 0.7 else "Robustness is acceptable for deployment"
            ]
        )
        
        robustness_results[model_metrics.model_id] = robustness_result
    
    return robustness_results

def create_sample_failure_results(model_metrics_list: List[ModelMetrics]) -> Dict[str, Dict[str, Any]]:
    """Create sample failure analysis results.
    
    Args:
        model_metrics_list: List of model metrics
        
    Returns:
        Dict[str, FailureAnalysisResult]: Failure results by model
    """
    failure_results = {}
    
    for model_metrics in model_metrics_list:
        success_rate = model_metrics.primary_metrics['success_rate'].value
        failure_rate = 1.0 - success_rate
        
        # Distribute failures across different types
        total_episodes = 100
        failed_episodes = int(total_episodes * failure_rate)
        
        failure_distribution = {
            'collision': max(1, int(failed_episodes * 0.4)),
            'off_lane': max(1, int(failed_episodes * 0.3)),
            'stuck': max(0, int(failed_episodes * 0.2)),
            'oscillation': max(0, int(failed_episodes * 0.1))
        }
        
        # Ensure total matches
        actual_total = sum(failure_distribution.values())
        if actual_total != failed_episodes and failed_episodes > 0:
            failure_distribution['collision'] += failed_episodes - actual_total
        
        # Create failure result as dictionary (matching failure_analyzer output format)
        common_patterns = [
            "Failures often occur at sharp turns" if 'collision' in failure_distribution else None,
            "Lane departure common in complex scenarios" if 'off_lane' in failure_distribution else None
        ]
        recommendations = [
            "Focus on collision avoidance training" if failure_distribution.get('collision', 0) > 2 else None,
            "Improve lane-following stability" if failure_distribution.get('off_lane', 0) > 2 else None
        ]
        
        # Filter out None values
        common_patterns = [p for p in common_patterns if p]
        recommendations = [r for r in recommendations if r]
        
        failure_result = {
            'report_metadata': {
                'generation_time': '2024-01-01T00:00:00',
                'analyzed_models': [model_metrics.model_id]
            },
            'failure_statistics': {
                'total_episodes': total_episodes,
                'failure_type_counts': failure_distribution,
                'failure_rate_by_map': {
                    'loop_empty': failure_rate * 0.8,
                    'zigzag': failure_rate * 1.2,
                    'techtrack': failure_rate * 1.1,
                    'multi_track': failure_rate * 1.3,
                    'huge_loop': failure_rate * 0.9
                }
            },
            'episode_summaries': [],
            'recommendations': recommendations
        }
        
        failure_results[model_metrics.model_id] = failure_result
    
    return failure_results

def demonstrate_basic_report_generation():
    """Demonstrate basic report generation functionality."""
    print("📊 Basic Report Generation Example")
    print("=" * 50)
    
    # Create sample model metrics
    model_candidates = [
        create_sample_model_metrics("elite_model_v3", 0.92, 0.02),      # Excellent
        create_sample_model_metrics("advanced_model_v2", 0.88, 0.03),   # Very good
        create_sample_model_metrics("standard_model_v4", 0.85, 0.04),   # Good
        create_sample_model_metrics("baseline_model_v1", 0.80, 0.05),   # Decent
        create_sample_model_metrics("experimental_v1", 0.75, 0.08),     # Experimental
    ]
    
    print(f"✅ Created {len(model_candidates)} model candidates")
    
    # Initialize report generator with custom configuration
    config = {
        'include_confidence_intervals': True,
        'include_statistical_tests': True,
        'include_pareto_analysis': True,
        'plot_style': 'seaborn-v0_8',
        'plot_dpi': 150,
        'save_plots': True,
        'generate_html': True
    }
    
    report_generator = ReportGenerator(config)
    print("✅ Initialized ReportGenerator")
    
    # Generate basic report
    report = report_generator.generate_comprehensive_report(
        model_metrics_list=model_candidates,
        report_id="basic_example_report"
    )
    
    print(f"✅ Generated report: {report.report_id}")
    print(f"📊 Champion model: {report.executive_summary.champion_model}")
    print(f"🎯 Deployment readiness: {report.executive_summary.deployment_readiness}")
    print(f"📈 Total plots generated: {len(report.plots)}")
    
    # Display leaderboard summary
    print("\n📋 Leaderboard Summary (Top 3):")
    print("-" * 60)
    print(f"{'Rank':<5} {'Model ID':<20} {'Score':<8} {'Success Rate':<12}")
    print("-" * 60)
    
    for entry in report.leaderboard[:3]:
        print(f"{entry.rank:<5} {entry.model_id:<20} {entry.composite_score:<8.3f} {entry.success_rate:<12.1%}")
    
    return report

def demonstrate_comprehensive_report_generation():
    """Demonstrate comprehensive report generation with all components."""
    print("\n📊 Comprehensive Report Generation Example")
    print("=" * 50)
    
    # Create sample model metrics
    model_candidates = [
        create_sample_model_metrics("champion_model_v5", 0.94, 0.01),
        create_sample_model_metrics("runner_up_v3", 0.91, 0.02),
        create_sample_model_metrics("solid_performer_v2", 0.87, 0.03),
        create_sample_model_metrics("baseline_v4", 0.82, 0.04),
        create_sample_model_metrics("experimental_v2", 0.78, 0.06),
        create_sample_model_metrics("legacy_model", 0.73, 0.05),
    ]
    
    print(f"✅ Created {len(model_candidates)} model candidates")
    
    # Generate champion selection results
    champion_selector = ChampionSelector()
    champion_selection_result = champion_selector.select_champion(model_candidates)
    print(f"✅ Champion selected: {champion_selection_result.new_champion_id}")
    
    # Generate robustness analysis results
    robustness_results = create_sample_robustness_results(model_candidates)
    print(f"✅ Generated robustness analysis for {len(robustness_results)} models")
    
    # Generate failure analysis results
    failure_results = create_sample_failure_results(model_candidates)
    print(f"✅ Generated failure analysis for {len(failure_results)} models")
    
    # Initialize report generator
    config = {
        'include_confidence_intervals': True,
        'include_statistical_tests': True,
        'include_pareto_analysis': True,
        'include_robustness_analysis': True,
        'include_failure_analysis': True,
        'plot_style': 'seaborn-v0_8',
        'plot_dpi': 200,
        'save_plots': True,
        'generate_html': True,
        'color_palette': 'Set2'
    }
    
    report_generator = ReportGenerator(config)
    
    # Generate comprehensive report
    report = report_generator.generate_comprehensive_report(
        model_metrics_list=model_candidates,
        champion_selection_result=champion_selection_result,
        robustness_results=robustness_results,
        failure_results=failure_results,
        report_id="comprehensive_example_report"
    )
    
    print(f"✅ Generated comprehensive report: {report.report_id}")
    
    # Display executive summary
    print("\n📋 Executive Summary:")
    print("-" * 40)
    print(f"Champion: {report.executive_summary.champion_model}")
    print(f"Total Models: {report.executive_summary.total_models_evaluated}")
    print(f"Deployment Readiness: {report.executive_summary.deployment_readiness}")
    
    print("\n🔍 Key Findings:")
    for i, finding in enumerate(report.executive_summary.key_findings, 1):
        print(f"  {i}. {finding}")
    
    print("\n💡 Recommendations:")
    for i, recommendation in enumerate(report.executive_summary.recommendations, 1):
        print(f"  {i}. {recommendation}")
    
    print("\n⚠️  Risk Assessment:")
    for category, risk in report.executive_summary.risk_assessment.items():
        print(f"  {category.title()}: {risk}")
    
    # Display performance highlights
    print("\n📊 Performance Highlights:")
    highlights = report.executive_summary.performance_highlights
    print(f"  Champion Composite Score: {highlights.get('champion_composite_score', 'N/A'):.3f}")
    print(f"  Champion Success Rate: {highlights.get('champion_success_rate', 0):.1%}")
    print(f"  Models Above 80% Success: {highlights.get('models_above_threshold', 0)}")
    
    # Display analysis summaries
    if report.robustness_analysis:
        print(f"\n🔬 Robustness Analysis: {report.robustness_analysis['total_models_analyzed']} models analyzed")
        if 'overall_rankings' in report.robustness_analysis:
            top_robust = report.robustness_analysis['overall_rankings'][0]
            print(f"  Most Robust Model: {top_robust[0]} (score: {top_robust[1]:.3f})")
    
    if report.failure_analysis:
        print(f"\n🚨 Failure Analysis: {report.failure_analysis['total_models_analyzed']} models analyzed")
        if report.failure_analysis['common_failure_modes']:
            print(f"  Common Failure Modes: {', '.join(report.failure_analysis['common_failure_modes'])}")
    
    print(f"\n📊 Visualizations Generated: {len(report.plots)}")
    for plot_name in report.plots.keys():
        print(f"  - {plot_name.replace('_', ' ').title()}")
    
    return report

def demonstrate_custom_report_configuration():
    """Demonstrate custom report configuration options."""
    print("\n📊 Custom Report Configuration Example")
    print("=" * 50)
    
    # Create sample data
    model_candidates = [
        create_sample_model_metrics("custom_model_a", 0.89, 0.02),
        create_sample_model_metrics("custom_model_b", 0.86, 0.03),
        create_sample_model_metrics("custom_model_c", 0.83, 0.04),
    ]
    
    # Custom configuration for minimal report
    minimal_config = {
        'include_confidence_intervals': False,
        'include_statistical_tests': False,
        'include_pareto_analysis': False,
        'include_robustness_analysis': False,
        'include_failure_analysis': False,
        'save_plots': False,
        'generate_html': False
    }
    
    print("📊 Generating minimal report...")
    minimal_generator = ReportGenerator(minimal_config)
    minimal_report = minimal_generator.generate_comprehensive_report(
        model_metrics_list=model_candidates,
        report_id="minimal_report"
    )
    
    print(f"✅ Minimal report generated with {len(minimal_report.plots)} plots")
    
    # Custom configuration for detailed report
    detailed_config = {
        'include_confidence_intervals': True,
        'include_statistical_tests': True,
        'include_pareto_analysis': True,
        'plot_style': 'ggplot',
        'plot_dpi': 300,
        'plot_format': 'pdf',
        'color_palette': 'viridis',
        'figure_size': (14, 10),
        'font_size': 12,
        'save_plots': True,
        'generate_html': True,
        'generate_pdf': False
    }
    
    print("📊 Generating detailed report with custom styling...")
    detailed_generator = ReportGenerator(detailed_config)
    detailed_report = detailed_generator.generate_comprehensive_report(
        model_metrics_list=model_candidates,
        report_id="detailed_custom_report"
    )
    
    print(f"✅ Detailed report generated with {len(detailed_report.plots)} plots")
    print(f"🎨 Plot format: {detailed_config['plot_format']}")
    print(f"🎨 Color palette: {detailed_config['color_palette']}")
    
    return minimal_report, detailed_report

def demonstrate_report_analysis():
    """Demonstrate analysis of generated reports."""
    print("\n📊 Report Analysis Example")
    print("=" * 50)
    
    # Generate a sample report
    model_candidates = [
        create_sample_model_metrics("analysis_model_1", 0.90, 0.02),
        create_sample_model_metrics("analysis_model_2", 0.85, 0.03),
        create_sample_model_metrics("analysis_model_3", 0.80, 0.04),
    ]
    
    report_generator = ReportGenerator()
    report = report_generator.generate_comprehensive_report(
        model_metrics_list=model_candidates,
        report_id="analysis_example"
    )
    
    # Analyze report contents
    print("🔍 Report Analysis:")
    print(f"  Report ID: {report.report_id}")
    print(f"  Generation Time: {report.generation_timestamp}")
    print(f"  Total Models: {len(report.leaderboard)}")
    
    # Analyze leaderboard
    print("\n📊 Leaderboard Analysis:")
    scores = [entry.composite_score for entry in report.leaderboard]
    print(f"  Score Range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"  Score Spread: {max(scores) - min(scores):.3f}")
    print(f"  Average Score: {np.mean(scores):.3f}")
    
    # Analyze performance tables
    print(f"\n📋 Performance Tables: {len(report.performance_tables)}")
    for table_name, table in report.performance_tables.items():
        print(f"  {table_name}: {table.table_data.shape[0]} rows × {table.table_data.shape[1]} columns")
        if table.best_performers:
            print(f"    Best performers: {len(table.best_performers)} categories")
    
    # Analyze statistical comparisons
    if report.statistical_comparisons.comparisons:
        significant_count = sum(1 for comp in report.statistical_comparisons.comparisons if comp.is_significant)
        total_count = len(report.statistical_comparisons.comparisons)
        print(f"\n📈 Statistical Comparisons: {significant_count}/{total_count} significant")
        print(f"  Correction method: {report.statistical_comparisons.correction_method}")
    
    # Analyze executive summary
    print(f"\n📋 Executive Summary:")
    print(f"  Key findings: {len(report.executive_summary.key_findings)}")
    print(f"  Recommendations: {len(report.executive_summary.recommendations)}")
    print(f"  Risk categories: {len(report.executive_summary.risk_assessment)}")
    
    return report

def main():
    """Run all report generator examples."""
    print("🚀 Starting Report Generator Examples")
    
    try:
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Run different examples
        print("\n" + "="*60)
        basic_report = demonstrate_basic_report_generation()
        
        print("\n" + "="*60)
        comprehensive_report = demonstrate_comprehensive_report_generation()
        
        print("\n" + "="*60)
        minimal_report, detailed_report = demonstrate_custom_report_configuration()
        
        print("\n" + "="*60)
        analysis_report = demonstrate_report_analysis()
        
        print("\n" + "="*60)
        print("✅ All Report Generator Examples Completed Successfully!")
        
        # Summary statistics
        print("\n📊 Example Summary:")
        print(f"📋 Basic report models: {len(basic_report.leaderboard)}")
        print(f"🏆 Comprehensive report champion: {comprehensive_report.executive_summary.champion_model}")
        print(f"📊 Minimal report plots: {len(minimal_report.plots)}")
        print(f"📊 Detailed report plots: {len(detailed_report.plots)}")
        print(f"🔍 Analysis report score range: {max([e.composite_score for e in analysis_report.leaderboard]) - min([e.composite_score for e in analysis_report.leaderboard]):.3f}")
        
        # Output directory information
        output_dir = Path("logs/evaluation_reports")
        if output_dir.exists():
            report_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            print(f"\n📁 Generated {len(report_dirs)} report directories in {output_dir}")
            for report_dir in report_dirs:
                files = list(report_dir.glob("*"))
                print(f"  {report_dir.name}: {len(files)} files")
        
    except Exception as e:
        print(f"❌ Error in report generator examples: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()