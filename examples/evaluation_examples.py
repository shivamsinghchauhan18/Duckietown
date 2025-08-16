#!/usr/bin/env python3
"""
Example evaluation scripts for common use cases with the Enhanced Duckietown RL evaluation system.

This module provides practical examples for different evaluation scenarios including
basic model comparison, champion selection, robustness analysis, and deployment readiness testing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator
from duckietown_utils.champion_selector import ChampionSelector
from duckietown_utils.report_generator import ReportGenerator
from duckietown_utils.robustness_analyzer import RobustnessAnalyzer
from config.evaluation_config import EvaluationConfig


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Set up logging for evaluation examples."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation_examples.log')
        ]
    )
    return logging.getLogger(__name__)


def example_1_basic_model_evaluation():
    """
    Example 1: Basic Model Evaluation
    
    Demonstrates how to evaluate a single model across basic test suites
    with standard configuration.
    """
    logger = setup_logging()
    logger.info("Running Example 1: Basic Model Evaluation")
    
    # Create basic evaluation configuration
    config = EvaluationConfig(
        suites=['base', 'hard'],
        seeds_per_map=25,
        policy_modes=['deterministic'],
        compute_ci=True,
        export_plots=True,
        record_videos=False  # Disable for faster execution
    )
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    
    # Evaluate single model
    model_path = 'models/my_trained_model.pkl'
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file {model_path} not found. Creating mock evaluation.")
        # For demonstration, create mock results
        results = create_mock_evaluation_results('my_model')
    else:
        results = orchestrator.run_single_evaluation(
            model_path=model_path,
            model_id='my_model'
        )
    
    # Print summary results
    logger.info(f"Evaluation Results for {results.model_id}:")
    logger.info(f"  Global Score: {results.global_score:.3f}")
    logger.info(f"  Success Rate: {results.suite_results['base'].metrics.success_rate:.3f}")
    logger.info(f"  Mean Reward: {results.suite_results['base'].metrics.mean_reward:.3f}")
    
    return results


def example_2_multi_model_comparison():
    """
    Example 2: Multi-Model Comparison
    
    Demonstrates how to evaluate and compare multiple models with
    statistical significance testing.
    """
    logger = setup_logging()
    logger.info("Running Example 2: Multi-Model Comparison")
    
    # Create configuration for model comparison
    config = EvaluationConfig(
        suites=['base', 'hard', 'ood'],
        seeds_per_map=50,
        policy_modes=['deterministic', 'stochastic'],
        compute_ci=True,
        bootstrap_resamples=10000,
        significance_correction='benjamini_hochberg',
        export_csv_json=True,
        export_plots=True
    )
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    
    # Define models to compare
    model_paths = [
        'models/baseline_model.pkl',
        'models/enhanced_model_v1.pkl',
        'models/enhanced_model_v2.pkl',
        'models/champion_model.pkl'
    ]
    
    model_ids = ['baseline', 'enhanced_v1', 'enhanced_v2', 'champion']
    
    # Check if models exist, create mock data if not
    existing_models = []
    for path, model_id in zip(model_paths, model_ids):
        if os.path.exists(path):
            existing_models.append((path, model_id))
        else:
            logger.warning(f"Model {path} not found, creating mock data")
    
    if not existing_models:
        # Create mock results for demonstration
        results = {}
        for model_id in model_ids:
            results[model_id] = create_mock_evaluation_results(model_id)
    else:
        # Evaluate existing models
        results = orchestrator.evaluate_models(
            [path for path, _ in existing_models],
            [model_id for _, model_id in existing_models]
        )
    
    # Generate comparison report
    reporter = ReportGenerator(config)
    report_artifacts = reporter.generate_comprehensive_report(
        results,
        output_dir='evaluation_reports/multi_model_comparison',
        report_name='model_comparison'
    )
    
    # Print comparison summary
    logger.info("Model Comparison Results:")
    sorted_results = sorted(results.items(), key=lambda x: x[1].global_score, reverse=True)
    
    for rank, (model_id, result) in enumerate(sorted_results, 1):
        logger.info(f"  {rank}. {model_id}: Score = {result.global_score:.3f} "
                   f"(CI: {result.global_score_ci[0]:.3f}-{result.global_score_ci[1]:.3f})")
    
    logger.info(f"Detailed report saved to: {report_artifacts.html_report_path}")
    
    return results, report_artifacts


def example_3_champion_selection():
    """
    Example 3: Champion Selection
    
    Demonstrates automated champion selection with statistical validation
    and Pareto front analysis.
    """
    logger = setup_logging()
    logger.info("Running Example 3: Champion Selection")
    
    # Create configuration with champion selection criteria
    config = EvaluationConfig(
        suites=['base', 'hard', 'law', 'ood'],
        seeds_per_map=75,
        policy_modes=['deterministic'],
        use_composite=True,
        pareto_axes=[
            ['SR', '-D', '-J'],  # Success Rate vs Lateral Deviation vs Smoothness
            ['SR', 'R']          # Success Rate vs Reward
        ],
        keep_top_k=3
    )
    
    # Initialize components
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    selector = ChampionSelector(config)
    
    # Candidate models for champion selection
    candidate_models = [
        ('models/candidate_1.pkl', 'candidate_1'),
        ('models/candidate_2.pkl', 'candidate_2'),
        ('models/candidate_3.pkl', 'candidate_3'),
        ('models/candidate_4.pkl', 'candidate_4'),
        ('models/candidate_5.pkl', 'candidate_5')
    ]
    
    # Evaluate candidates (using mock data for demonstration)
    candidate_results = []
    for i, (path, model_id) in enumerate(candidate_models):
        if os.path.exists(path):
            result = orchestrator.run_single_evaluation(path, model_id)
        else:
            result = create_mock_evaluation_results(model_id, base_score=0.7 + i*0.05)
        candidate_results.append(result)
    
    # Load current champion (if exists)
    current_champion = None
    champion_path = 'models/current_champion.pkl'
    if os.path.exists(champion_path):
        current_champion = orchestrator.run_single_evaluation(champion_path, 'current_champion')
    else:
        current_champion = create_mock_evaluation_results('current_champion', base_score=0.82)
    
    # Select new champion
    champion_selection = selector.select_champion(
        candidate_results,
        current_champion
    )
    
    # Analyze Pareto fronts
    pareto_analysis = selector.analyze_pareto_front(
        candidate_results,
        objectives=['SR', '-D', '-J']
    )
    
    # Print champion selection results
    logger.info("Champion Selection Results:")
    logger.info(f"  New Champion: {champion_selection.champion_id}")
    logger.info(f"  Champion Score: {champion_selection.champion_score:.3f}")
    logger.info(f"  Improvement over current: {champion_selection.improvement:.3f}")
    logger.info(f"  Statistical significance: p = {champion_selection.significance_p:.4f}")
    
    logger.info("Pareto Front Analysis:")
    logger.info(f"  Non-dominated models: {len(pareto_analysis.non_dominated_models)}")
    for model_id in pareto_analysis.non_dominated_models:
        logger.info(f"    - {model_id}")
    
    return champion_selection, pareto_analysis


def example_4_robustness_analysis():
    """
    Example 4: Robustness Analysis
    
    Demonstrates robustness analysis across environmental parameter sweeps
    to understand model sensitivity and operating boundaries.
    """
    logger = setup_logging()
    logger.info("Running Example 4: Robustness Analysis")
    
    # Create configuration for robustness analysis
    config = EvaluationConfig(
        suites=['base'],  # Use base suite for parameter sweeps
        seeds_per_map=30,
        policy_modes=['deterministic'],
        compute_ci=True
    )
    
    # Initialize components
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    robustness_analyzer = RobustnessAnalyzer(config)
    
    # Define parameter ranges for robustness testing
    parameter_ranges = {
        'lighting_intensity': [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7],
        'camera_noise': [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'friction_coefficient': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        'spawn_pose_noise': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # Base configuration for parameter sweeps
    base_config = {
        'lighting_intensity': 1.0,
        'camera_noise': 0.0,
        'friction_coefficient': 1.0,
        'spawn_pose_noise': 0.0
    }
    
    # Load model for robustness analysis
    model_path = 'models/robustness_test_model.pkl'
    if not os.path.exists(model_path):
        logger.warning(f"Model {model_path} not found, creating mock analysis")
        # Create mock robustness analysis
        robustness_results = create_mock_robustness_analysis(parameter_ranges)
    else:
        # Load model (placeholder - actual implementation would load the model)
        model = None  # load_model(model_path)
        
        # Perform robustness analysis
        robustness_results = robustness_analyzer.analyze_robustness(
            model=model,
            parameter_ranges=parameter_ranges,
            base_config=base_config
        )
    
    # Print robustness analysis results
    logger.info("Robustness Analysis Results:")
    for parameter, analysis in robustness_results.parameter_analyses.items():
        logger.info(f"  {parameter}:")
        logger.info(f"    AUC Robustness: {analysis.auc_robustness:.3f}")
        logger.info(f"    Sensitivity Threshold: {analysis.sensitivity_threshold:.3f}")
        logger.info(f"    Operating Range: {analysis.operating_range[0]:.2f} - {analysis.operating_range[1]:.2f}")
    
    # Generate robustness plots
    plot_paths = robustness_analyzer.generate_robustness_plots(
        robustness_results,
        output_dir='evaluation_reports/robustness_analysis'
    )
    
    logger.info(f"Robustness plots saved to: {plot_paths}")
    
    return robustness_results


def example_5_deployment_readiness():
    """
    Example 5: Deployment Readiness Testing
    
    Demonstrates comprehensive evaluation for deployment readiness with
    strict acceptance criteria and safety validation.
    """
    logger = setup_logging()
    logger.info("Running Example 5: Deployment Readiness Testing")
    
    # Create strict configuration for deployment testing
    config = EvaluationConfig(
        suites=['base', 'hard', 'ood', 'stress'],
        seeds_per_map=100,  # High statistical power
        policy_modes=['deterministic'],
        compute_ci=True,
        confidence_level=0.99,  # High confidence for safety
        bootstrap_resamples=15000,
        record_videos=True,
        save_worst_k=10  # Save more failure cases for analysis
    )
    
    # Define strict acceptance criteria
    acceptance_criteria = {
        'min_global_score': 0.85,
        'min_success_rate': 0.95,
        'max_lateral_deviation': 0.10,
        'max_heading_error': 5.0,  # degrees
        'min_stability': 0.8,
        'min_maps_passing': 0.90  # 90% of maps must meet criteria
    }
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    
    # Evaluate deployment candidate
    candidate_path = 'models/deployment_candidate.pkl'
    if not os.path.exists(candidate_path):
        logger.warning(f"Model {candidate_path} not found, creating mock evaluation")
        results = create_mock_evaluation_results('deployment_candidate', base_score=0.87)
    else:
        results = orchestrator.run_single_evaluation(
            model_path=candidate_path,
            model_id='deployment_candidate'
        )
    
    # Evaluate against acceptance criteria
    deployment_readiness = evaluate_deployment_readiness(results, acceptance_criteria)
    
    # Generate deployment report
    reporter = ReportGenerator(config)
    report_artifacts = reporter.generate_deployment_report(
        results,
        acceptance_criteria,
        output_dir='evaluation_reports/deployment_readiness'
    )
    
    # Print deployment readiness results
    logger.info("Deployment Readiness Assessment:")
    logger.info(f"  Overall Status: {'PASS' if deployment_readiness['ready'] else 'FAIL'}")
    logger.info(f"  Global Score: {results.global_score:.3f} (Required: {acceptance_criteria['min_global_score']:.3f})")
    
    for criterion, result in deployment_readiness['criteria_results'].items():
        status = 'PASS' if result['passed'] else 'FAIL'
        logger.info(f"  {criterion}: {status} ({result['value']:.3f})")
    
    if not deployment_readiness['ready']:
        logger.warning("Model NOT ready for deployment. Issues found:")
        for issue in deployment_readiness['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Model READY for deployment!")
    
    logger.info(f"Deployment report saved to: {report_artifacts.html_report_path}")
    
    return deployment_readiness, report_artifacts


def example_6_continuous_evaluation():
    """
    Example 6: Continuous Evaluation Pipeline
    
    Demonstrates setting up continuous evaluation for integration with
    training pipelines and CI/CD systems.
    """
    logger = setup_logging()
    logger.info("Running Example 6: Continuous Evaluation Pipeline")
    
    # Create configuration for continuous evaluation
    config = EvaluationConfig(
        suites=['base', 'hard'],  # Faster suites for CI
        seeds_per_map=25,
        policy_modes=['deterministic'],
        compute_ci=False,  # Faster execution
        export_plots=False,
        record_videos=False,
        timeout_seconds=120  # Shorter timeout for CI
    )
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    
    # Simulate continuous evaluation workflow
    def evaluate_training_checkpoint(checkpoint_path: str, iteration: int) -> Dict[str, Any]:
        """Evaluate a training checkpoint."""
        model_id = f'checkpoint_{iteration}'
        
        if os.path.exists(checkpoint_path):
            results = orchestrator.run_single_evaluation(checkpoint_path, model_id)
        else:
            # Mock results for demonstration
            results = create_mock_evaluation_results(model_id, base_score=0.6 + iteration*0.02)
        
        # Check if this is a new best model
        is_best = results.global_score > get_current_best_score()
        
        # Log results
        logger.info(f"Checkpoint {iteration}: Score = {results.global_score:.3f} "
                   f"{'(NEW BEST!)' if is_best else ''}")
        
        # Update champion if this is the best model
        if is_best:
            update_champion_model(checkpoint_path, results)
        
        return {
            'iteration': iteration,
            'score': results.global_score,
            'is_best': is_best,
            'results': results
        }
    
    # Simulate evaluating multiple training checkpoints
    checkpoint_results = []
    for iteration in range(1, 11):
        checkpoint_path = f'checkpoints/model_checkpoint_{iteration}.pkl'
        result = evaluate_training_checkpoint(checkpoint_path, iteration)
        checkpoint_results.append(result)
    
    # Generate training progress report
    generate_training_progress_report(checkpoint_results, 'evaluation_reports/continuous_evaluation')
    
    logger.info("Continuous evaluation completed. Best model updated if improved.")
    
    return checkpoint_results


def example_7_custom_metrics_evaluation():
    """
    Example 7: Custom Metrics Evaluation
    
    Demonstrates how to extend the evaluation system with custom metrics
    and specialized analysis for specific research objectives.
    """
    logger = setup_logging()
    logger.info("Running Example 7: Custom Metrics Evaluation")
    
    # Create configuration with custom metrics
    config = EvaluationConfig(
        suites=['base', 'hard'],
        seeds_per_map=30,
        policy_modes=['deterministic'],
        compute_ci=True
    )
    
    # Define custom metrics
    custom_metrics = {
        'lane_change_success_rate': calculate_lane_change_success_rate,
        'object_avoidance_efficiency': calculate_object_avoidance_efficiency,
        'energy_efficiency': calculate_energy_efficiency,
        'comfort_score': calculate_comfort_score
    }
    
    # Initialize orchestrator with custom metrics
    orchestrator = EvaluationOrchestrator(config, logger=logger)
    orchestrator.register_custom_metrics(custom_metrics)
    
    # Evaluate model with custom metrics
    model_path = 'models/custom_metrics_model.pkl'
    if not os.path.exists(model_path):
        logger.warning(f"Model {model_path} not found, creating mock evaluation")
        results = create_mock_evaluation_results_with_custom_metrics('custom_model')
    else:
        results = orchestrator.run_single_evaluation(model_path, 'custom_model')
    
    # Print custom metrics results
    logger.info("Custom Metrics Results:")
    for metric_name, value in results.custom_metrics.items():
        logger.info(f"  {metric_name}: {value:.3f}")
    
    # Generate custom analysis report
    custom_report = generate_custom_metrics_report(results, 'evaluation_reports/custom_metrics')
    
    logger.info(f"Custom metrics report saved to: {custom_report}")
    
    return results


# Helper functions for examples

def create_mock_evaluation_results(model_id: str, base_score: float = 0.75) -> 'ModelEvaluationResults':
    """Create mock evaluation results for demonstration purposes."""
    import random
    from dataclasses import dataclass
    from typing import Dict, Tuple, Optional, Any
    
    # Mock data structures (simplified)
    @dataclass
    class MockSuiteResults:
        success_rate: float
        mean_reward: float
        lateral_deviation: float
        heading_error: float
        smoothness: float
        
    @dataclass
    class MockMetrics:
        success_rate: float
        mean_reward: float
        lateral_deviation: float
        heading_error: float
        smoothness: float
        
    @dataclass
    class MockEvaluationResults:
        model_id: str
        global_score: float
        global_score_ci: Tuple[float, float]
        suite_results: Dict[str, Any]
        pareto_rank: int = 1
        champion_comparison: Optional[Any] = None
        failure_analysis: Optional[Any] = None
        robustness_analysis: Optional[Any] = None
        metadata: Dict[str, Any] = None
        custom_metrics: Dict[str, float] = None
    
    # Generate mock results with some randomness
    random.seed(hash(model_id) % 1000)  # Deterministic randomness based on model_id
    
    noise = random.uniform(-0.1, 0.1)
    score = max(0.0, min(1.0, base_score + noise))
    
    suite_results = {}
    for suite in ['base', 'hard', 'law', 'ood', 'stress']:
        suite_noise = random.uniform(-0.05, 0.05)
        suite_score = max(0.0, min(1.0, score + suite_noise))
        
        metrics = MockMetrics(
            success_rate=max(0.0, min(1.0, suite_score + random.uniform(-0.1, 0.1))),
            mean_reward=max(0.0, min(1.0, suite_score + random.uniform(-0.05, 0.05))),
            lateral_deviation=max(0.0, 0.2 - suite_score * 0.15 + random.uniform(-0.02, 0.02)),
            heading_error=max(0.0, 10.0 - suite_score * 8.0 + random.uniform(-1.0, 1.0)),
            smoothness=max(0.0, 0.3 - suite_score * 0.2 + random.uniform(-0.03, 0.03))
        )
        
        suite_results[suite] = MockSuiteResults(
            success_rate=metrics.success_rate,
            mean_reward=metrics.mean_reward,
            lateral_deviation=metrics.lateral_deviation,
            heading_error=metrics.heading_error,
            smoothness=metrics.smoothness
        )
        suite_results[suite].metrics = metrics
    
    return MockEvaluationResults(
        model_id=model_id,
        global_score=score,
        global_score_ci=(score - 0.02, score + 0.02),
        suite_results=suite_results,
        metadata={'mock': True}
    )


def create_mock_robustness_analysis(parameter_ranges: Dict[str, List[float]]) -> 'RobustnessAnalysis':
    """Create mock robustness analysis results."""
    import random
    from dataclasses import dataclass
    from typing import Dict, List, Tuple
    
    @dataclass
    class MockParameterAnalysis:
        auc_robustness: float
        sensitivity_threshold: float
        operating_range: Tuple[float, float]
        success_rates: List[float]
        parameter_values: List[float]
    
    @dataclass
    class MockRobustnessAnalysis:
        parameter_analyses: Dict[str, MockParameterAnalysis]
        overall_robustness: float
    
    parameter_analyses = {}
    overall_robustness_scores = []
    
    for param_name, param_values in parameter_ranges.items():
        # Generate mock success rates that decrease with parameter deviation
        baseline_idx = len(param_values) // 2
        baseline_value = param_values[baseline_idx]
        
        success_rates = []
        for i, value in enumerate(param_values):
            deviation = abs(value - baseline_value) / baseline_value if baseline_value != 0 else abs(value)
            base_success = 0.9
            degradation = min(0.4, deviation * 0.5 + random.uniform(-0.1, 0.1))
            success_rate = max(0.3, base_success - degradation)
            success_rates.append(success_rate)
        
        # Calculate AUC robustness (simplified)
        normalized_values = [(v - min(param_values)) / (max(param_values) - min(param_values)) 
                           for v in param_values]
        auc = sum(success_rates) / len(success_rates)
        
        # Find sensitivity threshold (where success rate drops below 0.8)
        sensitivity_threshold = baseline_value
        for i, sr in enumerate(success_rates):
            if sr < 0.8:
                sensitivity_threshold = param_values[i]
                break
        
        # Operating range (where success rate > 0.75)
        operating_range = (min(param_values), max(param_values))
        for i, sr in enumerate(success_rates):
            if sr > 0.75:
                operating_range = (param_values[i], operating_range[1])
                break
        for i in reversed(range(len(success_rates))):
            if success_rates[i] > 0.75:
                operating_range = (operating_range[0], param_values[i])
                break
        
        parameter_analyses[param_name] = MockParameterAnalysis(
            auc_robustness=auc,
            sensitivity_threshold=sensitivity_threshold,
            operating_range=operating_range,
            success_rates=success_rates,
            parameter_values=param_values
        )
        
        overall_robustness_scores.append(auc)
    
    return MockRobustnessAnalysis(
        parameter_analyses=parameter_analyses,
        overall_robustness=sum(overall_robustness_scores) / len(overall_robustness_scores)
    )


def create_mock_evaluation_results_with_custom_metrics(model_id: str) -> 'ModelEvaluationResults':
    """Create mock evaluation results with custom metrics."""
    results = create_mock_evaluation_results(model_id)
    
    # Add custom metrics
    results.custom_metrics = {
        'lane_change_success_rate': 0.85,
        'object_avoidance_efficiency': 0.92,
        'energy_efficiency': 0.78,
        'comfort_score': 0.88
    }
    
    return results


def evaluate_deployment_readiness(results: 'ModelEvaluationResults', 
                                criteria: Dict[str, float]) -> Dict[str, Any]:
    """Evaluate if a model meets deployment readiness criteria."""
    criteria_results = {}
    issues = []
    
    # Check global score
    if results.global_score >= criteria['min_global_score']:
        criteria_results['global_score'] = {'passed': True, 'value': results.global_score}
    else:
        criteria_results['global_score'] = {'passed': False, 'value': results.global_score}
        issues.append(f"Global score {results.global_score:.3f} below minimum {criteria['min_global_score']:.3f}")
    
    # Check success rate
    base_success_rate = results.suite_results['base'].metrics.success_rate
    if base_success_rate >= criteria['min_success_rate']:
        criteria_results['success_rate'] = {'passed': True, 'value': base_success_rate}
    else:
        criteria_results['success_rate'] = {'passed': False, 'value': base_success_rate}
        issues.append(f"Success rate {base_success_rate:.3f} below minimum {criteria['min_success_rate']:.3f}")
    
    # Check lateral deviation
    lateral_deviation = results.suite_results['base'].metrics.lateral_deviation
    if lateral_deviation <= criteria['max_lateral_deviation']:
        criteria_results['lateral_deviation'] = {'passed': True, 'value': lateral_deviation}
    else:
        criteria_results['lateral_deviation'] = {'passed': False, 'value': lateral_deviation}
        issues.append(f"Lateral deviation {lateral_deviation:.3f} above maximum {criteria['max_lateral_deviation']:.3f}")
    
    # Overall readiness
    all_passed = all(result['passed'] for result in criteria_results.values())
    
    return {
        'ready': all_passed,
        'criteria_results': criteria_results,
        'issues': issues
    }


def get_current_best_score() -> float:
    """Get current best model score (mock implementation)."""
    return 0.75  # Mock current best score


def update_champion_model(model_path: str, results: 'ModelEvaluationResults') -> None:
    """Update champion model (mock implementation)."""
    print(f"Updated champion model: {model_path} with score {results.global_score:.3f}")


def generate_training_progress_report(checkpoint_results: List[Dict], output_dir: str) -> str:
    """Generate training progress report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'training_progress.json')
    
    with open(report_path, 'w') as f:
        json.dump(checkpoint_results, f, indent=2, default=str)
    
    return report_path


def generate_custom_metrics_report(results: 'ModelEvaluationResults', output_dir: str) -> str:
    """Generate custom metrics report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'custom_metrics_report.json')
    
    report_data = {
        'model_id': results.model_id,
        'global_score': results.global_score,
        'custom_metrics': results.custom_metrics if hasattr(results, 'custom_metrics') else {}
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report_path


# Custom metric calculation functions (mock implementations)

def calculate_lane_change_success_rate(episode_data) -> float:
    """Calculate lane change success rate."""
    return 0.85  # Mock implementation


def calculate_object_avoidance_efficiency(episode_data) -> float:
    """Calculate object avoidance efficiency."""
    return 0.92  # Mock implementation


def calculate_energy_efficiency(episode_data) -> float:
    """Calculate energy efficiency metric."""
    return 0.78  # Mock implementation


def calculate_comfort_score(episode_data) -> float:
    """Calculate passenger comfort score."""
    return 0.88  # Mock implementation


def main():
    """Run all evaluation examples."""
    logger = setup_logging()
    logger.info("Starting Evaluation Examples")
    
    try:
        # Run all examples
        logger.info("=" * 60)
        example_1_basic_model_evaluation()
        
        logger.info("=" * 60)
        example_2_multi_model_comparison()
        
        logger.info("=" * 60)
        example_3_champion_selection()
        
        logger.info("=" * 60)
        example_4_robustness_analysis()
        
        logger.info("=" * 60)
        example_5_deployment_readiness()
        
        logger.info("=" * 60)
        example_6_continuous_evaluation()
        
        logger.info("=" * 60)
        example_7_custom_metrics_evaluation()
        
        logger.info("=" * 60)
        logger.info("All evaluation examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()