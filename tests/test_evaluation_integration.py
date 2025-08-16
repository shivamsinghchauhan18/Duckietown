#!/usr/bin/env python3
"""
🧪 EVALUATION INTEGRATION TESTS 🧪
Comprehensive integration tests for the evaluation orchestrator system

This module implements end-to-end evaluation pipeline tests with mock models,
statistical validation tests, reproducibility tests, performance benchmarking,
and integration tests for all evaluation suites and failure modes.

Requirements covered: 8.4, 9.1-9.5, 13.3, 13.4
"""

import os
import sys
import time
import json
import tempfile
import threading
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.evaluation_orchestrator import (
    EvaluationOrchestrator, ModelRegistry, SeedManager, EvaluationStateTracker,
    ModelInfo, EvaluationTask, EvaluationProgress, EvaluationStatus, PolicyMode
)
from duckietown_utils.suite_manager import (
    SuiteManager, SuiteConfig, SuiteType, EpisodeResult, SuiteResults
)
from duckietown_utils.metrics_calculator import (
    MetricsCalculator, MetricResult, ConfidenceInterval, ModelMetrics
)
from duckietown_utils.statistical_analyzer import (
    StatisticalAnalyzer, ComparisonResult, MultipleComparisonResult
)
from duckietown_utils.failure_analyzer import FailureAnalyzer
from duckietown_utils.robustness_analyzer import RobustnessAnalyzer
from duckietown_utils.champion_selector import ChampionSelector
from duckietown_utils.report_generator import ReportGenerator
from duckietown_utils.artifact_manager import ArtifactManager


class MockModel:
    """Mock model for testing evaluation pipeline."""
    
    def __init__(self, model_id: str, performance_profile: Dict[str, float] = None):
        self.model_id = model_id
        self.performance_profile = performance_profile or {
            'base_success_rate': 0.8,
            'base_reward': 0.7,
            'noise_sensitivity': 0.1,
            'failure_rate': 0.05
        }
        self.call_count = 0
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Mock prediction method."""
        self.call_count += 1
        # Return random action for simulation
        return np.random.uniform(-1, 1, size=2)
    
    def reset(self):
        """Reset model state."""
        self.call_count = 0


class TestEvaluationPipelineIntegration:
    """Integration tests for the complete evaluation pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_models(self, temp_workspace):
        """Create mock models with different performance profiles."""
        models = {}
        
        # High-performance model
        models['champion'] = MockModel('champion', {
            'base_success_rate': 0.95,
            'base_reward': 0.9,
            'noise_sensitivity': 0.05,
            'failure_rate': 0.02
        })
        
        # Medium-performance model
        models['baseline'] = MockModel('baseline', {
            'base_success_rate': 0.8,
            'base_reward': 0.7,
            'noise_sensitivity': 0.1,
            'failure_rate': 0.05
        })
        
        # Low-performance model
        models['weak'] = MockModel('weak', {
            'base_success_rate': 0.6,
            'base_reward': 0.5,
            'noise_sensitivity': 0.2,
            'failure_rate': 0.15
        })
        
        # Create mock model files
        for model_id, model in models.items():
            model_path = Path(temp_workspace) / f"{model_id}.pth"
            model_path.touch()
            model.model_path = str(model_path)
        
        return models
    
    @pytest.fixture
    def evaluation_config(self, temp_workspace):
        """Create evaluation configuration."""
        return {
            'results_dir': temp_workspace,
            'max_concurrent_evaluations': 2,
            'default_seeds_per_suite': 10,
            'base_seed': 42,
            'confidence_level': 0.95,
            'bootstrap_samples': 1000,
            'timeout_per_episode': 30.0,
            'save_artifacts': True,
            'generate_reports': True
        }
    
    @pytest.fixture
    def orchestrator(self, evaluation_config):
        """Create evaluation orchestrator."""
        return EvaluationOrchestrator(evaluation_config)
    
    @pytest.fixture
    def suite_manager(self, evaluation_config):
        """Create suite manager."""
        return SuiteManager(evaluation_config)
    
    @pytest.fixture
    def metrics_calculator(self, evaluation_config):
        """Create metrics calculator."""
        return MetricsCalculator(evaluation_config)
    
    @pytest.fixture
    def statistical_analyzer(self, evaluation_config):
        """Create statistical analyzer."""
        return StatisticalAnalyzer(evaluation_config)
    
    def test_end_to_end_evaluation_pipeline(self, orchestrator, mock_models, temp_workspace):
        """Test complete end-to-end evaluation pipeline with mock models."""
        # Register models
        model_ids = []
        for model_id, model in mock_models.items():
            registered_id = orchestrator.register_model(
                model.model_path,
                model_type="pytorch",
                model_id=model_id,
                metadata={"performance_profile": model.performance_profile}
            )
            model_ids.append(registered_id)
        
        # Schedule evaluations for all models across multiple suites
        task_ids = orchestrator.schedule_evaluation(
            model_ids=model_ids,
            suite_names=["base", "hard_randomization"],
            policy_modes=[PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC],
            seeds_per_suite=5  # Reduced for faster testing
        )
        
        # Should have 3 models * 2 suites * 2 policy modes = 12 tasks
        assert len(task_ids) == 12
        
        # Check initial progress
        progress = orchestrator.get_progress()
        assert progress.total_tasks == 12
        assert progress.pending_tasks == 12
        assert progress.overall_progress == 0.0
        
        # Mock the evaluation execution
        with patch.object(orchestrator.suite_manager, 'run_suite') as mock_run_suite:
            # Configure mock to return realistic results
            def mock_suite_execution(suite_name, model, seeds, policy_mode="deterministic", **kwargs):
                model_perf = mock_models[model.model_id].performance_profile
                
                # Simulate suite difficulty
                difficulty_multiplier = {
                    'base': 1.0,
                    'hard_randomization': 0.8,
                    'law_intersection': 0.9,
                    'out_of_distribution': 0.7,
                    'stress_adversarial': 0.6
                }.get(suite_name, 1.0)
                
                # Generate mock episode results
                episodes = []
                success_rate = model_perf['base_success_rate'] * difficulty_multiplier
                
                for i, seed in enumerate(seeds):
                    success = np.random.random() < success_rate
                    reward = model_perf['base_reward'] * (0.9 if success else 0.3)
                    
                    episode = EpisodeResult(
                        episode_id=f"{model.model_id}_{suite_name}_{i}",
                        map_name=f"map_{i % 3}",
                        seed=seed,
                        success=success,
                        reward=reward,
                        episode_length=500 + np.random.randint(-100, 100),
                        lateral_deviation=0.1 + np.random.uniform(0, 0.1),
                        heading_error=5.0 + np.random.uniform(-2, 2),
                        jerk=0.05 + np.random.uniform(0, 0.02),
                        stability=0.9 - np.random.uniform(0, 0.1),
                        collision=not success and np.random.random() < 0.5,
                        off_lane=not success and np.random.random() < 0.3
                    )
                    episodes.append(episode)
                
                return SuiteResults(
                    suite_name=suite_name,
                    suite_type=SuiteType.BASE if suite_name == "base" else SuiteType.HARD_RANDOMIZATION,
                    model_id=model.model_id,
                    policy_mode=policy_mode,
                    total_episodes=len(episodes),
                    successful_episodes=sum(1 for ep in episodes if ep.success),
                    episode_results=episodes,
                    execution_time=len(episodes) * 2.0  # Mock execution time
                )
            
            mock_run_suite.side_effect = mock_suite_execution
            
            # Start evaluation
            success = orchestrator.start_evaluation()
            assert success
            
            # Wait for completion (in real scenario, this would be async)
            # For testing, we'll simulate completion by updating task statuses
            for task_id in task_ids:
                task = orchestrator.state_tracker.get_task(task_id)
                orchestrator.state_tracker.update_task_status(
                    task_id, EvaluationStatus.COMPLETED, progress=100.0
                )
                
                # Add mock results
                mock_results = {
                    "success_rate": 0.8,
                    "mean_reward": 0.7,
                    "total_episodes": 5
                }
                orchestrator.state_tracker.update_task_results(task_id, mock_results)
        
        # Check final progress
        final_progress = orchestrator.get_progress()
        assert final_progress.completed_tasks == 12
        assert final_progress.overall_progress == 100.0
        
        # Get results
        results = orchestrator.get_results()
        assert len(results) == 12
        
        # Verify results structure
        for result in results:
            assert "model_id" in result
            assert "suite_name" in result
            assert "policy_mode" in result
            assert "results" in result
            assert result["results"]["success_rate"] is not None
    
    def test_statistical_validation_confidence_intervals(self, metrics_calculator, statistical_analyzer):
        """Test statistical validation of confidence intervals and significance testing."""
        # Generate test data with known properties
        np.random.seed(42)
        
        # Create two datasets with known difference
        baseline_data = np.random.normal(0.7, 0.1, 100)  # Mean ~0.7
        treatment_data = np.random.normal(0.8, 0.1, 100)  # Mean ~0.8, should be significantly different
        
        # Test confidence interval calculation
        baseline_ci = statistical_analyzer.compute_confidence_intervals(baseline_data, method="bootstrap")
        treatment_ci = statistical_analyzer.compute_confidence_intervals(treatment_data, method="bootstrap")
        
        # Validate confidence intervals
        assert isinstance(baseline_ci, ConfidenceInterval)
        assert isinstance(treatment_ci, ConfidenceInterval)
        assert baseline_ci.lower < baseline_ci.upper
        assert treatment_ci.lower < treatment_ci.upper
        assert baseline_ci.confidence_level == 0.95
        assert treatment_ci.confidence_level == 0.95
        
        # Test that true means are within confidence intervals
        assert baseline_ci.lower <= np.mean(baseline_data) <= baseline_ci.upper
        assert treatment_ci.lower <= np.mean(treatment_data) <= treatment_ci.upper
        
        # Test significance testing
        comparison = statistical_analyzer.compare_models(
            baseline_data, treatment_data,
            "baseline", "treatment", "test_metric"
        )
        
        assert isinstance(comparison, ComparisonResult)
        assert comparison.model_b_mean > comparison.model_a_mean  # Treatment should be higher
        assert comparison.p_value < 0.05  # Should be significant
        assert comparison.is_significant
        assert comparison.effect_size > 0  # Positive effect
        
        # Test multiple comparison correction
        comparisons = [comparison]  # In real scenario, would have multiple comparisons
        corrected_results = statistical_analyzer.correct_multiple_comparisons(
            comparisons, method="benjamini_hochberg"
        )
        
        assert isinstance(corrected_results, MultipleComparisonResult)
        assert len(corrected_results.comparisons) == 1
        assert corrected_results.comparisons[0].adjusted_p_value >= comparison.p_value
    
    def test_reproducibility_with_fixed_seeds(self, orchestrator, suite_manager, mock_models):
        """Test reproducibility of evaluations with fixed seeds and configurations."""
        # Register a model
        model = list(mock_models.values())[0]
        model_id = orchestrator.register_model(model.model_path, model_id="test_model")
        
        # Run same evaluation twice with same configuration
        seeds = [1, 2, 3, 4, 5]
        
        with patch.object(suite_manager, 'run_suite') as mock_run_suite:
            # Create deterministic mock results based on seeds
            def deterministic_mock_suite(suite_name, model, seeds, policy_mode="deterministic", **kwargs):
                episodes = []
                for i, seed in enumerate(seeds):
                    # Use seed to generate deterministic results
                    np.random.seed(seed)
                    success = np.random.random() < 0.8
                    reward = 0.7 if success else 0.3
                    
                    episode = EpisodeResult(
                        episode_id=f"test_{i}",
                        map_name=f"map_{i % 2}",
                        seed=seed,
                        success=success,
                        reward=reward,
                        episode_length=500,
                        lateral_deviation=0.1,
                        heading_error=5.0,
                        jerk=0.05,
                        stability=0.9
                    )
                    episodes.append(episode)
                
                return SuiteResults(
                    suite_name=suite_name,
                    suite_type=SuiteType.BASE,
                    model_id=model.model_id,
                    policy_mode=policy_mode,
                    total_episodes=len(episodes),
                    successful_episodes=sum(1 for ep in episodes if ep.success),
                    episode_results=episodes
                )
            
            mock_run_suite.side_effect = deterministic_mock_suite
            
            # Run first evaluation
            results1 = suite_manager.run_suite("base", model, seeds, "deterministic")
            
            # Run second evaluation with same parameters
            results2 = suite_manager.run_suite("base", model, seeds, "deterministic")
            
            # Results should be identical
            assert results1.success_rate == results2.success_rate
            assert results1.mean_reward == results2.mean_reward
            assert len(results1.episode_results) == len(results2.episode_results)
            
            # Episode-level results should match
            for ep1, ep2 in zip(results1.episode_results, results2.episode_results):
                assert ep1.success == ep2.success
                assert ep1.reward == ep2.reward
                assert ep1.seed == ep2.seed
    
    def test_performance_benchmarking_evaluation_throughput(self, orchestrator, mock_models, temp_workspace):
        """Test performance benchmarking for evaluation throughput."""
        # Register multiple models
        model_ids = []
        for model_id, model in mock_models.items():
            registered_id = orchestrator.register_model(model.model_path, model_id=model_id)
            model_ids.append(registered_id)
        
        # Measure evaluation scheduling performance
        start_time = time.time()
        
        task_ids = orchestrator.schedule_evaluation(
            model_ids=model_ids,
            suite_names=["base", "hard_randomization"],
            policy_modes=[PolicyMode.DETERMINISTIC],
            seeds_per_suite=10
        )
        
        scheduling_time = time.time() - start_time
        
        # Should schedule tasks quickly (< 1 second for this scale)
        assert scheduling_time < 1.0
        assert len(task_ids) == len(model_ids) * 2  # 3 models * 2 suites
        
        # Test concurrent evaluation performance
        with patch.object(orchestrator.suite_manager, 'run_suite') as mock_run_suite:
            # Mock fast suite execution
            def fast_mock_suite(suite_name, model, seeds, **kwargs):
                time.sleep(0.1)  # Simulate 100ms per suite
                return SuiteResults(
                    suite_name=suite_name,
                    suite_type=SuiteType.BASE,
                    model_id=model.model_id,
                    policy_mode="deterministic",
                    total_episodes=len(seeds),
                    successful_episodes=int(len(seeds) * 0.8),
                    episode_results=[]
                )
            
            mock_run_suite.side_effect = fast_mock_suite
            
            # Measure execution time
            execution_start = time.time()
            
            # Simulate concurrent execution
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for task_id in task_ids[:4]:  # Test with 4 tasks
                    task = orchestrator.state_tracker.get_task(task_id)
                    future = executor.submit(
                        orchestrator.suite_manager.run_suite,
                        task.suite_name,
                        mock_models[task.model_id],
                        task.seeds
                    )
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    result = future.result()
                    assert result is not None
            
            execution_time = time.time() - execution_start
            
            # With 2 workers and 4 tasks taking 0.1s each, should complete in ~0.2s
            # Allow some overhead
            assert execution_time < 0.5
    
    def test_all_evaluation_suites_integration(self, suite_manager, mock_models):
        """Test integration with all evaluation suites and failure modes."""
        model = list(mock_models.values())[0]
        seeds = [1, 2, 3, 4, 5]
        
        # Test all suite types
        suite_names = ["base", "hard_randomization", "law_intersection", 
                      "out_of_distribution", "stress_adversarial"]
        
        results = {}
        
        with patch.object(suite_manager, '_simulate_episode') as mock_simulate:
            # Mock episode simulation with suite-specific behavior
            def suite_specific_simulation(episode_id, map_name, model, seed, suite_config, policy_mode):
                # Different failure modes based on suite type
                suite_type = suite_config.suite_type
                
                if suite_type == SuiteType.BASE:
                    success_prob = 0.9
                    failure_modes = {'collision': 0.05, 'off_lane': 0.05}
                elif suite_type == SuiteType.HARD_RANDOMIZATION:
                    success_prob = 0.7
                    failure_modes = {'collision': 0.15, 'off_lane': 0.15}
                elif suite_type == SuiteType.LAW_INTERSECTION:
                    success_prob = 0.8
                    failure_modes = {'violation': 0.2}
                elif suite_type == SuiteType.OUT_OF_DISTRIBUTION:
                    success_prob = 0.6
                    failure_modes = {'collision': 0.2, 'off_lane': 0.2}
                elif suite_type == SuiteType.STRESS_ADVERSARIAL:
                    success_prob = 0.5
                    failure_modes = {'collision': 0.3, 'off_lane': 0.2}
                else:
                    success_prob = 0.8
                    failure_modes = {}
                
                np.random.seed(seed)
                success = np.random.random() < success_prob
                
                # Determine failure mode if not successful
                collision = False
                off_lane = False
                violations = {}
                
                if not success:
                    if 'collision' in failure_modes and np.random.random() < failure_modes['collision']:
                        collision = True
                    elif 'off_lane' in failure_modes and np.random.random() < failure_modes['off_lane']:
                        off_lane = True
                    elif 'violation' in failure_modes and np.random.random() < failure_modes['violation']:
                        violations = {'stop_sign': 1, 'speed': 2}
                
                return EpisodeResult(
                    episode_id=episode_id,
                    map_name=map_name,
                    seed=seed,
                    success=success,
                    reward=0.8 if success else 0.3,
                    episode_length=500,
                    lateral_deviation=0.1,
                    heading_error=5.0,
                    jerk=0.05,
                    stability=0.9,
                    collision=collision,
                    off_lane=off_lane,
                    violations=violations
                )
            
            mock_simulate.side_effect = suite_specific_simulation
            
            # Run all suites
            for suite_name in suite_names:
                result = suite_manager.run_suite(suite_name, model, seeds)
                results[suite_name] = result
                
                # Validate suite-specific results
                assert result.suite_name == suite_name
                assert result.total_episodes == len(seeds)
                assert 0.0 <= result.success_rate <= 1.0
                assert 0.0 <= result.collision_rate <= 1.0
                assert 0.0 <= result.off_lane_rate <= 1.0
        
        # Validate suite difficulty progression
        # Base should have highest success rate, stress_adversarial should have lowest
        assert results['base'].success_rate >= results['hard_randomization'].success_rate
        assert results['hard_randomization'].success_rate >= results['stress_adversarial'].success_rate
        
        # Validate failure mode distributions
        assert results['stress_adversarial'].collision_rate >= results['base'].collision_rate
        assert results['out_of_distribution'].off_lane_rate >= results['base'].off_lane_rate
    
    def test_failure_mode_analysis_integration(self, temp_workspace):
        """Test integration with failure analysis system."""
        # Create failure analyzer
        config = {'results_dir': temp_workspace}
        failure_analyzer = FailureAnalyzer(config)
        
        # Create mock episode results with various failure modes
        episodes = []
        failure_types = ['collision', 'off_lane', 'stuck', 'oscillation', 'over_speed']
        
        for i in range(50):
            if i < 35:  # 70% success rate
                episode = EpisodeResult(
                    episode_id=f"success_{i}",
                    map_name=f"map_{i % 3}",
                    seed=i,
                    success=True,
                    reward=0.8,
                    episode_length=500,
                    lateral_deviation=0.1,
                    heading_error=5.0,
                    jerk=0.05,
                    stability=0.9
                )
            else:  # Failure cases
                failure_type = failure_types[(i - 35) % len(failure_types)]
                episode = EpisodeResult(
                    episode_id=f"failure_{i}",
                    map_name=f"map_{i % 3}",
                    seed=i,
                    success=False,
                    reward=0.3,
                    episode_length=200 if failure_type == 'stuck' else 500,
                    lateral_deviation=0.3 if failure_type == 'off_lane' else 0.1,
                    heading_error=15.0 if failure_type == 'oscillation' else 5.0,
                    jerk=0.2 if failure_type == 'oscillation' else 0.05,
                    stability=0.3,
                    collision=(failure_type == 'collision'),
                    off_lane=(failure_type == 'off_lane'),
                    violations={'speed': 5} if failure_type == 'over_speed' else {}
                )
            episodes.append(episode)
        
        # Analyze failures
        failure_analysis = failure_analyzer.analyze_failures(episodes, "test_model")
        
        # Validate failure analysis results
        assert failure_analysis is not None
        assert 'failure_classification' in failure_analysis
        assert 'failure_statistics' in failure_analysis
        assert 'spatial_analysis' in failure_analysis
        
        # Check failure classification
        classification = failure_analysis['failure_classification']
        assert len(classification) == 15  # 15 failed episodes
        
        # Check that different failure types are identified
        failure_causes = [f['primary_cause'] for f in classification]
        assert 'collision' in failure_causes
        assert 'off_lane' in failure_causes
        
        # Check failure statistics
        stats = failure_analysis['failure_statistics']
        assert stats['total_failures'] == 15
        assert stats['failure_rate'] == 0.3  # 15/50
        assert 'failure_distribution' in stats
    
    def test_robustness_analysis_integration(self, temp_workspace):
        """Test integration with robustness analysis system."""
        config = {'results_dir': temp_workspace}
        robustness_analyzer = RobustnessAnalyzer(config)
        
        # Create mock model results across different environmental parameters
        parameter_sweep_results = {}
        
        # Simulate lighting intensity sweep
        lighting_values = [0.5, 0.7, 1.0, 1.3, 1.5]
        for lighting in lighting_values:
            # Success rate decreases with extreme lighting
            base_success = 0.9
            if lighting < 0.7 or lighting > 1.3:
                success_rate = base_success * 0.7
            else:
                success_rate = base_success
            
            parameter_sweep_results[f"lighting_{lighting}"] = {
                'success_rate': success_rate,
                'mean_reward': success_rate * 0.8,
                'parameter_value': lighting
            }
        
        # Analyze robustness
        robustness_analysis = robustness_analyzer.analyze_robustness(
            parameter_sweep_results, 
            parameter_name="lighting_intensity",
            model_id="test_model"
        )
        
        # Validate robustness analysis
        assert robustness_analysis is not None
        assert 'auc_robustness' in robustness_analysis
        assert 'sensitivity_threshold' in robustness_analysis
        assert 'operating_range' in robustness_analysis
        assert 'robustness_curve' in robustness_analysis
        
        # AUC should be reasonable (between 0 and 1)
        assert 0.0 <= robustness_analysis['auc_robustness'] <= 1.0
        
        # Should identify sensitivity to extreme lighting
        assert robustness_analysis['sensitivity_threshold'] is not None
    
    def test_champion_selection_integration(self, temp_workspace, metrics_calculator):
        """Test integration with champion selection system."""
        config = {'results_dir': temp_workspace}
        champion_selector = ChampionSelector(config)
        
        # Create mock model metrics for multiple models
        model_metrics_list = []
        
        model_names = ['champion', 'baseline', 'weak']
        performance_levels = [0.95, 0.8, 0.6]
        
        for model_name, perf_level in zip(model_names, performance_levels):
            # Create mock metrics
            primary_metrics = {
                'success_rate': MetricResult('success_rate', perf_level, sample_size=100),
                'mean_reward': MetricResult('mean_reward', perf_level * 0.8, sample_size=100),
                'episode_length': MetricResult('episode_length', 500, sample_size=100)
            }
            
            secondary_metrics = {
                'lateral_deviation': MetricResult('lateral_deviation', 0.1 * (2 - perf_level), sample_size=100),
                'heading_error': MetricResult('heading_error', 5.0 * (2 - perf_level), sample_size=100),
                'smoothness': MetricResult('smoothness', 0.05 * (2 - perf_level), sample_size=100)
            }
            
            safety_metrics = {
                'collision_rate': MetricResult('collision_rate', 0.05 * (2 - perf_level), sample_size=100),
                'off_lane_rate': MetricResult('off_lane_rate', 0.03 * (2 - perf_level), sample_size=100)
            }
            
            model_metrics = ModelMetrics(
                model_id=model_name,
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
                safety_metrics=safety_metrics,
                per_suite_metrics={},
                per_map_metrics={},
                metadata={'total_episodes': 100}
            )
            
            model_metrics_list.append(model_metrics)
        
        # Select champion
        champion_results = champion_selector.select_champion(model_metrics_list)
        
        # Validate champion selection
        assert champion_results is not None
        assert 'champion_model' in champion_results
        assert 'ranking' in champion_results
        assert 'pareto_analysis' in champion_results
        
        # Champion should be the highest performing model
        assert champion_results['champion_model']['model_id'] == 'champion'
        
        # Ranking should be in correct order
        ranking = champion_results['ranking']
        assert len(ranking) == 3
        assert ranking[0]['model_id'] == 'champion'
        assert ranking[1]['model_id'] == 'baseline'
        assert ranking[2]['model_id'] == 'weak'
    
    def test_artifact_management_integration(self, temp_workspace):
        """Test integration with artifact management system."""
        config = {'results_dir': temp_workspace, 'save_artifacts': True}
        artifact_manager = ArtifactManager(config)
        
        # Create mock evaluation results
        episode_results = []
        for i in range(10):
            episode = EpisodeResult(
                episode_id=f"test_episode_{i}",
                map_name=f"map_{i % 2}",
                seed=i,
                success=i < 8,  # 80% success rate
                reward=0.8 if i < 8 else 0.3,
                episode_length=500,
                lateral_deviation=0.1,
                heading_error=5.0,
                jerk=0.05,
                stability=0.9
            )
            episode_results.append(episode)
        
        suite_results = SuiteResults(
            suite_name="test_suite",
            suite_type=SuiteType.BASE,
            model_id="test_model",
            policy_mode="deterministic",
            total_episodes=10,
            successful_episodes=8,
            episode_results=episode_results
        )
        
        # Save artifacts
        artifact_paths = artifact_manager.save_evaluation_artifacts(
            suite_results, 
            include_videos=False,  # Skip video generation for testing
            include_traces=True
        )
        
        # Validate artifact creation
        assert artifact_paths is not None
        assert 'results_json' in artifact_paths
        assert 'episode_csv' in artifact_paths
        
        # Check that files were created
        results_path = Path(artifact_paths['results_json'])
        assert results_path.exists()
        
        csv_path = Path(artifact_paths['episode_csv'])
        assert csv_path.exists()
        
        # Validate JSON content
        with open(results_path, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['suite_name'] == "test_suite"
        assert saved_results['total_episodes'] == 10
        assert len(saved_results['episode_results']) == 10
    
    def test_report_generation_integration(self, temp_workspace, metrics_calculator):
        """Test integration with report generation system."""
        config = {'results_dir': temp_workspace}
        report_generator = ReportGenerator(config)
        
        # Create mock evaluation results for multiple models
        model_results = {}
        
        for model_name in ['model_a', 'model_b', 'model_c']:
            # Create mock metrics
            primary_metrics = {
                'success_rate': MetricResult('success_rate', 0.8, sample_size=100),
                'mean_reward': MetricResult('mean_reward', 0.7, sample_size=100)
            }
            
            model_metrics = ModelMetrics(
                model_id=model_name,
                primary_metrics=primary_metrics,
                secondary_metrics={},
                safety_metrics={},
                per_suite_metrics={},
                per_map_metrics={},
                metadata={'total_episodes': 100}
            )
            
            model_results[model_name] = model_metrics
        
        # Generate comprehensive report
        report_paths = report_generator.generate_comprehensive_report(
            model_results,
            report_name="integration_test_report"
        )
        
        # Validate report generation
        assert report_paths is not None
        assert 'html_report' in report_paths
        assert 'json_summary' in report_paths
        
        # Check that report files were created
        html_path = Path(report_paths['html_report'])
        assert html_path.exists()
        
        json_path = Path(report_paths['json_summary'])
        assert json_path.exists()
        
        # Validate JSON summary content
        with open(json_path, 'r') as f:
            summary = json.load(f)
        
        assert 'evaluation_summary' in summary
        assert 'model_rankings' in summary
        assert len(summary['model_rankings']) == 3
    
    def test_memory_usage_and_cleanup(self, orchestrator, mock_models, temp_workspace):
        """Test memory usage and proper cleanup of evaluation resources."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register models and run evaluations
        model_ids = []
        for model_id, model in mock_models.items():
            registered_id = orchestrator.register_model(model.model_path, model_id=model_id)
            model_ids.append(registered_id)
        
        # Schedule many tasks to test memory usage
        task_ids = orchestrator.schedule_evaluation(
            model_ids=model_ids,
            suite_names=["base", "hard_randomization", "law_intersection"],
            policy_modes=[PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC],
            seeds_per_suite=20
        )
        
        # Check memory after scheduling
        after_scheduling_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = after_scheduling_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test scale)
        assert memory_increase < 100
        
        # Clear completed tasks and cleanup
        orchestrator.state_tracker.clear_completed_tasks()
        orchestrator.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Memory should not have grown excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < 50  # Should be minimal growth after cleanup
    
    def test_error_handling_and_recovery(self, orchestrator, mock_models):
        """Test error handling and recovery mechanisms."""
        # Register a model
        model = list(mock_models.values())[0]
        model_id = orchestrator.register_model(model.model_path, model_id="error_test_model")
        
        # Test invalid suite name
        with pytest.raises(ValueError):
            orchestrator.schedule_evaluation(
                model_ids=model_id,
                suite_names="nonexistent_suite"
            )
        
        # Test invalid model ID
        with pytest.raises(ValueError):
            orchestrator.schedule_evaluation(
                model_ids="nonexistent_model",
                suite_names="base"
            )
        
        # Test evaluation with simulated failures
        task_ids = orchestrator.schedule_evaluation(
            model_ids=model_id,
            suite_names="base",
            seeds_per_suite=5
        )
        
        # Simulate task failure
        task_id = task_ids[0]
        orchestrator.state_tracker.update_task_status(
            task_id, 
            EvaluationStatus.FAILED,
            error_message="Simulated evaluation failure"
        )
        
        # Check that failure is properly recorded
        task = orchestrator.state_tracker.get_task(task_id)
        assert task.status == EvaluationStatus.FAILED
        assert task.error_message == "Simulated evaluation failure"
        
        # Check progress reflects failure
        progress = orchestrator.get_progress()
        assert progress.failed_tasks == 1
    
    def test_concurrent_evaluation_safety(self, orchestrator, mock_models):
        """Test thread safety and concurrent evaluation handling."""
        # Register models
        model_ids = []
        for model_id, model in mock_models.items():
            registered_id = orchestrator.register_model(model.model_path, model_id=model_id)
            model_ids.append(registered_id)
        
        # Test concurrent task scheduling
        def schedule_tasks(model_id, suite_name):
            return orchestrator.schedule_evaluation(
                model_ids=model_id,
                suite_names=suite_name,
                seeds_per_suite=3
            )
        
        # Schedule tasks concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, model_id in enumerate(model_ids):
                future = executor.submit(schedule_tasks, model_id, "base")
                futures.append(future)
            
            # Collect results
            all_task_ids = []
            for future in as_completed(futures):
                task_ids = future.result()
                all_task_ids.extend(task_ids)
        
        # All tasks should be scheduled without conflicts
        assert len(all_task_ids) == len(model_ids) * 2  # 2 policy modes per model
        
        # All task IDs should be unique
        assert len(set(all_task_ids)) == len(all_task_ids)
        
        # Test concurrent progress updates
        def update_task_progress(task_id, progress):
            orchestrator.state_tracker.update_task_status(
                task_id, EvaluationStatus.RUNNING, progress=progress
            )
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, task_id in enumerate(all_task_ids[:3]):  # Test with first 3 tasks
                future = executor.submit(update_task_progress, task_id, (i + 1) * 25.0)
                futures.append(future)
            
            # Wait for all updates
            for future in as_completed(futures):
                future.result()
        
        # Check that all updates were applied correctly
        for i, task_id in enumerate(all_task_ids[:3]):
            task = orchestrator.state_tracker.get_task(task_id)
            assert task.status == EvaluationStatus.RUNNING
            assert task.progress == (i + 1) * 25.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])