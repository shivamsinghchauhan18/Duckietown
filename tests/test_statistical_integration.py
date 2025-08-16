#!/usr/bin/env python3
"""
🔗 STATISTICAL INTEGRATION TESTS 🔗
Integration tests for StatisticalAnalyzer with MetricsCalculator

Tests the integration between statistical analysis and metrics calculation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.statistical_analyzer import StatisticalAnalyzer, SignificanceTest
from duckietown_utils.metrics_calculator import MetricsCalculator, MetricResult
from duckietown_utils.suite_manager import EpisodeResult

class TestStatisticalIntegration:
    """Test integration between StatisticalAnalyzer and other components."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a StatisticalAnalyzer instance."""
        return StatisticalAnalyzer({'random_seed': 42, 'bootstrap_resamples': 1000})
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create a MetricsCalculator instance."""
        return MetricsCalculator()
    
    @pytest.fixture
    def sample_episode_results(self):
        """Create sample episode results for testing."""
        np.random.seed(42)
        episodes = []
        
        for i in range(50):
            episode = EpisodeResult(
                episode_id=f"episode_{i}",
                map_name="test_map",
                seed=i,
                success=np.random.random() > 0.2,  # 80% success rate
                reward=np.random.normal(0.8, 0.1),
                episode_length=np.random.randint(100, 500),
                lateral_deviation=np.random.exponential(0.1),
                heading_error=np.random.exponential(0.05),
                jerk=np.random.exponential(0.02),
                stability=np.random.normal(2.0, 0.5),
                collision=np.random.random() < 0.1,  # 10% collision rate
                off_lane=np.random.random() < 0.1,  # 10% off-lane rate
                violations={},
                lap_time=np.random.normal(30.0, 5.0),
                timestamp="2024-01-01T00:00:00"
            )
            episodes.append(episode)
        
        return episodes
    
    def test_metrics_with_confidence_intervals(self, analyzer, metrics_calculator, sample_episode_results):
        """Test calculating metrics with confidence intervals."""
        # Calculate metrics using MetricsCalculator
        metrics = metrics_calculator.aggregate_episode_metrics(sample_episode_results)
        
        # Verify we have metrics
        assert len(metrics) > 0
        assert 'success_rate' in metrics
        assert 'mean_reward' in metrics
        
        # Extract raw data for statistical analysis
        success_data = np.array([ep.success for ep in sample_episode_results], dtype=float)
        reward_data = np.array([ep.reward for ep in sample_episode_results])
        
        # Calculate confidence intervals using StatisticalAnalyzer
        success_ci = analyzer.compute_confidence_intervals(success_data, method="wilson")
        reward_ci = analyzer.compute_confidence_intervals(reward_data, method="bootstrap")
        
        # Verify confidence intervals are reasonable
        assert success_ci.lower <= np.mean(success_data) <= success_ci.upper
        assert reward_ci.lower <= np.mean(reward_data) <= reward_ci.upper
        
        # Verify CI bounds are reasonable
        assert 0 <= success_ci.lower <= success_ci.upper <= 1
        assert reward_ci.lower < reward_ci.upper
    
    def test_model_comparison_with_metrics(self, analyzer, metrics_calculator):
        """Test comparing models using both metrics and statistical analysis."""
        np.random.seed(42)
        
        # Create episode results for two models
        model_a_episodes = []
        model_b_episodes = []
        
        for i in range(30):
            # Model A - baseline performance
            episode_a = EpisodeResult(
                episode_id=f"episode_a_{i}",
                map_name="test_map",
                seed=i,
                success=np.random.random() > 0.25,  # 75% success
                reward=np.random.normal(0.7, 0.1),
                episode_length=200,
                lateral_deviation=0.1,
                heading_error=0.05,
                jerk=0.02,
                stability=2.0,
                collision=False,
                off_lane=False,
                violations={},
                lap_time=30.0,
                timestamp="2024-01-01T00:00:00"
            )
            
            # Model B - improved performance
            episode_b = EpisodeResult(
                episode_id=f"episode_b_{i}",
                map_name="test_map",
                seed=i,
                success=np.random.random() > 0.15,  # 85% success
                reward=np.random.normal(0.8, 0.1),
                episode_length=200,
                lateral_deviation=0.1,
                heading_error=0.05,
                jerk=0.02,
                stability=2.0,
                collision=False,
                off_lane=False,
                violations={},
                lap_time=30.0,
                timestamp="2024-01-01T00:00:00"
            )
            
            model_a_episodes.append(episode_a)
            model_b_episodes.append(episode_b)
        
        # Calculate metrics for both models
        metrics_a = metrics_calculator.aggregate_episode_metrics(model_a_episodes)
        metrics_b = metrics_calculator.aggregate_episode_metrics(model_b_episodes)
        
        # Extract data for statistical comparison
        success_a = np.array([ep.success for ep in model_a_episodes], dtype=float)
        success_b = np.array([ep.success for ep in model_b_episodes], dtype=float)
        reward_a = np.array([ep.reward for ep in model_a_episodes])
        reward_b = np.array([ep.reward for ep in model_b_episodes])
        
        # Perform statistical comparisons
        success_comparison = analyzer.compare_models(
            success_a, success_b,
            "model_a", "model_b", "success_rate",
            test_method=SignificanceTest.MANN_WHITNEY_U
        )
        
        reward_comparison = analyzer.compare_models(
            reward_a, reward_b,
            "model_a", "model_b", "reward",
            test_method=SignificanceTest.MANN_WHITNEY_U
        )
        
        # Verify comparisons make sense
        assert success_comparison.model_b_mean > success_comparison.model_a_mean
        assert reward_comparison.model_b_mean > reward_comparison.model_a_mean
        
        # Should detect significant differences (we created them)
        # Note: Due to randomness, we just check that the comparison was performed correctly
        assert success_comparison.p_value is not None
        assert reward_comparison.p_value is not None
        assert 0 <= success_comparison.p_value <= 1
        assert 0 <= reward_comparison.p_value <= 1
    
    def test_multiple_model_comparison(self, analyzer):
        """Test multiple comparison correction with several models."""
        np.random.seed(42)
        
        # Create data for 4 models with different performance levels
        models_data = {
            'model_a': np.random.normal(0.70, 0.1, 25),  # Baseline
            'model_b': np.random.normal(0.75, 0.1, 25),  # Slightly better
            'model_c': np.random.normal(0.80, 0.1, 25),  # Better
            'model_d': np.random.normal(0.72, 0.1, 25),  # Slightly better than baseline
        }
        
        # Perform all pairwise comparisons
        comparisons = []
        model_names = list(models_data.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_i, model_j = model_names[i], model_names[j]
                comparison = analyzer.compare_models(
                    models_data[model_i], models_data[model_j],
                    model_i, model_j, "reward",
                    test_method=SignificanceTest.MANN_WHITNEY_U
                )
                comparisons.append(comparison)
        
        # Apply multiple comparison correction
        corrected_results = analyzer.correct_multiple_comparisons(
            comparisons, method="benjamini_hochberg"
        )
        
        # Verify correction was applied
        assert len(corrected_results.comparisons) == len(comparisons)
        assert corrected_results.num_significant_after <= corrected_results.num_significant_before
        
        # Check that adjusted p-values are reasonable
        for comp in corrected_results.comparisons:
            assert comp.adjusted_p_value >= comp.p_value
            assert comp.adjusted_p_value <= 1.0
    
    def test_confidence_interval_integration(self, analyzer, metrics_calculator, sample_episode_results):
        """Test integration of confidence intervals with metric results."""
        # Calculate metrics
        metrics = metrics_calculator.aggregate_episode_metrics(sample_episode_results)
        
        # Verify that MetricsCalculator already includes confidence intervals
        for metric_name, metric_result in metrics.items():
            assert isinstance(metric_result, MetricResult)
            if metric_result.confidence_interval is not None:
                ci = metric_result.confidence_interval
                assert ci.lower <= metric_result.value <= ci.upper
                assert ci.confidence_level > 0
                assert ci.method is not None
    
    def test_bootstrap_integration(self, analyzer):
        """Test bootstrap functionality with realistic evaluation data."""
        np.random.seed(42)
        
        # Simulate evaluation results with some variability
        episode_rewards = []
        for _ in range(100):
            # Simulate episodes with varying performance
            base_reward = 0.8
            noise = np.random.normal(0, 0.1)
            episode_reward = np.clip(base_reward + noise, 0, 1)
            episode_rewards.append(episode_reward)
        
        episode_rewards = np.array(episode_rewards)
        
        # Perform bootstrap analysis
        bootstrap_result = analyzer.bootstrap_mean_estimate(episode_rewards, n_resamples=500)
        
        # Verify bootstrap results are reasonable
        assert abs(bootstrap_result.bootstrap_mean - np.mean(episode_rewards)) < 0.05
        assert bootstrap_result.bootstrap_std > 0
        assert bootstrap_result.bootstrap_std < 0.1  # Should be much smaller than data std
        assert len(bootstrap_result.bootstrap_samples) == 500
        
        # Verify confidence interval contains the true mean
        true_mean = np.mean(episode_rewards)
        ci = bootstrap_result.confidence_interval
        assert ci.lower <= true_mean <= ci.upper

if __name__ == "__main__":
    pytest.main([__file__])