#!/usr/bin/env python3
"""
🧪 METRICS CALCULATOR TESTS 🧪
Comprehensive unit tests for the MetricsCalculator class

Tests cover metric calculations, composite scoring, normalization,
confidence intervals, and episode-level metric extraction.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import the modules to test
from duckietown_utils.metrics_calculator import (
    MetricsCalculator, MetricDefinition, MetricType, NormalizationScope,
    CompositeScoreConfig, MetricResult, ConfidenceInterval, ModelMetrics
)
from duckietown_utils.suite_manager import EpisodeResult, SuiteResults, SuiteType


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a MetricsCalculator instance for testing."""
        config = {
            'confidence_level': 0.95,
            'bootstrap_samples': 1000,  # Reduced for faster tests
            'composite_weights': {
                'success_rate': 0.45,
                'mean_reward': 0.25,
                'episode_length': 0.10,
                'lateral_deviation': 0.08,
                'heading_error': 0.06,
                'smoothness': 0.06
            }
        }
        return MetricsCalculator(config)
    
    @pytest.fixture
    def sample_episode_results(self) -> List[EpisodeResult]:
        """Create sample episode results for testing."""
        episodes = []
        
        # Create 20 sample episodes with varying performance
        for i in range(20):
            success = i < 15  # 75% success rate
            reward = 0.8 if success else 0.3
            episode_length = 500 + i * 10
            lateral_deviation = 0.1 + (i % 5) * 0.02
            heading_error = 5.0 + (i % 3) * 2.0
            jerk = 0.05 + (i % 4) * 0.01
            stability = 0.9 - (i % 6) * 0.05
            
            episode = EpisodeResult(
                episode_id=f"test_episode_{i}",
                map_name=f"map_{i % 3}",  # 3 different maps
                seed=i,
                success=success,
                reward=reward,
                episode_length=episode_length,
                lateral_deviation=lateral_deviation,
                heading_error=heading_error,
                jerk=jerk,
                stability=stability,
                collision=(not success and i % 2 == 0),
                off_lane=(not success and i % 2 == 1),
                violations={'speed': i % 3, 'stop_sign': i % 4} if i % 5 == 0 else {},
                lap_time=30.0 + i if success else None
            )
            episodes.append(episode)
        
        return episodes
    
    @pytest.fixture
    def sample_suite_results(self, sample_episode_results) -> SuiteResults:
        """Create sample suite results for testing."""
        return SuiteResults(
            suite_name="test_suite",
            suite_type=SuiteType.BASE,
            model_id="test_model",
            policy_mode="deterministic",
            total_episodes=len(sample_episode_results),
            successful_episodes=sum(1 for ep in sample_episode_results if ep.success),
            episode_results=sample_episode_results
        )
    
    def test_initialization(self, calculator):
        """Test MetricsCalculator initialization."""
        assert calculator is not None
        assert len(calculator.metric_definitions) > 0
        assert calculator.confidence_level == 0.95
        assert calculator.bootstrap_samples == 1000
        
        # Check that all expected metrics are defined
        expected_metrics = [
            'success_rate', 'mean_reward', 'episode_length', 'lateral_deviation',
            'heading_error', 'smoothness', 'stability', 'collision_rate',
            'off_lane_rate', 'violation_rate'
        ]
        
        for metric in expected_metrics:
            assert metric in calculator.metric_definitions
    
    def test_metric_definitions(self, calculator):
        """Test metric definitions are properly configured."""
        # Test primary metrics
        success_rate_def = calculator.get_metric_definition('success_rate')
        assert success_rate_def is not None
        assert success_rate_def.metric_type == MetricType.PRIMARY
        assert success_rate_def.higher_is_better is True
        assert success_rate_def.min_value == 0.0
        assert success_rate_def.max_value == 1.0
        
        # Test metric with lower is better
        lateral_dev_def = calculator.get_metric_definition('lateral_deviation')
        assert lateral_dev_def is not None
        assert lateral_dev_def.higher_is_better is False
        
        # Test safety metrics
        collision_def = calculator.get_metric_definition('collision_rate')
        assert collision_def is not None
        assert collision_def.metric_type == MetricType.SAFETY
        assert collision_def.higher_is_better is False
    
    def test_calculate_episode_metrics(self, calculator):
        """Test calculation of metrics for a single episode."""
        episode = EpisodeResult(
            episode_id="test_ep",
            map_name="test_map",
            seed=42,
            success=True,
            reward=0.85,
            episode_length=600,
            lateral_deviation=0.12,
            heading_error=7.5,
            jerk=0.08,
            stability=0.92,
            collision=False,
            off_lane=False,
            violations={'speed': 1, 'stop_sign': 0},
            lap_time=35.2
        )
        
        metrics = calculator.calculate_episode_metrics(episode)
        
        # Check all expected metrics are present
        assert 'success_rate' in metrics
        assert 'mean_reward' in metrics
        assert 'episode_length' in metrics
        assert 'lateral_deviation' in metrics
        assert 'heading_error' in metrics
        assert 'smoothness' in metrics
        assert 'stability' in metrics
        assert 'collision_rate' in metrics
        assert 'off_lane_rate' in metrics
        assert 'violation_rate' in metrics
        assert 'lap_time' in metrics
        
        # Check specific values
        assert metrics['success_rate'] == 1.0
        assert metrics['mean_reward'] == 0.85
        assert metrics['episode_length'] == 600.0
        assert metrics['lateral_deviation'] == 0.12
        assert metrics['collision_rate'] == 0.0
        assert metrics['off_lane_rate'] == 0.0
        assert metrics['violation_rate'] == 0.1  # 1 violation / 10 max
        assert metrics['lap_time'] == 35.2
    
    def test_aggregate_episode_metrics(self, calculator, sample_episode_results):
        """Test aggregation of metrics across multiple episodes."""
        aggregated = calculator.aggregate_episode_metrics(sample_episode_results)
        
        # Check all metrics are present
        assert len(aggregated) > 0
        
        # Check success rate calculation
        success_rate = aggregated['success_rate']
        assert success_rate.name == 'success_rate'
        assert success_rate.value == 0.75  # 15/20 episodes successful
        assert success_rate.sample_size == 20
        assert success_rate.confidence_interval is not None
        
        # Check mean reward
        mean_reward = aggregated['mean_reward']
        expected_reward = np.mean([ep.reward for ep in sample_episode_results])
        assert abs(mean_reward.value - expected_reward) < 1e-6
        
        # Check metadata
        assert 'std' in mean_reward.metadata
        assert 'min' in mean_reward.metadata
        assert 'max' in mean_reward.metadata
        assert 'median' in mean_reward.metadata
    
    def test_calculate_suite_metrics(self, calculator, sample_suite_results):
        """Test calculation of metrics for a complete suite."""
        suite_metrics = calculator.calculate_suite_metrics(sample_suite_results)
        
        assert len(suite_metrics) > 0
        assert 'success_rate' in suite_metrics
        assert 'mean_reward' in suite_metrics
        
        # Should be same as aggregating episode results directly
        direct_aggregated = calculator.aggregate_episode_metrics(sample_suite_results.episode_results)
        
        for metric_name in suite_metrics:
            assert abs(suite_metrics[metric_name].value - direct_aggregated[metric_name].value) < 1e-6
    
    def test_calculate_per_map_metrics(self, calculator, sample_episode_results):
        """Test calculation of per-map metrics."""
        per_map_metrics = calculator.calculate_per_map_metrics(sample_episode_results)
        
        # Should have metrics for 3 maps (map_0, map_1, map_2)
        assert len(per_map_metrics) == 3
        assert 'map_0' in per_map_metrics
        assert 'map_1' in per_map_metrics
        assert 'map_2' in per_map_metrics
        
        # Each map should have all metrics
        for map_name, map_metrics in per_map_metrics.items():
            assert 'success_rate' in map_metrics
            assert 'mean_reward' in map_metrics
            assert map_metrics['success_rate'].sample_size > 0
    
    def test_wilson_confidence_interval(self, calculator):
        """Test Wilson confidence interval calculation for proportions."""
        # Test with known values
        values = [1.0] * 15 + [0.0] * 5  # 75% success rate
        
        ci = calculator._wilson_confidence_interval(values)
        
        assert ci.confidence_level == 0.95
        assert ci.method == "wilson"
        assert 0.0 <= ci.lower <= ci.upper <= 1.0
        assert ci.lower < 0.75 < ci.upper  # Should contain true proportion
    
    def test_bootstrap_confidence_interval(self, calculator):
        """Test bootstrap confidence interval calculation."""
        # Test with normal distribution
        np.random.seed(42)  # For reproducible tests
        values = np.random.normal(0.5, 0.1, 100).tolist()
        
        ci = calculator._bootstrap_confidence_interval(values)
        
        assert ci.confidence_level == 0.95
        assert ci.method == "bootstrap"
        assert ci.lower < np.mean(values) < ci.upper
    
    def test_normalize_value_min_max(self, calculator):
        """Test min-max normalization."""
        definition = MetricDefinition(
            name='test_metric',
            metric_type=MetricType.PRIMARY,
            description='Test metric',
            higher_is_better=True,
            normalization_method='min_max'
        )
        
        norm_stats = {'min': 0.0, 'max': 10.0}
        
        # Test normalization
        assert calculator._normalize_value(0.0, norm_stats, definition) == 0.0
        assert calculator._normalize_value(10.0, norm_stats, definition) == 1.0
        assert calculator._normalize_value(5.0, norm_stats, definition) == 0.5
        
        # Test with lower is better
        definition.higher_is_better = False
        assert calculator._normalize_value(0.0, norm_stats, definition) == 1.0
        assert calculator._normalize_value(10.0, norm_stats, definition) == 0.0
        assert calculator._normalize_value(5.0, norm_stats, definition) == 0.5
    
    def test_normalize_value_z_score(self, calculator):
        """Test z-score normalization."""
        definition = MetricDefinition(
            name='test_metric',
            metric_type=MetricType.PRIMARY,
            description='Test metric',
            higher_is_better=True,
            normalization_method='z_score'
        )
        
        norm_stats = {'mean': 5.0, 'std': 2.0}
        
        # Test normalization (should use sigmoid function)
        normalized = calculator._normalize_value(5.0, norm_stats, definition)
        assert 0.4 < normalized < 0.6  # Should be around 0.5 for mean value
        
        # Higher value should give higher normalized value
        high_normalized = calculator._normalize_value(7.0, norm_stats, definition)
        assert high_normalized > normalized
    
    def test_compute_normalization_stats(self, calculator, sample_episode_results):
        """Test computation of normalization statistics."""
        # Create multiple metric sets
        metrics_list = []
        for i in range(3):
            episodes = sample_episode_results[i*5:(i+1)*5]  # 5 episodes each
            metrics = calculator.aggregate_episode_metrics(episodes)
            metrics_list.append(metrics)
        
        norm_stats = calculator.compute_normalization_stats(metrics_list)
        
        assert len(norm_stats) > 0
        
        # Check that each metric has proper statistics
        for metric_name, stats in norm_stats.items():
            assert 'min' in stats
            assert 'max' in stats
            assert 'mean' in stats
            assert 'std' in stats
            assert 'median' in stats
            assert 'mad' in stats
            assert 'count' in stats
            assert stats['count'] == 3  # 3 metric sets
    
    def test_normalize_metrics(self, calculator, sample_episode_results):
        """Test normalization of metrics."""
        metrics = calculator.aggregate_episode_metrics(sample_episode_results)
        
        # Create normalization stats
        norm_stats = {
            'success_rate': {'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.2},
            'mean_reward': {'min': 0.0, 'max': 1.0, 'mean': 0.6, 'std': 0.15}
        }
        
        normalized = calculator.normalize_metrics(metrics, norm_stats)
        
        # Check that normalized values are present
        assert 'success_rate' in normalized
        assert 'mean_reward' in normalized
        
        success_rate = normalized['success_rate']
        assert success_rate.normalized_value is not None
        assert 0.0 <= success_rate.normalized_value <= 1.0
        
        # Check that normalization stats are in metadata
        assert 'normalization_stats' in success_rate.metadata
    
    def test_calculate_composite_score(self, calculator):
        """Test composite score calculation."""
        # Create mock normalized metrics
        metrics = {
            'success_rate': MetricResult('success_rate', 0.8, normalized_value=0.8, sample_size=20),
            'mean_reward': MetricResult('mean_reward', 0.7, normalized_value=0.7, sample_size=20),
            'episode_length': MetricResult('episode_length', 500, normalized_value=0.6, sample_size=20),
            'lateral_deviation': MetricResult('lateral_deviation', 0.1, normalized_value=0.9, sample_size=20),
            'heading_error': MetricResult('heading_error', 5.0, normalized_value=0.8, sample_size=20),
            'smoothness': MetricResult('smoothness', 0.05, normalized_value=0.85, sample_size=20),
            'collision_rate': MetricResult('collision_rate', 0.1, normalized_value=0.9, sample_size=20)
        }
        
        composite = calculator.calculate_composite_score(metrics)
        
        assert composite.name == 'composite_score'
        assert 0.0 <= composite.value <= 1.0
        assert composite.normalized_value == composite.value
        assert 'components' in composite.metadata
        assert 'total_weight' in composite.metadata
        assert 'safety_penalty' in composite.metadata
        
        # Check that all components are included
        components = composite.metadata['components']
        expected_components = ['success_rate', 'mean_reward', 'episode_length', 
                             'lateral_deviation', 'heading_error', 'smoothness']
        
        for component in expected_components:
            assert component in components
    
    def test_calculate_safety_penalty(self, calculator):
        """Test safety penalty calculation."""
        # Create metrics with safety issues
        metrics = {
            'collision_rate': MetricResult('collision_rate', 0.2, normalized_value=0.8, sample_size=20),
            'off_lane_rate': MetricResult('off_lane_rate', 0.1, normalized_value=0.9, sample_size=20),
            'violation_rate': MetricResult('violation_rate', 0.15, normalized_value=0.85, sample_size=20)
        }
        
        penalty = calculator._calculate_safety_penalty(metrics)
        
        assert 0.0 <= penalty <= 1.0
        # Should be average of normalized safety metric values
        expected_penalty = (0.8 + 0.9 + 0.85) / 3
        assert abs(penalty - expected_penalty) < 1e-6
    
    def test_calculate_model_metrics(self, calculator, sample_episode_results):
        """Test calculation of comprehensive model metrics."""
        # Create multiple suite results
        suite_results_list = []
        for i in range(2):
            suite_results = SuiteResults(
                suite_name=f"suite_{i}",
                suite_type=SuiteType.BASE,
                model_id="test_model",
                policy_mode="deterministic",
                total_episodes=10,
                successful_episodes=8,
                episode_results=sample_episode_results[i*10:(i+1)*10]
            )
            suite_results_list.append(suite_results)
        
        model_metrics = calculator.calculate_model_metrics("test_model", suite_results_list)
        
        assert model_metrics.model_id == "test_model"
        assert len(model_metrics.primary_metrics) > 0
        assert len(model_metrics.secondary_metrics) > 0
        assert len(model_metrics.safety_metrics) > 0
        assert len(model_metrics.per_suite_metrics) == 2
        assert len(model_metrics.per_map_metrics) > 0
        
        # Check metadata
        assert model_metrics.metadata['total_episodes'] == 20
        assert model_metrics.metadata['total_suites'] == 2
        assert len(model_metrics.metadata['suite_names']) == 2
    
    def test_add_composite_score(self, calculator, sample_episode_results):
        """Test adding composite score to model metrics."""
        # Create model metrics
        suite_results = SuiteResults(
            suite_name="test_suite",
            suite_type=SuiteType.BASE,
            model_id="test_model",
            policy_mode="deterministic",
            total_episodes=len(sample_episode_results),
            successful_episodes=15,
            episode_results=sample_episode_results
        )
        
        model_metrics = calculator.calculate_model_metrics("test_model", [suite_results])
        
        # Create normalization stats
        all_metrics = {}
        all_metrics.update(model_metrics.primary_metrics)
        all_metrics.update(model_metrics.secondary_metrics)
        all_metrics.update(model_metrics.safety_metrics)
        
        norm_stats = calculator.compute_normalization_stats([all_metrics])
        
        # Add composite score
        model_metrics_with_composite = calculator.add_composite_score(model_metrics, norm_stats)
        
        assert model_metrics_with_composite.composite_score is not None
        assert model_metrics_with_composite.composite_score.name == 'composite_score'
        assert 0.0 <= model_metrics_with_composite.composite_score.value <= 1.0
    
    def test_list_available_metrics(self, calculator):
        """Test listing of available metrics."""
        metrics_by_type = calculator.list_available_metrics()
        
        assert 'primary' in metrics_by_type
        assert 'secondary' in metrics_by_type
        assert 'safety' in metrics_by_type
        assert 'composite' in metrics_by_type
        
        # Check that expected metrics are in correct categories
        assert 'success_rate' in metrics_by_type['primary']
        assert 'stability' in metrics_by_type['secondary']
        assert 'collision_rate' in metrics_by_type['safety']
        assert 'composite_score' in metrics_by_type['composite']
    
    def test_empty_episode_list(self, calculator):
        """Test handling of empty episode list."""
        empty_metrics = calculator.aggregate_episode_metrics([])
        assert len(empty_metrics) == 0
        
        model_metrics = calculator.calculate_model_metrics("test_model", [])
        assert model_metrics.model_id == "test_model"
        assert len(model_metrics.primary_metrics) == 0
    
    def test_single_episode(self, calculator):
        """Test handling of single episode."""
        episode = EpisodeResult(
            episode_id="single_ep",
            map_name="test_map",
            seed=42,
            success=True,
            reward=0.8,
            episode_length=500,
            lateral_deviation=0.1,
            heading_error=5.0,
            jerk=0.05,
            stability=0.9,
            collision=False,
            off_lane=False
        )
        
        metrics = calculator.aggregate_episode_metrics([episode])
        
        assert len(metrics) > 0
        assert metrics['success_rate'].value == 1.0
        assert metrics['mean_reward'].value == 0.8
        
        # Confidence intervals should handle single value
        assert metrics['success_rate'].confidence_interval is not None
    
    def test_invalid_metric_name(self, calculator):
        """Test handling of invalid metric names."""
        definition = calculator.get_metric_definition('nonexistent_metric')
        assert definition is None
    
    def test_composite_score_config_validation(self):
        """Test composite score configuration validation."""
        # Test invalid weights (don't sum to 1.0)
        with pytest.warns(UserWarning):
            config = CompositeScoreConfig(weights={'metric1': 0.5, 'metric2': 0.3})
    
    def test_metric_definition_validation(self):
        """Test metric definition validation."""
        # Test invalid normalization method
        with pytest.raises(ValueError):
            MetricDefinition(
                name='test',
                metric_type=MetricType.PRIMARY,
                description='Test',
                normalization_method='invalid_method'
            )
    
    def test_edge_cases_normalization(self, calculator):
        """Test edge cases in normalization."""
        definition = MetricDefinition(
            name='test_metric',
            metric_type=MetricType.PRIMARY,
            description='Test metric',
            higher_is_better=True,
            normalization_method='min_max'
        )
        
        # Test case where min == max (no variation)
        norm_stats = {'min': 5.0, 'max': 5.0}
        normalized = calculator._normalize_value(5.0, norm_stats, definition)
        assert normalized == 0.5  # Should default to middle value
        
        # Test z-score with zero std
        definition.normalization_method = 'z_score'
        norm_stats = {'mean': 5.0, 'std': 0.0}
        normalized = calculator._normalize_value(5.0, norm_stats, definition)
        assert normalized == 0.5
        
        # Test robust with zero MAD
        definition.normalization_method = 'robust'
        norm_stats = {'median': 5.0, 'mad': 0.0}
        normalized = calculator._normalize_value(5.0, norm_stats, definition)
        assert normalized == 0.5


if __name__ == '__main__':
    pytest.main([__file__])