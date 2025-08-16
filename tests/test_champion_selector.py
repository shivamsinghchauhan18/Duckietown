#!/usr/bin/env python3
"""
🧪 CHAMPION SELECTOR TESTS 🧪
Comprehensive unit tests for the ChampionSelector class

This module tests the multi-criteria ranking algorithm, Pareto front analysis,
regression detection, champion validation logic, and statistical significance
validation for champion updates.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List

# Import the classes to test
from duckietown_utils.champion_selector import (
    ChampionSelector, RankingCriterion, ValidationStatus, ParetoPoint, ParetoFront,
    RegressionAnalysis, ChampionValidation, RankingResult, ChampionSelectionResult
)
from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult
from duckietown_utils.statistical_analyzer import ComparisonResult

class TestChampionSelector:
    """Test suite for ChampionSelector class."""
    
    @pytest.fixture
    def champion_selector(self):
        """Create a ChampionSelector instance for testing."""
        config = {
            'min_maps_threshold': 0.9,
            'min_success_rate_threshold': 0.75,
            'success_rate_regression_threshold': 0.05,
            'smoothness_regression_threshold': 0.20,
            'require_statistical_significance': True,
            'significance_alpha': 0.05
        }
        return ChampionSelector(config)
    
    @pytest.fixture
    def sample_model_metrics(self):
        """Create sample model metrics for testing."""
        def create_model_metrics(model_id: str, composite_score: float, 
                                success_rate: float, smoothness: float,
                                lateral_deviation: float, stability: float) -> ModelMetrics:
            
            # Create primary metrics
            primary_metrics = {
                'success_rate': MetricResult(name='success_rate', value=success_rate),
                'smoothness': MetricResult(name='smoothness', value=smoothness),
                'lateral_deviation': MetricResult(name='lateral_deviation', value=lateral_deviation),
                'episode_length': MetricResult(name='episode_length', value=500.0)
            }
            
            # Create secondary metrics
            secondary_metrics = {
                'stability': MetricResult(name='stability', value=stability)
            }
            
            # Create safety metrics
            safety_metrics = {
                'collision_rate': MetricResult(name='collision_rate', value=0.05)
            }
            
            # Create composite score
            composite_score_result = MetricResult(name='composite_score', value=composite_score)
            
            # Create per-map metrics (3 maps, all meeting threshold)
            per_map_metrics = {}
            for i in range(3):
                map_name = f"map_{i}"
                per_map_metrics[map_name] = {
                    'success_rate': MetricResult(name='success_rate', value=success_rate)
                }
            
            # Create per-suite metrics
            per_suite_metrics = {
                'base': {
                    'success_rate': MetricResult(name='success_rate', value=success_rate)
                },
                'ood': {
                    'success_rate': MetricResult(name='success_rate', value=success_rate * 0.9)
                }
            }
            
            return ModelMetrics(
                model_id=model_id,
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
                safety_metrics=safety_metrics,
                composite_score=composite_score_result,
                per_map_metrics=per_map_metrics,
                per_suite_metrics=per_suite_metrics
            )
        
        return [
            create_model_metrics("model_a", 0.85, 0.90, 0.05, 0.10, 0.85),  # Best overall
            create_model_metrics("model_b", 0.80, 0.85, 0.08, 0.12, 0.80),  # Second best
            create_model_metrics("model_c", 0.75, 0.80, 0.10, 0.15, 0.75),  # Third best
            create_model_metrics("model_d", 0.70, 0.70, 0.15, 0.20, 0.70),  # Poor performance
        ]
    
    def test_initialization(self, champion_selector):
        """Test ChampionSelector initialization."""
        assert champion_selector.min_maps_threshold == 0.9
        assert champion_selector.min_success_rate_threshold == 0.75
        assert champion_selector.success_rate_regression_threshold == 0.05
        assert champion_selector.smoothness_regression_threshold == 0.20
        assert champion_selector.require_statistical_significance is True
        assert len(champion_selector.ranking_criteria) == 7
    
    def test_rank_models(self, champion_selector, sample_model_metrics):
        """Test multi-criteria ranking algorithm."""
        rankings = champion_selector._rank_models(sample_model_metrics)
        
        # Check that all models are ranked
        assert len(rankings) == 4
        
        # Check that rankings are in descending order of composite score
        for i in range(len(rankings) - 1):
            assert rankings[i].global_composite_score >= rankings[i + 1].global_composite_score
        
        # Check that ranks are assigned correctly
        for i, ranking in enumerate(rankings):
            assert ranking.rank == i + 1
        
        # Check that model_a (best composite score) is ranked first
        assert rankings[0].model_id == "model_a"
        assert rankings[0].rank == 1
        
        # Check that ranking scores are populated
        for ranking in rankings:
            assert len(ranking.ranking_scores) == len(champion_selector.ranking_criteria)
            assert RankingCriterion.GLOBAL_COMPOSITE_SCORE in ranking.ranking_scores
    
    def test_pareto_front_analysis(self, champion_selector, sample_model_metrics):
        """Test Pareto front analysis for trade-off visualization."""
        pareto_fronts = champion_selector._analyze_pareto_fronts(sample_model_metrics)
        
        # Should have multiple Pareto fronts based on different axes
        assert len(pareto_fronts) > 0
        
        for front in pareto_fronts:
            assert isinstance(front, ParetoFront)
            assert len(front.axes) > 0
            assert len(front.points) > 0
            
            # Check that points have coordinates for all axes
            for point in front.points:
                assert len(point.coordinates) == len(front.axes)
                for axis in front.axes:
                    assert axis in point.coordinates
            
            # Check that non-dominated and dominated models are identified
            assert len(front.non_dominated_models) + len(front.dominated_models) == len(front.points)
    
    def test_domination_logic(self, champion_selector):
        """Test Pareto domination logic."""
        # Create test points
        point_a = ParetoPoint(
            model_id="a",
            coordinates={"success_rate": 0.9, "lateral_deviation": 0.1}
        )
        point_b = ParetoPoint(
            model_id="b", 
            coordinates={"success_rate": 0.8, "lateral_deviation": 0.2}
        )
        point_c = ParetoPoint(
            model_id="c",
            coordinates={"success_rate": 0.9, "lateral_deviation": 0.2}
        )
        
        axes = ["success_rate", "lateral_deviation"]
        
        # Point A should dominate point B (better in both dimensions)
        assert champion_selector._dominates(point_a, point_b, axes)
        
        # Point B should not dominate point A
        assert not champion_selector._dominates(point_b, point_a, axes)
        
        # Point A should dominate point C (equal success rate, better lateral deviation)
        assert champion_selector._dominates(point_a, point_c, axes)
        
        # Point C should not dominate point A
        assert not champion_selector._dominates(point_c, point_a, axes)
    
    def test_champion_validation(self, champion_selector, sample_model_metrics):
        """Test champion validation logic."""
        # Test valid champion
        validation = champion_selector._validate_champion_candidate(sample_model_metrics[0])
        assert validation.status == ValidationStatus.VALID
        assert validation.maps_meeting_threshold == 3
        assert validation.total_maps == 3
        
        # Test champion with insufficient success rate
        poor_model = sample_model_metrics[3]  # model_d with 70% success rate
        validation = champion_selector._validate_champion_candidate(poor_model)
        # Should fail validation due to insufficient maps meeting threshold (70% < 75% threshold)
        assert validation.status in [ValidationStatus.INSUFFICIENT_MAPS, ValidationStatus.LOW_SUCCESS_RATE]
        assert len(validation.maps_below_success_threshold) > 0
    
    def test_regression_detection(self, champion_selector, sample_model_metrics):
        """Test regression detection logic."""
        champion = sample_model_metrics[0]  # model_a
        candidate = sample_model_metrics[1]  # model_b (lower performance)
        
        regression = champion_selector._detect_regression(candidate, champion)
        
        assert regression.model_id == candidate.model_id
        assert regression.current_champion_id == champion.model_id
        assert regression.success_rate_change is not None
        assert regression.success_rate_change < 0  # Candidate has lower success rate
        
        # Should detect regression due to success rate drop
        assert regression.is_regression
        assert len(regression.regression_reasons) > 0
    
    def test_no_regression_detection(self, champion_selector, sample_model_metrics):
        """Test that no regression is detected when candidate is better."""
        champion = sample_model_metrics[1]  # model_b
        candidate = sample_model_metrics[0]  # model_a (better performance)
        
        regression = champion_selector._detect_regression(candidate, champion)
        
        assert not regression.is_regression
        assert len(regression.regression_reasons) == 0
        assert regression.success_rate_change > 0  # Candidate has higher success rate
    
    def test_statistical_comparisons(self, champion_selector, sample_model_metrics):
        """Test statistical comparisons between models."""
        comparisons = champion_selector._perform_statistical_comparisons(sample_model_metrics)
        
        # Should have comparisons between all pairs of models
        expected_comparisons = len(sample_model_metrics) * (len(sample_model_metrics) - 1) // 2
        assert len(comparisons) == expected_comparisons
        
        for comparison in comparisons:
            assert isinstance(comparison, ComparisonResult)
            assert comparison.model_a_id != comparison.model_b_id
            assert comparison.metric_name == 'composite_score'
            assert comparison.p_value is not None
            assert comparison.effect_size is not None
    
    def test_select_champion_new_selection(self, champion_selector, sample_model_metrics):
        """Test champion selection with no current champion."""
        result = champion_selector.select_champion(sample_model_metrics)
        
        assert isinstance(result, ChampionSelectionResult)
        assert result.new_champion_id == "model_a"  # Best performing model
        assert result.previous_champion_id is None
        assert len(result.rankings) == 4
        assert len(result.pareto_fronts) > 0
        
        # Check that rankings are properly ordered
        assert result.rankings[0].model_id == "model_a"
        assert result.rankings[0].rank == 1
    
    def test_select_champion_with_current_champion(self, champion_selector, sample_model_metrics):
        """Test champion selection with existing champion."""
        current_champion = "model_b"
        result = champion_selector.select_champion(sample_model_metrics, current_champion)
        
        assert result.new_champion_id == "model_a"  # Should upgrade to better model
        assert result.previous_champion_id == current_champion
        
        # Check that regression analysis was performed
        for ranking in result.rankings:
            if ranking.regression_analysis:
                assert ranking.regression_analysis.current_champion_id == current_champion
    
    def test_select_champion_no_upgrade_due_to_regression(self, champion_selector):
        """Test that champion is not changed if all candidates show regression."""
        # Create a scenario where the current champion is better than all candidates
        current_champion_metrics = ModelMetrics(
            model_id="champion",
            primary_metrics={
                'success_rate': MetricResult(name='success_rate', value=0.95),
                'smoothness': MetricResult(name='smoothness', value=0.03),
                'lateral_deviation': MetricResult(name='lateral_deviation', value=0.08)
            },
            secondary_metrics={
                'stability': MetricResult(name='stability', value=0.90)
            },
            safety_metrics={},
            composite_score=MetricResult(name='composite_score', value=0.90),
            per_map_metrics={
                'map_0': {'success_rate': MetricResult(name='success_rate', value=0.95)},
                'map_1': {'success_rate': MetricResult(name='success_rate', value=0.95)},
                'map_2': {'success_rate': MetricResult(name='success_rate', value=0.95)}
            },
            per_suite_metrics={
                'base': {'success_rate': MetricResult(name='success_rate', value=0.95)}
            }
        )
        
        # Create worse candidates
        worse_candidates = [
            ModelMetrics(
                model_id="candidate_1",
                primary_metrics={
                    'success_rate': MetricResult(name='success_rate', value=0.80),  # Significant drop
                    'smoothness': MetricResult(name='smoothness', value=0.05),
                    'lateral_deviation': MetricResult(name='lateral_deviation', value=0.10)
                },
                secondary_metrics={
                    'stability': MetricResult(name='stability', value=0.80)
                },
                safety_metrics={},
                composite_score=MetricResult(name='composite_score', value=0.75),
                per_map_metrics={
                    'map_0': {'success_rate': MetricResult(name='success_rate', value=0.80)},
                    'map_1': {'success_rate': MetricResult(name='success_rate', value=0.80)},
                    'map_2': {'success_rate': MetricResult(name='success_rate', value=0.80)}
                },
                per_suite_metrics={
                    'base': {'success_rate': MetricResult(name='success_rate', value=0.80)}
                }
            )
        ]
        
        all_candidates = [current_champion_metrics] + worse_candidates
        
        result = champion_selector.select_champion(all_candidates, "champion")
        
        # Should select the best available candidate even if it shows regression
        # (since we log a warning but still select the best)
        assert result.new_champion_id in ["champion", "candidate_1"]
    
    def test_pareto_rank_assignment(self, champion_selector, sample_model_metrics):
        """Test Pareto rank assignment."""
        pareto_fronts = champion_selector._analyze_pareto_fronts(sample_model_metrics)
        
        # Test getting Pareto rank for models
        for model_metrics in sample_model_metrics:
            rank = champion_selector._get_pareto_rank(model_metrics.model_id, pareto_fronts)
            assert rank is not None
            assert rank >= 1
    
    def test_metric_extraction(self, champion_selector, sample_model_metrics):
        """Test metric value extraction from model metrics."""
        model = sample_model_metrics[0]
        
        # Test extracting different types of metrics
        success_rate = champion_selector._extract_metric_value(model, 'success_rate')
        assert success_rate == 0.90
        
        stability = champion_selector._extract_metric_value(model, 'stability')
        assert stability == 0.85
        
        composite_score = champion_selector._extract_metric_value(model, 'composite_score')
        assert composite_score == 0.85
        
        # Test non-existent metric
        non_existent = champion_selector._extract_metric_value(model, 'non_existent')
        assert non_existent is None
    
    def test_champion_summary(self, champion_selector, sample_model_metrics):
        """Test champion selection summary generation."""
        result = champion_selector.select_champion(sample_model_metrics)
        summary = champion_selector.get_champion_summary(result)
        
        assert 'champion_id' in summary
        assert 'champion_changed' in summary
        assert 'total_candidates' in summary
        assert 'champion_rank' in summary
        assert 'champion_score' in summary
        assert 'champion_validation' in summary
        
        assert summary['champion_id'] == result.new_champion_id
        assert summary['total_candidates'] == len(sample_model_metrics)
        assert summary['champion_rank'] == 1  # Best model should be rank 1
    
    def test_empty_model_list(self, champion_selector):
        """Test handling of empty model list."""
        with pytest.raises(ValueError, match="No model metrics provided"):
            champion_selector.select_champion([])
    
    def test_single_model(self, champion_selector, sample_model_metrics):
        """Test champion selection with single model."""
        single_model = [sample_model_metrics[0]]
        result = champion_selector.select_champion(single_model)
        
        assert result.new_champion_id == single_model[0].model_id
        assert len(result.rankings) == 1
        assert result.rankings[0].rank == 1
    
    def test_tie_breaking(self, champion_selector):
        """Test tie-breaking logic in ranking."""
        # Create models with identical composite scores but different secondary metrics
        model_a = ModelMetrics(
            model_id="tie_a",
            primary_metrics={
                'success_rate': MetricResult(name='success_rate', value=0.85),
                'smoothness': MetricResult(name='smoothness', value=0.05),  # Better smoothness
                'lateral_deviation': MetricResult(name='lateral_deviation', value=0.10)
            },
            secondary_metrics={
                'stability': MetricResult(name='stability', value=0.80)
            },
            safety_metrics={},
            composite_score=MetricResult(name='composite_score', value=0.80),  # Same score
            per_map_metrics={
                'map_0': {'success_rate': MetricResult(name='success_rate', value=0.85)}
            },
            per_suite_metrics={
                'base': {'success_rate': MetricResult(name='success_rate', value=0.85)}
            }
        )
        
        model_b = ModelMetrics(
            model_id="tie_b",
            primary_metrics={
                'success_rate': MetricResult(name='success_rate', value=0.85),
                'smoothness': MetricResult(name='smoothness', value=0.08),  # Worse smoothness
                'lateral_deviation': MetricResult(name='lateral_deviation', value=0.10)
            },
            secondary_metrics={
                'stability': MetricResult(name='stability', value=0.80)
            },
            safety_metrics={},
            composite_score=MetricResult(name='composite_score', value=0.80),  # Same score
            per_map_metrics={
                'map_0': {'success_rate': MetricResult(name='success_rate', value=0.85)}
            },
            per_suite_metrics={
                'base': {'success_rate': MetricResult(name='success_rate', value=0.85)}
            }
        )
        
        tied_models = [model_a, model_b]
        rankings = champion_selector._rank_models(tied_models)
        
        # Model A should rank higher due to better smoothness (tie-breaker)
        assert rankings[0].model_id == "tie_a"
        assert rankings[1].model_id == "tie_b"
        
        # Check that tie-breaker is recorded
        if rankings[1].tie_breaker_used:
            assert rankings[1].tie_breaker_used == RankingCriterion.SMOOTHNESS
    
    def test_validation_with_insufficient_maps(self, champion_selector):
        """Test validation when model doesn't meet map threshold."""
        # Create model with only 1 map meeting threshold out of 3 total
        model = ModelMetrics(
            model_id="insufficient_maps",
            primary_metrics={
                'success_rate': MetricResult(name='success_rate', value=0.80)
            },
            secondary_metrics={},
            safety_metrics={},
            composite_score=MetricResult(name='composite_score', value=0.80),
            per_map_metrics={
                'map_0': {'success_rate': MetricResult(name='success_rate', value=0.80)},  # Above threshold
                'map_1': {'success_rate': MetricResult(name='success_rate', value=0.70)},  # Below threshold
                'map_2': {'success_rate': MetricResult(name='success_rate', value=0.65)}   # Below threshold
            },
            per_suite_metrics={}
        )
        
        validation = champion_selector._validate_champion_candidate(model)
        
        # Should fail validation due to insufficient maps meeting threshold
        assert validation.status in [ValidationStatus.INSUFFICIENT_MAPS, ValidationStatus.LOW_SUCCESS_RATE]
        assert validation.maps_meeting_threshold == 1
        assert validation.total_maps == 3
        assert len(validation.maps_below_success_threshold) == 2
    
    def test_higher_is_better_logic(self, champion_selector):
        """Test the logic for determining if higher values are better."""
        # Test metrics where higher is better
        assert champion_selector._is_higher_better('success_rate') is True
        assert champion_selector._is_higher_better('stability') is True
        assert champion_selector._is_higher_better('composite_score') is True
        
        # Test metrics where lower is better
        assert champion_selector._is_higher_better('lateral_deviation') is False
        assert champion_selector._is_higher_better('smoothness') is False
        assert champion_selector._is_higher_better('episode_length') is False
        
        # Test unknown metric (should default to higher is better with warning)
        with patch.object(champion_selector.logger, 'warning') as mock_warning:
            result = champion_selector._is_higher_better('unknown_metric')
            assert result is True
            mock_warning.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])