#!/usr/bin/env python3
"""
🧪 ROBUSTNESS ANALYZER TESTS 🧪
Comprehensive unit tests for the RobustnessAnalyzer class

This module tests all functionality of the RobustnessAnalyzer including parameter sweeps,
AUC calculations, sensitivity threshold detection, operating range recommendations,
and multi-model comparisons.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

# Import the modules to test
from duckietown_utils.robustness_analyzer import (
    RobustnessAnalyzer, ParameterSweepConfig, ParameterType, RobustnessMetric,
    ParameterPoint, RobustnessCurve, RobustnessAnalysisResult, MultiModelRobustnessComparison
)
from duckietown_utils.suite_manager import EpisodeResult
from duckietown_utils.metrics_calculator import ConfidenceInterval

class TestParameterSweepConfig:
    """Test ParameterSweepConfig functionality."""
    
    def test_valid_config_creation(self):
        """Test creating a valid parameter sweep configuration."""
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            num_points=10,
            sweep_method="linear"
        )
        
        assert config.parameter_type == ParameterType.LIGHTING_INTENSITY
        assert config.parameter_name == "lighting_intensity"
        assert config.min_value == 0.5
        assert config.max_value == 2.0
        assert config.num_points == 10
        assert config.sweep_method == "linear"
    
    def test_custom_sweep_validation(self):
        """Test validation for custom sweep method."""
        # Should raise error without custom_values
        with pytest.raises(ValueError, match="Custom sweep method requires custom_values"):
            ParameterSweepConfig(
                parameter_type=ParameterType.LIGHTING_INTENSITY,
                parameter_name="lighting_intensity",
                min_value=0.5,
                max_value=2.0,
                sweep_method="custom"
            )
        
        # Should work with custom_values
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            sweep_method="custom",
            custom_values=[0.5, 1.0, 1.5, 2.0]
        )
        assert config.custom_values == [0.5, 1.0, 1.5, 2.0]
    
    def test_invalid_range_validation(self):
        """Test validation for invalid parameter ranges."""
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            ParameterSweepConfig(
                parameter_type=ParameterType.LIGHTING_INTENSITY,
                parameter_name="lighting_intensity",
                min_value=2.0,
                max_value=1.0
            )
    
    def test_insufficient_points_validation(self):
        """Test validation for insufficient number of points."""
        with pytest.raises(ValueError, match="num_points must be at least 3"):
            ParameterSweepConfig(
                parameter_type=ParameterType.LIGHTING_INTENSITY,
                parameter_name="lighting_intensity",
                min_value=0.5,
                max_value=2.0,
                num_points=2
            )

class TestRobustnessAnalyzer:
    """Test RobustnessAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a RobustnessAnalyzer instance for testing."""
        config = {
            'confidence_level': 0.95,
            'sensitivity_threshold': 0.1,
            'min_operating_performance': 0.75,
            'robustness_weights': {
                'success_rate_auc': 0.5,
                'reward_auc': 0.3,
                'stability_auc': 0.2
            }
        }
        return RobustnessAnalyzer(config)
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episode results for testing."""
        episodes = []
        for i in range(10):
            episode = EpisodeResult(
                episode_id=f"run_{i}",
                map_name="loop_empty",
                seed=i,
                success=i >= 3,  # 70% success rate
                reward=0.8 - 0.05 * i,  # Decreasing reward
                episode_length=1000 + 10 * i,
                lateral_deviation=0.1 + 0.01 * i,
                heading_error=2.0 + 0.2 * i,
                jerk=0.5 + 0.05 * i,
                stability=2.0 - 0.1 * i,
                collision=i < 2,
                off_lane=i == 2,
                violations={},
                lap_time=30.0 + i,
                metadata={
                    "model_id": "test_model",
                    "mode": "deterministic",
                    "suite": "base"
                },
                timestamp="2024-01-01T00:00:00"
            )
            episodes.append(episode)
        return episodes
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.confidence_level == 0.95
        assert analyzer.sensitivity_threshold == 0.1
        assert analyzer.min_operating_performance == 0.75
        assert analyzer.robustness_weights['success_rate_auc'] == 0.5
    
    def test_generate_linear_sweep_values(self, analyzer):
        """Test generating linear parameter sweep values."""
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            num_points=5,
            sweep_method="linear"
        )
        
        values = analyzer.generate_parameter_sweep_values(config)
        expected = [0.5, 0.875, 1.25, 1.625, 2.0]
        
        assert len(values) == 5
        np.testing.assert_array_almost_equal(values, expected)
    
    def test_generate_log_sweep_values(self, analyzer):
        """Test generating logarithmic parameter sweep values."""
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.1,
            max_value=10.0,
            num_points=3,
            sweep_method="log"
        )
        
        values = analyzer.generate_parameter_sweep_values(config)
        expected = [0.1, 1.0, 10.0]
        
        assert len(values) == 3
        np.testing.assert_array_almost_equal(values, expected)
    
    def test_generate_custom_sweep_values(self, analyzer):
        """Test generating custom parameter sweep values."""
        custom_values = [0.5, 0.8, 1.2, 1.8, 2.0]
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            sweep_method="custom",
            custom_values=custom_values
        )
        
        values = analyzer.generate_parameter_sweep_values(config)
        assert values == sorted(custom_values)
    
    def test_invalid_sweep_method(self, analyzer):
        """Test handling of invalid sweep method."""
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            sweep_method="invalid"
        )
        
        with pytest.raises(ValueError, match="Unknown sweep method"):
            analyzer.generate_parameter_sweep_values(config)
    
    def test_log_sweep_negative_values(self, analyzer):
        """Test log sweep with negative values raises error."""
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=-1.0,
            max_value=2.0,
            sweep_method="log"
        )
        
        with pytest.raises(ValueError, match="Log sweep requires positive min_value"):
            analyzer.generate_parameter_sweep_values(config)
    
    def test_analyze_parameter_point(self, analyzer, sample_episodes):
        """Test analyzing a single parameter point."""
        point = analyzer._analyze_parameter_point(1.0, sample_episodes)
        
        assert point.parameter_value == 1.0
        assert point.success_rate == 0.7  # 7 out of 10 successful
        assert point.sample_size == 10
        assert point.success_rate_ci is not None
        assert point.reward_ci is not None
        assert 'collision_rate' in point.metadata
        assert 'off_lane_rate' in point.metadata
    
    def test_analyze_parameter_point_empty(self, analyzer):
        """Test analyzing parameter point with no episodes."""
        point = analyzer._analyze_parameter_point(1.0, [])
        
        assert point.parameter_value == 1.0
        assert point.success_rate == 0.0
        assert point.mean_reward == 0.0
        assert point.stability == 0.0
        assert point.sample_size == 0
    
    def test_calculate_auc_success_rate(self, analyzer):
        """Test AUC calculation for success rate."""
        # Create test points
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.9, mean_reward=0.8, stability=2.0),
            ParameterPoint(parameter_value=1.0, success_rate=0.8, mean_reward=0.7, stability=1.8),
            ParameterPoint(parameter_value=1.5, success_rate=0.6, mean_reward=0.6, stability=1.5),
            ParameterPoint(parameter_value=2.0, success_rate=0.4, mean_reward=0.5, stability=1.2)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0
        )
        
        auc = analyzer._calculate_auc(sweep_points, 'success_rate', config)
        assert auc > 0  # Should be positive
        assert auc <= 1.0  # Should be normalized
    
    def test_calculate_auc_insufficient_points(self, analyzer):
        """Test AUC calculation with insufficient points."""
        sweep_points = [
            ParameterPoint(parameter_value=1.0, success_rate=0.8, mean_reward=0.7, stability=1.8)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0
        )
        
        auc = analyzer._calculate_auc(sweep_points, 'success_rate', config)
        assert auc == 0.0
    
    def test_detect_sensitivity_threshold(self, analyzer):
        """Test sensitivity threshold detection."""
        # Create points with degrading performance
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.9, mean_reward=0.8, stability=2.0),
            ParameterPoint(parameter_value=1.0, success_rate=0.85, mean_reward=0.75, stability=1.8),
            ParameterPoint(parameter_value=1.5, success_rate=0.7, mean_reward=0.6, stability=1.5),  # Below threshold
            ParameterPoint(parameter_value=2.0, success_rate=0.5, mean_reward=0.4, stability=1.0)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            baseline_value=1.0
        )
        
        threshold = analyzer._detect_sensitivity_threshold(sweep_points, config)
        assert threshold == 1.5  # First point below 90% of baseline (0.85)
    
    def test_detect_sensitivity_threshold_no_degradation(self, analyzer):
        """Test sensitivity threshold detection with no degradation."""
        # All points have good performance
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.9, mean_reward=0.8, stability=2.0),
            ParameterPoint(parameter_value=1.0, success_rate=0.88, mean_reward=0.78, stability=1.9),
            ParameterPoint(parameter_value=1.5, success_rate=0.86, mean_reward=0.76, stability=1.8),
            ParameterPoint(parameter_value=2.0, success_rate=0.84, mean_reward=0.74, stability=1.7)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            baseline_value=1.0
        )
        
        threshold = analyzer._detect_sensitivity_threshold(sweep_points, config)
        assert threshold is None
    
    def test_determine_operating_range(self, analyzer):
        """Test operating range determination."""
        # Create points with some below threshold
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.6, mean_reward=0.5, stability=1.5),  # Below threshold
            ParameterPoint(parameter_value=1.0, success_rate=0.8, mean_reward=0.7, stability=1.8),  # Above threshold
            ParameterPoint(parameter_value=1.5, success_rate=0.85, mean_reward=0.75, stability=1.9),  # Above threshold
            ParameterPoint(parameter_value=2.0, success_rate=0.7, mean_reward=0.6, stability=1.6)  # Below threshold
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0
        )
        
        op_range = analyzer._determine_operating_range(sweep_points, config)
        assert op_range == (1.0, 1.5)
    
    def test_determine_operating_range_no_acceptable_points(self, analyzer):
        """Test operating range with no acceptable points."""
        # All points below threshold
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.6, mean_reward=0.5, stability=1.5),
            ParameterPoint(parameter_value=1.0, success_rate=0.65, mean_reward=0.55, stability=1.6),
            ParameterPoint(parameter_value=1.5, success_rate=0.7, mean_reward=0.6, stability=1.7),
            ParameterPoint(parameter_value=2.0, success_rate=0.6, mean_reward=0.5, stability=1.5)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0
        )
        
        op_range = analyzer._determine_operating_range(sweep_points, config)
        assert op_range is None
    
    def test_find_baseline_performance_specified(self, analyzer):
        """Test finding baseline performance with specified value."""
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.8, mean_reward=0.7, stability=1.8),
            ParameterPoint(parameter_value=1.0, success_rate=0.9, mean_reward=0.8, stability=2.0),
            ParameterPoint(parameter_value=1.5, success_rate=0.85, mean_reward=0.75, stability=1.9),
            ParameterPoint(parameter_value=2.0, success_rate=0.7, mean_reward=0.6, stability=1.6)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            baseline_value=1.0
        )
        
        baseline = analyzer._find_baseline_performance(sweep_points, config)
        assert baseline.parameter_value == 1.0
    
    def test_find_baseline_performance_best(self, analyzer):
        """Test finding baseline performance as best performing point."""
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.8, mean_reward=0.7, stability=1.8),
            ParameterPoint(parameter_value=1.0, success_rate=0.9, mean_reward=0.8, stability=2.0),  # Best
            ParameterPoint(parameter_value=1.5, success_rate=0.85, mean_reward=0.75, stability=1.9),
            ParameterPoint(parameter_value=2.0, success_rate=0.7, mean_reward=0.6, stability=1.6)
        ]
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0
        )
        
        baseline = analyzer._find_baseline_performance(sweep_points, config)
        assert baseline.parameter_value == 1.0
        assert baseline.success_rate == 0.9
    
    def test_identify_degradation_points(self, analyzer):
        """Test identifying degradation points."""
        baseline = ParameterPoint(parameter_value=1.0, success_rate=0.9, mean_reward=0.8, stability=2.0)
        
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.85, mean_reward=0.75, stability=1.9),  # OK
            baseline,
            ParameterPoint(parameter_value=1.5, success_rate=0.7, mean_reward=0.6, stability=1.5),  # Degraded
            ParameterPoint(parameter_value=2.0, success_rate=0.6, mean_reward=0.5, stability=1.2)   # Degraded
        ]
        
        degradation_points = analyzer._identify_degradation_points(sweep_points, baseline)
        
        assert len(degradation_points) == 2
        assert degradation_points[0].parameter_value == 1.5
        assert degradation_points[1].parameter_value == 2.0
    
    def test_analyze_parameter_sweep(self, analyzer, sample_episodes):
        """Test complete parameter sweep analysis."""
        # Create parameter results
        parameter_results = {
            0.5: sample_episodes[:5],  # First 5 episodes
            1.0: sample_episodes[2:7],  # Middle 5 episodes
            1.5: sample_episodes[5:],   # Last 5 episodes
        }
        
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=1.5,
            baseline_value=1.0
        )
        
        curve = analyzer.analyze_parameter_sweep("test_model", parameter_results, config)
        
        assert curve.model_id == "test_model"
        assert curve.parameter_name == "lighting_intensity"
        assert len(curve.sweep_points) == 3
        assert curve.auc_success_rate >= 0
        assert curve.auc_reward >= 0
        assert curve.auc_stability >= 0
        assert curve.baseline_performance is not None
    
    def test_calculate_overall_robustness_score(self, analyzer):
        """Test overall robustness score calculation."""
        # Create mock curves
        curves = {
            'lighting': Mock(auc_success_rate=0.8, auc_reward=0.7, auc_stability=0.6),
            'friction': Mock(auc_success_rate=0.9, auc_reward=0.8, auc_stability=0.7)
        }
        
        score = analyzer._calculate_overall_robustness_score(curves)
        
        # Expected: average of (0.5*0.8 + 0.3*0.7 + 0.2*0.6) and (0.5*0.9 + 0.3*0.8 + 0.2*0.7)
        expected_curve1 = 0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.6
        expected_curve2 = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
        expected_overall = (expected_curve1 + expected_curve2) / 2
        
        assert abs(score - expected_overall) < 0.001
    
    def test_calculate_overall_robustness_score_empty(self, analyzer):
        """Test overall robustness score with no curves."""
        score = analyzer._calculate_overall_robustness_score({})
        assert score == 0.0
    
    def test_generate_recommendations(self, analyzer):
        """Test recommendation generation."""
        # Create mock curves with various characteristics
        lighting_curve = Mock()
        lighting_curve.operating_range = (0.8, 1.2)
        lighting_curve.auc_success_rate = 0.6  # Low AUC
        lighting_curve.auc_reward = 0.5
        lighting_curve.auc_stability = 0.4
        lighting_curve.metadata = {'parameter_range': (0.5, 2.0)}
        
        friction_curve = Mock()
        friction_curve.operating_range = None  # No operating range
        friction_curve.auc_success_rate = 0.8
        friction_curve.auc_reward = 0.7
        friction_curve.auc_stability = 0.6
        friction_curve.metadata = {'parameter_range': (0.1, 1.0)}
        
        curves = {
            'lighting': lighting_curve,
            'friction': friction_curve
        }
        
        recommendations = analyzer._generate_recommendations(curves)
        
        assert len(recommendations) > 0
        # Should have recommendations about low AUC and missing operating range
        rec_text = ' '.join(recommendations)
        assert 'Low robustness detected' in rec_text or 'No safe operating range' in rec_text
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_robustness_curve(self, mock_show, mock_savefig, analyzer):
        """Test plotting robustness curve."""
        # Create a mock curve
        sweep_points = [
            ParameterPoint(parameter_value=0.5, success_rate=0.9, mean_reward=0.8, stability=2.0),
            ParameterPoint(parameter_value=1.0, success_rate=0.8, mean_reward=0.7, stability=1.8),
            ParameterPoint(parameter_value=1.5, success_rate=0.6, mean_reward=0.6, stability=1.5)
        ]
        
        curve = RobustnessCurve(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            model_id="test_model",
            sweep_points=sweep_points,
            auc_success_rate=0.7,
            auc_reward=0.6,
            auc_stability=0.5,
            sensitivity_threshold=1.2,
            operating_range=(0.8, 1.2)
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = analyzer.plot_robustness_curve(curve, tmp.name)
            assert fig is not None
    
    def test_export_robustness_results_json(self, analyzer):
        """Test exporting robustness results to JSON."""
        # Create a mock result
        result = RobustnessAnalysisResult(
            model_id="test_model",
            parameter_curves={},
            overall_robustness_score=0.8,
            robustness_ranking=1,
            sensitivity_summary={'lighting': 1.2},
            operating_ranges={'lighting': (0.8, 1.5)},
            recommendations=["Test recommendation"]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            analyzer.export_robustness_results(result, tmp.name, format='json')
            
            # Verify file was created and contains expected data
            with open(tmp.name, 'r') as f:
                data = json.load(f)
                assert data['model_id'] == "test_model"
                assert data['overall_robustness_score'] == 0.8
                assert data['operating_ranges']['lighting'] == [0.8, 1.5]
    
    def test_export_robustness_results_invalid_format(self, analyzer):
        """Test exporting with invalid format."""
        result = RobustnessAnalysisResult(
            model_id="test_model",
            parameter_curves={},
            overall_robustness_score=0.8
        )
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            analyzer.export_robustness_results(result, "test.txt", format='invalid')

class TestMultiModelComparison:
    """Test multi-model robustness comparison functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer for multi-model tests."""
        return RobustnessAnalyzer()
    
    @pytest.fixture
    def sample_model_results(self):
        """Create sample model results for comparison."""
        results = {}
        
        for i, model_id in enumerate(['model_a', 'model_b', 'model_c']):
            # Create mock curves
            curves = {
                'lighting': Mock(
                    auc_success_rate=0.8 - 0.1 * i,
                    auc_reward=0.7 - 0.1 * i,
                    auc_stability=0.6 - 0.1 * i
                )
            }
            
            result = RobustnessAnalysisResult(
                model_id=model_id,
                parameter_curves=curves,
                overall_robustness_score=0.75 - 0.1 * i,
                sensitivity_summary={'lighting': 1.0 + 0.2 * i},
                operating_ranges={'lighting': (0.8 - 0.1 * i, 1.2 + 0.1 * i)}
            )
            results[model_id] = result
        
        return results
    
    def test_compare_model_robustness(self, analyzer, sample_model_results):
        """Test comparing robustness across multiple models."""
        comparison = analyzer.compare_model_robustness(sample_model_results)
        
        assert len(comparison.model_results) == 3
        assert len(comparison.robustness_rankings) == 3
        
        # Check rankings are in descending order
        scores = [score for _, score in comparison.robustness_rankings]
        assert scores == sorted(scores, reverse=True)
        
        # Check best model is ranked first
        best_model, best_score = comparison.robustness_rankings[0]
        assert best_model == 'model_a'
        assert best_score == 0.75
    
    def test_parameter_rankings(self, analyzer, sample_model_results):
        """Test per-parameter rankings in multi-model comparison."""
        comparison = analyzer.compare_model_robustness(sample_model_results)
        
        assert 'lighting' in comparison.parameter_rankings
        lighting_rankings = comparison.parameter_rankings['lighting']
        
        # Should have 3 models ranked
        assert len(lighting_rankings) == 3
        
        # Check rankings are in descending order by AUC
        aucs = [auc for _, auc in lighting_rankings]
        assert aucs == sorted(aucs, reverse=True)
    
    def test_sensitivity_comparison(self, analyzer, sample_model_results):
        """Test sensitivity comparison across models."""
        comparison = analyzer.compare_model_robustness(sample_model_results)
        
        assert 'lighting' in comparison.sensitivity_comparison
        lighting_sensitivity = comparison.sensitivity_comparison['lighting']
        
        assert 'model_a' in lighting_sensitivity
        assert 'model_b' in lighting_sensitivity
        assert 'model_c' in lighting_sensitivity
        
        # Check sensitivity values are as expected
        assert lighting_sensitivity['model_a'] == 1.0
        assert lighting_sensitivity['model_b'] == 1.2
        assert lighting_sensitivity['model_c'] == 1.4
    
    def test_best_operating_ranges(self, analyzer, sample_model_results):
        """Test best operating ranges calculation."""
        comparison = analyzer.compare_model_robustness(sample_model_results)
        
        assert 'lighting' in comparison.best_operating_ranges
        best_range = comparison.best_operating_ranges['lighting']
        
        # Should be the union of all ranges: min of mins, max of maxes
        # model_a: (0.8, 1.2), model_b: (0.7, 1.3), model_c: (0.6, 1.4)
        expected_range = (0.6, 1.4)
        assert abs(best_range[0] - expected_range[0]) < 1e-10
        assert abs(best_range[1] - expected_range[1]) < 1e-10
    
    def test_ranking_updates(self, analyzer, sample_model_results):
        """Test that individual model rankings are updated."""
        comparison = analyzer.compare_model_robustness(sample_model_results)
        
        # Check that rankings were updated in individual results
        assert comparison.model_results['model_a'].robustness_ranking == 1
        assert comparison.model_results['model_b'].robustness_ranking == 2
        assert comparison.model_results['model_c'].robustness_ranking == 3

class TestIntegration:
    """Integration tests for complete robustness analysis workflow."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer for integration tests."""
        return RobustnessAnalyzer({
            'confidence_level': 0.95,
            'sensitivity_threshold': 0.15,
            'min_operating_performance': 0.7
        })
    
    @pytest.fixture
    def complete_episode_data(self):
        """Create complete episode data for integration testing."""
        data = {}
        
        # Create episodes for different parameter values
        param_values = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        
        for param_val in param_values:
            episodes = []
            # Success rate decreases as parameter increases
            success_prob = max(0.1, 1.0 - 0.4 * (param_val - 0.5))
            
            for i in range(20):  # 20 episodes per parameter value
                success = np.random.random() < success_prob
                episode = EpisodeResult(
                    episode_id=f"run_{param_val}_{i}",
                    map_name="loop_empty",
                    seed=i,
                    success=success,
                    reward=0.9 - 0.2 * (param_val - 0.5) + np.random.normal(0, 0.05),
                    episode_length=1000 + int(100 * (param_val - 0.5)) + np.random.randint(-50, 50),
                    lateral_deviation=0.05 + 0.1 * (param_val - 0.5) + np.random.normal(0, 0.01),
                    heading_error=1.0 + 2 * (param_val - 0.5) + np.random.normal(0, 0.5),
                    jerk=0.3 + 0.2 * (param_val - 0.5) + np.random.normal(0, 0.05),
                    stability=2.5 - 0.5 * (param_val - 0.5) + np.random.normal(0, 0.1),
                    collision=not success and np.random.random() < 0.7,
                    off_lane=not success and np.random.random() < 0.3,
                    violations={},
                    lap_time=30.0 + 5 * (param_val - 0.5) + np.random.normal(0, 2),
                    metadata={
                        "model_id": "integration_test_model",
                        "mode": "deterministic",
                        "suite": "base"
                    },
                    timestamp="2024-01-01T00:00:00"
                )
                episodes.append(episode)
            
            data[param_val] = episodes
        
        return data
    
    def test_complete_analysis_workflow(self, analyzer, complete_episode_data):
        """Test complete robustness analysis workflow."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create sweep configuration
        config = ParameterSweepConfig(
            parameter_type=ParameterType.LIGHTING_INTENSITY,
            parameter_name="lighting_intensity",
            min_value=0.5,
            max_value=2.0,
            num_points=7,
            sweep_method="linear",
            baseline_value=1.0
        )
        
        # Analyze parameter sweep
        curve = analyzer.analyze_parameter_sweep(
            "integration_test_model", 
            complete_episode_data, 
            config
        )
        
        # Verify curve properties
        assert curve.model_id == "integration_test_model"
        assert curve.parameter_name == "lighting_intensity"
        assert len(curve.sweep_points) == 7
        assert curve.auc_success_rate > 0
        assert curve.baseline_performance is not None
        
        # Verify degradation is detected (success rate should decrease with parameter)
        success_rates = [point.success_rate for point in curve.sweep_points]
        # Should generally decrease (allowing for some noise)
        assert success_rates[0] > success_rates[-1]
        
        # Verify sensitivity threshold is detected
        assert curve.sensitivity_threshold is not None
        
        # Create full analysis
        parameter_sweep_results = {"lighting_intensity": complete_episode_data}
        sweep_configs = {"lighting_intensity": config}
        
        analysis_result = analyzer.analyze_model_robustness(
            "integration_test_model",
            parameter_sweep_results,
            sweep_configs
        )
        
        # Verify analysis result
        assert analysis_result.model_id == "integration_test_model"
        assert "lighting_intensity" in analysis_result.parameter_curves
        assert analysis_result.overall_robustness_score > 0
        assert len(analysis_result.recommendations) > 0
        
        # Test export functionality
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            analyzer.export_robustness_results(analysis_result, tmp.name, format='json')
            
            # Verify export worked
            with open(tmp.name, 'r') as f:
                exported_data = json.load(f)
                assert exported_data['model_id'] == "integration_test_model"
                assert 'parameter_curves' in exported_data

if __name__ == "__main__":
    pytest.main([__file__])