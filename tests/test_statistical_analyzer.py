#!/usr/bin/env python3
"""
🧪 STATISTICAL ANALYZER TESTS 🧪
Comprehensive unit tests for the StatisticalAnalyzer class

Tests confidence interval calculations, bootstrap resampling, significance testing,
and multiple comparison correction methods.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.statistical_analyzer import (
    StatisticalAnalyzer, SignificanceTest, EffectSizeMethod,
    ComparisonResult, MultipleComparisonResult, BootstrapResult
)
from duckietown_utils.metrics_calculator import ConfidenceInterval

class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a StatisticalAnalyzer instance for testing."""
        config = {
            'confidence_level': 0.95,
            'bootstrap_resamples': 1000,  # Reduced for faster tests
            'alpha': 0.05,
            'random_seed': 42
        }
        return StatisticalAnalyzer(config)
    
    @pytest.fixture
    def sample_data_normal(self):
        """Generate normal sample data for testing."""
        np.random.seed(42)
        return np.random.normal(10, 2, 100)
    
    @pytest.fixture
    def sample_data_binary(self):
        """Generate binary sample data for testing."""
        np.random.seed(42)
        return np.random.binomial(1, 0.7, 100).astype(float)
    
    @pytest.fixture
    def paired_data(self):
        """Generate paired sample data for testing."""
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        treatment = baseline + np.random.normal(1, 1, 50)  # Treatment effect
        return baseline, treatment
    
    def test_initialization(self):
        """Test StatisticalAnalyzer initialization."""
        config = {
            'confidence_level': 0.99,
            'bootstrap_resamples': 5000,
            'alpha': 0.01,
            'random_seed': 123
        }
        analyzer = StatisticalAnalyzer(config)
        
        assert analyzer.confidence_level == 0.99
        assert analyzer.bootstrap_resamples == 5000
        assert analyzer.alpha == 0.01
        assert analyzer.random_seed == 123
    
    def test_initialization_defaults(self):
        """Test StatisticalAnalyzer initialization with defaults."""
        analyzer = StatisticalAnalyzer()
        
        assert analyzer.confidence_level == 0.95
        assert analyzer.bootstrap_resamples == 10000
        assert analyzer.alpha == 0.05
        assert analyzer.random_seed == 42
    
    def test_bootstrap_confidence_interval(self, analyzer, sample_data_normal):
        """Test bootstrap confidence interval calculation."""
        ci = analyzer.compute_confidence_intervals(sample_data_normal, method="bootstrap")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == "bootstrap"
        assert ci.confidence_level == 0.95
        assert ci.lower < ci.upper
        
        # Check that the true mean is within the CI (should be true most of the time)
        true_mean = np.mean(sample_data_normal)
        assert ci.lower <= true_mean <= ci.upper
    
    def test_normal_confidence_interval(self, analyzer, sample_data_normal):
        """Test normal (t-distribution) confidence interval calculation."""
        ci = analyzer.compute_confidence_intervals(sample_data_normal, method="normal")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == "normal"
        assert ci.confidence_level == 0.95
        assert ci.lower < ci.upper
    
    def test_wilson_confidence_interval(self, analyzer, sample_data_binary):
        """Test Wilson confidence interval for proportions."""
        ci = analyzer.compute_confidence_intervals(sample_data_binary, method="wilson")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == "wilson"
        assert ci.confidence_level == 0.95
        assert 0 <= ci.lower <= ci.upper <= 1
    
    def test_confidence_interval_insufficient_data(self, analyzer):
        """Test confidence interval with insufficient data."""
        small_data = np.array([1.0, 2.0])  # Less than min_sample_size
        ci = analyzer.compute_confidence_intervals(small_data, method="bootstrap")
        
        assert "insufficient_data" in ci.method
        assert ci.lower == ci.upper  # Should be the mean
    
    def test_bootstrap_mean_estimate(self, analyzer, sample_data_normal):
        """Test bootstrap mean estimation."""
        result = analyzer.bootstrap_mean_estimate(sample_data_normal, n_resamples=500)
        
        assert isinstance(result, BootstrapResult)
        assert result.n_resamples == 500
        assert len(result.bootstrap_samples) == 500
        
        # Bootstrap mean should be close to original mean
        original_mean = np.mean(sample_data_normal)
        assert abs(result.bootstrap_mean - original_mean) < 0.5
        
        # Bootstrap std should be reasonable
        assert result.bootstrap_std > 0
        assert result.bootstrap_std < 1.0  # Should be much smaller than data std
    
    def test_paired_t_test(self, analyzer, paired_data):
        """Test paired t-test comparison."""
        baseline, treatment = paired_data
        
        comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            test_method=SignificanceTest.PAIRED_T_TEST
        )
        
        assert isinstance(comparison, ComparisonResult)
        assert comparison.model_a_id == "baseline"
        assert comparison.model_b_id == "treatment"
        assert comparison.test_method == "paired_t_test"
        assert comparison.sample_size_a == len(baseline)
        assert comparison.sample_size_b == len(treatment)
        
        # Treatment should have higher mean
        assert comparison.model_b_mean > comparison.model_a_mean
        assert comparison.difference > 0
        
        # Should be significant (we added a treatment effect)
        assert comparison.p_value < 0.05
        assert comparison.is_significant
    
    def test_wilcoxon_test(self, analyzer, paired_data):
        """Test Wilcoxon signed-rank test."""
        baseline, treatment = paired_data
        
        comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            test_method=SignificanceTest.WILCOXON
        )
        
        assert comparison.test_method == "wilcoxon"
        assert comparison.p_value is not None
        assert comparison.effect_size is not None
    
    def test_mann_whitney_u_test(self, analyzer):
        """Test Mann-Whitney U test for unpaired data."""
        np.random.seed(42)
        group_a = np.random.normal(10, 2, 30)
        group_b = np.random.normal(12, 2, 35)  # Different sample sizes
        
        comparison = analyzer.compare_models(
            group_a, group_b,
            "group_a", "group_b", "test_metric",
            test_method=SignificanceTest.MANN_WHITNEY_U
        )
        
        assert comparison.test_method == "mann_whitney_u"
        assert comparison.sample_size_a == 30
        assert comparison.sample_size_b == 35
        assert comparison.p_value is not None
    
    def test_bootstrap_significance_test(self, analyzer, paired_data):
        """Test bootstrap significance test."""
        baseline, treatment = paired_data
        
        comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            test_method=SignificanceTest.BOOTSTRAP
        )
        
        assert comparison.test_method == "bootstrap"
        assert comparison.p_value is not None
        assert 0 <= comparison.p_value <= 1
    
    def test_cohens_d_effect_size(self, analyzer, paired_data):
        """Test Cohen's d effect size calculation."""
        baseline, treatment = paired_data
        
        comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            effect_size_method=EffectSizeMethod.COHENS_D
        )
        
        assert comparison.effect_size_method == "cohens_d"
        assert comparison.effect_size is not None
        assert comparison.effect_size > 0  # Treatment should be better
    
    def test_cliffs_delta_effect_size(self, analyzer, paired_data):
        """Test Cliff's delta effect size calculation."""
        baseline, treatment = paired_data
        
        comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            effect_size_method=EffectSizeMethod.CLIFFS_DELTA
        )
        
        assert comparison.effect_size_method == "cliffs_delta"
        assert comparison.effect_size is not None
        assert -1 <= comparison.effect_size <= 1
    
    def test_hedges_g_effect_size(self, analyzer, paired_data):
        """Test Hedges' g effect size calculation."""
        baseline, treatment = paired_data
        
        comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            effect_size_method=EffectSizeMethod.HEDGES_G
        )
        
        assert comparison.effect_size_method == "hedges_g"
        assert comparison.effect_size is not None
        # Hedges' g should be slightly smaller than Cohen's d
        cohens_d_comparison = analyzer.compare_models(
            baseline, treatment,
            "baseline", "treatment", "test_metric",
            effect_size_method=EffectSizeMethod.COHENS_D
        )
        assert abs(comparison.effect_size) <= abs(cohens_d_comparison.effect_size)
    
    def test_benjamini_hochberg_correction(self, analyzer):
        """Test Benjamini-Hochberg multiple comparison correction."""
        # Create mock comparisons with known p-values
        p_values = [0.001, 0.01, 0.03, 0.05, 0.08, 0.15, 0.3, 0.5]
        comparisons = []
        
        for i, p_val in enumerate(p_values):
            comparison = ComparisonResult(
                model_a_id=f"model_a_{i}",
                model_b_id=f"model_b_{i}",
                metric_name="test_metric",
                model_a_mean=10.0,
                model_b_mean=11.0,
                difference=1.0,
                p_value=p_val,
                is_significant=p_val < 0.05,
                test_method="paired_t_test"
            )
            comparisons.append(comparison)
        
        result = analyzer.correct_multiple_comparisons(
            comparisons, method="benjamini_hochberg"
        )
        
        assert isinstance(result, MultipleComparisonResult)
        assert result.correction_method == "benjamini_hochberg"
        assert len(result.comparisons) == len(comparisons)
        
        # Check that adjusted p-values are reasonable
        for i, comp in enumerate(result.comparisons):
            assert comp.adjusted_p_value is not None
            assert comp.adjusted_p_value >= comp.p_value  # Should be >= original
            assert comp.adjusted_p_value <= 1.0
        
        # Should have fewer significant results after correction
        assert result.num_significant_after <= result.num_significant_before
    
    def test_bonferroni_correction(self, analyzer):
        """Test Bonferroni multiple comparison correction."""
        p_values = [0.001, 0.01, 0.03, 0.05]
        comparisons = []
        
        for i, p_val in enumerate(p_values):
            comparison = ComparisonResult(
                model_a_id=f"model_a_{i}",
                model_b_id=f"model_b_{i}",
                metric_name="test_metric",
                model_a_mean=10.0,
                model_b_mean=11.0,
                difference=1.0,
                p_value=p_val,
                is_significant=p_val < 0.05,
                test_method="paired_t_test"
            )
            comparisons.append(comparison)
        
        result = analyzer.correct_multiple_comparisons(
            comparisons, method="bonferroni"
        )
        
        assert result.correction_method == "bonferroni"
        
        # Bonferroni should multiply p-values by number of tests
        for i, comp in enumerate(result.comparisons):
            expected_adjusted = min(1.0, p_values[i] * len(p_values))
            assert abs(comp.adjusted_p_value - expected_adjusted) < 1e-10
    
    def test_holm_correction(self, analyzer):
        """Test Holm multiple comparison correction."""
        p_values = [0.001, 0.01, 0.03, 0.05]
        comparisons = []
        
        for i, p_val in enumerate(p_values):
            comparison = ComparisonResult(
                model_a_id=f"model_a_{i}",
                model_b_id=f"model_b_{i}",
                metric_name="test_metric",
                model_a_mean=10.0,
                model_b_mean=11.0,
                difference=1.0,
                p_value=p_val,
                is_significant=p_val < 0.05,
                test_method="paired_t_test"
            )
            comparisons.append(comparison)
        
        result = analyzer.correct_multiple_comparisons(
            comparisons, method="holm"
        )
        
        assert result.correction_method == "holm"
        assert len(result.comparisons) == len(comparisons)
    
    def test_empty_comparisons_correction(self, analyzer):
        """Test multiple comparison correction with empty list."""
        result = analyzer.correct_multiple_comparisons([], method="benjamini_hochberg")
        
        assert isinstance(result, MultipleComparisonResult)
        assert len(result.comparisons) == 0
        assert result.num_significant_before == 0
        assert result.num_significant_after == 0
    
    def test_get_interpretation(self, analyzer):
        """Test interpretation of comparison results."""
        # Significant result with large effect
        comparison = ComparisonResult(
            model_a_id="model_a",
            model_b_id="model_b",
            metric_name="success_rate",
            model_a_mean=0.7,
            model_b_mean=0.9,
            difference=0.2,
            p_value=0.001,
            adjusted_p_value=0.005,
            is_significant=True,
            effect_size=0.8,
            effect_size_method="cohens_d",
            test_method="paired_t_test"
        )
        
        interpretation = analyzer.get_interpretation(comparison)
        
        assert "significant" in interpretation['significance'].lower()
        assert "large" in interpretation['effect_size'].lower()
        assert "model_b" in interpretation['practical']
        assert "success_rate" in interpretation['practical']
    
    def test_get_interpretation_non_significant(self, analyzer):
        """Test interpretation of non-significant results."""
        comparison = ComparisonResult(
            model_a_id="model_a",
            model_b_id="model_b",
            metric_name="success_rate",
            model_a_mean=0.75,
            model_b_mean=0.76,
            difference=0.01,
            p_value=0.8,
            is_significant=False,
            effect_size=0.1,
            effect_size_method="cohens_d",
            test_method="paired_t_test"
        )
        
        interpretation = analyzer.get_interpretation(comparison)
        
        assert "not" in interpretation['significance'].lower()
        assert "negligible" in interpretation['effect_size'].lower()
    
    def test_invalid_test_method(self, analyzer, paired_data):
        """Test error handling for invalid test method."""
        baseline, treatment = paired_data
        
        with pytest.raises(ValueError, match="Unknown test method"):
            analyzer.compare_models(
                baseline, treatment,
                "baseline", "treatment", "test_metric",
                test_method="invalid_test"
            )
    
    def test_invalid_effect_size_method(self, analyzer, paired_data):
        """Test error handling for invalid effect size method."""
        baseline, treatment = paired_data
        
        with pytest.raises(ValueError, match="Unknown effect size method"):
            analyzer.compare_models(
                baseline, treatment,
                "baseline", "treatment", "test_metric",
                effect_size_method="invalid_method"
            )
    
    def test_invalid_confidence_interval_method(self, analyzer, sample_data_normal):
        """Test error handling for invalid CI method."""
        with pytest.raises(ValueError, match="Unknown confidence interval method"):
            analyzer.compute_confidence_intervals(sample_data_normal, method="invalid_method")
    
    def test_invalid_correction_method(self, analyzer):
        """Test error handling for invalid correction method."""
        comparison = ComparisonResult(
            model_a_id="model_a",
            model_b_id="model_b",
            metric_name="test_metric",
            model_a_mean=10.0,
            model_b_mean=11.0,
            difference=1.0,
            p_value=0.05,
            is_significant=True,
            test_method="paired_t_test"
        )
        
        with pytest.raises(ValueError, match="Unknown correction method"):
            analyzer.correct_multiple_comparisons([comparison], method="invalid_method")
    
    def test_paired_test_unequal_sizes(self, analyzer):
        """Test error handling for paired tests with unequal sample sizes."""
        data_a = np.array([1, 2, 3])
        data_b = np.array([4, 5])  # Different size
        
        with pytest.raises(ValueError, match="equal sample sizes"):
            analyzer.compare_models(
                data_a, data_b,
                "model_a", "model_b", "test_metric",
                test_method=SignificanceTest.PAIRED_T_TEST
            )
    
    def test_zero_variance_data(self, analyzer):
        """Test handling of zero variance data."""
        # All values are the same
        data_a = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        data_b = np.array([6.0, 6.0, 6.0, 6.0, 6.0])
        
        comparison = analyzer.compare_models(
            data_a, data_b,
            "model_a", "model_b", "test_metric"
        )
        
        # Should handle gracefully without errors
        assert comparison.effect_size is not None
        assert not np.isnan(comparison.effect_size)
    
    def test_empty_data(self, analyzer):
        """Test handling of empty data arrays."""
        data_a = np.array([])
        data_b = np.array([1.0])
        
        # Should handle gracefully or raise appropriate error
        try:
            comparison = analyzer.compare_models(
                data_a, data_b,
                "model_a", "model_b", "test_metric"
            )
            # If it doesn't raise an error, check the result is reasonable
            assert comparison is not None
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty data
            pass
    
    def test_confidence_level_parameter(self, analyzer, sample_data_normal):
        """Test custom confidence level parameter."""
        ci_95 = analyzer.compute_confidence_intervals(
            sample_data_normal, method="bootstrap", confidence_level=0.95
        )
        ci_99 = analyzer.compute_confidence_intervals(
            sample_data_normal, method="bootstrap", confidence_level=0.99
        )
        
        # 99% CI should be wider than 95% CI
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        assert width_99 > width_95
    
    def test_reproducibility(self, analyzer, sample_data_normal):
        """Test that results are reproducible with same seed."""
        # Reset seed and run twice
        np.random.seed(42)
        ci1 = analyzer.compute_confidence_intervals(sample_data_normal, method="bootstrap")
        
        np.random.seed(42)
        ci2 = analyzer.compute_confidence_intervals(sample_data_normal, method="bootstrap")
        
        # Results should be identical
        assert abs(ci1.lower - ci2.lower) < 1e-10
        assert abs(ci1.upper - ci2.upper) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__])