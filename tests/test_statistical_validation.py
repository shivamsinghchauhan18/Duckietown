#!/usr/bin/env python3
"""
🧪 STATISTICAL VALIDATION TESTS 🧪
Specialized tests for statistical validation of confidence intervals and significance testing

This module focuses on validating the statistical rigor of the evaluation system,
including confidence interval accuracy, significance test reliability, and
multiple comparison correction validation.

Requirements covered: 8.4, 13.1, 13.2
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from scipy import stats
from typing import List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.statistical_analyzer import (
    StatisticalAnalyzer, SignificanceTest, EffectSizeMethod,
    ComparisonResult, MultipleComparisonResult
)
from duckietown_utils.metrics_calculator import ConfidenceInterval


class TestStatisticalValidation:
    """Test suite for statistical validation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create statistical analyzer with high precision for validation."""
        config = {
            'confidence_level': 0.95,
            'bootstrap_resamples': 10000,  # High precision for validation
            'alpha': 0.05,
            'random_seed': 42
        }
        return StatisticalAnalyzer(config)
    
    def test_confidence_interval_coverage_validation(self, analyzer):
        """Test that confidence intervals have correct coverage probability."""
        # Test with known normal distribution
        true_mean = 10.0
        true_std = 2.0
        sample_size = 100
        n_trials = 1000
        
        coverage_count = 0
        
        for trial in range(n_trials):
            # Generate sample from known distribution
            np.random.seed(trial)
            sample = np.random.normal(true_mean, true_std, sample_size)
            
            # Compute confidence interval
            ci = analyzer.compute_confidence_intervals(sample, method="bootstrap")
            
            # Check if true mean is within CI
            if ci.lower <= true_mean <= ci.upper:
                coverage_count += 1
        
        # Coverage should be approximately 95%
        coverage_rate = coverage_count / n_trials
        
        # Allow for some sampling variation (should be within 2 standard errors)
        expected_coverage = 0.95
        se_coverage = np.sqrt(expected_coverage * (1 - expected_coverage) / n_trials)
        margin_of_error = 2 * se_coverage
        
        assert abs(coverage_rate - expected_coverage) <= margin_of_error, \
            f"Coverage rate {coverage_rate:.3f} is outside expected range"
    
    def test_wilson_confidence_interval_accuracy(self, analyzer):
        """Test Wilson confidence interval accuracy for proportions."""
        # Test with known binomial distribution
        true_p = 0.7
        sample_size = 100
        n_trials = 1000
        
        coverage_count = 0
        
        for trial in range(n_trials):
            # Generate binomial sample
            np.random.seed(trial)
            sample = np.random.binomial(1, true_p, sample_size).astype(float)
            
            # Compute Wilson CI
            ci = analyzer.compute_confidence_intervals(sample, method="wilson")
            
            # Check coverage
            if ci.lower <= true_p <= ci.upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_trials
        
        # Wilson interval should have good coverage even for proportions
        assert coverage_rate >= 0.93, f"Wilson CI coverage {coverage_rate:.3f} too low"
        assert coverage_rate <= 0.97, f"Wilson CI coverage {coverage_rate:.3f} too high"
    
    def test_significance_test_type_i_error_rate(self, analyzer):
        """Test that significance tests have correct Type I error rate."""
        # Test with identical distributions (null hypothesis is true)
        n_trials = 1000
        sample_size = 50
        alpha = 0.05
        
        false_positive_count = 0
        
        for trial in range(n_trials):
            # Generate two samples from same distribution
            np.random.seed(trial)
            sample_a = np.random.normal(10, 2, sample_size)
            sample_b = np.random.normal(10, 2, sample_size)  # Same distribution
            
            # Perform significance test
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                "model_a", "model_b", "test_metric",
                test_method=SignificanceTest.PAIRED_T_TEST
            )
            
            if comparison.is_significant:
                false_positive_count += 1
        
        type_i_error_rate = false_positive_count / n_trials
        
        # Type I error rate should be approximately alpha (5%)
        expected_rate = alpha
        se_rate = np.sqrt(expected_rate * (1 - expected_rate) / n_trials)
        margin_of_error = 2 * se_rate
        
        assert abs(type_i_error_rate - expected_rate) <= margin_of_error, \
            f"Type I error rate {type_i_error_rate:.3f} deviates from expected {expected_rate:.3f}"
    
    def test_significance_test_power_validation(self, analyzer):
        """Test statistical power of significance tests."""
        # Test with known effect size
        effect_size = 0.5  # Medium effect size
        sample_size = 50
        n_trials = 1000
        
        significant_count = 0
        
        for trial in range(n_trials):
            # Generate samples with known difference
            np.random.seed(trial)
            sample_a = np.random.normal(10, 2, sample_size)
            sample_b = np.random.normal(10 + effect_size * 2, 2, sample_size)  # Effect size of 0.5
            
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                "model_a", "model_b", "test_metric",
                test_method=SignificanceTest.PAIRED_T_TEST
            )
            
            if comparison.is_significant:
                significant_count += 1
        
        statistical_power = significant_count / n_trials
        
        # For medium effect size (0.5) with n=50, power should be > 0.7
        assert statistical_power >= 0.7, \
            f"Statistical power {statistical_power:.3f} too low for medium effect size"
    
    def test_effect_size_accuracy(self, analyzer):
        """Test accuracy of effect size calculations."""
        # Test Cohen's d with known effect size
        true_effect_size = 0.8  # Large effect size
        sample_size = 100
        n_trials = 100
        
        effect_sizes = []
        
        for trial in range(n_trials):
            np.random.seed(trial)
            sample_a = np.random.normal(10, 2, sample_size)
            sample_b = np.random.normal(10 + true_effect_size * 2, 2, sample_size)
            
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                "model_a", "model_b", "test_metric",
                effect_size_method=EffectSizeMethod.COHENS_D
            )
            
            effect_sizes.append(comparison.effect_size)
        
        mean_effect_size = np.mean(effect_sizes)
        
        # Mean effect size should be close to true effect size
        assert abs(mean_effect_size - true_effect_size) < 0.1, \
            f"Mean effect size {mean_effect_size:.3f} deviates from true {true_effect_size:.3f}"
    
    def test_benjamini_hochberg_fdr_control(self, analyzer):
        """Test that Benjamini-Hochberg procedure controls FDR."""
        # Create mixture of true and false hypotheses
        n_true_null = 800  # 80% true null hypotheses
        n_false_null = 200  # 20% false null hypotheses
        sample_size = 50
        effect_size = 1.0  # Large effect for false nulls
        
        comparisons = []
        true_nulls = []  # Track which comparisons are true nulls
        
        # Generate comparisons
        for i in range(n_true_null + n_false_null):
            np.random.seed(i)
            
            if i < n_true_null:
                # True null: same distribution
                sample_a = np.random.normal(10, 2, sample_size)
                sample_b = np.random.normal(10, 2, sample_size)
                true_nulls.append(True)
            else:
                # False null: different distributions
                sample_a = np.random.normal(10, 2, sample_size)
                sample_b = np.random.normal(10 + effect_size * 2, 2, sample_size)
                true_nulls.append(False)
            
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                f"model_a_{i}", f"model_b_{i}", "test_metric"
            )
            comparisons.append(comparison)
        
        # Apply Benjamini-Hochberg correction
        corrected_results = analyzer.correct_multiple_comparisons(
            comparisons, method="benjamini_hochberg"
        )
        
        # Calculate FDR
        significant_comparisons = [
            (comp, true_nulls[i]) 
            for i, comp in enumerate(corrected_results.comparisons)
            if comp.is_significant_adjusted
        ]
        
        if len(significant_comparisons) > 0:
            false_discoveries = sum(1 for _, is_true_null in significant_comparisons if is_true_null)
            fdr = false_discoveries / len(significant_comparisons)
            
            # FDR should be controlled at alpha level (0.05)
            assert fdr <= 0.1, f"FDR {fdr:.3f} exceeds expected control level"
        
        # Should have reasonable power (detect some false nulls)
        detected_false_nulls = sum(
            1 for _, is_true_null in significant_comparisons if not is_true_null
        )
        assert detected_false_nulls > 0, "No false nulls detected - power too low"
    
    def test_bootstrap_consistency(self, analyzer):
        """Test consistency of bootstrap confidence intervals."""
        # Generate sample data
        np.random.seed(42)
        sample_data = np.random.normal(10, 2, 100)
        
        # Compute multiple bootstrap CIs with same data
        cis = []
        for i in range(10):
            np.random.seed(42)  # Same seed for reproducibility
            ci = analyzer.compute_confidence_intervals(sample_data, method="bootstrap")
            cis.append((ci.lower, ci.upper))
        
        # All CIs should be identical (same seed)
        for i in range(1, len(cis)):
            assert abs(cis[i][0] - cis[0][0]) < 1e-10, "Bootstrap CIs not reproducible"
            assert abs(cis[i][1] - cis[0][1]) < 1e-10, "Bootstrap CIs not reproducible"
        
        # Test with different seeds - should get different but reasonable CIs
        cis_different_seeds = []
        for i in range(10):
            np.random.seed(i)  # Different seeds
            ci = analyzer.compute_confidence_intervals(sample_data, method="bootstrap")
            cis_different_seeds.append((ci.lower, ci.upper))
        
        # CIs should vary but all contain approximately the same range
        widths = [upper - lower for lower, upper in cis_different_seeds]
        mean_width = np.mean(widths)
        
        # All widths should be reasonably similar
        for width in widths:
            assert abs(width - mean_width) / mean_width < 0.1, \
                "Bootstrap CI widths vary too much across seeds"
    
    def test_multiple_comparison_correction_ordering(self, analyzer):
        """Test that multiple comparison corrections preserve p-value ordering."""
        # Create comparisons with known p-value ordering
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.5]
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
        
        # Test different correction methods
        for method in ["benjamini_hochberg", "bonferroni", "holm"]:
            corrected_results = analyzer.correct_multiple_comparisons(
                comparisons, method=method
            )
            
            # Adjusted p-values should preserve ordering
            adjusted_p_values = [comp.adjusted_p_value for comp in corrected_results.comparisons]
            
            for i in range(1, len(adjusted_p_values)):
                assert adjusted_p_values[i] >= adjusted_p_values[i-1], \
                    f"{method} correction doesn't preserve p-value ordering"
            
            # Adjusted p-values should be >= original p-values
            for orig_comp, adj_comp in zip(comparisons, corrected_results.comparisons):
                assert adj_comp.adjusted_p_value >= orig_comp.p_value, \
                    f"{method} correction reduces p-value"
    
    def test_confidence_interval_width_consistency(self, analyzer):
        """Test that confidence interval widths behave consistently."""
        base_sample_size = 50
        true_mean = 10.0
        true_std = 2.0
        
        # Test different sample sizes
        sample_sizes = [25, 50, 100, 200]
        ci_widths = []
        
        for n in sample_sizes:
            np.random.seed(42)
            sample = np.random.normal(true_mean, true_std, n)
            ci = analyzer.compute_confidence_intervals(sample, method="bootstrap")
            ci_widths.append(ci.upper - ci.lower)
        
        # CI width should decrease with increasing sample size
        for i in range(1, len(ci_widths)):
            assert ci_widths[i] < ci_widths[i-1], \
                "CI width doesn't decrease with larger sample size"
        
        # Test different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        np.random.seed(42)
        sample = np.random.normal(true_mean, true_std, base_sample_size)
        
        level_widths = []
        for level in confidence_levels:
            ci = analyzer.compute_confidence_intervals(
                sample, method="bootstrap", confidence_level=level
            )
            level_widths.append(ci.upper - ci.lower)
        
        # CI width should increase with higher confidence level
        for i in range(1, len(level_widths)):
            assert level_widths[i] > level_widths[i-1], \
                "CI width doesn't increase with higher confidence level"
    
    def test_nonparametric_test_robustness(self, analyzer):
        """Test robustness of nonparametric tests to distribution assumptions."""
        sample_size = 100
        n_trials = 100
        
        # Test with non-normal distributions
        distributions = [
            ("exponential", lambda: np.random.exponential(2, sample_size)),
            ("uniform", lambda: np.random.uniform(0, 10, sample_size)),
            ("bimodal", lambda: np.concatenate([
                np.random.normal(5, 1, sample_size//2),
                np.random.normal(15, 1, sample_size//2)
            ]))
        ]
        
        for dist_name, dist_generator in distributions:
            significant_count = 0
            
            for trial in range(n_trials):
                np.random.seed(trial)
                
                # Generate samples from same distribution (null hypothesis true)
                sample_a = dist_generator()
                sample_b = dist_generator()
                
                # Use nonparametric test
                comparison = analyzer.compare_models(
                    sample_a, sample_b,
                    "model_a", "model_b", "test_metric",
                    test_method=SignificanceTest.MANN_WHITNEY_U
                )
                
                if comparison.is_significant:
                    significant_count += 1
            
            type_i_error_rate = significant_count / n_trials
            
            # Type I error rate should still be controlled for non-normal distributions
            assert type_i_error_rate <= 0.1, \
                f"Type I error rate {type_i_error_rate:.3f} too high for {dist_name} distribution"
    
    def test_effect_size_interpretation_consistency(self, analyzer):
        """Test consistency of effect size interpretations."""
        sample_size = 100
        
        # Test different effect sizes
        effect_sizes = [0.1, 0.3, 0.5, 0.8, 1.2]  # Small to very large
        
        for true_effect in effect_sizes:
            np.random.seed(42)
            sample_a = np.random.normal(10, 2, sample_size)
            sample_b = np.random.normal(10 + true_effect * 2, 2, sample_size)
            
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                "model_a", "model_b", "test_metric",
                effect_size_method=EffectSizeMethod.COHENS_D
            )
            
            interpretation = analyzer.get_interpretation(comparison)
            
            # Check that interpretation matches expected categories
            if true_effect < 0.2:
                assert "negligible" in interpretation['effect_size'].lower()
            elif true_effect < 0.5:
                assert "small" in interpretation['effect_size'].lower()
            elif true_effect < 0.8:
                assert "medium" in interpretation['effect_size'].lower()
            else:
                assert "large" in interpretation['effect_size'].lower()


class TestReproducibilityValidation:
    """Test suite for reproducibility validation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with fixed seed."""
        return StatisticalAnalyzer({'random_seed': 12345})
    
    def test_seed_reproducibility(self, analyzer):
        """Test that results are reproducible with same seed."""
        sample_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Run same analysis multiple times
        results = []
        for _ in range(5):
            np.random.seed(12345)  # Reset seed
            ci = analyzer.compute_confidence_intervals(sample_data, method="bootstrap")
            results.append((ci.lower, ci.upper))
        
        # All results should be identical
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10
            assert abs(results[i][1] - results[0][1]) < 1e-10
    
    def test_configuration_reproducibility(self, analyzer):
        """Test that same configuration produces same results."""
        # Create two analyzers with same configuration
        config = {
            'confidence_level': 0.95,
            'bootstrap_resamples': 1000,
            'alpha': 0.05,
            'random_seed': 42
        }
        
        analyzer1 = StatisticalAnalyzer(config)
        analyzer2 = StatisticalAnalyzer(config)
        
        sample_data = np.random.normal(10, 2, 50)
        
        # Both should produce identical results
        np.random.seed(42)
        ci1 = analyzer1.compute_confidence_intervals(sample_data, method="bootstrap")
        
        np.random.seed(42)
        ci2 = analyzer2.compute_confidence_intervals(sample_data, method="bootstrap")
        
        assert abs(ci1.lower - ci2.lower) < 1e-10
        assert abs(ci1.upper - ci2.upper) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])