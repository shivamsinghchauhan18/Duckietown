#!/usr/bin/env python3
"""
📊 STATISTICAL ANALYZER 📊
Rigorous statistical analysis for model evaluation

This module implements the StatisticalAnalyzer class with confidence interval calculations,
bootstrap resampling for robust mean estimates, significance testing with paired comparisons,
and Benjamini-Hochberg multiple comparison correction.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import metrics calculator for data structures
from duckietown_utils.metrics_calculator import MetricResult, ConfidenceInterval

class SignificanceTest(Enum):
    """Types of significance tests."""
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP = "bootstrap"

class EffectSizeMethod(Enum):
    """Methods for calculating effect size."""
    COHENS_D = "cohens_d"
    CLIFFS_DELTA = "cliffs_delta"
    HEDGES_G = "hedges_g"

@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two models."""
    model_a_id: str
    model_b_id: str
    metric_name: str
    model_a_mean: float
    model_b_mean: float
    difference: float
    p_value: float
    adjusted_p_value: Optional[float] = None
    is_significant: bool = False
    effect_size: Optional[float] = None
    effect_size_method: Optional[str] = None
    test_method: str = "paired_t_test"
    confidence_interval: Optional[ConfidenceInterval] = None
    sample_size_a: int = 0
    sample_size_b: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultipleComparisonResult:
    """Result of multiple comparison correction."""
    comparisons: List[ComparisonResult]
    correction_method: str
    alpha: float = 0.05
    num_significant_before: int = 0
    num_significant_after: int = 0
    rejected_hypotheses: List[bool] = field(default_factory=list)

@dataclass
class BootstrapResult:
    """Result of bootstrap resampling."""
    original_statistic: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: ConfidenceInterval
    bootstrap_samples: np.ndarray
    n_resamples: int

class StatisticalAnalyzer:
    """Comprehensive statistical analyzer for model evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the statistical analyzer.
        
        Args:
            config: Configuration dictionary for the analyzer
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.bootstrap_resamples = self.config.get('bootstrap_resamples', 10000)
        self.alpha = self.config.get('alpha', 0.05)
        self.min_sample_size = self.config.get('min_sample_size', 5)
        
        # Random seed for reproducibility
        self.random_seed = self.config.get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        self.logger.info("📊 Statistical Analyzer initialized")
        self.logger.info(f"🎯 Confidence level: {self.confidence_level}")
        self.logger.info(f"🔄 Bootstrap resamples: {self.bootstrap_resamples}")
        self.logger.info(f"📏 Alpha level: {self.alpha}")
    
    def compute_confidence_intervals(self, data: np.ndarray, 
                                   method: str = "bootstrap",
                                   confidence_level: Optional[float] = None) -> ConfidenceInterval:
        """Compute confidence interval for data.
        
        Args:
            data: Array of data values
            method: Method for CI calculation ('bootstrap', 'normal', 'wilson')
            confidence_level: Confidence level (defaults to instance setting)
            
        Returns:
            ConfidenceInterval: Confidence interval result
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        data = np.asarray(data)
        
        if len(data) < self.min_sample_size:
            self.logger.warning(f"Sample size {len(data)} below minimum {self.min_sample_size}")
            mean_val = np.mean(data) if len(data) > 0 else 0.0
            return ConfidenceInterval(
                lower=mean_val,
                upper=mean_val,
                confidence_level=confidence_level,
                method=f"{method}_insufficient_data"
            )
        
        if method == "bootstrap":
            return self._bootstrap_confidence_interval(data, confidence_level)
        elif method == "normal":
            return self._normal_confidence_interval(data, confidence_level)
        elif method == "wilson":
            return self._wilson_confidence_interval(data, confidence_level)
        else:
            raise ValueError(f"Unknown confidence interval method: {method}")
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, 
                                     confidence_level: float) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval.
        
        Args:
            data: Array of data values
            confidence_level: Confidence level
            
        Returns:
            ConfidenceInterval: Bootstrap confidence interval
        """
        n = len(data)
        bootstrap_means = []
        
        # Generate bootstrap samples
        for _ in range(self.bootstrap_resamples):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method="bootstrap"
        )
    
    def _normal_confidence_interval(self, data: np.ndarray, 
                                  confidence_level: float) -> ConfidenceInterval:
        """Calculate normal (t-distribution) confidence interval.
        
        Args:
            data: Array of data values
            confidence_level: Confidence level
            
        Returns:
            ConfidenceInterval: Normal confidence interval
        """
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)  # Standard error of the mean
        
        # t-critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin = t_critical * std_err
        
        return ConfidenceInterval(
            lower=float(mean - margin),
            upper=float(mean + margin),
            confidence_level=confidence_level,
            method="normal"
        )
    
    def _wilson_confidence_interval(self, data: np.ndarray, 
                                  confidence_level: float) -> ConfidenceInterval:
        """Calculate Wilson confidence interval for proportions.
        
        Args:
            data: Array of binary values (0.0 or 1.0)
            confidence_level: Confidence level
            
        Returns:
            ConfidenceInterval: Wilson confidence interval
        """
        n = len(data)
        p = np.mean(data)
        
        # Z-score for confidence level
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha/2)
        
        # Wilson interval calculation
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method="wilson"
        )
    
    def bootstrap_mean_estimate(self, data: np.ndarray, 
                               n_resamples: Optional[int] = None) -> BootstrapResult:
        """Perform bootstrap resampling for robust mean estimation.
        
        Args:
            data: Array of data values
            n_resamples: Number of bootstrap resamples (defaults to instance setting)
            
        Returns:
            BootstrapResult: Bootstrap resampling result
        """
        if n_resamples is None:
            n_resamples = self.bootstrap_resamples
        
        data = np.asarray(data)
        n = len(data)
        original_mean = np.mean(data)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_resamples):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate statistics
        bootstrap_mean = np.mean(bootstrap_means)
        bootstrap_std = np.std(bootstrap_means)
        
        # Calculate confidence interval
        ci = self._bootstrap_confidence_interval(data, self.confidence_level)
        
        return BootstrapResult(
            original_statistic=float(original_mean),
            bootstrap_mean=float(bootstrap_mean),
            bootstrap_std=float(bootstrap_std),
            confidence_interval=ci,
            bootstrap_samples=bootstrap_means,
            n_resamples=n_resamples
        )
    
    def compare_models(self, model_a_data: np.ndarray, model_b_data: np.ndarray,
                      model_a_id: str, model_b_id: str, metric_name: str,
                      test_method: SignificanceTest = SignificanceTest.PAIRED_T_TEST,
                      effect_size_method: EffectSizeMethod = EffectSizeMethod.COHENS_D) -> ComparisonResult:
        """Compare two models using statistical significance testing.
        
        Args:
            model_a_data: Data from model A
            model_b_data: Data from model B
            model_a_id: ID of model A
            model_b_id: ID of model B
            metric_name: Name of the metric being compared
            test_method: Statistical test method to use
            effect_size_method: Method for calculating effect size
            
        Returns:
            ComparisonResult: Statistical comparison result
        """
        model_a_data = np.asarray(model_a_data)
        model_b_data = np.asarray(model_b_data)
        
        # Basic statistics
        mean_a = np.mean(model_a_data)
        mean_b = np.mean(model_b_data)
        difference = mean_b - mean_a
        
        # Perform significance test
        if test_method == SignificanceTest.PAIRED_T_TEST:
            if len(model_a_data) != len(model_b_data):
                raise ValueError("Paired t-test requires equal sample sizes")
            statistic, p_value = ttest_rel(model_b_data, model_a_data)
            test_name = "paired_t_test"
        
        elif test_method == SignificanceTest.WILCOXON:
            if len(model_a_data) != len(model_b_data):
                raise ValueError("Wilcoxon test requires equal sample sizes")
            statistic, p_value = wilcoxon(model_b_data, model_a_data)
            test_name = "wilcoxon"
        
        elif test_method == SignificanceTest.MANN_WHITNEY_U:
            statistic, p_value = mannwhitneyu(model_b_data, model_a_data, alternative='two-sided')
            test_name = "mann_whitney_u"
        
        elif test_method == SignificanceTest.BOOTSTRAP:
            p_value = self._bootstrap_significance_test(model_a_data, model_b_data)
            statistic = difference
            test_name = "bootstrap"
        
        else:
            raise ValueError(f"Unknown test method: {test_method}")
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(
            model_a_data, model_b_data, effect_size_method
        )
        
        # Calculate confidence interval for the difference
        if test_method in [SignificanceTest.PAIRED_T_TEST, SignificanceTest.WILCOXON]:
            # For paired tests, calculate CI of differences
            differences = model_b_data - model_a_data
            ci = self.compute_confidence_intervals(differences, method="bootstrap")
        else:
            # For unpaired tests, use bootstrap CI of difference in means
            ci = self._bootstrap_difference_ci(model_a_data, model_b_data)
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        return ComparisonResult(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            metric_name=metric_name,
            model_a_mean=float(mean_a),
            model_b_mean=float(mean_b),
            difference=float(difference),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=float(effect_size),
            effect_size_method=effect_size_method.value,
            test_method=test_name,
            confidence_interval=ci,
            sample_size_a=len(model_a_data),
            sample_size_b=len(model_b_data),
            metadata={
                'test_statistic': float(statistic),
                'alpha': self.alpha
            }
        )
    
    def _bootstrap_significance_test(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Perform bootstrap significance test.
        
        Args:
            data_a: Data from group A
            data_b: Data from group B
            
        Returns:
            float: P-value from bootstrap test
        """
        # Observed difference
        observed_diff = np.mean(data_b) - np.mean(data_a)
        
        # Pool the data under null hypothesis
        pooled_data = np.concatenate([data_a, data_b])
        n_a, n_b = len(data_a), len(data_b)
        
        # Bootstrap under null hypothesis
        bootstrap_diffs = []
        for _ in range(self.bootstrap_resamples):
            # Resample from pooled data
            resampled = np.random.choice(pooled_data, size=n_a + n_b, replace=True)
            bootstrap_a = resampled[:n_a]
            bootstrap_b = resampled[n_a:]
            bootstrap_diff = np.mean(bootstrap_b) - np.mean(bootstrap_a)
            bootstrap_diffs.append(bootstrap_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return float(p_value)
    
    def _bootstrap_difference_ci(self, data_a: np.ndarray, data_b: np.ndarray) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval for difference in means.
        
        Args:
            data_a: Data from group A
            data_b: Data from group B
            
        Returns:
            ConfidenceInterval: CI for difference in means
        """
        bootstrap_diffs = []
        n_a, n_b = len(data_a), len(data_b)
        
        for _ in range(self.bootstrap_resamples):
            bootstrap_a = np.random.choice(data_a, size=n_a, replace=True)
            bootstrap_b = np.random.choice(data_b, size=n_b, replace=True)
            diff = np.mean(bootstrap_b) - np.mean(bootstrap_a)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_diffs, lower_percentile)
        upper = np.percentile(bootstrap_diffs, upper_percentile)
        
        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            confidence_level=self.confidence_level,
            method="bootstrap_difference"
        )
    
    def _calculate_effect_size(self, data_a: np.ndarray, data_b: np.ndarray,
                              method: EffectSizeMethod) -> float:
        """Calculate effect size between two groups.
        
        Args:
            data_a: Data from group A
            data_b: Data from group B
            method: Effect size calculation method
            
        Returns:
            float: Effect size value
        """
        if method == EffectSizeMethod.COHENS_D:
            return self._cohens_d(data_a, data_b)
        elif method == EffectSizeMethod.CLIFFS_DELTA:
            return self._cliffs_delta(data_a, data_b)
        elif method == EffectSizeMethod.HEDGES_G:
            return self._hedges_g(data_a, data_b)
        else:
            raise ValueError(f"Unknown effect size method: {method}")
    
    def _cohens_d(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Calculate Cohen's d effect size.
        
        Args:
            data_a: Data from group A
            data_b: Data from group B
            
        Returns:
            float: Cohen's d value
        """
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        var_a, var_b = np.var(data_a, ddof=1), np.var(data_b, ddof=1)
        n_a, n_b = len(data_a), len(data_b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean_b - mean_a) / pooled_std
    
    def _cliffs_delta(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Calculate Cliff's delta effect size.
        
        Args:
            data_a: Data from group A
            data_b: Data from group B
            
        Returns:
            float: Cliff's delta value (-1 to 1)
        """
        n_a, n_b = len(data_a), len(data_b)
        
        if n_a == 0 or n_b == 0:
            return 0.0
        
        # Count pairs where B > A and A > B
        greater = 0
        less = 0
        
        for a_val in data_a:
            for b_val in data_b:
                if b_val > a_val:
                    greater += 1
                elif a_val > b_val:
                    less += 1
        
        total_pairs = n_a * n_b
        return (greater - less) / total_pairs
    
    def _hedges_g(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d).
        
        Args:
            data_a: Data from group A
            data_b: Data from group B
            
        Returns:
            float: Hedges' g value
        """
        cohens_d = self._cohens_d(data_a, data_b)
        n_a, n_b = len(data_a), len(data_b)
        
        # Bias correction factor
        df = n_a + n_b - 2
        correction = 1 - (3 / (4 * df - 1))
        
        return cohens_d * correction
    
    def correct_multiple_comparisons(self, comparisons: List[ComparisonResult],
                                   method: str = "benjamini_hochberg",
                                   alpha: Optional[float] = None) -> MultipleComparisonResult:
        """Apply multiple comparison correction to a list of comparisons.
        
        Args:
            comparisons: List of comparison results
            method: Correction method ('benjamini_hochberg', 'bonferroni', 'holm')
            alpha: Significance level (defaults to instance setting)
            
        Returns:
            MultipleComparisonResult: Corrected comparison results
        """
        if alpha is None:
            alpha = self.alpha
        
        if not comparisons:
            return MultipleComparisonResult(
                comparisons=[],
                correction_method=method,
                alpha=alpha,
                num_significant_before=0,
                num_significant_after=0,
                rejected_hypotheses=[]
            )
        
        # Extract p-values
        p_values = np.array([comp.p_value for comp in comparisons])
        num_significant_before = np.sum(p_values < alpha)
        
        # Apply correction
        if method == "benjamini_hochberg":
            rejected, adjusted_p_values = self._benjamini_hochberg_correction(p_values, alpha)
        elif method == "bonferroni":
            rejected, adjusted_p_values = self._bonferroni_correction(p_values, alpha)
        elif method == "holm":
            rejected, adjusted_p_values = self._holm_correction(p_values, alpha)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Update comparison results
        corrected_comparisons = []
        for i, comparison in enumerate(comparisons):
            corrected_comparison = ComparisonResult(
                model_a_id=comparison.model_a_id,
                model_b_id=comparison.model_b_id,
                metric_name=comparison.metric_name,
                model_a_mean=comparison.model_a_mean,
                model_b_mean=comparison.model_b_mean,
                difference=comparison.difference,
                p_value=comparison.p_value,
                adjusted_p_value=float(adjusted_p_values[i]),
                is_significant=bool(rejected[i]),
                effect_size=comparison.effect_size,
                effect_size_method=comparison.effect_size_method,
                test_method=comparison.test_method,
                confidence_interval=comparison.confidence_interval,
                sample_size_a=comparison.sample_size_a,
                sample_size_b=comparison.sample_size_b,
                metadata=comparison.metadata.copy()
            )
            corrected_comparison.metadata['correction_method'] = method
            corrected_comparisons.append(corrected_comparison)
        
        num_significant_after = np.sum(rejected)
        
        return MultipleComparisonResult(
            comparisons=corrected_comparisons,
            correction_method=method,
            alpha=alpha,
            num_significant_before=int(num_significant_before),
            num_significant_after=int(num_significant_after),
            rejected_hypotheses=rejected.tolist()
        )
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray, 
                                     alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Benjamini-Hochberg (FDR) correction.
        
        Args:
            p_values: Array of p-values
            alpha: Significance level
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rejected hypotheses, adjusted p-values)
        """
        m = len(p_values)
        if m == 0:
            return np.array([]), np.array([])
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate adjusted p-values
        adjusted_p_values = np.zeros_like(sorted_p_values)
        
        # Work backwards through sorted p-values
        for i in range(m - 1, -1, -1):
            if i == m - 1:
                adjusted_p_values[i] = sorted_p_values[i]
            else:
                adjusted_p_values[i] = min(
                    adjusted_p_values[i + 1],
                    sorted_p_values[i] * m / (i + 1)
                )
        
        # Ensure adjusted p-values don't exceed 1
        adjusted_p_values = np.minimum(adjusted_p_values, 1.0)
        
        # Determine rejected hypotheses
        rejected_sorted = adjusted_p_values <= alpha
        
        # Map back to original order
        rejected = np.zeros(m, dtype=bool)
        final_adjusted_p_values = np.zeros(m)
        
        for i, orig_idx in enumerate(sorted_indices):
            rejected[orig_idx] = rejected_sorted[i]
            final_adjusted_p_values[orig_idx] = adjusted_p_values[i]
        
        return rejected, final_adjusted_p_values
    
    def _bonferroni_correction(self, p_values: np.ndarray, 
                              alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Bonferroni correction.
        
        Args:
            p_values: Array of p-values
            alpha: Significance level
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rejected hypotheses, adjusted p-values)
        """
        m = len(p_values)
        adjusted_p_values = np.minimum(p_values * m, 1.0)
        rejected = adjusted_p_values <= alpha
        
        return rejected, adjusted_p_values
    
    def _holm_correction(self, p_values: np.ndarray, 
                        alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Holm correction.
        
        Args:
            p_values: Array of p-values
            alpha: Significance level
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rejected hypotheses, adjusted p-values)
        """
        m = len(p_values)
        if m == 0:
            return np.array([]), np.array([])
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate adjusted p-values
        adjusted_p_values = np.zeros_like(sorted_p_values)
        rejected_sorted = np.zeros(m, dtype=bool)
        
        for i in range(m):
            adjusted_p_values[i] = min(
                1.0,
                sorted_p_values[i] * (m - i)
            )
            
            # For Holm, reject if this and all previous are significant
            if adjusted_p_values[i] <= alpha:
                rejected_sorted[i] = True
            else:
                # Stop rejecting once we find a non-significant result
                break
        
        # Map back to original order
        rejected = np.zeros(m, dtype=bool)
        final_adjusted_p_values = np.zeros(m)
        
        for i, orig_idx in enumerate(sorted_indices):
            rejected[orig_idx] = rejected_sorted[i]
            final_adjusted_p_values[orig_idx] = adjusted_p_values[i]
        
        return rejected, final_adjusted_p_values
    
    def get_interpretation(self, comparison: ComparisonResult) -> Dict[str, str]:
        """Get human-readable interpretation of comparison results.
        
        Args:
            comparison: Comparison result to interpret
            
        Returns:
            Dict[str, str]: Interpretation dictionary
        """
        interpretation = {}
        
        # Significance interpretation
        if comparison.is_significant:
            if comparison.adjusted_p_value is not None:
                interpretation['significance'] = f"Statistically significant (adjusted p = {comparison.adjusted_p_value:.4f})"
            else:
                interpretation['significance'] = f"Statistically significant (p = {comparison.p_value:.4f})"
        else:
            interpretation['significance'] = "Not statistically significant"
        
        # Effect size interpretation
        if comparison.effect_size is not None:
            if comparison.effect_size_method in ['cohens_d', 'hedges_g']:
                if abs(comparison.effect_size) < 0.2:
                    effect_magnitude = "negligible"
                elif abs(comparison.effect_size) < 0.5:
                    effect_magnitude = "small"
                elif abs(comparison.effect_size) < 0.8:
                    effect_magnitude = "medium"
                else:
                    effect_magnitude = "large"
            elif comparison.effect_size_method == 'cliffs_delta':
                if abs(comparison.effect_size) < 0.147:
                    effect_magnitude = "negligible"
                elif abs(comparison.effect_size) < 0.33:
                    effect_magnitude = "small"
                elif abs(comparison.effect_size) < 0.474:
                    effect_magnitude = "medium"
                else:
                    effect_magnitude = "large"
            else:
                effect_magnitude = "unknown"
            
            interpretation['effect_size'] = f"{effect_magnitude.capitalize()} effect size ({comparison.effect_size:.3f})"
        
        # Practical interpretation
        if comparison.difference > 0:
            better_model = comparison.model_b_id
            worse_model = comparison.model_a_id
        else:
            better_model = comparison.model_a_id
            worse_model = comparison.model_b_id
        
        interpretation['practical'] = f"{better_model} performs better than {worse_model} on {comparison.metric_name}"
        
        return interpretation