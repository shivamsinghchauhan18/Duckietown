#!/usr/bin/env python3
"""
📊 STATISTICAL ANALYZER EXAMPLE 📊
Example usage of the StatisticalAnalyzer class

This example demonstrates confidence interval calculations, bootstrap resampling,
significance testing, and multiple comparison correction.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.statistical_analyzer import (
    StatisticalAnalyzer, SignificanceTest, EffectSizeMethod
)

def main():
    """Demonstrate StatisticalAnalyzer functionality."""
    print("🔬 Statistical Analyzer Example")
    print("=" * 50)
    
    # Initialize analyzer
    config = {
        'confidence_level': 0.95,
        'bootstrap_resamples': 5000,
        'alpha': 0.05,
        'random_seed': 42
    }
    analyzer = StatisticalAnalyzer(config)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Model performance data (success rates)
    model_a_success = np.random.binomial(1, 0.75, 100).astype(float)  # 75% success rate
    model_b_success = np.random.binomial(1, 0.85, 100).astype(float)  # 85% success rate
    model_c_success = np.random.binomial(1, 0.80, 100).astype(float)  # 80% success rate
    
    # Model performance data (continuous metric - reward)
    model_a_reward = np.random.normal(0.7, 0.1, 100)
    model_b_reward = np.random.normal(0.8, 0.1, 100)
    model_c_reward = np.random.normal(0.75, 0.1, 100)
    
    print("\n📈 1. CONFIDENCE INTERVALS")
    print("-" * 30)
    
    # Calculate confidence intervals for success rates
    ci_a_success = analyzer.compute_confidence_intervals(model_a_success, method="wilson")
    ci_b_success = analyzer.compute_confidence_intervals(model_b_success, method="wilson")
    
    print(f"Model A Success Rate: {np.mean(model_a_success):.3f}")
    print(f"  95% CI: [{ci_a_success.lower:.3f}, {ci_a_success.upper:.3f}] ({ci_a_success.method})")
    
    print(f"Model B Success Rate: {np.mean(model_b_success):.3f}")
    print(f"  95% CI: [{ci_b_success.lower:.3f}, {ci_b_success.upper:.3f}] ({ci_b_success.method})")
    
    # Calculate confidence intervals for rewards
    ci_a_reward = analyzer.compute_confidence_intervals(model_a_reward, method="bootstrap")
    ci_b_reward = analyzer.compute_confidence_intervals(model_b_reward, method="bootstrap")
    
    print(f"\nModel A Reward: {np.mean(model_a_reward):.3f}")
    print(f"  95% CI: [{ci_a_reward.lower:.3f}, {ci_a_reward.upper:.3f}] ({ci_a_reward.method})")
    
    print(f"Model B Reward: {np.mean(model_b_reward):.3f}")
    print(f"  95% CI: [{ci_b_reward.lower:.3f}, {ci_b_reward.upper:.3f}] ({ci_b_reward.method})")
    
    print("\n🔄 2. BOOTSTRAP RESAMPLING")
    print("-" * 30)
    
    # Demonstrate bootstrap mean estimation
    bootstrap_result = analyzer.bootstrap_mean_estimate(model_a_reward, n_resamples=1000)
    
    print(f"Original Mean: {bootstrap_result.original_statistic:.4f}")
    print(f"Bootstrap Mean: {bootstrap_result.bootstrap_mean:.4f}")
    print(f"Bootstrap Std: {bootstrap_result.bootstrap_std:.4f}")
    print(f"Bootstrap CI: [{bootstrap_result.confidence_interval.lower:.4f}, {bootstrap_result.confidence_interval.upper:.4f}]")
    print(f"Number of Resamples: {bootstrap_result.n_resamples}")
    
    print("\n🔍 3. SIGNIFICANCE TESTING")
    print("-" * 30)
    
    # Compare models using different statistical tests
    
    # Paired t-test (assuming paired data)
    comparison_ttest = analyzer.compare_models(
        model_a_reward, model_b_reward,
        "Model_A", "Model_B", "reward",
        test_method=SignificanceTest.PAIRED_T_TEST,
        effect_size_method=EffectSizeMethod.COHENS_D
    )
    
    print("Paired T-Test Results:")
    print(f"  Model A Mean: {comparison_ttest.model_a_mean:.4f}")
    print(f"  Model B Mean: {comparison_ttest.model_b_mean:.4f}")
    print(f"  Difference: {comparison_ttest.difference:.4f}")
    print(f"  P-value: {comparison_ttest.p_value:.6f}")
    print(f"  Significant: {comparison_ttest.is_significant}")
    print(f"  Effect Size (Cohen's d): {comparison_ttest.effect_size:.4f}")
    
    # Wilcoxon signed-rank test
    comparison_wilcoxon = analyzer.compare_models(
        model_a_reward, model_b_reward,
        "Model_A", "Model_B", "reward",
        test_method=SignificanceTest.WILCOXON,
        effect_size_method=EffectSizeMethod.CLIFFS_DELTA
    )
    
    print(f"\nWilcoxon Test Results:")
    print(f"  P-value: {comparison_wilcoxon.p_value:.6f}")
    print(f"  Significant: {comparison_wilcoxon.is_significant}")
    print(f"  Effect Size (Cliff's δ): {comparison_wilcoxon.effect_size:.4f}")
    
    # Bootstrap significance test
    comparison_bootstrap = analyzer.compare_models(
        model_a_reward, model_b_reward,
        "Model_A", "Model_B", "reward",
        test_method=SignificanceTest.BOOTSTRAP,
        effect_size_method=EffectSizeMethod.HEDGES_G
    )
    
    print(f"\nBootstrap Test Results:")
    print(f"  P-value: {comparison_bootstrap.p_value:.6f}")
    print(f"  Significant: {comparison_bootstrap.is_significant}")
    print(f"  Effect Size (Hedges' g): {comparison_bootstrap.effect_size:.4f}")
    
    print("\n📊 4. MULTIPLE COMPARISON CORRECTION")
    print("-" * 40)
    
    # Create multiple comparisons
    comparisons = []
    
    # Compare all pairs of models
    model_pairs = [
        (model_a_reward, model_b_reward, "Model_A", "Model_B"),
        (model_a_reward, model_c_reward, "Model_A", "Model_C"),
        (model_b_reward, model_c_reward, "Model_B", "Model_C"),
        (model_a_success, model_b_success, "Model_A", "Model_B"),
        (model_a_success, model_c_success, "Model_A", "Model_C"),
        (model_b_success, model_c_success, "Model_B", "Model_C")
    ]
    
    for i, (data_a, data_b, id_a, id_b) in enumerate(model_pairs):
        metric_name = "reward" if i < 3 else "success_rate"
        comparison = analyzer.compare_models(
            data_a, data_b, id_a, id_b, metric_name,
            test_method=SignificanceTest.PAIRED_T_TEST
        )
        comparisons.append(comparison)
    
    print("Original P-values:")
    for i, comp in enumerate(comparisons):
        print(f"  {comp.model_a_id} vs {comp.model_b_id} ({comp.metric_name}): p = {comp.p_value:.6f}")
    
    # Apply Benjamini-Hochberg correction
    bh_result = analyzer.correct_multiple_comparisons(
        comparisons, method="benjamini_hochberg"
    )
    
    print(f"\nBenjamini-Hochberg Correction:")
    print(f"  Significant before correction: {bh_result.num_significant_before}")
    print(f"  Significant after correction: {bh_result.num_significant_after}")
    
    print("\nAdjusted P-values:")
    for comp in bh_result.comparisons:
        status = "✓" if comp.is_significant else "✗"
        print(f"  {comp.model_a_id} vs {comp.model_b_id} ({comp.metric_name}): "
              f"p_adj = {comp.adjusted_p_value:.6f} {status}")
    
    # Apply Bonferroni correction for comparison
    bonf_result = analyzer.correct_multiple_comparisons(
        comparisons, method="bonferroni"
    )
    
    print(f"\nBonferroni Correction:")
    print(f"  Significant after correction: {bonf_result.num_significant_after}")
    
    print("\n🎯 5. RESULT INTERPRETATION")
    print("-" * 30)
    
    # Get interpretation for the most significant comparison
    best_comparison = min(bh_result.comparisons, key=lambda x: x.adjusted_p_value)
    interpretation = analyzer.get_interpretation(best_comparison)
    
    print(f"Best Comparison: {best_comparison.model_a_id} vs {best_comparison.model_b_id}")
    print(f"Metric: {best_comparison.metric_name}")
    print(f"Statistical Significance: {interpretation['significance']}")
    print(f"Effect Size: {interpretation['effect_size']}")
    print(f"Practical Interpretation: {interpretation['practical']}")
    
    print("\n✅ Statistical Analysis Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()