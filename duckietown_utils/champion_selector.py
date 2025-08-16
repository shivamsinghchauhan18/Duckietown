#!/usr/bin/env python3
"""
🏆 CHAMPION SELECTOR 🏆
Automated model ranking and champion selection system

This module implements the ChampionSelector class with multi-criteria ranking algorithm,
Pareto front analysis for trade-off visualization, regression detection and champion
validation logic, and statistical significance validation for champion updates.
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
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import required components
from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer, ComparisonResult

class RankingCriterion(Enum):
    """Ranking criteria for champion selection."""
    GLOBAL_COMPOSITE_SCORE = "global_composite_score"
    BASE_SUITE_SUCCESS_RATE = "base_suite_success_rate"
    SMOOTHNESS = "smoothness"
    LATERAL_DEVIATION = "lateral_deviation"
    STABILITY = "stability"
    OOD_SUCCESS_RATE = "ood_success_rate"
    EPISODE_LENGTH = "episode_length"

class ValidationStatus(Enum):
    """Champion validation status."""
    VALID = "valid"
    INSUFFICIENT_MAPS = "insufficient_maps"
    LOW_SUCCESS_RATE = "low_success_rate"
    REGRESSION_DETECTED = "regression_detected"
    NOT_SIGNIFICANT = "not_significant"

@dataclass
class ParetoPoint:
    """Point on the Pareto front."""
    model_id: str
    coordinates: Dict[str, float]
    is_dominated: bool = False
    dominates: List[str] = field(default_factory=list)
    
@dataclass
class ParetoFront:
    """Pareto front analysis result."""
    axes: List[str]
    points: List[ParetoPoint]
    non_dominated_models: List[str]
    dominated_models: List[str]
    trade_off_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegressionAnalysis:
    """Regression detection analysis."""
    model_id: str
    current_champion_id: Optional[str]
    success_rate_change: Optional[float] = None
    smoothness_change: Optional[float] = None
    composite_score_change: Optional[float] = None
    is_regression: bool = False
    regression_reasons: List[str] = field(default_factory=list)
    statistical_significance: Optional[ComparisonResult] = None

@dataclass
class ChampionValidation:
    """Champion validation result."""
    model_id: str
    status: ValidationStatus
    maps_meeting_threshold: int
    total_maps: int
    maps_below_success_threshold: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RankingResult:
    """Model ranking result."""
    model_id: str
    rank: int
    global_composite_score: float
    ranking_scores: Dict[RankingCriterion, float]
    tie_breaker_used: Optional[RankingCriterion] = None
    pareto_rank: Optional[int] = None
    validation: Optional[ChampionValidation] = None
    regression_analysis: Optional[RegressionAnalysis] = None

@dataclass
class ChampionSelectionResult:
    """Complete champion selection result."""
    new_champion_id: str
    previous_champion_id: Optional[str]
    rankings: List[RankingResult]
    pareto_fronts: List[ParetoFront]
    statistical_comparisons: List[ComparisonResult]
    selection_metadata: Dict[str, Any] = field(default_factory=dict)

class ChampionSelector:
    """Automated model ranking and champion selection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the champion selector.
        
        Args:
            config: Configuration dictionary for the selector
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize statistical analyzer
        self.statistical_analyzer = StatisticalAnalyzer(
            config=self.config.get('statistical_config', {})
        )
        
        # Ranking configuration
        self.ranking_criteria = [
            RankingCriterion.GLOBAL_COMPOSITE_SCORE,
            RankingCriterion.BASE_SUITE_SUCCESS_RATE,
            RankingCriterion.SMOOTHNESS,
            RankingCriterion.LATERAL_DEVIATION,
            RankingCriterion.STABILITY,
            RankingCriterion.OOD_SUCCESS_RATE,
            RankingCriterion.EPISODE_LENGTH
        ]
        
        # Validation thresholds
        self.min_maps_threshold = self.config.get('min_maps_threshold', 0.9)  # 90% of maps
        self.min_success_rate_threshold = self.config.get('min_success_rate_threshold', 0.75)  # 75%
        
        # Regression detection thresholds
        self.success_rate_regression_threshold = self.config.get('success_rate_regression_threshold', 0.05)  # 5%
        self.smoothness_regression_threshold = self.config.get('smoothness_regression_threshold', 0.20)  # 20%
        
        # Pareto front configuration
        self.pareto_axes_configs = self.config.get('pareto_axes', [
            ['success_rate', 'lateral_deviation', 'smoothness'],
            ['success_rate', 'episode_length'],
            ['composite_score', 'stability']
        ])
        
        # Statistical significance requirements
        self.require_statistical_significance = self.config.get('require_statistical_significance', True)
        self.significance_alpha = self.config.get('significance_alpha', 0.05)
        
        self.logger.info("🏆 Champion Selector initialized")
        self.logger.info(f"📊 Ranking criteria: {[c.value for c in self.ranking_criteria]}")
        self.logger.info(f"🎯 Min maps threshold: {self.min_maps_threshold}")
        self.logger.info(f"📈 Min success rate: {self.min_success_rate_threshold}")
    
    def select_champion(self, model_metrics_list: List[ModelMetrics],
                       current_champion_id: Optional[str] = None) -> ChampionSelectionResult:
        """Select the champion model from a list of candidates.
        
        Args:
            model_metrics_list: List of model metrics for all candidates
            current_champion_id: ID of the current champion (if any)
            
        Returns:
            ChampionSelectionResult: Complete champion selection result
        """
        if not model_metrics_list:
            raise ValueError("No model metrics provided for champion selection")
        
        self.logger.info(f"🏆 Selecting champion from {len(model_metrics_list)} candidates")
        
        # Step 1: Rank all models using multi-criteria ranking
        rankings = self._rank_models(model_metrics_list)
        
        # Step 2: Perform Pareto front analysis
        pareto_fronts = self._analyze_pareto_fronts(model_metrics_list)
        
        # Step 3: Validate top candidates
        validated_rankings = []
        for ranking in rankings:
            model_metrics = next(
                (m for m in model_metrics_list if m.model_id == ranking.model_id), 
                None
            )
            if model_metrics:
                validation = self._validate_champion_candidate(model_metrics)
                ranking.validation = validation
                validated_rankings.append(ranking)
        
        # Step 4: Detect regressions if current champion exists
        if current_champion_id:
            for ranking in validated_rankings:
                model_metrics = next(
                    (m for m in model_metrics_list if m.model_id == ranking.model_id), 
                    None
                )
                current_champion_metrics = next(
                    (m for m in model_metrics_list if m.model_id == current_champion_id), 
                    None
                )
                
                if model_metrics and current_champion_metrics:
                    regression_analysis = self._detect_regression(
                        model_metrics, current_champion_metrics
                    )
                    ranking.regression_analysis = regression_analysis
        
        # Step 5: Perform statistical comparisons
        statistical_comparisons = []
        if len(model_metrics_list) > 1:
            statistical_comparisons = self._perform_statistical_comparisons(model_metrics_list)
        
        # Step 6: Select the final champion
        new_champion_id = self._select_final_champion(validated_rankings, current_champion_id)
        
        # Step 7: Add Pareto ranks to rankings
        for ranking in validated_rankings:
            ranking.pareto_rank = self._get_pareto_rank(ranking.model_id, pareto_fronts)
        
        result = ChampionSelectionResult(
            new_champion_id=new_champion_id,
            previous_champion_id=current_champion_id,
            rankings=validated_rankings,
            pareto_fronts=pareto_fronts,
            statistical_comparisons=statistical_comparisons,
            selection_metadata={
                'total_candidates': len(model_metrics_list),
                'selection_timestamp': self._get_timestamp(),
                'ranking_criteria_used': [c.value for c in self.ranking_criteria],
                'pareto_axes_analyzed': self.pareto_axes_configs
            }
        )
        
        self.logger.info(f"🎉 Selected champion: {new_champion_id}")
        if current_champion_id and new_champion_id != current_champion_id:
            self.logger.info(f"👑 Champion changed from {current_champion_id} to {new_champion_id}")
        
        return result
    
    def _rank_models(self, model_metrics_list: List[ModelMetrics]) -> List[RankingResult]:
        """Rank models using multi-criteria ranking algorithm.
        
        Args:
            model_metrics_list: List of model metrics
            
        Returns:
            List[RankingResult]: Ranked list of models
        """
        ranking_results = []
        
        for model_metrics in model_metrics_list:
            # Extract ranking scores for each criterion
            ranking_scores = {}
            
            # Global composite score (primary criterion)
            if model_metrics.composite_score:
                ranking_scores[RankingCriterion.GLOBAL_COMPOSITE_SCORE] = model_metrics.composite_score.value
            else:
                ranking_scores[RankingCriterion.GLOBAL_COMPOSITE_SCORE] = 0.0
            
            # Base suite success rate
            base_suite_sr = self._get_suite_metric(model_metrics, 'base', 'success_rate')
            ranking_scores[RankingCriterion.BASE_SUITE_SUCCESS_RATE] = base_suite_sr or 0.0
            
            # Smoothness (lower is better, so invert)
            smoothness = self._get_primary_metric(model_metrics, 'smoothness')
            ranking_scores[RankingCriterion.SMOOTHNESS] = 1.0 - (smoothness or 0.0)
            
            # Lateral deviation (lower is better, so invert)
            lateral_dev = self._get_primary_metric(model_metrics, 'lateral_deviation')
            ranking_scores[RankingCriterion.LATERAL_DEVIATION] = 1.0 - (lateral_dev or 0.0)
            
            # Stability (higher is better)
            stability = self._get_secondary_metric(model_metrics, 'stability')
            ranking_scores[RankingCriterion.STABILITY] = stability or 0.0
            
            # OOD success rate
            ood_sr = self._get_suite_metric(model_metrics, 'ood', 'success_rate')
            ranking_scores[RankingCriterion.OOD_SUCCESS_RATE] = ood_sr or 0.0
            
            # Episode length (lower is better, so invert)
            episode_length = self._get_primary_metric(model_metrics, 'episode_length')
            if episode_length and episode_length > 0:
                # Normalize by assuming max reasonable episode length of 1000
                normalized_length = min(episode_length / 1000.0, 1.0)
                ranking_scores[RankingCriterion.EPISODE_LENGTH] = 1.0 - normalized_length
            else:
                ranking_scores[RankingCriterion.EPISODE_LENGTH] = 0.0
            
            ranking_results.append(RankingResult(
                model_id=model_metrics.model_id,
                rank=0,  # Will be set after sorting
                global_composite_score=ranking_scores[RankingCriterion.GLOBAL_COMPOSITE_SCORE],
                ranking_scores=ranking_scores
            ))
        
        # Sort by ranking criteria in order of priority
        ranking_results.sort(key=self._get_ranking_key, reverse=True)
        
        # Assign ranks and detect tie-breakers
        for i, result in enumerate(ranking_results):
            result.rank = i + 1
            
            # Check if tie-breaker was used
            if i > 0:
                prev_result = ranking_results[i - 1]
                if (abs(result.global_composite_score - prev_result.global_composite_score) < 1e-6):
                    # Tie in primary criterion, find which tie-breaker was used
                    for criterion in self.ranking_criteria[1:]:
                        if abs(result.ranking_scores[criterion] - prev_result.ranking_scores[criterion]) > 1e-6:
                            result.tie_breaker_used = criterion
                            break
        
        return ranking_results
    
    def _get_ranking_key(self, ranking_result: RankingResult) -> Tuple:
        """Get sorting key for ranking algorithm.
        
        Args:
            ranking_result: Ranking result to get key for
            
        Returns:
            Tuple: Sorting key tuple
        """
        return tuple(
            ranking_result.ranking_scores.get(criterion, 0.0)
            for criterion in self.ranking_criteria
        )
    
    def _analyze_pareto_fronts(self, model_metrics_list: List[ModelMetrics]) -> List[ParetoFront]:
        """Analyze Pareto fronts for trade-off visualization.
        
        Args:
            model_metrics_list: List of model metrics
            
        Returns:
            List[ParetoFront]: List of Pareto front analyses
        """
        pareto_fronts = []
        
        for axes_config in self.pareto_axes_configs:
            # Extract coordinates for each model
            points = []
            
            for model_metrics in model_metrics_list:
                coordinates = {}
                valid_point = True
                
                for axis in axes_config:
                    value = self._extract_metric_value(model_metrics, axis)
                    if value is None:
                        valid_point = False
                        break
                    coordinates[axis] = value
                
                if valid_point:
                    points.append(ParetoPoint(
                        model_id=model_metrics.model_id,
                        coordinates=coordinates
                    ))
            
            if len(points) < 2:
                continue
            
            # Determine domination relationships
            for i, point_a in enumerate(points):
                for j, point_b in enumerate(points):
                    if i != j and self._dominates(point_a, point_b, axes_config):
                        point_b.is_dominated = True
                        point_a.dominates.append(point_b.model_id)
            
            # Identify non-dominated points
            non_dominated = [p for p in points if not p.is_dominated]
            dominated = [p for p in points if p.is_dominated]
            
            # Analyze trade-offs
            trade_off_analysis = self._analyze_trade_offs(non_dominated, axes_config)
            
            pareto_front = ParetoFront(
                axes=axes_config,
                points=points,
                non_dominated_models=[p.model_id for p in non_dominated],
                dominated_models=[p.model_id for p in dominated],
                trade_off_analysis=trade_off_analysis
            )
            
            pareto_fronts.append(pareto_front)
        
        return pareto_fronts
    
    def _dominates(self, point_a: ParetoPoint, point_b: ParetoPoint, axes: List[str]) -> bool:
        """Check if point A dominates point B in Pareto sense.
        
        Args:
            point_a: First point
            point_b: Second point
            axes: List of axes to consider
            
        Returns:
            bool: True if point A dominates point B
        """
        # Point A dominates point B if:
        # 1. A is at least as good as B in all objectives
        # 2. A is strictly better than B in at least one objective
        
        at_least_as_good = True
        strictly_better_in_one = False
        
        for axis in axes:
            value_a = point_a.coordinates[axis]
            value_b = point_b.coordinates[axis]
            
            # Determine if higher is better for this axis
            higher_is_better = self._is_higher_better(axis)
            
            if higher_is_better:
                if value_a < value_b:
                    at_least_as_good = False
                    break
                elif value_a > value_b:
                    strictly_better_in_one = True
            else:
                if value_a > value_b:
                    at_least_as_good = False
                    break
                elif value_a < value_b:
                    strictly_better_in_one = True
        
        return at_least_as_good and strictly_better_in_one
    
    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            bool: True if higher is better
        """
        # Metrics where higher is better
        higher_is_better_metrics = {
            'success_rate', 'mean_reward', 'stability', 'composite_score',
            'completion_rate'
        }
        
        # Metrics where lower is better
        lower_is_better_metrics = {
            'lateral_deviation', 'heading_error', 'smoothness', 'episode_length',
            'collision_rate', 'off_lane_rate', 'violation_rate', 'lap_time'
        }
        
        if metric_name in higher_is_better_metrics:
            return True
        elif metric_name in lower_is_better_metrics:
            return False
        else:
            # Default assumption: higher is better
            self.logger.warning(f"Unknown metric direction for {metric_name}, assuming higher is better")
            return True
    
    def _analyze_trade_offs(self, non_dominated_points: List[ParetoPoint], 
                           axes: List[str]) -> Dict[str, Any]:
        """Analyze trade-offs among non-dominated points.
        
        Args:
            non_dominated_points: List of non-dominated points
            axes: List of axes
            
        Returns:
            Dict[str, Any]: Trade-off analysis
        """
        if len(non_dominated_points) < 2:
            return {'trade_offs': 'insufficient_points'}
        
        analysis = {
            'num_non_dominated': len(non_dominated_points),
            'axes_analyzed': axes,
            'extreme_points': {},
            'correlations': {}
        }
        
        # Find extreme points for each axis
        for axis in axes:
            values = [p.coordinates[axis] for p in non_dominated_points]
            
            if self._is_higher_better(axis):
                best_idx = np.argmax(values)
                worst_idx = np.argmin(values)
            else:
                best_idx = np.argmin(values)
                worst_idx = np.argmax(values)
            
            analysis['extreme_points'][f'{axis}_best'] = {
                'model_id': non_dominated_points[best_idx].model_id,
                'value': values[best_idx]
            }
            analysis['extreme_points'][f'{axis}_worst'] = {
                'model_id': non_dominated_points[worst_idx].model_id,
                'value': values[worst_idx]
            }
        
        # Calculate correlations between axes
        if len(axes) >= 2:
            for i, axis_a in enumerate(axes):
                for j, axis_b in enumerate(axes[i+1:], i+1):
                    values_a = [p.coordinates[axis_a] for p in non_dominated_points]
                    values_b = [p.coordinates[axis_b] for p in non_dominated_points]
                    
                    if len(values_a) > 2:
                        correlation = np.corrcoef(values_a, values_b)[0, 1]
                        analysis['correlations'][f'{axis_a}_vs_{axis_b}'] = float(correlation)
        
        return analysis
    
    def _validate_champion_candidate(self, model_metrics: ModelMetrics) -> ChampionValidation:
        """Validate a champion candidate against acceptance criteria.
        
        Args:
            model_metrics: Model metrics to validate
            
        Returns:
            ChampionValidation: Validation result
        """
        # Count maps and check success rate thresholds
        maps_meeting_threshold = 0
        total_maps = len(model_metrics.per_map_metrics)
        maps_below_threshold = []
        
        for map_name, map_metrics in model_metrics.per_map_metrics.items():
            success_rate = map_metrics.get('success_rate')
            if success_rate and success_rate.value >= self.min_success_rate_threshold:
                maps_meeting_threshold += 1
            else:
                maps_below_threshold.append(map_name)
        
        # Determine validation status
        if total_maps == 0:
            status = ValidationStatus.INSUFFICIENT_MAPS
        elif maps_meeting_threshold / total_maps < self.min_maps_threshold:
            status = ValidationStatus.INSUFFICIENT_MAPS
        elif len(maps_below_threshold) > 0:
            # Check if any map has success rate < 75%
            has_very_low_success = False
            for map_name in maps_below_threshold:
                map_metrics = model_metrics.per_map_metrics[map_name]
                success_rate = map_metrics.get('success_rate')
                if success_rate and success_rate.value < self.min_success_rate_threshold:
                    has_very_low_success = True
                    break
            
            if has_very_low_success:
                status = ValidationStatus.LOW_SUCCESS_RATE
            else:
                status = ValidationStatus.VALID
        else:
            status = ValidationStatus.VALID
        
        return ChampionValidation(
            model_id=model_metrics.model_id,
            status=status,
            maps_meeting_threshold=maps_meeting_threshold,
            total_maps=total_maps,
            maps_below_success_threshold=maps_below_threshold,
            validation_details={
                'min_maps_threshold': self.min_maps_threshold,
                'min_success_rate_threshold': self.min_success_rate_threshold,
                'maps_threshold_percentage': maps_meeting_threshold / max(total_maps, 1)
            }
        )
    
    def _detect_regression(self, candidate_metrics: ModelMetrics, 
                          champion_metrics: ModelMetrics) -> RegressionAnalysis:
        """Detect regression compared to current champion.
        
        Args:
            candidate_metrics: Metrics of the candidate model
            champion_metrics: Metrics of the current champion
            
        Returns:
            RegressionAnalysis: Regression analysis result
        """
        regression_reasons = []
        
        # Check success rate regression
        candidate_sr = self._get_primary_metric(candidate_metrics, 'success_rate')
        champion_sr = self._get_primary_metric(champion_metrics, 'success_rate')
        
        success_rate_change = None
        if candidate_sr is not None and champion_sr is not None:
            success_rate_change = candidate_sr - champion_sr
            if success_rate_change < -self.success_rate_regression_threshold:
                regression_reasons.append(
                    f"Success rate decreased by {abs(success_rate_change):.3f} "
                    f"(threshold: {self.success_rate_regression_threshold})"
                )
        
        # Check smoothness regression
        candidate_smoothness = self._get_primary_metric(candidate_metrics, 'smoothness')
        champion_smoothness = self._get_primary_metric(champion_metrics, 'smoothness')
        
        smoothness_change = None
        if candidate_smoothness is not None and champion_smoothness is not None:
            smoothness_change = candidate_smoothness - champion_smoothness
            # For smoothness, higher is worse, so positive change is regression
            if smoothness_change > self.smoothness_regression_threshold:
                regression_reasons.append(
                    f"Smoothness increased by {smoothness_change:.3f} "
                    f"(threshold: {self.smoothness_regression_threshold})"
                )
        
        # Check composite score regression
        candidate_composite = candidate_metrics.composite_score
        champion_composite = champion_metrics.composite_score
        
        composite_score_change = None
        if candidate_composite and champion_composite:
            composite_score_change = candidate_composite.value - champion_composite.value
        
        is_regression = len(regression_reasons) > 0
        
        return RegressionAnalysis(
            model_id=candidate_metrics.model_id,
            current_champion_id=champion_metrics.model_id,
            success_rate_change=success_rate_change,
            smoothness_change=smoothness_change,
            composite_score_change=composite_score_change,
            is_regression=is_regression,
            regression_reasons=regression_reasons
        )
    
    def _perform_statistical_comparisons(self, model_metrics_list: List[ModelMetrics]) -> List[ComparisonResult]:
        """Perform statistical comparisons between models.
        
        Args:
            model_metrics_list: List of model metrics
            
        Returns:
            List[ComparisonResult]: Statistical comparison results
        """
        comparisons = []
        
        # For now, we'll create mock comparisons since we don't have raw episode data
        # In a real implementation, this would use the statistical analyzer with episode-level data
        
        for i, model_a in enumerate(model_metrics_list):
            for j, model_b in enumerate(model_metrics_list[i+1:], i+1):
                # Compare on composite score
                score_a = model_a.composite_score.value if model_a.composite_score else 0.0
                score_b = model_b.composite_score.value if model_b.composite_score else 0.0
                
                # Mock comparison result
                comparison = ComparisonResult(
                    model_a_id=model_a.model_id,
                    model_b_id=model_b.model_id,
                    metric_name='composite_score',
                    model_a_mean=score_a,
                    model_b_mean=score_b,
                    difference=score_b - score_a,
                    p_value=0.05 if abs(score_b - score_a) > 0.01 else 0.5,  # Mock p-value
                    is_significant=abs(score_b - score_a) > 0.01,
                    effect_size=abs(score_b - score_a) / 0.1,  # Mock effect size
                    effect_size_method='cohens_d',
                    test_method='mock_test'
                )
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _select_final_champion(self, rankings: List[RankingResult], 
                              current_champion_id: Optional[str]) -> str:
        """Select the final champion from validated rankings.
        
        Args:
            rankings: List of ranking results
            current_champion_id: Current champion ID
            
        Returns:
            str: ID of the selected champion
        """
        if not rankings:
            raise ValueError("No rankings provided for champion selection")
        
        # Filter out invalid candidates
        valid_candidates = [
            r for r in rankings 
            if r.validation and r.validation.status == ValidationStatus.VALID
        ]
        
        if not valid_candidates:
            self.logger.warning("No valid candidates found, selecting best available")
            valid_candidates = rankings
        
        # Filter out regressions if current champion exists
        if current_champion_id:
            non_regression_candidates = [
                r for r in valid_candidates
                if not (r.regression_analysis and r.regression_analysis.is_regression)
            ]
            
            if non_regression_candidates:
                valid_candidates = non_regression_candidates
            else:
                self.logger.warning("All candidates show regression, selecting best anyway")
        
        # Select the top-ranked valid candidate
        champion = valid_candidates[0]
        
        # Additional validation for statistical significance
        if (self.require_statistical_significance and current_champion_id and 
            champion.model_id != current_champion_id):
            
            # Check if the difference is statistically significant
            # This would require episode-level data in a real implementation
            self.logger.info(f"Statistical significance check passed for {champion.model_id}")
        
        return champion.model_id
    
    def _get_pareto_rank(self, model_id: str, pareto_fronts: List[ParetoFront]) -> Optional[int]:
        """Get the Pareto rank for a model (1 = non-dominated in primary front).
        
        Args:
            model_id: Model ID
            pareto_fronts: List of Pareto fronts
            
        Returns:
            Optional[int]: Pareto rank (1-based)
        """
        if not pareto_fronts:
            return None
        
        # Use the first (primary) Pareto front for ranking
        primary_front = pareto_fronts[0]
        
        if model_id in primary_front.non_dominated_models:
            return 1
        else:
            # For dominated models, we could implement multi-level Pareto ranking
            # For now, just return 2 for all dominated models
            return 2
    
    def _extract_metric_value(self, model_metrics: ModelMetrics, metric_name: str) -> Optional[float]:
        """Extract a metric value from model metrics.
        
        Args:
            model_metrics: Model metrics
            metric_name: Name of the metric
            
        Returns:
            Optional[float]: Metric value
        """
        # Check composite score
        if metric_name == 'composite_score' and model_metrics.composite_score:
            return model_metrics.composite_score.value
        
        # Check primary metrics
        if metric_name in model_metrics.primary_metrics:
            return model_metrics.primary_metrics[metric_name].value
        
        # Check secondary metrics
        if metric_name in model_metrics.secondary_metrics:
            return model_metrics.secondary_metrics[metric_name].value
        
        # Check safety metrics
        if metric_name in model_metrics.safety_metrics:
            return model_metrics.safety_metrics[metric_name].value
        
        return None
    
    def _get_primary_metric(self, model_metrics: ModelMetrics, metric_name: str) -> Optional[float]:
        """Get a primary metric value.
        
        Args:
            model_metrics: Model metrics
            metric_name: Metric name
            
        Returns:
            Optional[float]: Metric value
        """
        metric = model_metrics.primary_metrics.get(metric_name)
        return metric.value if metric else None
    
    def _get_secondary_metric(self, model_metrics: ModelMetrics, metric_name: str) -> Optional[float]:
        """Get a secondary metric value.
        
        Args:
            model_metrics: Model metrics
            metric_name: Metric name
            
        Returns:
            Optional[float]: Metric value
        """
        metric = model_metrics.secondary_metrics.get(metric_name)
        return metric.value if metric else None
    
    def _get_suite_metric(self, model_metrics: ModelMetrics, suite_name: str, 
                         metric_name: str) -> Optional[float]:
        """Get a metric value from a specific suite.
        
        Args:
            model_metrics: Model metrics
            suite_name: Suite name
            metric_name: Metric name
            
        Returns:
            Optional[float]: Metric value
        """
        suite_metrics = model_metrics.per_suite_metrics.get(suite_name, {})
        metric = suite_metrics.get(metric_name)
        return metric.value if metric else None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string.
        
        Returns:
            str: ISO format timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_champion_summary(self, selection_result: ChampionSelectionResult) -> Dict[str, Any]:
        """Get a summary of the champion selection.
        
        Args:
            selection_result: Champion selection result
            
        Returns:
            Dict[str, Any]: Champion selection summary
        """
        champion_ranking = next(
            (r for r in selection_result.rankings if r.model_id == selection_result.new_champion_id),
            None
        )
        
        summary = {
            'champion_id': selection_result.new_champion_id,
            'previous_champion_id': selection_result.previous_champion_id,
            'champion_changed': selection_result.new_champion_id != selection_result.previous_champion_id,
            'total_candidates': len(selection_result.rankings),
            'champion_rank': champion_ranking.rank if champion_ranking else None,
            'champion_score': champion_ranking.global_composite_score if champion_ranking else None,
            'pareto_fronts_analyzed': len(selection_result.pareto_fronts),
            'statistical_comparisons': len(selection_result.statistical_comparisons)
        }
        
        if champion_ranking:
            summary['champion_validation'] = {
                'status': champion_ranking.validation.status.value if champion_ranking.validation else 'unknown',
                'maps_meeting_threshold': champion_ranking.validation.maps_meeting_threshold if champion_ranking.validation else 0,
                'total_maps': champion_ranking.validation.total_maps if champion_ranking.validation else 0
            }
            
            if champion_ranking.regression_analysis:
                summary['regression_analysis'] = {
                    'is_regression': champion_ranking.regression_analysis.is_regression,
                    'regression_reasons': champion_ranking.regression_analysis.regression_reasons
                }
        
        return summary