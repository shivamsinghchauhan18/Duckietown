#!/usr/bin/env python3
"""
📊 METRICS CALCULATOR 📊
Comprehensive metrics calculation for model evaluation

This module implements the MetricsCalculator class with all primary and secondary metrics,
composite score calculation with configurable weights, per-map and per-suite metric
normalization, and episode-level metric extraction and aggregation.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import suite manager for episode results
from duckietown_utils.suite_manager import EpisodeResult, SuiteResults

class MetricType(Enum):
    """Types of metrics for categorization."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    COMPOSITE = "composite"
    SAFETY = "safety"

class NormalizationScope(Enum):
    """Normalization scope options."""
    GLOBAL = "global"
    PER_MAP = "per_map"
    PER_SUITE = "per_suite"
    PER_MAP_SUITE = "per_map_suite"

@dataclass
class MetricDefinition:
    """Definition of a metric including calculation and normalization parameters."""
    name: str
    metric_type: MetricType
    description: str
    higher_is_better: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    normalization_method: str = "min_max"  # 'min_max', 'z_score', 'robust'
    
    def __post_init__(self):
        """Validate metric definition."""
        if self.normalization_method not in ['min_max', 'z_score', 'robust']:
            raise ValueError(f"Invalid normalization method: {self.normalization_method}")

@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = "wilson"  # 'wilson', 'bootstrap', 'normal'

@dataclass
class MetricResult:
    """Result of a metric calculation."""
    name: str
    value: float
    normalized_value: Optional[float] = None
    confidence_interval: Optional[ConfidenceInterval] = None
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompositeScoreConfig:
    """Configuration for composite score calculation."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'success_rate': 0.45,
        'mean_reward': 0.25,
        'episode_length': 0.10,
        'lateral_deviation': 0.08,
        'heading_error': 0.06,
        'smoothness': 0.06
    })
    normalization_scope: NormalizationScope = NormalizationScope.PER_MAP_SUITE
    include_safety_penalty: bool = True
    safety_penalty_weight: float = 0.2
    
    def __post_init__(self):
        """Validate composite score configuration."""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            warnings.warn(f"Composite score weights sum to {total_weight:.3f}, not 1.0")

@dataclass
class ModelMetrics:
    """Complete metrics for a model across all evaluations."""
    model_id: str
    primary_metrics: Dict[str, MetricResult]
    secondary_metrics: Dict[str, MetricResult]
    safety_metrics: Dict[str, MetricResult]
    composite_score: Optional[MetricResult] = None
    per_map_metrics: Dict[str, Dict[str, MetricResult]] = field(default_factory=dict)
    per_suite_metrics: Dict[str, Dict[str, MetricResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCalculator:
    """Comprehensive metrics calculator for model evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary for the calculator
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric definitions
        self.metric_definitions = self._initialize_metric_definitions()
        
        # Composite score configuration
        self.composite_config = CompositeScoreConfig(
            weights=self.config.get('composite_weights', {}),
            normalization_scope=NormalizationScope(
                self.config.get('normalization_scope', 'per_map_suite')
            ),
            include_safety_penalty=self.config.get('include_safety_penalty', True),
            safety_penalty_weight=self.config.get('safety_penalty_weight', 0.2)
        )
        
        # Confidence interval settings
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.bootstrap_samples = self.config.get('bootstrap_samples', 10000)
        
        # Normalization data storage
        self.normalization_stats: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("📊 Metrics Calculator initialized")
        self.logger.info(f"🎯 Composite weights: {dict(self.composite_config.weights)}")
        self.logger.info(f"📏 Normalization scope: {self.composite_config.normalization_scope.value}")
    
    def _initialize_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Initialize standard metric definitions."""
        definitions = {}
        
        # Primary metrics
        definitions['success_rate'] = MetricDefinition(
            name='success_rate',
            metric_type=MetricType.PRIMARY,
            description='Percentage of episodes completing without collision/off-lane',
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            normalization_method='min_max'
        )
        
        definitions['mean_reward'] = MetricDefinition(
            name='mean_reward',
            metric_type=MetricType.PRIMARY,
            description='Mean normalized reward across episodes',
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            normalization_method='min_max'
        )
        
        definitions['episode_length'] = MetricDefinition(
            name='episode_length',
            metric_type=MetricType.PRIMARY,
            description='Mean episode length in steps or lap time',
            higher_is_better=False,  # Shorter is better for efficiency
            min_value=0.0,
            normalization_method='min_max'
        )
        
        definitions['lateral_deviation'] = MetricDefinition(
            name='lateral_deviation',
            metric_type=MetricType.PRIMARY,
            description='Mean distance from lane center',
            higher_is_better=False,
            min_value=0.0,
            normalization_method='min_max'
        )
        
        definitions['heading_error'] = MetricDefinition(
            name='heading_error',
            metric_type=MetricType.PRIMARY,
            description='Mean angular deviation from desired heading',
            higher_is_better=False,
            min_value=0.0,
            normalization_method='min_max'
        )
        
        definitions['smoothness'] = MetricDefinition(
            name='smoothness',
            metric_type=MetricType.PRIMARY,
            description='Mean absolute steering changes (jerk)',
            higher_is_better=False,
            min_value=0.0,
            normalization_method='min_max'
        )
        
        # Secondary metrics
        definitions['stability'] = MetricDefinition(
            name='stability',
            metric_type=MetricType.SECONDARY,
            description='Reward consistency (μ/σ)',
            higher_is_better=True,
            min_value=0.0,
            normalization_method='min_max'
        )
        
        definitions['lap_time'] = MetricDefinition(
            name='lap_time',
            metric_type=MetricType.SECONDARY,
            description='Mean lap completion time',
            higher_is_better=False,
            min_value=0.0,
            normalization_method='min_max'
        )
        
        definitions['completion_rate'] = MetricDefinition(
            name='completion_rate',
            metric_type=MetricType.SECONDARY,
            description='Percentage of episodes reaching the end',
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            normalization_method='min_max'
        )
        
        # Safety metrics
        definitions['collision_rate'] = MetricDefinition(
            name='collision_rate',
            metric_type=MetricType.SAFETY,
            description='Percentage of episodes with collisions',
            higher_is_better=False,
            min_value=0.0,
            max_value=1.0,
            normalization_method='min_max'
        )
        
        definitions['off_lane_rate'] = MetricDefinition(
            name='off_lane_rate',
            metric_type=MetricType.SAFETY,
            description='Percentage of episodes with lane departures',
            higher_is_better=False,
            min_value=0.0,
            max_value=1.0,
            normalization_method='min_max'
        )
        
        definitions['violation_rate'] = MetricDefinition(
            name='violation_rate',
            metric_type=MetricType.SAFETY,
            description='Normalized traffic rule violations per episode',
            higher_is_better=False,
            min_value=0.0,
            normalization_method='min_max'
        )
        
        return definitions
    
    def calculate_episode_metrics(self, episode_result: EpisodeResult) -> Dict[str, float]:
        """Calculate metrics for a single episode.
        
        Args:
            episode_result: Results from a single episode
            
        Returns:
            Dict[str, float]: Dictionary of metric values
        """
        metrics = {}
        
        # Primary metrics
        metrics['success_rate'] = 1.0 if episode_result.success else 0.0
        metrics['mean_reward'] = episode_result.reward
        metrics['episode_length'] = float(episode_result.episode_length)
        metrics['lateral_deviation'] = episode_result.lateral_deviation
        metrics['heading_error'] = episode_result.heading_error
        metrics['smoothness'] = episode_result.jerk
        
        # Secondary metrics
        metrics['stability'] = episode_result.stability
        if episode_result.lap_time is not None:
            metrics['lap_time'] = episode_result.lap_time
        metrics['completion_rate'] = 1.0 if episode_result.success else 0.0
        
        # Safety metrics
        metrics['collision_rate'] = 1.0 if episode_result.collision else 0.0
        metrics['off_lane_rate'] = 1.0 if episode_result.off_lane else 0.0
        
        # Calculate violation rate
        total_violations = sum(episode_result.violations.values()) if episode_result.violations else 0
        metrics['violation_rate'] = float(total_violations) / 10.0  # Normalize by max expected violations
        
        return metrics
    
    def aggregate_episode_metrics(self, episode_results: List[EpisodeResult]) -> Dict[str, MetricResult]:
        """Aggregate metrics across multiple episodes.
        
        Args:
            episode_results: List of episode results
            
        Returns:
            Dict[str, MetricResult]: Aggregated metrics with confidence intervals
        """
        if not episode_results:
            return {}
        
        # Extract episode-level metrics
        episode_metrics = [self.calculate_episode_metrics(ep) for ep in episode_results]
        
        aggregated_metrics = {}
        
        for metric_name in self.metric_definitions.keys():
            # Get values for this metric
            values = [ep_metrics.get(metric_name, 0.0) for ep_metrics in episode_metrics]
            values = [v for v in values if v is not None]  # Remove None values
            
            if not values:
                continue
            
            # Calculate basic statistics
            mean_value = np.mean(values)
            
            # Calculate confidence interval
            ci = self._calculate_confidence_interval(values, metric_name)
            
            # Create metric result
            aggregated_metrics[metric_name] = MetricResult(
                name=metric_name,
                value=mean_value,
                confidence_interval=ci,
                sample_size=len(values),
                metadata={
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            )
        
        return aggregated_metrics
    
    def calculate_suite_metrics(self, suite_results: SuiteResults) -> Dict[str, MetricResult]:
        """Calculate metrics for a complete suite.
        
        Args:
            suite_results: Results from a complete evaluation suite
            
        Returns:
            Dict[str, MetricResult]: Suite-level metrics
        """
        return self.aggregate_episode_metrics(suite_results.episode_results)
    
    def calculate_per_map_metrics(self, episode_results: List[EpisodeResult]) -> Dict[str, Dict[str, MetricResult]]:
        """Calculate metrics grouped by map.
        
        Args:
            episode_results: List of episode results
            
        Returns:
            Dict[str, Dict[str, MetricResult]]: Metrics grouped by map name
        """
        # Group episodes by map
        episodes_by_map = {}
        for episode in episode_results:
            map_name = episode.map_name
            if map_name not in episodes_by_map:
                episodes_by_map[map_name] = []
            episodes_by_map[map_name].append(episode)
        
        # Calculate metrics for each map
        per_map_metrics = {}
        for map_name, map_episodes in episodes_by_map.items():
            per_map_metrics[map_name] = self.aggregate_episode_metrics(map_episodes)
        
        return per_map_metrics
    
    def normalize_metrics(self, metrics: Dict[str, MetricResult], 
                         normalization_data: Dict[str, Dict[str, float]],
                         scope: NormalizationScope = NormalizationScope.GLOBAL) -> Dict[str, MetricResult]:
        """Normalize metrics using the specified normalization data and scope.
        
        Args:
            metrics: Dictionary of metric results to normalize
            normalization_data: Normalization statistics (min, max, mean, std)
            scope: Normalization scope
            
        Returns:
            Dict[str, MetricResult]: Metrics with normalized values
        """
        normalized_metrics = {}
        
        for metric_name, metric_result in metrics.items():
            if metric_name not in normalization_data:
                # No normalization data available, keep original
                normalized_metrics[metric_name] = metric_result
                continue
            
            definition = self.metric_definitions.get(metric_name)
            if not definition:
                normalized_metrics[metric_name] = metric_result
                continue
            
            norm_stats = normalization_data[metric_name]
            normalized_value = self._normalize_value(
                metric_result.value,
                norm_stats,
                definition
            )
            
            # Create new metric result with normalized value
            normalized_result = MetricResult(
                name=metric_result.name,
                value=metric_result.value,
                normalized_value=normalized_value,
                confidence_interval=metric_result.confidence_interval,
                sample_size=metric_result.sample_size,
                metadata=metric_result.metadata.copy()
            )
            normalized_result.metadata['normalization_stats'] = norm_stats
            
            normalized_metrics[metric_name] = normalized_result
        
        return normalized_metrics
    
    def _normalize_value(self, value: float, norm_stats: Dict[str, float], 
                        definition: MetricDefinition) -> float:
        """Normalize a single value using the specified method.
        
        Args:
            value: Value to normalize
            norm_stats: Normalization statistics
            definition: Metric definition
            
        Returns:
            float: Normalized value
        """
        if definition.normalization_method == 'min_max':
            min_val = norm_stats.get('min', definition.min_value or 0.0)
            max_val = norm_stats.get('max', definition.max_value or 1.0)
            
            if max_val == min_val:
                return 0.5  # Default to middle value if no variation
            
            normalized = (value - min_val) / (max_val - min_val)
            
            # Invert if lower is better
            if not definition.higher_is_better:
                normalized = 1.0 - normalized
            
            return np.clip(normalized, 0.0, 1.0)
        
        elif definition.normalization_method == 'z_score':
            mean_val = norm_stats.get('mean', 0.0)
            std_val = norm_stats.get('std', 1.0)
            
            if std_val == 0:
                return 0.5
            
            z_score = (value - mean_val) / std_val
            
            # Convert z-score to 0-1 range using sigmoid
            normalized = 1.0 / (1.0 + np.exp(-z_score))
            
            # Invert if lower is better
            if not definition.higher_is_better:
                normalized = 1.0 - normalized
            
            return normalized
        
        elif definition.normalization_method == 'robust':
            median_val = norm_stats.get('median', 0.0)
            mad_val = norm_stats.get('mad', 1.0)  # Median Absolute Deviation
            
            if mad_val == 0:
                return 0.5
            
            robust_score = (value - median_val) / mad_val
            
            # Convert to 0-1 range using tanh
            normalized = (np.tanh(robust_score / 2.0) + 1.0) / 2.0
            
            # Invert if lower is better
            if not definition.higher_is_better:
                normalized = 1.0 - normalized
            
            return normalized
        
        else:
            # Default to min-max normalization
            return self._normalize_value(value, norm_stats, 
                                       MetricDefinition(definition.name, definition.metric_type,
                                                      definition.description, definition.higher_is_better,
                                                      definition.min_value, definition.max_value, 'min_max'))
    
    def calculate_composite_score(self, metrics: Dict[str, MetricResult]) -> MetricResult:
        """Calculate composite score from normalized metrics.
        
        Args:
            metrics: Dictionary of normalized metric results
            
        Returns:
            MetricResult: Composite score result
        """
        score_components = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Calculate weighted sum of normalized metrics
        for metric_name, weight in self.composite_config.weights.items():
            if metric_name in metrics:
                metric_result = metrics[metric_name]
                normalized_value = metric_result.normalized_value
                
                if normalized_value is not None:
                    score_components[metric_name] = {
                        'value': metric_result.value,
                        'normalized_value': normalized_value,
                        'weight': weight,
                        'contribution': normalized_value * weight
                    }
                    weighted_sum += normalized_value * weight
                    total_weight += weight
        
        # Apply safety penalty if configured
        safety_penalty = 0.0
        if self.composite_config.include_safety_penalty:
            safety_penalty = self._calculate_safety_penalty(metrics)
            weighted_sum -= safety_penalty * self.composite_config.safety_penalty_weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 0.0
        
        # Ensure score is in [0, 1] range
        composite_score = np.clip(composite_score, 0.0, 1.0)
        
        return MetricResult(
            name='composite_score',
            value=composite_score,
            normalized_value=composite_score,
            sample_size=min([m.sample_size for m in metrics.values() if m.sample_size > 0], default=0),
            metadata={
                'components': score_components,
                'total_weight': total_weight,
                'safety_penalty': safety_penalty,
                'weights_used': dict(self.composite_config.weights)
            }
        )
    
    def _calculate_safety_penalty(self, metrics: Dict[str, MetricResult]) -> float:
        """Calculate safety penalty from safety metrics.
        
        Args:
            metrics: Dictionary of metric results
            
        Returns:
            float: Safety penalty value (0.0 to 1.0)
        """
        safety_metrics = ['collision_rate', 'off_lane_rate', 'violation_rate']
        penalty = 0.0
        count = 0
        
        for metric_name in safety_metrics:
            if metric_name in metrics:
                metric_result = metrics[metric_name]
                # Use normalized value if available, otherwise raw value
                value = metric_result.normalized_value or metric_result.value
                penalty += value
                count += 1
        
        return penalty / max(count, 1)
    
    def _calculate_confidence_interval(self, values: List[float], metric_name: str) -> ConfidenceInterval:
        """Calculate confidence interval for a metric.
        
        Args:
            values: List of metric values
            metric_name: Name of the metric
            
        Returns:
            ConfidenceInterval: Confidence interval result
        """
        if len(values) < 2:
            return ConfidenceInterval(lower=values[0] if values else 0.0, 
                                    upper=values[0] if values else 0.0,
                                    confidence_level=self.confidence_level,
                                    method="insufficient_data")
        
        # For success rate and other proportions, use Wilson interval
        if metric_name in ['success_rate', 'collision_rate', 'off_lane_rate', 'completion_rate']:
            return self._wilson_confidence_interval(values)
        
        # For continuous metrics, use bootstrap
        return self._bootstrap_confidence_interval(values)
    
    def _wilson_confidence_interval(self, values: List[float]) -> ConfidenceInterval:
        """Calculate Wilson confidence interval for proportions.
        
        Args:
            values: List of binary values (0.0 or 1.0)
            
        Returns:
            ConfidenceInterval: Wilson confidence interval
        """
        n = len(values)
        p = np.mean(values)
        
        # Z-score for confidence level
        z = 1.96 if self.confidence_level == 0.95 else 2.576  # 95% or 99%
        
        # Wilson interval calculation
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method="wilson"
        )
    
    def _bootstrap_confidence_interval(self, values: List[float]) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval.
        
        Args:
            values: List of metric values
            
        Returns:
            ConfidenceInterval: Bootstrap confidence interval
        """
        values_array = np.array(values)
        n = len(values_array)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            bootstrap_sample = np.random.choice(values_array, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method="bootstrap"
        )
    
    def compute_normalization_stats(self, all_metrics: List[Dict[str, MetricResult]], 
                                  scope: NormalizationScope = NormalizationScope.GLOBAL) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics from multiple metric sets.
        
        Args:
            all_metrics: List of metric dictionaries from different models/suites
            scope: Normalization scope
            
        Returns:
            Dict[str, Dict[str, float]]: Normalization statistics for each metric
        """
        if not all_metrics:
            return {}
        
        # Collect all values for each metric
        metric_values = {}
        for metrics_dict in all_metrics:
            for metric_name, metric_result in metrics_dict.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(metric_result.value)
        
        # Calculate statistics for each metric
        normalization_stats = {}
        for metric_name, values in metric_values.items():
            if not values:
                continue
            
            values_array = np.array(values)
            
            stats = {
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'median': float(np.median(values_array)),
                'mad': float(np.median(np.abs(values_array - np.median(values_array)))),
                'count': len(values)
            }
            
            normalization_stats[metric_name] = stats
        
        return normalization_stats
    
    def calculate_model_metrics(self, model_id: str, suite_results_list: List[SuiteResults]) -> ModelMetrics:
        """Calculate comprehensive metrics for a model across all suites.
        
        Args:
            model_id: ID of the model
            suite_results_list: List of suite results for the model
            
        Returns:
            ModelMetrics: Complete metrics for the model
        """
        if not suite_results_list:
            return ModelMetrics(
                model_id=model_id,
                primary_metrics={},
                secondary_metrics={},
                safety_metrics={}
            )
        
        # Aggregate all episodes across suites
        all_episodes = []
        for suite_results in suite_results_list:
            all_episodes.extend(suite_results.episode_results)
        
        # Calculate overall metrics
        overall_metrics = self.aggregate_episode_metrics(all_episodes)
        
        # Calculate per-map metrics
        per_map_metrics = self.calculate_per_map_metrics(all_episodes)
        
        # Calculate per-suite metrics
        per_suite_metrics = {}
        for suite_results in suite_results_list:
            suite_metrics = self.calculate_suite_metrics(suite_results)
            per_suite_metrics[suite_results.suite_name] = suite_metrics
        
        # Separate metrics by type
        primary_metrics = {}
        secondary_metrics = {}
        safety_metrics = {}
        
        for metric_name, metric_result in overall_metrics.items():
            definition = self.metric_definitions.get(metric_name)
            if not definition:
                continue
            
            if definition.metric_type == MetricType.PRIMARY:
                primary_metrics[metric_name] = metric_result
            elif definition.metric_type == MetricType.SECONDARY:
                secondary_metrics[metric_name] = metric_result
            elif definition.metric_type == MetricType.SAFETY:
                safety_metrics[metric_name] = metric_result
        
        return ModelMetrics(
            model_id=model_id,
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
            safety_metrics=safety_metrics,
            per_map_metrics=per_map_metrics,
            per_suite_metrics=per_suite_metrics,
            metadata={
                'total_episodes': len(all_episodes),
                'total_suites': len(suite_results_list),
                'suite_names': [sr.suite_name for sr in suite_results_list]
            }
        )
    
    def add_composite_score(self, model_metrics: ModelMetrics, 
                           normalization_stats: Optional[Dict[str, Dict[str, float]]] = None) -> ModelMetrics:
        """Add composite score to model metrics.
        
        Args:
            model_metrics: Model metrics to add composite score to
            normalization_stats: Optional normalization statistics
            
        Returns:
            ModelMetrics: Model metrics with composite score added
        """
        # Combine all metrics for composite score calculation
        all_metrics = {}
        all_metrics.update(model_metrics.primary_metrics)
        all_metrics.update(model_metrics.secondary_metrics)
        all_metrics.update(model_metrics.safety_metrics)
        
        # Normalize metrics if normalization stats provided
        if normalization_stats:
            all_metrics = self.normalize_metrics(all_metrics, normalization_stats)
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(all_metrics)
        
        # Add to model metrics
        model_metrics.composite_score = composite_score
        
        return model_metrics
    
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Optional[MetricDefinition]: Metric definition if found
        """
        return self.metric_definitions.get(metric_name)
    
    def list_available_metrics(self) -> Dict[str, List[str]]:
        """List all available metrics by type.
        
        Returns:
            Dict[str, List[str]]: Metrics grouped by type
        """
        metrics_by_type = {
            'primary': [],
            'secondary': [],
            'safety': [],
            'composite': ['composite_score']
        }
        
        for metric_name, definition in self.metric_definitions.items():
            metrics_by_type[definition.metric_type.value].append(metric_name)
        
        return metrics_by_type