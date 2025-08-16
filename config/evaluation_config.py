"""
Evaluation Configuration Management System for Enhanced Duckietown RL
Provides comprehensive parameter validation, YAML loading, and configuration templates
for the Master Evaluation Orchestrator.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import os
import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class SuiteConfig:
    """Configuration for individual evaluation suites."""
    name: str
    enabled: bool = True
    seeds_per_map: int = 50
    maps: List[str] = field(default_factory=lambda: ['loop_empty', 'small_loop', 'zigzag_dists'])
    policy_modes: List[str] = field(default_factory=lambda: ['deterministic', 'stochastic'])
    timeout_steps: int = 2000
    
    # Suite-specific parameters
    environmental_noise: float = 0.0
    traffic_density: float = 0.0
    lighting_variation: float = 0.0
    texture_variation: float = 0.0
    camera_noise: float = 0.0
    friction_variation: float = 0.0
    
    def __post_init__(self):
        """Validate suite configuration parameters."""
        if not self.name:
            raise ValueError("Suite name cannot be empty")
        
        if self.seeds_per_map <= 0:
            raise ValueError(f"seeds_per_map must be positive, got {self.seeds_per_map}")
        
        if not self.maps:
            raise ValueError("Suite must specify at least one map")
        
        valid_modes = ['deterministic', 'stochastic']
        for mode in self.policy_modes:
            if mode not in valid_modes:
                raise ValueError(f"Invalid policy mode '{mode}'. Valid modes: {valid_modes}")
        
        if self.timeout_steps <= 0:
            raise ValueError(f"timeout_steps must be positive, got {self.timeout_steps}")
        
        # Validate environmental parameters
        for param_name, param_value in [
            ('environmental_noise', self.environmental_noise),
            ('traffic_density', self.traffic_density),
            ('lighting_variation', self.lighting_variation),
            ('texture_variation', self.texture_variation),
            ('camera_noise', self.camera_noise),
            ('friction_variation', self.friction_variation)
        ]:
            if not 0.0 <= param_value <= 1.0:
                raise ValueError(f"{param_name} must be between 0.0 and 1.0, got {param_value}")


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10000
    min_episodes_for_ci: int = 10
    
    # Primary metrics weights for composite score
    success_rate_weight: float = 0.45
    reward_weight: float = 0.25
    episode_length_weight: float = 0.10
    lateral_deviation_weight: float = 0.08
    heading_error_weight: float = 0.06
    smoothness_weight: float = 0.06
    
    # Normalization settings
    normalization_scope: str = 'per_map_suite'  # 'global', 'per_suite', 'per_map_suite'
    use_composite_score: bool = True
    
    def __post_init__(self):
        """Validate metrics configuration parameters."""
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError(f"confidence_level must be between 0.5 and 0.99, got {self.confidence_level}")
        
        if self.bootstrap_resamples < 1000:
            raise ValueError(f"bootstrap_resamples should be at least 1000, got {self.bootstrap_resamples}")
        
        if self.min_episodes_for_ci <= 0:
            raise ValueError(f"min_episodes_for_ci must be positive, got {self.min_episodes_for_ci}")
        
        # Validate individual weights are non-negative (before checking sum)
        weights = [
            ('success_rate_weight', self.success_rate_weight),
            ('reward_weight', self.reward_weight),
            ('episode_length_weight', self.episode_length_weight),
            ('lateral_deviation_weight', self.lateral_deviation_weight),
            ('heading_error_weight', self.heading_error_weight),
            ('smoothness_weight', self.smoothness_weight)
        ]
        
        for weight_name, weight_value in weights:
            if weight_value < 0:
                raise ValueError(f"{weight_name} must be non-negative, got {weight_value}")
        
        # Validate weights sum to 1.0 (approximately) - after individual validation
        total_weight = (
            self.success_rate_weight + self.reward_weight + self.episode_length_weight +
            self.lateral_deviation_weight + self.heading_error_weight + self.smoothness_weight
        )
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Metric weights must sum to 1.0, got {total_weight}")
        
        valid_scopes = ['global', 'per_suite', 'per_map_suite']
        if self.normalization_scope not in valid_scopes:
            raise ValueError(f"normalization_scope must be one of {valid_scopes}, got {self.normalization_scope}")


@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis."""
    significance_level: float = 0.05
    multiple_comparison_correction: str = 'benjamini_hochberg'
    min_effect_size: float = 0.1
    paired_tests: bool = True
    
    # Statistical test preferences
    use_parametric_tests: bool = False  # Use non-parametric by default
    bootstrap_comparisons: bool = True
    
    def __post_init__(self):
        """Validate statistical configuration parameters."""
        if not 0.001 <= self.significance_level <= 0.1:
            raise ValueError(f"significance_level must be between 0.001 and 0.1, got {self.significance_level}")
        
        valid_corrections = ['benjamini_hochberg', 'bonferroni', 'holm', 'none']
        if self.multiple_comparison_correction not in valid_corrections:
            raise ValueError(f"multiple_comparison_correction must be one of {valid_corrections}, got {self.multiple_comparison_correction}")
        
        if self.min_effect_size < 0:
            raise ValueError(f"min_effect_size must be non-negative, got {self.min_effect_size}")


@dataclass
class FailureAnalysisConfig:
    """Configuration for failure analysis."""
    enabled: bool = True
    record_failure_videos: bool = True
    max_failure_videos: int = 10
    generate_heatmaps: bool = True
    save_episode_traces: bool = True
    
    # Failure classification thresholds
    stuck_threshold_steps: int = 100
    oscillation_threshold: float = 0.5
    overspeed_threshold: float = 2.0
    
    def __post_init__(self):
        """Validate failure analysis configuration parameters."""
        if self.max_failure_videos < 0:
            raise ValueError(f"max_failure_videos must be non-negative, got {self.max_failure_videos}")
        
        if self.stuck_threshold_steps <= 0:
            raise ValueError(f"stuck_threshold_steps must be positive, got {self.stuck_threshold_steps}")
        
        if self.oscillation_threshold <= 0:
            raise ValueError(f"oscillation_threshold must be positive, got {self.oscillation_threshold}")
        
        if self.overspeed_threshold <= 0:
            raise ValueError(f"overspeed_threshold must be positive, got {self.overspeed_threshold}")


@dataclass
class RobustnessConfig:
    """Configuration for robustness analysis."""
    enabled: bool = True
    parameter_sweeps: Dict[str, List[float]] = field(default_factory=lambda: {
        'lighting_intensity': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
        'texture_domain': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'camera_pitch': [-0.2, -0.1, 0.0, 0.1, 0.2],
        'friction_coefficient': [0.6, 0.8, 1.0, 1.2, 1.4],
        'traffic_density': [0.0, 0.1, 0.3, 0.5, 0.7]
    })
    
    # Robustness analysis settings
    min_success_rate_threshold: float = 0.75
    sensitivity_threshold: float = 0.1  # 10% degradation
    
    def __post_init__(self):
        """Validate robustness configuration parameters."""
        if not 0.0 <= self.min_success_rate_threshold <= 1.0:
            raise ValueError(f"min_success_rate_threshold must be between 0.0 and 1.0, got {self.min_success_rate_threshold}")
        
        if not 0.0 <= self.sensitivity_threshold <= 1.0:
            raise ValueError(f"sensitivity_threshold must be between 0.0 and 1.0, got {self.sensitivity_threshold}")
        
        # Validate parameter sweep ranges
        for param_name, param_values in self.parameter_sweeps.items():
            if not param_values:
                raise ValueError(f"Parameter sweep for '{param_name}' cannot be empty")
            
            if len(param_values) < 2:
                raise ValueError(f"Parameter sweep for '{param_name}' must have at least 2 values")


@dataclass
class ChampionSelectionConfig:
    """Configuration for champion selection."""
    enabled: bool = True
    
    # Champion validation thresholds
    min_maps_passing: float = 0.9  # 90% of maps must meet thresholds
    min_success_rate: float = 0.75
    max_regression_threshold: float = 0.05  # 5% success rate decrease
    max_smoothness_increase: float = 0.20  # 20% smoothness increase
    
    # Pareto analysis
    pareto_axes: List[List[str]] = field(default_factory=lambda: [
        ['success_rate', '-lateral_deviation', '-smoothness']
    ])
    
    def __post_init__(self):
        """Validate champion selection configuration parameters."""
        if not 0.0 <= self.min_maps_passing <= 1.0:
            raise ValueError(f"min_maps_passing must be between 0.0 and 1.0, got {self.min_maps_passing}")
        
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError(f"min_success_rate must be between 0.0 and 1.0, got {self.min_success_rate}")
        
        if self.max_regression_threshold < 0:
            raise ValueError(f"max_regression_threshold must be non-negative, got {self.max_regression_threshold}")
        
        if self.max_smoothness_increase < 0:
            raise ValueError(f"max_smoothness_increase must be non-negative, got {self.max_smoothness_increase}")


@dataclass
class ArtifactConfig:
    """Configuration for evaluation artifacts and outputs."""
    output_directory: str = "evaluation_results"
    keep_top_k_models: int = 5
    
    # Export settings
    export_csv: bool = True
    export_json: bool = True
    export_plots: bool = True
    export_videos: bool = True
    
    # Compression and cleanup
    compress_artifacts: bool = True
    cleanup_intermediate: bool = True
    max_artifact_age_days: int = 30
    
    def __post_init__(self):
        """Validate artifact configuration parameters."""
        if self.keep_top_k_models <= 0:
            raise ValueError(f"keep_top_k_models must be positive, got {self.keep_top_k_models}")
        
        if self.max_artifact_age_days <= 0:
            raise ValueError(f"max_artifact_age_days must be positive, got {self.max_artifact_age_days}")
        
        if not self.output_directory:
            raise ValueError("output_directory cannot be empty")


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility and experiment tracking."""
    fix_seed_list: bool = True
    seed_base: int = 42
    cudnn_deterministic: bool = True
    log_git_sha: bool = True
    log_environment_info: bool = True
    
    # Version tracking
    track_model_versions: bool = True
    track_config_versions: bool = True
    
    def __post_init__(self):
        """Validate reproducibility configuration parameters."""
        if self.seed_base < 0:
            raise ValueError(f"seed_base must be non-negative, got {self.seed_base}")


@dataclass
class EvaluationConfig:
    """
    Comprehensive configuration for the Master Evaluation Orchestrator.
    Provides parameter validation, YAML loading, and configuration templates.
    """
    # Core evaluation settings
    suites: Dict[str, SuiteConfig] = field(default_factory=dict)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    failure_analysis: FailureAnalysisConfig = field(default_factory=FailureAnalysisConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    champion_selection: ChampionSelectionConfig = field(default_factory=ChampionSelectionConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    
    # Global settings
    parallel_evaluation: bool = True
    max_parallel_workers: int = 4
    evaluation_timeout_hours: float = 24.0
    
    def __post_init__(self):
        """Validate global evaluation configuration parameters."""
        if self.max_parallel_workers <= 0:
            raise ValueError(f"max_parallel_workers must be positive, got {self.max_parallel_workers}")
        
        if self.evaluation_timeout_hours <= 0:
            raise ValueError(f"evaluation_timeout_hours must be positive, got {self.evaluation_timeout_hours}")
        
        # Ensure at least one suite is configured
        if not self.suites:
            # Create default suites if none specified
            self.suites = self._create_default_suites()
    
    def _create_default_suites(self) -> Dict[str, SuiteConfig]:
        """Create default evaluation suites."""
        return {
            'base': SuiteConfig(
                name='base',
                environmental_noise=0.0,
                traffic_density=0.0
            ),
            'hard': SuiteConfig(
                name='hard',
                environmental_noise=0.7,
                traffic_density=0.3,
                lighting_variation=0.5,
                texture_variation=0.4,
                camera_noise=0.2,
                friction_variation=0.2
            ),
            'law': SuiteConfig(
                name='law',
                maps=['4way', 'udem1', 'regress_4way_adam'],
                environmental_noise=0.1,
                traffic_density=0.2
            ),
            'ood': SuiteConfig(
                name='ood',
                environmental_noise=0.8,
                lighting_variation=0.8,
                texture_variation=0.9,
                camera_noise=0.5
            )
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'EvaluationConfig':
        """
        Load evaluation configuration from YAML file with validation.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            EvaluationConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If configuration validation fails
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Evaluation configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}")
        
        if config_dict is None:
            config_dict = {}
        
        # Validate against schema
        cls._validate_config_schema(config_dict)
        
        # Create nested configuration objects
        try:
            # Parse suites
            suites = {}
            if 'suites' in config_dict:
                for suite_name, suite_data in config_dict['suites'].items():
                    suite_data['name'] = suite_name
                    suites[suite_name] = SuiteConfig(**suite_data)
            
            # Parse other components
            metrics_config = MetricsConfig(**config_dict.get('metrics', {}))
            statistical_config = StatisticalConfig(**config_dict.get('statistical', {}))
            failure_analysis_config = FailureAnalysisConfig(**config_dict.get('failure_analysis', {}))
            robustness_config = RobustnessConfig(**config_dict.get('robustness', {}))
            champion_selection_config = ChampionSelectionConfig(**config_dict.get('champion_selection', {}))
            artifacts_config = ArtifactConfig(**config_dict.get('artifacts', {}))
            reproducibility_config = ReproducibilityConfig(**config_dict.get('reproducibility', {}))
            
            return cls(
                suites=suites,
                metrics=metrics_config,
                statistical=statistical_config,
                failure_analysis=failure_analysis_config,
                robustness=robustness_config,
                champion_selection=champion_selection_config,
                artifacts=artifacts_config,
                reproducibility=reproducibility_config,
                parallel_evaluation=config_dict.get('parallel_evaluation', True),
                max_parallel_workers=config_dict.get('max_parallel_workers', 4),
                evaluation_timeout_hours=config_dict.get('evaluation_timeout_hours', 24.0)
            )
        except (TypeError, ValueError) as e:
            raise ValidationError(f"Evaluation configuration validation failed: {e}")
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save evaluation configuration to YAML file.
        
        Args:
            yaml_path: Path where to save YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Evaluation configuration saved to {yaml_path}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters at runtime with validation.
        
        Args:
            updates: Dictionary of parameter updates in nested format
        
        Raises:
            ValidationError: If updated parameters are invalid
        """
        # Create a copy of current config as dict
        current_dict = asdict(self)
        
        # Apply updates
        self._recursive_update(current_dict, updates)
        
        # Validate the updated configuration
        self._validate_config_schema(current_dict)
        
        # Recreate configuration objects with updated values
        try:
            if 'suites' in updates:
                suites = {}
                for suite_name, suite_data in current_dict['suites'].items():
                    suite_data['name'] = suite_name
                    suites[suite_name] = SuiteConfig(**suite_data)
                self.suites = suites
            
            if 'metrics' in updates:
                self.metrics = MetricsConfig(**current_dict['metrics'])
            
            if 'statistical' in updates:
                self.statistical = StatisticalConfig(**current_dict['statistical'])
            
            if 'failure_analysis' in updates:
                self.failure_analysis = FailureAnalysisConfig(**current_dict['failure_analysis'])
            
            if 'robustness' in updates:
                self.robustness = RobustnessConfig(**current_dict['robustness'])
            
            if 'champion_selection' in updates:
                self.champion_selection = ChampionSelectionConfig(**current_dict['champion_selection'])
            
            if 'artifacts' in updates:
                self.artifacts = ArtifactConfig(**current_dict['artifacts'])
            
            if 'reproducibility' in updates:
                self.reproducibility = ReproducibilityConfig(**current_dict['reproducibility'])
            
            # Update global settings
            for key in ['parallel_evaluation', 'max_parallel_workers', 'evaluation_timeout_hours']:
                if key in updates:
                    setattr(self, key, current_dict[key])
                    
        except (TypeError, ValueError) as e:
            raise ValidationError(f"Evaluation configuration update validation failed: {e}")
        
        logger.info("Evaluation configuration updated successfully")
    
    def get_suite_config(self, suite_name: str) -> SuiteConfig:
        """
        Get configuration for a specific evaluation suite.
        
        Args:
            suite_name: Name of the evaluation suite
            
        Returns:
            SuiteConfig for the specified suite
            
        Raises:
            ValueError: If suite name is not found
        """
        if suite_name not in self.suites:
            raise ValueError(f"Suite '{suite_name}' not found. Available suites: {list(self.suites.keys())}")
        
        return self.suites[suite_name]
    
    def get_enabled_suites(self) -> List[str]:
        """
        Get list of enabled evaluation suites.
        
        Returns:
            List of enabled suite names
        """
        return [name for name, config in self.suites.items() if config.enabled]
    
    def validate_runtime_parameters(self) -> List[str]:
        """
        Validate runtime parameters and return list of warnings.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check if any suites are enabled
        enabled_suites = self.get_enabled_suites()
        if not enabled_suites:
            warnings.append("No evaluation suites are enabled")
        
        # Check for reasonable number of seeds
        for suite_name, suite_config in self.suites.items():
            if suite_config.enabled and suite_config.seeds_per_map < 10:
                warnings.append(f"Suite '{suite_name}' has very few seeds ({suite_config.seeds_per_map}), results may not be statistically reliable")
        
        # Check bootstrap resamples
        if self.metrics.bootstrap_resamples < 5000:
            warnings.append(f"Bootstrap resamples ({self.metrics.bootstrap_resamples}) is low, consider increasing for more reliable confidence intervals")
        
        # Check parallel workers vs available resources
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        if self.max_parallel_workers > available_cores:
            warnings.append(f"max_parallel_workers ({self.max_parallel_workers}) exceeds available CPU cores ({available_cores})")
        
        return warnings
    
    @staticmethod
    def _validate_config_schema(config_dict: Dict[str, Any]) -> None:
        """
        Validate configuration dictionary against JSON schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Raises:
            ValidationError: If configuration doesn't match schema
        """
        # Define comprehensive schema for evaluation configuration
        schema = {
            "type": "object",
            "properties": {
                "suites": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "seeds_per_map": {"type": "integer", "minimum": 1},
                                "maps": {"type": "array", "items": {"type": "string"}},
                                "policy_modes": {"type": "array", "items": {"type": "string", "enum": ["deterministic", "stochastic"]}},
                                "timeout_steps": {"type": "integer", "minimum": 1},
                                "environmental_noise": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "traffic_density": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "lighting_variation": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "texture_variation": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "camera_noise": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "friction_variation": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            }
                        }
                    }
                },
                "metrics": {
                    "type": "object",
                    "properties": {
                        "compute_confidence_intervals": {"type": "boolean"},
                        "confidence_level": {"type": "number", "minimum": 0.5, "maximum": 0.99},
                        "bootstrap_resamples": {"type": "integer", "minimum": 1000},
                        "success_rate_weight": {"type": "number", "minimum": 0.0},
                        "reward_weight": {"type": "number", "minimum": 0.0},
                        "episode_length_weight": {"type": "number", "minimum": 0.0},
                        "lateral_deviation_weight": {"type": "number", "minimum": 0.0},
                        "heading_error_weight": {"type": "number", "minimum": 0.0},
                        "smoothness_weight": {"type": "number", "minimum": 0.0},
                        "normalization_scope": {"type": "string", "enum": ["global", "per_suite", "per_map_suite"]},
                        "use_composite_score": {"type": "boolean"}
                    }
                },
                "statistical": {
                    "type": "object",
                    "properties": {
                        "significance_level": {"type": "number", "minimum": 0.001, "maximum": 0.1},
                        "multiple_comparison_correction": {"type": "string", "enum": ["benjamini_hochberg", "bonferroni", "holm", "none"]},
                        "min_effect_size": {"type": "number", "minimum": 0.0},
                        "paired_tests": {"type": "boolean"},
                        "use_parametric_tests": {"type": "boolean"},
                        "bootstrap_comparisons": {"type": "boolean"}
                    }
                },
                "parallel_evaluation": {"type": "boolean"},
                "max_parallel_workers": {"type": "integer", "minimum": 1},
                "evaluation_timeout_hours": {"type": "number", "minimum": 0.0}
            }
        }
        
        try:
            validate(instance=config_dict, schema=schema)
        except ValidationError as e:
            raise ValidationError(f"Evaluation configuration schema validation failed: {e.message}")
    
    @staticmethod
    def _recursive_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                EvaluationConfig._recursive_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Configuration template functions
def create_basic_evaluation_config() -> EvaluationConfig:
    """Create a basic evaluation configuration for quick testing."""
    return EvaluationConfig(
        suites={
            'base': SuiteConfig(
                name='base',
                seeds_per_map=10,
                maps=['loop_empty'],
                policy_modes=['deterministic']
            )
        },
        metrics=MetricsConfig(
            bootstrap_resamples=1000,
            compute_confidence_intervals=False
        ),
        failure_analysis=FailureAnalysisConfig(
            record_failure_videos=False,
            generate_heatmaps=False
        ),
        robustness=RobustnessConfig(enabled=False),
        max_parallel_workers=1
    )


def create_comprehensive_evaluation_config() -> EvaluationConfig:
    """Create a comprehensive evaluation configuration for full model assessment."""
    return EvaluationConfig(
        suites={
            'base': SuiteConfig(
                name='base',
                seeds_per_map=50,
                environmental_noise=0.0,
                traffic_density=0.0
            ),
            'hard': SuiteConfig(
                name='hard',
                seeds_per_map=50,
                environmental_noise=0.7,
                traffic_density=0.3,
                lighting_variation=0.5,
                texture_variation=0.4,
                camera_noise=0.2,
                friction_variation=0.2
            ),
            'law': SuiteConfig(
                name='law',
                seeds_per_map=30,
                maps=['4way', 'udem1', 'regress_4way_adam'],
                environmental_noise=0.1,
                traffic_density=0.2
            ),
            'ood': SuiteConfig(
                name='ood',
                seeds_per_map=40,
                environmental_noise=0.8,
                lighting_variation=0.8,
                texture_variation=0.9,
                camera_noise=0.5
            ),
            'stress': SuiteConfig(
                name='stress',
                seeds_per_map=20,
                environmental_noise=1.0,
                traffic_density=0.8,
                lighting_variation=1.0,
                texture_variation=1.0,
                camera_noise=0.8,
                friction_variation=0.5
            )
        },
        metrics=MetricsConfig(
            compute_confidence_intervals=True,
            bootstrap_resamples=10000
        ),
        failure_analysis=FailureAnalysisConfig(
            record_failure_videos=True,
            max_failure_videos=20,
            generate_heatmaps=True
        ),
        robustness=RobustnessConfig(
            enabled=True,
            parameter_sweeps={
                'lighting_intensity': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
                'texture_domain': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                'camera_pitch': [-0.2, -0.1, 0.0, 0.1, 0.2],
                'friction_coefficient': [0.6, 0.8, 1.0, 1.2, 1.4],
                'traffic_density': [0.0, 0.1, 0.3, 0.5, 0.7]
            }
        ),
        max_parallel_workers=4
    )


def create_research_evaluation_config() -> EvaluationConfig:
    """Create an evaluation configuration optimized for research and publication."""
    return EvaluationConfig(
        suites={
            'base': SuiteConfig(
                name='base',
                seeds_per_map=100,
                maps=['loop_empty', 'small_loop', 'zigzag_dists', 'straight_road'],
                policy_modes=['deterministic', 'stochastic']
            ),
            'robustness': SuiteConfig(
                name='robustness',
                seeds_per_map=50,
                environmental_noise=0.5,
                lighting_variation=0.3,
                texture_variation=0.3,
                camera_noise=0.1
            ),
            'generalization': SuiteConfig(
                name='generalization',
                seeds_per_map=75,
                maps=['udem1', 'regress_4way_adam', 'ETHZ_autolab_technical_track'],
                environmental_noise=0.3,
                traffic_density=0.2
            )
        },
        metrics=MetricsConfig(
            compute_confidence_intervals=True,
            confidence_level=0.95,
            bootstrap_resamples=20000
        ),
        statistical=StatisticalConfig(
            significance_level=0.01,  # More stringent for research
            multiple_comparison_correction='benjamini_hochberg',
            bootstrap_comparisons=True
        ),
        reproducibility=ReproducibilityConfig(
            fix_seed_list=True,
            cudnn_deterministic=True,
            log_git_sha=True,
            log_environment_info=True,
            track_model_versions=True,
            track_config_versions=True
        ),
        max_parallel_workers=6
    )


# Configuration utilities
def load_evaluation_config(config_path: Union[str, Path] = None, template: str = 'comprehensive') -> EvaluationConfig:
    """
    Load evaluation configuration from file or create from template.
    
    Args:
        config_path: Path to configuration file. If None, creates from template.
        template: Template name ('basic', 'comprehensive', 'research')
        
    Returns:
        EvaluationConfig instance
    """
    if config_path is None:
        template_map = {
            'basic': create_basic_evaluation_config,
            'comprehensive': create_comprehensive_evaluation_config,
            'research': create_research_evaluation_config
        }
        
        if template not in template_map:
            logger.warning(f"Unknown template '{template}', using 'comprehensive'")
            template = 'comprehensive'
        
        logger.info(f"Creating evaluation configuration from '{template}' template")
        return template_map[template]()
    
    try:
        config = EvaluationConfig.from_yaml(config_path)
        logger.info(f"Evaluation configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load evaluation configuration from {config_path}: {e}")
        logger.info("Using comprehensive template as fallback")
        return create_comprehensive_evaluation_config()


def validate_evaluation_config_file(config_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate an evaluation configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        config = EvaluationConfig.from_yaml(config_path)
        warnings = config.validate_runtime_parameters()
        if warnings:
            errors.extend([f"Warning: {w}" for w in warnings])
        return True, errors
    except Exception as e:
        errors.append(f"Validation failed: {e}")
        return False, errors