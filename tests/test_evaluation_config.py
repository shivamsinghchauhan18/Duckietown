"""
Unit tests for evaluation configuration management system.
Tests parameter validation, YAML loading, and error handling.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from config.evaluation_config import (
    EvaluationConfig, SuiteConfig, MetricsConfig, StatisticalConfig,
    FailureAnalysisConfig, RobustnessConfig, ChampionSelectionConfig,
    ArtifactConfig, ReproducibilityConfig,
    create_basic_evaluation_config, create_comprehensive_evaluation_config,
    create_research_evaluation_config, load_evaluation_config,
    validate_evaluation_config_file
)
from jsonschema import ValidationError


class TestSuiteConfig:
    """Test SuiteConfig validation and functionality."""
    
    def test_valid_suite_config(self):
        """Test creation of valid suite configuration."""
        config = SuiteConfig(
            name="test_suite",
            seeds_per_map=50,
            maps=["loop_empty", "small_loop"],
            policy_modes=["deterministic", "stochastic"],
            environmental_noise=0.5
        )
        
        assert config.name == "test_suite"
        assert config.seeds_per_map == 50
        assert config.environmental_noise == 0.5
        assert config.enabled is True  # default
    
    def test_empty_name_validation(self):
        """Test validation of empty suite name."""
        with pytest.raises(ValueError, match="Suite name cannot be empty"):
            SuiteConfig(name="")
    
    def test_negative_seeds_validation(self):
        """Test validation of negative seeds per map."""
        with pytest.raises(ValueError, match="seeds_per_map must be positive"):
            SuiteConfig(name="test", seeds_per_map=-1)
    
    def test_empty_maps_validation(self):
        """Test validation of empty maps list."""
        with pytest.raises(ValueError, match="Suite must specify at least one map"):
            SuiteConfig(name="test", maps=[])
    
    def test_invalid_policy_mode_validation(self):
        """Test validation of invalid policy modes."""
        with pytest.raises(ValueError, match="Invalid policy mode"):
            SuiteConfig(name="test", policy_modes=["invalid_mode"])
    
    def test_environmental_parameter_validation(self):
        """Test validation of environmental parameters."""
        # Test valid range
        config = SuiteConfig(name="test", environmental_noise=0.5)
        assert config.environmental_noise == 0.5
        
        # Test invalid ranges
        with pytest.raises(ValueError, match="environmental_noise must be between 0.0 and 1.0"):
            SuiteConfig(name="test", environmental_noise=-0.1)
        
        with pytest.raises(ValueError, match="traffic_density must be between 0.0 and 1.0"):
            SuiteConfig(name="test", traffic_density=1.5)


class TestMetricsConfig:
    """Test MetricsConfig validation and functionality."""
    
    def test_valid_metrics_config(self):
        """Test creation of valid metrics configuration."""
        config = MetricsConfig(
            confidence_level=0.95,
            bootstrap_resamples=10000,
            success_rate_weight=0.5,
            reward_weight=0.3,
            episode_length_weight=0.1,
            lateral_deviation_weight=0.05,
            heading_error_weight=0.03,
            smoothness_weight=0.02
        )
        
        assert config.confidence_level == 0.95
        assert config.bootstrap_resamples == 10000
    
    def test_confidence_level_validation(self):
        """Test validation of confidence level."""
        with pytest.raises(ValueError, match="confidence_level must be between 0.5 and 0.99"):
            MetricsConfig(confidence_level=0.3)
        
        with pytest.raises(ValueError, match="confidence_level must be between 0.5 and 0.99"):
            MetricsConfig(confidence_level=1.0)
    
    def test_bootstrap_resamples_validation(self):
        """Test validation of bootstrap resamples."""
        with pytest.raises(ValueError, match="bootstrap_resamples should be at least 1000"):
            MetricsConfig(bootstrap_resamples=500)
    
    def test_weights_sum_validation(self):
        """Test validation that weights sum to 1.0."""
        with pytest.raises(ValueError, match="Metric weights must sum to 1.0"):
            MetricsConfig(
                success_rate_weight=0.5,
                reward_weight=0.3,
                episode_length_weight=0.1,
                lateral_deviation_weight=0.05,
                heading_error_weight=0.03,
                smoothness_weight=0.05  # Total = 1.03
            )
    
    def test_negative_weight_validation(self):
        """Test validation of negative weights."""
        with pytest.raises(ValueError, match="success_rate_weight must be non-negative"):
            MetricsConfig(success_rate_weight=-0.1)
    
    def test_normalization_scope_validation(self):
        """Test validation of normalization scope."""
        with pytest.raises(ValueError, match="normalization_scope must be one of"):
            MetricsConfig(normalization_scope="invalid_scope")


class TestStatisticalConfig:
    """Test StatisticalConfig validation and functionality."""
    
    def test_valid_statistical_config(self):
        """Test creation of valid statistical configuration."""
        config = StatisticalConfig(
            significance_level=0.05,
            multiple_comparison_correction='benjamini_hochberg',
            min_effect_size=0.1
        )
        
        assert config.significance_level == 0.05
        assert config.multiple_comparison_correction == 'benjamini_hochberg'
    
    def test_significance_level_validation(self):
        """Test validation of significance level."""
        with pytest.raises(ValueError, match="significance_level must be between 0.001 and 0.1"):
            StatisticalConfig(significance_level=0.0005)
        
        with pytest.raises(ValueError, match="significance_level must be between 0.001 and 0.1"):
            StatisticalConfig(significance_level=0.2)
    
    def test_correction_method_validation(self):
        """Test validation of multiple comparison correction method."""
        with pytest.raises(ValueError, match="multiple_comparison_correction must be one of"):
            StatisticalConfig(multiple_comparison_correction='invalid_method')
    
    def test_negative_effect_size_validation(self):
        """Test validation of negative effect size."""
        with pytest.raises(ValueError, match="min_effect_size must be non-negative"):
            StatisticalConfig(min_effect_size=-0.1)


class TestFailureAnalysisConfig:
    """Test FailureAnalysisConfig validation and functionality."""
    
    def test_valid_failure_analysis_config(self):
        """Test creation of valid failure analysis configuration."""
        config = FailureAnalysisConfig(
            max_failure_videos=20,
            stuck_threshold_steps=100,
            oscillation_threshold=0.5
        )
        
        assert config.max_failure_videos == 20
        assert config.stuck_threshold_steps == 100
    
    def test_negative_max_videos_validation(self):
        """Test validation of negative max failure videos."""
        with pytest.raises(ValueError, match="max_failure_videos must be non-negative"):
            FailureAnalysisConfig(max_failure_videos=-1)
    
    def test_threshold_validation(self):
        """Test validation of various thresholds."""
        with pytest.raises(ValueError, match="stuck_threshold_steps must be positive"):
            FailureAnalysisConfig(stuck_threshold_steps=0)
        
        with pytest.raises(ValueError, match="oscillation_threshold must be positive"):
            FailureAnalysisConfig(oscillation_threshold=0)
        
        with pytest.raises(ValueError, match="overspeed_threshold must be positive"):
            FailureAnalysisConfig(overspeed_threshold=-1)


class TestRobustnessConfig:
    """Test RobustnessConfig validation and functionality."""
    
    def test_valid_robustness_config(self):
        """Test creation of valid robustness configuration."""
        config = RobustnessConfig(
            parameter_sweeps={
                'lighting': [0.5, 1.0, 1.5],
                'friction': [0.8, 1.0, 1.2]
            },
            min_success_rate_threshold=0.75
        )
        
        assert 'lighting' in config.parameter_sweeps
        assert config.min_success_rate_threshold == 0.75
    
    def test_threshold_validation(self):
        """Test validation of robustness thresholds."""
        with pytest.raises(ValueError, match="min_success_rate_threshold must be between 0.0 and 1.0"):
            RobustnessConfig(min_success_rate_threshold=1.5)
        
        with pytest.raises(ValueError, match="sensitivity_threshold must be between 0.0 and 1.0"):
            RobustnessConfig(sensitivity_threshold=-0.1)
    
    def test_empty_parameter_sweep_validation(self):
        """Test validation of empty parameter sweeps."""
        with pytest.raises(ValueError, match="Parameter sweep for .* cannot be empty"):
            RobustnessConfig(parameter_sweeps={'lighting': []})
    
    def test_insufficient_parameter_values_validation(self):
        """Test validation of insufficient parameter values."""
        with pytest.raises(ValueError, match="Parameter sweep for .* must have at least 2 values"):
            RobustnessConfig(parameter_sweeps={'lighting': [1.0]})


class TestChampionSelectionConfig:
    """Test ChampionSelectionConfig validation and functionality."""
    
    def test_valid_champion_selection_config(self):
        """Test creation of valid champion selection configuration."""
        config = ChampionSelectionConfig(
            min_maps_passing=0.9,
            min_success_rate=0.75,
            max_regression_threshold=0.05
        )
        
        assert config.min_maps_passing == 0.9
        assert config.min_success_rate == 0.75
    
    def test_threshold_validation(self):
        """Test validation of champion selection thresholds."""
        with pytest.raises(ValueError, match="min_maps_passing must be between 0.0 and 1.0"):
            ChampionSelectionConfig(min_maps_passing=1.5)
        
        with pytest.raises(ValueError, match="min_success_rate must be between 0.0 and 1.0"):
            ChampionSelectionConfig(min_success_rate=-0.1)
        
        with pytest.raises(ValueError, match="max_regression_threshold must be non-negative"):
            ChampionSelectionConfig(max_regression_threshold=-0.1)


class TestArtifactConfig:
    """Test ArtifactConfig validation and functionality."""
    
    def test_valid_artifact_config(self):
        """Test creation of valid artifact configuration."""
        config = ArtifactConfig(
            output_directory="test_results",
            keep_top_k_models=5,
            max_artifact_age_days=30
        )
        
        assert config.output_directory == "test_results"
        assert config.keep_top_k_models == 5
    
    def test_validation(self):
        """Test validation of artifact configuration parameters."""
        with pytest.raises(ValueError, match="keep_top_k_models must be positive"):
            ArtifactConfig(keep_top_k_models=0)
        
        with pytest.raises(ValueError, match="max_artifact_age_days must be positive"):
            ArtifactConfig(max_artifact_age_days=-1)
        
        with pytest.raises(ValueError, match="output_directory cannot be empty"):
            ArtifactConfig(output_directory="")


class TestReproducibilityConfig:
    """Test ReproducibilityConfig validation and functionality."""
    
    def test_valid_reproducibility_config(self):
        """Test creation of valid reproducibility configuration."""
        config = ReproducibilityConfig(
            seed_base=42,
            fix_seed_list=True,
            log_git_sha=True
        )
        
        assert config.seed_base == 42
        assert config.fix_seed_list is True
    
    def test_negative_seed_validation(self):
        """Test validation of negative seed base."""
        with pytest.raises(ValueError, match="seed_base must be non-negative"):
            ReproducibilityConfig(seed_base=-1)


class TestEvaluationConfig:
    """Test main EvaluationConfig class."""
    
    def test_valid_evaluation_config(self):
        """Test creation of valid evaluation configuration."""
        config = EvaluationConfig(
            max_parallel_workers=4,
            evaluation_timeout_hours=24.0
        )
        
        assert config.max_parallel_workers == 4
        assert config.evaluation_timeout_hours == 24.0
        assert len(config.suites) > 0  # Should create default suites
    
    def test_global_parameter_validation(self):
        """Test validation of global parameters."""
        with pytest.raises(ValueError, match="max_parallel_workers must be positive"):
            EvaluationConfig(max_parallel_workers=0)
        
        with pytest.raises(ValueError, match="evaluation_timeout_hours must be positive"):
            EvaluationConfig(evaluation_timeout_hours=-1)
    
    def test_default_suites_creation(self):
        """Test that default suites are created when none specified."""
        config = EvaluationConfig()
        
        assert 'base' in config.suites
        assert 'hard' in config.suites
        assert 'law' in config.suites
        assert 'ood' in config.suites
    
    def test_get_suite_config(self):
        """Test getting suite configuration."""
        config = EvaluationConfig()
        
        base_suite = config.get_suite_config('base')
        assert base_suite.name == 'base'
        
        with pytest.raises(ValueError, match="Suite 'nonexistent' not found"):
            config.get_suite_config('nonexistent')
    
    def test_get_enabled_suites(self):
        """Test getting enabled suites."""
        config = EvaluationConfig()
        enabled = config.get_enabled_suites()
        
        assert isinstance(enabled, list)
        assert len(enabled) > 0
        assert all(config.suites[name].enabled for name in enabled)
    
    def test_validate_runtime_parameters(self):
        """Test runtime parameter validation."""
        config = EvaluationConfig()
        warnings = config.validate_runtime_parameters()
        
        assert isinstance(warnings, list)
        # Should not have warnings for default config
    
    @patch('multiprocessing.cpu_count', return_value=2)
    def test_validate_runtime_parameters_warnings(self, mock_cpu_count):
        """Test runtime parameter validation with warnings."""
        config = EvaluationConfig(
            max_parallel_workers=4,  # More than available cores
            metrics=MetricsConfig(bootstrap_resamples=1000)  # Low resamples
        )
        
        warnings = config.validate_runtime_parameters()
        
        assert len(warnings) >= 1
        assert any('exceeds available CPU cores' in w for w in warnings)


class TestYAMLOperations:
    """Test YAML loading and saving operations."""
    
    def test_yaml_round_trip(self):
        """Test saving and loading configuration to/from YAML."""
        config = create_basic_evaluation_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save to YAML
            config.to_yaml(temp_path)
            assert temp_path.exists()
            
            # Load from YAML
            loaded_config = EvaluationConfig.from_yaml(temp_path)
            
            # Compare key attributes
            assert loaded_config.max_parallel_workers == config.max_parallel_workers
            assert loaded_config.evaluation_timeout_hours == config.evaluation_timeout_hours
            assert len(loaded_config.suites) == len(config.suites)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_yaml_loading_nonexistent_file(self):
        """Test loading from nonexistent YAML file."""
        with pytest.raises(FileNotFoundError):
            EvaluationConfig.from_yaml("nonexistent.yml")
    
    def test_yaml_loading_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(yaml.YAMLError):
                EvaluationConfig.from_yaml(temp_path)
        finally:
            temp_path.unlink()
    
    def test_yaml_loading_invalid_config(self):
        """Test loading YAML with invalid configuration."""
        invalid_config = {
            'max_parallel_workers': -1,  # Invalid
            'suites': {
                'test': {
                    'name': 'test',
                    'seeds_per_map': -5  # Invalid
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValidationError):
                EvaluationConfig.from_yaml(temp_path)
        finally:
            temp_path.unlink()


class TestConfigurationUpdate:
    """Test configuration update functionality."""
    
    def test_valid_update(self):
        """Test valid configuration update."""
        config = EvaluationConfig()
        
        updates = {
            'max_parallel_workers': 8,
            'metrics': {
                'bootstrap_resamples': 15000
            }
        }
        
        config.update(updates)
        
        assert config.max_parallel_workers == 8
        assert config.metrics.bootstrap_resamples == 15000
    
    def test_invalid_update(self):
        """Test invalid configuration update."""
        config = EvaluationConfig()
        
        invalid_updates = {
            'max_parallel_workers': -1  # Invalid
        }
        
        with pytest.raises(ValidationError):
            config.update(invalid_updates)


class TestConfigurationTemplates:
    """Test configuration template functions."""
    
    def test_basic_template(self):
        """Test basic evaluation configuration template."""
        config = create_basic_evaluation_config()
        
        assert config.max_parallel_workers == 1
        assert not config.metrics.compute_confidence_intervals
        assert not config.robustness.enabled
        assert len(config.suites) == 1
        assert 'base' in config.suites
    
    def test_comprehensive_template(self):
        """Test comprehensive evaluation configuration template."""
        config = create_comprehensive_evaluation_config()
        
        assert config.max_parallel_workers == 4
        assert config.metrics.compute_confidence_intervals
        assert config.robustness.enabled
        assert len(config.suites) >= 4
        assert all(suite in config.suites for suite in ['base', 'hard', 'law', 'ood'])
    
    def test_research_template(self):
        """Test research evaluation configuration template."""
        config = create_research_evaluation_config()
        
        assert config.max_parallel_workers == 6
        assert config.metrics.bootstrap_resamples == 20000
        assert config.statistical.significance_level == 0.01
        assert config.reproducibility.log_git_sha
        assert config.reproducibility.track_model_versions


class TestConfigurationUtilities:
    """Test configuration utility functions."""
    
    def test_load_evaluation_config_default(self):
        """Test loading evaluation config with default template."""
        config = load_evaluation_config(template='comprehensive')
        
        assert isinstance(config, EvaluationConfig)
        assert len(config.suites) >= 4
    
    def test_load_evaluation_config_unknown_template(self):
        """Test loading evaluation config with unknown template."""
        config = load_evaluation_config(template='unknown')
        
        # Should fallback to comprehensive
        assert isinstance(config, EvaluationConfig)
        assert len(config.suites) >= 4
    
    def test_load_evaluation_config_from_file(self):
        """Test loading evaluation config from file."""
        # Create a temporary config file
        config_data = {
            'max_parallel_workers': 2,
            'suites': {
                'test': {
                    'name': 'test',
                    'seeds_per_map': 10,
                    'maps': ['loop_empty']
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = load_evaluation_config(temp_path)
            assert config.max_parallel_workers == 2
            assert 'test' in config.suites
        finally:
            temp_path.unlink()
    
    def test_validate_evaluation_config_file_valid(self):
        """Test validation of valid configuration file."""
        config = create_basic_evaluation_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.to_yaml(temp_path)
            is_valid, errors = validate_evaluation_config_file(temp_path)
            
            assert is_valid
            # May have warnings but should be valid
        finally:
            temp_path.unlink()
    
    def test_validate_evaluation_config_file_invalid(self):
        """Test validation of invalid configuration file."""
        invalid_config = {
            'max_parallel_workers': -1,
            'suites': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)
        
        try:
            is_valid, errors = validate_evaluation_config_file(temp_path)
            
            assert not is_valid
            assert len(errors) > 0
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])