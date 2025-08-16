#!/usr/bin/env python3
"""
🧪 INTEGRATION TEST CONFIGURATION 🧪
Configuration settings for integration tests

This module provides configuration settings and utilities for running
integration tests across the evaluation system.

Requirements covered: 8.4, 9.1-9.5, 13.3, 13.4
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    
    # Test execution settings
    timeout_per_suite: int = 300  # seconds
    max_concurrent_tests: int = 4
    verbose_output: bool = False
    save_artifacts: bool = True
    
    # Performance test settings
    performance_test_episodes: int = 100
    performance_test_models: int = 5
    memory_limit_mb: int = 1000
    cpu_time_limit_seconds: int = 60
    
    # Statistical validation settings
    statistical_test_samples: int = 1000
    confidence_level: float = 0.95
    bootstrap_resamples: int = 1000
    significance_alpha: float = 0.05
    
    # Reproducibility settings
    base_seed: int = 42
    random_seed: int = 42
    reproducibility_runs: int = 3
    
    # Test data settings
    mock_episode_count: int = 50
    mock_model_count: int = 3
    test_suite_names: List[str] = field(default_factory=lambda: [
        'base', 'hard_randomization', 'law_intersection', 
        'out_of_distribution', 'stress_adversarial'
    ])
    
    # File system settings
    temp_dir: str = field(default_factory=lambda: tempfile.mkdtemp())
    results_dir: str = field(default_factory=lambda: str(Path(__file__).parent / 'integration_test_results'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for component configuration."""
        return {
            'results_dir': self.temp_dir,
            'base_seed': self.base_seed,
            'random_seed': self.random_seed,
            'bootstrap_samples': self.bootstrap_resamples,
            'confidence_level': self.confidence_level,
            'alpha': self.significance_alpha,
            'max_concurrent_evaluations': self.max_concurrent_tests,
            'default_seeds_per_suite': 10,
            'timeout_per_episode': 30.0,
            'save_artifacts': self.save_artifacts,
            'generate_reports': False,  # Disable for testing
            'deterministic_mode': True
        }


# Default configuration instance
DEFAULT_CONFIG = IntegrationTestConfig()


# Test suite configurations
TEST_SUITE_CONFIGS = {
    'evaluation_integration': {
        'description': 'End-to-end evaluation pipeline integration tests',
        'timeout': 300,
        'requirements': ['8.4', '9.1', '9.2', '9.3', '9.4', '9.5', '13.3', '13.4'],
        'test_categories': [
            'end_to_end_pipeline',
            'mock_model_evaluation',
            'suite_integration',
            'failure_mode_analysis',
            'artifact_management',
            'memory_usage',
            'error_handling',
            'concurrent_safety'
        ]
    },
    'statistical_validation': {
        'description': 'Statistical validation and confidence interval tests',
        'timeout': 180,
        'requirements': ['8.4', '13.1', '13.2'],
        'test_categories': [
            'confidence_interval_coverage',
            'significance_test_accuracy',
            'effect_size_validation',
            'multiple_comparison_correction',
            'bootstrap_consistency',
            'nonparametric_robustness'
        ]
    },
    'performance_benchmarking': {
        'description': 'Performance and throughput benchmarking tests',
        'timeout': 240,
        'requirements': ['8.4', '13.3', '13.4'],
        'test_categories': [
            'throughput_benchmarks',
            'memory_efficiency',
            'cpu_utilization',
            'scalability_tests',
            'concurrent_performance'
        ]
    },
    'reproducibility_validation': {
        'description': 'Reproducibility and seed validation tests',
        'timeout': 120,
        'requirements': ['8.4', '13.3', '13.4'],
        'test_categories': [
            'seed_reproducibility',
            'configuration_consistency',
            'cross_run_validation',
            'artifact_reproducibility',
            'environment_tracking'
        ]
    }
}


# Performance benchmarks and thresholds
PERFORMANCE_THRESHOLDS = {
    'model_registration_rate': 50,  # models per second
    'task_scheduling_rate': 500,    # tasks per second
    'episode_processing_rate': 1000,  # episodes per second
    'metrics_calculation_rate': 1000,  # episodes per second
    'statistical_analysis_rate': 10,   # comparisons per second
    'memory_per_episode_mb': 0.05,     # MB per episode
    'memory_per_model_mb': 1.0,        # MB per model
    'max_memory_increase_mb': 500,     # Maximum memory increase
    'concurrent_efficiency': 0.7       # Minimum concurrency efficiency
}


# Statistical validation parameters
STATISTICAL_VALIDATION_PARAMS = {
    'coverage_tolerance': 0.02,      # Tolerance for CI coverage
    'type_i_error_tolerance': 0.02,  # Tolerance for Type I error rate
    'effect_size_tolerance': 0.1,    # Tolerance for effect size accuracy
    'power_threshold': 0.7,          # Minimum statistical power
    'fdr_threshold': 0.1             # Maximum false discovery rate
}


# Test data generation parameters
TEST_DATA_PARAMS = {
    'episode_success_rates': [0.6, 0.75, 0.9],  # Different performance levels
    'reward_ranges': [(0.3, 0.5), (0.5, 0.8), (0.8, 0.95)],
    'episode_length_range': (400, 600),
    'lateral_deviation_range': (0.05, 0.25),
    'heading_error_range': (2.0, 15.0),
    'jerk_range': (0.02, 0.15),
    'stability_range': (0.6, 0.95),
    'map_names': ['map_0', 'map_1', 'map_2', 'map_3', 'map_4']
}


def get_test_config(config_overrides: Dict[str, Any] = None) -> IntegrationTestConfig:
    """Get test configuration with optional overrides."""
    config = IntegrationTestConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def create_mock_performance_profiles() -> Dict[str, Dict[str, float]]:
    """Create mock performance profiles for different model types."""
    return {
        'champion': {
            'base_success_rate': 0.95,
            'base_reward': 0.9,
            'noise_sensitivity': 0.05,
            'failure_rate': 0.02,
            'robustness_score': 0.9
        },
        'baseline': {
            'base_success_rate': 0.8,
            'base_reward': 0.7,
            'noise_sensitivity': 0.1,
            'failure_rate': 0.05,
            'robustness_score': 0.7
        },
        'weak': {
            'base_success_rate': 0.6,
            'base_reward': 0.5,
            'noise_sensitivity': 0.2,
            'failure_rate': 0.15,
            'robustness_score': 0.5
        },
        'unstable': {
            'base_success_rate': 0.7,
            'base_reward': 0.6,
            'noise_sensitivity': 0.3,
            'failure_rate': 0.2,
            'robustness_score': 0.4
        },
        'overfitted': {
            'base_success_rate': 0.9,
            'base_reward': 0.85,
            'noise_sensitivity': 0.4,
            'failure_rate': 0.1,
            'robustness_score': 0.3
        }
    }


def get_suite_difficulty_multipliers() -> Dict[str, float]:
    """Get difficulty multipliers for different test suites."""
    return {
        'base': 1.0,
        'hard_randomization': 0.8,
        'law_intersection': 0.9,
        'out_of_distribution': 0.7,
        'stress_adversarial': 0.6
    }


def get_failure_mode_probabilities() -> Dict[str, Dict[str, float]]:
    """Get failure mode probabilities for different suites."""
    return {
        'base': {
            'collision': 0.05,
            'off_lane': 0.05,
            'stuck': 0.02,
            'oscillation': 0.01,
            'over_speed': 0.01,
            'violation': 0.01
        },
        'hard_randomization': {
            'collision': 0.15,
            'off_lane': 0.15,
            'stuck': 0.05,
            'oscillation': 0.03,
            'over_speed': 0.02,
            'violation': 0.02
        },
        'law_intersection': {
            'collision': 0.08,
            'off_lane': 0.08,
            'stuck': 0.03,
            'oscillation': 0.02,
            'over_speed': 0.05,
            'violation': 0.15
        },
        'out_of_distribution': {
            'collision': 0.20,
            'off_lane': 0.20,
            'stuck': 0.08,
            'oscillation': 0.05,
            'over_speed': 0.03,
            'violation': 0.04
        },
        'stress_adversarial': {
            'collision': 0.30,
            'off_lane': 0.20,
            'stuck': 0.10,
            'oscillation': 0.08,
            'over_speed': 0.05,
            'violation': 0.07
        }
    }


def validate_test_environment() -> Dict[str, bool]:
    """Validate that the test environment is properly set up."""
    validation_results = {}
    
    # Check Python version
    import sys
    validation_results['python_version'] = sys.version_info >= (3, 7)
    
    # Check required packages
    required_packages = [
        'numpy', 'pytest', 'psutil', 'pathlib', 'tempfile',
        'threading', 'concurrent.futures', 'json', 'time'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('.', '_'))
            validation_results[f'package_{package}'] = True
        except ImportError:
            validation_results[f'package_{package}'] = False
    
    # Check file system permissions
    try:
        test_dir = Path(tempfile.mkdtemp())
        test_file = test_dir / 'test.txt'
        test_file.write_text('test')
        test_file.unlink()
        test_dir.rmdir()
        validation_results['filesystem_access'] = True
    except Exception:
        validation_results['filesystem_access'] = False
    
    # Check memory availability
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        validation_results['sufficient_memory'] = available_memory_gb >= 2.0
    except Exception:
        validation_results['sufficient_memory'] = False
    
    return validation_results


def get_test_requirements_mapping() -> Dict[str, List[str]]:
    """Get mapping of test categories to requirements."""
    return {
        'end_to_end_pipeline': ['8.1', '8.2', '13.3', '13.4'],
        'statistical_validation': ['8.4', '12.2', '13.1', '13.2'],
        'performance_benchmarking': ['8.4', '13.3', '13.4'],
        'reproducibility_validation': ['8.4', '13.3', '13.4'],
        'suite_integration': ['9.1', '9.2', '9.3', '9.4', '9.5'],
        'failure_analysis': ['10.1', '10.2', '10.3', '10.4', '10.5'],
        'robustness_analysis': ['11.1', '11.2', '11.3', '11.4', '11.5'],
        'champion_selection': ['12.1', '12.2', '12.3', '12.4', '12.5'],
        'artifact_management': ['13.2', '13.4', '13.5'],
        'report_generation': ['13.1', '13.2', '13.5']
    }


if __name__ == '__main__':
    # Validate test environment when run directly
    print("🔍 Validating test environment...")
    
    validation_results = validate_test_environment()
    
    print("\n📋 Validation Results:")
    for check, result in validation_results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}")
    
    all_passed = all(validation_results.values())
    
    if all_passed:
        print("\n✅ Test environment validation passed!")
    else:
        print("\n❌ Test environment validation failed!")
        failed_checks = [check for check, result in validation_results.items() if not result]
        print(f"Failed checks: {', '.join(failed_checks)}")
    
    # Print configuration summary
    config = DEFAULT_CONFIG
    print(f"\n📊 Default Configuration:")
    print(f"   Timeout per suite: {config.timeout_per_suite}s")
    print(f"   Max concurrent tests: {config.max_concurrent_tests}")
    print(f"   Performance test episodes: {config.performance_test_episodes}")
    print(f"   Statistical test samples: {config.statistical_test_samples}")
    print(f"   Base seed: {config.base_seed}")
    print(f"   Results directory: {config.results_dir}")