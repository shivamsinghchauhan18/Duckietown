#!/usr/bin/env python3
"""
Example usage of the Evaluation Configuration Management System.
Demonstrates configuration creation, validation, loading, and customization.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.evaluation_config import (
    EvaluationConfig, SuiteConfig, MetricsConfig,
    create_basic_evaluation_config, create_comprehensive_evaluation_config,
    create_research_evaluation_config, load_evaluation_config,
    validate_evaluation_config_file
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_basic_usage():
    """Demonstrate basic configuration usage."""
    logger.info("=== Basic Configuration Usage ===")
    
    # Create a basic configuration
    config = create_basic_evaluation_config()
    logger.info(f"Created basic config with {len(config.suites)} suites")
    logger.info(f"Parallel workers: {config.max_parallel_workers}")
    logger.info(f"Bootstrap resamples: {config.metrics.bootstrap_resamples}")
    
    # Get enabled suites
    enabled_suites = config.get_enabled_suites()
    logger.info(f"Enabled suites: {enabled_suites}")
    
    # Check runtime parameters
    warnings = config.validate_runtime_parameters()
    if warnings:
        logger.warning(f"Configuration warnings: {warnings}")
    else:
        logger.info("No configuration warnings")


def demonstrate_template_comparison():
    """Demonstrate different configuration templates."""
    logger.info("\n=== Template Comparison ===")
    
    templates = {
        'basic': create_basic_evaluation_config(),
        'comprehensive': create_comprehensive_evaluation_config(),
        'research': create_research_evaluation_config()
    }
    
    for name, config in templates.items():
        logger.info(f"\n{name.upper()} Template:")
        logger.info(f"  Suites: {len(config.suites)}")
        logger.info(f"  Parallel workers: {config.max_parallel_workers}")
        logger.info(f"  Bootstrap resamples: {config.metrics.bootstrap_resamples}")
        logger.info(f"  Confidence intervals: {config.metrics.compute_confidence_intervals}")
        logger.info(f"  Robustness analysis: {config.robustness.enabled}")
        logger.info(f"  Failure videos: {config.failure_analysis.record_failure_videos}")


def demonstrate_custom_configuration():
    """Demonstrate creating custom configuration."""
    logger.info("\n=== Custom Configuration ===")
    
    # Create custom suite configuration
    custom_suite = SuiteConfig(
        name="custom_test",
        seeds_per_map=25,
        maps=["loop_empty", "small_loop"],
        policy_modes=["deterministic"],
        environmental_noise=0.3,
        traffic_density=0.1
    )
    
    # Create custom metrics configuration
    custom_metrics = MetricsConfig(
        compute_confidence_intervals=True,
        bootstrap_resamples=5000,
        success_rate_weight=0.5,
        reward_weight=0.3,
        episode_length_weight=0.1,
        lateral_deviation_weight=0.05,
        heading_error_weight=0.03,
        smoothness_weight=0.02
    )
    
    # Create full custom configuration
    custom_config = EvaluationConfig(
        suites={"custom_test": custom_suite},
        metrics=custom_metrics,
        max_parallel_workers=2,
        evaluation_timeout_hours=12.0
    )
    
    logger.info("Created custom configuration:")
    logger.info(f"  Suite: {custom_suite.name}")
    logger.info(f"  Seeds per map: {custom_suite.seeds_per_map}")
    logger.info(f"  Environmental noise: {custom_suite.environmental_noise}")
    logger.info(f"  Success rate weight: {custom_metrics.success_rate_weight}")


def demonstrate_yaml_operations():
    """Demonstrate YAML loading and saving."""
    logger.info("\n=== YAML Operations ===")
    
    # Create configuration
    config = create_comprehensive_evaluation_config()
    
    # Save to YAML
    yaml_path = Path("temp_evaluation_config.yml")
    config.to_yaml(yaml_path)
    logger.info(f"Saved configuration to {yaml_path}")
    
    # Load from YAML
    loaded_config = EvaluationConfig.from_yaml(yaml_path)
    logger.info(f"Loaded configuration from {yaml_path}")
    logger.info(f"  Loaded suites: {list(loaded_config.suites.keys())}")
    logger.info(f"  Parallel workers: {loaded_config.max_parallel_workers}")
    
    # Validate configuration file
    is_valid, errors = validate_evaluation_config_file(yaml_path)
    logger.info(f"Configuration validation: {'VALID' if is_valid else 'INVALID'}")
    if errors:
        for error in errors:
            logger.info(f"  {error}")
    
    # Clean up
    yaml_path.unlink()
    logger.info("Cleaned up temporary file")


def demonstrate_configuration_updates():
    """Demonstrate runtime configuration updates."""
    logger.info("\n=== Configuration Updates ===")
    
    config = create_basic_evaluation_config()
    logger.info(f"Initial bootstrap resamples: {config.metrics.bootstrap_resamples}")
    logger.info(f"Initial parallel workers: {config.max_parallel_workers}")
    
    # Update configuration
    updates = {
        'max_parallel_workers': 4,
        'metrics': {
            'bootstrap_resamples': 15000,
            'compute_confidence_intervals': True
        },
        'failure_analysis': {
            'record_failure_videos': True,
            'max_failure_videos': 15
        }
    }
    
    config.update(updates)
    logger.info(f"Updated bootstrap resamples: {config.metrics.bootstrap_resamples}")
    logger.info(f"Updated parallel workers: {config.max_parallel_workers}")
    logger.info(f"Updated CI computation: {config.metrics.compute_confidence_intervals}")
    logger.info(f"Updated failure videos: {config.failure_analysis.record_failure_videos}")


def demonstrate_template_loading():
    """Demonstrate loading configurations from templates."""
    logger.info("\n=== Template Loading ===")
    
    # Load template configurations
    template_path = Path(__file__).parent.parent / "config" / "templates"
    
    if template_path.exists():
        for template_file in template_path.glob("*.yml"):
            logger.info(f"\nLoading template: {template_file.name}")
            try:
                config = EvaluationConfig.from_yaml(template_file)
                logger.info(f"  Successfully loaded {template_file.name}")
                logger.info(f"  Suites: {list(config.suites.keys())}")
                logger.info(f"  Enabled suites: {len(config.get_enabled_suites())}")
                
                # Validate
                warnings = config.validate_runtime_parameters()
                if warnings:
                    logger.info(f"  Warnings: {len(warnings)}")
                else:
                    logger.info("  No warnings")
                    
            except Exception as e:
                logger.error(f"  Failed to load {template_file.name}: {e}")
    else:
        logger.info("Template directory not found, loading from code templates")
        
        # Load from code templates
        templates = {
            'basic': lambda: load_evaluation_config(template='basic'),
            'comprehensive': lambda: load_evaluation_config(template='comprehensive'),
            'research': lambda: load_evaluation_config(template='research')
        }
        
        for name, loader in templates.items():
            config = loader()
            logger.info(f"  Loaded {name} template")
            logger.info(f"    Suites: {len(config.suites)}")
            logger.info(f"    Timeout: {config.evaluation_timeout_hours}h")


def demonstrate_error_handling():
    """Demonstrate error handling and validation."""
    logger.info("\n=== Error Handling ===")
    
    # Test invalid suite configuration
    try:
        invalid_suite = SuiteConfig(
            name="",  # Invalid: empty name
            seeds_per_map=10
        )
    except ValueError as e:
        logger.info(f"Caught expected error for empty suite name: {e}")
    
    # Test invalid metrics configuration
    try:
        invalid_metrics = MetricsConfig(
            success_rate_weight=0.6,
            reward_weight=0.6,  # Total > 1.0
            episode_length_weight=0.1,
            lateral_deviation_weight=0.1,
            heading_error_weight=0.1,
            smoothness_weight=0.1
        )
    except ValueError as e:
        logger.info(f"Caught expected error for invalid weights: {e}")
    
    # Test invalid configuration update
    config = create_basic_evaluation_config()
    try:
        config.update({
            'max_parallel_workers': -1  # Invalid
        })
    except Exception as e:
        logger.info(f"Caught expected error for invalid update: {e}")
    
    # Test loading nonexistent file
    try:
        EvaluationConfig.from_yaml("nonexistent_config.yml")
    except FileNotFoundError as e:
        logger.info(f"Caught expected error for nonexistent file: {e}")


def main():
    """Run all demonstration functions."""
    logger.info("Evaluation Configuration Management System Examples")
    logger.info("=" * 60)
    
    try:
        demonstrate_basic_usage()
        demonstrate_template_comparison()
        demonstrate_custom_configuration()
        demonstrate_yaml_operations()
        demonstrate_configuration_updates()
        demonstrate_template_loading()
        demonstrate_error_handling()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise


if __name__ == "__main__":
    main()