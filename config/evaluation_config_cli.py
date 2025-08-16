#!/usr/bin/env python3
"""
Command-line interface for evaluation configuration management.
Provides utilities for creating, validating, and managing evaluation configurations.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from evaluation_config import (
    EvaluationConfig,
    create_basic_evaluation_config,
    create_comprehensive_evaluation_config,
    create_research_evaluation_config,
    load_evaluation_config,
    validate_evaluation_config_file
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_config_command(args):
    """Create a new evaluation configuration from template."""
    template_map = {
        'basic': create_basic_evaluation_config,
        'comprehensive': create_comprehensive_evaluation_config,
        'research': create_research_evaluation_config
    }
    
    if args.template not in template_map:
        logger.error(f"Unknown template '{args.template}'. Available: {list(template_map.keys())}")
        return 1
    
    try:
        config = template_map[args.template]()
        config.to_yaml(args.output)
        logger.info(f"Created {args.template} evaluation configuration at {args.output}")
        
        # Show summary
        enabled_suites = config.get_enabled_suites()
        logger.info(f"Configuration summary:")
        logger.info(f"  Template: {args.template}")
        logger.info(f"  Suites: {len(config.suites)} ({len(enabled_suites)} enabled)")
        logger.info(f"  Parallel workers: {config.max_parallel_workers}")
        logger.info(f"  Bootstrap resamples: {config.metrics.bootstrap_resamples}")
        logger.info(f"  Confidence intervals: {config.metrics.compute_confidence_intervals}")
        
        return 0
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        return 1


def validate_config_command(args):
    """Validate an evaluation configuration file."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        is_valid, errors = validate_evaluation_config_file(config_path)
        
        if is_valid:
            logger.info(f"Configuration file {config_path} is VALID")
            
            # Load and show warnings
            config = EvaluationConfig.from_yaml(config_path)
            warnings = config.validate_runtime_parameters()
            
            if warnings:
                logger.info("Runtime warnings:")
                for warning in warnings:
                    logger.warning(f"  {warning}")
            else:
                logger.info("No runtime warnings")
            
            return 0
        else:
            logger.error(f"Configuration file {config_path} is INVALID")
            for error in errors:
                logger.error(f"  {error}")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def info_config_command(args):
    """Show information about an evaluation configuration."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        config = EvaluationConfig.from_yaml(config_path)
        
        print(f"Evaluation Configuration: {config_path}")
        print("=" * 60)
        
        # Global settings
        print(f"Parallel evaluation: {config.parallel_evaluation}")
        print(f"Max parallel workers: {config.max_parallel_workers}")
        print(f"Evaluation timeout: {config.evaluation_timeout_hours}h")
        print()
        
        # Suites
        print(f"Evaluation Suites ({len(config.suites)} total):")
        enabled_suites = config.get_enabled_suites()
        for suite_name, suite_config in config.suites.items():
            status = "ENABLED" if suite_config.enabled else "DISABLED"
            print(f"  {suite_name}: {status}")
            print(f"    Seeds per map: {suite_config.seeds_per_map}")
            print(f"    Maps: {len(suite_config.maps)}")
            print(f"    Policy modes: {suite_config.policy_modes}")
            print(f"    Environmental noise: {suite_config.environmental_noise}")
            print(f"    Traffic density: {suite_config.traffic_density}")
        print()
        
        # Metrics
        print("Metrics Configuration:")
        print(f"  Confidence intervals: {config.metrics.compute_confidence_intervals}")
        print(f"  Bootstrap resamples: {config.metrics.bootstrap_resamples}")
        print(f"  Composite score: {config.metrics.use_composite_score}")
        print(f"  Success rate weight: {config.metrics.success_rate_weight}")
        print(f"  Reward weight: {config.metrics.reward_weight}")
        print()
        
        # Statistical analysis
        print("Statistical Analysis:")
        print(f"  Significance level: {config.statistical.significance_level}")
        print(f"  Multiple comparison correction: {config.statistical.multiple_comparison_correction}")
        print(f"  Bootstrap comparisons: {config.statistical.bootstrap_comparisons}")
        print()
        
        # Analysis modules
        print("Analysis Modules:")
        print(f"  Failure analysis: {config.failure_analysis.enabled}")
        print(f"  Robustness analysis: {config.robustness.enabled}")
        print(f"  Champion selection: {config.champion_selection.enabled}")
        print()
        
        # Artifacts
        print("Artifacts:")
        print(f"  Output directory: {config.artifacts.output_directory}")
        print(f"  Keep top K models: {config.artifacts.keep_top_k_models}")
        print(f"  Export CSV: {config.artifacts.export_csv}")
        print(f"  Export plots: {config.artifacts.export_plots}")
        print(f"  Export videos: {config.artifacts.export_videos}")
        print()
        
        # Reproducibility
        print("Reproducibility:")
        print(f"  Fix seed list: {config.reproducibility.fix_seed_list}")
        print(f"  CUDNN deterministic: {config.reproducibility.cudnn_deterministic}")
        print(f"  Log git SHA: {config.reproducibility.log_git_sha}")
        print(f"  Track model versions: {config.reproducibility.track_model_versions}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1


def update_config_command(args):
    """Update specific parameters in an evaluation configuration."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        config = EvaluationConfig.from_yaml(config_path)
        
        # Parse updates from command line arguments
        updates = {}
        
        if args.parallel_workers is not None:
            updates['max_parallel_workers'] = args.parallel_workers
        
        if args.bootstrap_resamples is not None:
            updates['metrics'] = updates.get('metrics', {})
            updates['metrics']['bootstrap_resamples'] = args.bootstrap_resamples
        
        if args.confidence_intervals is not None:
            updates['metrics'] = updates.get('metrics', {})
            updates['metrics']['compute_confidence_intervals'] = args.confidence_intervals
        
        if args.enable_robustness is not None:
            updates['robustness'] = updates.get('robustness', {})
            updates['robustness']['enabled'] = args.enable_robustness
        
        if args.enable_failure_analysis is not None:
            updates['failure_analysis'] = updates.get('failure_analysis', {})
            updates['failure_analysis']['enabled'] = args.enable_failure_analysis
        
        if updates:
            config.update(updates)
            
            # Save updated configuration
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = config_path
            
            config.to_yaml(output_path)
            logger.info(f"Updated configuration saved to {output_path}")
            
            # Show what was updated
            logger.info("Updates applied:")
            for key, value in updates.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("No updates specified")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        return 1


def compare_configs_command(args):
    """Compare two evaluation configurations."""
    config1_path = Path(args.config1)
    config2_path = Path(args.config2)
    
    if not config1_path.exists():
        logger.error(f"Configuration file not found: {config1_path}")
        return 1
    
    if not config2_path.exists():
        logger.error(f"Configuration file not found: {config2_path}")
        return 1
    
    try:
        config1 = EvaluationConfig.from_yaml(config1_path)
        config2 = EvaluationConfig.from_yaml(config2_path)
        
        print(f"Configuration Comparison")
        print("=" * 60)
        print(f"Config 1: {config1_path}")
        print(f"Config 2: {config2_path}")
        print()
        
        # Compare key parameters
        comparisons = [
            ("Parallel workers", config1.max_parallel_workers, config2.max_parallel_workers),
            ("Evaluation timeout", f"{config1.evaluation_timeout_hours}h", f"{config2.evaluation_timeout_hours}h"),
            ("Number of suites", len(config1.suites), len(config2.suites)),
            ("Enabled suites", len(config1.get_enabled_suites()), len(config2.get_enabled_suites())),
            ("Bootstrap resamples", config1.metrics.bootstrap_resamples, config2.metrics.bootstrap_resamples),
            ("Confidence intervals", config1.metrics.compute_confidence_intervals, config2.metrics.compute_confidence_intervals),
            ("Robustness analysis", config1.robustness.enabled, config2.robustness.enabled),
            ("Failure analysis", config1.failure_analysis.enabled, config2.failure_analysis.enabled),
            ("Champion selection", config1.champion_selection.enabled, config2.champion_selection.enabled),
        ]
        
        print("Parameter Comparison:")
        for param_name, val1, val2 in comparisons:
            status = "SAME" if val1 == val2 else "DIFFERENT"
            print(f"  {param_name:20} | {str(val1):15} | {str(val2):15} | {status}")
        
        print()
        
        # Compare suites
        suites1 = set(config1.suites.keys())
        suites2 = set(config2.suites.keys())
        
        common_suites = suites1 & suites2
        only_in_1 = suites1 - suites2
        only_in_2 = suites2 - suites1
        
        print("Suite Comparison:")
        print(f"  Common suites: {sorted(common_suites)}")
        if only_in_1:
            print(f"  Only in config 1: {sorted(only_in_1)}")
        if only_in_2:
            print(f"  Only in config 2: {sorted(only_in_2)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to compare configurations: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluation Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a basic configuration
  python evaluation_config_cli.py create --template basic --output basic_eval.yml
  
  # Validate a configuration file
  python evaluation_config_cli.py validate --config my_config.yml
  
  # Show configuration information
  python evaluation_config_cli.py info --config my_config.yml
  
  # Update configuration parameters
  python evaluation_config_cli.py update --config my_config.yml --parallel-workers 8 --bootstrap-resamples 15000
  
  # Compare two configurations
  python evaluation_config_cli.py compare --config1 basic.yml --config2 comprehensive.yml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new evaluation configuration')
    create_parser.add_argument('--template', choices=['basic', 'comprehensive', 'research'], 
                              default='comprehensive', help='Configuration template to use')
    create_parser.add_argument('--output', '-o', required=True, help='Output configuration file path')
    create_parser.set_defaults(func=create_config_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate an evaluation configuration')
    validate_parser.add_argument('--config', '-c', required=True, help='Configuration file to validate')
    validate_parser.set_defaults(func=validate_config_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show configuration information')
    info_parser.add_argument('--config', '-c', required=True, help='Configuration file to analyze')
    info_parser.set_defaults(func=info_config_command)
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update configuration parameters')
    update_parser.add_argument('--config', '-c', required=True, help='Configuration file to update')
    update_parser.add_argument('--output', '-o', help='Output file (default: overwrite input)')
    update_parser.add_argument('--parallel-workers', type=int, help='Number of parallel workers')
    update_parser.add_argument('--bootstrap-resamples', type=int, help='Number of bootstrap resamples')
    update_parser.add_argument('--confidence-intervals', type=bool, help='Enable/disable confidence intervals')
    update_parser.add_argument('--enable-robustness', type=bool, help='Enable/disable robustness analysis')
    update_parser.add_argument('--enable-failure-analysis', type=bool, help='Enable/disable failure analysis')
    update_parser.set_defaults(func=update_config_command)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two configurations')
    compare_parser.add_argument('--config1', required=True, help='First configuration file')
    compare_parser.add_argument('--config2', required=True, help='Second configuration file')
    compare_parser.set_defaults(func=compare_configs_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())