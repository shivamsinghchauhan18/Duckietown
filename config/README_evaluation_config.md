# Evaluation Configuration Management System

The Evaluation Configuration Management System provides comprehensive parameter validation, YAML loading, and configuration templates for the Master Evaluation Orchestrator in the Enhanced Duckietown RL project.

## Features

- **Comprehensive Parameter Validation**: All configuration parameters are validated with meaningful error messages
- **YAML Configuration Loading**: Load and save configurations from/to YAML files with schema validation
- **Configuration Templates**: Pre-built templates for different evaluation scenarios (basic, comprehensive, research)
- **Runtime Parameter Updates**: Update configuration parameters at runtime with validation
- **Configuration Utilities**: Helper functions for loading, validating, and managing configurations
- **Command-Line Interface**: CLI tools for configuration management and validation

## Quick Start

### Creating a Configuration

```python
from config.evaluation_config import create_comprehensive_evaluation_config

# Create a comprehensive evaluation configuration
config = create_comprehensive_evaluation_config()

# Save to YAML file
config.to_yaml("my_evaluation_config.yml")
```

### Loading from YAML

```python
from config.evaluation_config import EvaluationConfig

# Load configuration from YAML file
config = EvaluationConfig.from_yaml("my_evaluation_config.yml")

# Get enabled suites
enabled_suites = config.get_enabled_suites()
print(f"Enabled suites: {enabled_suites}")
```

### Using Templates

```python
from config.evaluation_config import (
    create_basic_evaluation_config,
    create_comprehensive_evaluation_config,
    create_research_evaluation_config
)

# Create different configuration templates
basic_config = create_basic_evaluation_config()
comprehensive_config = create_comprehensive_evaluation_config()
research_config = create_research_evaluation_config()
```

## Configuration Structure

The evaluation configuration is organized into several main components:

### Global Settings
- `parallel_evaluation`: Enable/disable parallel evaluation
- `max_parallel_workers`: Number of parallel workers
- `evaluation_timeout_hours`: Maximum evaluation time

### Evaluation Suites
Each suite defines a set of evaluation conditions:
- `name`: Suite identifier
- `enabled`: Whether the suite is active
- `seeds_per_map`: Number of random seeds per map
- `maps`: List of maps to evaluate on
- `policy_modes`: Policy modes to test (deterministic/stochastic)
- `environmental_noise`: Environmental noise level (0.0-1.0)
- `traffic_density`: Traffic density level (0.0-1.0)
- Various other environmental parameters

### Metrics Configuration
- `compute_confidence_intervals`: Enable confidence interval calculation
- `bootstrap_resamples`: Number of bootstrap resamples
- Composite score weights for different metrics
- `normalization_scope`: How to normalize metrics

### Statistical Analysis
- `significance_level`: Statistical significance threshold
- `multiple_comparison_correction`: Correction method for multiple comparisons
- `bootstrap_comparisons`: Enable bootstrap-based comparisons

### Analysis Modules
- **Failure Analysis**: Configuration for failure classification and analysis
- **Robustness Analysis**: Parameter sweeps and sensitivity analysis
- **Champion Selection**: Model ranking and selection criteria

### Artifacts and Reproducibility
- Output directory and file management settings
- Reproducibility settings for deterministic evaluation

## Configuration Templates

### Basic Template
Minimal configuration for quick testing:
- Single evaluation suite (base)
- Reduced number of seeds
- Disabled advanced features
- Fast execution

```python
config = create_basic_evaluation_config()
```

### Comprehensive Template
Full-featured configuration for thorough assessment:
- Multiple evaluation suites (base, hard, law, ood, stress)
- Full statistical analysis
- All analysis modules enabled
- Production-ready settings

```python
config = create_comprehensive_evaluation_config()
```

### Research Template
High-rigor configuration for research and publication:
- Large number of seeds for statistical power
- Stringent significance levels
- Comprehensive reproducibility tracking
- Extended evaluation suites

```python
config = create_research_evaluation_config()
```

## Command-Line Interface

The system includes a CLI for configuration management:

### Create Configuration
```bash
python config/evaluation_config_cli.py create --template comprehensive --output my_config.yml
```

### Validate Configuration
```bash
python config/evaluation_config_cli.py validate --config my_config.yml
```

### Show Configuration Info
```bash
python config/evaluation_config_cli.py info --config my_config.yml
```

### Update Configuration
```bash
python config/evaluation_config_cli.py update --config my_config.yml --parallel-workers 8 --bootstrap-resamples 15000
```

### Compare Configurations
```bash
python config/evaluation_config_cli.py compare --config1 basic.yml --config2 comprehensive.yml
```

## Configuration Validation

The system provides comprehensive validation at multiple levels:

### Schema Validation
All configurations are validated against a JSON schema that defines:
- Required and optional parameters
- Parameter types and ranges
- Valid enumeration values

### Parameter Validation
Individual configuration classes validate their parameters:
- Range checks for numerical values
- Enumeration validation for categorical values
- Cross-parameter consistency checks

### Runtime Validation
Additional validation checks for runtime considerations:
- Resource availability (CPU cores, memory)
- Statistical reliability (minimum sample sizes)
- Configuration completeness

## Error Handling

The system provides detailed error messages for common issues:

```python
try:
    config = EvaluationConfig.from_yaml("config.yml")
except FileNotFoundError:
    print("Configuration file not found")
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
```

## Examples

### Custom Suite Configuration

```python
from config.evaluation_config import SuiteConfig, EvaluationConfig

# Create custom suite
custom_suite = SuiteConfig(
    name="custom_test",
    seeds_per_map=25,
    maps=["loop_empty", "small_loop"],
    environmental_noise=0.3,
    traffic_density=0.1
)

# Create configuration with custom suite
config = EvaluationConfig(
    suites={"custom_test": custom_suite},
    max_parallel_workers=2
)
```

### Runtime Configuration Updates

```python
# Load existing configuration
config = EvaluationConfig.from_yaml("config.yml")

# Update parameters
updates = {
    'max_parallel_workers': 8,
    'metrics': {
        'bootstrap_resamples': 15000,
        'compute_confidence_intervals': True
    }
}

config.update(updates)
config.to_yaml("updated_config.yml")
```

### Configuration Validation

```python
from config.evaluation_config import validate_evaluation_config_file

# Validate configuration file
is_valid, errors = validate_evaluation_config_file("config.yml")

if is_valid:
    print("Configuration is valid")
else:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

## Integration with Evaluation Orchestrator

The configuration system is designed to integrate seamlessly with the Master Evaluation Orchestrator:

```python
from config.evaluation_config import load_evaluation_config
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator

# Load configuration
config = load_evaluation_config("evaluation_config.yml")

# Create orchestrator with configuration
orchestrator = EvaluationOrchestrator(config)

# Run evaluation
results = orchestrator.evaluate_models(model_paths)
```

## Best Practices

### Configuration Management
1. Use version control for configuration files
2. Document configuration changes and rationale
3. Validate configurations before running evaluations
4. Use templates as starting points for custom configurations

### Parameter Selection
1. Start with appropriate templates for your use case
2. Adjust parameters based on available computational resources
3. Consider statistical power when setting sample sizes
4. Balance evaluation thoroughness with execution time

### Reproducibility
1. Enable reproducibility settings for research
2. Track configuration versions and git commits
3. Document evaluation environment and dependencies
4. Archive configurations with evaluation results

## Troubleshooting

### Common Issues

**Configuration validation fails**
- Check parameter ranges and types
- Ensure required parameters are specified
- Verify enumeration values are valid

**Runtime warnings about statistical reliability**
- Increase number of seeds per map
- Increase bootstrap resamples for confidence intervals
- Consider using more stringent significance levels

**Resource-related warnings**
- Adjust parallel workers based on available CPU cores
- Monitor memory usage during evaluation
- Consider reducing evaluation scope for resource-constrained environments

### Getting Help

1. Check configuration validation messages for specific issues
2. Use the CLI info command to inspect configuration details
3. Review example configurations and templates
4. Consult the evaluation orchestrator documentation for integration issues

## API Reference

### Main Classes

- `EvaluationConfig`: Main configuration class
- `SuiteConfig`: Individual evaluation suite configuration
- `MetricsConfig`: Metrics calculation configuration
- `StatisticalConfig`: Statistical analysis configuration
- `FailureAnalysisConfig`: Failure analysis configuration
- `RobustnessConfig`: Robustness analysis configuration
- `ChampionSelectionConfig`: Champion selection configuration
- `ArtifactConfig`: Artifact management configuration
- `ReproducibilityConfig`: Reproducibility settings

### Utility Functions

- `create_basic_evaluation_config()`: Create basic template
- `create_comprehensive_evaluation_config()`: Create comprehensive template
- `create_research_evaluation_config()`: Create research template
- `load_evaluation_config()`: Load configuration from file or template
- `validate_evaluation_config_file()`: Validate configuration file

### CLI Commands

- `create`: Create new configuration from template
- `validate`: Validate configuration file
- `info`: Show configuration information
- `update`: Update configuration parameters
- `compare`: Compare two configurations

For detailed API documentation, see the docstrings in the source code.