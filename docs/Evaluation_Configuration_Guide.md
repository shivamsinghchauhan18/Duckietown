# Evaluation Configuration Guide

## Overview

This guide provides comprehensive documentation for configuring the Enhanced Duckietown RL evaluation system. The configuration system uses YAML files with extensive validation to ensure reproducible and reliable evaluations.

## Configuration Structure

### Main Configuration File

The main evaluation configuration is defined in `EvaluationConfig` and can be loaded from YAML files:

```yaml
# evaluation_config.yml
suite_configuration:
  suites: ['base', 'hard', 'law', 'ood', 'stress']
  seeds_per_map: 50
  policy_modes: ['deterministic', 'stochastic']
  timeout_seconds: 300

metrics_configuration:
  compute_ci: true
  bootstrap_resamples: 10000
  confidence_level: 0.95
  significance_correction: 'benjamini_hochberg'
  
scoring_configuration:
  use_composite: true
  normalization_scope: 'per_map_suite'
  composite_weights:
    success_rate: 0.45
    mean_reward: 0.25
    episode_length: 0.10
    lateral_deviation: 0.08
    heading_error: 0.06
    smoothness: 0.06
  pareto_axes:
    - ['SR', '-D', '-J']  # Success Rate vs Lateral Deviation vs Smoothness
    - ['SR', 'R']         # Success Rate vs Reward

artifact_configuration:
  keep_top_k: 5
  export_csv_json: true
  export_plots: true
  record_videos: true
  save_worst_k: 5
  compression_level: 6

reproducibility_configuration:
  fix_seed_list: true
  cudnn_deterministic: true
  log_git_sha: true
  log_environment: true
```

## Configuration Sections

### Suite Configuration

Controls which test suites to run and their execution parameters.

#### Parameters

##### suites
- **Type**: `List[str]`
- **Default**: `['base', 'hard', 'law', 'ood']`
- **Description**: List of evaluation suites to execute
- **Valid Values**: 
  - `'base'`: Clean environmental conditions
  - `'hard'`: Heavy randomization with noise and traffic
  - `'law'`: Traffic rule compliance testing
  - `'ood'`: Out-of-distribution conditions
  - `'stress'`: Adversarial conditions with sensor failures

**Example**:
```yaml
suite_configuration:
  suites: ['base', 'hard', 'ood']  # Run only these three suites
```

##### seeds_per_map
- **Type**: `int`
- **Default**: `50`
- **Range**: `10-1000`
- **Description**: Number of random seeds to use per map for statistical reliability
- **Recommendation**: Minimum 25 for basic evaluation, 50+ for publication-quality results

**Example**:
```yaml
suite_configuration:
  seeds_per_map: 100  # High statistical power
```

##### policy_modes
- **Type**: `List[str]`
- **Default**: `['deterministic', 'stochastic']`
- **Description**: Policy execution modes to test
- **Valid Values**:
  - `'deterministic'`: Use mean action from policy
  - `'stochastic'`: Sample actions from policy distribution

**Example**:
```yaml
suite_configuration:
  policy_modes: ['deterministic']  # Test only deterministic mode
```

##### timeout_seconds
- **Type**: `int`
- **Default**: `300`
- **Range**: `60-3600`
- **Description**: Maximum time allowed per episode before timeout

### Metrics Configuration

Controls metric calculation and statistical analysis parameters.

#### Parameters

##### compute_ci
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to compute confidence intervals for metrics
- **Impact**: Disabling improves performance but reduces statistical rigor

##### bootstrap_resamples
- **Type**: `int`
- **Default**: `10000`
- **Range**: `1000-100000`
- **Description**: Number of bootstrap samples for confidence interval estimation
- **Recommendation**: 10000 for good accuracy, 1000 for faster computation

##### confidence_level
- **Type**: `float`
- **Default**: `0.95`
- **Range**: `0.80-0.99`
- **Description**: Confidence level for interval estimation (e.g., 0.95 = 95% CI)

##### significance_correction
- **Type**: `str`
- **Default**: `'benjamini_hochberg'`
- **Valid Values**: `'benjamini_hochberg'`, `'bonferroni'`, `'none'`
- **Description**: Multiple comparison correction method
- **Recommendation**: Use `'benjamini_hochberg'` for good power with FDR control

**Example**:
```yaml
metrics_configuration:
  compute_ci: true
  bootstrap_resamples: 5000  # Faster computation
  confidence_level: 0.90     # 90% confidence intervals
  significance_correction: 'bonferroni'  # Conservative correction
```

### Scoring Configuration

Controls composite score calculation and model ranking.

#### Parameters

##### use_composite
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to calculate composite scores for ranking
- **Note**: Required for champion selection

##### normalization_scope
- **Type**: `str`
- **Default**: `'per_map_suite'`
- **Valid Values**: `'global'`, `'per_suite'`, `'per_map'`, `'per_map_suite'`
- **Description**: Scope for metric normalization
- **Impact**:
  - `'global'`: Normalize across all maps and suites
  - `'per_suite'`: Normalize within each suite
  - `'per_map'`: Normalize within each map
  - `'per_map_suite'`: Normalize within each map-suite combination

##### composite_weights
- **Type**: `Dict[str, float]`
- **Description**: Weights for composite score calculation
- **Constraint**: Must sum to 1.0
- **Default Weights**:
  - `success_rate`: 0.45 (Primary importance)
  - `mean_reward`: 0.25 (Secondary importance)
  - `episode_length`: 0.10
  - `lateral_deviation`: 0.08
  - `heading_error`: 0.06
  - `smoothness`: 0.06

**Custom Weights Example**:
```yaml
scoring_configuration:
  composite_weights:
    success_rate: 0.60      # Emphasize success
    mean_reward: 0.20
    lateral_deviation: 0.15  # Emphasize precision
    episode_length: 0.05
    heading_error: 0.00     # Ignore heading
    smoothness: 0.00        # Ignore smoothness
```

##### pareto_axes
- **Type**: `List[List[str]]`
- **Description**: Objective combinations for Pareto front analysis
- **Format**: Each inner list defines objectives for one Pareto front
- **Prefix Convention**: Use `-` prefix for objectives to minimize (e.g., `-D` for lateral deviation)

**Example**:
```yaml
scoring_configuration:
  pareto_axes:
    - ['SR', '-D']          # Success Rate vs Lateral Deviation
    - ['SR', 'R', '-J']     # Success Rate vs Reward vs Smoothness
    - ['-D', '-H', '-J']    # Precision-focused Pareto front
```

### Artifact Configuration

Controls output generation and storage.

#### Parameters

##### keep_top_k
- **Type**: `int`
- **Default**: `5`
- **Range**: `1-50`
- **Description**: Number of top-performing models to keep detailed artifacts for

##### export_csv_json
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to export episode-level data in CSV/JSON format

##### export_plots
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to generate visualization plots

##### record_videos
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to record videos of episodes
- **Impact**: Significantly increases storage requirements

##### save_worst_k
- **Type**: `int`
- **Default**: `5`
- **Range**: `0-20`
- **Description**: Number of worst-performing episodes to save videos for

##### compression_level
- **Type**: `int`
- **Default**: `6`
- **Range**: `0-9`
- **Description**: Compression level for artifact storage (0=none, 9=maximum)

### Reproducibility Configuration

Ensures reproducible evaluation results.

#### Parameters

##### fix_seed_list
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to use fixed seed lists for reproducibility
- **Impact**: Ensures identical results across runs

##### cudnn_deterministic
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to enable deterministic CUDA operations
- **Impact**: Reduces performance but ensures reproducibility on GPU

##### log_git_sha
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to log Git SHA for version tracking

##### log_environment
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to log environment information (Python version, packages, etc.)

## Suite-Specific Configuration

### Base Suite Configuration

```yaml
base_suite:
  lighting_intensity: 1.0
  texture_domain: 'default'
  camera_noise: 0.0
  friction_coefficient: 1.0
  traffic_density: 0.0
  spawn_pose_noise: 0.0
```

### Hard Randomization Suite Configuration

```yaml
hard_suite:
  lighting_intensity_range: [0.3, 1.7]
  texture_domain: 'heavy_randomization'
  camera_noise: 0.15
  friction_coefficient_range: [0.7, 1.3]
  traffic_density: 0.3
  spawn_pose_noise: 0.2
```

### Law/Intersection Suite Configuration

```yaml
law_suite:
  include_stop_signs: true
  include_intersections: true
  include_right_of_way: true
  traffic_light_probability: 0.5
  pedestrian_probability: 0.2
```

### Out-of-Distribution Suite Configuration

```yaml
ood_suite:
  texture_domain: 'unseen'
  lighting_conditions: ['night', 'rain', 'fog']
  sensor_noise_multiplier: 2.0
  camera_angle_offset: 15.0  # degrees
  wheel_bias: 0.1
```

### Stress/Adversarial Suite Configuration

```yaml
stress_suite:
  sensor_dropout_probability: 0.1
  wheel_bias_range: [-0.2, 0.2]
  moving_obstacle_density: 0.4
  emergency_brake_probability: 0.05
  communication_delay: 0.1  # seconds
```

## Configuration Templates

### Quick Evaluation Template

For rapid model testing during development:

```yaml
# quick_evaluation.yml
suite_configuration:
  suites: ['base']
  seeds_per_map: 10
  policy_modes: ['deterministic']
  timeout_seconds: 120

metrics_configuration:
  compute_ci: false
  bootstrap_resamples: 1000

artifact_configuration:
  export_plots: false
  record_videos: false
  save_worst_k: 0
```

### Research Publication Template

For rigorous evaluation suitable for research publication:

```yaml
# research_evaluation.yml
suite_configuration:
  suites: ['base', 'hard', 'law', 'ood', 'stress']
  seeds_per_map: 100
  policy_modes: ['deterministic', 'stochastic']
  timeout_seconds: 300

metrics_configuration:
  compute_ci: true
  bootstrap_resamples: 10000
  confidence_level: 0.95
  significance_correction: 'benjamini_hochberg'

scoring_configuration:
  use_composite: true
  normalization_scope: 'per_map_suite'
  pareto_axes:
    - ['SR', '-D', '-J']
    - ['SR', 'R']
    - ['-D', '-H']

artifact_configuration:
  keep_top_k: 10
  export_csv_json: true
  export_plots: true
  record_videos: true
  save_worst_k: 10

reproducibility_configuration:
  fix_seed_list: true
  cudnn_deterministic: true
  log_git_sha: true
  log_environment: true
```

### Deployment Readiness Template

For evaluating models before production deployment:

```yaml
# deployment_evaluation.yml
suite_configuration:
  suites: ['base', 'hard', 'ood', 'stress']
  seeds_per_map: 75
  policy_modes: ['deterministic']
  timeout_seconds: 180

metrics_configuration:
  compute_ci: true
  bootstrap_resamples: 5000
  confidence_level: 0.99  # High confidence for safety

scoring_configuration:
  composite_weights:
    success_rate: 0.70      # Emphasize safety
    lateral_deviation: 0.15  # Emphasize precision
    mean_reward: 0.10
    smoothness: 0.05
    episode_length: 0.00
    heading_error: 0.00

# Strict acceptance criteria
acceptance_criteria:
  min_success_rate: 0.95
  max_lateral_deviation: 0.1
  min_maps_passing: 0.90
```

## Configuration Validation

### Automatic Validation

The system automatically validates configuration parameters:

```python
from config.evaluation_config import EvaluationConfig, validate_config

# Load and validate configuration
config = EvaluationConfig.from_yaml('my_config.yml')
validation_result = validate_config(config)

if not validation_result.is_valid:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Common Validation Errors

1. **Weight Sum Error**: Composite weights don't sum to 1.0
2. **Invalid Suite Name**: Unknown suite specified
3. **Parameter Range Error**: Parameter outside valid range
4. **Missing Required Field**: Required configuration field not provided
5. **Type Mismatch**: Parameter has wrong type

### Validation Example

```python
# This will raise validation errors:
invalid_config = {
    'suite_configuration': {
        'suites': ['base', 'invalid_suite'],  # Error: invalid suite
        'seeds_per_map': 5,                   # Error: too few seeds
    },
    'scoring_configuration': {
        'composite_weights': {
            'success_rate': 0.5,              # Error: weights don't sum to 1.0
            'mean_reward': 0.3,
        }
    }
}
```

## Environment Variables

Some configuration can be overridden with environment variables:

```bash
# Override default configuration file
export EVALUATION_CONFIG_PATH="/path/to/custom_config.yml"

# Override number of parallel workers
export EVALUATION_WORKERS=8

# Override output directory
export EVALUATION_OUTPUT_DIR="/path/to/results"

# Enable debug logging
export EVALUATION_DEBUG=true

# Override GPU device
export EVALUATION_DEVICE="cuda:1"
```

## Configuration Best Practices

### Performance Optimization

1. **Reduce Seeds for Development**: Use 10-25 seeds during development, 50+ for final evaluation
2. **Disable Expensive Features**: Turn off video recording and CI computation for quick tests
3. **Parallel Execution**: Use multiple workers for suite execution
4. **Memory Management**: Monitor memory usage with large seed counts

### Statistical Rigor

1. **Sufficient Seeds**: Use at least 25 seeds per map for reliable statistics
2. **Multiple Comparison Correction**: Always use correction for multiple model comparisons
3. **Confidence Intervals**: Enable CI computation for publication-quality results
4. **Reproducibility**: Always enable reproducibility settings for research

### Storage Management

1. **Selective Video Recording**: Only record videos for worst episodes to save space
2. **Compression**: Use appropriate compression levels for artifacts
3. **Cleanup**: Regularly clean up old evaluation results
4. **Top-K Selection**: Limit detailed artifacts to top-performing models

### Configuration Organization

1. **Template Usage**: Start with appropriate templates and customize
2. **Version Control**: Keep configuration files in version control
3. **Documentation**: Document custom configurations and their purpose
4. **Validation**: Always validate configurations before long-running evaluations

## Troubleshooting Configuration Issues

### Common Issues and Solutions

#### Issue: "Composite weights don't sum to 1.0"
```yaml
# Problem:
composite_weights:
  success_rate: 0.5
  mean_reward: 0.3  # Sum = 0.8, not 1.0

# Solution:
composite_weights:
  success_rate: 0.5
  mean_reward: 0.3
  lateral_deviation: 0.2  # Now sums to 1.0
```

#### Issue: "Invalid suite name"
```yaml
# Problem:
suites: ['base', 'custom_suite']  # 'custom_suite' not recognized

# Solution:
suites: ['base', 'hard', 'ood']  # Use valid suite names
```

#### Issue: "Insufficient seeds for statistical analysis"
```yaml
# Problem:
seeds_per_map: 5  # Too few for reliable statistics

# Solution:
seeds_per_map: 25  # Minimum recommended
```

### Debug Configuration

Enable debug mode for detailed configuration information:

```yaml
debug_configuration:
  verbose_logging: true
  log_configuration: true
  validate_on_load: true
  print_warnings: true
```

This comprehensive configuration guide provides all the information needed to properly configure the evaluation system for different use cases and requirements.