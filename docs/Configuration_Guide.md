# Enhanced Duckietown RL Configuration Guide

## Table of Contents
1. [Configuration Overview](#configuration-overview)
2. [YOLO Detection Configuration](#yolo-detection-configuration)
3. [Object Avoidance Configuration](#object-avoidance-configuration)
4. [Lane Changing Configuration](#lane-changing-configuration)
5. [Reward System Configuration](#reward-system-configuration)
6. [Logging Configuration](#logging-configuration)
7. [Performance Configuration](#performance-configuration)
8. [Environment-Specific Configurations](#environment-specific-configurations)
9. [Troubleshooting Configuration Issues](#troubleshooting-configuration-issues)

## Configuration Overview

The Enhanced Duckietown RL system uses YAML configuration files to manage all system parameters. The configuration system is hierarchical and supports parameter validation, default values, and environment-specific overrides.

### Configuration File Structure
```yaml
# Main configuration sections
yolo:           # YOLO object detection settings
  model_path: "models/yolo5s.pt"
  confidence_threshold: 0.5
  device: "cuda"

object_avoidance:  # Object avoidance behavior settings
  safety_distance: 0.5
  avoidance_strength: 1.0

lane_changing:     # Lane changing behavior settings
  lane_change_threshold: 0.3
  safety_margin: 2.0

rewards:          # Multi-objective reward settings
  weights:
    lane_following: 1.0
    object_avoidance: 2.0

logging:          # Logging and debugging settings
  level: "INFO"
  log_detections: true

performance:      # Performance optimization settings
  max_fps: 30
  gpu_memory_fraction: 0.8
```

### Loading Configuration
```python
from config.enhanced_config import EnhancedRLConfig

# Load from default file
config = EnhancedRLConfig.from_yaml('config/enhanced_config.yml')

# Load from custom file
config = EnhancedRLConfig.from_yaml('my_custom_config.yml')

# Override specific parameters
config.yolo.confidence_threshold = 0.7
config.rewards.weights['safety'] = -10.0

# Validate configuration
config.validate()
```

## YOLO Detection Configuration

### Core YOLO Parameters

#### `model_path` (string, required)
**Description**: Path to the YOLO v5 model file (.pt format)

**Options**:
- `"models/yolo5n.pt"` - Nano model (fastest, least accurate)
- `"models/yolo5s.pt"` - Small model (balanced speed/accuracy) **[Recommended]**
- `"models/yolo5m.pt"` - Medium model (good accuracy, moderate speed)
- `"models/yolo5l.pt"` - Large model (high accuracy, slower)
- `"models/yolo5x.pt"` - Extra large model (highest accuracy, slowest)

**Recommendations**:
- Use `yolo5s.pt` for real-time applications
- Use `yolo5m.pt` or larger for offline analysis
- Ensure model file exists and is accessible

```yaml
yolo:
  model_path: "models/yolo5s.pt"  # Recommended for real-time use
```

#### `confidence_threshold` (float, 0.0-1.0, default: 0.5)
**Description**: Minimum confidence score for object detections

**Impact**:
- **Lower values (0.3-0.5)**: More detections, including uncertain ones
- **Higher values (0.6-0.8)**: Fewer, more confident detections
- **Very high values (0.8+)**: Only very confident detections

**Recommendations**:
- Start with 0.5 for balanced performance
- Increase to 0.6-0.7 for safety-critical applications
- Decrease to 0.3-0.4 for environments with poor lighting

```yaml
yolo:
  confidence_threshold: 0.6  # Conservative setting for safety
```

#### `device` (string, default: "cuda")
**Description**: Device for YOLO inference

**Options**:
- `"cuda"` - Use GPU acceleration (recommended if available)
- `"cpu"` - Use CPU inference (slower but always available)
- `"cuda:0"` - Specific GPU device

**Performance Impact**:
- GPU: ~10-50ms per frame
- CPU: ~100-500ms per frame

```yaml
yolo:
  device: "cuda"  # Use GPU if available
```

#### `max_detections` (integer, default: 10)
**Description**: Maximum number of detections to process per frame

**Recommendations**:
- Use 5-10 for simple environments
- Use 15-20 for complex environments with many objects
- Higher values increase processing time

```yaml
yolo:
  max_detections: 15  # For complex environments
```

### Advanced YOLO Parameters

#### `input_size` (integer, default: 640)
**Description**: Input image size for YOLO processing

**Trade-offs**:
- **Smaller (320-416)**: Faster processing, lower accuracy
- **Standard (640)**: Balanced performance **[Recommended]**
- **Larger (832-1280)**: Higher accuracy, slower processing

```yaml
yolo:
  input_size: 640  # Standard size
```

#### `nms_threshold` (float, 0.0-1.0, default: 0.45)
**Description**: Non-maximum suppression threshold for removing duplicate detections

**Recommendations**:
- Use 0.45 for standard applications
- Increase to 0.6 if missing closely spaced objects
- Decrease to 0.3 if getting too many duplicate detections

```yaml
yolo:
  nms_threshold: 0.45  # Standard setting
```

## Object Avoidance Configuration

### Core Avoidance Parameters

#### `safety_distance` (float, meters, default: 0.5)
**Description**: Distance threshold for triggering avoidance behavior

**Recommendations**:
- **Conservative (0.7-1.0m)**: Safe for high-speed scenarios
- **Standard (0.5-0.7m)**: Balanced for most applications **[Recommended]**
- **Aggressive (0.3-0.5m)**: For tight spaces, requires precise detection

```yaml
object_avoidance:
  safety_distance: 0.6  # Conservative setting
```

#### `avoidance_strength` (float, 0.0-3.0, default: 1.0)
**Description**: Strength of avoidance forces applied to actions

**Impact**:
- **Low (0.5-0.8)**: Gentle avoidance, may not avoid fast-approaching objects
- **Medium (1.0-1.5)**: Standard avoidance **[Recommended]**
- **High (2.0-3.0)**: Strong avoidance, may cause jerky movements

```yaml
object_avoidance:
  avoidance_strength: 1.2  # Slightly stronger than default
```

#### `min_clearance` (float, meters, default: 0.2)
**Description**: Minimum clearance to maintain from objects

**Recommendations**:
- Should be less than `safety_distance`
- Use 0.15-0.25m for most scenarios
- Increase for larger robots or uncertain detection

```yaml
object_avoidance:
  min_clearance: 0.25  # Conservative clearance
```

### Advanced Avoidance Parameters

#### `smoothing_factor` (float, 0.0-1.0, default: 0.3)
**Description**: Action smoothing to prevent oscillations

**Impact**:
- **Low (0.1-0.2)**: Responsive but may oscillate
- **Medium (0.3-0.5)**: Balanced smoothness **[Recommended]**
- **High (0.6-0.8)**: Smooth but slow response

```yaml
object_avoidance:
  smoothing_factor: 0.4  # Smooth movements
```

#### `priority_weights` (dict, default: see below)
**Description**: Priority weights for different object classes

```yaml
object_avoidance:
  priority_weights:
    person: 3.0      # Highest priority
    bicycle: 2.5
    car: 2.0
    truck: 2.0
    duckiebot: 1.5
    cone: 1.0        # Lowest priority
```

## Lane Changing Configuration

### Core Lane Changing Parameters

#### `lane_change_threshold` (float, 0.0-1.0, default: 0.3)
**Description**: Threshold for initiating lane changes based on obstacle blocking

**Recommendations**:
- **Conservative (0.2-0.3)**: Change lanes early when obstacles detected
- **Standard (0.3-0.4)**: Balanced approach **[Recommended]**
- **Aggressive (0.4-0.6)**: Only change when significantly blocked

```yaml
lane_changing:
  lane_change_threshold: 0.35  # Slightly conservative
```

#### `safety_margin` (float, meters, default: 2.0)
**Description**: Required clear distance in target lane for safe lane change

**Recommendations**:
- **Conservative (2.5-3.0m)**: Very safe, may miss opportunities
- **Standard (2.0-2.5m)**: Balanced safety **[Recommended]**
- **Aggressive (1.5-2.0m)**: More opportunities, higher risk

```yaml
lane_changing:
  safety_margin: 2.2  # Conservative safety margin
```

#### `max_lane_change_time` (float, seconds, default: 3.0)
**Description**: Maximum time allowed to complete a lane change

**Recommendations**:
- Use 2.5-3.5 seconds for most scenarios
- Increase for slower robots or complex maneuvers
- Decrease for faster, more agile robots

```yaml
lane_changing:
  max_lane_change_time: 3.5  # Allow extra time
```

### Advanced Lane Changing Parameters

#### `evaluation_frequency` (integer, steps, default: 10)
**Description**: How often to evaluate lane changing opportunities

**Trade-offs**:
- **Frequent (5-8 steps)**: Responsive but higher computational cost
- **Standard (10-15 steps)**: Balanced **[Recommended]**
- **Infrequent (20+ steps)**: Lower cost but may miss opportunities

```yaml
lane_changing:
  evaluation_frequency: 12  # Slightly less frequent
```

#### `abort_conditions` (dict)
**Description**: Conditions for aborting lane changes

```yaml
lane_changing:
  abort_conditions:
    max_lateral_deviation: 0.5  # meters
    min_forward_progress: 0.1   # m/s
    safety_violation: true      # abort on any safety issue
```

## Reward System Configuration

### Reward Weights

#### Core Reward Components
```yaml
rewards:
  weights:
    lane_following: 1.0      # Base lane following reward
    object_avoidance: 2.0    # Reward for avoiding objects
    lane_changing: 1.5       # Reward for successful lane changes
    efficiency: 0.5          # Reward for forward progress
    safety: -5.0             # Penalty for safety violations (negative)
```

**Tuning Guidelines**:

**Lane Following Weight (0.5-2.0)**
- **High (1.5-2.0)**: Prioritizes staying in lane
- **Medium (1.0-1.5)**: Balanced approach **[Recommended]**
- **Low (0.5-1.0)**: Allows more deviation for other objectives

**Object Avoidance Weight (1.0-3.0)**
- **High (2.5-3.0)**: Very conservative, avoids all objects
- **Medium (1.5-2.5)**: Balanced avoidance **[Recommended]**
- **Low (1.0-1.5)**: More aggressive, closer approaches

**Lane Changing Weight (0.5-2.0)**
- **High (1.5-2.0)**: Encourages lane changes when beneficial
- **Medium (1.0-1.5)**: Balanced **[Recommended]**
- **Low (0.5-1.0)**: Conservative, fewer lane changes

**Safety Penalty (-10.0 to -2.0)**
- **Strong (-8.0 to -10.0)**: Very risk-averse
- **Medium (-5.0 to -8.0)**: Balanced safety **[Recommended]**
- **Weak (-2.0 to -5.0)**: More risk-tolerant

### Advanced Reward Parameters

#### `safety_penalty_scale` (float, default: 10.0)
**Description**: Multiplier for safety penalty calculations

```yaml
rewards:
  safety_penalty_scale: 12.0  # Stronger safety penalties
```

#### `efficiency_bonus_scale` (float, default: 1.0)
**Description**: Multiplier for efficiency bonus calculations

```yaml
rewards:
  efficiency_bonus_scale: 1.2  # Encourage faster progress
```

#### `reward_normalization` (boolean, default: true)
**Description**: Whether to normalize reward components

```yaml
rewards:
  reward_normalization: true  # Recommended for stable training
```

## Logging Configuration

### Core Logging Parameters

#### `level` (string, default: "INFO")
**Description**: Logging level for system messages

**Options**:
- `"DEBUG"` - Detailed debugging information
- `"INFO"` - General information **[Recommended for training]**
- `"WARNING"` - Only warnings and errors
- `"ERROR"` - Only errors

```yaml
logging:
  level: "INFO"  # Standard logging level
```

#### `output_dir` (string, default: "logs")
**Description**: Directory for log files

```yaml
logging:
  output_dir: "logs/experiment_1"  # Organized by experiment
```

### Specific Logging Controls

#### Detection Logging
```yaml
logging:
  log_detections: true          # Log object detections
  detection_log_frequency: 1    # Log every N detections
  include_detection_images: false  # Save detection images (large files)
```

#### Action Logging
```yaml
logging:
  log_actions: true            # Log action decisions
  log_action_reasoning: true   # Include reasoning for actions
  action_log_frequency: 1      # Log every N actions
```

#### Reward Logging
```yaml
logging:
  log_rewards: true           # Log reward calculations
  log_reward_components: true # Log individual reward components
  reward_log_frequency: 1     # Log every N rewards
```

#### Performance Logging
```yaml
logging:
  log_performance: true       # Log performance metrics
  performance_log_frequency: 10  # Log every N steps
  include_system_metrics: true   # Include CPU/GPU usage
```

## Performance Configuration

### Core Performance Parameters

#### `max_fps` (integer, default: 30)
**Description**: Maximum frames per second for environment

**Recommendations**:
- **High (30+ FPS)**: Smooth real-time operation
- **Medium (15-30 FPS)**: Balanced performance **[Recommended]**
- **Low (10-15 FPS)**: Acceptable for training, not real-time

```yaml
performance:
  max_fps: 25  # Slightly reduced for stability
```

#### `gpu_memory_fraction` (float, 0.1-1.0, default: 0.8)
**Description**: Fraction of GPU memory to allocate

**Recommendations**:
- Use 0.6-0.8 for single model training
- Use 0.4-0.6 when running multiple processes
- Monitor GPU memory usage and adjust accordingly

```yaml
performance:
  gpu_memory_fraction: 0.7  # Conservative allocation
```

### Advanced Performance Parameters

#### `batch_processing` (boolean, default: false)
**Description**: Enable batch processing for multiple environments

```yaml
performance:
  batch_processing: true     # For parallel training
  batch_size: 4             # Number of environments per batch
```

#### `optimization_level` (string, default: "balanced")
**Description**: Performance optimization level

**Options**:
- `"speed"` - Optimize for speed over accuracy
- `"balanced"` - Balance speed and accuracy **[Recommended]**
- `"accuracy"` - Optimize for accuracy over speed

```yaml
performance:
  optimization_level: "balanced"
```

## Environment-Specific Configurations

### Simple Loop Environment
```yaml
# config/simple_loop_config.yml
yolo:
  model_path: "models/yolo5s.pt"
  confidence_threshold: 0.5

object_avoidance:
  safety_distance: 0.4
  avoidance_strength: 0.8

lane_changing:
  enabled: false  # No lane changing in simple loop

rewards:
  weights:
    lane_following: 2.0
    object_avoidance: 1.5
    efficiency: 1.0
    safety: -3.0
```

### Multi-Lane Highway Environment
```yaml
# config/highway_config.yml
yolo:
  model_path: "models/yolo5m.pt"  # Better accuracy for complex scenes
  confidence_threshold: 0.6

object_avoidance:
  safety_distance: 0.8  # Higher speed requires more distance
  avoidance_strength: 1.5

lane_changing:
  lane_change_threshold: 0.25  # More aggressive lane changing
  safety_margin: 3.0  # Larger safety margin for highway speeds
  max_lane_change_time: 4.0

rewards:
  weights:
    lane_following: 1.0
    object_avoidance: 2.5
    lane_changing: 2.0  # Encourage lane changing
    efficiency: 1.5     # Reward speed on highway
    safety: -8.0        # Strong safety penalty
```

### Urban Environment with Pedestrians
```yaml
# config/urban_config.yml
yolo:
  model_path: "models/yolo5l.pt"  # High accuracy for pedestrian detection
  confidence_threshold: 0.4  # Lower threshold for pedestrians

object_avoidance:
  safety_distance: 1.0  # Large safety distance for pedestrians
  avoidance_strength: 2.0
  priority_weights:
    person: 5.0  # Highest priority for pedestrians
    bicycle: 3.0
    car: 2.0

lane_changing:
  lane_change_threshold: 0.4  # Conservative lane changing
  safety_margin: 2.5

rewards:
  weights:
    lane_following: 1.2
    object_avoidance: 3.0  # High priority on avoidance
    lane_changing: 1.0
    efficiency: 0.3  # Lower priority on speed
    safety: -10.0  # Very strong safety penalty
```

## Troubleshooting Configuration Issues

### Common Configuration Problems

#### YOLO Model Issues
```yaml
# Problem: Model not loading
yolo:
  model_path: "models/yolo5s.pt"  # Ensure file exists
  device: "cpu"  # Fallback if GPU issues

# Problem: Poor detection performance
yolo:
  confidence_threshold: 0.3  # Lower threshold
  input_size: 832  # Higher resolution
```

#### Performance Issues
```yaml
# Problem: Too slow
performance:
  max_fps: 15  # Reduce target FPS
  gpu_memory_fraction: 0.9  # Use more GPU memory

yolo:
  model_path: "models/yolo5n.pt"  # Use faster model
  input_size: 416  # Reduce input size
```

#### Training Instability
```yaml
# Problem: Reward oscillations
rewards:
  reward_normalization: true  # Enable normalization
  weights:
    safety: -3.0  # Reduce penalty magnitude

# Problem: Action oscillations
object_avoidance:
  smoothing_factor: 0.6  # Increase smoothing
```

### Configuration Validation

#### Automatic Validation
```python
from config.enhanced_config import EnhancedRLConfig

try:
    config = EnhancedRLConfig.from_yaml('my_config.yml')
    config.validate()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

#### Manual Validation Checklist
1. **File Paths**: Ensure all model and data files exist
2. **Parameter Ranges**: Check all values are within valid ranges
3. **Dependencies**: Verify required dependencies are installed
4. **Hardware**: Confirm GPU availability if using CUDA
5. **Memory**: Ensure sufficient RAM/GPU memory for configuration

### Best Practices

#### Configuration Management
1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for different environments
3. **Documentation**: Document custom parameter choices
4. **Testing**: Test configurations before long training runs
5. **Backup**: Keep backup of working configurations

#### Parameter Tuning Strategy
1. **Start Simple**: Begin with default parameters
2. **One at a Time**: Change one parameter at a time
3. **Monitor Impact**: Track performance metrics during changes
4. **Document Results**: Keep notes on parameter effects
5. **Systematic Search**: Use grid search or Bayesian optimization for complex tuning

This configuration guide provides comprehensive information for optimizing the Enhanced Duckietown RL system for various scenarios and requirements.