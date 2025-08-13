# Enhanced Configuration Management System

The Enhanced Configuration Management System provides comprehensive parameter validation, YAML loading, and runtime configuration updates for the Enhanced Duckietown RL system.

## Overview

The configuration system is built around the `EnhancedRLConfig` dataclass which contains all configuration parameters for:
- YOLO object detection
- Object avoidance behavior
- Lane changing behavior
- Multi-objective reward function
- Logging and debugging
- Performance optimization

## Key Features

- **Comprehensive Validation**: All parameters are validated with meaningful error messages
- **YAML Support**: Load and save configurations from/to YAML files
- **Runtime Updates**: Modify configuration parameters during training or inference
- **Configuration Presets**: Predefined configurations for common use cases
- **Update History**: Track all configuration changes with timestamps
- **Schema Validation**: JSON schema validation for configuration files

## Basic Usage

### Creating a Default Configuration

```python
from config.enhanced_config import EnhancedRLConfig

# Create default configuration
config = EnhancedRLConfig()

# Access component configurations
print(f"YOLO confidence threshold: {config.yolo.confidence_threshold}")
print(f"Safety distance: {config.object_avoidance.safety_distance}")
print(f"Enabled features: {config.enabled_features}")
```

### Loading from YAML File

```python
# Load configuration from YAML file
config = EnhancedRLConfig.from_yaml("config/enhanced_config.yml")

# Save configuration to YAML file
config.to_yaml("my_config.yml")
```

### Using Configuration Manager

```python
from config.config_utils import ConfigurationManager

# Create configuration manager
config = EnhancedRLConfig()
manager = ConfigurationManager(config)

# Update individual components
manager.update_yolo_config(confidence_threshold=0.8, device="cpu")
manager.update_object_avoidance_config(safety_distance=0.7)

# Batch updates
updates = {
    'yolo': {'confidence_threshold': 0.9},
    'reward': {'lane_following_weight': 1.5},
    'debug_mode': True
}
manager.batch_update(updates)

# Feature management
manager.enable_feature('lane_changing')
manager.disable_feature('yolo')
```

## Configuration Components

### YOLO Configuration

```python
yolo:
  model_path: "yolov5s.pt"          # Path to YOLO model
  confidence_threshold: 0.5          # Detection confidence threshold (0.0-1.0)
  device: "auto"                     # Device: "cuda", "cpu", or "auto"
  input_size: 640                    # Input image size (must be divisible by 32)
  max_detections: 100                # Maximum detections per frame
```

### Object Avoidance Configuration

```python
object_avoidance:
  safety_distance: 0.5               # Safety distance from objects (meters)
  avoidance_strength: 1.0            # Avoidance force multiplier
  min_clearance: 0.2                 # Minimum clearance (meters)
  max_avoidance_angle: 0.5           # Maximum avoidance angle (radians)
  smoothing_factor: 0.8              # Action smoothing factor (0.0-1.0)
```

### Lane Changing Configuration

```python
lane_changing:
  lane_change_threshold: 0.3         # Threshold for initiating lane change
  safety_margin: 2.0                 # Safety margin (meters)
  max_lane_change_time: 3.0          # Maximum lane change time (seconds)
  min_lane_width: 0.4                # Minimum lane width (meters)
  evaluation_distance: 5.0           # Evaluation distance (meters)
```

### Reward Configuration

```python
reward:
  lane_following_weight: 1.0         # Lane following reward weight
  object_avoidance_weight: 0.5       # Object avoidance reward weight
  lane_change_weight: 0.3            # Lane change reward weight
  efficiency_weight: 0.2             # Efficiency reward weight
  safety_penalty_weight: -2.0        # Safety penalty weight (negative)
  collision_penalty: -10.0           # Collision penalty (negative)
```

### Logging Configuration

```python
logging:
  log_level: "INFO"                  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_detections: true               # Log object detections
  log_actions: true                  # Log action decisions
  log_rewards: true                  # Log reward components
  log_performance: true              # Log performance metrics
  log_file_path: null                # Log file path (null for no file logging)
  console_logging: true              # Enable console logging
```

### Performance Configuration

```python
performance:
  max_fps: 30.0                      # Maximum frames per second
  detection_batch_size: 1            # Batch size for object detection
  use_gpu_acceleration: true         # Use GPU acceleration
  memory_limit_gb: 4.0               # Memory limit in GB
```

## Configuration Presets

The system includes predefined presets for common use cases:

### Development Preset

```python
from config.config_utils import apply_preset

apply_preset(manager, 'development')
```

- Debug mode enabled
- Verbose logging
- CPU-only processing
- Lower performance settings

### Production Preset

```python
apply_preset(manager, 'production')
```

- Debug mode disabled
- Optimized logging
- GPU acceleration enabled
- Production performance settings

### High Performance Preset

```python
apply_preset(manager, 'high_performance')
```

- Minimal logging
- Maximum performance settings
- Batch processing enabled
- Higher memory limits

### Safe Driving Preset

```python
apply_preset(manager, 'safe_driving')
```

- Conservative safety distances
- Lower detection thresholds
- Higher safety penalties
- Longer lane change times

## Validation and Error Handling

The configuration system provides comprehensive validation:

```python
# Parameter validation
try:
    config = YOLOConfig(confidence_threshold=1.5)  # Invalid: > 1.0
except ValueError as e:
    print(f"Validation error: {e}")

# Configuration updates with validation
try:
    manager.update_yolo_config(confidence_threshold=2.0)  # Invalid
except ValidationError as e:
    print(f"Update failed: {e}")

# Validate current configuration
is_valid = manager.validate_current_config()
```

## Runtime Configuration Updates

### During Training

```python
# Adjust parameters during training
manager.update_yolo_config(confidence_threshold=0.6)
manager.update_reward_config(lane_following_weight=1.2)

# Switch to inference mode
manager.batch_update({
    'logging': {'log_level': 'WARNING'},
    'performance': {'max_fps': 60.0},
    'debug_mode': False
})
```

### Feature Toggling

```python
# Disable features for CPU-only inference
manager.disable_feature('yolo')
manager.disable_feature('object_avoidance')

# Enable features for full functionality
manager.enable_feature('yolo')
manager.enable_feature('lane_changing')
```

## Configuration Monitoring

### Export Configuration Summary

```python
summary = manager.export_config_summary()
print(f"Current configuration: {summary}")
```

### Track Update History

```python
history = manager.get_update_history()
for update in history:
    print(f"{update['timestamp']}: {update['update_type']}")
```

### Configuration Persistence

```python
# Save current configuration
manager.save_config("current_config.yml")

# Reload configuration
manager.reload_config("saved_config.yml")
```

## Integration with Training Pipeline

### Environment Setup

```python
from config.enhanced_config import load_enhanced_config
from config.config_utils import create_config_manager

# Load configuration
config = load_enhanced_config("config/enhanced_config.yml")
manager = create_config_manager("config/enhanced_config.yml")

# Use configuration in wrappers
yolo_config = config.get_feature_config('yolo')
avoidance_config = config.get_feature_config('object_avoidance')
```

### Training Loop Integration

```python
# Check if features are enabled
if config.is_feature_enabled('yolo'):
    # Initialize YOLO wrapper
    pass

if config.is_feature_enabled('object_avoidance'):
    # Initialize object avoidance wrapper
    pass

# Runtime adjustments
if training_step % 1000 == 0:
    # Adjust confidence threshold based on training progress
    new_threshold = min(0.8, 0.3 + training_step / 10000)
    manager.update_yolo_config(confidence_threshold=new_threshold)
```

## Best Practices

1. **Use Configuration Manager**: Always use `ConfigurationManager` for runtime updates
2. **Validate Early**: Load and validate configurations at startup
3. **Use Presets**: Start with predefined presets and customize as needed
4. **Monitor Changes**: Track configuration updates for debugging
5. **Persist Important Configurations**: Save successful configurations for reproducibility
6. **Handle Errors Gracefully**: Always catch and handle validation errors
7. **Document Custom Configurations**: Add comments to custom YAML files

## Error Messages and Troubleshooting

### Common Validation Errors

- `confidence_threshold must be between 0.0 and 1.0`: YOLO confidence threshold out of range
- `safety_distance must be positive`: Negative safety distance
- `min_clearance must be less than safety_distance`: Invalid clearance configuration
- `Unknown feature 'feature_name'`: Invalid feature name in enabled_features list

### File Loading Errors

- `Configuration file not found`: YAML file doesn't exist
- `Failed to parse YAML file`: Invalid YAML syntax
- `Configuration schema validation failed`: Configuration doesn't match expected schema

### Runtime Update Errors

- `Configuration update validation failed`: Invalid parameter values in update
- `No file path provided`: Missing file path for save/reload operations

## Example Configuration File

```yaml
# Enhanced Duckietown RL Configuration
enabled_features:
  - "yolo"
  - "object_avoidance"
  - "lane_changing"
  - "multi_objective_reward"

debug_mode: false

yolo:
  model_path: "yolov5s.pt"
  confidence_threshold: 0.5
  device: "auto"
  input_size: 640
  max_detections: 100

object_avoidance:
  safety_distance: 0.5
  avoidance_strength: 1.0
  min_clearance: 0.2
  max_avoidance_angle: 0.5
  smoothing_factor: 0.8

lane_changing:
  lane_change_threshold: 0.3
  safety_margin: 2.0
  max_lane_change_time: 3.0
  min_lane_width: 0.4
  evaluation_distance: 5.0

reward:
  lane_following_weight: 1.0
  object_avoidance_weight: 0.5
  lane_change_weight: 0.3
  efficiency_weight: 0.2
  safety_penalty_weight: -2.0
  collision_penalty: -10.0

logging:
  log_level: "INFO"
  log_detections: true
  log_actions: true
  log_rewards: true
  log_performance: true
  log_file_path: null
  console_logging: true

performance:
  max_fps: 30.0
  detection_batch_size: 1
  use_gpu_acceleration: true
  memory_limit_gb: 4.0
```