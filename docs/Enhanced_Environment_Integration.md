# Enhanced Environment Integration

This document describes the Enhanced Duckietown RL Environment Integration Module, which extends the existing environment setup with advanced capabilities for object detection, avoidance, and lane changing.

## Overview

The Enhanced Environment Integration Module provides a seamless way to add advanced RL capabilities to the existing Duckietown environment while maintaining backward compatibility. It follows a modular wrapper-based architecture where each capability can be enabled or disabled independently.

## Key Features

- **Modular Design**: Each enhanced capability is implemented as a separate wrapper
- **Backward Compatibility**: Works with existing training configurations
- **Flexible Configuration**: Enable/disable features as needed
- **Error Handling**: Graceful degradation when components fail
- **Comprehensive Logging**: Detailed logging for debugging and analysis

## Architecture

The integration module extends the existing `launch_and_wrap_env` function with `launch_and_wrap_enhanced_env`, which applies enhanced wrappers in the correct order:

1. **Observation Wrappers**: YOLO detection, enhanced observations
2. **Action Wrappers**: Object avoidance, lane changing
3. **Reward Wrappers**: Multi-objective rewards

## Usage

### Basic Usage

```python
from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig

# Create standard environment configuration
env_config = {
    'training_map': 'small_loop',
    'episode_max_steps': 500,
    'mode': 'train',
    'action_type': 'continuous',
    'reward_function': 'Posangle',
    # ... other standard config options
}

# Create enhanced configuration
enhanced_config = EnhancedRLConfig(
    enabled_features=['yolo', 'object_avoidance', 'lane_changing']
)

# Launch enhanced environment
env = launch_and_wrap_enhanced_env(env_config, enhanced_config)

# Use environment normally
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

### Configuration from File

```python
from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import load_enhanced_config

# Load configuration from YAML file
enhanced_config = load_enhanced_config('config/enhanced_config.yml')

# Launch environment
env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
```

### Selective Features

```python
# Enable only specific features
enhanced_config = EnhancedRLConfig(
    enabled_features=['object_avoidance']  # Only object avoidance
)

env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
```

### Backward Compatibility

```python
# No enhanced features - works like standard environment
enhanced_config = EnhancedRLConfig(enabled_features=[])

env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
```

## Configuration Options

### Enhanced Features

- `yolo`: YOLO v5 object detection
- `object_avoidance`: Potential field-based obstacle avoidance
- `lane_changing`: Dynamic lane changing behavior
- `multi_objective_reward`: Multi-objective reward function

### YOLO Configuration

```python
enhanced_config.yolo.model_path = "yolov5s.pt"
enhanced_config.yolo.confidence_threshold = 0.5
enhanced_config.yolo.device = "cuda"  # or "cpu", "auto"
enhanced_config.yolo.input_size = 640
enhanced_config.yolo.max_detections = 100
```

### Object Avoidance Configuration

```python
enhanced_config.object_avoidance.safety_distance = 0.5
enhanced_config.object_avoidance.avoidance_strength = 1.0
enhanced_config.object_avoidance.min_clearance = 0.2
enhanced_config.object_avoidance.max_avoidance_angle = 0.5
enhanced_config.object_avoidance.smoothing_factor = 0.8
```

### Lane Changing Configuration

```python
enhanced_config.lane_changing.lane_change_threshold = 0.3
enhanced_config.lane_changing.safety_margin = 2.0
enhanced_config.lane_changing.max_lane_change_time = 3.0
enhanced_config.lane_changing.min_lane_width = 0.4
enhanced_config.lane_changing.evaluation_distance = 5.0
```

### Reward Configuration

```python
enhanced_config.reward.lane_following_weight = 1.0
enhanced_config.reward.object_avoidance_weight = 0.5
enhanced_config.reward.lane_change_weight = 0.3
enhanced_config.reward.efficiency_weight = 0.2
enhanced_config.reward.safety_penalty_weight = -2.0
enhanced_config.reward.collision_penalty = -10.0
```

## Wrapper Order and Compatibility

The integration module applies wrappers in a specific order to ensure compatibility:

1. **Standard Wrappers** (from existing `wrap_env`)
2. **YOLO Object Detection Wrapper**
3. **Enhanced Observation Wrapper**
4. **Object Avoidance Action Wrapper**
5. **Lane Changing Action Wrapper**
6. **Multi-Objective Reward Wrapper**

### Compatibility Checks

The module performs automatic compatibility validation:

- **YOLO + Grayscale**: Error - YOLO requires RGB images
- **Enhanced Actions + Discrete**: Warning - works better with continuous actions
- **Multi-Objective + Custom Rewards**: Warning - potential conflicts
- **Dependencies**: Warnings for missing feature dependencies

## Error Handling

The integration module provides robust error handling:

### Debug Mode
```python
enhanced_config.debug_mode = True  # Raises exceptions immediately
```

### Production Mode
```python
enhanced_config.debug_mode = False  # Graceful degradation
```

In production mode, if a wrapper fails to initialize:
1. Error is logged
2. Warning is issued
3. Environment continues without that wrapper
4. Other wrappers still function normally

## Inspection and Debugging

### Get Wrapper Information

```python
from duckietown_utils.env import get_enhanced_wrappers

obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)

print(f"Enhanced wrappers: {len(enhanced_wrappers)}")
for wrapper in enhanced_wrappers:
    print(f"  - {wrapper.__class__.__name__}")
```

### Logging Configuration

```python
enhanced_config.logging.log_level = "DEBUG"
enhanced_config.logging.log_detections = True
enhanced_config.logging.log_actions = True
enhanced_config.logging.log_rewards = True
enhanced_config.logging.log_performance = True
```

## Integration with PPO Training

The enhanced environment integrates seamlessly with PPO training:

```python
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

def env_creator(env_config):
    enhanced_config = load_enhanced_config('config/enhanced_config.yml')
    return launch_and_wrap_enhanced_env(env_config, enhanced_config)

# Register environment
tune.register_env("enhanced_duckietown", env_creator)

# Configure PPO trainer
config = {
    "env": "enhanced_duckietown",
    "env_config": env_config,
    # ... other PPO config options
}

trainer = PPOTrainer(config=config)
```

## Performance Considerations

### YOLO Performance
- GPU acceleration recommended for real-time performance
- Adjust `input_size` and `max_detections` based on hardware
- Monitor memory usage with `memory_limit_gb` setting

### Action Processing
- Object avoidance and lane changing add computational overhead
- Use `smoothing_factor` to balance responsiveness vs. stability
- Consider reducing `max_fps` if processing can't keep up

### Memory Usage
- YOLO model loading requires significant GPU memory
- Frame stacking with YOLO detection increases memory usage
- Monitor with `performance.memory_limit_gb` setting

## Testing

The integration module includes comprehensive tests:

```bash
# Run integration tests
python -m pytest tests/test_enhanced_environment_integration.py -v

# Run example script
python examples/enhanced_environment_integration_example.py
```

## Troubleshooting

### Common Issues

1. **YOLO Model Not Found**
   - Ensure model path is correct
   - Check if model file exists
   - Verify model format compatibility

2. **GPU Memory Issues**
   - Reduce YOLO input size
   - Use CPU inference as fallback
   - Adjust memory limits

3. **Action Space Conflicts**
   - Verify action space compatibility
   - Check wrapper order
   - Review action transformation logic

4. **Reward Function Conflicts**
   - Disable conflicting reward wrappers
   - Adjust reward weights
   - Use debug mode for detailed logging

### Debug Mode

Enable debug mode for detailed error information:

```python
enhanced_config.debug_mode = True
```

This will:
- Raise exceptions immediately instead of graceful degradation
- Provide detailed stack traces
- Enable verbose logging
- Show wrapper initialization details

## Migration from Standard Environment

To migrate from standard environment to enhanced environment:

1. **Update imports**:
   ```python
   # Old
   from duckietown_utils.env import launch_and_wrap_env
   
   # New
   from duckietown_utils.env import launch_and_wrap_enhanced_env
   from config.enhanced_config import EnhancedRLConfig
   ```

2. **Create enhanced configuration**:
   ```python
   enhanced_config = EnhancedRLConfig(enabled_features=[])  # Start with no features
   ```

3. **Update environment creation**:
   ```python
   # Old
   env = launch_and_wrap_env(env_config)
   
   # New
   env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
   ```

4. **Gradually enable features**:
   ```python
   enhanced_config.enabled_features = ['object_avoidance']  # Add one feature at a time
   ```

## Future Extensions

The integration module is designed for extensibility:

- New wrapper types can be easily added
- Configuration system supports new parameters
- Validation system can be extended
- Error handling can be customized

To add a new enhanced wrapper:

1. Implement the wrapper class
2. Add it to the wrapper imports
3. Update `_apply_enhanced_wrappers` function
4. Add configuration parameters
5. Update validation logic
6. Add tests and documentation

## Conclusion

The Enhanced Environment Integration Module provides a robust, flexible, and backward-compatible way to add advanced RL capabilities to the Duckietown environment. Its modular design allows for selective feature usage while maintaining the simplicity of the original environment interface.