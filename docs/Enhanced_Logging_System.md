# Enhanced Logging System

The Enhanced Logging System provides comprehensive, structured logging capabilities for the Duckietown RL environment. It captures object detections, action decisions, reward calculations, and performance metrics in a structured JSON format for analysis and debugging.

## Features

- **Structured Logging**: All logs are stored in JSON/JSONL format for easy parsing and analysis
- **Multiple Log Types**: Separate logging for detections, actions, rewards, and performance metrics
- **Performance Tracking**: Built-in FPS calculation and timing measurements
- **Context Management**: Easy-to-use context managers for timing operations
- **Schema Validation**: Comprehensive test suite ensures log format consistency
- **Configurable Output**: Support for both console and file output with customizable log levels

## Quick Start

### Basic Usage

```python
from duckietown_utils.enhanced_logger import initialize_logger

# Initialize the logger
logger = initialize_logger(
    log_dir="logs/my_experiment",
    log_level="INFO",
    log_detections=True,
    log_actions=True,
    log_rewards=True,
    log_performance=True
)

# Log object detection results
detections = [
    {
        'class': 'duckiebot',
        'confidence': 0.85,
        'bbox': [100, 50, 200, 150],
        'distance': 1.2,
        'relative_position': [0.5, 0.0]
    }
]

logger.log_object_detection(
    frame_id=1,
    detections=detections,
    processing_time_ms=25.5,
    confidence_threshold=0.5
)
```

### Using with Wrapper Classes

```python
from duckietown_utils.logging_context import LoggingMixin

class MyWrapper(LoggingMixin):
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Log wrapper initialization
        self._log_wrapper_initialization({
            'wrapper_type': 'MyWrapper',
            'config': {'param1': 'value1'}
        })
    
    def step(self, action):
        frame_id = self._increment_frame_id()
        
        # Log action decision
        self._logger.log_action_decision(
            frame_id=frame_id,
            original_action=action,
            modified_action=action,  # or modified version
            action_type='lane_following',
            reasoning='Normal lane following behavior',
            triggering_conditions={'clear_path': True},
            safety_checks={'collision_check': True},
            wrapper_source=self._wrapper_name
        )
        
        return self.env.step(action)
```

## Log Types and Formats

### Object Detection Logs

Stored in `detections_YYYYMMDD_HHMMSS.jsonl`:

```json
{
    "timestamp": 1692123456.789,
    "frame_id": 42,
    "detections": [
        {
            "class": "duckiebot",
            "confidence": 0.85,
            "bbox": [100, 50, 200, 150],
            "distance": 1.2,
            "relative_position": [0.5, 0.0]
        }
    ],
    "processing_time_ms": 25.5,
    "total_objects": 1,
    "safety_critical": false,
    "confidence_threshold": 0.5
}
```

### Action Decision Logs

Stored in `actions_YYYYMMDD_HHMMSS.jsonl`:

```json
{
    "timestamp": 1692123456.789,
    "frame_id": 42,
    "original_action": [0.5, 0.0],
    "modified_action": [0.3, 0.2],
    "action_type": "object_avoidance",
    "reasoning": "Avoiding obstacle detected at 0.4m distance",
    "triggering_conditions": {
        "obstacle_distance": 0.4,
        "obstacle_class": "duckiebot"
    },
    "safety_checks": {
        "clearance_check": true,
        "collision_check": true
    },
    "wrapper_source": "ObjectAvoidanceActionWrapper"
}
```

### Reward Component Logs

Stored in `rewards_YYYYMMDD_HHMMSS.jsonl`:

```json
{
    "timestamp": 1692123456.789,
    "frame_id": 42,
    "total_reward": 0.9,
    "reward_components": {
        "lane_following": 0.8,
        "object_avoidance": 0.2,
        "safety_penalty": -0.1
    },
    "reward_weights": {
        "lane_following": 1.0,
        "object_avoidance": 0.5,
        "safety_penalty": 2.0
    },
    "episode_step": 100,
    "cumulative_reward": 45.2
}
```

### Performance Metrics Logs

Stored in `performance_YYYYMMDD_HHMMSS.jsonl`:

```json
{
    "timestamp": 1692123456.789,
    "frame_id": 42,
    "fps": 15.5,
    "detection_time_ms": 35.2,
    "action_processing_time_ms": 5.1,
    "reward_calculation_time_ms": 2.3,
    "total_step_time_ms": 42.6,
    "memory_usage_mb": 512.0,
    "gpu_memory_usage_mb": 256.0
}
```

## Context Managers

The logging system provides convenient context managers for timing operations:

### Detection Timing

```python
from duckietown_utils.logging_context import log_detection_timing

with log_detection_timing(frame_id=1, logger=logger) as log_detections:
    # Perform object detection
    detections = detect_objects(image)
    
    # Log results with automatic timing
    processing_time = log_detections(detections, confidence_threshold=0.5)
```

### Action Timing

```python
from duckietown_utils.logging_context import log_action_timing

with log_action_timing(frame_id=1, wrapper_source="MyWrapper", logger=logger) as log_action:
    # Process action
    modified_action = process_action(original_action)
    
    # Log with timing
    processing_time = log_action(
        original_action=original_action,
        modified_action=modified_action,
        action_type="object_avoidance",
        reasoning="Avoiding detected obstacle",
        triggering_conditions={"obstacle_detected": True},
        safety_checks={"clearance_ok": True}
    )
```

## Configuration Options

### Logger Initialization

```python
logger = initialize_logger(
    log_dir="logs/experiment",          # Directory for log files
    log_level="INFO",                   # Logging level (DEBUG, INFO, WARNING, ERROR)
    log_detections=True,                # Enable detection logging
    log_actions=True,                   # Enable action logging
    log_rewards=True,                   # Enable reward logging
    log_performance=True,               # Enable performance logging
    console_output=True,                # Enable console output
    file_output=True                    # Enable file output
)
```

### Performance Tracking

The system automatically tracks:
- Frame rates (FPS)
- Processing times for each component
- Memory usage (RAM and GPU)
- Frame IDs for correlation across log types

## Log Analysis

### Reading JSONL Files

```python
import json
from pathlib import Path

log_dir = Path("logs/experiment")
detection_file = list(log_dir.glob("detections_*.jsonl"))[0]

with open(detection_file, 'r') as f:
    for line in f:
        log_entry = json.loads(line.strip())
        print(f"Frame {log_entry['frame_id']}: {log_entry['total_objects']} objects detected")
```

### Performance Analysis

```python
import json
import pandas as pd

# Load performance logs into DataFrame
performance_file = list(log_dir.glob("performance_*.jsonl"))[0]
performance_data = []

with open(performance_file, 'r') as f:
    for line in f:
        performance_data.append(json.loads(line.strip()))

df = pd.DataFrame(performance_data)

# Analyze performance metrics
print(f"Average FPS: {df['fps'].mean():.2f}")
print(f"Average detection time: {df['detection_time_ms'].mean():.2f}ms")
print(f"Average total step time: {df['total_step_time_ms'].mean():.2f}ms")
```

## Error Handling

The logging system includes robust error handling:

```python
# Log errors with context
try:
    risky_operation()
except Exception as e:
    logger.log_error("Operation failed", exception=e, context="additional_info")

# Log warnings
logger.log_warning("Performance degraded", fps=current_fps, threshold=target_fps)
```

## Best Practices

1. **Use Context Managers**: Prefer context managers for automatic timing
2. **Consistent Frame IDs**: Use the performance tracker to maintain consistent frame IDs
3. **Structured Data**: Include relevant context in triggering_conditions and safety_checks
4. **Regular Analysis**: Periodically analyze logs to identify performance bottlenecks
5. **Log Rotation**: Consider implementing log rotation for long-running experiments

## Integration with Existing Wrappers

The logging system is designed to integrate seamlessly with existing wrapper classes:

```python
# In your wrapper's __init__ method
from duckietown_utils.logging_context import LoggingMixin

class MyEnhancedWrapper(gym.Wrapper, LoggingMixin):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        LoggingMixin.__init__(self)
        
        # Your wrapper initialization code
        self._log_wrapper_initialization(config_dict)
```

## Testing

The logging system includes comprehensive unit tests:

```bash
# Run all logging tests
python -m unittest tests.test_enhanced_logger -v
python -m unittest tests.test_log_format_validation -v
python -m unittest tests.test_logging_integration -v
```

## Example Usage

See `examples/enhanced_logging_example.py` for a complete demonstration of the logging system capabilities.

## Requirements Satisfied

This logging system satisfies the following requirements from the specification:

- **5.1**: Structured logging for object detections with confidence scores and bounding boxes
- **5.2**: Action decision logging with reasoning and triggering conditions
- **5.3**: Reward component logging for debugging and analysis
- **5.4**: Performance metrics logging for frame rates and processing times
- **5.5**: Comprehensive error handling and logging
- **6.5**: Testing capabilities with log format validation