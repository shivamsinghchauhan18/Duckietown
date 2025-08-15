# Enhanced Duckietown RL API Documentation

## Overview

This document provides comprehensive API documentation for all wrapper classes in the Enhanced Duckietown RL system. The system extends the base Duckietown environment with object detection, avoidance, lane changing, and multi-objective reward capabilities.

## Core Wrapper Classes

### YOLOObjectDetectionWrapper

**Purpose**: Integrates YOLO v5 object detection into the observation pipeline for real-time obstacle detection.

**Class Definition**:
```python
class YOLOObjectDetectionWrapper(gym.ObservationWrapper):
    """
    Wrapper that adds YOLO v5 object detection to environment observations.
    
    This wrapper processes camera observations through a YOLO v5 model to detect
    objects in the environment and includes detection results in the observation space.
    """
```

**Constructor Parameters**:
- `env` (gym.Env): The environment to wrap
- `model_path` (str): Path to the YOLO v5 model file
- `confidence_threshold` (float, default=0.5): Minimum confidence for detections
- `device` (str, default='cuda'): Device for YOLO inference ('cuda' or 'cpu')
- `max_detections` (int, default=10): Maximum number of detections per frame

**Methods**:

#### `observation(self, observation: np.ndarray) -> Dict[str, Any]`
Processes the raw observation through YOLO detection.

**Parameters**:
- `observation` (np.ndarray): Raw camera observation from environment

**Returns**:
- `Dict[str, Any]`: Enhanced observation containing:
  - `image`: Original camera image
  - `detections`: List of detection dictionaries
  - `detection_count`: Number of objects detected
  - `safety_critical`: Boolean indicating objects within safety distance

**Detection Dictionary Format**:
```python
{
    'class': str,           # Object class name
    'confidence': float,    # Detection confidence (0.0-1.0)
    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
    'distance': float,      # Estimated distance to object
    'relative_position': [x, y]  # Position relative to robot
}
```

**Example Usage**:
```python
from duckietown_utils.wrappers import YOLOObjectDetectionWrapper

env = YOLOObjectDetectionWrapper(
    env=base_env,
    model_path='models/yolo5s.pt',
    confidence_threshold=0.6,
    device='cuda'
)

obs = env.reset()
print(f"Detected {obs['detection_count']} objects")
for detection in obs['detections']:
    print(f"Found {detection['class']} at distance {detection['distance']:.2f}m")
```

---

### ObjectAvoidanceActionWrapper

**Purpose**: Modifies actions to avoid detected objects while maintaining lane following behavior.

**Class Definition**:
```python
class ObjectAvoidanceActionWrapper(gym.ActionWrapper):
    """
    Wrapper that modifies actions to avoid detected objects using potential field algorithm.
    
    This wrapper analyzes object detections and applies repulsive forces to steer away
    from obstacles while maintaining attraction to the lane center.
    """
```

**Constructor Parameters**:
- `env` (gym.Env): The environment to wrap (must have object detections)
- `safety_distance` (float, default=0.5): Distance threshold for avoidance activation
- `avoidance_strength` (float, default=1.0): Strength of avoidance forces
- `min_clearance` (float, default=0.2): Minimum clearance to maintain from objects
- `smoothing_factor` (float, default=0.3): Action smoothing to prevent oscillations

**Methods**:

#### `action(self, action: np.ndarray) -> np.ndarray`
Modifies the input action to avoid detected objects.

**Parameters**:
- `action` (np.ndarray): Original action from the agent [velocity, steering]

**Returns**:
- `np.ndarray`: Modified action with avoidance behavior applied

**Algorithm**:
1. Calculate repulsive forces from all detected objects within safety distance
2. Apply attractive force toward lane center
3. Combine forces using weighted sum based on priorities
4. Modify original action based on combined force vector
5. Apply smoothing to prevent jerky movements

**Example Usage**:
```python
from duckietown_utils.wrappers import ObjectAvoidanceActionWrapper

env = ObjectAvoidanceActionWrapper(
    env=detection_env,
    safety_distance=0.6,
    avoidance_strength=1.2,
    min_clearance=0.25
)

action = agent.predict(obs)
modified_action = env.action(action)  # Automatically applies avoidance
```

---

### LaneChangingActionWrapper

**Purpose**: Enables dynamic lane changing decisions based on obstacles and traffic conditions.

**Class Definition**:
```python
class LaneChangingActionWrapper(gym.ActionWrapper):
    """
    Wrapper that adds lane changing capabilities to the action space.
    
    This wrapper implements a state machine for safe lane changing, including
    lane evaluation, trajectory planning, and execution with safety checks.
    """
```

**Constructor Parameters**:
- `env` (gym.Env): The environment to wrap
- `lane_change_threshold` (float, default=0.3): Threshold for initiating lane changes
- `safety_margin` (float, default=2.0): Required clear distance for lane changes
- `max_lane_change_time` (float, default=3.0): Maximum time to complete lane change
- `evaluation_frequency` (int, default=10): Steps between lane evaluations

**Methods**:

#### `action(self, action: np.ndarray) -> np.ndarray`
Processes action through lane changing state machine.

**Parameters**:
- `action` (np.ndarray): Original action from agent

**Returns**:
- `np.ndarray`: Action modified for lane changing if applicable

#### `get_lane_change_state(self) -> Dict[str, Any]`
Returns current lane changing state information.

**Returns**:
- `Dict[str, Any]`: State information including:
  - `phase`: Current phase ('following', 'evaluating', 'initiating', 'executing')
  - `target_lane`: Target lane ID if changing
  - `progress`: Lane change progress (0.0-1.0)
  - `safety_checks`: Results of safety evaluations

**State Machine Phases**:
1. **Lane Following**: Normal lane following behavior
2. **Evaluating Change**: Assessing need and safety for lane change
3. **Initiating Change**: Beginning lane change maneuver
4. **Executing Change**: Completing lane change trajectory

**Example Usage**:
```python
from duckietown_utils.wrappers import LaneChangingActionWrapper

env = LaneChangingActionWrapper(
    env=avoidance_env,
    lane_change_threshold=0.4,
    safety_margin=2.5
)

action = agent.predict(obs)
state = env.get_lane_change_state()
if state['phase'] == 'executing':
    print(f"Lane change {state['progress']*100:.1f}% complete")
```

---

### EnhancedObservationWrapper

**Purpose**: Combines object detection results with traditional observations for RL agent consumption.

**Class Definition**:
```python
class EnhancedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that flattens and normalizes enhanced observations for RL training.
    
    This wrapper converts complex detection and state information into a flat
    feature vector suitable for neural network processing.
    """
```

**Constructor Parameters**:
- `env` (gym.Env): The environment to wrap
- `include_detection_features` (bool, default=True): Include object detection features
- `include_lane_features` (bool, default=True): Include lane state features
- `feature_vector_size` (int, default=128): Size of output feature vector
- `normalize_features` (bool, default=True): Apply feature normalization

**Methods**:

#### `observation(self, observation: Dict) -> np.ndarray`
Converts enhanced observation dictionary to flat feature vector.

**Parameters**:
- `observation` (Dict): Enhanced observation from previous wrappers

**Returns**:
- `np.ndarray`: Flattened and normalized feature vector

**Feature Vector Components**:
- Image features (CNN-encoded or downsampled)
- Object presence indicators (per class)
- Closest object distance and bearing
- Lane occupancy status
- Safety critical flags
- Lane change state information

**Example Usage**:
```python
from duckietown_utils.wrappers import EnhancedObservationWrapper

env = EnhancedObservationWrapper(
    env=lane_changing_env,
    feature_vector_size=256,
    normalize_features=True
)

obs = env.reset()
print(f"Feature vector shape: {obs.shape}")  # (256,)
```

---

### MultiObjectiveRewardWrapper

**Purpose**: Provides balanced rewards for lane following, object avoidance, and lane changing behaviors.

**Class Definition**:
```python
class MultiObjectiveRewardWrapper(gym.RewardWrapper):
    """
    Wrapper that implements multi-objective reward function for enhanced behaviors.
    
    This wrapper combines rewards from multiple objectives including lane following,
    object avoidance, lane changing efficiency, and safety penalties.
    """
```

**Constructor Parameters**:
- `env` (gym.Env): The environment to wrap
- `reward_weights` (Dict[str, float]): Weights for different reward components
- `safety_penalty_scale` (float, default=10.0): Scale factor for safety penalties
- `efficiency_bonus_scale` (float, default=1.0): Scale factor for efficiency bonuses

**Default Reward Weights**:
```python
{
    'lane_following': 1.0,      # Lane following accuracy
    'object_avoidance': 2.0,    # Safe object avoidance
    'lane_changing': 1.5,       # Successful lane changes
    'efficiency': 0.5,          # Forward progress
    'safety': -5.0              # Safety violations (negative)
}
```

**Methods**:

#### `reward(self, reward: float) -> float`
Calculates multi-objective reward based on current state.

**Parameters**:
- `reward` (float): Base reward from environment

**Returns**:
- `float`: Combined multi-objective reward

#### `get_reward_components(self) -> Dict[str, float]`
Returns breakdown of individual reward components.

**Returns**:
- `Dict[str, float]`: Individual reward components for analysis

**Reward Components**:
1. **Lane Following**: Based on lane position and heading alignment
2. **Object Avoidance**: Positive for maintaining safe distances
3. **Lane Changing**: Rewards successful lane changes when beneficial
4. **Efficiency**: Encourages forward progress and speed
5. **Safety Penalty**: Penalizes collisions and unsafe maneuvers

**Example Usage**:
```python
from duckietown_utils.wrappers import MultiObjectiveRewardWrapper

reward_weights = {
    'lane_following': 1.2,
    'object_avoidance': 2.5,
    'lane_changing': 1.0,
    'efficiency': 0.8,
    'safety': -8.0
}

env = MultiObjectiveRewardWrapper(
    env=observation_env,
    reward_weights=reward_weights
)

obs, reward, done, info = env.step(action)
components = env.get_reward_components()
print(f"Total reward: {reward:.3f}")
for component, value in components.items():
    print(f"  {component}: {value:.3f}")
```

## Wrapper Composition

The wrappers are designed to be composed in a specific order for optimal functionality:

```python
def create_enhanced_env(base_env):
    """Create fully enhanced environment with all wrappers."""
    
    # 1. Add object detection
    env = YOLOObjectDetectionWrapper(
        env=base_env,
        model_path='models/yolo5s.pt',
        confidence_threshold=0.5
    )
    
    # 2. Add object avoidance
    env = ObjectAvoidanceActionWrapper(
        env=env,
        safety_distance=0.5,
        avoidance_strength=1.0
    )
    
    # 3. Add lane changing
    env = LaneChangingActionWrapper(
        env=env,
        lane_change_threshold=0.3,
        safety_margin=2.0
    )
    
    # 4. Flatten observations
    env = EnhancedObservationWrapper(
        env=env,
        feature_vector_size=128
    )
    
    # 5. Add multi-objective rewards
    env = MultiObjectiveRewardWrapper(
        env=env,
        reward_weights={
            'lane_following': 1.0,
            'object_avoidance': 2.0,
            'lane_changing': 1.5,
            'efficiency': 0.5,
            'safety': -5.0
        }
    )
    
    return env
```

## Error Handling

All wrappers implement comprehensive error handling:

### Common Error Types
- **Model Loading Errors**: YOLO model file not found or corrupted
- **GPU Memory Errors**: Insufficient GPU memory for YOLO inference
- **Invalid Actions**: Actions outside valid range
- **Safety Violations**: Collision detection and emergency stops

### Error Recovery Strategies
- **Graceful Degradation**: Fall back to simpler algorithms when advanced features fail
- **State Reset**: Reset wrapper state on critical errors
- **Logging**: Comprehensive error logging for debugging

### Example Error Handling
```python
try:
    env = YOLOObjectDetectionWrapper(env, model_path='invalid_path.pt')
except ModelLoadError as e:
    print(f"YOLO model loading failed: {e}")
    # Fall back to basic obstacle detection
    env = BasicObstacleDetectionWrapper(env)
```

## Performance Considerations

### Real-time Requirements
- **Detection Latency**: < 50ms per frame
- **Action Processing**: < 10ms per step
- **Memory Usage**: < 2GB GPU memory for YOLO inference

### Optimization Tips
1. Use appropriate YOLO model size (yolo5s for speed, yolo5x for accuracy)
2. Adjust image resolution based on performance requirements
3. Enable GPU acceleration when available
4. Use batch processing for multiple environments

### Monitoring Performance
```python
import time

start_time = time.time()
obs, reward, done, info = env.step(action)
step_time = time.time() - start_time

if step_time > 0.1:  # 100ms threshold
    print(f"Warning: Step took {step_time*1000:.1f}ms")
```

This API documentation provides comprehensive information for developers working with the Enhanced Duckietown RL system, including detailed parameter descriptions, usage examples, and best practices for each wrapper class.