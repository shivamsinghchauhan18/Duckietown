# Design Document

## Overview

The Enhanced Duckietown RL system extends the existing lane-following agent with advanced object detection and avoidance capabilities using YOLO v5, dynamic lane changing behaviors, and comprehensive integration with PPO training. The design maintains compatibility with the existing wrapper architecture while adding new layers of functionality for complex autonomous driving scenarios.

The system follows a modular wrapper-based architecture where each capability (object detection, avoidance, lane changing) is implemented as separate gym wrappers that can be composed together. This approach ensures maintainability, testability, and allows for selective feature activation during training and inference.

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    A[Duckietown Simulator] --> B[Observation Wrappers]
    B --> C[YOLO Object Detection Wrapper]
    C --> D[Enhanced Observation Wrapper]
    D --> E[PPO Agent]
    E --> F[Action Wrappers]
    F --> G[Object Avoidance Action Wrapper]
    G --> H[Lane Changing Action Wrapper]
    H --> I[Reward Wrappers]
    I --> J[Multi-Objective Reward Wrapper]
    J --> A
    
    K[YOLO v5 Model] --> C
    L[Configuration Manager] --> M[All Components]
    N[Logger] --> M
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant Sim as Simulator
    participant OD as Object Detection
    participant OA as Object Avoidance
    participant LC as Lane Changing
    participant PPO as PPO Agent
    participant RW as Reward Wrapper
    
    Sim->>OD: Raw observation
    OD->>OA: Observation + detections
    OA->>LC: Enhanced observation
    LC->>PPO: Final observation
    PPO->>LC: Action decision
    LC->>OA: Lane change action
    OA->>Sim: Final action
    Sim->>RW: Step result
    RW->>PPO: Multi-objective reward
```

## Components and Interfaces

### 1. YOLO Object Detection Wrapper

**Purpose**: Integrates YOLO v5 object detection into the observation pipeline.

**Interface**:
```python
class YOLOObjectDetectionWrapper(gym.ObservationWrapper):
    def __init__(self, env, model_path: str, confidence_threshold: float = 0.5)
    def observation(self, observation: np.ndarray) -> Dict[str, Any]
```

**Key Features**:
- Real-time object detection using pre-trained YOLO v5 model
- Configurable confidence thresholds
- Bounding box extraction and object classification
- Integration with existing observation space
- Performance optimization for real-time processing

**Output Format**:
```python
{
    'image': np.ndarray,  # Original image
    'detections': [
        {
            'class': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'distance': float,  # Estimated distance
            'relative_position': [x, y]  # Relative to robot
        }
    ],
    'detection_count': int,
    'safety_critical': bool  # Any objects within safety distance
}
```

### 2. Object Avoidance Action Wrapper

**Purpose**: Modifies actions to avoid detected objects while maintaining lane following.

**Interface**:
```python
class ObjectAvoidanceActionWrapper(gym.ActionWrapper):
    def __init__(self, env, safety_distance: float = 0.5, avoidance_strength: float = 1.0)
    def action(self, action: np.ndarray) -> np.ndarray
```

**Key Features**:
- Potential field-based avoidance algorithm
- Configurable safety distances and avoidance strength
- Smooth action modifications to prevent jerky movements
- Priority-based avoidance for multiple objects
- Integration with existing action space

**Avoidance Algorithm**:
1. Calculate repulsive forces from detected objects
2. Apply attractive force toward lane center
3. Combine forces using weighted sum
4. Modify original action based on combined force vector
5. Apply smoothing to prevent oscillations

### 3. Lane Changing Action Wrapper

**Purpose**: Enables dynamic lane changing decisions based on obstacles and traffic conditions.

**Interface**:
```python
class LaneChangingActionWrapper(gym.ActionWrapper):
    def __init__(self, env, lane_change_threshold: float = 0.3, safety_margin: float = 2.0)
    def action(self, action: np.ndarray) -> np.ndarray
```

**Key Features**:
- Lane occupancy detection and analysis
- Safe lane change trajectory planning
- Multi-step lane change execution
- Fallback to current lane if unsafe
- State machine for lane change phases

**Lane Change State Machine**:
```mermaid
stateDiagram-v2
    [*] --> LaneFollowing
    LaneFollowing --> EvaluatingChange: Obstacle detected
    EvaluatingChange --> InitiatingChange: Safe lane available
    EvaluatingChange --> LaneFollowing: No safe lane
    InitiatingChange --> ExecutingChange: Lane change started
    ExecutingChange --> LaneFollowing: Lane change completed
    ExecutingChange --> LaneFollowing: Emergency abort
```

### 4. Enhanced Observation Wrapper

**Purpose**: Combines object detection results with traditional observations for the RL agent.

**Interface**:
```python
class EnhancedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, include_detection_features: bool = True)
    def observation(self, observation: Dict) -> np.ndarray
```

**Key Features**:
- Flattens detection information into feature vector
- Maintains compatibility with existing observation space
- Configurable feature inclusion
- Normalization of detection features

**Feature Vector Components**:
- Original image features (flattened/encoded)
- Object presence indicators (per class)
- Closest object distance and bearing
- Lane occupancy status
- Safety critical flags

### 5. Multi-Objective Reward Wrapper

**Purpose**: Provides balanced rewards for lane following, object avoidance, and lane changing behaviors.

**Interface**:
```python
class MultiObjectiveRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, reward_weights: Dict[str, float])
    def reward(self, reward: float) -> float
```

**Reward Components**:
- **Lane Following Reward**: Based on existing DtRewardPosAngle
- **Object Avoidance Reward**: Positive for maintaining safe distances
- **Lane Change Reward**: Positive for successful lane changes when needed
- **Efficiency Reward**: Encourages forward progress
- **Safety Penalty**: Negative for collisions or unsafe maneuvers

**Reward Calculation**:
```python
total_reward = (
    w_lane * lane_following_reward +
    w_avoid * object_avoidance_reward +
    w_change * lane_change_reward +
    w_efficiency * efficiency_reward +
    w_safety * safety_penalty
)
```

## Data Models

### Detection Data Structure
```python
@dataclass
class ObjectDetection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    distance: float
    relative_position: Tuple[float, float]
    timestamp: float
    
@dataclass
class DetectionResult:
    detections: List[ObjectDetection]
    processing_time: float
    frame_id: int
    safety_critical: bool
```

### Lane Change State
```python
@dataclass
class LaneChangeState:
    current_phase: str  # 'following', 'evaluating', 'initiating', 'executing'
    target_lane: Optional[int]
    progress: float  # 0.0 to 1.0
    start_time: float
    safety_checks: Dict[str, bool]
    
@dataclass
class LaneInfo:
    lane_id: int
    occupancy: float  # 0.0 to 1.0
    safe_distance_ahead: float
    safe_distance_behind: float
```

### Configuration Schema
```python
@dataclass
class EnhancedRLConfig:
    # YOLO Configuration
    yolo_model_path: str
    yolo_confidence_threshold: float = 0.5
    yolo_device: str = 'cuda'
    
    # Object Avoidance Configuration
    safety_distance: float = 0.5
    avoidance_strength: float = 1.0
    min_clearance: float = 0.2
    
    # Lane Changing Configuration
    lane_change_threshold: float = 0.3
    safety_margin: float = 2.0
    max_lane_change_time: float = 3.0
    
    # Reward Configuration
    reward_weights: Dict[str, float]
    
    # Logging Configuration
    log_level: str = 'INFO'
    log_detections: bool = True
    log_actions: bool = True
    log_rewards: bool = True
```

## Error Handling

### Object Detection Errors
- **Model Loading Failures**: Fallback to basic obstacle detection using simulator data
- **Inference Timeouts**: Skip detection for current frame, use previous results
- **Memory Issues**: Reduce batch size or image resolution
- **GPU Unavailability**: Fallback to CPU inference with performance warning

### Action Wrapper Errors
- **Invalid Actions**: Clip to valid action space and log warning
- **Safety Violations**: Override with emergency stop action
- **Lane Change Failures**: Abort lane change and return to lane following

### Integration Errors
- **Wrapper Compatibility**: Validate wrapper order and compatibility at initialization
- **Configuration Errors**: Comprehensive validation with meaningful error messages
- **Runtime Exceptions**: Graceful degradation with logging

## Testing Strategy

### Unit Testing
- **Individual Wrapper Testing**: Test each wrapper in isolation with mock environments
- **YOLO Integration Testing**: Validate object detection accuracy and performance
- **Action Modification Testing**: Verify action transformations maintain safety constraints
- **Reward Calculation Testing**: Validate multi-objective reward computation

### Integration Testing
- **Full Pipeline Testing**: Test complete wrapper stack with real simulator
- **Performance Testing**: Validate real-time processing requirements (>= 10 FPS)
- **Safety Testing**: Verify collision avoidance in various scenarios
- **Lane Change Testing**: Validate lane change execution and safety checks

### Scenario Testing
- **Static Obstacle Scenarios**: Test avoidance of stationary objects
- **Dynamic Obstacle Scenarios**: Test interaction with moving objects
- **Multi-Lane Scenarios**: Test lane changing in complex traffic situations
- **Edge Cases**: Test behavior at map boundaries, intersections, and tight spaces

### Performance Benchmarks
- **Detection Latency**: < 50ms per frame
- **Action Processing**: < 10ms per step
- **Memory Usage**: < 2GB GPU memory for YOLO inference
- **Training Stability**: Convergence within 1M timesteps

### Logging and Debugging
- **Structured Logging**: JSON-formatted logs with timestamps and context
- **Performance Metrics**: Frame rates, processing times, memory usage
- **Decision Logging**: Action decisions, lane change reasoning, safety checks
- **Visualization Tools**: Real-time display of detections, actions, and rewards

The design ensures modularity, maintainability, and extensibility while providing robust autonomous driving capabilities that build upon the existing Duckietown RL framework.