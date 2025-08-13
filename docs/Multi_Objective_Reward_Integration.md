# Multi-Objective Reward Wrapper Integration Guide

## Overview

The `MultiObjectiveRewardWrapper` is a comprehensive reward wrapper that combines multiple reward components for enhanced Duckietown RL training. It provides a balanced reward signal that considers lane following, object avoidance, lane changing, efficiency, and safety penalties.

## Features

- **Multi-Component Rewards**: Combines five different reward components with configurable weights
- **Configurable Weights**: Easily adjust the importance of different behaviors
- **Runtime Updates**: Modify reward weights during training for curriculum learning
- **Comprehensive Logging**: Detailed reward component tracking for analysis
- **Safety Integration**: Built-in safety penalties for collision avoidance

## Reward Components

### 1. Lane Following Reward
- **Purpose**: Encourages staying in the center of the lane with correct orientation
- **Range**: -1.0 to 1.0
- **Calculation**: Based on distance from lane center and orientation alignment
- **Default Weight**: 1.0

### 2. Object Avoidance Reward
- **Purpose**: Rewards maintaining safe distances from detected objects
- **Range**: -1.0 to 0.1
- **Calculation**: Based on proximity to detected objects and safety distances
- **Default Weight**: 0.5

### 3. Lane Changing Reward
- **Purpose**: Rewards successful lane changes when needed
- **Range**: 0.0 to 1.0
- **Calculation**: Based on lane change execution progress and completion
- **Default Weight**: 0.3

### 4. Efficiency Reward
- **Purpose**: Encourages forward progress and movement efficiency
- **Range**: -0.5 to 1.0
- **Calculation**: Based on forward movement and trajectory optimization
- **Default Weight**: 0.2

### 5. Safety Penalty
- **Purpose**: Penalizes unsafe behaviors like collisions and off-road driving
- **Range**: -∞ to 0.0 (always negative or zero)
- **Calculation**: Based on collision detection, lane violations, and unsafe maneuvers
- **Default Weight**: -2.0

## Installation and Setup

### Prerequisites

```bash
# Install required dependencies
pip install gym gym-duckietown numpy

# Install enhanced Duckietown utilities
pip install -e .
```

### Basic Usage

```python
from duckietown_utils.wrappers import MultiObjectiveRewardWrapper
import gym

# Create base environment
env = gym.make('Duckietown-udem1-v0')

# Apply multi-objective reward wrapper
env = MultiObjectiveRewardWrapper(env)

# Use the environment
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, done, info = env.step(action)
    
    # Access reward components
    if 'custom_rewards' in info:
        components = info['custom_rewards']
        print(f"Lane Following: {components['lane_following']:.3f}")
        print(f"Total Reward: {components['total']:.3f}")
    
    if done:
        obs = env.reset()
```

### Custom Weight Configuration

```python
# Define custom reward weights
custom_weights = {
    'lane_following': 1.5,      # Increase lane following importance
    'object_avoidance': 1.0,    # Increase safety importance
    'lane_changing': 0.2,       # Reduce lane changing frequency
    'efficiency': 0.1,          # Reduce speed emphasis
    'safety_penalty': -3.0      # Increase safety penalty
}

# Create wrapper with custom weights
env = MultiObjectiveRewardWrapper(env, custom_weights)
```

## Integration with Enhanced Environment

### Complete Wrapper Stack

```python
from duckietown_utils.wrappers import (
    YOLOObjectDetectionWrapper,
    EnhancedObservationWrapper,
    ObjectAvoidanceActionWrapper,
    LaneChangingActionWrapper,
    MultiObjectiveRewardWrapper
)

def create_enhanced_environment():
    # Start with base environment
    env = gym.make('Duckietown-udem1-v0')
    
    # Add observation wrappers
    env = YOLOObjectDetectionWrapper(env, model_path='models/yolov5s.pt')
    env = EnhancedObservationWrapper(env)
    
    # Add action wrappers
    env = ObjectAvoidanceActionWrapper(env, safety_distance=0.5)
    env = LaneChangingActionWrapper(env, lane_change_threshold=0.3)
    
    # Add reward wrapper (should be last)
    reward_weights = {
        'lane_following': 1.0,
        'object_avoidance': 0.8,
        'lane_changing': 0.4,
        'efficiency': 0.3,
        'safety_penalty': -2.5
    }
    env = MultiObjectiveRewardWrapper(env, reward_weights)
    
    return env
```

## Configuration Strategies

### Conservative Driving Configuration
```python
conservative_weights = {
    'lane_following': 1.0,
    'object_avoidance': 1.5,    # High safety priority
    'lane_changing': 0.1,       # Minimal lane changes
    'efficiency': 0.05,         # Low speed priority
    'safety_penalty': -5.0      # Very high safety penalty
}
```

### Aggressive Driving Configuration
```python
aggressive_weights = {
    'lane_following': 0.8,
    'object_avoidance': 0.4,
    'lane_changing': 0.8,       # More lane changes
    'efficiency': 0.7,          # High speed priority
    'safety_penalty': -1.5      # Lower safety penalty
}
```

### Balanced Configuration (Default)
```python
balanced_weights = {
    'lane_following': 1.0,
    'object_avoidance': 0.5,
    'lane_changing': 0.3,
    'efficiency': 0.2,
    'safety_penalty': -2.0
}
```

## Training Integration

### PPO Training with RLlib

```python
import ray
from ray.rllib.algorithms.ppo import PPO

def train_with_multi_objective_reward():
    ray.init()
    
    config = {
        'env': 'enhanced_duckietown_env',
        'env_config': {
            'reward_weights': {
                'lane_following': 1.0,
                'object_avoidance': 0.8,
                'lane_changing': 0.4,
                'efficiency': 0.3,
                'safety_penalty': -3.0
            }
        },
        'num_workers': 4,
        'train_batch_size': 4000,
        'framework': 'torch'
    }
    
    agent = PPO(config=config)
    
    for i in range(1000):
        result = agent.train()
        
        # Monitor reward components
        if 'custom_rewards' in result.get('info', {}):
            rewards = result['info']['custom_rewards']
            print(f"Episode {i} - Total: {rewards['total']:.3f}")
    
    ray.shutdown()
```

### Curriculum Learning

```python
def curriculum_training():
    """Example of curriculum learning with dynamic weight adjustment."""
    
    # Start with conservative weights
    initial_weights = {
        'lane_following': 1.0,
        'object_avoidance': 2.0,    # High safety initially
        'lane_changing': 0.1,       # Low complexity
        'efficiency': 0.05,
        'safety_penalty': -5.0
    }
    
    env = MultiObjectiveRewardWrapper(base_env, initial_weights)
    
    # Training phases
    for phase in range(3):
        print(f"Training Phase {phase + 1}")
        
        # Adjust weights based on phase
        if phase == 1:  # Intermediate phase
            env.update_weights({
                'object_avoidance': 1.0,
                'lane_changing': 0.3,
                'efficiency': 0.2
            })
        elif phase == 2:  # Advanced phase
            env.update_weights({
                'object_avoidance': 0.5,
                'lane_changing': 0.5,
                'efficiency': 0.4,
                'safety_penalty': -2.0
            })
        
        # Train for this phase
        train_phase(env, episodes=500)
```

## Monitoring and Debugging

### Reward Component Analysis

```python
def analyze_reward_components(env, episodes=100):
    """Analyze reward component distribution over episodes."""
    
    component_history = {
        'lane_following': [],
        'object_avoidance': [],
        'lane_changing': [],
        'efficiency': [],
        'safety_penalty': [],
        'total': []
    }
    
    for episode in range(episodes):
        obs = env.reset()
        episode_rewards = {key: [] for key in component_history.keys()}
        
        done = False
        while not done:
            action = your_agent.act(obs)  # Replace with your agent
            obs, reward, done, info = env.step(action)
            
            if 'custom_rewards' in info:
                for component, value in info['custom_rewards'].items():
                    episode_rewards[component].append(value)
        
        # Store episode averages
        for component in component_history.keys():
            if episode_rewards[component]:
                avg_reward = np.mean(episode_rewards[component])
                component_history[component].append(avg_reward)
    
    # Analyze results
    for component, values in component_history.items():
        print(f"{component}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
```

### Real-time Monitoring

```python
def monitor_training(env):
    """Real-time monitoring of reward components during training."""
    
    obs = env.reset()
    step_count = 0
    
    while True:
        action = your_agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Log every 100 steps
        if step_count % 100 == 0 and 'custom_rewards' in info:
            components = info['custom_rewards']
            print(f"Step {step_count}:")
            for component, value in components.items():
                print(f"  {component}: {value:.3f}")
        
        if done:
            obs = env.reset()
```

## API Reference

### MultiObjectiveRewardWrapper

#### Constructor
```python
MultiObjectiveRewardWrapper(env, reward_weights=None)
```

**Parameters:**
- `env`: The environment to wrap
- `reward_weights`: Optional dictionary of reward component weights

#### Methods

##### `reward(reward: float) -> float`
Calculate the multi-objective reward.

##### `step(action) -> Tuple[np.ndarray, float, bool, dict]`
Step the environment and return multi-objective reward.

##### `reset(**kwargs) -> np.ndarray`
Reset the environment and reward tracking.

##### `update_weights(new_weights: Dict[str, float])`
Update reward weights during runtime.

##### `get_reward_components() -> Dict[str, float]`
Get current reward component values.

##### `get_reward_weights() -> Dict[str, float]`
Get current reward weights.

## Best Practices

### 1. Wrapper Order
Always apply the MultiObjectiveRewardWrapper last in the wrapper stack:
```python
env = BaseEnvironment()
env = ObservationWrapper(env)
env = ActionWrapper(env)
env = MultiObjectiveRewardWrapper(env)  # Last!
```

### 2. Weight Tuning
- Start with conservative weights emphasizing safety
- Gradually increase efficiency and lane changing weights
- Monitor reward component distributions during training
- Use curriculum learning for complex behaviors

### 3. Debugging
- Always monitor individual reward components
- Log reward weights used in each training session
- Validate that reward components are in expected ranges
- Use visualization tools to understand agent behavior

### 4. Performance Considerations
- The wrapper adds minimal computational overhead
- Reward calculations are optimized for real-time use
- Consider disabling detailed logging in production training

## Troubleshooting

### Common Issues

1. **Reward Components Not Logged**
   - Ensure the wrapper is applied last in the stack
   - Check that `info['custom_rewards']` exists in step returns

2. **Unexpected Reward Values**
   - Verify reward weights are set correctly
   - Check individual component calculations
   - Ensure environment provides necessary information

3. **Training Instability**
   - Reduce safety penalty magnitude
   - Increase lane following weight
   - Use curriculum learning approach

4. **Poor Performance**
   - Check that all required wrappers are applied
   - Verify YOLO detection is working correctly
   - Monitor reward component balance

### Debug Mode

Enable debug logging for detailed reward information:
```python
import logging
logging.getLogger('duckietown_utils.wrappers.multi_objective_reward_wrapper').setLevel(logging.DEBUG)
```

## Examples and Tutorials

See the following files for complete examples:
- `examples/multi_objective_reward_integration_example.py`
- `tests/test_multi_objective_reward_wrapper.py`
- `tests/test_multi_objective_reward_wrapper_simple.py`

## Contributing

When modifying the MultiObjectiveRewardWrapper:
1. Ensure all tests pass
2. Update documentation for any API changes
3. Add tests for new functionality
4. Follow the existing code style and patterns