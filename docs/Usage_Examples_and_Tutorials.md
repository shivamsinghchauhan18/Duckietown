# Enhanced Duckietown RL Usage Examples and Tutorials

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [Basic Training Tutorial](#basic-training-tutorial)
3. [Advanced Configuration Tutorial](#advanced-configuration-tutorial)
4. [Custom Wrapper Development](#custom-wrapper-development)
5. [Performance Optimization](#performance-optimization)
6. [Debugging and Visualization](#debugging-and-visualization)

## Quick Start Guide

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Duckietown simulator installed

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd enhanced-duckietown-rl

# Set up the environment
./setup_enhanced_environment.sh

# Activate the environment
conda activate enhanced-duckietown-rl
```

### Basic Usage
```python
from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig

# Load configuration
config = EnhancedRLConfig.from_yaml('config/enhanced_config.yml')

# Create enhanced environment
env = launch_and_wrap_enhanced_env(
    map_name='loop_obstacles',
    config=config
)

# Basic interaction
obs = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
        
env.close()
```

## Basic Training Tutorial

### Step 1: Environment Setup
```python
import gym
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig

def create_env(env_config):
    """Environment factory function for RLLib."""
    config = EnhancedRLConfig.from_yaml('config/enhanced_config.yml')
    return launch_and_wrap_enhanced_env(
        map_name=env_config.get('map_name', 'loop_obstacles'),
        config=config
    )

# Register environment
from ray.tune.registry import register_env
register_env("enhanced_duckietown", create_env)
```

### Step 2: Training Configuration
```python
# PPO training configuration
training_config = {
    "env": "enhanced_duckietown",
    "env_config": {
        "map_name": "loop_obstacles"
    },
    "framework": "torch",
    "num_workers": 4,
    "num_envs_per_worker": 1,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "callbacks": "duckietown_utils.enhanced_rllib_callbacks.EnhancedCallbacks"
}
```

### Step 3: Training Execution
```python
import ray
from ray.rllib.agents.ppo import PPOTrainer

# Initialize Ray
ray.init()

# Create trainer
trainer = PPOTrainer(config=training_config)

# Training loop
for iteration in range(1000):
    result = trainer.train()
    
    # Print progress
    if iteration % 10 == 0:
        print(f"Iteration {iteration}")
        print(f"  Episode Reward Mean: {result['episode_reward_mean']:.2f}")
        print(f"  Episode Length Mean: {result['episode_len_mean']:.2f}")
    
    # Save checkpoint
    if iteration % 100 == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint saved at: {checkpoint}")

# Clean up
trainer.stop()
ray.shutdown()
```

### Step 4: Model Evaluation
```python
from ray.rllib.agents.ppo import PPOTrainer

# Load trained model
trainer = PPOTrainer(config=training_config)
trainer.restore("path/to/checkpoint")

# Create evaluation environment
env = create_env({"map_name": "loop_obstacles"})

# Evaluate model
total_reward = 0
obs = env.reset()

for step in range(1000):
    action = trainer.compute_action(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        print(f"Episode finished with total reward: {total_reward}")
        total_reward = 0
        obs = env.reset()

env.close()
```

## Advanced Configuration Tutorial

### Custom Configuration File
Create a custom configuration file `my_config.yml`:

```yaml
# YOLO Configuration
yolo:
  model_path: "models/yolo5m.pt"
  confidence_threshold: 0.6
  device: "cuda"
  max_detections: 15

# Object Avoidance Configuration
object_avoidance:
  safety_distance: 0.7
  avoidance_strength: 1.2
  min_clearance: 0.3
  smoothing_factor: 0.4

# Lane Changing Configuration
lane_changing:
  lane_change_threshold: 0.4
  safety_margin: 2.5
  max_lane_change_time: 4.0
  evaluation_frequency: 8

# Reward Configuration
rewards:
  weights:
    lane_following: 1.2
    object_avoidance: 2.5
    lane_changing: 1.8
    efficiency: 0.6
    safety: -8.0
  safety_penalty_scale: 12.0
  efficiency_bonus_scale: 1.2

# Logging Configuration
logging:
  level: "DEBUG"
  log_detections: true
  log_actions: true
  log_rewards: true
  log_performance: true
  output_dir: "logs/custom_training"
```

### Using Custom Configuration
```python
from config.enhanced_config import EnhancedRLConfig

# Load custom configuration
config = EnhancedRLConfig.from_yaml('my_config.yml')

# Override specific parameters
config.yolo.confidence_threshold = 0.7
config.rewards.weights['safety'] = -10.0

# Create environment with custom config
env = launch_and_wrap_enhanced_env(
    map_name='multi_track',
    config=config
)
```

### Curriculum Learning Setup
```python
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    def __init__(self):
        self.stages = [
            {
                'name': 'basic_lane_following',
                'map_name': 'loop_empty',
                'max_steps': 100000,
                'success_threshold': 0.8
            },
            {
                'name': 'static_obstacles',
                'map_name': 'loop_obstacles',
                'max_steps': 200000,
                'success_threshold': 0.7
            },
            {
                'name': 'dynamic_obstacles',
                'map_name': 'loop_dyn_duckiebots',
                'max_steps': 300000,
                'success_threshold': 0.6
            },
            {
                'name': 'multi_lane',
                'map_name': 'multi_track',
                'max_steps': 400000,
                'success_threshold': 0.5
            }
        ]
        self.current_stage = 0

def curriculum_env_creator(env_config):
    """Environment creator with curriculum learning."""
    curriculum = env_config.get('curriculum', CurriculumConfig())
    stage = curriculum.stages[curriculum.current_stage]
    
    config = EnhancedRLConfig.from_yaml('config/enhanced_config.yml')
    
    return launch_and_wrap_enhanced_env(
        map_name=stage['map_name'],
        config=config
    )
```

## Custom Wrapper Development

### Creating a Custom Observation Wrapper
```python
import gym
import numpy as np
from typing import Dict, Any

class CustomFeatureWrapper(gym.ObservationWrapper):
    """Custom wrapper that adds domain-specific features."""
    
    def __init__(self, env, feature_config: Dict[str, Any]):
        super().__init__(env)
        self.feature_config = feature_config
        
        # Modify observation space
        original_space = env.observation_space
        additional_features = feature_config.get('additional_features', 10)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_space.shape[0] + additional_features,),
            dtype=np.float32
        )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Add custom features to observation."""
        
        # Extract custom features
        custom_features = self._extract_custom_features(observation)
        
        # Combine with original observation
        enhanced_obs = np.concatenate([observation, custom_features])
        
        return enhanced_obs.astype(np.float32)
    
    def _extract_custom_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract domain-specific features."""
        features = []
        
        # Example: Add statistical features
        if self.feature_config.get('include_statistics', True):
            features.extend([
                np.mean(obs),
                np.std(obs),
                np.min(obs),
                np.max(obs)
            ])
        
        # Example: Add frequency domain features
        if self.feature_config.get('include_fft', False):
            fft_features = np.abs(np.fft.fft(obs.flatten()))[:6]
            features.extend(fft_features)
        
        return np.array(features, dtype=np.float32)

# Usage
feature_config = {
    'additional_features': 10,
    'include_statistics': True,
    'include_fft': True
}

env = CustomFeatureWrapper(base_env, feature_config)
```

### Creating a Custom Action Wrapper
```python
class AdaptiveActionWrapper(gym.ActionWrapper):
    """Wrapper that adapts actions based on performance history."""
    
    def __init__(self, env, adaptation_config: Dict[str, Any]):
        super().__init__(env)
        self.adaptation_config = adaptation_config
        self.performance_history = []
        self.adaptation_factor = 1.0
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """Adapt action based on recent performance."""
        
        # Apply adaptation
        adapted_action = action * self.adaptation_factor
        
        # Clip to valid range
        adapted_action = np.clip(
            adapted_action,
            self.action_space.low,
            self.action_space.high
        )
        
        return adapted_action
    
    def step(self, action):
        """Override step to track performance."""
        obs, reward, done, info = super().step(action)
        
        # Update performance history
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Update adaptation factor
        self._update_adaptation_factor()
        
        return obs, reward, done, info
    
    def _update_adaptation_factor(self):
        """Update adaptation factor based on performance."""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = np.mean(self.performance_history[-10:])
        overall_performance = np.mean(self.performance_history)
        
        if recent_performance < overall_performance * 0.8:
            # Performance declining, reduce action magnitude
            self.adaptation_factor *= 0.95
        elif recent_performance > overall_performance * 1.2:
            # Performance improving, increase action magnitude
            self.adaptation_factor *= 1.05
        
        # Clip adaptation factor
        self.adaptation_factor = np.clip(self.adaptation_factor, 0.5, 2.0)
```

## Performance Optimization

### GPU Memory Optimization
```python
import torch

def optimize_yolo_memory():
    """Optimize YOLO model for memory efficiency."""
    
    # Enable mixed precision
    torch.backends.cudnn.benchmark = True
    
    # Set memory fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Clear cache periodically
    def clear_cache_callback():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return clear_cache_callback

# Use in training
clear_cache = optimize_yolo_memory()

for episode in range(num_episodes):
    # Training code here
    
    if episode % 10 == 0:
        clear_cache()
```

### Parallel Environment Setup
```python
from ray.rllib.env import ParallelRolloutWorker
from ray.rllib.env.vector_env import VectorEnv

def create_parallel_envs(num_envs: int = 4):
    """Create parallel environments for faster training."""
    
    def env_creator(env_config):
        config = EnhancedRLConfig.from_yaml('config/enhanced_config.yml')
        return launch_and_wrap_enhanced_env(
            map_name=env_config.get('map_name', 'loop_obstacles'),
            config=config
        )
    
    # Create vector environment
    envs = [env_creator({'map_name': 'loop_obstacles'}) for _ in range(num_envs)]
    vector_env = VectorEnv(envs)
    
    return vector_env

# Usage in training
parallel_env = create_parallel_envs(num_envs=8)
```

### Performance Monitoring
```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    """Monitor system performance during training."""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.gpu_usage = []
    
    def start_step(self):
        """Start timing a step."""
        self.step_start = time.time()
    
    def end_step(self):
        """End timing a step and record metrics."""
        step_time = time.time() - self.step_start
        self.step_times.append(step_time)
        
        # Record memory usage
        memory = psutil.virtual_memory().percent
        self.memory_usage.append(memory)
        
        # Record GPU usage
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].memoryUtil * 100
                self.gpu_usage.append(gpu_usage)
        except:
            pass
    
    def get_stats(self):
        """Get performance statistics."""
        return {
            'avg_step_time': np.mean(self.step_times),
            'max_step_time': np.max(self.step_times),
            'avg_memory_usage': np.mean(self.memory_usage),
            'avg_gpu_usage': np.mean(self.gpu_usage) if self.gpu_usage else 0
        }

# Usage
monitor = PerformanceMonitor()

for step in range(1000):
    monitor.start_step()
    
    # Training step
    obs, reward, done, info = env.step(action)
    
    monitor.end_step()
    
    if step % 100 == 0:
        stats = monitor.get_stats()
        print(f"Performance stats: {stats}")
```

## Debugging and Visualization

### Real-time Visualization
```python
import matplotlib.pyplot as plt
from duckietown_utils.visualization_manager import VisualizationManager

def setup_realtime_visualization():
    """Set up real-time visualization for debugging."""
    
    viz_manager = VisualizationManager(
        enable_detection_viz=True,
        enable_action_viz=True,
        enable_reward_viz=True
    )
    
    return viz_manager

# Usage
viz = setup_realtime_visualization()

obs = env.reset()
for step in range(1000):
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    
    # Update visualization
    viz.update(obs, action, reward, info)
    
    if done:
        obs = env.reset()
```

### Debug Logging Setup
```python
import logging
from duckietown_utils.enhanced_logger import EnhancedLogger

def setup_debug_logging():
    """Set up comprehensive debug logging."""
    
    logger = EnhancedLogger(
        log_level='DEBUG',
        log_dir='logs/debug_session',
        enable_structured_logging=True
    )
    
    # Configure specific loggers
    logger.configure_detection_logging(log_confidence=True, log_bboxes=True)
    logger.configure_action_logging(log_reasoning=True)
    logger.configure_reward_logging(log_components=True)
    
    return logger

# Usage
debug_logger = setup_debug_logging()

# Training with debug logging
for episode in range(100):
    obs = env.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Log step information
        debug_logger.log_step(obs, action, reward, info)
        
        if done:
            debug_logger.log_episode_end(episode_reward)
            break
```

### Troubleshooting Common Issues
```python
def diagnose_environment_issues(env):
    """Diagnose common environment setup issues."""
    
    issues = []
    
    # Check YOLO model
    try:
        if hasattr(env, 'yolo_wrapper'):
            test_obs = env.observation_space.sample()
            detections = env.yolo_wrapper.detect(test_obs)
            if len(detections) == 0:
                issues.append("YOLO model not detecting objects in test image")
    except Exception as e:
        issues.append(f"YOLO detection error: {e}")
    
    # Check action space
    try:
        test_action = env.action_space.sample()
        if not env.action_space.contains(test_action):
            issues.append("Action space validation failed")
    except Exception as e:
        issues.append(f"Action space error: {e}")
    
    # Check observation space
    try:
        obs = env.reset()
        if not env.observation_space.contains(obs):
            issues.append("Observation space validation failed")
    except Exception as e:
        issues.append(f"Observation space error: {e}")
    
    return issues

# Usage
issues = diagnose_environment_issues(env)
if issues:
    print("Environment issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Environment validation passed!")
```

This comprehensive tutorial provides practical examples and guidance for using the Enhanced Duckietown RL system effectively, from basic setup to advanced customization and optimization techniques.