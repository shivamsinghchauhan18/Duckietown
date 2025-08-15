# Enhanced Duckietown RL Documentation

Welcome to the Enhanced Duckietown RL system documentation. This system extends the base Duckietown environment with advanced object detection, avoidance, lane changing, and multi-objective reward capabilities using YOLO v5 and PPO training.

## 📚 Documentation Overview

### Core Documentation
- **[API Documentation](API_Documentation.md)** - Comprehensive API reference for all wrapper classes
- **[Configuration Guide](Configuration_Guide.md)** - Complete guide to system configuration and parameter tuning
- **[Usage Examples and Tutorials](Usage_Examples_and_Tutorials.md)** - Step-by-step tutorials and practical examples
- **[Troubleshooting Guide](Troubleshooting_Guide.md)** - Solutions to common issues and problems

### Component-Specific Guides
- **[YOLO Integration](YOLO_Integration.md)** - Object detection setup and optimization
- **[Object Avoidance Integration](Object_Avoidance_Integration.md)** - Avoidance behavior configuration
- **[Lane Changing Integration](Lane_Changing_Integration.md)** - Lane changing implementation details
- **[Multi-Objective Reward Integration](Multi_Objective_Reward_Integration.md)** - Reward system design
- **[Enhanced Configuration System](Enhanced_Configuration_System.md)** - Configuration management
- **[Enhanced Logging System](Enhanced_Logging_System.md)** - Logging and monitoring
- **[Enhanced Environment Integration](Enhanced_Environment_Integration.md)** - Environment setup
- **[Error Handling and Recovery System](Error_Handling_and_Recovery_System.md)** - Error management
- **[Debugging and Visualization Tools](Debugging_and_Visualization_Tools.md)** - Development tools

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd enhanced-duckietown-rl

# Set up the environment
./setup_enhanced_environment.sh

# Activate the environment
conda activate enhanced-duckietown-rl
```

### 2. Basic Usage
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
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### 3. Training Example
```bash
# Run complete training example
python examples/complete_enhanced_training_example.py \
    --config config/enhanced_config.yml \
    --experiment-name my_experiment \
    --iterations 1000
```

## 📖 Learning Path

### For Beginners
1. Start with **[Usage Examples and Tutorials](Usage_Examples_and_Tutorials.md)** - Quick Start Guide
2. Read **[Configuration Guide](Configuration_Guide.md)** - Basic configuration
3. Try the basic training example
4. Consult **[Troubleshooting Guide](Troubleshooting_Guide.md)** if issues arise

### For Developers
1. Review **[API Documentation](API_Documentation.md)** for detailed class references
2. Study component-specific guides for areas of interest
3. Examine example implementations in the `examples/` directory
4. Use **[Debugging and Visualization Tools](Debugging_and_Visualization_Tools.md)** for development

### For Researchers
1. Understand the system architecture from **[Enhanced Environment Integration](Enhanced_Environment_Integration.md)**
2. Study **[Multi-Objective Reward Integration](Multi_Objective_Reward_Integration.md)** for reward design
3. Use **[Configuration Guide](Configuration_Guide.md)** for parameter tuning
4. Try **[Curriculum Learning Example](../examples/curriculum_learning_example.py)** for advanced training

## 🔧 System Components

### Core Wrappers
- **YOLOObjectDetectionWrapper** - Real-time object detection using YOLO v5
- **ObjectAvoidanceActionWrapper** - Potential field-based obstacle avoidance
- **LaneChangingActionWrapper** - Dynamic lane changing with safety checks
- **EnhancedObservationWrapper** - Feature extraction and normalization
- **MultiObjectiveRewardWrapper** - Balanced multi-objective rewards

### Configuration System
- **EnhancedRLConfig** - Centralized configuration management
- **YAML-based configuration** - Human-readable parameter files
- **Parameter validation** - Automatic validation with meaningful errors
- **Environment-specific configs** - Optimized settings for different scenarios

### Logging and Debugging
- **EnhancedLogger** - Structured logging with multiple output formats
- **VisualizationManager** - Real-time visualization of detections and actions
- **Performance monitoring** - System resource and timing metrics
- **Debug utilities** - Tools for troubleshooting and analysis

## 📊 Example Configurations

### Simple Lane Following
```yaml
# For basic lane following without obstacles
yolo:
  enabled: false
object_avoidance:
  enabled: false
lane_changing:
  enabled: false
rewards:
  weights:
    lane_following: 2.0
    efficiency: 1.0
    safety: -3.0
```

### Full Enhanced Mode
```yaml
# For complete enhanced functionality
yolo:
  model_path: "models/yolov5s.pt"
  confidence_threshold: 0.5
object_avoidance:
  safety_distance: 0.6
  avoidance_strength: 1.2
lane_changing:
  lane_change_threshold: 0.3
  safety_margin: 2.5
rewards:
  weights:
    lane_following: 1.0
    object_avoidance: 2.0
    lane_changing: 1.5
    efficiency: 0.8
    safety: -5.0
```

## 🎯 Training Examples

### Basic Training
```bash
python examples/complete_enhanced_training_example.py \
    --config config/enhanced_config.yml \
    --experiment-name basic_training
```

### Curriculum Learning
```bash
python examples/curriculum_learning_example.py \
    --config config/enhanced_config.yml \
    --experiment-name curriculum_training
```

### Hyperparameter Tuning
```bash
python examples/hyperparameter_tuning_example.py \
    --config config/enhanced_config.yml \
    --experiment-name hyperparameter_search \
    --num-samples 100
```

## 🔍 Debugging and Monitoring

### Enable Debug Logging
```python
from duckietown_utils.enhanced_logger import EnhancedLogger

logger = EnhancedLogger(
    log_level='DEBUG',
    log_dir='logs/debug_session',
    enable_structured_logging=True
)
```

### Real-time Visualization
```python
from duckietown_utils.visualization_manager import VisualizationManager

viz = VisualizationManager(
    enable_detection_viz=True,
    enable_action_viz=True,
    enable_reward_viz=True
)
```

### Performance Monitoring
```python
import time
import psutil

def monitor_performance():
    start_time = time.time()
    obs, reward, done, info = env.step(action)
    step_time = time.time() - start_time
    
    memory_usage = psutil.virtual_memory().percent
    print(f"Step time: {step_time*1000:.1f}ms, Memory: {memory_usage:.1f}%")
```

## 🛠️ Development Workflow

### 1. Environment Setup
- Use provided setup script
- Verify GPU availability
- Test basic functionality

### 2. Configuration
- Start with default configuration
- Modify parameters incrementally
- Validate configuration before training

### 3. Testing
- Run unit tests: `python -m pytest tests/`
- Test individual components
- Validate wrapper composition

### 4. Training
- Start with simple scenarios
- Monitor training progress
- Use checkpointing for long runs

### 5. Evaluation
- Test trained models
- Analyze performance metrics
- Compare different configurations

## 📈 Performance Optimization

### GPU Optimization
```python
import torch

# Optimize GPU usage
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.8)

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Memory Management
```python
# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use efficient data structures
from collections import deque
buffer = deque(maxlen=1000)  # Fixed-size buffer
```

### Processing Optimization
```python
# Reduce YOLO input size for speed
yolo:
  input_size: 416  # Smaller than default 640

# Use frame skipping
class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
```

## 🤝 Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Include unit tests for new features

### Documentation
- Update relevant documentation files
- Add examples for new features
- Include configuration parameters
- Update troubleshooting guide

### Testing
- Run existing tests: `python -m pytest tests/`
- Add tests for new functionality
- Test with different configurations
- Verify backward compatibility

## 📞 Support

### Getting Help
1. Check **[Troubleshooting Guide](Troubleshooting_Guide.md)** for common issues
2. Review **[API Documentation](API_Documentation.md)** for detailed references
3. Look at **[Usage Examples](Usage_Examples_and_Tutorials.md)** for implementation patterns
4. Enable debug logging for detailed error information

### Reporting Issues
When reporting issues, please include:
- System configuration (OS, Python version, GPU)
- Complete error messages and stack traces
- Configuration file used
- Steps to reproduce the issue
- Expected vs. actual behavior

### Feature Requests
For new features:
- Describe the use case and motivation
- Provide implementation suggestions if possible
- Consider backward compatibility
- Include example usage scenarios

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Duckietown project for the base simulator
- Ultralytics for YOLO v5 implementation
- Ray team for RLLib framework
- Contributors and maintainers

---

For the most up-to-date information, please refer to the individual documentation files and the project repository.