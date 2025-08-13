# Debugging and Visualization Tools

This document describes the comprehensive debugging and visualization tools available for the Enhanced Duckietown RL system. These tools provide real-time monitoring, log analysis, performance profiling, and debugging capabilities.

## Overview

The debugging and visualization system consists of several components:

1. **Real-time Visualization Tools** - Live monitoring of detections, actions, rewards, and performance
2. **Log Analysis Utilities** - Comprehensive analysis of structured log files
3. **Performance Profiling** - Code profiling and bottleneck identification
4. **Debug Reporting** - Automated generation of debug reports with recommendations
5. **Visualization Manager** - Unified interface for coordinating all visualization components

## Components

### 1. Real-Time Detection Visualizer

Provides live visualization of YOLO object detections with bounding boxes, confidence scores, and distance estimates.

**Features:**
- Real-time bounding box overlay on camera feed
- Configurable confidence thresholds
- Color-coded object classes
- Distance and confidence information display
- Detection count monitoring

**Usage:**
```python
from duckietown_utils.visualization_utils import RealTimeDetectionVisualizer

visualizer = RealTimeDetectionVisualizer(
    window_name="Object Detections",
    confidence_threshold=0.5
)

# In your detection loop
visualizer.show_frame(image, detections)
```

### 2. Action Decision Visualizer

Visualizes action decisions with reasoning and safety analysis over time.

**Features:**
- Action values plotted over time
- Action type distribution pie chart
- Safety critical action tracking
- Latest action reasoning display
- Historical action pattern analysis

**Usage:**
```python
from duckietown_utils.visualization_utils import ActionDecisionVisualizer, ActionVisualization

visualizer = ActionDecisionVisualizer(max_history=100)
visualizer.show()

# Add action data
action_viz = ActionVisualization(
    action_type="object_avoidance",
    action_values=np.array([-0.3, 0.5]),
    reasoning="Avoiding duckiebot at 1.2m distance",
    timestamp=time.time(),
    safety_critical=True
)
visualizer.add_action(action_viz)
```

### 3. Reward Component Visualizer

Analyzes and visualizes reward components for training analysis.

**Features:**
- Total reward trend over time
- Individual component contributions
- Recent reward distribution
- Component statistics and correlations
- Episode-based reward analysis

**Usage:**
```python
from duckietown_utils.visualization_utils import RewardComponentVisualizer, RewardVisualization

visualizer = RewardComponentVisualizer(max_history=1000)
visualizer.show()

# Add reward data
reward_viz = RewardVisualization(
    components={
        'lane_following': 0.5,
        'object_avoidance': 0.2,
        'safety_penalty': -0.1
    },
    total_reward=0.6,
    timestamp=time.time()
)
visualizer.add_reward(reward_viz)
```

### 4. Performance Monitoring Dashboard

Real-time monitoring of system performance metrics.

**Features:**
- FPS monitoring with target thresholds
- Detection processing time tracking
- Action processing time analysis
- GPU memory usage monitoring
- System health indicators
- Performance statistics summary

**Usage:**
```python
from duckietown_utils.visualization_utils import PerformanceMonitoringDashboard

dashboard = PerformanceMonitoringDashboard(max_history=1000)
dashboard.show()

# Add performance metrics
dashboard.add_metrics(
    fps=15.0,
    detection_time=25.0,  # ms
    action_time=5.0,      # ms
    memory_usage=1024.0   # MB
)
```

### 5. Log Analysis Utilities

Comprehensive analysis of structured log files with automated report generation.

**Features:**
- Multi-format log parsing (JSONL, structured logs)
- Detection performance analysis
- Action pattern analysis
- Reward trend analysis
- Performance metrics analysis
- Automated visualization plot generation
- Debug report generation with recommendations

**Usage:**
```python
from duckietown_utils.debug_utils import LogAnalyzer, create_debug_session

# Analyze logs
analyzer = LogAnalyzer("logs/enhanced_logging_demo")
analyzer.parse_log_files()

# Run specific analyses
detection_analysis = analyzer.analyze_detection_performance()
action_analysis = analyzer.analyze_action_patterns()
reward_analysis = analyzer.analyze_reward_trends()

# Generate comprehensive report
report = analyzer.generate_debug_report("debug_report.md")

# Create visualization plots
analyzer.create_visualization_plots("analysis_plots/")

# Or use convenience function
analyzer, report = create_debug_session("logs/enhanced_logging_demo")
```

### 6. Performance Profiler

Code profiling for identifying performance bottlenecks.

**Features:**
- Section-based timing with context managers
- Statistical analysis of timing data
- Bottleneck identification
- Performance regression detection
- Thread-safe profiling

**Usage:**
```python
from duckietown_utils.debug_utils import DebugProfiler, ProfileSection

profiler = DebugProfiler()

# Method 1: Manual timing
profiler.start_timer("detection")
# ... detection code ...
profiler.end_timer("detection")

# Method 2: Context manager
with ProfileSection(profiler, "action_processing"):
    # ... action processing code ...
    pass

# Get statistics
stats = profiler.get_stats()
profiler.print_stats()
```

### 7. Visualization Manager

Unified interface for coordinating all visualization components.

**Features:**
- Centralized configuration management
- Thread-safe data queues
- Automatic update loops
- Integrated profiling
- State saving and loading
- Debug session creation

**Usage:**
```python
from duckietown_utils.visualization_manager import (
    VisualizationManager, 
    VisualizationConfig,
    create_visualization_manager
)

# Create with custom configuration
config = VisualizationConfig(
    enable_detection_viz=True,
    enable_action_viz=True,
    enable_reward_viz=True,
    enable_performance_viz=True,
    detection_confidence_threshold=0.7,
    max_action_history=200
)

manager = VisualizationManager(config)
manager.start()

# Update data
manager.update_detections(image, detections)
manager.update_action("lane_following", action_values, "Following lane center")
manager.update_reward(reward_components, total_reward)
manager.update_performance(fps, detection_time, action_time, memory_usage)

# Profile code sections
with manager.profile_section("critical_section"):
    # ... critical code ...
    pass

# Stop and get results
manager.stop()
stats = manager.get_profiling_stats()

# Or use convenience function
manager = create_visualization_manager(enable_all=True, log_directory="logs/")
```

## Command-Line Interface

The `debug_enhanced_rl.py` script provides a convenient command-line interface for all debugging tools.

### Available Commands

#### Analyze Logs
```bash
python debug_enhanced_rl.py analyze logs/enhanced_logging_demo
python debug_enhanced_rl.py analyze logs/enhanced_logging_demo --output-dir analysis_results/
```

#### Generate Debug Report
```bash
python debug_enhanced_rl.py report logs/enhanced_logging_demo
python debug_enhanced_rl.py report logs/enhanced_logging_demo --output debug_report.md
```

#### Real-Time Visualization
```bash
python debug_enhanced_rl.py visualize --real-time
python debug_enhanced_rl.py visualize --real-time --duration 120
```

#### Performance Profiling
```bash
python debug_enhanced_rl.py profile --duration 30
python debug_enhanced_rl.py profile --log-dir logs/enhanced_logging_demo
```

#### List Available Logs
```bash
python debug_enhanced_rl.py list-logs
```

## Integration with Enhanced RL System

### In Training Scripts

```python
from duckietown_utils.visualization_manager import create_visualization_manager

# Create visualization manager
viz_manager = create_visualization_manager(enable_all=True)
viz_manager.start()

# In training loop
for episode in range(num_episodes):
    obs = env.reset()
    
    while not done:
        # Profile action selection
        with viz_manager.profile_section("action_selection"):
            action = agent.act(obs)
        
        # Profile environment step
        with viz_manager.profile_section("env_step"):
            obs, reward, done, info = env.step(action)
        
        # Update visualizations
        if 'detections' in info:
            viz_manager.update_detections(obs['image'], info['detections'])
        
        if 'action_reasoning' in info:
            viz_manager.update_action(
                info['action_type'], 
                action, 
                info['action_reasoning'],
                info.get('safety_critical', False)
            )
        
        if 'reward_components' in info:
            viz_manager.update_reward(info['reward_components'], reward)
        
        # Update performance metrics
        viz_manager.update_performance(
            info.get('fps', 0),
            info.get('detection_time', 0),
            info.get('action_time', 0),
            info.get('memory_usage', 0)
        )

# Stop visualization and save results
viz_manager.stop()
viz_manager.save_current_state("training_session_debug/")
```

### In Wrapper Classes

```python
class EnhancedWrapper(gym.Wrapper):
    def __init__(self, env, viz_manager=None):
        super().__init__(env)
        self.viz_manager = viz_manager
    
    def step(self, action):
        if self.viz_manager:
            with self.viz_manager.profile_section("wrapper_processing"):
                # ... wrapper logic ...
                pass
        
        obs, reward, done, info = self.env.step(action)
        
        # Update visualizations
        if self.viz_manager and 'detections' in info:
            self.viz_manager.update_detections(obs['image'], info['detections'])
        
        return obs, reward, done, info
```

## Configuration Options

### VisualizationConfig Parameters

```python
@dataclass
class VisualizationConfig:
    # Enable/disable components
    enable_detection_viz: bool = True
    enable_action_viz: bool = True
    enable_reward_viz: bool = True
    enable_performance_viz: bool = True
    enable_profiling: bool = True
    
    # Detection visualization settings
    detection_confidence_threshold: float = 0.5
    detection_window_name: str = "Object Detections"
    
    # History settings
    max_action_history: int = 100
    max_reward_history: int = 1000
    max_performance_history: int = 1000
    
    # Update frequencies (Hz)
    detection_update_freq: float = 30.0
    action_update_freq: float = 10.0
    reward_update_freq: float = 5.0
    performance_update_freq: float = 2.0
```

## Performance Considerations

### Real-Time Visualization
- Detection visualization runs at camera frame rate (30 FPS)
- Action/reward visualizations update at lower frequencies to reduce overhead
- Use separate threads for visualization updates to avoid blocking main loop

### Memory Usage
- History buffers are limited by `max_*_history` parameters
- Visualization plots are updated incrementally
- Log analysis loads entire files into memory - consider file size limits

### Profiling Overhead
- Profiling adds minimal overhead (~0.1ms per section)
- Can be disabled entirely by setting `enable_profiling=False`
- Use context managers for automatic cleanup

## Troubleshooting

### Common Issues

1. **Visualization windows not appearing**
   - Check if running in headless environment
   - Ensure matplotlib backend supports GUI
   - Try setting `DISPLAY` environment variable

2. **Log analysis fails**
   - Verify log file format (JSONL for structured logs)
   - Check file permissions
   - Ensure log directory exists and contains valid files

3. **Performance issues**
   - Reduce visualization update frequencies
   - Disable unused visualization components
   - Limit history buffer sizes

4. **Memory leaks**
   - Ensure `stop()` is called on visualization manager
   - Check for circular references in callback functions
   - Monitor memory usage with performance dashboard

### Debug Tips

1. **Use profiling to identify bottlenecks**
   ```python
   with viz_manager.profile_section("suspected_bottleneck"):
       # ... code to profile ...
       pass
   ```

2. **Enable debug logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Save visualization state for later analysis**
   ```python
   viz_manager.save_current_state("debug_session/")
   ```

4. **Use log analysis for post-mortem debugging**
   ```bash
   python debug_enhanced_rl.py analyze logs/problematic_session/
   ```

## Examples

See `examples/visualization_debugging_example.py` for comprehensive usage examples and demos of all visualization tools.

## Testing

Run the test suite to verify all components:

```bash
python -m pytest tests/test_visualization_tools.py -v
```

The test suite covers:
- All visualization components
- Log analysis functionality
- Performance profiling
- Visualization manager coordination
- Error handling and edge cases