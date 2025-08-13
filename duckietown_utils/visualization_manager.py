"""
Visualization Manager for coordinating all debugging and visualization tools.

This module provides a unified interface for managing all visualization components:
- Real-time detection visualization
- Action decision visualization
- Reward component visualization
- Performance monitoring dashboard
- Log analysis and debugging
"""

import threading
import queue
import time
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
from dataclasses import dataclass

from .visualization_utils import (
    RealTimeDetectionVisualizer,
    ActionDecisionVisualizer,
    RewardComponentVisualizer,
    PerformanceMonitoringDashboard,
    DetectionVisualization,
    ActionVisualization,
    RewardVisualization
)
from .debug_utils import LogAnalyzer, DebugProfiler, ProfileSection


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
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
    
    # Update frequencies (in Hz)
    detection_update_freq: float = 30.0
    action_update_freq: float = 10.0
    reward_update_freq: float = 5.0
    performance_update_freq: float = 2.0


class VisualizationManager:
    """Unified manager for all visualization components."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Initialize visualizers
        self.detection_viz = None
        self.action_viz = None
        self.reward_viz = None
        self.performance_viz = None
        self.profiler = None
        
        # Data queues for thread-safe communication
        self.detection_queue = queue.Queue(maxsize=100)
        self.action_queue = queue.Queue(maxsize=100)
        self.reward_queue = queue.Queue(maxsize=100)
        self.performance_queue = queue.Queue(maxsize=100)
        
        # Control flags
        self.running = False
        self.threads = []
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize visualization components based on configuration."""
        if self.config.enable_detection_viz:
            self.detection_viz = RealTimeDetectionVisualizer(
                window_name=self.config.detection_window_name,
                confidence_threshold=self.config.detection_confidence_threshold
            )
        
        if self.config.enable_action_viz:
            self.action_viz = ActionDecisionVisualizer(
                max_history=self.config.max_action_history
            )
        
        if self.config.enable_reward_viz:
            self.reward_viz = RewardComponentVisualizer(
                max_history=self.config.max_reward_history
            )
        
        if self.config.enable_performance_viz:
            self.performance_viz = PerformanceMonitoringDashboard(
                max_history=self.config.max_performance_history
            )
        
        if self.config.enable_profiling:
            self.profiler = DebugProfiler()
    
    def start(self):
        """Start all visualization components and update threads."""
        if self.running:
            return
        
        self.running = True
        
        # Start visualization windows
        if self.action_viz:
            self.action_viz.show()
        
        if self.reward_viz:
            self.reward_viz.show()
        
        if self.performance_viz:
            self.performance_viz.show()
        
        # Start update threads
        if self.config.enable_action_viz:
            thread = threading.Thread(target=self._action_update_loop, daemon=True)
            thread.start()
            self.threads.append(thread)
        
        if self.config.enable_reward_viz:
            thread = threading.Thread(target=self._reward_update_loop, daemon=True)
            thread.start()
            self.threads.append(thread)
        
        if self.config.enable_performance_viz:
            thread = threading.Thread(target=self._performance_update_loop, daemon=True)
            thread.start()
            self.threads.append(thread)
    
    def stop(self):
        """Stop all visualization components."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Close visualizers
        if self.detection_viz:
            self.detection_viz.close()
        
        if self.action_viz:
            self.action_viz.close()
        
        if self.reward_viz:
            self.reward_viz.close()
        
        if self.performance_viz:
            self.performance_viz.close()
        
        # Print profiling results
        if self.profiler:
            self.profiler.print_stats()
    
    def update_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]):
        """Update detection visualization."""
        if not self.config.enable_detection_viz or not self.detection_viz:
            return
        
        try:
            # Add to queue for thread-safe processing
            self.detection_queue.put_nowait((image.copy(), detections))
        except queue.Full:
            pass  # Skip if queue is full
        
        # Process immediately for real-time display
        self.detection_viz.show_frame(image, detections)
    
    def update_action(self, action_type: str, action_values: np.ndarray, 
                     reasoning: str, safety_critical: bool = False):
        """Update action decision visualization."""
        if not self.config.enable_action_viz or not self.action_viz:
            return
        
        action_viz = ActionVisualization(
            action_type=action_type,
            action_values=action_values,
            reasoning=reasoning,
            timestamp=time.time(),
            safety_critical=safety_critical
        )
        
        try:
            self.action_queue.put_nowait(action_viz)
        except queue.Full:
            pass  # Skip if queue is full
    
    def update_reward(self, components: Dict[str, float], total_reward: float):
        """Update reward component visualization."""
        if not self.config.enable_reward_viz or not self.reward_viz:
            return
        
        reward_viz = RewardVisualization(
            components=components,
            total_reward=total_reward,
            timestamp=time.time()
        )
        
        try:
            self.reward_queue.put_nowait(reward_viz)
        except queue.Full:
            pass  # Skip if queue is full
    
    def update_performance(self, fps: float, detection_time: float, 
                          action_time: float, memory_usage: float):
        """Update performance monitoring dashboard."""
        if not self.config.enable_performance_viz or not self.performance_viz:
            return
        
        try:
            self.performance_queue.put_nowait((fps, detection_time, action_time, memory_usage))
        except queue.Full:
            pass  # Skip if queue is full
    
    def profile_section(self, name: str):
        """Get a context manager for profiling a code section."""
        if self.profiler:
            return ProfileSection(self.profiler, name)
        else:
            return DummyProfileSection()
    
    def get_profiling_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current profiling statistics."""
        if self.profiler:
            return self.profiler.get_stats()
        return {}
    
    def reset_profiling(self):
        """Reset profiling statistics."""
        if self.profiler:
            self.profiler.reset()
    
    def _action_update_loop(self):
        """Update loop for action visualization."""
        update_interval = 1.0 / self.config.action_update_freq
        
        while self.running:
            try:
                # Process all queued actions
                while not self.action_queue.empty():
                    action_viz = self.action_queue.get_nowait()
                    self.action_viz.add_action(action_viz)
                
                time.sleep(update_interval)
            except Exception as e:
                print(f"Error in action update loop: {e}")
                time.sleep(update_interval)
    
    def _reward_update_loop(self):
        """Update loop for reward visualization."""
        update_interval = 1.0 / self.config.reward_update_freq
        
        while self.running:
            try:
                # Process all queued rewards
                while not self.reward_queue.empty():
                    reward_viz = self.reward_queue.get_nowait()
                    self.reward_viz.add_reward(reward_viz)
                
                time.sleep(update_interval)
            except Exception as e:
                print(f"Error in reward update loop: {e}")
                time.sleep(update_interval)
    
    def _performance_update_loop(self):
        """Update loop for performance monitoring."""
        update_interval = 1.0 / self.config.performance_update_freq
        
        while self.running:
            try:
                # Process all queued performance metrics
                while not self.performance_queue.empty():
                    fps, detection_time, action_time, memory_usage = self.performance_queue.get_nowait()
                    self.performance_viz.add_metrics(fps, detection_time, action_time, memory_usage)
                
                time.sleep(update_interval)
            except Exception as e:
                print(f"Error in performance update loop: {e}")
                time.sleep(update_interval)
    
    def create_debug_session(self, log_directory: str) -> str:
        """Create a comprehensive debug session with log analysis."""
        analyzer = LogAnalyzer(log_directory)
        report = analyzer.generate_debug_report()
        analyzer.create_visualization_plots()
        
        return report
    
    def save_current_state(self, output_path: str):
        """Save current visualization state and profiling data."""
        import json
        from pathlib import Path
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Save profiling stats
        if self.profiler:
            stats = self.profiler.get_stats()
            with open(output_dir / 'profiling_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
        
        # Save configuration
        config_dict = {
            'enable_detection_viz': self.config.enable_detection_viz,
            'enable_action_viz': self.config.enable_action_viz,
            'enable_reward_viz': self.config.enable_reward_viz,
            'enable_performance_viz': self.config.enable_performance_viz,
            'enable_profiling': self.config.enable_profiling,
            'detection_confidence_threshold': self.config.detection_confidence_threshold,
            'max_action_history': self.config.max_action_history,
            'max_reward_history': self.config.max_reward_history,
            'max_performance_history': self.config.max_performance_history
        }
        
        with open(output_dir / 'visualization_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Visualization state saved to {output_dir}")


class DummyProfileSection:
    """Dummy context manager when profiling is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Convenience function for quick setup
def create_visualization_manager(enable_all: bool = True, 
                               log_directory: Optional[str] = None) -> VisualizationManager:
    """Create a visualization manager with common settings."""
    config = VisualizationConfig(
        enable_detection_viz=enable_all,
        enable_action_viz=enable_all,
        enable_reward_viz=enable_all,
        enable_performance_viz=enable_all,
        enable_profiling=enable_all
    )
    
    manager = VisualizationManager(config)
    
    if log_directory:
        # Generate initial debug report
        report = manager.create_debug_session(log_directory)
        print("Debug report generated:")
        print(report[:500] + "..." if len(report) > 500 else report)
    
    return manager