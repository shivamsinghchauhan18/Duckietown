"""
Visualization utilities for debugging and monitoring the enhanced Duckietown RL system.

This module provides real-time visualization tools for:
- Object detections and bounding boxes
- Action decisions with reasoning
- Reward components analysis
- Performance metrics monitoring
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass
from collections import deque
import threading
import queue


@dataclass
class DetectionVisualization:
    """Data structure for detection visualization."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    distance: float
    color: Tuple[int, int, int] = (0, 255, 0)


@dataclass
class ActionVisualization:
    """Data structure for action visualization."""
    action_type: str
    action_values: np.ndarray
    reasoning: str
    timestamp: float
    safety_critical: bool = False


@dataclass
class RewardVisualization:
    """Data structure for reward component visualization."""
    components: Dict[str, float]
    total_reward: float
    timestamp: float


class RealTimeDetectionVisualizer:
    """Real-time visualization for object detections and bounding boxes."""
    
    def __init__(self, window_name: str = "Object Detections", 
                 confidence_threshold: float = 0.5):
        self.window_name = window_name
        self.confidence_threshold = confidence_threshold
        self.colors = {
            'duckiebot': (0, 255, 0),    # Green
            'duckie': (255, 255, 0),     # Yellow
            'cone': (0, 165, 255),       # Orange
            'truck': (0, 0, 255),        # Red
            'bus': (255, 0, 255),        # Magenta
            'default': (128, 128, 128)   # Gray
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        for detection in detections:
            if detection['confidence'] < self.confidence_threshold:
                continue
                
            # Extract detection info
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            distance = detection.get('distance', 0.0)
            
            # Get color for class
            color = self.colors.get(class_name, self.colors['default'])
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.thickness)
            
            # Create label text
            label = f"{class_name}: {confidence:.2f}"
            if distance > 0:
                label += f" ({distance:.2f}m)"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness)
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_height - baseline - 5),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - baseline - 2),
                       self.font, self.font_scale, (255, 255, 255), 1)
        
        # Add detection count
        detection_count = len([d for d in detections 
                             if d['confidence'] >= self.confidence_threshold])
        count_text = f"Detections: {detection_count}"
        cv2.putText(vis_image, count_text, (10, 30),
                   self.font, self.font_scale, (255, 255, 255), 2)
        
        return vis_image
    
    def show_frame(self, image: np.ndarray, detections: List[Dict[str, Any]]):
        """Display a single frame with detections."""
        vis_image = self.visualize_detections(image, detections)
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
    
    def close(self):
        """Close the visualization window."""
        cv2.destroyWindow(self.window_name)


class ActionDecisionVisualizer:
    """Visualization for action decisions with reasoning display."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.action_history = deque(maxlen=max_history)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Action Decision Analysis')
        
        # Initialize plots
        self._setup_plots()
        
    def _setup_plots(self):
        """Setup the subplot layout."""
        # Action values over time
        self.axes[0, 0].set_title('Action Values Over Time')
        self.axes[0, 0].set_xlabel('Time Step')
        self.axes[0, 0].set_ylabel('Action Value')
        
        # Action type distribution
        self.axes[0, 1].set_title('Action Type Distribution')
        
        # Safety critical actions
        self.axes[1, 0].set_title('Safety Critical Actions')
        self.axes[1, 0].set_xlabel('Time Step')
        self.axes[1, 0].set_ylabel('Safety Critical')
        
        # Reasoning text area
        self.axes[1, 1].set_title('Latest Action Reasoning')
        self.axes[1, 1].axis('off')
        
    def add_action(self, action_viz: ActionVisualization):
        """Add a new action to the visualization."""
        self.action_history.append(action_viz)
        self._update_plots()
        
    def _update_plots(self):
        """Update all plots with current data."""
        if not self.action_history:
            return
            
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        self._setup_plots()
        
        # Extract data
        timestamps = [a.timestamp for a in self.action_history]
        action_values = [a.action_values for a in self.action_history]
        action_types = [a.action_type for a in self.action_history]
        safety_flags = [a.safety_critical for a in self.action_history]
        
        # Plot action values over time
        if action_values:
            action_array = np.array(action_values)
            for i in range(action_array.shape[1]):
                self.axes[0, 0].plot(timestamps, action_array[:, i], 
                                   label=f'Action {i}')
            self.axes[0, 0].legend()
            self.axes[0, 0].set_title('Action Values Over Time')
        
        # Plot action type distribution
        if action_types:
            type_counts = {}
            for action_type in action_types:
                type_counts[action_type] = type_counts.get(action_type, 0) + 1
            
            self.axes[0, 1].pie(type_counts.values(), labels=type_counts.keys(),
                              autopct='%1.1f%%')
            self.axes[0, 1].set_title('Action Type Distribution')
        
        # Plot safety critical actions
        if safety_flags:
            self.axes[1, 0].plot(timestamps, safety_flags, 'ro-', markersize=3)
            self.axes[1, 0].set_ylim(-0.1, 1.1)
            self.axes[1, 0].set_title('Safety Critical Actions')
        
        # Show latest reasoning
        if self.action_history:
            latest = self.action_history[-1]
            reasoning_text = f"Type: {latest.action_type}\n"
            reasoning_text += f"Values: {latest.action_values}\n"
            reasoning_text += f"Safety Critical: {latest.safety_critical}\n\n"
            reasoning_text += f"Reasoning:\n{latest.reasoning}"
            
            self.axes[1, 1].text(0.05, 0.95, reasoning_text,
                               transform=self.axes[1, 1].transAxes,
                               verticalalignment='top',
                               fontsize=10,
                               wrap=True)
            self.axes[1, 1].set_title('Latest Action Reasoning')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def show(self):
        """Show the visualization window."""
        plt.show(block=False)
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)


class RewardComponentVisualizer:
    """Visualization for reward components analysis."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.reward_history = deque(maxlen=max_history)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Reward Component Analysis')
        
        # Component colors
        self.component_colors = {
            'lane_following': 'blue',
            'object_avoidance': 'green',
            'lane_changing': 'orange',
            'efficiency': 'purple',
            'safety_penalty': 'red',
            'total': 'black'
        }
        
    def add_reward(self, reward_viz: RewardVisualization):
        """Add a new reward to the visualization."""
        self.reward_history.append(reward_viz)
        self._update_plots()
        
    def _update_plots(self):
        """Update all plots with current reward data."""
        if not self.reward_history:
            return
            
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Extract data
        timestamps = [r.timestamp for r in self.reward_history]
        total_rewards = [r.total_reward for r in self.reward_history]
        
        # Get all component names
        all_components = set()
        for reward in self.reward_history:
            all_components.update(reward.components.keys())
        
        # Plot total reward over time
        self.axes[0, 0].plot(timestamps, total_rewards, 
                           color=self.component_colors.get('total', 'black'),
                           linewidth=2, label='Total Reward')
        self.axes[0, 0].set_title('Total Reward Over Time')
        self.axes[0, 0].set_xlabel('Time Step')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot component rewards over time
        for component in all_components:
            component_values = []
            for reward in self.reward_history:
                component_values.append(reward.components.get(component, 0.0))
            
            color = self.component_colors.get(component, 'gray')
            self.axes[0, 1].plot(timestamps, component_values, 
                               color=color, label=component, alpha=0.7)
        
        self.axes[0, 1].set_title('Reward Components Over Time')
        self.axes[0, 1].set_xlabel('Time Step')
        self.axes[0, 1].set_ylabel('Component Reward')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot recent reward distribution
        if self.reward_history:
            recent_rewards = list(self.reward_history)[-50:]  # Last 50 rewards
            recent_components = {}
            
            for component in all_components:
                recent_components[component] = np.mean([
                    r.components.get(component, 0.0) for r in recent_rewards
                ])
            
            # Filter out zero components
            recent_components = {k: v for k, v in recent_components.items() if abs(v) > 1e-6}
            
            if recent_components:
                colors = [self.component_colors.get(comp, 'gray') 
                         for comp in recent_components.keys()]
                self.axes[1, 0].bar(recent_components.keys(), 
                                  recent_components.values(), color=colors)
                self.axes[1, 0].set_title('Average Component Rewards (Recent)')
                self.axes[1, 0].set_ylabel('Average Reward')
                plt.setp(self.axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot reward statistics
        if len(self.reward_history) > 10:
            stats_text = f"Total Episodes: {len(self.reward_history)}\n"
            stats_text += f"Mean Total Reward: {np.mean(total_rewards):.3f}\n"
            stats_text += f"Std Total Reward: {np.std(total_rewards):.3f}\n"
            stats_text += f"Max Total Reward: {np.max(total_rewards):.3f}\n"
            stats_text += f"Min Total Reward: {np.min(total_rewards):.3f}\n\n"
            
            # Component statistics
            for component in all_components:
                component_values = [r.components.get(component, 0.0) 
                                  for r in self.reward_history]
                stats_text += f"{component}:\n"
                stats_text += f"  Mean: {np.mean(component_values):.3f}\n"
                stats_text += f"  Std: {np.std(component_values):.3f}\n"
            
            self.axes[1, 1].text(0.05, 0.95, stats_text,
                               transform=self.axes[1, 1].transAxes,
                               verticalalignment='top',
                               fontsize=9,
                               family='monospace')
            self.axes[1, 1].set_title('Reward Statistics')
            self.axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def show(self):
        """Show the visualization window."""
        plt.show(block=False)
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)


class PerformanceMonitoringDashboard:
    """Performance monitoring dashboard for training metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Performance Monitoring Dashboard')
        
        # Performance metrics
        self.fps_history = deque(maxlen=max_history)
        self.detection_time_history = deque(maxlen=max_history)
        self.action_time_history = deque(maxlen=max_history)
        self.memory_usage_history = deque(maxlen=max_history)
        
        self.start_time = time.time()
        
    def add_metrics(self, fps: float, detection_time: float, 
                   action_time: float, memory_usage: float):
        """Add new performance metrics."""
        timestamp = time.time() - self.start_time
        
        self.fps_history.append((timestamp, fps))
        self.detection_time_history.append((timestamp, detection_time))
        self.action_time_history.append((timestamp, action_time))
        self.memory_usage_history.append((timestamp, memory_usage))
        
        self._update_dashboard()
        
    def _update_dashboard(self):
        """Update the performance dashboard."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # FPS over time
        if self.fps_history:
            times, fps_values = zip(*self.fps_history)
            self.axes[0, 0].plot(times, fps_values, 'b-', linewidth=2)
            self.axes[0, 0].axhline(y=10, color='r', linestyle='--', 
                                  label='Min Required (10 FPS)')
            self.axes[0, 0].set_title('Frames Per Second')
            self.axes[0, 0].set_ylabel('FPS')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Detection time
        if self.detection_time_history:
            times, det_times = zip(*self.detection_time_history)
            self.axes[0, 1].plot(times, det_times, 'g-', linewidth=2)
            self.axes[0, 1].axhline(y=50, color='r', linestyle='--',
                                  label='Max Target (50ms)')
            self.axes[0, 1].set_title('Detection Processing Time')
            self.axes[0, 1].set_ylabel('Time (ms)')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Action processing time
        if self.action_time_history:
            times, action_times = zip(*self.action_time_history)
            self.axes[0, 2].plot(times, action_times, 'orange', linewidth=2)
            self.axes[0, 2].axhline(y=10, color='r', linestyle='--',
                                  label='Max Target (10ms)')
            self.axes[0, 2].set_title('Action Processing Time')
            self.axes[0, 2].set_ylabel('Time (ms)')
            self.axes[0, 2].legend()
            self.axes[0, 2].grid(True, alpha=0.3)
        
        # Memory usage
        if self.memory_usage_history:
            times, memory_values = zip(*self.memory_usage_history)
            self.axes[1, 0].plot(times, memory_values, 'purple', linewidth=2)
            self.axes[1, 0].axhline(y=2048, color='r', linestyle='--',
                                  label='Max Target (2GB)')
            self.axes[1, 0].set_title('GPU Memory Usage')
            self.axes[1, 0].set_ylabel('Memory (MB)')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Performance statistics
        if len(self.fps_history) > 10:
            fps_values = [fps for _, fps in self.fps_history]
            det_times = [dt for _, dt in self.detection_time_history]
            action_times = [at for _, at in self.action_time_history]
            memory_values = [mem for _, mem in self.memory_usage_history]
            
            stats_text = "Performance Statistics:\n\n"
            stats_text += f"FPS:\n"
            stats_text += f"  Current: {fps_values[-1]:.1f}\n"
            stats_text += f"  Mean: {np.mean(fps_values):.1f}\n"
            stats_text += f"  Min: {np.min(fps_values):.1f}\n\n"
            
            stats_text += f"Detection Time (ms):\n"
            stats_text += f"  Current: {det_times[-1]:.1f}\n"
            stats_text += f"  Mean: {np.mean(det_times):.1f}\n"
            stats_text += f"  Max: {np.max(det_times):.1f}\n\n"
            
            stats_text += f"Action Time (ms):\n"
            stats_text += f"  Current: {action_times[-1]:.1f}\n"
            stats_text += f"  Mean: {np.mean(action_times):.1f}\n"
            stats_text += f"  Max: {np.max(action_times):.1f}\n\n"
            
            stats_text += f"Memory (MB):\n"
            stats_text += f"  Current: {memory_values[-1]:.0f}\n"
            stats_text += f"  Mean: {np.mean(memory_values):.0f}\n"
            stats_text += f"  Max: {np.max(memory_values):.0f}\n"
            
            self.axes[1, 1].text(0.05, 0.95, stats_text,
                               transform=self.axes[1, 1].transAxes,
                               verticalalignment='top',
                               fontsize=10,
                               family='monospace')
            self.axes[1, 1].set_title('Current Statistics')
            self.axes[1, 1].axis('off')
        
        # System health indicators
        health_indicators = []
        if self.fps_history:
            current_fps = self.fps_history[-1][1]
            health_indicators.append(("FPS", current_fps >= 10, f"{current_fps:.1f}"))
        
        if self.detection_time_history:
            current_det_time = self.detection_time_history[-1][1]
            health_indicators.append(("Detection", current_det_time <= 50, f"{current_det_time:.1f}ms"))
        
        if self.action_time_history:
            current_action_time = self.action_time_history[-1][1]
            health_indicators.append(("Action", current_action_time <= 10, f"{current_action_time:.1f}ms"))
        
        if self.memory_usage_history:
            current_memory = self.memory_usage_history[-1][1]
            health_indicators.append(("Memory", current_memory <= 2048, f"{current_memory:.0f}MB"))
        
        if health_indicators:
            y_pos = 0.9
            self.axes[1, 2].text(0.05, 0.95, "System Health:",
                               transform=self.axes[1, 2].transAxes,
                               fontsize=12, fontweight='bold')
            
            for name, healthy, value in health_indicators:
                color = 'green' if healthy else 'red'
                status = '✓' if healthy else '✗'
                self.axes[1, 2].text(0.05, y_pos, f"{status} {name}: {value}",
                                   transform=self.axes[1, 2].transAxes,
                                   color=color, fontsize=11)
                y_pos -= 0.15
            
            self.axes[1, 2].set_title('Health Status')
            self.axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def show(self):
        """Show the dashboard."""
        plt.show(block=False)
    
    def close(self):
        """Close the dashboard."""
        plt.close(self.fig)