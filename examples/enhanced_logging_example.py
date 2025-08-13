#!/usr/bin/env python3
"""
Enhanced Logging System Example

This example demonstrates how to use the comprehensive logging system
for the enhanced Duckietown RL environment.
"""

import numpy as np
import time
from pathlib import Path

from duckietown_utils.enhanced_logger import initialize_logger, get_logger
from duckietown_utils.logging_context import (
    log_detection_timing, log_action_timing, log_reward_timing,
    LoggingMixin, get_performance_tracker
)


class ExampleWrapper(LoggingMixin):
    """Example wrapper demonstrating logging integration."""
    
    def __init__(self, log_dir="logs/example"):
        super().__init__()
        
        # Initialize logger with custom configuration
        self.logger = initialize_logger(
            log_dir=log_dir,
            log_level="INFO",
            log_detections=True,
            log_actions=True,
            log_rewards=True,
            log_performance=True,
            console_output=True,
            file_output=True
        )
        
        self._log_wrapper_initialization({
            'wrapper_type': 'ExampleWrapper',
            'log_dir': log_dir,
            'features': ['detection', 'action', 'reward', 'performance']
        })
        
        self.performance_tracker = get_performance_tracker()
    
    def simulate_object_detection(self, frame_id: int):
        """Simulate object detection with logging."""
        with log_detection_timing(frame_id, self.logger) as log_detections:
            # Simulate YOLO detection processing
            time.sleep(0.02)  # Simulate 20ms processing time
            
            # Create mock detection results
            detections = [
                {
                    'class': 'duckiebot',
                    'confidence': 0.85,
                    'bbox': [100, 50, 200, 150],
                    'distance': 1.2,
                    'relative_position': [0.5, 0.0]
                },
                {
                    'class': 'obstacle',
                    'confidence': 0.72,
                    'bbox': [250, 80, 320, 180],
                    'distance': 0.8,
                    'relative_position': [-0.3, 0.1]
                }
            ]
            
            # Log the detection results
            processing_time = log_detections(detections, confidence_threshold=0.5)
            
            return detections, processing_time
    
    def simulate_action_processing(self, frame_id: int, detections):
        """Simulate action processing with logging."""
        with log_action_timing(frame_id, "ExampleWrapper", self.logger) as log_action:
            # Simulate action processing
            time.sleep(0.005)  # Simulate 5ms processing time
            
            original_action = np.array([0.5, 0.0])
            
            # Determine action modification based on detections
            if any(det['distance'] < 1.0 for det in detections):
                # Object avoidance needed
                modified_action = np.array([0.3, 0.2])
                action_type = 'object_avoidance'
                reasoning = f"Avoiding {len(detections)} detected objects"
                triggering_conditions = {
                    'closest_object_distance': min(det['distance'] for det in detections),
                    'object_count': len(detections)
                }
                safety_checks = {
                    'clearance_check': True,
                    'collision_check': True,
                    'speed_limit_check': True
                }
            else:
                # Normal lane following
                modified_action = original_action
                action_type = 'lane_following'
                reasoning = "No obstacles detected, continuing lane following"
                triggering_conditions = {'clear_path': True}
                safety_checks = {'lane_position_check': True}
            
            # Log the action decision
            processing_time = log_action(
                original_action=original_action,
                modified_action=modified_action,
                action_type=action_type,
                reasoning=reasoning,
                triggering_conditions=triggering_conditions,
                safety_checks=safety_checks
            )
            
            return modified_action, processing_time
    
    def simulate_reward_calculation(self, frame_id: int, episode_step: int, action, detections):
        """Simulate reward calculation with logging."""
        with log_reward_timing(frame_id, episode_step, self.logger) as log_reward:
            # Simulate reward processing
            time.sleep(0.002)  # Simulate 2ms processing time
            
            # Calculate reward components
            lane_following_reward = 0.8 if np.linalg.norm(action) < 0.6 else 0.4
            
            # Object avoidance reward
            min_distance = min((det['distance'] for det in detections), default=float('inf'))
            if min_distance < 0.5:
                object_avoidance_reward = -0.5  # Penalty for being too close
            elif min_distance < 1.0:
                object_avoidance_reward = 0.2   # Reward for maintaining safe distance
            else:
                object_avoidance_reward = 0.0   # Neutral when no objects nearby
            
            # Efficiency reward
            efficiency_reward = 0.1 if action[0] > 0.3 else 0.0
            
            # Safety penalty
            safety_penalty = -1.0 if min_distance < 0.3 else 0.0
            
            reward_components = {
                'lane_following': lane_following_reward,
                'object_avoidance': object_avoidance_reward,
                'efficiency': efficiency_reward,
                'safety_penalty': safety_penalty
            }
            
            reward_weights = {
                'lane_following': 1.0,
                'object_avoidance': 0.8,
                'efficiency': 0.3,
                'safety_penalty': 2.0
            }
            
            # Calculate total reward
            total_reward = sum(
                reward_components[component] * reward_weights[component]
                for component in reward_components
            )
            
            cumulative_reward = episode_step * 0.5  # Mock cumulative reward
            
            # Log the reward calculation
            processing_time = log_reward(
                total_reward=total_reward,
                reward_components=reward_components,
                reward_weights=reward_weights,
                cumulative_reward=cumulative_reward
            )
            
            return total_reward, processing_time
    
    def simulate_episode(self, num_steps: int = 50):
        """Simulate a complete episode with comprehensive logging."""
        print(f"Starting episode simulation with {num_steps} steps...")
        
        episode_start_time = time.time()
        
        for step in range(num_steps):
            frame_id = self.performance_tracker.increment_frame()
            
            # Simulate object detection
            detections, detection_time = self.simulate_object_detection(frame_id)
            
            # Simulate action processing
            action, action_time = self.simulate_action_processing(frame_id, detections)
            
            # Simulate reward calculation
            reward, reward_time = self.simulate_reward_calculation(frame_id, step + 1, action, detections)
            
            # Log performance metrics
            self.logger.log_performance_metrics(
                frame_id=frame_id,
                detection_time_ms=detection_time,
                action_processing_time_ms=action_time,
                reward_calculation_time_ms=reward_time
            )
            
            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} completed")
        
        episode_duration = time.time() - episode_start_time
        
        print(f"Episode completed in {episode_duration:.2f} seconds")
        print(f"Average FPS: {num_steps / episode_duration:.2f}")
        
        # Print logging summary
        summary = self.logger.get_log_summary()
        print("\nLogging Summary:")
        print(f"  Log directory: {summary['log_dir']}")
        print(f"  Features enabled: {summary['features_enabled']}")
        print(f"  Current FPS: {summary['current_fps']:.2f}")
        print(f"  Total frames processed: {summary['total_frames_processed']}")


def demonstrate_error_logging():
    """Demonstrate error and warning logging."""
    logger = get_logger()
    
    print("\nDemonstrating error and warning logging...")
    
    # Log a warning
    logger.log_warning("This is a test warning", 
                      component="example", 
                      severity="low")
    
    # Log an error with exception
    try:
        raise ValueError("This is a test error for demonstration")
    except ValueError as e:
        logger.log_error("Demonstration error occurred", 
                        exception=e,
                        component="example",
                        operation="demonstration")
    
    print("Error and warning logging completed")


def demonstrate_log_analysis():
    """Demonstrate how to analyze log files."""
    logger = get_logger()
    log_dir = Path(logger.log_dir)
    
    print(f"\nAnalyzing log files in: {log_dir}")
    
    # List all log files
    log_files = {
        'detections': list(log_dir.glob("detections_*.jsonl")),
        'actions': list(log_dir.glob("actions_*.jsonl")),
        'rewards': list(log_dir.glob("rewards_*.jsonl")),
        'performance': list(log_dir.glob("performance_*.jsonl")),
        'main': list(log_dir.glob("enhanced_rl_*.log"))
    }
    
    for log_type, files in log_files.items():
        if files:
            print(f"  {log_type.capitalize()} logs: {len(files)} file(s)")
            
            if log_type != 'main':  # JSONL files
                with open(files[0], 'r') as f:
                    lines = f.readlines()
                    print(f"    - {len(lines)} entries in {files[0].name}")
                    
                    if lines:
                        # Show first entry as example
                        import json
                        first_entry = json.loads(lines[0].strip())
                        print(f"    - First entry keys: {list(first_entry.keys())}")
        else:
            print(f"  {log_type.capitalize()} logs: No files found")


def main():
    """Main demonstration function."""
    print("Enhanced Duckietown RL Logging System Demonstration")
    print("=" * 60)
    
    # Create example wrapper with logging
    wrapper = ExampleWrapper(log_dir="logs/enhanced_logging_demo")
    
    # Simulate an episode
    wrapper.simulate_episode(num_steps=30)
    
    # Demonstrate error logging
    demonstrate_error_logging()
    
    # Analyze generated logs
    demonstrate_log_analysis()
    
    print("\nDemonstration completed!")
    print("Check the 'logs/enhanced_logging_demo' directory for generated log files.")


if __name__ == "__main__":
    main()