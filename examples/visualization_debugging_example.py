#!/usr/bin/env python3
"""
Example script demonstrating the debugging and visualization tools.

This script shows how to use the visualization manager and debug utilities
for monitoring and debugging the enhanced Duckietown RL system.
"""

import numpy as np
import time
import random
from pathlib import Path
import sys

# Add the parent directory to the path to import duckietown_utils
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.visualization_manager import (
    VisualizationManager, 
    VisualizationConfig,
    create_visualization_manager
)
from duckietown_utils.debug_utils import LogAnalyzer, create_debug_session


def simulate_detection_data():
    """Simulate object detection data."""
    # Create a dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulate detections
    detections = []
    num_detections = random.randint(0, 3)
    
    classes = ['duckiebot', 'duckie', 'cone', 'truck']
    
    for _ in range(num_detections):
        x1, y1 = random.randint(0, 500), random.randint(0, 400)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 100)
        
        detection = {
            'class': random.choice(classes),
            'confidence': random.uniform(0.3, 0.95),
            'bbox': [x1, y1, x2, y2],
            'distance': random.uniform(0.5, 3.0),
            'relative_position': [random.uniform(-1, 1), random.uniform(-1, 1)]
        }
        detections.append(detection)
    
    return image, detections


def simulate_action_data():
    """Simulate action decision data."""
    action_types = ['lane_following', 'object_avoidance', 'lane_changing', 'emergency_stop']
    action_type = random.choice(action_types)
    
    # Simulate action values (steering, throttle)
    action_values = np.array([
        random.uniform(-1, 1),  # steering
        random.uniform(0, 1)    # throttle
    ])
    
    # Generate reasoning based on action type
    reasoning_templates = {
        'lane_following': "Following lane center, no obstacles detected",
        'object_avoidance': f"Avoiding {random.choice(['duckiebot', 'cone', 'duckie'])} at {random.uniform(0.5, 2.0):.1f}m distance",
        'lane_changing': f"Changing to {'left' if random.random() > 0.5 else 'right'} lane due to obstacle",
        'emergency_stop': "Emergency stop triggered due to safety violation"
    }
    
    reasoning = reasoning_templates[action_type]
    safety_critical = action_type in ['emergency_stop'] or random.random() < 0.1
    
    return action_type, action_values, reasoning, safety_critical


def simulate_reward_data():
    """Simulate reward component data."""
    components = {
        'lane_following': random.uniform(-0.1, 0.8),
        'object_avoidance': random.uniform(-0.2, 0.5),
        'lane_changing': random.uniform(-0.1, 0.3),
        'efficiency': random.uniform(-0.1, 0.2),
        'safety_penalty': random.uniform(-1.0, 0.0)
    }
    
    total_reward = sum(components.values())
    return components, total_reward


def simulate_performance_data():
    """Simulate performance metrics."""
    # Simulate realistic performance with occasional drops
    base_fps = 15.0
    fps = base_fps + random.uniform(-3, 3)
    if random.random() < 0.05:  # Occasional performance drops
        fps *= 0.5
    
    detection_time = random.uniform(20, 80)
    if random.random() < 0.1:  # Occasional spikes
        detection_time *= 2
    
    action_time = random.uniform(2, 15)
    memory_usage = random.uniform(800, 2200)
    
    return max(fps, 1), detection_time, action_time, memory_usage


def demo_real_time_visualization():
    """Demonstrate real-time visualization capabilities."""
    print("=== Real-Time Visualization Demo ===")
    print("This demo shows real-time visualization of detection, action, reward, and performance data.")
    print("Close the visualization windows to stop the demo.\n")
    
    # Create visualization manager
    config = VisualizationConfig(
        enable_detection_viz=True,
        enable_action_viz=True,
        enable_reward_viz=True,
        enable_performance_viz=True,
        enable_profiling=True
    )
    
    viz_manager = VisualizationManager(config)
    viz_manager.start()
    
    try:
        # Simulate data for 60 seconds
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < 60:
            step += 1
            
            # Profile the simulation step
            with viz_manager.profile_section("simulation_step"):
                # Simulate detection data
                with viz_manager.profile_section("detection_simulation"):
                    image, detections = simulate_detection_data()
                    viz_manager.update_detections(image, detections)
                
                # Simulate action data (less frequent)
                if step % 3 == 0:
                    with viz_manager.profile_section("action_simulation"):
                        action_type, action_values, reasoning, safety_critical = simulate_action_data()
                        viz_manager.update_action(action_type, action_values, reasoning, safety_critical)
                
                # Simulate reward data (less frequent)
                if step % 5 == 0:
                    with viz_manager.profile_section("reward_simulation"):
                        components, total_reward = simulate_reward_data()
                        viz_manager.update_reward(components, total_reward)
                
                # Simulate performance data
                with viz_manager.profile_section("performance_simulation"):
                    fps, detection_time, action_time, memory_usage = simulate_performance_data()
                    viz_manager.update_performance(fps, detection_time, action_time, memory_usage)
            
            # Control simulation speed
            time.sleep(0.1)
            
            # Print progress every 10 seconds
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Demo running... {elapsed:.1f}s elapsed")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        print("\nStopping visualization...")
        viz_manager.stop()
        
        # Show profiling results
        print("\nProfiling Results:")
        stats = viz_manager.get_profiling_stats()
        for section, stat in stats.items():
            print(f"  {section}: {stat['mean']:.2f}ms avg ({stat['count']} calls)")


def demo_log_analysis():
    """Demonstrate log analysis capabilities."""
    print("\n=== Log Analysis Demo ===")
    
    # Check if log directory exists
    log_dir = Path("logs/enhanced_logging_demo")
    if not log_dir.exists():
        print(f"Log directory {log_dir} not found.")
        print("Run the enhanced logging example first to generate logs.")
        return
    
    print(f"Analyzing logs in: {log_dir}")
    
    # Create debug session
    analyzer, report = create_debug_session(str(log_dir))
    
    print("\n=== Debug Report ===")
    print(report)
    
    # Show analysis plots location
    plots_dir = log_dir / "analysis_plots"
    if plots_dir.exists():
        print(f"\nAnalysis plots saved to: {plots_dir}")
        plot_files = list(plots_dir.glob("*.png"))
        for plot_file in plot_files:
            print(f"  - {plot_file.name}")


def demo_custom_visualization():
    """Demonstrate custom visualization setup."""
    print("\n=== Custom Visualization Demo ===")
    
    # Create custom configuration
    config = VisualizationConfig(
        enable_detection_viz=True,
        enable_action_viz=False,  # Disable action viz
        enable_reward_viz=True,
        enable_performance_viz=False,  # Disable performance viz
        enable_profiling=True,
        detection_confidence_threshold=0.7,  # Higher threshold
        max_reward_history=500  # Smaller history
    )
    
    viz_manager = VisualizationManager(config)
    viz_manager.start()
    
    print("Running custom visualization for 20 seconds...")
    print("Only detection and reward visualizations are enabled.")
    
    try:
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < 20:
            step += 1
            
            # Only update enabled visualizations
            image, detections = simulate_detection_data()
            viz_manager.update_detections(image, detections)
            
            if step % 5 == 0:
                components, total_reward = simulate_reward_data()
                viz_manager.update_reward(components, total_reward)
            
            time.sleep(0.2)
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        viz_manager.stop()
        print("Custom visualization demo completed.")


def demo_save_state():
    """Demonstrate saving visualization state."""
    print("\n=== Save State Demo ===")
    
    viz_manager = create_visualization_manager(enable_all=True)
    viz_manager.start()
    
    # Generate some data
    for i in range(50):
        with viz_manager.profile_section("demo_data_generation"):
            image, detections = simulate_detection_data()
            viz_manager.update_detections(image, detections)
            
            if i % 3 == 0:
                action_type, action_values, reasoning, safety_critical = simulate_action_data()
                viz_manager.update_action(action_type, action_values, reasoning, safety_critical)
            
            if i % 5 == 0:
                components, total_reward = simulate_reward_data()
                viz_manager.update_reward(components, total_reward)
        
        time.sleep(0.05)
    
    # Save current state
    output_path = "visualization_state_demo"
    viz_manager.save_current_state(output_path)
    
    viz_manager.stop()
    print(f"Visualization state saved to: {output_path}")


def main():
    """Main function to run all demos."""
    print("Enhanced Duckietown RL - Debugging and Visualization Tools Demo")
    print("=" * 60)
    
    demos = [
        ("Real-Time Visualization", demo_real_time_visualization),
        ("Log Analysis", demo_log_analysis),
        ("Custom Visualization", demo_custom_visualization),
        ("Save State", demo_save_state)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  0. Run all demos")
    
    try:
        choice = input("\nSelect demo to run (0-4): ").strip()
        
        if choice == "0":
            for name, demo_func in demos:
                print(f"\n{'='*20} {name} {'='*20}")
                demo_func()
                input("\nPress Enter to continue to next demo...")
        elif choice in ["1", "2", "3", "4"]:
            demo_idx = int(choice) - 1
            name, demo_func = demos[demo_idx]
            print(f"\n{'='*20} {name} {'='*20}")
            demo_func()
        else:
            print("Invalid choice. Exiting.")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user. Exiting.")
    except Exception as e:
        print(f"Error running demo: {e}")
    
    print("\nDemo completed. Thank you!")


if __name__ == "__main__":
    main()