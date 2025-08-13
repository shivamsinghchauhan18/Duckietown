#!/usr/bin/env python3
"""
Complete Debugging and Visualization Integration Example

This example demonstrates how to integrate all debugging and visualization tools
into a complete enhanced Duckietown RL training session with comprehensive
monitoring, profiling, and analysis capabilities.
"""

import numpy as np
import time
import gym
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.visualization_manager import (
    create_visualization_manager,
    VisualizationConfig
)
from duckietown_utils.enhanced_logger import EnhancedLogger
from duckietown_utils.logging_context import LoggingContext
from duckietown_utils.debug_utils import create_debug_session


class MockEnhancedDuckietownEnv:
    """Mock enhanced Duckietown environment for demonstration."""
    
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self.step_count = 0
        self.episode_count = 0
        
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        self.episode_count += 1
        
        # Generate mock observation
        obs = {
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'speed': 0.0,
            'steering': 0.0
        }
        
        return obs
    
    def step(self, action):
        """Step environment."""
        self.step_count += 1
        
        # Simulate processing times
        detection_start = time.time()
        detections = self._simulate_detections()
        detection_time = (time.time() - detection_start) * 1000
        
        action_start = time.time()
        action_info = self._simulate_action_processing(action)
        action_time = (time.time() - action_start) * 1000
        
        # Generate observation
        obs = {
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'speed': float(action[1]),
            'steering': float(action[0])
        }
        
        # Calculate reward components
        reward_components = self._calculate_reward_components(action, detections)
        total_reward = sum(reward_components.values())
        
        # Determine if episode is done
        done = self.step_count >= 100 or total_reward < -2.0
        
        # Create info dict
        info = {
            'detections': detections,
            'action_type': action_info['type'],
            'action_reasoning': action_info['reasoning'],
            'safety_critical': action_info['safety_critical'],
            'reward_components': reward_components,
            'fps': 1.0 / max(0.001, (detection_time + action_time) / 1000),
            'detection_time': detection_time,
            'action_time': action_time,
            'memory_usage': np.random.uniform(800, 1500),  # Mock memory usage
            'episode': self.episode_count,
            'step': self.step_count
        }
        
        return obs, total_reward, done, info
    
    def _simulate_detections(self) -> List[Dict[str, Any]]:
        """Simulate YOLO detections."""
        detections = []
        num_detections = np.random.randint(0, 4)
        
        classes = ['duckiebot', 'duckie', 'cone', 'truck', 'bus']
        
        for _ in range(num_detections):
            x1, y1 = np.random.randint(0, 500), np.random.randint(0, 400)
            x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(50, 100)
            
            detection = {
                'class': np.random.choice(classes),
                'confidence': np.random.uniform(0.3, 0.95),
                'bbox': [x1, y1, x2, y2],
                'distance': np.random.uniform(0.5, 3.0),
                'relative_position': [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            }
            detections.append(detection)
        
        return detections
    
    def _simulate_action_processing(self, action) -> Dict[str, Any]:
        """Simulate action processing and reasoning."""
        steering, throttle = action
        
        # Determine action type based on action values
        if abs(steering) > 0.5:
            if steering > 0:
                action_type = "turning_right"
                reasoning = f"Turning right with steering {steering:.2f}"
            else:
                action_type = "turning_left"
                reasoning = f"Turning left with steering {steering:.2f}"
        elif throttle < 0.1:
            action_type = "stopping"
            reasoning = "Stopping or very slow movement"
        elif abs(steering) < 0.1:
            action_type = "lane_following"
            reasoning = "Following lane center"
        else:
            action_type = "adjusting"
            reasoning = f"Minor steering adjustment: {steering:.2f}"
        
        # Determine if safety critical
        safety_critical = (
            throttle > 0.8 or  # High speed
            abs(steering) > 0.7 or  # Sharp turn
            np.random.random() < 0.05  # Random safety events
        )
        
        if safety_critical:
            reasoning += " [SAFETY CRITICAL]"
        
        return {
            'type': action_type,
            'reasoning': reasoning,
            'safety_critical': safety_critical
        }
    
    def _calculate_reward_components(self, action, detections) -> Dict[str, float]:
        """Calculate reward components."""
        steering, throttle = action
        
        # Lane following reward (higher for straight driving)
        lane_following = 0.5 * (1.0 - abs(steering))
        
        # Object avoidance reward (penalty for close objects)
        object_avoidance = 0.0
        for detection in detections:
            if detection['distance'] < 1.0:
                object_avoidance -= 0.3 * (1.0 - detection['distance'])
        
        # Efficiency reward (encourage moderate speed)
        efficiency = 0.2 * throttle * (1.0 - abs(steering))
        
        # Safety penalty (for aggressive actions)
        safety_penalty = 0.0
        if abs(steering) > 0.7:
            safety_penalty -= 0.5
        if throttle > 0.8:
            safety_penalty -= 0.3
        
        # Lane changing reward (small bonus for controlled lane changes)
        lane_changing = 0.0
        if 0.3 < abs(steering) < 0.6 and throttle > 0.3:
            lane_changing = 0.1
        
        return {
            'lane_following': lane_following,
            'object_avoidance': object_avoidance,
            'efficiency': efficiency,
            'safety_penalty': safety_penalty,
            'lane_changing': lane_changing
        }


class MockAgent:
    """Mock RL agent for demonstration."""
    
    def __init__(self):
        self.step_count = 0
        
    def act(self, observation):
        """Select action based on observation."""
        self.step_count += 1
        
        # Simple policy: mostly go straight with occasional turns
        if self.step_count % 20 == 0:
            # Occasional turn
            steering = np.random.uniform(-0.6, 0.6)
        else:
            # Mostly straight
            steering = np.random.uniform(-0.2, 0.2)
        
        # Moderate throttle with some variation
        throttle = np.random.uniform(0.3, 0.7)
        
        return np.array([steering, throttle], dtype=np.float32)


def run_enhanced_training_with_debugging(num_episodes: int = 5, 
                                       log_directory: str = "logs/complete_debugging_demo"):
    """Run enhanced training session with comprehensive debugging and visualization."""
    
    print("=== Enhanced Duckietown RL Training with Complete Debugging ===")
    print(f"Episodes: {num_episodes}")
    print(f"Log directory: {log_directory}")
    print()
    
    # Setup logging
    log_path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure enhanced logger
    logger = EnhancedLogger(
        name="enhanced_training",
        log_directory=str(log_path),
        enable_structured_logging=True,
        enable_performance_logging=True,
        log_level=logging.INFO
    )
    
    # Create visualization manager
    viz_config = VisualizationConfig(
        enable_detection_viz=True,
        enable_action_viz=True,
        enable_reward_viz=True,
        enable_performance_viz=True,
        enable_profiling=True,
        detection_confidence_threshold=0.4,
        max_action_history=200,
        max_reward_history=1000
    )
    
    viz_manager = create_visualization_manager(enable_all=True)
    viz_manager.config = viz_config
    viz_manager.start()
    
    # Initialize environment and agent
    env = MockEnhancedDuckietownEnv()
    agent = MockAgent()
    
    try:
        total_steps = 0
        episode_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            with LoggingContext(logger, episode=episode):
                # Profile episode initialization
                with viz_manager.profile_section("episode_initialization"):
                    obs = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    done = False
                
                logger.info(f"Starting episode {episode + 1}")
                
                while not done:
                    total_steps += 1
                    episode_steps += 1
                    
                    # Profile action selection
                    with viz_manager.profile_section("action_selection"):
                        action = agent.act(obs)
                    
                    # Profile environment step
                    with viz_manager.profile_section("environment_step"):
                        obs, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    
                    # Update visualizations
                    with viz_manager.profile_section("visualization_update"):
                        # Detection visualization
                        viz_manager.update_detections(obs['image'], info['detections'])
                        
                        # Action visualization
                        viz_manager.update_action(
                            info['action_type'],
                            action,
                            info['action_reasoning'],
                            info['safety_critical']
                        )
                        
                        # Reward visualization
                        viz_manager.update_reward(info['reward_components'], reward)
                        
                        # Performance visualization
                        viz_manager.update_performance(
                            info['fps'],
                            info['detection_time'],
                            info['action_time'],
                            info['memory_usage']
                        )
                    
                    # Log structured data
                    with viz_manager.profile_section("logging"):
                        logger.log_detection_data(
                            detections=info['detections'],
                            processing_time=info['detection_time']
                        )
                        
                        logger.log_action_data(
                            action_type=info['action_type'],
                            action_values=action.tolist(),
                            reasoning=info['action_reasoning'],
                            safety_critical=info['safety_critical']
                        )
                        
                        logger.log_reward_data(
                            reward_components=info['reward_components'],
                            total_reward=reward,
                            episode=episode + 1
                        )
                        
                        logger.log_performance_data(
                            fps=info['fps'],
                            detection_time=info['detection_time'],
                            action_time=info['action_time'],
                            memory_usage=info['memory_usage']
                        )
                    
                    # Print progress
                    if episode_steps % 20 == 0:
                        print(f"  Step {episode_steps}: reward={reward:.3f}, "
                              f"total={episode_reward:.3f}, fps={info['fps']:.1f}")
                    
                    # Small delay to make visualization visible
                    time.sleep(0.05)
                
                episode_rewards.append(episode_reward)
                logger.info(f"Episode {episode + 1} completed: "
                           f"steps={episode_steps}, reward={episode_reward:.3f}")
                
                print(f"  Episode {episode + 1} completed: "
                      f"steps={episode_steps}, reward={episode_reward:.3f}")
        
        # Training summary
        print(f"\n=== Training Summary ===")
        print(f"Total episodes: {num_episodes}")
        print(f"Total steps: {total_steps}")
        print(f"Average episode reward: {np.mean(episode_rewards):.3f}")
        print(f"Best episode reward: {np.max(episode_rewards):.3f}")
        print(f"Worst episode reward: {np.min(episode_rewards):.3f}")
        
        # Get profiling results
        profiling_stats = viz_manager.get_profiling_stats()
        if profiling_stats:
            print(f"\n=== Profiling Results ===")
            for section, stats in profiling_stats.items():
                print(f"  {section}: {stats['mean']:.2f}ms avg "
                      f"({stats['count']} calls, {stats['total']:.1f}ms total)")
        
        # Save visualization state
        viz_state_path = log_path / "visualization_state"
        viz_manager.save_current_state(str(viz_state_path))
        print(f"\nVisualization state saved to: {viz_state_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Stop visualization
        print("\nStopping visualization...")
        viz_manager.stop()
        
        # Close logger
        logger.close()
        
        print(f"Logs saved to: {log_path}")
        
        return log_path


def analyze_training_session(log_directory: str):
    """Analyze the completed training session."""
    print(f"\n=== Analyzing Training Session ===")
    print(f"Log directory: {log_directory}")
    
    try:
        # Create debug session
        analyzer, report = create_debug_session(log_directory)
        
        print("\n=== Debug Report ===")
        print(report)
        
        # Check for analysis plots
        plots_dir = Path(log_directory) / "analysis_plots"
        if plots_dir.exists():
            print(f"\nAnalysis plots generated:")
            for plot_file in plots_dir.glob("*.png"):
                print(f"  - {plot_file.name}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing training session: {e}")
        return False


def main():
    """Main function to run the complete debugging integration example."""
    print("Enhanced Duckietown RL - Complete Debugging Integration Example")
    print("=" * 70)
    
    try:
        # Run training with debugging
        log_directory = run_enhanced_training_with_debugging(
            num_episodes=3,
            log_directory="logs/complete_debugging_demo"
        )
        
        # Wait a moment for files to be written
        time.sleep(1)
        
        # Analyze the training session
        analyze_training_session(str(log_directory))
        
        print(f"\n=== Example Completed Successfully ===")
        print(f"Check the following for results:")
        print(f"  - Logs: {log_directory}")
        print(f"  - Analysis plots: {log_directory}/analysis_plots/")
        print(f"  - Visualization state: {log_directory}/visualization_state/")
        
        print(f"\nTo run analysis again:")
        print(f"  python debug_enhanced_rl.py analyze {log_directory}")
        print(f"  python debug_enhanced_rl.py report {log_directory}")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()