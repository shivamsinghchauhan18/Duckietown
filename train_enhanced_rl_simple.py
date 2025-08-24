#!/usr/bin/env python3
"""
🚀 SIMPLIFIED ENHANCED RL TRAINING 🚀

Quick demonstration of the enhanced RL system without full dependencies.
This shows the core concepts and can run in 10-15 minutes.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedDuckietownEnv:
    """Simplified Duckietown environment for demonstration."""
    
    def __init__(self):
        self.observation_space_shape = (120, 160, 3)
        self.action_space_shape = (2,)  # [steering, throttle]
        
        self.current_step = 0
        self.max_steps = 500
        self.lane_center = 80  # Center of 160-width image
        self.robot_position = 80
        self.robot_speed = 0.3
        
        # Simulate some obstacles
        self.obstacles = [
            {'x': 60, 'y': 200, 'active': True},
            {'x': 100, 'y': 350, 'active': True},
            {'x': 40, 'y': 450, 'active': True}
        ]
        
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.robot_position = 80 + np.random.uniform(-10, 10)
        self.robot_speed = 0.3
        
        # Reset obstacles
        for obs in self.obstacles:
            obs['y'] = np.random.uniform(200, 500)
            obs['active'] = True
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment."""
        steering, throttle = action
        
        # Update robot state
        self.robot_position += steering * 10  # Steering effect
        self.robot_position = np.clip(self.robot_position, 10, 150)
        self.robot_speed = throttle * 0.5
        
        # Move obstacles closer
        for obs in self.obstacles:
            obs['y'] -= self.robot_speed * 50
            if obs['y'] < 0:
                obs['y'] = np.random.uniform(400, 600)
        
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if done
        done = self.current_step >= self.max_steps or self._check_collision()
        
        info = {
            'step': self.current_step,
            'robot_position': self.robot_position,
            'collision': self._check_collision(),
            'lane_deviation': abs(self.robot_position - self.lane_center) / 80
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation."""
        # Create a simple synthetic image
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        
        # Draw lane lines
        image[:, 20:25, :] = [255, 255, 255]  # Left lane
        image[:, 135:140, :] = [255, 255, 255]  # Right lane
        
        # Draw road
        image[:, 25:135, :] = [100, 100, 100]
        
        # Draw robot position indicator
        robot_x = int(self.robot_position)
        image[100:110, robot_x-5:robot_x+5, :] = [0, 255, 0]  # Green robot
        
        # Draw obstacles
        for obs in self.obstacles:
            if obs['active'] and 0 < obs['y'] < 120:
                obs_x = int(obs['x'])
                obs_y = int(obs['y'])
                if 0 <= obs_x < 160 and 0 <= obs_y < 120:
                    image[max(0, obs_y-5):min(120, obs_y+5), 
                          max(0, obs_x-5):min(160, obs_x+5), :] = [255, 0, 0]  # Red obstacle
        
        return image
    
    def _calculate_reward(self, action):
        """Calculate reward."""
        steering, throttle = action
        
        # Lane following reward
        lane_deviation = abs(self.robot_position - self.lane_center) / 80
        lane_reward = 1.0 - lane_deviation
        
        # Speed reward
        speed_reward = throttle * 0.5
        
        # Smoothness penalty
        smoothness_penalty = abs(steering) * 0.1
        
        # Obstacle avoidance reward
        obstacle_reward = 0.0
        min_distance = float('inf')
        
        for obs in self.obstacles:
            if obs['active']:
                distance = np.sqrt((self.robot_position - obs['x'])**2 + (100 - obs['y'])**2)
                min_distance = min(min_distance, distance)
        
        if min_distance < 20:
            obstacle_reward = -2.0  # Penalty for being too close
        elif min_distance < 40:
            obstacle_reward = -0.5
        else:
            obstacle_reward = 0.1  # Small reward for maintaining distance
        
        total_reward = lane_reward + speed_reward - smoothness_penalty + obstacle_reward
        
        return total_reward
    
    def _check_collision(self):
        """Check for collision."""
        for obs in self.obstacles:
            if obs['active']:
                distance = np.sqrt((self.robot_position - obs['x'])**2 + (100 - obs['y'])**2)
                if distance < 15:
                    return True
        
        # Check lane boundaries
        if self.robot_position < 25 or self.robot_position > 135:
            return True
        
        return False


class SimplifiedYOLODetector:
    """Simplified YOLO detector for demonstration."""
    
    def __init__(self):
        self.confidence_threshold = 0.5
    
    def detect_objects(self, image):
        """Detect objects in image."""
        detections = []
        
        # Simple red object detection (obstacles)
        red_pixels = np.where((image[:, :, 0] > 200) & (image[:, :, 1] < 50) & (image[:, :, 2] < 50))
        
        if len(red_pixels[0]) > 0:
            # Find bounding boxes
            y_coords = red_pixels[0]
            x_coords = red_pixels[1]
            
            if len(y_coords) > 0:
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                
                # Estimate distance (closer objects appear lower in image)
                distance = (120 - y_max) / 120 * 5.0  # 0-5 meters
                
                detection = {
                    'class': 'obstacle',
                    'confidence': 0.9,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'distance': distance,
                    'relative_position': [(x_min + x_max) / 2 - 80, distance]
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'detection_count': len(detections),
            'safety_critical': any(d['distance'] < 1.0 for d in detections)
        }


class EnhancedDQNNetwork(nn.Module):
    """Enhanced DQN network with multi-modal inputs."""
    
    def __init__(self):
        super().__init__()
        
        # Image encoder (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Detection encoder
        self.detection_encoder = nn.Sequential(
            nn.Linear(45, 64),  # 5 detections * 9 features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Safety encoder
        self.safety_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Calculate CNN output size after flattening
        # After conv layers: 120x160 -> 30x40 -> 15x20 -> 15x20 = 19200 features
        cnn_output_size = 64 * 15 * 20  # 19200
        
        # Fusion layer
        fusion_size = cnn_output_size + 32 + 8  # 19240
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Dueling DQN heads
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 actions: steering, throttle
        )
    
    def forward(self, obs):
        """Forward pass."""
        image = obs['image']
        detection_features = obs['detection_features']
        safety_features = obs['safety_features']
        
        # Process image
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        image_features = self.image_encoder(image)
        # Already flattened by nn.Flatten()
        
        # Process detection features
        detection_encoded = self.detection_encoder(detection_features)
        
        # Process safety features
        safety_encoded = self.safety_encoder(safety_features)
        
        # Fuse features
        fused = torch.cat([image_features, detection_encoded, safety_encoded], dim=1)
        hidden = self.fusion_layer(fused)
        
        # Dueling DQN
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class SimplifiedEnhancedAgent:
    """Simplified enhanced RL agent."""
    
    def __init__(self):
        # Setup device (use CPU for compatibility)
        # Note: MPS has some compatibility issues with adaptive pooling
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (most compatible)")
        
        # Create network
        self.q_network = EnhancedDQNNetwork().to(self.device)
        self.target_network = EnhancedDQNNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.replay_buffer = []
        self.batch_size = 32
        self.buffer_size = 10000
        
        # YOLO detector
        self.yolo_detector = SimplifiedYOLODetector()
        
        # Statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'avg_reward': 0,
            'success_rate': 0,
            'collision_rate': 0
        }
    
    def process_observation(self, raw_obs):
        """Process raw observation into enhanced format."""
        # Run YOLO detection
        detection_result = self.yolo_detector.detect_objects(raw_obs)
        
        # Create detection features (5 detections * 9 features = 45)
        detection_features = np.zeros(45, dtype=np.float32)
        
        for i, detection in enumerate(detection_result['detections'][:5]):
            start_idx = i * 9
            bbox = detection['bbox']
            rel_pos = detection['relative_position']
            
            detection_features[start_idx:start_idx+9] = [
                1.0,  # class_id (simplified)
                detection['confidence'],
                bbox[0], bbox[1], bbox[2], bbox[3],
                rel_pos[0], rel_pos[1], detection['distance']
            ]
        
        # Create safety features
        safety_features = np.array([
            len(detection_result['detections']) / 5.0,  # Normalized detection count
            1.0 if detection_result['safety_critical'] else 0.0,  # Safety flag
            0.05  # Mock inference time
        ], dtype=np.float32)
        
        return {
            'image': raw_obs.astype(np.float32) / 255.0,
            'detection_features': detection_features,
            'safety_features': safety_features
        }
    
    def select_action(self, obs, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            # Random action
            return np.random.uniform(-1, 1, 2)
        
        # Greedy action
        with torch.no_grad():
            obs_tensor = {
                'image': torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device),
                'detection_features': torch.FloatTensor(obs['detection_features']).unsqueeze(0).to(self.device),
                'safety_features': torch.FloatTensor(obs['safety_features']).unsqueeze(0).to(self.device)
            }
            
            q_values = self.q_network(obs_tensor)
            action = torch.tanh(q_values).cpu().numpy().squeeze()
            
            return action
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((obs, action, reward, next_obs, done))
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Prepare batch tensors
        obs_batch = {
            'image': torch.FloatTensor([t[0]['image'] for t in batch]).to(self.device),
            'detection_features': torch.FloatTensor([t[0]['detection_features'] for t in batch]).to(self.device),
            'safety_features': torch.FloatTensor([t[0]['safety_features'] for t in batch]).to(self.device)
        }
        
        next_obs_batch = {
            'image': torch.FloatTensor([t[3]['image'] for t in batch]).to(self.device),
            'detection_features': torch.FloatTensor([t[3]['detection_features'] for t in batch]).to(self.device),
            'safety_features': torch.FloatTensor([t[3]['safety_features'] for t in batch]).to(self.device)
        }
        
        action_batch = torch.FloatTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        done_batch = torch.BoolTensor([t[4] for t in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(obs_batch)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_batch)
            target_q = reward_batch.unsqueeze(1) + (0.99 * next_q_values * ~done_batch.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_simplified_enhanced_rl():
    """Train the simplified enhanced RL agent."""
    logger.info("🚀 Starting Simplified Enhanced RL Training")
    logger.info("=" * 60)
    
    # Create environment and agent
    env = SimplifiedDuckietownEnv()
    agent = SimplifiedEnhancedAgent()
    
    # Training parameters
    num_episodes = 200
    target_update_freq = 10
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_episodes = 0
    collision_episodes = 0
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs = env.reset()
        processed_obs = agent.process_observation(obs)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(processed_obs, training=True)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            next_processed_obs = agent.process_observation(next_obs)
            
            # Store transition
            agent.store_transition(processed_obs, action, reward, next_processed_obs, done)
            
            # Train agent
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
            
            # Update state
            processed_obs = next_processed_obs
            episode_reward += reward
            episode_length += 1
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if not info['collision']:
            success_episodes += 1
        else:
            collision_episodes += 1
        
        # Log progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            success_rate = success_episodes / (episode + 1)
            
            logger.info(f"Episode {episode:3d} | "
                       f"Reward: {episode_reward:6.2f} | "
                       f"Avg Reward: {avg_reward:6.2f} | "
                       f"Success Rate: {success_rate:.2%} | "
                       f"Epsilon: {agent.epsilon:.3f}")
    
    # Training completed
    training_time = time.time() - start_time
    final_success_rate = success_episodes / num_episodes
    
    logger.info("=" * 60)
    logger.info("🎉 Training Completed!")
    logger.info(f"⏱️  Training Time: {training_time/60:.1f} minutes")
    logger.info(f"📊 Episodes: {num_episodes}")
    logger.info(f"🏆 Success Rate: {final_success_rate:.1%}")
    logger.info(f"💥 Collision Rate: {collision_episodes/num_episodes:.1%}")
    logger.info(f"📈 Final Avg Reward: {np.mean(episode_rewards[-20:]):.2f}")
    
    # Save results
    results = {
        'training_time_minutes': training_time / 60,
        'num_episodes': num_episodes,
        'success_rate': final_success_rate,
        'collision_rate': collision_episodes / num_episodes,
        'final_avg_reward': float(np.mean(episode_rewards[-20:])),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'device_used': str(agent.device)
    }
    
    # Save model
    model_path = Path("models/simplified_enhanced_model.pth")
    model_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'training_results': results
    }, model_path)
    
    logger.info(f"💾 Model saved to: {model_path}")
    
    # Create performance plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    window_size = 20
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg)
        plt.title(f'Moving Average Reward (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
    
    plt.subplot(2, 2, 3)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 4)
    # Success rate over time
    success_over_time = []
    for i in range(1, len(episode_rewards) + 1):
        # Count successes in last 20 episodes (or all if less than 20)
        start_idx = max(0, i - 20)
        recent_episodes = episode_rewards[start_idx:i]
        # Assume success if reward > 0 (simplified)
        recent_successes = sum(1 for r in recent_episodes if r > 0)
        success_rate = recent_successes / len(recent_episodes)
        success_over_time.append(success_rate)
    
    plt.plot(success_over_time)
    plt.title('Success Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    plot_path = Path("logs/simplified_training_results.png")
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"📊 Performance plot saved to: {plot_path}")
    
    return results


if __name__ == "__main__":
    try:
        results = train_simplified_enhanced_rl()
        
        print("\n🎉 SIMPLIFIED ENHANCED RL TRAINING COMPLETED!")
        print(f"✅ Success Rate: {results['success_rate']:.1%}")
        print(f"⏱️  Training Time: {results['training_time_minutes']:.1f} minutes")
        print(f"🖥️  Device Used: {results['device_used']}")
        print(f"📈 Final Performance: {results['final_avg_reward']:.2f}")
        
        print("\n🚀 Ready for full enhanced RL training!")
        print("Run: python run_enhanced_rl_pipeline.py --quick")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()