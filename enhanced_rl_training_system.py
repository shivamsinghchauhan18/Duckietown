#!/usr/bin/env python3
"""
🚀 ENHANCED DUCKIETOWN RL TRAINING SYSTEM 🚀

Complete implementation of the enhanced RL system with:
- YOLO v5 object detection integration
- Object avoidance and lane changing capabilities
- Multi-objective reward optimization
- Metal framework GPU acceleration (macOS)
- Rigorous evaluation and testing
- Production-ready deployment

This system bridges the gap between the advanced infrastructure and deployment.
"""

import os
import sys
import time
import json
import yaml
import logging
import threading
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import argparse

# Core ML and RL imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Gym and environment imports
import gym
from gym import spaces

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our enhanced infrastructure
from duckietown_utils.env import launch_and_wrap_enhanced_env
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
from config.enhanced_config import EnhancedRLConfig, load_enhanced_config
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator
from duckietown_utils.enhanced_logger import EnhancedLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Enhanced training configuration."""
    # Training parameters
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 50_000
    train_freq: int = 4
    target_update_interval: int = 10_000
    
    # Enhanced features
    use_yolo: bool = True
    use_object_avoidance: bool = True
    use_lane_changing: bool = True
    use_multi_objective_reward: bool = True
    
    # Metal framework (macOS GPU acceleration)
    use_metal: bool = True
    metal_device: str = "mps"  # Metal Performance Shaders
    
    # Evaluation
    eval_freq: int = 50_000
    eval_episodes: int = 20
    
    # Logging and checkpointing
    log_interval: int = 1000
    save_freq: int = 100_000
    
    # Maps for training
    training_maps: List[str] = None
    
    def __post_init__(self):
        if self.training_maps is None:
            self.training_maps = [
                'loop_empty',
                'small_loop', 
                'zigzag_dists',
                'loop_obstacles',
                'loop_pedestrians'
            ]


class EnhancedDQNNetwork(nn.Module):
    """Enhanced DQN network with YOLO feature integration."""
    
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Determine input dimensions
        if isinstance(observation_space, spaces.Dict):
            # Enhanced observation space with YOLO features
            self.use_enhanced_obs = True
            self.image_shape = observation_space.spaces.get('image', spaces.Box(0, 255, (120, 160, 3))).shape
            self.detection_features = observation_space.spaces.get('detection_features', spaces.Box(-np.inf, np.inf, (90,))).shape[0]
            self.safety_features = observation_space.spaces.get('safety_features', spaces.Box(-np.inf, np.inf, (5,))).shape[0]
        else:
            # Flattened observation space
            self.use_enhanced_obs = False
            self.obs_dim = observation_space.shape[0]
        
        self.action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n
        
        if self.use_enhanced_obs:
            self._build_enhanced_network()
        else:
            self._build_standard_network()
    
    def _build_enhanced_network(self):
        """Build network for enhanced observations with YOLO integration."""
        # Image processing branch (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 5))
        )
        
        # Calculate CNN output size
        cnn_output_size = 64 * 4 * 5  # 1280
        
        # Detection features branch (MLP)
        self.detection_encoder = nn.Sequential(
            nn.Linear(self.detection_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Safety features branch (MLP)
        self.safety_encoder = nn.Sequential(
            nn.Linear(self.safety_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_size = cnn_output_size + 64 + 16  # 1360
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Action value heads
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def _build_standard_network(self):
        """Build network for standard flattened observations."""
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, self.action_dim)
    
    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through the network."""
        if self.use_enhanced_obs:
            return self._forward_enhanced(obs)
        else:
            return self._forward_standard(obs)
    
    def _forward_enhanced(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for enhanced observations."""
        # Process image
        image = obs['image']
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Ensure correct channel order (HWC -> CHW)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Process detection features
        detection_features = self.detection_encoder(obs['detection_features'])
        
        # Process safety features
        safety_features = self.safety_encoder(obs['safety_features'])
        
        # Fuse all features
        fused_features = torch.cat([image_features, detection_features, safety_features], dim=1)
        hidden = self.fusion_layer(fused_features)
        
        # Dueling DQN architecture
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def _forward_standard(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for standard observations."""
        hidden = self.network(obs)
        
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class EnhancedDQNAgent:
    """Enhanced DQN agent with YOLO integration and Metal support."""
    
    def __init__(self, env: gym.Env, config: TrainingConfig):
        self.env = env
        self.config = config
        
        # Setup device (Metal support for macOS)
        self.device = self._setup_device()
        
        # Create networks
        self.q_network = EnhancedDQNNetwork(env.observation_space, env.action_space, config).to(self.device)
        self.target_network = EnhancedDQNNetwork(env.observation_space, env.action_space, config).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = []
        
        # Training state
        self.timesteps = 0
        self.episodes = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Logging
        self.logger = EnhancedLogger("enhanced_rl_training")
        self.writer = SummaryWriter(f"logs/enhanced_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        logger.info(f"Enhanced DQN Agent initialized on device: {self.device}")
        logger.info(f"Q-Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device with Metal support."""
        if self.config.use_metal and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Metal Performance Shaders (MPS) for GPU acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA for GPU acceleration")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for computation")
        
        return device
    
    def select_action(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            # Random action
            if hasattr(self.env.action_space, 'sample'):
                return self.env.action_space.sample()
            else:
                return np.random.randint(0, self.env.action_space.n)
        
        # Greedy action
        with torch.no_grad():
            if isinstance(obs, dict):
                obs_tensor = {k: torch.FloatTensor(v).to(self.device) for k, v in obs.items()}
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            q_values = self.q_network(obs_tensor)
            
            if hasattr(self.env.action_space, 'shape'):
                # Continuous action space - use the Q-values directly
                return q_values.cpu().numpy().squeeze()
            else:
                # Discrete action space
                return q_values.argmax().cpu().numpy()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((obs, action, reward, next_obs, done))
        
        # Keep buffer size manageable
        if len(self.replay_buffer) > self.config.buffer_size:
            self.replay_buffer.pop(0)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.replay_buffer), self.config.batch_size, replace=False)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for idx in batch:
            obs, action, reward, next_obs, done = self.replay_buffer[idx]
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        
        # Convert to tensors
        if isinstance(obs_batch[0], dict):
            obs_tensor = {}
            next_obs_tensor = {}
            for key in obs_batch[0].keys():
                obs_tensor[key] = torch.FloatTensor([obs[key] for obs in obs_batch]).to(self.device)
                next_obs_tensor[key] = torch.FloatTensor([obs[key] for obs in next_obs_batch]).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
            next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        done_tensor = torch.BoolTensor(done_batch).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(obs_tensor)
        if hasattr(self.env.action_space, 'shape'):
            # Continuous actions - use MSE loss
            current_q = current_q_values
        else:
            # Discrete actions
            current_q = current_q_values.gather(1, action_tensor.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor)
            if hasattr(self.env.action_space, 'shape'):
                next_q = next_q_values
            else:
                next_q = next_q_values.max(1)[0]
            
            target_q = reward_tensor + (0.99 * next_q * ~done_tensor)
        
        # Compute loss
        if hasattr(self.env.action_space, 'shape'):
            loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        else:
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Log training metrics
        self.writer.add_scalar('Training/Loss', loss.item(), self.timesteps)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, self.timesteps)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps': self.timesteps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.timesteps = checkpoint['timesteps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        
        logger.info(f"Model loaded from {path}")


class EnhancedRLTrainer:
    """Enhanced RL trainer with full integration."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.training_config = TrainingConfig()
        self.enhanced_config = load_enhanced_config(config_path) if config_path else load_enhanced_config()
        
        # Setup directories
        self.log_dir = Path("logs/enhanced_rl_training")
        self.model_dir = Path("models/enhanced_rl")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.training_active = False
        self.best_reward = -float('inf')
        self.evaluation_results = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Enhanced RL Trainer initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Training interrupted. Saving model...")
        self.training_active = False
        if hasattr(self, 'agent'):
            self.agent.save_model(str(self.model_dir / "interrupted_model.pth"))
        sys.exit(0)
    
    def create_enhanced_environment(self, map_name: str = None) -> gym.Env:
        """Create enhanced environment with all wrappers."""
        # Base environment configuration
        env_config = {
            'training_map': map_name or 'loop_empty',
            'episode_max_steps': 1000,
            'domain_rand': True,
            'dynamics_rand': True,
            'camera_rand': True,
            'accepted_start_angle_deg': 60,
            'distortion': True,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.3,
            'mode': 'train',
            'aido_wrapper': False,
            'crop_image_top': True,
            'top_crop_divider': 3,
            'resized_input_shape': (120, 160, 3),
            'frame_stacking': False,
            'action_type': 'continuous',
            'reward_function': 'Posangle'
        }
        
        # Create enhanced environment
        env = launch_and_wrap_enhanced_env(env_config, self.enhanced_config)
        
        logger.info(f"Enhanced environment created for map: {map_name}")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        return env
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting Enhanced RL Training")
        logger.info("=" * 60)
        logger.info(f"Total timesteps: {self.training_config.total_timesteps:,}")
        logger.info(f"Using YOLO: {self.training_config.use_yolo}")
        logger.info(f"Using Object Avoidance: {self.training_config.use_object_avoidance}")
        logger.info(f"Using Lane Changing: {self.training_config.use_lane_changing}")
        logger.info(f"Using Metal: {self.training_config.use_metal}")
        logger.info("=" * 60)
        
        self.training_active = True
        
        # Create environment
        env = self.create_enhanced_environment()
        
        # Create agent
        self.agent = EnhancedDQNAgent(env, self.training_config)
        
        # Training loop
        episode_rewards = []
        episode_lengths = []
        
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for timestep in range(self.training_config.total_timesteps):
            if not self.training_active:
                break
            
            self.agent.timesteps = timestep
            
            # Select action
            action = self.agent.select_action(obs, training=True)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            self.agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train agent
            if timestep >= self.training_config.learning_starts:
                if timestep % self.training_config.train_freq == 0:
                    loss = self.agent.train_step()
                
                # Update target network
                if timestep % self.training_config.target_update_interval == 0:
                    self.agent.update_target_network()
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                self.agent.episodes += 1
                
                # Log episode
                if len(episode_rewards) % self.training_config.log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    
                    logger.info(f"Episode {self.agent.episodes:,} | "
                              f"Timestep {timestep:,} | "
                              f"Avg Reward: {avg_reward:.2f} | "
                              f"Avg Length: {avg_length:.1f} | "
                              f"Epsilon: {self.agent.epsilon:.3f}")
                    
                    self.agent.writer.add_scalar('Episode/Reward', episode_reward, self.agent.episodes)
                    self.agent.writer.add_scalar('Episode/Length', episode_length, self.agent.episodes)
                    self.agent.writer.add_scalar('Episode/AvgReward100', avg_reward, self.agent.episodes)
                
                # Reset episode
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Evaluation
            if timestep % self.training_config.eval_freq == 0 and timestep > 0:
                eval_reward = self.evaluate_agent(env)
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.agent.save_model(str(self.model_dir / "best_model.pth"))
                    logger.info(f"New best model saved! Reward: {eval_reward:.2f}")
            
            # Save checkpoint
            if timestep % self.training_config.save_freq == 0 and timestep > 0:
                self.agent.save_model(str(self.model_dir / f"checkpoint_{timestep}.pth"))
        
        # Final save
        self.agent.save_model(str(self.model_dir / "final_model.pth"))
        
        # Final evaluation
        logger.info("🏁 Training completed! Running final evaluation...")
        self.run_comprehensive_evaluation()
        
        logger.info("✅ Enhanced RL Training completed successfully!")
    
    def evaluate_agent(self, env: gym.Env, num_episodes: int = None) -> float:
        """Evaluate agent performance."""
        num_episodes = num_episodes or self.training_config.eval_episodes
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(obs, training=False)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        logger.info(f"Evaluation: {avg_reward:.2f} ± {std_reward:.2f} over {num_episodes} episodes")
        
        self.agent.writer.add_scalar('Evaluation/AvgReward', avg_reward, self.agent.timesteps)
        self.agent.writer.add_scalar('Evaluation/StdReward', std_reward, self.agent.timesteps)
        
        return avg_reward
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across all maps."""
        logger.info("🔬 Running comprehensive evaluation...")
        
        # Create evaluation orchestrator
        eval_config = {
            'suites': ['base', 'hard'],
            'seeds_per_map': 20,
            'policy_modes': ['deterministic'],
            'compute_ci': True
        }
        
        evaluator = EvaluationOrchestrator(eval_config)
        
        # Evaluate on all training maps
        results = {}
        for map_name in self.training_config.training_maps:
            logger.info(f"Evaluating on map: {map_name}")
            
            # Create environment for this map
            env = self.create_enhanced_environment(map_name)
            
            # Run evaluation
            map_results = self.evaluate_agent(env, num_episodes=50)
            results[map_name] = map_results
            
            env.close()
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.log_dir / f"comprehensive_evaluation_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'training_timesteps': self.agent.timesteps,
                'training_episodes': self.agent.episodes,
                'best_reward': self.best_reward,
                'map_results': results,
                'config': asdict(self.training_config)
            }, f, indent=2)
        
        logger.info(f"Comprehensive evaluation results saved to {results_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced Duckietown RL Training")
    parser.add_argument('--config', type=str, help='Path to enhanced config file')
    parser.add_argument('--timesteps', type=int, default=5_000_000, help='Total training timesteps')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO detection')
    parser.add_argument('--no-metal', action='store_true', help='Disable Metal acceleration')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedRLTrainer(args.config)
    
    # Update config based on arguments
    trainer.training_config.total_timesteps = args.timesteps
    trainer.training_config.use_yolo = not args.no_yolo
    trainer.training_config.use_metal = not args.no_metal
    
    if args.eval_only:
        # Load best model and evaluate
        model_path = trainer.model_dir / "best_model.pth"
        if model_path.exists():
            env = trainer.create_enhanced_environment()
            trainer.agent = EnhancedDQNAgent(env, trainer.training_config)
            trainer.agent.load_model(str(model_path))
            trainer.run_comprehensive_evaluation()
        else:
            logger.error("No trained model found for evaluation")
    else:
        # Run training
        trainer.train()


if __name__ == "__main__":
    main()