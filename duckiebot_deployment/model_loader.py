#!/usr/bin/env python3
"""
Model loader utility for Duckiebot deployment.
Handles loading different types of RL models for inference.
"""

import torch
import numpy as np
from pathlib import Path
import logging
import importlib.util
import sys
from typing import Union, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DuckiebotRLModel(torch.nn.Module):
    """
    Duckiebot RL Model for lane following.
    This is a copy of the model class for deployment compatibility.
    """
    
    def __init__(self, input_shape=(3, 120, 160), hidden_dim=256):
        super(DuckiebotRLModel, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        
        # CNN feature extractor
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 5))
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_size = cnn_output.numel()
        
        # Fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.cnn_output_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Tanh()
        )
        
        # Value head
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        if x.max() > 1.0:
            x = x / 255.0
        
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        hidden = self.fc(features)
        
        action = self.policy_head(hidden)
        value = self.value_head(hidden)
        
        return {'action': action, 'value': value}
    
    def get_action(self, observation):
        """Get action from observation."""
        self.eval()
        
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.FloatTensor(observation)
            else:
                obs_tensor = observation
            
            if len(obs_tensor.shape) == 3:
                if obs_tensor.shape[0] == 3:
                    pass
                else:
                    obs_tensor = obs_tensor.permute(2, 0, 1)
            
            output = self.forward(obs_tensor)
            action = output['action'].squeeze().cpu().numpy()
            
            return action


class ModelLoader:
    """Utility class for loading different types of RL models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_pytorch_model(self, model_path: str) -> torch.nn.Module:
        """
        Load PyTorch model from .pth file.
        
        Args:
            model_path: Path to .pth file
            
        Returns:
            Loaded PyTorch model
        """
        self.logger.info(f"Loading PyTorch model from: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # State dict format
                model_class = checkpoint.get('model_class', 'DuckiebotRLModel')
                input_shape = checkpoint.get('input_shape', (3, 120, 160))
                hidden_dim = checkpoint.get('hidden_dim', 256)
                
                self.logger.info(f"Model class: {model_class}")
                self.logger.info(f"Input shape: {input_shape}")
                
                # Create model instance
                if model_class == 'DuckiebotRLModel':
                    model = DuckiebotRLModel(input_shape=input_shape, hidden_dim=hidden_dim)
                else:
                    raise ValueError(f"Unknown model class: {model_class}")
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.logger.info("✅ PyTorch model loaded successfully")
                return model
                
            elif hasattr(checkpoint, 'get_action'):
                # Complete model format
                checkpoint.eval()
                self.logger.info("✅ Complete PyTorch model loaded successfully")
                return checkpoint
                
            else:
                raise ValueError("Invalid model format - expected state dict or complete model")
                
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def load_rllib_checkpoint(self, checkpoint_path: str):
        """
        Load Ray RLLib checkpoint.
        
        Args:
            checkpoint_path: Path to RLLib checkpoint directory
            
        Returns:
            RLLib trainer object
        """
        self.logger.info(f"Loading RLLib checkpoint from: {checkpoint_path}")
        
        try:
            from ray.rllib.agents.ppo import PPOTrainer
            
            # Create trainer with minimal config
            trainer = PPOTrainer(config={
                'env': 'DummyEnv',
                'framework': 'torch',
                'num_workers': 0,
                'explore': False
            })
            
            # Restore from checkpoint
            trainer.restore(checkpoint_path)
            
            self.logger.info("✅ RLLib checkpoint loaded successfully")
            return trainer
            
        except ImportError:
            self.logger.error("Ray RLLib not available")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load RLLib checkpoint: {e}")
            raise
    
    def load_model(self, model_path: str) -> Union[torch.nn.Module, Any]:
        """
        Auto-detect and load model from path.
        
        Args:
            model_path: Path to model file or checkpoint
            
        Returns:
            Loaded model object
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"Auto-detecting model type: {model_path}")
        
        if model_path.suffix in ['.pth', '.pt']:
            # PyTorch model
            return self.load_pytorch_model(str(model_path))
            
        elif model_path.is_dir() and (model_path / 'checkpoint_metadata.json').exists():
            # RLLib checkpoint directory
            return self.load_rllib_checkpoint(str(model_path))
            
        elif model_path.is_dir():
            # Look for checkpoint files in directory
            checkpoint_files = list(model_path.glob('checkpoint-*'))
            if checkpoint_files:
                return self.load_rllib_checkpoint(str(model_path))
            else:
                raise ValueError(f"No valid checkpoint found in directory: {model_path}")
        
        else:
            raise ValueError(f"Unknown model format: {model_path}")
    
    def create_inference_wrapper(self, model) -> 'InferenceWrapper':
        """
        Create a unified inference wrapper for any model type.
        
        Args:
            model: Loaded model object
            
        Returns:
            InferenceWrapper instance
        """
        return InferenceWrapper(model)


class InferenceWrapper:
    """
    Unified wrapper for different model types to provide consistent inference interface.
    """
    
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Detect model type
        if hasattr(model, 'get_action'):
            self.model_type = 'pytorch_custom'
        elif hasattr(model, 'compute_action'):
            self.model_type = 'rllib'
        elif isinstance(model, torch.nn.Module):
            self.model_type = 'pytorch_standard'
        else:
            self.model_type = 'unknown'
        
        self.logger.info(f"Created inference wrapper for model type: {self.model_type}")
    
    def compute_action(self, observation: np.ndarray, explore: bool = False) -> np.ndarray:
        """
        Unified interface for computing actions.
        
        Args:
            observation: Input observation
            explore: Whether to use exploration (ignored for inference)
            
        Returns:
            Action array [steering, throttle]
        """
        try:
            if self.model_type == 'pytorch_custom':
                # Custom PyTorch model with get_action method
                return self.model.get_action(observation)
                
            elif self.model_type == 'rllib':
                # Ray RLLib trainer
                return self.model.compute_action(observation, explore=explore)
                
            elif self.model_type == 'pytorch_standard':
                # Standard PyTorch model
                self.model.eval()
                
                with torch.no_grad():
                    if isinstance(observation, np.ndarray):
                        obs_tensor = torch.FloatTensor(observation)
                    else:
                        obs_tensor = observation
                    
                    # Handle different input formats
                    if len(obs_tensor.shape) == 3:
                        if obs_tensor.shape[0] != 3:  # HWC format
                            obs_tensor = obs_tensor.permute(2, 0, 1)
                        obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim
                    
                    # Forward pass
                    output = self.model(obs_tensor)
                    
                    # Extract action
                    if isinstance(output, dict):
                        action = output['action'].squeeze().cpu().numpy()
                    else:
                        action = output.squeeze().cpu().numpy()
                    
                    return action
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            # Return safe default action
            return np.array([0.0, 0.0])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_type': self.model_type,
            'has_get_action': hasattr(self.model, 'get_action'),
            'has_compute_action': hasattr(self.model, 'compute_action'),
            'is_pytorch_module': isinstance(self.model, torch.nn.Module)
        }
        
        if hasattr(self.model, 'input_shape'):
            info['input_shape'] = self.model.input_shape
        
        return info


def load_model_for_deployment(model_path: str) -> InferenceWrapper:
    """
    Convenience function to load any model type for deployment.
    
    Args:
        model_path: Path to model file or checkpoint
        
    Returns:
        InferenceWrapper ready for deployment
    """
    loader = ModelLoader()
    model = loader.load_model(model_path)
    wrapper = loader.create_inference_wrapper(model)
    
    logger.info(f"Model loaded and ready for deployment: {wrapper.get_model_info()}")
    
    return wrapper


if __name__ == "__main__":
    # Test the model loader
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "champion_model.pth"
    
    print(f"Testing model loader with: {model_path}")
    
    try:
        wrapper = load_model_for_deployment(model_path)
        
        # Test inference
        test_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        action = wrapper.compute_action(test_obs)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model info: {wrapper.get_model_info()}")
        print(f"   Test action: {action}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)