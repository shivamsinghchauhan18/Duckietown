#!/usr/bin/env python3
"""
WSL + RTX 3060 Dependency Resolver for Enhanced Duckietown RL

This module provides intelligent dependency resolution for WSL environments
with NVIDIA GPU support, handling common conflicts gracefully.
"""

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class WSLDependencyResolver:
    """Intelligent dependency resolver for WSL + GPU environments."""
    
    def __init__(self):
        self.resolved_deps = {}
        self.failed_deps = {}
        self.gpu_available = False
        self.wsl_detected = False
        
        # Detect environment
        self._detect_environment()
        
    def _detect_environment(self):
        """Detect WSL and GPU environment."""
        # Check for WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    self.wsl_detected = True
                    logger.info("WSL environment detected")
        except:
            pass
            
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {gpu_name}")
        except:
            pass
    
    def resolve_all_dependencies(self) -> Dict[str, bool]:
        """Resolve all enhanced RL dependencies with conflict handling."""
        results = {}
        
        # Core dependencies (must have)
        core_deps = [
            ('torch', self._resolve_pytorch),
            ('numpy', self._resolve_numpy),
            ('gym', self._resolve_gym),
        ]
        
        # Enhanced dependencies (graceful fallback)
        enhanced_deps = [
            ('tensorboard', self._resolve_tensorboard),
            ('ultralytics', self._resolve_yolo),
            ('gym_duckietown', self._resolve_gym_duckietown),
            ('pyglet', self._resolve_pyglet),
        ]
        
        # Resolve core dependencies first
        for name, resolver in core_deps:
            try:
                results[name] = resolver()
                if not results[name]:
                    logger.error(f"Critical dependency {name} failed to resolve")
            except Exception as e:
                logger.error(f"Failed to resolve {name}: {e}")
                results[name] = False
        
        # Resolve enhanced dependencies with fallbacks
        for name, resolver in enhanced_deps:
            try:
                results[name] = resolver()
            except Exception as e:
                logger.warning(f"Enhanced dependency {name} failed: {e}")
                results[name] = False
        
        return results
    
    def _resolve_pytorch(self) -> bool:
        """Resolve PyTorch with CUDA support for RTX 3060."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"PyTorch with CUDA already available: {torch.__version__}")
                return True
        except ImportError:
            pass
        
        # Install PyTorch with CUDA 11.8 (compatible with RTX 3060)
        try:
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Verify installation
            import torch
            return torch.cuda.is_available()
        except Exception as e:
            logger.error(f"Failed to install PyTorch: {e}")
            return False
    
    def _resolve_numpy(self) -> bool:
        """Resolve NumPy with version compatibility."""
        try:
            import numpy as np
            # Check for NumPy 2.0 compatibility issues
            version = np.__version__
            if version.startswith('2.'):
                logger.warning("NumPy 2.0 detected - may cause gym compatibility issues")
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2.0"], 
                             check=True, capture_output=True)
                return True
            except:
                return False
    
    def _resolve_gym(self) -> bool:
        """Resolve Gym vs Gymnasium compatibility."""
        try:
            # Try gymnasium first (maintained)
            import gymnasium as gym
            logger.info("Using Gymnasium (recommended)")
            return True
        except ImportError:
            try:
                # Fallback to old gym with NumPy compatibility
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "gym==0.21.0", "numpy<2.0"
                ], check=True, capture_output=True)
                import gym
                logger.info("Using legacy Gym with NumPy<2.0")
                return True
            except:
                return False
    
    def _resolve_tensorboard(self) -> bool:
        """Resolve TensorBoard with TensorFlow compatibility."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            logger.info("TensorBoard available via PyTorch")
            return True
        except ImportError:
            try:
                # Install compatible versions
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "tensorboard", "protobuf<4.0"
                ], check=True, capture_output=True)
                from torch.utils.tensorboard import SummaryWriter
                return True
            except Exception as e:
                logger.warning(f"TensorBoard installation failed: {e}")
                return False
    
    def _resolve_yolo(self) -> bool:
        """Resolve YOLO with GPU support."""
        try:
            from ultralytics import YOLO
            logger.info("YOLO (Ultralytics) available")
            return True
        except ImportError:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "ultralytics", "opencv-python"
                ], check=True, capture_output=True)
                from ultralytics import YOLO
                return True
            except Exception as e:
                logger.warning(f"YOLO installation failed: {e}")
                return False
    
    def _resolve_gym_duckietown(self) -> bool:
        """Resolve gym-duckietown with WSL compatibility."""
        try:
            import gym_duckietown
            logger.info("gym-duckietown available")
            return True
        except ImportError:
            if self.wsl_detected:
                # WSL-specific installation
                try:
                    # Set up virtual display for WSL
                    os.environ['DISPLAY'] = ':0'
                    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
                    
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        "gym-duckietown", "pyglet==1.5.27"
                    ], check=True, capture_output=True)
                    return True
                except Exception as e:
                    logger.warning(f"gym-duckietown WSL installation failed: {e}")
                    return False
            else:
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "gym-duckietown"
                    ], check=True, capture_output=True)
                    return True
                except:
                    return False
    
    def _resolve_pyglet(self) -> bool:
        """Resolve Pyglet with WSL OpenGL compatibility."""
        try:
            import pyglet
            return True
        except ImportError:
            try:
                if self.wsl_detected:
                    # WSL-specific Pyglet setup
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        "pyglet==1.5.27"  # Stable version for WSL
                    ], check=True, capture_output=True)
                else:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "pyglet"
                    ], check=True, capture_output=True)
                return True
            except:
                return False
    
    def create_compatibility_layer(self) -> str:
        """Create compatibility imports for missing dependencies."""
        compat_code = '''
# Auto-generated compatibility layer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Gym/Gymnasium compatibility
try:
    import gymnasium as gym
    gym.make = gym.make  # Ensure compatibility
except ImportError:
    try:
        import gym
    except ImportError:
        class MockGym:
            def make(self, *args, **kwargs):
                raise ImportError("Neither gym nor gymnasium available")
        gym = MockGym()

# TensorBoard compatibility
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class MockSummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass
    SummaryWriter = MockSummaryWriter

# YOLO compatibility
try:
    from ultralytics import YOLO
except ImportError:
    class MockYOLO:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): 
            return [type('MockResult', (), {'boxes': None})()]
    YOLO = MockYOLO

# gym-duckietown compatibility
try:
    import gym_duckietown
    GYM_DUCKIETOWN_AVAILABLE = True
except ImportError:
    GYM_DUCKIETOWN_AVAILABLE = False
    warnings.warn("gym-duckietown not available - using fallback mode")
'''
        
        # Write compatibility layer
        compat_path = Path("duckietown_utils/wsl_compatibility.py")
        compat_path.parent.mkdir(exist_ok=True)
        with open(compat_path, 'w') as f:
            f.write(compat_code)
        
        return str(compat_path)

def setup_wsl_environment():
    """Setup WSL environment for enhanced RL training."""
    resolver = WSLDependencyResolver()
    
    print("🔧 Setting up WSL + RTX 3060 environment for Enhanced Duckietown RL...")
    
    # Resolve dependencies
    results = resolver.resolve_all_dependencies()
    
    # Create compatibility layer
    compat_path = resolver.create_compatibility_layer()
    
    # Report results
    print("\n📊 Dependency Resolution Results:")
    for dep, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")
    
    print(f"\n🛡️ Compatibility layer created: {compat_path}")
    
    # WSL-specific optimizations
    if resolver.wsl_detected:
        print("\n🐧 Applying WSL optimizations...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
    return results

if __name__ == "__main__":
    setup_wsl_environment()