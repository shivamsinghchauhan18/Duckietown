"""
Compatibility layer for handling missing dependencies gracefully.

This module provides fallback implementations and graceful degradation
when optional dependencies are not available.
"""

import warnings
import logging
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages optional dependencies and provides fallbacks."""
    
    def __init__(self):
        self.available_packages = {}
        self.missing_packages = {}
        self.fallback_implementations = {}
        
        # Check package availability
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check which dependencies are available."""
        dependencies = {
            'gym': self._check_gym,
            'gym_duckietown': self._check_gym_duckietown,
            'ray': self._check_ray,
            'ultralytics': self._check_ultralytics,
            'cv2': self._check_opencv,
            'torch': self._check_torch,
            'matplotlib': self._check_matplotlib,
            'pytest': self._check_pytest
        }
        
        for name, check_func in dependencies.items():
            try:
                package = check_func()
                self.available_packages[name] = package
                logger.debug(f"✓ {name} available")
            except ImportError as e:
                self.missing_packages[name] = str(e)
                logger.debug(f"✗ {name} not available: {e}")
                warnings.warn(f"{name} not available. Some features will be disabled.")
    
    def _check_gym(self):
        """Check gym availability."""
        import gym
        return gym
    
    def _check_gym_duckietown(self):
        """Check gym-duckietown availability."""
        import gym_duckietown
        return gym_duckietown
    
    def _check_ray(self):
        """Check Ray availability."""
        import ray
        return ray
    
    def _check_ultralytics(self):
        """Check ultralytics availability."""
        from ultralytics import YOLO
        return YOLO
    
    def _check_opencv(self):
        """Check OpenCV availability."""
        import cv2
        return cv2
    
    def _check_torch(self):
        """Check PyTorch availability."""
        import torch
        return torch
    
    def _check_matplotlib(self):
        """Check matplotlib availability."""
        import matplotlib.pyplot as plt
        return plt
    
    def _check_pytest(self):
        """Check pytest availability."""
        import pytest
        return pytest
    
    def is_available(self, package_name: str) -> bool:
        """Check if a package is available."""
        return package_name in self.available_packages
    
    def get_package(self, package_name: str) -> Any:
        """Get a package if available, otherwise return None."""
        return self.available_packages.get(package_name)
    
    def require_package(self, package_name: str, feature_name: str = None) -> Any:
        """Require a package or raise informative error."""
        if package_name in self.available_packages:
            return self.available_packages[package_name]
        
        feature_msg = f" for {feature_name}" if feature_name else ""
        error_msg = f"{package_name} is required{feature_msg} but not available"
        
        if package_name in self.missing_packages:
            error_msg += f": {self.missing_packages[package_name]}"
        
        raise ImportError(error_msg)
    
    def get_fallback(self, package_name: str, fallback_impl: Any = None) -> Any:
        """Get package or fallback implementation."""
        if package_name in self.available_packages:
            return self.available_packages[package_name]
        
        if fallback_impl is not None:
            return fallback_impl
        
        if package_name in self.fallback_implementations:
            return self.fallback_implementations[package_name]
        
        # Return a mock object as last resort
        logger.warning(f"Using mock implementation for {package_name}")
        return Mock()


# Global dependency manager instance
dependency_manager = DependencyManager()


# Convenience functions
def is_gym_available() -> bool:
    """Check if gym is available."""
    return dependency_manager.is_available('gym')


def is_gym_duckietown_available() -> bool:
    """Check if gym-duckietown is available."""
    return dependency_manager.is_available('gym_duckietown')


def is_ray_available() -> bool:
    """Check if Ray is available."""
    return dependency_manager.is_available('ray')


def is_yolo_available() -> bool:
    """Check if YOLO (ultralytics) is available."""
    return dependency_manager.is_available('ultralytics')


def is_opencv_available() -> bool:
    """Check if OpenCV is available."""
    return dependency_manager.is_available('cv2')


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return dependency_manager.is_available('torch')


def is_matplotlib_available() -> bool:
    """Check if matplotlib is available."""
    return dependency_manager.is_available('matplotlib')


def is_pytest_available() -> bool:
    """Check if pytest is available."""
    return dependency_manager.is_available('pytest')


# Fallback implementations
class MockGymEnv:
    """Mock gym environment for when gym is not available."""
    
    def __init__(self, *args, **kwargs):
        self.observation_space = Mock()
        self.action_space = Mock()
        self.observation_space.shape = (120, 160, 3)
        self.action_space.shape = (2,)
        self.step_count = 0
    
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        return {'image': [[0] * 160 for _ in range(120)] * 3}
    
    def step(self, action):
        """Step environment."""
        self.step_count += 1
        obs = {'image': [[0] * 160 for _ in range(120)] * 3}
        reward = 0.0
        done = self.step_count >= 100
        info = {}
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render environment."""
        pass
    
    def close(self):
        """Close environment."""
        pass


class MockYOLO:
    """Mock YOLO model for when ultralytics is not available."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        logger.warning("Using mock YOLO implementation - no actual detection will occur")
    
    def __call__(self, image, *args, **kwargs):
        """Mock inference."""
        return MockYOLOResults()
    
    def predict(self, image, *args, **kwargs):
        """Mock prediction."""
        return [MockYOLOResults()]


class MockYOLOResults:
    """Mock YOLO results."""
    
    def __init__(self):
        self.boxes = Mock()
        self.boxes.data = []  # Empty tensor-like object
        self.boxes.xyxy = []
        self.boxes.conf = []
        self.boxes.cls = []


class MockRay:
    """Mock Ray for when ray is not available."""
    
    @staticmethod
    def init(*args, **kwargs):
        """Mock Ray init."""
        logger.warning("Ray not available - using mock implementation")
    
    @staticmethod
    def shutdown():
        """Mock Ray shutdown."""
        pass
    
    class tune:
        """Mock Ray Tune."""
        
        @staticmethod
        def run(*args, **kwargs):
            """Mock tune run."""
            return {'training_iteration': 1, 'episode_reward_mean': 0.0}
        
        @staticmethod
        def register_env(name, env_creator):
            """Mock env registration."""
            pass


# Register fallback implementations
dependency_manager.fallback_implementations.update({
    'gym': type('MockGym', (), {
        'Env': MockGymEnv,
        'spaces': type('MockSpaces', (), {
            'Box': Mock,
            'Discrete': Mock
        })()
    })(),
    'ultralytics': type('MockUltralytics', (), {
        'YOLO': MockYOLO
    })(),
    'ray': MockRay()
})


def safe_import(package_name: str, feature_name: str = None, 
               fallback: Any = None, required: bool = False):
    """Safely import a package with fallback options."""
    if required:
        return dependency_manager.require_package(package_name, feature_name)
    else:
        return dependency_manager.get_fallback(package_name, fallback)


def check_system_compatibility() -> Dict[str, Any]:
    """Check overall system compatibility."""
    compatibility_report = {
        'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        'available_packages': list(dependency_manager.available_packages.keys()),
        'missing_packages': list(dependency_manager.missing_packages.keys()),
        'core_functionality': True,
        'optional_features': {},
        'recommendations': []
    }
    
    # Check core functionality
    core_packages = ['numpy', 'torch', 'cv2']
    missing_core = [pkg for pkg in core_packages 
                   if not dependency_manager.is_available(pkg)]
    
    if missing_core:
        compatibility_report['core_functionality'] = False
        compatibility_report['recommendations'].append(
            f"Install core packages: {', '.join(missing_core)}"
        )
    
    # Check optional features
    optional_features = {
        'gym_environments': dependency_manager.is_available('gym'),
        'duckietown_environments': dependency_manager.is_available('gym_duckietown'),
        'object_detection': dependency_manager.is_available('ultralytics'),
        'distributed_training': dependency_manager.is_available('ray'),
        'visualization': dependency_manager.is_available('matplotlib'),
        'testing': dependency_manager.is_available('pytest')
    }
    
    compatibility_report['optional_features'] = optional_features
    
    # Generate recommendations
    if not optional_features['gym_environments']:
        compatibility_report['recommendations'].append(
            "Install gym for full environment support: pip install gym"
        )
    
    if not optional_features['duckietown_environments']:
        compatibility_report['recommendations'].append(
            "Install gym-duckietown: pip install git+https://github.com/duckietown/gym-duckietown.git"
        )
    
    if not optional_features['object_detection']:
        compatibility_report['recommendations'].append(
            "Install ultralytics for YOLO support: pip install ultralytics"
        )
    
    if not optional_features['distributed_training']:
        compatibility_report['recommendations'].append(
            "Install Ray for distributed training: pip install ray[rllib]"
        )
    
    return compatibility_report


def print_compatibility_report():
    """Print a formatted compatibility report."""
    report = check_system_compatibility()
    
    print("🔍 System Compatibility Report")
    print("=" * 40)
    print(f"Python Version: {report['python_version']}")
    print(f"Core Functionality: {'✅' if report['core_functionality'] else '❌'}")
    
    print(f"\n📦 Available Packages ({len(report['available_packages'])}):")
    for package in sorted(report['available_packages']):
        print(f"  ✅ {package}")
    
    if report['missing_packages']:
        print(f"\n❌ Missing Packages ({len(report['missing_packages'])}):")
        for package in sorted(report['missing_packages']):
            print(f"  ❌ {package}")
    
    print(f"\n🎯 Optional Features:")
    for feature, available in report['optional_features'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    return report


if __name__ == "__main__":
    print_compatibility_report()