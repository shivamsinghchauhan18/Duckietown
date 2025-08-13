"""
Configuration utilities for Enhanced Duckietown RL system.
Provides runtime parameter adjustment and configuration management tools.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
from datetime import datetime

from .enhanced_config import EnhancedRLConfig, ValidationError

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Manages configuration updates, validation, and persistence for Enhanced RL system.
    """
    
    def __init__(self, config: EnhancedRLConfig, config_file_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Initial configuration
            config_file_path: Path to configuration file for persistence
        """
        self.config = config
        self.config_file_path = Path(config_file_path) if config_file_path else None
        self.update_history: List[Dict[str, Any]] = []
    
    def update_yolo_config(self, **kwargs) -> None:
        """
        Update YOLO configuration parameters.
        
        Args:
            **kwargs: YOLO configuration parameters to update
        """
        self._update_component_config('yolo', kwargs)
    
    def update_object_avoidance_config(self, **kwargs) -> None:
        """
        Update object avoidance configuration parameters.
        
        Args:
            **kwargs: Object avoidance configuration parameters to update
        """
        self._update_component_config('object_avoidance', kwargs)
    
    def update_lane_changing_config(self, **kwargs) -> None:
        """
        Update lane changing configuration parameters.
        
        Args:
            **kwargs: Lane changing configuration parameters to update
        """
        self._update_component_config('lane_changing', kwargs)
    
    def update_reward_config(self, **kwargs) -> None:
        """
        Update reward configuration parameters.
        
        Args:
            **kwargs: Reward configuration parameters to update
        """
        self._update_component_config('reward', kwargs)
    
    def update_logging_config(self, **kwargs) -> None:
        """
        Update logging configuration parameters.
        
        Args:
            **kwargs: Logging configuration parameters to update
        """
        self._update_component_config('logging', kwargs)
    
    def update_performance_config(self, **kwargs) -> None:
        """
        Update performance configuration parameters.
        
        Args:
            **kwargs: Performance configuration parameters to update
        """
        self._update_component_config('performance', kwargs)
    
    def enable_feature(self, feature_name: str) -> None:
        """
        Enable a specific feature.
        
        Args:
            feature_name: Name of feature to enable
        """
        if feature_name not in self.config.enabled_features:
            self.config.enabled_features.append(feature_name)
            self._log_update('enable_feature', {'feature': feature_name})
            logger.info(f"Feature '{feature_name}' enabled")
    
    def disable_feature(self, feature_name: str) -> None:
        """
        Disable a specific feature.
        
        Args:
            feature_name: Name of feature to disable
        """
        if feature_name in self.config.enabled_features:
            self.config.enabled_features.remove(feature_name)
            self._log_update('disable_feature', {'feature': feature_name})
            logger.info(f"Feature '{feature_name}' disabled")
    
    def set_debug_mode(self, debug_mode: bool) -> None:
        """
        Set debug mode on or off.
        
        Args:
            debug_mode: True to enable debug mode, False to disable
        """
        old_value = self.config.debug_mode
        self.config.debug_mode = debug_mode
        self._log_update('set_debug_mode', {'old_value': old_value, 'new_value': debug_mode})
        logger.info(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
    
    def batch_update(self, updates: Dict[str, Any]) -> None:
        """
        Apply multiple configuration updates in a single transaction.
        
        Args:
            updates: Dictionary of updates to apply
        """
        try:
            self.config.update(updates)
            self._log_update('batch_update', updates)
            logger.info("Batch configuration update completed successfully")
        except ValidationError as e:
            logger.error(f"Batch update failed: {e}")
            raise
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration. If None, uses initialized path.
        """
        save_path = Path(file_path) if file_path else self.config_file_path
        
        if save_path is None:
            raise ValueError("No file path provided for saving configuration")
        
        self.config.to_yaml(save_path)
        logger.info(f"Configuration saved to {save_path}")
    
    def reload_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Reload configuration from file.
        
        Args:
            file_path: Path to load configuration from. If None, uses initialized path.
        """
        load_path = Path(file_path) if file_path else self.config_file_path
        
        if load_path is None:
            raise ValueError("No file path provided for reloading configuration")
        
        self.config = EnhancedRLConfig.from_yaml(load_path)
        self._log_update('reload_config', {'file_path': str(load_path)})
        logger.info(f"Configuration reloaded from {load_path}")
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """
        Get history of configuration updates.
        
        Returns:
            List of update records with timestamps
        """
        return self.update_history.copy()
    
    def export_config_summary(self) -> Dict[str, Any]:
        """
        Export a summary of current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            'enabled_features': self.config.enabled_features,
            'debug_mode': self.config.debug_mode,
            'yolo_confidence_threshold': self.config.yolo.confidence_threshold,
            'safety_distance': self.config.object_avoidance.safety_distance,
            'lane_change_threshold': self.config.lane_changing.lane_change_threshold,
            'reward_weights': {
                'lane_following': self.config.reward.lane_following_weight,
                'object_avoidance': self.config.reward.object_avoidance_weight,
                'lane_change': self.config.reward.lane_change_weight,
                'efficiency': self.config.reward.efficiency_weight
            },
            'log_level': self.config.logging.log_level,
            'max_fps': self.config.performance.max_fps
        }
    
    def validate_current_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Create a new config from current values to trigger validation
            from dataclasses import asdict
            config_dict = asdict(self.config)
            EnhancedRLConfig._validate_config_schema(config_dict)
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _update_component_config(self, component: str, updates: Dict[str, Any]) -> None:
        """
        Update a specific component configuration.
        
        Args:
            component: Component name
            updates: Updates to apply
        """
        try:
            self.config.update({component: updates})
            self._log_update(f'update_{component}', updates)
            logger.info(f"{component.title()} configuration updated successfully")
        except ValidationError as e:
            logger.error(f"Failed to update {component} configuration: {e}")
            raise
    
    def _log_update(self, update_type: str, update_data: Dict[str, Any]) -> None:
        """
        Log configuration update to history.
        
        Args:
            update_type: Type of update performed
            update_data: Data that was updated
        """
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'update_type': update_type,
            'data': update_data
        }
        self.update_history.append(update_record)


class ConfigPresets:
    """
    Predefined configuration presets for common use cases.
    """
    
    @staticmethod
    def get_development_preset() -> Dict[str, Any]:
        """Get configuration preset for development/debugging."""
        return {
            'debug_mode': True,
            'logging': {
                'log_level': 'DEBUG',
                'log_detections': True,
                'log_actions': True,
                'log_rewards': True,
                'log_performance': True,
                'console_logging': True
            },
            'performance': {
                'max_fps': 10.0,
                'use_gpu_acceleration': False
            },
            'yolo': {
                'confidence_threshold': 0.3,
                'device': 'cpu'
            }
        }
    
    @staticmethod
    def get_production_preset() -> Dict[str, Any]:
        """Get configuration preset for production/training."""
        return {
            'debug_mode': False,
            'logging': {
                'log_level': 'INFO',
                'log_detections': False,
                'log_actions': False,
                'log_rewards': True,
                'log_performance': True,
                'console_logging': False
            },
            'performance': {
                'max_fps': 30.0,
                'use_gpu_acceleration': True
            },
            'yolo': {
                'confidence_threshold': 0.5,
                'device': 'cuda'
            }
        }
    
    @staticmethod
    def get_high_performance_preset() -> Dict[str, Any]:
        """Get configuration preset for high-performance training."""
        return {
            'debug_mode': False,
            'logging': {
                'log_level': 'WARNING',
                'log_detections': False,
                'log_actions': False,
                'log_rewards': False,
                'log_performance': False,
                'console_logging': False
            },
            'performance': {
                'max_fps': 60.0,
                'detection_batch_size': 4,
                'use_gpu_acceleration': True,
                'memory_limit_gb': 8.0
            },
            'yolo': {
                'confidence_threshold': 0.6,
                'device': 'cuda',
                'input_size': 416  # Smaller for faster inference
            }
        }
    
    @staticmethod
    def get_safe_driving_preset() -> Dict[str, Any]:
        """Get configuration preset optimized for safe driving."""
        return {
            'object_avoidance': {
                'safety_distance': 0.8,
                'avoidance_strength': 1.5,
                'min_clearance': 0.3
            },
            'lane_changing': {
                'lane_change_threshold': 0.2,
                'safety_margin': 3.0,
                'max_lane_change_time': 4.0
            },
            'reward': {
                'safety_penalty_weight': -5.0,
                'collision_penalty': -20.0,
                'object_avoidance_weight': 1.0
            },
            'yolo': {
                'confidence_threshold': 0.4  # Lower threshold for better detection
            }
        }


def apply_preset(config_manager: ConfigurationManager, preset_name: str) -> None:
    """
    Apply a predefined configuration preset.
    
    Args:
        config_manager: Configuration manager instance
        preset_name: Name of preset to apply
    """
    presets = {
        'development': ConfigPresets.get_development_preset(),
        'production': ConfigPresets.get_production_preset(),
        'high_performance': ConfigPresets.get_high_performance_preset(),
        'safe_driving': ConfigPresets.get_safe_driving_preset()
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
    
    preset_config = presets[preset_name]
    config_manager.batch_update(preset_config)
    logger.info(f"Applied '{preset_name}' configuration preset")


def create_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """
    Create a configuration manager with optional config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigurationManager instance
    """
    if config_path:
        config = EnhancedRLConfig.from_yaml(config_path)
    else:
        config = EnhancedRLConfig()
    
    return ConfigurationManager(config, config_path)