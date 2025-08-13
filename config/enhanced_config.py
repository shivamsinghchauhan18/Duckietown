"""
Enhanced Configuration Management System for Duckietown RL
Provides comprehensive parameter validation, YAML loading, and runtime configuration updates.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import os
import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """Configuration for YOLO object detection."""
    model_path: str = "yolov5s.pt"
    confidence_threshold: float = 0.5
    device: str = "cuda"
    input_size: int = 640
    max_detections: int = 100
    
    def __post_init__(self):
        """Validate YOLO configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"YOLO confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")
        
        if self.device not in ["cuda", "cpu", "auto"]:
            raise ValueError(f"YOLO device must be 'cuda', 'cpu', or 'auto', got {self.device}")
        
        if self.input_size <= 0 or self.input_size % 32 != 0:
            raise ValueError(f"YOLO input_size must be positive and divisible by 32, got {self.input_size}")
        
        if self.max_detections <= 0:
            raise ValueError(f"YOLO max_detections must be positive, got {self.max_detections}")


@dataclass
class ObjectAvoidanceConfig:
    """Configuration for object avoidance behavior."""
    safety_distance: float = 0.5
    avoidance_strength: float = 1.0
    min_clearance: float = 0.2
    max_avoidance_angle: float = 0.5
    smoothing_factor: float = 0.8
    
    def __post_init__(self):
        """Validate object avoidance configuration parameters."""
        if self.safety_distance <= 0:
            raise ValueError(f"Object avoidance safety_distance must be positive, got {self.safety_distance}")
        
        if self.avoidance_strength <= 0:
            raise ValueError(f"Object avoidance avoidance_strength must be positive, got {self.avoidance_strength}")
        
        if self.min_clearance <= 0:
            raise ValueError(f"Object avoidance min_clearance must be positive, got {self.min_clearance}")
        
        if self.min_clearance >= self.safety_distance:
            raise ValueError(f"Object avoidance min_clearance ({self.min_clearance}) must be less than safety_distance ({self.safety_distance})")
        
        if not 0.0 <= self.max_avoidance_angle <= 1.57:  # ~90 degrees in radians
            raise ValueError(f"Object avoidance max_avoidance_angle must be between 0.0 and 1.57 radians, got {self.max_avoidance_angle}")
        
        if not 0.0 <= self.smoothing_factor <= 1.0:
            raise ValueError(f"Object avoidance smoothing_factor must be between 0.0 and 1.0, got {self.smoothing_factor}")


@dataclass
class LaneChangingConfig:
    """Configuration for lane changing behavior."""
    lane_change_threshold: float = 0.3
    safety_margin: float = 2.0
    max_lane_change_time: float = 3.0
    min_lane_width: float = 0.4
    evaluation_distance: float = 5.0
    
    def __post_init__(self):
        """Validate lane changing configuration parameters."""
        if not 0.0 <= self.lane_change_threshold <= 1.0:
            raise ValueError(f"Lane changing lane_change_threshold must be between 0.0 and 1.0, got {self.lane_change_threshold}")
        
        if self.safety_margin <= 0:
            raise ValueError(f"Lane changing safety_margin must be positive, got {self.safety_margin}")
        
        if self.max_lane_change_time <= 0:
            raise ValueError(f"Lane changing max_lane_change_time must be positive, got {self.max_lane_change_time}")
        
        if self.min_lane_width <= 0:
            raise ValueError(f"Lane changing min_lane_width must be positive, got {self.min_lane_width}")
        
        if self.evaluation_distance <= 0:
            raise ValueError(f"Lane changing evaluation_distance must be positive, got {self.evaluation_distance}")


@dataclass
class RewardConfig:
    """Configuration for multi-objective reward function."""
    lane_following_weight: float = 1.0
    object_avoidance_weight: float = 0.5
    lane_change_weight: float = 0.3
    efficiency_weight: float = 0.2
    safety_penalty_weight: float = -2.0
    collision_penalty: float = -10.0
    
    def __post_init__(self):
        """Validate reward configuration parameters."""
        if self.lane_following_weight < 0:
            raise ValueError(f"Reward lane_following_weight must be non-negative, got {self.lane_following_weight}")
        
        if self.object_avoidance_weight < 0:
            raise ValueError(f"Reward object_avoidance_weight must be non-negative, got {self.object_avoidance_weight}")
        
        if self.lane_change_weight < 0:
            raise ValueError(f"Reward lane_change_weight must be non-negative, got {self.lane_change_weight}")
        
        if self.efficiency_weight < 0:
            raise ValueError(f"Reward efficiency_weight must be non-negative, got {self.efficiency_weight}")
        
        if self.safety_penalty_weight > 0:
            raise ValueError(f"Reward safety_penalty_weight must be non-positive, got {self.safety_penalty_weight}")
        
        if self.collision_penalty > 0:
            raise ValueError(f"Reward collision_penalty must be non-positive, got {self.collision_penalty}")


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""
    log_level: str = "INFO"
    log_detections: bool = True
    log_actions: bool = True
    log_rewards: bool = True
    log_performance: bool = True
    log_file_path: Optional[str] = None
    console_logging: bool = True
    
    def __post_init__(self):
        """Validate logging configuration parameters."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Logging log_level must be one of {valid_log_levels}, got {self.log_level}")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    max_fps: float = 30.0
    detection_batch_size: int = 1
    use_gpu_acceleration: bool = True
    memory_limit_gb: float = 4.0
    
    def __post_init__(self):
        """Validate performance configuration parameters."""
        if self.max_fps <= 0:
            raise ValueError(f"Performance max_fps must be positive, got {self.max_fps}")
        
        if self.detection_batch_size <= 0:
            raise ValueError(f"Performance detection_batch_size must be positive, got {self.detection_batch_size}")
        
        if self.memory_limit_gb <= 0:
            raise ValueError(f"Performance memory_limit_gb must be positive, got {self.memory_limit_gb}")


@dataclass
class EnhancedRLConfig:
    """
    Comprehensive configuration for Enhanced Duckietown RL system.
    Provides parameter validation, YAML loading, and runtime updates.
    """
    # Component configurations
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    object_avoidance: ObjectAvoidanceConfig = field(default_factory=ObjectAvoidanceConfig)
    lane_changing: LaneChangingConfig = field(default_factory=LaneChangingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Global settings
    enabled_features: List[str] = field(default_factory=lambda: ["yolo", "object_avoidance", "lane_changing"])
    debug_mode: bool = False
    
    def __post_init__(self):
        """Validate global configuration parameters."""
        valid_features = ["yolo", "object_avoidance", "lane_changing", "multi_objective_reward"]
        for feature in self.enabled_features:
            if feature not in valid_features:
                raise ValueError(f"Unknown feature '{feature}'. Valid features: {valid_features}")
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'EnhancedRLConfig':
        """
        Load configuration from YAML file with validation.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            EnhancedRLConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If configuration validation fails
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}")
        
        if config_dict is None:
            config_dict = {}
        
        # Validate against schema
        cls._validate_config_schema(config_dict)
        
        # Create nested configuration objects
        try:
            yolo_config = YOLOConfig(**config_dict.get('yolo', {}))
            object_avoidance_config = ObjectAvoidanceConfig(**config_dict.get('object_avoidance', {}))
            lane_changing_config = LaneChangingConfig(**config_dict.get('lane_changing', {}))
            reward_config = RewardConfig(**config_dict.get('reward', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            performance_config = PerformanceConfig(**config_dict.get('performance', {}))
            
            return cls(
                yolo=yolo_config,
                object_avoidance=object_avoidance_config,
                lane_changing=lane_changing_config,
                reward=reward_config,
                logging=logging_config,
                performance=performance_config,
                enabled_features=config_dict.get('enabled_features', ["yolo", "object_avoidance", "lane_changing"]),
                debug_mode=config_dict.get('debug_mode', False)
            )
        except TypeError as e:
            raise ValidationError(f"Configuration validation failed: {e}")
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path where to save YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {yaml_path}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters at runtime with validation.
        
        Args:
            updates: Dictionary of parameter updates in nested format
                    e.g., {'yolo': {'confidence_threshold': 0.7}, 'debug_mode': True}
        
        Raises:
            ValidationError: If updated parameters are invalid
        """
        # Create a copy of current config as dict
        current_dict = asdict(self)
        
        # Apply updates
        self._recursive_update(current_dict, updates)
        
        # Validate the updated configuration
        self._validate_config_schema(current_dict)
        
        # Create new config objects with updated values
        try:
            if 'yolo' in updates:
                yolo_dict = current_dict['yolo']
                self.yolo = YOLOConfig(**yolo_dict)
            
            if 'object_avoidance' in updates:
                oa_dict = current_dict['object_avoidance']
                self.object_avoidance = ObjectAvoidanceConfig(**oa_dict)
            
            if 'lane_changing' in updates:
                lc_dict = current_dict['lane_changing']
                self.lane_changing = LaneChangingConfig(**lc_dict)
            
            if 'reward' in updates:
                reward_dict = current_dict['reward']
                self.reward = RewardConfig(**reward_dict)
            
            if 'logging' in updates:
                logging_dict = current_dict['logging']
                self.logging = LoggingConfig(**logging_dict)
            
            if 'performance' in updates:
                perf_dict = current_dict['performance']
                self.performance = PerformanceConfig(**perf_dict)
            
            if 'enabled_features' in updates:
                self.enabled_features = current_dict['enabled_features']
            
            if 'debug_mode' in updates:
                self.debug_mode = current_dict['debug_mode']
                
        except (TypeError, ValueError) as e:
            raise ValidationError(f"Configuration update validation failed: {e}")
        
        logger.info("Configuration updated successfully")
    
    def get_feature_config(self, feature_name: str) -> Any:
        """
        Get configuration for a specific feature.
        
        Args:
            feature_name: Name of the feature ('yolo', 'object_avoidance', etc.)
            
        Returns:
            Configuration object for the specified feature
            
        Raises:
            ValueError: If feature name is invalid
        """
        feature_map = {
            'yolo': self.yolo,
            'object_avoidance': self.object_avoidance,
            'lane_changing': self.lane_changing,
            'reward': self.reward,
            'logging': self.logging,
            'performance': self.performance
        }
        
        if feature_name not in feature_map:
            raise ValueError(f"Unknown feature '{feature_name}'. Available: {list(feature_map.keys())}")
        
        return feature_map[feature_name]
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return feature_name in self.enabled_features
    
    @staticmethod
    def _validate_config_schema(config_dict: Dict[str, Any]) -> None:
        """
        Validate configuration dictionary against JSON schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Raises:
            ValidationError: If configuration doesn't match schema
        """
        schema = {
            "type": "object",
            "properties": {
                "yolo": {
                    "type": "object",
                    "properties": {
                        "model_path": {"type": "string"},
                        "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "device": {"type": "string", "enum": ["cuda", "cpu", "auto"]},
                        "input_size": {"type": "integer", "minimum": 32},
                        "max_detections": {"type": "integer", "minimum": 1}
                    }
                },
                "object_avoidance": {
                    "type": "object",
                    "properties": {
                        "safety_distance": {"type": "number", "minimum": 0.0},
                        "avoidance_strength": {"type": "number", "minimum": 0.0},
                        "min_clearance": {"type": "number", "minimum": 0.0},
                        "max_avoidance_angle": {"type": "number", "minimum": 0.0, "maximum": 1.57},
                        "smoothing_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                },
                "lane_changing": {
                    "type": "object",
                    "properties": {
                        "lane_change_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "safety_margin": {"type": "number", "minimum": 0.0},
                        "max_lane_change_time": {"type": "number", "minimum": 0.0},
                        "min_lane_width": {"type": "number", "minimum": 0.0},
                        "evaluation_distance": {"type": "number", "minimum": 0.0}
                    }
                },
                "reward": {
                    "type": "object",
                    "properties": {
                        "lane_following_weight": {"type": "number", "minimum": 0.0},
                        "object_avoidance_weight": {"type": "number", "minimum": 0.0},
                        "lane_change_weight": {"type": "number", "minimum": 0.0},
                        "efficiency_weight": {"type": "number", "minimum": 0.0},
                        "safety_penalty_weight": {"type": "number", "maximum": 0.0},
                        "collision_penalty": {"type": "number", "maximum": 0.0}
                    }
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                        "log_detections": {"type": "boolean"},
                        "log_actions": {"type": "boolean"},
                        "log_rewards": {"type": "boolean"},
                        "log_performance": {"type": "boolean"},
                        "log_file_path": {"type": ["string", "null"]},
                        "console_logging": {"type": "boolean"}
                    }
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "max_fps": {"type": "number", "minimum": 0.0},
                        "detection_batch_size": {"type": "integer", "minimum": 1},
                        "use_gpu_acceleration": {"type": "boolean"},
                        "memory_limit_gb": {"type": "number", "minimum": 0.0}
                    }
                },
                "enabled_features": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["yolo", "object_avoidance", "lane_changing", "multi_objective_reward"]}
                },
                "debug_mode": {"type": "boolean"}
            }
        }
        
        try:
            validate(instance=config_dict, schema=schema)
        except ValidationError as e:
            raise ValidationError(f"Configuration schema validation failed: {e.message}")
    
    @staticmethod
    def _recursive_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                EnhancedRLConfig._recursive_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Configuration utilities
def create_default_config() -> EnhancedRLConfig:
    """Create a default enhanced RL configuration."""
    return EnhancedRLConfig()


def load_enhanced_config(config_path: Union[str, Path] = None) -> EnhancedRLConfig:
    """
    Load enhanced configuration from file or create default.
    
    Args:
        config_path: Path to configuration file. If None, creates default config.
        
    Returns:
        EnhancedRLConfig instance
    """
    if config_path is None:
        logger.info("No config path provided, using default configuration")
        return create_default_config()
    
    try:
        config = EnhancedRLConfig.from_yaml(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return create_default_config()


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """
    Validate a configuration file without loading it.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        EnhancedRLConfig.from_yaml(config_path)
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False