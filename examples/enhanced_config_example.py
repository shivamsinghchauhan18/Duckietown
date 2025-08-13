#!/usr/bin/env python3
"""
Example usage of Enhanced Configuration Management System.
Demonstrates configuration loading, validation, updates, and presets.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.enhanced_config import EnhancedRLConfig, ValidationError
from config.config_utils import ConfigurationManager, apply_preset, create_config_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_basic_config_usage():
    """Demonstrate basic configuration creation and usage."""
    logger.info("=== Basic Configuration Usage ===")
    
    # Create default configuration
    config = EnhancedRLConfig()
    logger.info(f"Default YOLO confidence threshold: {config.yolo.confidence_threshold}")
    logger.info(f"Default safety distance: {config.object_avoidance.safety_distance}")
    logger.info(f"Enabled features: {config.enabled_features}")
    
    # Access feature configurations
    yolo_config = config.get_feature_config('yolo')
    logger.info(f"YOLO device: {yolo_config.device}")
    
    # Check if features are enabled
    logger.info(f"YOLO enabled: {config.is_feature_enabled('yolo')}")
    logger.info(f"Lane changing enabled: {config.is_feature_enabled('lane_changing')}")


def demonstrate_yaml_operations():
    """Demonstrate YAML loading and saving operations."""
    logger.info("\n=== YAML Operations ===")
    
    # Create a configuration and save to YAML
    config = EnhancedRLConfig()
    config.yolo.confidence_threshold = 0.7
    config.debug_mode = True
    
    yaml_path = Path("temp_config.yml")
    config.to_yaml(yaml_path)
    logger.info(f"Configuration saved to {yaml_path}")
    
    # Load configuration from YAML
    loaded_config = EnhancedRLConfig.from_yaml(yaml_path)
    logger.info(f"Loaded YOLO confidence threshold: {loaded_config.yolo.confidence_threshold}")
    logger.info(f"Loaded debug mode: {loaded_config.debug_mode}")
    
    # Clean up
    yaml_path.unlink()
    logger.info("Temporary YAML file cleaned up")


def demonstrate_configuration_manager():
    """Demonstrate configuration manager functionality."""
    logger.info("\n=== Configuration Manager ===")
    
    # Create configuration manager
    config = EnhancedRLConfig()
    manager = ConfigurationManager(config)
    
    # Update individual component configurations
    logger.info("Updating YOLO configuration...")
    manager.update_yolo_config(confidence_threshold=0.8, device="cpu")
    logger.info(f"New YOLO confidence threshold: {config.yolo.confidence_threshold}")
    
    logger.info("Updating object avoidance configuration...")
    manager.update_object_avoidance_config(safety_distance=0.7, avoidance_strength=1.2)
    logger.info(f"New safety distance: {config.object_avoidance.safety_distance}")
    
    # Feature management
    logger.info("Managing features...")
    manager.disable_feature('lane_changing')
    logger.info(f"Features after disabling lane changing: {config.enabled_features}")
    
    manager.enable_feature('lane_changing')
    logger.info(f"Features after re-enabling lane changing: {config.enabled_features}")
    
    # Batch updates
    logger.info("Performing batch update...")
    batch_updates = {
        'yolo': {'confidence_threshold': 0.9},
        'reward': {'lane_following_weight': 1.5},
        'debug_mode': True
    }
    manager.batch_update(batch_updates)
    logger.info(f"After batch update - YOLO threshold: {config.yolo.confidence_threshold}")
    logger.info(f"After batch update - Lane following weight: {config.reward.lane_following_weight}")
    
    # Export configuration summary
    summary = manager.export_config_summary()
    logger.info("Configuration summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Show update history
    history = manager.get_update_history()
    logger.info(f"Number of configuration updates: {len(history)}")
    if history:
        logger.info(f"Last update: {history[-1]['update_type']}")


def demonstrate_presets():
    """Demonstrate configuration presets."""
    logger.info("\n=== Configuration Presets ===")
    
    config = EnhancedRLConfig()
    manager = ConfigurationManager(config)
    
    # Apply development preset
    logger.info("Applying development preset...")
    apply_preset(manager, 'development')
    logger.info(f"Debug mode: {config.debug_mode}")
    logger.info(f"Log level: {config.logging.log_level}")
    logger.info(f"GPU acceleration: {config.performance.use_gpu_acceleration}")
    
    # Apply production preset
    logger.info("Applying production preset...")
    apply_preset(manager, 'production')
    logger.info(f"Debug mode: {config.debug_mode}")
    logger.info(f"Log level: {config.logging.log_level}")
    logger.info(f"GPU acceleration: {config.performance.use_gpu_acceleration}")
    
    # Apply safe driving preset
    logger.info("Applying safe driving preset...")
    apply_preset(manager, 'safe_driving')
    logger.info(f"Safety distance: {config.object_avoidance.safety_distance}")
    logger.info(f"Safety penalty weight: {config.reward.safety_penalty_weight}")
    logger.info(f"YOLO confidence threshold: {config.yolo.confidence_threshold}")


def demonstrate_validation():
    """Demonstrate configuration validation."""
    logger.info("\n=== Configuration Validation ===")
    
    config = EnhancedRLConfig()
    manager = ConfigurationManager(config)
    
    # Valid update
    try:
        manager.update_yolo_config(confidence_threshold=0.8)
        logger.info("Valid YOLO update successful")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
    
    # Invalid update
    try:
        manager.update_yolo_config(confidence_threshold=1.5)  # Invalid value
        logger.info("Invalid YOLO update successful (this shouldn't happen)")
    except ValidationError as e:
        logger.info(f"Validation correctly caught invalid value: {e}")
    
    # Invalid batch update
    try:
        invalid_updates = {
            'object_avoidance': {'safety_distance': -0.5},  # Invalid
            'yolo': {'device': 'invalid_device'}  # Invalid
        }
        manager.batch_update(invalid_updates)
        logger.info("Invalid batch update successful (this shouldn't happen)")
    except ValidationError as e:
        logger.info(f"Validation correctly caught invalid batch update: {e}")
    
    # Validate current configuration
    is_valid = manager.validate_current_config()
    logger.info(f"Current configuration is valid: {is_valid}")


def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    logger.info("\n=== Error Handling ===")
    
    # Try to load non-existent file
    try:
        config = EnhancedRLConfig.from_yaml("non_existent_config.yml")
        logger.info("Loaded non-existent file (this shouldn't happen)")
    except FileNotFoundError as e:
        logger.info(f"Correctly handled missing file: {e}")
    
    # Try to create invalid configuration
    try:
        from config.enhanced_config import YOLOConfig
        invalid_yolo = YOLOConfig(confidence_threshold=2.0)
        logger.info("Created invalid YOLO config (this shouldn't happen)")
    except ValueError as e:
        logger.info(f"Correctly caught invalid YOLO config: {e}")
    
    # Try to access invalid feature
    config = EnhancedRLConfig()
    try:
        invalid_feature = config.get_feature_config('invalid_feature')
        logger.info("Got invalid feature config (this shouldn't happen)")
    except ValueError as e:
        logger.info(f"Correctly caught invalid feature access: {e}")


def main():
    """Main demonstration function."""
    logger.info("Enhanced Configuration Management System Demo")
    logger.info("=" * 50)
    
    try:
        demonstrate_basic_config_usage()
        demonstrate_yaml_operations()
        demonstrate_configuration_manager()
        demonstrate_presets()
        demonstrate_validation()
        demonstrate_error_handling()
        
        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()