"""
Unit tests for Enhanced Configuration Management System.
Tests parameter validation, YAML loading, and runtime updates.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import unittest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.enhanced_config import (
    EnhancedRLConfig, YOLOConfig, ObjectAvoidanceConfig, LaneChangingConfig,
    RewardConfig, LoggingConfig, PerformanceConfig, ValidationError,
    create_default_config, load_enhanced_config, validate_config_file
)
from config.config_utils import ConfigurationManager, ConfigPresets, apply_preset


class TestYOLOConfig(unittest.TestCase):
    """Test YOLO configuration validation."""
    
    def test_valid_yolo_config(self):
        """Test valid YOLO configuration."""
        config = YOLOConfig(
            model_path="yolov5s.pt",
            confidence_threshold=0.7,
            device="cuda",
            input_size=640,
            max_detections=50
        )
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.device, "cuda")
    
    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold validation."""
        with self.assertRaises(ValueError) as context:
            YOLOConfig(confidence_threshold=1.5)
        self.assertIn("confidence_threshold must be between 0.0 and 1.0", str(context.exception))
        
        with self.assertRaises(ValueError):
            YOLOConfig(confidence_threshold=-0.1)
    
    def test_invalid_device(self):
        """Test invalid device validation."""
        with self.assertRaises(ValueError) as context:
            YOLOConfig(device="invalid")
        self.assertIn("device must be 'cuda', 'cpu', or 'auto'", str(context.exception))
    
    def test_invalid_input_size(self):
        """Test invalid input size validation."""
        with self.assertRaises(ValueError) as context:
            YOLOConfig(input_size=100)  # Not divisible by 32
        self.assertIn("input_size must be positive and divisible by 32", str(context.exception))
        
        with self.assertRaises(ValueError):
            YOLOConfig(input_size=-640)
    
    def test_invalid_max_detections(self):
        """Test invalid max detections validation."""
        with self.assertRaises(ValueError) as context:
            YOLOConfig(max_detections=0)
        self.assertIn("max_detections must be positive", str(context.exception))


class TestObjectAvoidanceConfig(unittest.TestCase):
    """Test object avoidance configuration validation."""
    
    def test_valid_object_avoidance_config(self):
        """Test valid object avoidance configuration."""
        config = ObjectAvoidanceConfig(
            safety_distance=0.8,
            avoidance_strength=1.2,
            min_clearance=0.3,
            max_avoidance_angle=0.7,
            smoothing_factor=0.9
        )
        self.assertEqual(config.safety_distance, 0.8)
        self.assertEqual(config.min_clearance, 0.3)
    
    def test_invalid_safety_distance(self):
        """Test invalid safety distance validation."""
        with self.assertRaises(ValueError) as context:
            ObjectAvoidanceConfig(safety_distance=-0.1)
        self.assertIn("safety_distance must be positive", str(context.exception))
    
    def test_min_clearance_greater_than_safety_distance(self):
        """Test min clearance greater than safety distance validation."""
        with self.assertRaises(ValueError) as context:
            ObjectAvoidanceConfig(safety_distance=0.3, min_clearance=0.5)
        self.assertIn("min_clearance (0.5) must be less than safety_distance (0.3)", str(context.exception))
    
    def test_invalid_max_avoidance_angle(self):
        """Test invalid max avoidance angle validation."""
        with self.assertRaises(ValueError) as context:
            ObjectAvoidanceConfig(max_avoidance_angle=2.0)  # > π/2
        self.assertIn("max_avoidance_angle must be between 0.0 and 1.57", str(context.exception))
    
    def test_invalid_smoothing_factor(self):
        """Test invalid smoothing factor validation."""
        with self.assertRaises(ValueError) as context:
            ObjectAvoidanceConfig(smoothing_factor=1.5)
        self.assertIn("smoothing_factor must be between 0.0 and 1.0", str(context.exception))


class TestLaneChangingConfig(unittest.TestCase):
    """Test lane changing configuration validation."""
    
    def test_valid_lane_changing_config(self):
        """Test valid lane changing configuration."""
        config = LaneChangingConfig(
            lane_change_threshold=0.4,
            safety_margin=2.5,
            max_lane_change_time=4.0,
            min_lane_width=0.5,
            evaluation_distance=6.0
        )
        self.assertEqual(config.lane_change_threshold, 0.4)
        self.assertEqual(config.safety_margin, 2.5)
    
    def test_invalid_lane_change_threshold(self):
        """Test invalid lane change threshold validation."""
        with self.assertRaises(ValueError) as context:
            LaneChangingConfig(lane_change_threshold=1.5)
        self.assertIn("lane_change_threshold must be between 0.0 and 1.0", str(context.exception))
    
    def test_invalid_safety_margin(self):
        """Test invalid safety margin validation."""
        with self.assertRaises(ValueError) as context:
            LaneChangingConfig(safety_margin=-1.0)
        self.assertIn("safety_margin must be positive", str(context.exception))


class TestRewardConfig(unittest.TestCase):
    """Test reward configuration validation."""
    
    def test_valid_reward_config(self):
        """Test valid reward configuration."""
        config = RewardConfig(
            lane_following_weight=1.5,
            object_avoidance_weight=0.8,
            lane_change_weight=0.4,
            efficiency_weight=0.3,
            safety_penalty_weight=-3.0,
            collision_penalty=-15.0
        )
        self.assertEqual(config.lane_following_weight, 1.5)
        self.assertEqual(config.safety_penalty_weight, -3.0)
    
    def test_invalid_positive_weights(self):
        """Test invalid positive weight validation."""
        with self.assertRaises(ValueError) as context:
            RewardConfig(lane_following_weight=-0.5)
        self.assertIn("lane_following_weight must be non-negative", str(context.exception))
    
    def test_invalid_penalty_weights(self):
        """Test invalid penalty weight validation."""
        with self.assertRaises(ValueError) as context:
            RewardConfig(safety_penalty_weight=1.0)
        self.assertIn("safety_penalty_weight must be non-positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            RewardConfig(collision_penalty=5.0)
        self.assertIn("collision_penalty must be non-positive", str(context.exception))


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration validation."""
    
    def test_valid_logging_config(self):
        """Test valid logging configuration."""
        config = LoggingConfig(
            log_level="DEBUG",
            log_detections=True,
            log_file_path="/tmp/test.log"
        )
        self.assertEqual(config.log_level, "DEBUG")
        self.assertTrue(config.log_detections)
    
    def test_invalid_log_level(self):
        """Test invalid log level validation."""
        with self.assertRaises(ValueError) as context:
            LoggingConfig(log_level="INVALID")
        self.assertIn("log_level must be one of", str(context.exception))


class TestPerformanceConfig(unittest.TestCase):
    """Test performance configuration validation."""
    
    def test_valid_performance_config(self):
        """Test valid performance configuration."""
        config = PerformanceConfig(
            max_fps=60.0,
            detection_batch_size=4,
            memory_limit_gb=8.0
        )
        self.assertEqual(config.max_fps, 60.0)
        self.assertEqual(config.detection_batch_size, 4)
    
    def test_invalid_max_fps(self):
        """Test invalid max FPS validation."""
        with self.assertRaises(ValueError) as context:
            PerformanceConfig(max_fps=-10.0)
        self.assertIn("max_fps must be positive", str(context.exception))
    
    def test_invalid_detection_batch_size(self):
        """Test invalid detection batch size validation."""
        with self.assertRaises(ValueError) as context:
            PerformanceConfig(detection_batch_size=0)
        self.assertIn("detection_batch_size must be positive", str(context.exception))


class TestEnhancedRLConfig(unittest.TestCase):
    """Test enhanced RL configuration."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = EnhancedRLConfig()
        self.assertIsInstance(config.yolo, YOLOConfig)
        self.assertIsInstance(config.object_avoidance, ObjectAvoidanceConfig)
        self.assertIsInstance(config.reward, RewardConfig)
        self.assertEqual(config.debug_mode, False)
    
    def test_invalid_enabled_features(self):
        """Test invalid enabled features validation."""
        with self.assertRaises(ValueError) as context:
            EnhancedRLConfig(enabled_features=["invalid_feature"])
        self.assertIn("Unknown feature 'invalid_feature'", str(context.exception))
    
    def test_yaml_loading_valid_file(self):
        """Test YAML loading with valid file."""
        config_data = {
            'yolo': {'confidence_threshold': 0.7},
            'object_avoidance': {'safety_distance': 0.8},
            'enabled_features': ['yolo', 'object_avoidance'],
            'debug_mode': True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = EnhancedRLConfig.from_yaml(temp_path)
            self.assertEqual(config.yolo.confidence_threshold, 0.7)
            self.assertEqual(config.object_avoidance.safety_distance, 0.8)
            self.assertTrue(config.debug_mode)
        finally:
            os.unlink(temp_path)
    
    def test_yaml_loading_nonexistent_file(self):
        """Test YAML loading with nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            EnhancedRLConfig.from_yaml("nonexistent.yml")
    
    def test_yaml_loading_invalid_yaml(self):
        """Test YAML loading with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                EnhancedRLConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_yaml_saving(self):
        """Test YAML saving functionality."""
        config = EnhancedRLConfig()
        config.yolo.confidence_threshold = 0.8
        config.debug_mode = True
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_yaml(temp_path)
            
            # Load and verify
            loaded_config = EnhancedRLConfig.from_yaml(temp_path)
            self.assertEqual(loaded_config.yolo.confidence_threshold, 0.8)
            self.assertTrue(loaded_config.debug_mode)
        finally:
            os.unlink(temp_path)
    
    def test_config_update(self):
        """Test configuration update functionality."""
        config = EnhancedRLConfig()
        
        updates = {
            'yolo': {'confidence_threshold': 0.9},
            'debug_mode': True,
            'enabled_features': ['yolo']
        }
        
        config.update(updates)
        
        self.assertEqual(config.yolo.confidence_threshold, 0.9)
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.enabled_features, ['yolo'])
    
    def test_config_update_invalid_values(self):
        """Test configuration update with invalid values."""
        config = EnhancedRLConfig()
        
        updates = {
            'yolo': {'confidence_threshold': 1.5}  # Invalid value
        }
        
        with self.assertRaises(ValidationError):
            config.update(updates)
    
    def test_get_feature_config(self):
        """Test getting feature configuration."""
        config = EnhancedRLConfig()
        
        yolo_config = config.get_feature_config('yolo')
        self.assertIsInstance(yolo_config, YOLOConfig)
        
        with self.assertRaises(ValueError):
            config.get_feature_config('invalid_feature')
    
    def test_is_feature_enabled(self):
        """Test feature enabled check."""
        config = EnhancedRLConfig(enabled_features=['yolo', 'object_avoidance'])
        
        self.assertTrue(config.is_feature_enabled('yolo'))
        self.assertTrue(config.is_feature_enabled('object_avoidance'))
        self.assertFalse(config.is_feature_enabled('lane_changing'))


class TestConfigurationManager(unittest.TestCase):
    """Test configuration manager functionality."""
    
    def setUp(self):
        """Set up test configuration manager."""
        self.config = EnhancedRLConfig()
        self.manager = ConfigurationManager(self.config)
    
    def test_update_yolo_config(self):
        """Test YOLO configuration update."""
        self.manager.update_yolo_config(confidence_threshold=0.8, device="cpu")
        
        self.assertEqual(self.config.yolo.confidence_threshold, 0.8)
        self.assertEqual(self.config.yolo.device, "cpu")
    
    def test_update_object_avoidance_config(self):
        """Test object avoidance configuration update."""
        self.manager.update_object_avoidance_config(safety_distance=0.7)
        
        self.assertEqual(self.config.object_avoidance.safety_distance, 0.7)
    
    def test_enable_disable_feature(self):
        """Test feature enable/disable functionality."""
        # Initially enabled features
        initial_features = self.config.enabled_features.copy()
        
        # Disable a feature
        if 'yolo' in initial_features:
            self.manager.disable_feature('yolo')
            self.assertNotIn('yolo', self.config.enabled_features)
        
        # Enable a feature
        self.manager.enable_feature('yolo')
        self.assertIn('yolo', self.config.enabled_features)
    
    def test_set_debug_mode(self):
        """Test debug mode setting."""
        self.manager.set_debug_mode(True)
        self.assertTrue(self.config.debug_mode)
        
        self.manager.set_debug_mode(False)
        self.assertFalse(self.config.debug_mode)
    
    def test_batch_update(self):
        """Test batch configuration update."""
        updates = {
            'yolo': {'confidence_threshold': 0.6},
            'object_avoidance': {'safety_distance': 0.9},
            'debug_mode': True
        }
        
        self.manager.batch_update(updates)
        
        self.assertEqual(self.config.yolo.confidence_threshold, 0.6)
        self.assertEqual(self.config.object_avoidance.safety_distance, 0.9)
        self.assertTrue(self.config.debug_mode)
    
    def test_batch_update_invalid(self):
        """Test batch update with invalid values."""
        updates = {
            'yolo': {'confidence_threshold': 2.0}  # Invalid
        }
        
        with self.assertRaises(ValidationError):
            self.manager.batch_update(updates)
    
    def test_update_history(self):
        """Test update history tracking."""
        initial_history_length = len(self.manager.get_update_history())
        
        self.manager.update_yolo_config(confidence_threshold=0.7)
        
        history = self.manager.get_update_history()
        self.assertEqual(len(history), initial_history_length + 1)
        self.assertEqual(history[-1]['update_type'], 'update_yolo')
    
    def test_export_config_summary(self):
        """Test configuration summary export."""
        summary = self.manager.export_config_summary()
        
        self.assertIn('enabled_features', summary)
        self.assertIn('debug_mode', summary)
        self.assertIn('yolo_confidence_threshold', summary)
        self.assertIn('reward_weights', summary)
    
    def test_validate_current_config(self):
        """Test current configuration validation."""
        self.assertTrue(self.manager.validate_current_config())
        
        # Manually corrupt config to test validation failure
        self.config.yolo.confidence_threshold = 2.0  # Invalid value
        # Note: This won't trigger validation until we try to create a new config
        # The validation happens during config creation, not during direct attribute assignment


class TestConfigPresets(unittest.TestCase):
    """Test configuration presets."""
    
    def test_development_preset(self):
        """Test development preset."""
        preset = ConfigPresets.get_development_preset()
        
        self.assertTrue(preset['debug_mode'])
        self.assertEqual(preset['logging']['log_level'], 'DEBUG')
        self.assertFalse(preset['performance']['use_gpu_acceleration'])
    
    def test_production_preset(self):
        """Test production preset."""
        preset = ConfigPresets.get_production_preset()
        
        self.assertFalse(preset['debug_mode'])
        self.assertEqual(preset['logging']['log_level'], 'INFO')
        self.assertTrue(preset['performance']['use_gpu_acceleration'])
    
    def test_high_performance_preset(self):
        """Test high performance preset."""
        preset = ConfigPresets.get_high_performance_preset()
        
        self.assertEqual(preset['performance']['max_fps'], 60.0)
        self.assertEqual(preset['performance']['detection_batch_size'], 4)
        self.assertEqual(preset['yolo']['input_size'], 416)
    
    def test_safe_driving_preset(self):
        """Test safe driving preset."""
        preset = ConfigPresets.get_safe_driving_preset()
        
        self.assertEqual(preset['object_avoidance']['safety_distance'], 0.8)
        self.assertEqual(preset['reward']['safety_penalty_weight'], -5.0)
        self.assertEqual(preset['yolo']['confidence_threshold'], 0.4)
    
    def test_apply_preset(self):
        """Test applying preset to configuration manager."""
        config = EnhancedRLConfig()
        manager = ConfigurationManager(config)
        
        apply_preset(manager, 'development')
        
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.logging.log_level, 'DEBUG')
    
    def test_apply_invalid_preset(self):
        """Test applying invalid preset."""
        config = EnhancedRLConfig()
        manager = ConfigurationManager(config)
        
        with self.assertRaises(ValueError):
            apply_preset(manager, 'invalid_preset')


class TestConfigUtilities(unittest.TestCase):
    """Test configuration utility functions."""
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()
        self.assertIsInstance(config, EnhancedRLConfig)
    
    def test_load_enhanced_config_no_path(self):
        """Test loading enhanced config without path."""
        config = load_enhanced_config()
        self.assertIsInstance(config, EnhancedRLConfig)
    
    def test_load_enhanced_config_with_path(self):
        """Test loading enhanced config with valid path."""
        config_data = {'debug_mode': True}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_enhanced_config(temp_path)
            self.assertTrue(config.debug_mode)
        finally:
            os.unlink(temp_path)
    
    def test_load_enhanced_config_invalid_path(self):
        """Test loading enhanced config with invalid path."""
        config = load_enhanced_config("nonexistent.yml")
        self.assertIsInstance(config, EnhancedRLConfig)  # Should return default
    
    def test_validate_config_file_valid(self):
        """Test validating valid configuration file."""
        config_data = {'debug_mode': True}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            self.assertTrue(validate_config_file(temp_path))
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_file_invalid(self):
        """Test validating invalid configuration file."""
        self.assertFalse(validate_config_file("nonexistent.yml"))


if __name__ == '__main__':
    unittest.main()