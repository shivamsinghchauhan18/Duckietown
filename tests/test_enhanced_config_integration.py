"""
Integration tests for Enhanced Configuration Management System.
Tests integration with existing Duckietown RL components.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.enhanced_config import EnhancedRLConfig
from config.config_utils import ConfigurationManager, create_config_manager


class TestEnhancedConfigIntegration(unittest.TestCase):
    """Test integration of enhanced configuration with existing system."""
    
    def test_config_manager_creation(self):
        """Test creating configuration manager."""
        manager = create_config_manager()
        self.assertIsInstance(manager, ConfigurationManager)
        self.assertIsInstance(manager.config, EnhancedRLConfig)
    
    def test_config_file_persistence(self):
        """Test configuration file persistence."""
        # Create configuration
        config = EnhancedRLConfig()
        config.yolo.confidence_threshold = 0.8
        config.debug_mode = True
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create manager with file path
            manager = ConfigurationManager(config, temp_path)
            manager.save_config()
            
            # Create new manager and reload
            new_manager = create_config_manager(temp_path)
            
            # Verify configuration was loaded correctly
            self.assertEqual(new_manager.config.yolo.confidence_threshold, 0.8)
            self.assertTrue(new_manager.config.debug_mode)
            
        finally:
            os.unlink(temp_path)
    
    def test_runtime_configuration_updates(self):
        """Test runtime configuration updates during training simulation."""
        manager = create_config_manager()
        
        # Simulate training phase adjustments
        manager.update_yolo_config(confidence_threshold=0.6)
        manager.update_object_avoidance_config(safety_distance=0.8)
        
        # Verify updates
        self.assertEqual(manager.config.yolo.confidence_threshold, 0.6)
        self.assertEqual(manager.config.object_avoidance.safety_distance, 0.8)
        
        # Simulate switching to inference mode
        manager.batch_update({
            'yolo': {'confidence_threshold': 0.7},
            'logging': {'log_level': 'WARNING'},
            'debug_mode': False
        })
        
        # Verify inference configuration
        self.assertEqual(manager.config.yolo.confidence_threshold, 0.7)
        self.assertEqual(manager.config.logging.log_level, 'WARNING')
        self.assertFalse(manager.config.debug_mode)
    
    def test_feature_toggling_during_runtime(self):
        """Test enabling/disabling features during runtime."""
        manager = create_config_manager()
        
        # Start with all features enabled
        initial_features = manager.config.enabled_features.copy()
        self.assertIn('yolo', initial_features)
        self.assertIn('object_avoidance', initial_features)
        
        # Disable YOLO for CPU-only inference
        manager.disable_feature('yolo')
        self.assertNotIn('yolo', manager.config.enabled_features)
        
        # Re-enable for GPU inference
        manager.enable_feature('yolo')
        self.assertIn('yolo', manager.config.enabled_features)
    
    def test_configuration_validation_in_pipeline(self):
        """Test configuration validation in training pipeline context."""
        manager = create_config_manager()
        
        # Test valid training configuration
        training_config = {
            'yolo': {
                'confidence_threshold': 0.5,
                'device': 'cuda'
            },
            'performance': {
                'max_fps': 30.0,
                'use_gpu_acceleration': True
            },
            'reward': {
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.5
            }
        }
        
        # Should not raise any exceptions
        manager.batch_update(training_config)
        
        # Verify configuration is valid for training
        self.assertTrue(manager.validate_current_config())
        
        # Test configuration summary for monitoring
        summary = manager.export_config_summary()
        self.assertIn('yolo_confidence_threshold', summary)
        self.assertIn('reward_weights', summary)
    
    def test_error_recovery_in_configuration(self):
        """Test error recovery when invalid configurations are provided."""
        manager = create_config_manager()
        
        # Store original configuration
        original_threshold = manager.config.yolo.confidence_threshold
        
        # Try to apply invalid configuration
        try:
            manager.update_yolo_config(confidence_threshold=2.0)  # Invalid
            self.fail("Should have raised ValidationError")
        except Exception:
            # Configuration should remain unchanged after error
            self.assertEqual(manager.config.yolo.confidence_threshold, original_threshold)
    
    def test_configuration_export_for_logging(self):
        """Test configuration export for logging and monitoring."""
        manager = create_config_manager()
        
        # Update some configurations
        manager.update_yolo_config(confidence_threshold=0.7)
        manager.update_reward_config(lane_following_weight=1.2)
        
        # Export summary for logging
        summary = manager.export_config_summary()
        
        # Verify all important parameters are included
        expected_keys = [
            'enabled_features', 'debug_mode', 'yolo_confidence_threshold',
            'safety_distance', 'lane_change_threshold', 'reward_weights',
            'log_level', 'max_fps'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Verify values are correct
        self.assertEqual(summary['yolo_confidence_threshold'], 0.7)
        self.assertEqual(summary['reward_weights']['lane_following'], 1.2)


if __name__ == '__main__':
    unittest.main()