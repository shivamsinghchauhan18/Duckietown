"""
Unit tests for the enhanced logging system.

Tests logging functionality, log format validation, and performance tracking.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from duckietown_utils.enhanced_logger import (
    EnhancedLogger, ObjectDetectionLog, ActionDecisionLog, 
    RewardComponentLog, PerformanceMetricsLog, JsonFormatter,
    get_logger, initialize_logger
)
from duckietown_utils.logging_context import (
    PerformanceTracker, log_timing, log_detection_timing,
    log_action_timing, log_reward_timing, LoggingMixin,
    create_structured_log_entry
)


class TestObjectDetectionLog(unittest.TestCase):
    """Test ObjectDetectionLog data structure."""
    
    def test_object_detection_log_creation(self):
        """Test creating ObjectDetectionLog with valid data."""
        detections = [
            {
                'class': 'duckiebot',
                'confidence': 0.85,
                'bbox': [100, 50, 200, 150],
                'distance': 1.2,
                'relative_position': [0.5, 0.0]
            }
        ]
        
        log_entry = ObjectDetectionLog(
            timestamp=time.time(),
            frame_id=42,
            detections=detections,
            processing_time_ms=25.5,
            total_objects=1,
            safety_critical=False,
            confidence_threshold=0.5
        )
        
        self.assertEqual(log_entry.frame_id, 42)
        self.assertEqual(log_entry.total_objects, 1)
        self.assertFalse(log_entry.safety_critical)
        self.assertEqual(len(log_entry.detections), 1)
        
        # Test dictionary conversion
        log_dict = log_entry.to_dict()
        self.assertIsInstance(log_dict, dict)
        self.assertEqual(log_dict['frame_id'], 42)
        self.assertEqual(log_dict['total_objects'], 1)
    
    def test_safety_critical_detection(self):
        """Test safety critical flag with close objects."""
        detections = [
            {
                'class': 'obstacle',
                'confidence': 0.9,
                'bbox': [150, 75, 250, 175],
                'distance': 0.3,  # Within safety distance
                'relative_position': [0.0, 0.0]
            }
        ]
        
        log_entry = ObjectDetectionLog(
            timestamp=time.time(),
            frame_id=1,
            detections=detections,
            processing_time_ms=30.0,
            total_objects=1,
            safety_critical=True,
            confidence_threshold=0.5
        )
        
        self.assertTrue(log_entry.safety_critical)


class TestActionDecisionLog(unittest.TestCase):
    """Test ActionDecisionLog data structure."""
    
    def test_action_decision_log_creation(self):
        """Test creating ActionDecisionLog with valid data."""
        log_entry = ActionDecisionLog(
            timestamp=time.time(),
            frame_id=10,
            original_action=[0.5, 0.0],
            modified_action=[0.3, 0.2],
            action_type='object_avoidance',
            reasoning='Avoiding obstacle detected at 0.4m distance',
            triggering_conditions={'obstacle_distance': 0.4, 'obstacle_class': 'duckiebot'},
            safety_checks={'clearance_check': True, 'collision_check': True},
            wrapper_source='ObjectAvoidanceActionWrapper'
        )
        
        self.assertEqual(log_entry.frame_id, 10)
        self.assertEqual(log_entry.action_type, 'object_avoidance')
        self.assertEqual(log_entry.original_action, [0.5, 0.0])
        self.assertEqual(log_entry.modified_action, [0.3, 0.2])
        
        # Test dictionary conversion
        log_dict = log_entry.to_dict()
        self.assertIsInstance(log_dict, dict)
        self.assertEqual(log_dict['action_type'], 'object_avoidance')
        self.assertIn('triggering_conditions', log_dict)


class TestRewardComponentLog(unittest.TestCase):
    """Test RewardComponentLog data structure."""
    
    def test_reward_component_log_creation(self):
        """Test creating RewardComponentLog with valid data."""
        reward_components = {
            'lane_following': 0.8,
            'object_avoidance': 0.2,
            'lane_changing': 0.0,
            'safety_penalty': -0.1
        }
        
        reward_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5,
            'lane_changing': 0.3,
            'safety_penalty': 2.0
        }
        
        log_entry = RewardComponentLog(
            timestamp=time.time(),
            frame_id=5,
            total_reward=0.9,
            reward_components=reward_components,
            reward_weights=reward_weights,
            episode_step=100,
            cumulative_reward=45.2
        )
        
        self.assertEqual(log_entry.frame_id, 5)
        self.assertEqual(log_entry.total_reward, 0.9)
        self.assertEqual(log_entry.episode_step, 100)
        self.assertEqual(len(log_entry.reward_components), 4)
        
        # Test dictionary conversion
        log_dict = log_entry.to_dict()
        self.assertIsInstance(log_dict, dict)
        self.assertEqual(log_dict['total_reward'], 0.9)


class TestPerformanceMetricsLog(unittest.TestCase):
    """Test PerformanceMetricsLog data structure."""
    
    def test_performance_metrics_log_creation(self):
        """Test creating PerformanceMetricsLog with valid data."""
        log_entry = PerformanceMetricsLog(
            timestamp=time.time(),
            frame_id=20,
            fps=15.5,
            detection_time_ms=35.2,
            action_processing_time_ms=5.1,
            reward_calculation_time_ms=2.3,
            total_step_time_ms=42.6,
            memory_usage_mb=512.0,
            gpu_memory_usage_mb=256.0
        )
        
        self.assertEqual(log_entry.frame_id, 20)
        self.assertEqual(log_entry.fps, 15.5)
        self.assertEqual(log_entry.detection_time_ms, 35.2)
        self.assertEqual(log_entry.memory_usage_mb, 512.0)
        
        # Test dictionary conversion
        log_dict = log_entry.to_dict()
        self.assertIsInstance(log_dict, dict)
        self.assertEqual(log_dict['fps'], 15.5)


class TestEnhancedLogger(unittest.TestCase):
    """Test EnhancedLogger functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EnhancedLogger(
            log_dir=self.temp_dir,
            log_level="DEBUG",
            console_output=False,
            file_output=True
        )
    
    def test_logger_initialization(self):
        """Test logger initialization with custom configuration."""
        self.assertEqual(str(self.logger.log_dir), self.temp_dir)
        self.assertTrue(self.logger.log_detections)
        self.assertTrue(self.logger.log_actions)
        self.assertTrue(self.logger.log_rewards)
        self.assertTrue(self.logger.log_performance)
    
    def test_object_detection_logging(self):
        """Test object detection logging functionality."""
        detections = [
            {
                'class': 'duckiebot',
                'confidence': 0.85,
                'bbox': [100, 50, 200, 150],
                'distance': 1.2,
                'relative_position': [0.5, 0.0]
            }
        ]
        
        self.logger.log_object_detection(
            frame_id=1,
            detections=detections,
            processing_time_ms=25.5,
            confidence_threshold=0.5
        )
        
        # Check that detection log file was created
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        self.assertEqual(len(detection_files), 1)
        
        # Check log content
        with open(detection_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            self.assertEqual(log_data['frame_id'], 1)
            self.assertEqual(log_data['total_objects'], 1)
            self.assertEqual(log_data['processing_time_ms'], 25.5)
            self.assertFalse(log_data['safety_critical'])
    
    def test_action_decision_logging(self):
        """Test action decision logging functionality."""
        original_action = np.array([0.5, 0.0])
        modified_action = np.array([0.3, 0.2])
        
        self.logger.log_action_decision(
            frame_id=2,
            original_action=original_action,
            modified_action=modified_action,
            action_type='object_avoidance',
            reasoning='Avoiding obstacle',
            triggering_conditions={'obstacle_distance': 0.4},
            safety_checks={'clearance_check': True},
            wrapper_source='TestWrapper'
        )
        
        # Check that action log file was created
        action_files = list(Path(self.temp_dir).glob("actions_*.jsonl"))
        self.assertEqual(len(action_files), 1)
        
        # Check log content
        with open(action_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            self.assertEqual(log_data['frame_id'], 2)
            self.assertEqual(log_data['action_type'], 'object_avoidance')
            self.assertEqual(log_data['original_action'], [0.5, 0.0])
            self.assertEqual(log_data['modified_action'], [0.3, 0.2])
    
    def test_reward_component_logging(self):
        """Test reward component logging functionality."""
        reward_components = {
            'lane_following': 0.8,
            'object_avoidance': 0.2
        }
        
        reward_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5
        }
        
        self.logger.log_reward_components(
            frame_id=3,
            total_reward=1.0,
            reward_components=reward_components,
            reward_weights=reward_weights,
            episode_step=50,
            cumulative_reward=25.5
        )
        
        # Check that reward log file was created
        reward_files = list(Path(self.temp_dir).glob("rewards_*.jsonl"))
        self.assertEqual(len(reward_files), 1)
        
        # Check log content
        with open(reward_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            self.assertEqual(log_data['frame_id'], 3)
            self.assertEqual(log_data['total_reward'], 1.0)
            self.assertEqual(log_data['episode_step'], 50)
    
    def test_performance_metrics_logging(self):
        """Test performance metrics logging functionality."""
        self.logger.log_performance_metrics(
            frame_id=4,
            detection_time_ms=30.0,
            action_processing_time_ms=5.0,
            reward_calculation_time_ms=2.0,
            memory_usage_mb=512.0
        )
        
        # Check that performance log file was created
        performance_files = list(Path(self.temp_dir).glob("performance_*.jsonl"))
        self.assertEqual(len(performance_files), 1)
        
        # Check log content
        with open(performance_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            self.assertEqual(log_data['frame_id'], 4)
            self.assertEqual(log_data['detection_time_ms'], 30.0)
            self.assertEqual(log_data['action_processing_time_ms'], 5.0)
            self.assertGreater(log_data['fps'], 0)
    
    def test_error_logging(self):
        """Test error logging functionality."""
        test_exception = ValueError("Test error")
        
        self.logger.log_error("Test error message", exception=test_exception, context="test")
        
        # Error should be logged to main log file
        main_log_files = list(Path(self.temp_dir).glob("enhanced_rl_*.log"))
        self.assertEqual(len(main_log_files), 1)
    
    def test_log_summary(self):
        """Test log summary functionality."""
        summary = self.logger.get_log_summary()
        
        self.assertIn('log_dir', summary)
        self.assertIn('features_enabled', summary)
        self.assertIn('current_fps', summary)
        self.assertIn('total_frames_processed', summary)
        
        self.assertTrue(summary['features_enabled']['detections'])
        self.assertTrue(summary['features_enabled']['actions'])


class TestJsonFormatter(unittest.TestCase):
    """Test JsonFormatter functionality."""
    
    def test_json_formatter(self):
        """Test JSON formatting of log records."""
        formatter = JsonFormatter()
        
        # Create a mock log record
        import logging
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.custom_field = "custom_value"
        record.frame_id = 42
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertEqual(log_data['level'], 'INFO')
        self.assertEqual(log_data['message'], 'Test message')
        self.assertEqual(log_data['custom_field'], 'custom_value')
        self.assertEqual(log_data['frame_id'], 42)


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.tracker = PerformanceTracker()
    
    def test_timing_operations(self):
        """Test timing operations."""
        self.tracker.start_timing("test_operation")
        time.sleep(0.01)  # Sleep for 10ms
        duration = self.tracker.end_timing("test_operation")
        
        self.assertGreater(duration, 5.0)  # Should be at least 5ms
        self.assertLess(duration, 50.0)    # Should be less than 50ms
    
    def test_frame_id_tracking(self):
        """Test frame ID tracking."""
        initial_frame = self.tracker.get_frame_id()
        
        frame1 = self.tracker.increment_frame()
        frame2 = self.tracker.increment_frame()
        
        self.assertEqual(frame1, initial_frame + 1)
        self.assertEqual(frame2, initial_frame + 2)
        self.assertEqual(self.tracker.get_frame_id(), frame2)
    
    def test_memory_usage(self):
        """Test memory usage tracking."""
        memory_info = self.tracker.get_memory_usage()
        
        self.assertIn('memory_usage_mb', memory_info)
        self.assertIn('gpu_memory_usage_mb', memory_info)
        
        if memory_info['memory_usage_mb'] is not None:
            self.assertGreater(memory_info['memory_usage_mb'], 0)


class TestLoggingContext(unittest.TestCase):
    """Test logging context managers."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EnhancedLogger(
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
    
    def test_log_timing_context(self):
        """Test log_timing context manager."""
        with log_timing("test_operation", logger=self.logger) as timing:
            time.sleep(0.01)  # Sleep for 10ms
        
        # Check that timing was logged
        main_log_files = list(Path(self.temp_dir).glob("enhanced_rl_*.log"))
        self.assertEqual(len(main_log_files), 1)
    
    def test_log_detection_timing(self):
        """Test log_detection_timing context manager."""
        with log_detection_timing(frame_id=1, logger=self.logger) as log_func:
            detections = [{'class': 'test', 'confidence': 0.9}]
            processing_time = log_func(detections)
        
        self.assertGreater(processing_time, 0)
        
        # Check that detection was logged
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        self.assertEqual(len(detection_files), 1)
    
    def test_log_action_timing(self):
        """Test log_action_timing context manager."""
        with log_action_timing(frame_id=2, wrapper_source="TestWrapper", logger=self.logger) as log_func:
            processing_time = log_func(
                original_action=[0.5, 0.0],
                modified_action=[0.3, 0.2],
                action_type="test",
                reasoning="test reasoning",
                triggering_conditions={},
                safety_checks={}
            )
        
        self.assertGreater(processing_time, 0)
        
        # Check that action was logged
        action_files = list(Path(self.temp_dir).glob("actions_*.jsonl"))
        self.assertEqual(len(action_files), 1)


class TestLoggingMixin(unittest.TestCase):
    """Test LoggingMixin functionality."""
    
    def test_logging_mixin(self):
        """Test LoggingMixin integration."""
        class TestClass(LoggingMixin):
            def __init__(self):
                super().__init__()
        
        test_obj = TestClass()
        
        # Test that logger and performance tracker are available
        self.assertIsNotNone(test_obj._logger)
        self.assertIsNotNone(test_obj._performance_tracker)
        self.assertEqual(test_obj._wrapper_name, "TestClass")
        
        # Test frame ID methods
        frame_id = test_obj._get_frame_id()
        new_frame_id = test_obj._increment_frame_id()
        self.assertEqual(new_frame_id, frame_id + 1)


class TestStructuredLogEntry(unittest.TestCase):
    """Test structured log entry creation."""
    
    def test_create_structured_log_entry(self):
        """Test creating structured log entries."""
        entry = create_structured_log_entry(
            log_type="test",
            frame_id=42,
            data={"key": "value"},
            timing_ms=25.5
        )
        
        self.assertEqual(entry['log_type'], "test")
        self.assertEqual(entry['frame_id'], 42)
        self.assertEqual(entry['key'], "value")
        self.assertEqual(entry['processing_time_ms'], 25.5)
        self.assertIn('timestamp', entry)


class TestGlobalLoggerFunctions(unittest.TestCase):
    """Test global logger functions."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return the same instance
        self.assertIs(logger1, logger2)
    
    def test_initialize_logger(self):
        """Test initialize_logger function."""
        temp_dir = tempfile.mkdtemp()
        
        logger = initialize_logger(
            log_dir=temp_dir,
            log_level="DEBUG",
            console_output=False
        )
        
        self.assertEqual(str(logger.log_dir), temp_dir)
        
        # Should be the same as get_logger() after initialization
        self.assertIs(get_logger(), logger)


if __name__ == '__main__':
    unittest.main()