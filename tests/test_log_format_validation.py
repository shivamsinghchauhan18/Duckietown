"""
Unit tests for log format validation and schema compliance.

Tests that logged data conforms to expected schemas and formats.
"""

import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError

from duckietown_utils.enhanced_logger import EnhancedLogger


class TestLogFormatValidation(unittest.TestCase):
    """Test log format validation and schema compliance."""
    
    def setUp(self):
        """Set up test environment with schemas."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EnhancedLogger(
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
        
        # Define JSON schemas for validation
        self.detection_schema = {
            "type": "object",
            "required": ["timestamp", "frame_id", "detections", "processing_time_ms", 
                        "total_objects", "safety_critical", "confidence_threshold"],
            "properties": {
                "timestamp": {"type": "number"},
                "frame_id": {"type": "integer", "minimum": 0},
                "detections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["class", "confidence", "bbox"],
                        "properties": {
                            "class": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "distance": {"type": "number", "minimum": 0},
                            "relative_position": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        }
                    }
                },
                "processing_time_ms": {"type": "number", "minimum": 0},
                "total_objects": {"type": "integer", "minimum": 0},
                "safety_critical": {"type": "boolean"},
                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        
        self.action_schema = {
            "type": "object",
            "required": ["timestamp", "frame_id", "original_action", "modified_action",
                        "action_type", "reasoning", "triggering_conditions", 
                        "safety_checks", "wrapper_source"],
            "properties": {
                "timestamp": {"type": "number"},
                "frame_id": {"type": "integer", "minimum": 0},
                "original_action": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "modified_action": {
                    "type": "array", 
                    "items": {"type": "number"}
                },
                "action_type": {"type": "string"},
                "reasoning": {"type": "string"},
                "triggering_conditions": {"type": "object"},
                "safety_checks": {"type": "object"},
                "wrapper_source": {"type": "string"}
            }
        }  
      
        self.reward_schema = {
            "type": "object",
            "required": ["timestamp", "frame_id", "total_reward", "reward_components",
                        "reward_weights", "episode_step", "cumulative_reward"],
            "properties": {
                "timestamp": {"type": "number"},
                "frame_id": {"type": "integer", "minimum": 0},
                "total_reward": {"type": "number"},
                "reward_components": {"type": "object"},
                "reward_weights": {"type": "object"},
                "episode_step": {"type": "integer", "minimum": 0},
                "cumulative_reward": {"type": "number"}
            }
        }
        
        self.performance_schema = {
            "type": "object",
            "required": ["timestamp", "frame_id", "fps", "detection_time_ms",
                        "action_processing_time_ms", "reward_calculation_time_ms",
                        "total_step_time_ms"],
            "properties": {
                "timestamp": {"type": "number"},
                "frame_id": {"type": "integer", "minimum": 0},
                "fps": {"type": "number", "minimum": 0},
                "detection_time_ms": {"type": "number", "minimum": 0},
                "action_processing_time_ms": {"type": "number", "minimum": 0},
                "reward_calculation_time_ms": {"type": "number", "minimum": 0},
                "total_step_time_ms": {"type": "number", "minimum": 0},
                "memory_usage_mb": {"type": ["number", "null"], "minimum": 0},
                "gpu_memory_usage_mb": {"type": ["number", "null"], "minimum": 0}
            }
        }
    
    def test_detection_log_format_validation(self):
        """Test that detection logs conform to expected schema."""
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
        
        # Read and validate log file
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        self.assertEqual(len(detection_files), 1)
        
        with open(detection_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            # Validate against schema
            try:
                validate(instance=log_data, schema=self.detection_schema)
            except ValidationError as e:
                self.fail(f"Detection log format validation failed: {e}")
    
    def test_action_log_format_validation(self):
        """Test that action logs conform to expected schema."""
        self.logger.log_action_decision(
            frame_id=2,
            original_action=[0.5, 0.0],
            modified_action=[0.3, 0.2],
            action_type='object_avoidance',
            reasoning='Avoiding obstacle detected at 0.4m distance',
            triggering_conditions={'obstacle_distance': 0.4, 'obstacle_class': 'duckiebot'},
            safety_checks={'clearance_check': True, 'collision_check': True},
            wrapper_source='ObjectAvoidanceActionWrapper'
        )
        
        # Read and validate log file
        action_files = list(Path(self.temp_dir).glob("actions_*.jsonl"))
        self.assertEqual(len(action_files), 1)
        
        with open(action_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            # Validate against schema
            try:
                validate(instance=log_data, schema=self.action_schema)
            except ValidationError as e:
                self.fail(f"Action log format validation failed: {e}")
    
    def test_reward_log_format_validation(self):
        """Test that reward logs conform to expected schema."""
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
        
        self.logger.log_reward_components(
            frame_id=3,
            total_reward=0.9,
            reward_components=reward_components,
            reward_weights=reward_weights,
            episode_step=100,
            cumulative_reward=45.2
        )
        
        # Read and validate log file
        reward_files = list(Path(self.temp_dir).glob("rewards_*.jsonl"))
        self.assertEqual(len(reward_files), 1)
        
        with open(reward_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            # Validate against schema
            try:
                validate(instance=log_data, schema=self.reward_schema)
            except ValidationError as e:
                self.fail(f"Reward log format validation failed: {e}")
    
    def test_performance_log_format_validation(self):
        """Test that performance logs conform to expected schema."""
        self.logger.log_performance_metrics(
            frame_id=4,
            detection_time_ms=30.0,
            action_processing_time_ms=5.0,
            reward_calculation_time_ms=2.0,
            memory_usage_mb=512.0,
            gpu_memory_usage_mb=256.0
        )
        
        # Read and validate log file
        performance_files = list(Path(self.temp_dir).glob("performance_*.jsonl"))
        self.assertEqual(len(performance_files), 1)
        
        with open(performance_files[0], 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            # Validate against schema
            try:
                validate(instance=log_data, schema=self.performance_schema)
            except ValidationError as e:
                self.fail(f"Performance log format validation failed: {e}")
    
    def test_multiple_log_entries_validation(self):
        """Test validation of multiple log entries in sequence."""
        # Log multiple detection entries
        for i in range(3):
            detections = [
                {
                    'class': f'object_{i}',
                    'confidence': 0.7 + i * 0.1,
                    'bbox': [100 + i*10, 50, 200 + i*10, 150],
                    'distance': 1.0 + i * 0.2,
                    'relative_position': [0.5, i * 0.1]
                }
            ]
            
            self.logger.log_object_detection(
                frame_id=i,
                detections=detections,
                processing_time_ms=20.0 + i * 5,
                confidence_threshold=0.5
            )
        
        # Validate all entries
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        self.assertEqual(len(detection_files), 1)
        
        with open(detection_files[0], 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            
            for i, line in enumerate(lines):
                log_data = json.loads(line.strip())
                
                # Validate against schema
                try:
                    validate(instance=log_data, schema=self.detection_schema)
                except ValidationError as e:
                    self.fail(f"Detection log entry {i} validation failed: {e}")
                
                # Validate specific values
                self.assertEqual(log_data['frame_id'], i)
                self.assertEqual(log_data['detections'][0]['class'], f'object_{i}')
    
    def test_invalid_log_data_handling(self):
        """Test handling of invalid log data."""
        # Test with invalid detection data (missing required fields)
        invalid_detections = [
            {
                'class': 'test',
                # Missing confidence and bbox
            }
        ]
        
        # This should not crash but should still log the data
        self.logger.log_object_detection(
            frame_id=999,
            detections=invalid_detections,
            processing_time_ms=10.0,
            confidence_threshold=0.5
        )
        
        # The log should still be written (logger is permissive)
        # but validation would fail if we tried to validate it
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        if detection_files:
            with open(detection_files[0], 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    log_data = json.loads(last_line)
                    
                    # This should fail validation due to missing fields in detection
                    with self.assertRaises(ValidationError):
                        validate(instance=log_data, schema=self.detection_schema)
    
    def test_timestamp_format_validation(self):
        """Test that timestamps are in correct format."""
        self.logger.log_object_detection(
            frame_id=1,
            detections=[],
            processing_time_ms=10.0,
            confidence_threshold=0.5
        )
        
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        with open(detection_files[0], 'r') as f:
            log_data = json.loads(f.readline().strip())
            
            timestamp = log_data['timestamp']
            self.assertIsInstance(timestamp, (int, float))
            
            # Timestamp should be reasonable (within last hour and next hour)
            import time
            current_time = time.time()
            self.assertGreater(timestamp, current_time - 3600)  # Not more than 1 hour ago
            self.assertLess(timestamp, current_time + 3600)     # Not more than 1 hour in future
    
    def test_jsonl_format_compliance(self):
        """Test that log files are valid JSONL format."""
        # Log multiple entries
        for i in range(5):
            self.logger.log_object_detection(
                frame_id=i,
                detections=[],
                processing_time_ms=10.0,
                confidence_threshold=0.5
            )
        
        detection_files = list(Path(self.temp_dir).glob("detections_*.jsonl"))
        with open(detection_files[0], 'r') as f:
            lines = f.readlines()
            
            # Each line should be valid JSON
            for i, line in enumerate(lines):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        self.fail(f"Line {i+1} is not valid JSON: {e}")
            
            # Should have 5 lines
            non_empty_lines = [line for line in lines if line.strip()]
            self.assertEqual(len(non_empty_lines), 5)


if __name__ == '__main__':
    unittest.main()