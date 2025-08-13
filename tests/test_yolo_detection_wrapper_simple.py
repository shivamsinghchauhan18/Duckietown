"""
Simplified tests for YOLO Object Detection Wrapper core functionality.
Tests the wrapper logic without requiring gym dependency.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestYOLODetectionWrapperCore(unittest.TestCase):
    """Test core functionality of YOLO Detection Wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock gym and spaces to avoid import errors
        self.gym_mock = Mock()
        self.spaces_mock = Mock()
        
        # Create mock classes
        self.gym_mock.ObservationWrapper = object
        self.gym_mock.Env = object
        self.spaces_mock.Box = Mock()
        self.spaces_mock.Dict = Mock()
        
        # Patch imports
        sys.modules['gym'] = self.gym_mock
        sys.modules['gym.spaces'] = self.spaces_mock
        
        # Now import the wrapper
        from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
        self.YOLOObjectDetectionWrapper = YOLOObjectDetectionWrapper
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove mocked modules
        if 'gym' in sys.modules:
            del sys.modules['gym']
        if 'gym.spaces' in sys.modules:
            del sys.modules['gym.spaces']
    
    @patch('duckietown_utils.wrappers.yolo_detection_wrapper.create_yolo_inference_system')
    def test_safety_critical_detection_logic(self, mock_create_yolo):
        """Test safety critical detection logic."""
        # Create mock environment
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (480, 640, 3)
        
        # Create mock YOLO system
        mock_yolo_system = Mock()
        mock_create_yolo.return_value = mock_yolo_system
        
        # Create wrapper instance
        wrapper = self.YOLOObjectDetectionWrapper(mock_env)
        
        # Test safety critical detection (close and in front)
        detections_critical = [
            {
                'distance': 0.5,  # Close
                'relative_position': [0.0, 0.1]  # In front
            }
        ]
        
        result = wrapper._check_safety_critical(detections_critical)
        self.assertTrue(result)
        
        # Test non-critical detection (far away)
        detections_safe = [
            {
                'distance': 2.0,  # Far
                'relative_position': [0.0, 0.1]  # In front
            }
        ]
        
        result = wrapper._check_safety_critical(detections_safe)
        self.assertFalse(result)
        
        # Test non-critical detection (behind robot)
        detections_behind = [
            {
                'distance': 0.3,  # Close
                'relative_position': [0.0, -0.5]  # Behind
            }
        ]
        
        result = wrapper._check_safety_critical(detections_behind)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()