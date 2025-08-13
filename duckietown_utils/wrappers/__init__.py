"""
Duckietown RL Environment Wrappers.

This module provides various gym wrappers for enhancing the Duckietown RL environment
with additional capabilities like object detection, action modification, and reward shaping.
"""

from .observation_wrappers import (
    ResizeWrapper,
    ClipImageWrapper,
    NormalizeWrapper,
    ChannelsLast2ChannelsFirstWrapper,
    ObservationBufferWrapper,
    RGB2GrayscaleWrapper,
    MotionBlurWrapper,
    RandomFrameRepeatingWrapper
)

from .yolo_detection_wrapper import YOLOObjectDetectionWrapper
from .enhanced_observation_wrapper import EnhancedObservationWrapper
from .object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper

__all__ = [
    # Observation wrappers
    'ResizeWrapper',
    'ClipImageWrapper', 
    'NormalizeWrapper',
    'ChannelsLast2ChannelsFirstWrapper',
    'ObservationBufferWrapper',
    'RGB2GrayscaleWrapper',
    'MotionBlurWrapper',
    'RandomFrameRepeatingWrapper',
    
    # YOLO detection wrapper
    'YOLOObjectDetectionWrapper',
    
    # Enhanced observation wrapper
    'EnhancedObservationWrapper',
    
    # Action wrappers
    'ObjectAvoidanceActionWrapper'
]