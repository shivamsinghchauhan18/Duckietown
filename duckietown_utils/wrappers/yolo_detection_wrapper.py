"""
YOLO v5 Object Detection Wrapper for Duckietown

This wrapper integrates YOLO v5 object detection into the Duckietown environment
to enable real-time object detection for object avoidance and lane changing decisions.

Authors: Generated for Dynamic Lane Changing and Object Avoidance
License: MIT
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    print("Warning: gym not available, using minimal gym interface")
    GYM_AVAILABLE = False
    
    # Minimal gym interface for compatibility
    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
        
        def sample(self):
            import numpy as np
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
    
    class spaces:
        Box = Box
    
    class Env:
        def __init__(self):
            self.observation_space = None
            self.action_space = None
        
        def reset(self):
            raise NotImplementedError
        
        def step(self, action):
            raise NotImplementedError
        
        def render(self, mode='human'):
            pass
        
        def close(self):
            pass
    
    class Wrapper(Env):
        def __init__(self, env):
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)
        
        def step(self, action):
            return self.env.step(action)
    
    class ActionWrapper(Wrapper):
        def action(self, action):
            return action
        
        def step(self, action):
            return self.env.step(self.action(action))
    
    class gym:
        spaces = spaces
        Env = Env
        Wrapper = Wrapper
        ActionWrapper = ActionWrapper
    import sys
    sys.path.append('.')

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLO/torch not available, using dummy implementation")
    YOLO_AVAILABLE = False
    YOLO = None

logger = logging.getLogger(__name__)


class YOLODetectionWrapper(gym.Wrapper):
    """
    Wrapper that adds YOLO v5 object detection capabilities to the Duckietown environment.
    
    This wrapper:
    - Performs real-time object detection on observations
    - Provides detection results in info dict
    - Maintains detection visualization capability
    - Supports configurable confidence thresholds
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        """
        Initialize YOLO Detection Wrapper
        
        Args:
            env: Gym environment to wrap
            config: Configuration dictionary with YOLO settings
        """
        super().__init__(env)
        
        # Configuration with defaults
        self.config = config or {}
        self.model_name = self.config.get('yolo_model', 'yolov5s')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.device = self.config.get('device', 'cpu')  # Use CPU by default for compatibility
        self.enable_visualization = self.config.get('enable_visualization', False)
        self.detection_classes = self.config.get('detection_classes', None)  # None means all classes
        
        # Initialize YOLO model
        self._initialize_yolo_model()
        
        # Detection storage
        self.last_detections = []
        self.detection_history = []
        self.max_history_length = self.config.get('max_history_length', 10)
        
        # Performance tracking
        self.detection_times = []
        
        logger.info(f"YOLODetectionWrapper initialized with model: {self.model_name}")
    
    def _initialize_yolo_model(self):
        """Initialize the YOLO model"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available, using dummy model")
            self.model = None
            return
            
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"YOLO model {self.model_name} loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Use a dummy model for testing if YOLO fails to load
            self.model = None
    
    def reset(self, **kwargs):
        """Reset environment and clear detection history"""
        obs = self.env.reset(**kwargs)
        self.last_detections = []
        self.detection_history = []
        self.detection_times = []
        return obs
    
    def step(self, action):
        """Step environment and perform object detection"""
        obs, reward, done, info = self.env.step(action)
        
        # Perform object detection on the observation
        detections = self._detect_objects(obs)
        
        # Store detections in info
        info['yolo_detections'] = {
            'detections': detections,
            'detection_count': len(detections),
            'last_detection_time': self.detection_times[-1] if self.detection_times else 0,
            'detection_history': self.detection_history.copy()
        }
        
        # Store for future reference
        self.last_detections = detections
        self._update_detection_history(detections)
        
        return obs, reward, done, info
    
    def _detect_objects(self, observation: np.ndarray) -> List[Dict]:
        """
        Perform object detection on the observation
        
        Args:
            observation: RGB image observation from environment
            
        Returns:
            List of detection dictionaries with bbox, confidence, class info
        """
        if self.model is None:
            return []
        
        try:
            import time
            start_time = time.time()
            
            # Prepare image for YOLO (expects RGB)
            if len(observation.shape) == 4:
                # Handle stacked frames - use the last frame
                image = observation[..., -3:]
            else:
                image = observation
            
            # Convert to uint8 if necessary
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Ensure RGB format
            if image.shape[-1] == 3:
                # Already RGB, just make sure it's in correct format
                detection_image = image
            else:
                # Handle grayscale or other formats
                detection_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run YOLO detection
            results = self.model(detection_image, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    classes = result.boxes.cls.cpu().numpy()  # Class indices
                    
                    for i in range(len(boxes)):
                        class_id = int(classes[i])
                        confidence = float(confidences[i])
                        bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                        
                        # Filter by detection classes if specified
                        if self.detection_classes is None or class_id in self.detection_classes:
                            detection = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.model.names[class_id] if hasattr(self.model, 'names') else f'class_{class_id}',
                                'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            }
                            detections.append(detection)
            
            # Track detection time
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            if len(self.detection_times) > self.max_history_length:
                self.detection_times.pop(0)
                
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def _update_detection_history(self, detections: List[Dict]):
        """Update detection history for temporal analysis"""
        self.detection_history.append({
            'timestamp': len(self.detection_history),
            'detections': detections.copy(),
            'detection_count': len(detections)
        })
        
        # Limit history length
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)
    
    def get_detection_summary(self) -> Dict:
        """Get summary of recent detections"""
        if not self.detection_history:
            return {'total_detections': 0, 'avg_detections_per_frame': 0, 'class_distribution': {}}
        
        total_detections = sum(frame['detection_count'] for frame in self.detection_history)
        avg_detections = total_detections / len(self.detection_history)
        
        # Class distribution
        class_counts = {}
        for frame in self.detection_history:
            for detection in frame['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections,
            'class_distribution': class_counts,
            'frames_analyzed': len(self.detection_history),
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0
        }
    
    def render_detections(self, observation: np.ndarray, detections: List[Dict] = None) -> np.ndarray:
        """
        Render detection bounding boxes on the observation
        
        Args:
            observation: RGB image observation
            detections: List of detection dicts (uses last_detections if None)
            
        Returns:
            Image with detection visualizations
        """
        if detections is None:
            detections = self.last_detections
        
        if not detections:
            return observation
        
        # Create a copy to avoid modifying original
        rendered_image = observation.copy()
        if rendered_image.dtype != np.uint8:
            rendered_image = (rendered_image * 255).astype(np.uint8)
        
        # Handle stacked frames
        if len(rendered_image.shape) == 4:
            rendered_image = rendered_image[..., -3:]
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(rendered_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return rendered_image


class YOLOObjectAvoidanceWrapper(YOLODetectionWrapper):
    """
    Extended YOLO wrapper that adds object avoidance logic and lane changing decisions
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        super().__init__(env, config)
        
        # Object avoidance parameters
        self.avoidance_distance_threshold = self.config.get('avoidance_distance_threshold', 0.3)
        self.critical_classes = self.config.get('critical_classes', ['person', 'car', 'truck', 'bus', 'bicycle'])
        self.lane_change_trigger_threshold = self.config.get('lane_change_trigger_threshold', 0.5)
        
        # State tracking
        self.avoidance_active = False
        self.lane_change_recommended = False
        self.last_avoidance_decision = None
    
    def step(self, action):
        """Step with object avoidance analysis"""
        obs, reward, done, info = super().step(action)
        
        # Analyze detections for avoidance decisions
        avoidance_info = self._analyze_for_avoidance(info['yolo_detections']['detections'])
        info['object_avoidance'] = avoidance_info
        
        return obs, reward, done, info
    
    def _analyze_for_avoidance(self, detections: List[Dict]) -> Dict:
        """
        Analyze detections to determine if avoidance or lane changing is needed
        
        Args:
            detections: List of current detections
            
        Returns:
            Dictionary with avoidance recommendations
        """
        critical_objects = []
        lane_change_needed = False
        
        # Filter for critical objects
        for detection in detections:
            if detection['class_name'] in self.critical_classes:
                # Calculate relative position and size
                center_x, center_y = detection['center']
                bbox_area = detection['area']
                
                # Estimate distance based on object size (larger = closer)
                # This is a simple heuristic, could be improved with depth estimation
                estimated_distance = 1.0 / (1.0 + bbox_area / 10000)  # Rough estimation
                
                if estimated_distance < self.avoidance_distance_threshold:
                    critical_objects.append({
                        'detection': detection,
                        'estimated_distance': estimated_distance,
                        'position': 'center' if 0.3 < center_x / 640 < 0.7 else 
                                   'left' if center_x / 640 < 0.3 else 'right'
                    })
        
        # Determine if lane change is recommended
        if critical_objects:
            # Check if objects are in the center path
            center_objects = [obj for obj in critical_objects if obj['position'] == 'center']
            if center_objects:
                lane_change_needed = True
                self.lane_change_recommended = True
            
            self.avoidance_active = True
        else:
            self.avoidance_active = False
            self.lane_change_recommended = False
        
        avoidance_decision = {
            'avoidance_active': self.avoidance_active,
            'lane_change_recommended': self.lane_change_recommended,
            'critical_objects': critical_objects,
            'critical_object_count': len(critical_objects),
            'closest_object_distance': min([obj['estimated_distance'] for obj in critical_objects]) if critical_objects else float('inf')
        }
        
        self.last_avoidance_decision = avoidance_decision
        return avoidance_decision