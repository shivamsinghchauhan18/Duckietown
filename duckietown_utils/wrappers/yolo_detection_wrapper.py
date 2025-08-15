"""
YOLO Object Detection Wrapper for Duckietown RL Environment.

This module implements a gym observation wrapper that integrates YOLO v5 object detection
into the Duckietown RL environment, providing real-time object detection capabilities
with configurable confidence thresholds and safety monitoring.
"""

import logging
import time
from typing import Dict, Any, Optional, Union
import warnings

import gym
import numpy as np
from gym import spaces

from ..yolo_utils import create_yolo_inference_system, YOLOInferenceWrapper
from ..error_handling import (
    ErrorHandlingMixin, ErrorContext, ErrorSeverity, RecoveryStrategy,
    YOLOInferenceError, error_handling_context
)

logger = logging.getLogger(__name__)


class YOLOObjectDetectionWrapper(ErrorHandlingMixin, gym.ObservationWrapper):
    """
    Gym observation wrapper that adds YOLO v5 object detection to Duckietown environment.
    
    This wrapper extends the observation space to include object detection results,
    providing real-time detection of objects in the robot's field of view with
    configurable confidence thresholds and safety monitoring.
    
    The wrapper maintains compatibility with existing observation processing while
    adding comprehensive object detection information that can be used by RL agents
    for object avoidance and navigation decisions.
    """
    
    def __init__(
        self,
        env: gym.Env,
        model_path: str = "yolov5s.pt",
        confidence_threshold: float = 0.5,
        device: str = 'auto',
        max_detections: int = 10,
        safety_distance_threshold: float = 1.0,
        include_image_in_obs: bool = True,
        flatten_detections: bool = False,
        detection_timeout: float = 0.1
    ):
        """
        Initialize YOLO Object Detection Wrapper.
        
        Args:
            env: Base gym environment to wrap
            model_path: Path to YOLO model file (.pt format)
            confidence_threshold: Minimum confidence threshold for detections (0.0-1.0)
            device: Device for inference ('cpu', 'cuda', 'auto')
            max_detections: Maximum number of detections to return
            safety_distance_threshold: Distance threshold for safety-critical detection (meters)
            include_image_in_obs: Whether to include original image in observation
            flatten_detections: Whether to flatten detection data into feature vector
            detection_timeout: Maximum time allowed for detection inference (seconds)
        """
        super().__init__(env)
        
        # Store configuration
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.max_detections = max_detections
        self.safety_distance_threshold = safety_distance_threshold
        self.include_image_in_obs = include_image_in_obs
        self.flatten_detections = flatten_detections
        self.detection_timeout = detection_timeout
        
        # Initialize YOLO system
        self.yolo_system = None
        self._detection_enabled = False
        self._last_detection_result = None
        self._detection_stats = {
            'total_detections': 0,
            'failed_detections': 0,
            'timeout_detections': 0,
            'safety_critical_detections': 0
        }
        
        # Initialize error handling and safety systems
        super().__init__(env)  # Initialize ErrorHandlingMixin
        
        # Initialize YOLO inference system
        self._initialize_yolo_system()
        
        # Update observation space
        self._setup_observation_space()
        
        logger.info(f"YOLOObjectDetectionWrapper initialized with model: {model_path}")
    
    def _initialize_yolo_system(self) -> bool:
        """
        Initialize YOLO inference system with robust error handling.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        context = ErrorContext(
            component="YOLOObjectDetectionWrapper",
            operation="initialize_yolo_system",
            error_type="",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            max_retries=3
        )
        
        for attempt in range(context.max_retries):
            try:
                context.retry_count = attempt
                
                self.yolo_system = create_yolo_inference_system(
                    model_path=self.model_path,
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                    max_detections=self.max_detections
                )
                
                if self.yolo_system is not None:
                    self._detection_enabled = True
                    logger.info("YOLO system initialized successfully")
                    return True
                else:
                    raise YOLOInferenceError("YOLO system creation returned None")
                    
            except ImportError as e:
                # Missing dependencies - graceful degradation
                recovery_result = self._handle_error(e, context, None)
                logger.warning("YOLO dependencies not available - running without object detection")
                self._detection_enabled = False
                return False
                
            except FileNotFoundError as e:
                # Model file not found - try fallback or graceful degradation
                if attempt < context.max_retries - 1:
                    # Try with default model
                    if self.model_path != "yolov5s.pt":
                        logger.warning(f"Model {self.model_path} not found, trying default model")
                        self.model_path = "yolov5s.pt"
                        continue
                
                recovery_result = self._handle_error(e, context, None)
                self._detection_enabled = False
                return False
                
            except RuntimeError as e:
                # CUDA/GPU errors - try CPU fallback
                if "cuda" in str(e).lower() and self.device != 'cpu':
                    logger.warning("CUDA error detected, falling back to CPU")
                    self.device = 'cpu'
                    continue
                
                recovery_result = self._handle_error(e, context, None)
                if attempt == context.max_retries - 1:
                    self._detection_enabled = False
                    return False
                    
            except Exception as e:
                recovery_result = self._handle_error(e, context, None)
                if attempt == context.max_retries - 1:
                    logger.error(f"Failed to initialize YOLO system after {context.max_retries} attempts")
                    self._detection_enabled = False
                    return False
                    
            # Brief delay before retry
            time.sleep(0.1 * (attempt + 1))
        
        return False
    
    def _setup_observation_space(self):
        """Setup the observation space based on configuration."""
        if self.flatten_detections:
            # Create flattened observation space
            self._setup_flattened_observation_space()
        else:
            # Create dictionary observation space
            self._setup_dict_observation_space()
    
    def _setup_dict_observation_space(self):
        """Setup dictionary-based observation space."""
        obs_spaces = {}
        
        # Include original image if requested
        if self.include_image_in_obs:
            obs_spaces['image'] = self.env.observation_space
        
        # Detection results space (variable length list, represented as dict)
        obs_spaces['detections'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_detections, 9),  # [class_id, conf, x1, y1, x2, y2, rel_x, rel_y, distance]
            dtype=np.float32
        )
        
        # Detection metadata
        obs_spaces['detection_count'] = spaces.Box(
            low=0,
            high=self.max_detections,
            shape=(1,),
            dtype=np.int32
        )
        
        obs_spaces['safety_critical'] = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.int32
        )
        
        obs_spaces['inference_time'] = spaces.Box(
            low=0.0,
            high=1.0,  # Max 1 second inference time
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(obs_spaces)
    
    def _setup_flattened_observation_space(self):
        """Setup flattened observation space for compatibility with simple RL algorithms."""
        # Calculate total feature size
        original_size = np.prod(self.env.observation_space.shape) if self.include_image_in_obs else 0
        detection_features_size = self.max_detections * 9  # 9 features per detection
        metadata_size = 3  # detection_count, safety_critical, inference_time
        
        total_size = original_size + detection_features_size + metadata_size
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
    
    def observation(self, observation: np.ndarray) -> Union[Dict[str, Any], np.ndarray]:
        """
        Process observation with YOLO object detection and error recovery.
        
        Args:
            observation: Original observation from environment
            
        Returns:
            Enhanced observation with detection results
        """
        # Check if detection should be re-enabled
        self._check_detection_re_enable()
        
        # Perform object detection with error handling
        try:
            detection_result = self._detect_objects(observation)
        except Exception as e:
            # Final fallback - return empty detection result
            logger.error(f"Critical error in observation processing: {e}")
            detection_result = self._get_empty_detection_result()
        
        # Update statistics
        self._update_detection_stats(detection_result)
        
        # Format observation based on configuration with error handling
        try:
            if self.flatten_detections:
                return self._create_flattened_observation(observation, detection_result)
            else:
                return self._create_dict_observation(observation, detection_result)
        except Exception as e:
            logger.error(f"Error in observation formatting: {e}")
            # Return minimal safe observation
            return self._create_safe_fallback_observation(observation)
    
    def _check_detection_re_enable(self):
        """Check if detection should be re-enabled after temporary disable."""
        if hasattr(self, '_disable_counter') and self._disable_counter > 0:
            self._disable_counter -= 1
            if self._disable_counter == 0:
                logger.info("Re-enabling object detection after temporary disable")
                self._detection_enabled = True
    
    def _create_safe_fallback_observation(self, original_obs: np.ndarray) -> Union[Dict[str, Any], np.ndarray]:
        """
        Create a safe fallback observation when all else fails.
        
        Args:
            original_obs: Original observation
            
        Returns:
            Safe observation format
        """
        empty_result = self._get_empty_detection_result()
        
        if self.flatten_detections:
            # Return flattened observation with zeros for detection features
            if self.include_image_in_obs:
                img_features = original_obs.flatten()
            else:
                img_features = np.array([])
            
            detection_features = np.zeros(self.max_detections * 9, dtype=np.float32)
            metadata = np.array([0, 0, 0.0], dtype=np.float32)  # count, safety, time
            
            if len(img_features) > 0:
                return np.concatenate([img_features, detection_features, metadata])
            else:
                return np.concatenate([detection_features, metadata])
        else:
            # Return dict observation with safe defaults
            obs_dict = {}
            if self.include_image_in_obs:
                obs_dict['image'] = original_obs
            
            obs_dict['detections'] = np.zeros((self.max_detections, 9), dtype=np.float32)
            obs_dict['detection_count'] = np.array([0], dtype=np.int32)
            obs_dict['safety_critical'] = np.array([0], dtype=np.int32)
            obs_dict['inference_time'] = np.array([0.0], dtype=np.float32)
            
            return obs_dict
    
    def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform object detection with comprehensive error handling and recovery.
        
        Args:
            image: Input image for detection
            
        Returns:
            Detection results dictionary
        """
        if not self._detection_enabled or self.yolo_system is None:
            return self._get_empty_detection_result()
        
        # Validate input image
        if image is None or image.size == 0:
            logger.warning("Invalid input image for detection")
            self._detection_stats['failed_detections'] += 1
            return self._get_empty_detection_result()
        
        context = ErrorContext(
            component="YOLOObjectDetectionWrapper",
            operation="detect_objects",
            error_type="",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            max_retries=2
        )
        
        start_time = time.time()
        
        try:
            # Check for memory issues before inference
            if hasattr(self.yolo_system, 'check_memory_usage'):
                memory_ok = self.yolo_system.check_memory_usage()
                if not memory_ok:
                    raise MemoryError("Insufficient memory for YOLO inference")
            
            # Perform detection with timeout monitoring
            result = self.yolo_system.detect_objects(image)
            
            # Check timeout
            inference_time = time.time() - start_time
            if inference_time > self.detection_timeout:
                logger.warning(f"Detection timeout exceeded: {inference_time:.3f}s")
                self._detection_stats['timeout_detections'] += 1
                
                # If timeout is severe, disable detection temporarily
                if inference_time > self.detection_timeout * 2:
                    logger.warning("Severe timeout detected, temporarily disabling detection")
                    self._detection_enabled = False
                    # Re-enable after a few frames
                    self._schedule_detection_re_enable()
                
                return self._get_empty_detection_result()
            
            # Validate detection result
            if not self._validate_detection_result(result):
                raise YOLOInferenceError("Invalid detection result format")
            
            # Update safety distance threshold in result
            result['safety_critical'] = self._check_safety_critical(
                result.get('detections', [])
            )
            
            result['inference_time'] = inference_time
            self._last_detection_result = result
            return result
            
        except MemoryError as e:
            # Handle memory issues with graceful degradation
            recovery_result = self._handle_error(e, context, self._get_empty_detection_result())
            self._detection_stats['failed_detections'] += 1
            
            # Temporarily disable detection to free memory
            logger.warning("Memory error in detection, temporarily disabling")
            self._detection_enabled = False
            self._schedule_detection_re_enable()
            
            return recovery_result.fallback_value
            
        except RuntimeError as e:
            # Handle CUDA/inference errors
            if "cuda" in str(e).lower():
                # GPU error - try to recover or fallback to CPU
                logger.warning("CUDA error in detection, attempting recovery")
                if hasattr(self.yolo_system, 'fallback_to_cpu'):
                    self.yolo_system.fallback_to_cpu()
                
            recovery_result = self._handle_error(e, context, self._get_empty_detection_result())
            self._detection_stats['failed_detections'] += 1
            return recovery_result.fallback_value
            
        except Exception as e:
            # Handle all other errors
            recovery_result = self._handle_error(e, context, self._get_empty_detection_result())
            self._detection_stats['failed_detections'] += 1
            
            # If too many consecutive failures, disable detection temporarily
            if self._detection_stats['failed_detections'] % 10 == 0:
                logger.warning(f"High failure rate ({self._detection_stats['failed_detections']} failures), temporarily disabling detection")
                self._detection_enabled = False
                self._schedule_detection_re_enable()
            
            return recovery_result.fallback_value
    
    def _validate_detection_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate detection result format and content.
        
        Args:
            result: Detection result to validate
            
        Returns:
            bool: True if result is valid
        """
        if not isinstance(result, dict):
            return False
        
        required_keys = ['detections', 'detection_count']
        for key in required_keys:
            if key not in result:
                return False
        
        detections = result.get('detections', [])
        if not isinstance(detections, list):
            return False
        
        # Validate individual detections
        for detection in detections:
            if not isinstance(detection, dict):
                return False
            
            required_detection_keys = ['class', 'confidence', 'bbox']
            for key in required_detection_keys:
                if key not in detection:
                    return False
        
        return True
    
    def _schedule_detection_re_enable(self):
        """Schedule re-enabling of detection after temporary disable."""
        # Simple counter-based re-enable (could be enhanced with timer)
        if not hasattr(self, '_disable_counter'):
            self._disable_counter = 0
        self._disable_counter = 30  # Re-enable after 30 frames
    
    def _check_safety_critical(self, detections: list) -> bool:
        """
        Check if any detections are safety critical based on distance and position.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            bool: True if any detection is safety critical
        """
        for detection in detections:
            distance = detection.get('distance', float('inf'))
            rel_pos = detection.get('relative_position', [0, 0])
            
            # Check if object is close and in front of robot
            if (distance < self.safety_distance_threshold and 
                rel_pos[1] > -0.2):  # Allow some tolerance for objects slightly behind
                return True
        
        return False
    
    def _get_empty_detection_result(self) -> Dict[str, Any]:
        """Get empty detection result for error/timeout cases."""
        return {
            'detections': [],
            'detection_count': 0,
            'inference_time': 0.0,
            'frame_shape': None,
            'safety_critical': False
        }
    
    def _create_dict_observation(
        self, 
        original_obs: np.ndarray, 
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create dictionary-based observation.
        
        Args:
            original_obs: Original environment observation
            detection_result: YOLO detection results
            
        Returns:
            Dictionary observation with detection data
        """
        obs_dict = {}
        
        # Include original image if requested
        if self.include_image_in_obs:
            obs_dict['image'] = original_obs
        
        # Process detections into fixed-size array
        detection_array = np.zeros((self.max_detections, 9), dtype=np.float32)
        detections = detection_result.get('detections', [])
        
        for i, detection in enumerate(detections[:self.max_detections]):
            # Map class name to ID (simple hash for now)
            class_id = hash(detection.get('class', 'unknown')) % 1000
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            rel_pos = detection.get('relative_position', [0.0, 0.0])
            distance = detection.get('distance', 0.0)
            
            detection_array[i] = [
                class_id, confidence, 
                bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
                rel_pos[0], rel_pos[1], distance
            ]
        
        obs_dict['detections'] = detection_array
        obs_dict['detection_count'] = np.array([detection_result.get('detection_count', 0)], dtype=np.int32)
        obs_dict['safety_critical'] = np.array([int(detection_result.get('safety_critical', False))], dtype=np.int32)
        obs_dict['inference_time'] = np.array([detection_result.get('inference_time', 0.0)], dtype=np.float32)
        
        return obs_dict
    
    def _create_flattened_observation(
        self, 
        original_obs: np.ndarray, 
        detection_result: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create flattened observation for simple RL algorithms.
        
        Args:
            original_obs: Original environment observation
            detection_result: YOLO detection results
            
        Returns:
            Flattened numpy array with all observation data
        """
        components = []
        
        # Include flattened original image if requested
        if self.include_image_in_obs:
            components.append(original_obs.flatten())
        
        # Add detection features
        detection_features = np.zeros(self.max_detections * 9, dtype=np.float32)
        detections = detection_result.get('detections', [])
        
        for i, detection in enumerate(detections[:self.max_detections]):
            start_idx = i * 9
            class_id = hash(detection.get('class', 'unknown')) % 1000
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            rel_pos = detection.get('relative_position', [0.0, 0.0])
            distance = detection.get('distance', 0.0)
            
            detection_features[start_idx:start_idx+9] = [
                class_id, confidence,
                bbox[0], bbox[1], bbox[2], bbox[3],
                rel_pos[0], rel_pos[1], distance
            ]
        
        components.append(detection_features)
        
        # Add metadata
        metadata = np.array([
            detection_result.get('detection_count', 0),
            int(detection_result.get('safety_critical', False)),
            detection_result.get('inference_time', 0.0)
        ], dtype=np.float32)
        
        components.append(metadata)
        
        return np.concatenate(components)
    
    def _update_detection_stats(self, detection_result: Dict[str, Any]):
        """Update detection statistics."""
        self._detection_stats['total_detections'] += 1
        
        if detection_result.get('safety_critical', False):
            self._detection_stats['safety_critical_detections'] += 1
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get detection performance statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        stats = self._detection_stats.copy()
        
        if self.yolo_system is not None:
            yolo_stats = self.yolo_system.get_performance_stats()
            stats.update(yolo_stats)
        
        # Calculate success rate
        total = stats['total_detections']
        if total > 0:
            stats['success_rate'] = 1.0 - (stats['failed_detections'] + stats['timeout_detections']) / total
            stats['safety_critical_rate'] = stats['safety_critical_detections'] / total
        else:
            stats['success_rate'] = 0.0
            stats['safety_critical_rate'] = 0.0
        
        return stats
    
    def reset_detection_stats(self):
        """Reset detection statistics."""
        self._detection_stats = {
            'total_detections': 0,
            'failed_detections': 0,
            'timeout_detections': 0,
            'safety_critical_detections': 0
        }
        
        if self.yolo_system is not None:
            self.yolo_system.reset_stats()
    
    def is_detection_enabled(self) -> bool:
        """Check if object detection is enabled and working."""
        return self._detection_enabled and self.yolo_system is not None
    
    def get_last_detection_result(self) -> Optional[Dict[str, Any]]:
        """Get the last detection result for debugging."""
        return self._last_detection_result
    
    def reload_yolo_model(self) -> bool:
        """
        Reload YOLO model (useful for error recovery).
        
        Returns:
            bool: True if reload successful
        """
        logger.info("Reloading YOLO model...")
        return self._initialize_yolo_system()
    
    def reset(self, **kwargs):
        """Reset environment and clear detection history."""
        # Reset detection statistics for new episode
        self._last_detection_result = None
        
        # Reset base environment
        observation = self.env.reset(**kwargs)
        
        # Return processed observation
        return self.observation(observation)