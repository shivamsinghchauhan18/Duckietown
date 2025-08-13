"""
Enhanced Observation Wrapper for Duckietown RL Environment.

This module implements a gym observation wrapper that combines YOLO object detection
results with traditional observations, creating a unified observation space suitable
for PPO training with enhanced autonomous driving capabilities.
"""

import logging
from typing import Dict, Any, Union, Optional, Tuple
import warnings

import gym
import numpy as np
from gym import spaces

logger = logging.getLogger(__name__)


class EnhancedObservationWrapper(gym.ObservationWrapper):
    """
    Gym observation wrapper that combines object detection data with traditional observations.
    
    This wrapper processes observations from the YOLO Object Detection Wrapper and creates
    a unified observation space that includes both visual features and structured detection
    information. It provides configurable feature extraction and normalization to ensure
    compatibility with PPO training requirements.
    
    The wrapper can operate in two modes:
    1. Flattened mode: Creates a single feature vector combining all information
    2. Dictionary mode: Maintains structured observation with separate components
    """
    
    def __init__(
        self,
        env: gym.Env,
        include_detection_features: bool = True,
        include_image_features: bool = True,
        max_detections: int = 10,
        detection_feature_size: int = 9,
        image_feature_method: str = 'flatten',  # 'flatten', 'encode', 'none'
        normalize_features: bool = True,
        feature_scaling_method: str = 'minmax',  # 'minmax', 'standard', 'none'
        output_mode: str = 'flattened',  # 'flattened', 'dict'
        safety_feature_weight: float = 2.0,
        distance_normalization_factor: float = 10.0
    ):
        """
        Initialize Enhanced Observation Wrapper.
        
        Args:
            env: Base gym environment to wrap (should include YOLO detection wrapper)
            include_detection_features: Whether to include object detection features
            include_image_features: Whether to include image-based features
            max_detections: Maximum number of detections to process
            detection_feature_size: Number of features per detection
            image_feature_method: Method for processing image features
            normalize_features: Whether to normalize feature values
            feature_scaling_method: Method for feature scaling
            output_mode: Output format ('flattened' or 'dict')
            safety_feature_weight: Weight multiplier for safety-critical features
            distance_normalization_factor: Factor for normalizing distance values
        """
        super().__init__(env)
        
        # Store configuration
        self.include_detection_features = include_detection_features
        self.include_image_features = include_image_features
        self.max_detections = max_detections
        self.detection_feature_size = detection_feature_size
        self.image_feature_method = image_feature_method
        self.normalize_features = normalize_features
        self.feature_scaling_method = feature_scaling_method
        self.output_mode = output_mode
        self.safety_feature_weight = safety_feature_weight
        self.distance_normalization_factor = distance_normalization_factor
        
        # Feature statistics for normalization
        self._feature_stats = {
            'detection_min': np.zeros(max_detections * detection_feature_size),
            'detection_max': np.ones(max_detections * detection_feature_size),
            'detection_mean': np.zeros(max_detections * detection_feature_size),
            'detection_std': np.ones(max_detections * detection_feature_size),
            'image_min': 0.0,
            'image_max': 1.0,
            'image_mean': 0.5,
            'image_std': 0.25
        }
        
        # Initialize feature statistics
        self._initialize_feature_stats()
        
        # Setup observation space
        self._setup_observation_space()
        
        # Performance tracking
        self._processing_stats = {
            'total_observations': 0,
            'feature_extraction_time': 0.0,
            'normalization_time': 0.0
        }
        
        logger.info(f"EnhancedObservationWrapper initialized with mode: {output_mode}")
    
    def _initialize_feature_stats(self):
        """Initialize feature statistics for normalization."""
        # Detection feature ranges (based on expected YOLO output)
        detection_ranges = {
            'class_id': (0, 1000),      # Hash-based class IDs
            'confidence': (0.0, 1.0),   # Confidence scores
            'bbox_x1': (0, 640),        # Bounding box coordinates
            'bbox_y1': (0, 480),
            'bbox_x2': (0, 640),
            'bbox_y2': (0, 480),
            'rel_x': (-2.0, 2.0),       # Relative position
            'rel_y': (-2.0, 2.0),
            'distance': (0.0, 10.0)     # Distance in meters
        }
        
        # Set up detection feature statistics
        for i in range(self.max_detections):
            for j, (feature_name, (min_val, max_val)) in enumerate(detection_ranges.items()):
                idx = i * self.detection_feature_size + j
                if idx < len(self._feature_stats['detection_min']):
                    self._feature_stats['detection_min'][idx] = min_val
                    self._feature_stats['detection_max'][idx] = max_val
                    self._feature_stats['detection_mean'][idx] = (min_val + max_val) / 2
                    self._feature_stats['detection_std'][idx] = (max_val - min_val) / 4
    
    def _setup_observation_space(self):
        """Setup the observation space based on configuration."""
        if self.output_mode == 'flattened':
            self._setup_flattened_observation_space()
        else:
            self._setup_dict_observation_space()
    
    def _setup_flattened_observation_space(self):
        """Setup flattened observation space."""
        total_features = 0
        
        # Image features
        if self.include_image_features:
            if self.image_feature_method == 'flatten':
                # Assume input is already processed by previous wrappers
                if hasattr(self.env.observation_space, 'shape'):
                    if isinstance(self.env.observation_space, spaces.Dict):
                        # Handle dict observation space from YOLO wrapper
                        if 'image' in self.env.observation_space.spaces:
                            image_shape = self.env.observation_space.spaces['image'].shape
                        else:
                            # Fallback to a reasonable default
                            image_shape = (120, 160, 3)
                    else:
                        image_shape = self.env.observation_space.shape
                    total_features += np.prod(image_shape)
                else:
                    # Default image feature size
                    total_features += 120 * 160 * 3
            elif self.image_feature_method == 'encode':
                # Encoded image features (e.g., from CNN)
                total_features += 512  # Typical CNN feature size
        
        # Detection features
        if self.include_detection_features:
            total_features += self.max_detections * self.detection_feature_size
        
        # Safety and metadata features
        total_features += 5  # detection_count, safety_critical, inference_time, avg_distance, closest_distance
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )
        
        logger.info(f"Flattened observation space size: {total_features}")
    
    def _setup_dict_observation_space(self):
        """Setup dictionary observation space."""
        obs_spaces = {}
        
        # Image features
        if self.include_image_features:
            if isinstance(self.env.observation_space, spaces.Dict):
                if 'image' in self.env.observation_space.spaces:
                    obs_spaces['image'] = self.env.observation_space.spaces['image']
            else:
                obs_spaces['image'] = self.env.observation_space
        
        # Detection features
        if self.include_detection_features:
            obs_spaces['detection_features'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_detections * self.detection_feature_size,),
                dtype=np.float32
            )
        
        # Safety and metadata features
        obs_spaces['safety_features'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),  # detection_count, safety_critical, inference_time, avg_distance, closest_distance
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(obs_spaces)
    
    def observation(self, observation: Union[Dict[str, Any], np.ndarray]) -> Union[Dict[str, Any], np.ndarray]:
        """
        Process observation with enhanced feature extraction.
        
        Args:
            observation: Input observation (from YOLO wrapper or base environment)
            
        Returns:
            Enhanced observation with combined features
        """
        import time
        start_time = time.time()
        
        # Handle different input formats
        if isinstance(observation, dict):
            processed_obs = self._process_dict_observation(observation)
        else:
            # Handle raw image observation (no YOLO wrapper)
            processed_obs = self._process_image_observation(observation)
        
        # Update processing statistics
        self._processing_stats['total_observations'] += 1
        self._processing_stats['feature_extraction_time'] += time.time() - start_time
        
        return processed_obs
    
    def _process_dict_observation(self, observation: Dict[str, Any]) -> Union[Dict[str, Any], np.ndarray]:
        """
        Process dictionary observation from YOLO wrapper.
        
        Args:
            observation: Dictionary observation with detection results
            
        Returns:
            Processed observation
        """
        # Extract components
        image = observation.get('image', None)
        detections = observation.get('detections', np.zeros((self.max_detections, self.detection_feature_size)))
        detection_count = observation.get('detection_count', np.array([0]))[0]
        safety_critical = observation.get('safety_critical', np.array([0]))[0]
        inference_time = observation.get('inference_time', np.array([0.0]))[0]
        
        # Extract image features
        image_features = self._extract_image_features(image) if self.include_image_features else None
        
        # Extract detection features
        detection_features = self._extract_detection_features(detections) if self.include_detection_features else None
        
        # Extract safety and metadata features
        safety_features = self._extract_safety_features(
            detections, detection_count, safety_critical, inference_time
        )
        
        # Combine features based on output mode
        if self.output_mode == 'flattened':
            return self._create_flattened_observation(image_features, detection_features, safety_features)
        else:
            return self._create_dict_observation(image, image_features, detection_features, safety_features)
    
    def _process_image_observation(self, observation: np.ndarray) -> Union[Dict[str, Any], np.ndarray]:
        """
        Process raw image observation (fallback when no YOLO wrapper).
        
        Args:
            observation: Raw image observation
            
        Returns:
            Processed observation with empty detection features
        """
        # Extract image features
        image_features = self._extract_image_features(observation) if self.include_image_features else None
        
        # Create empty detection features
        detection_features = np.zeros(self.max_detections * self.detection_feature_size) if self.include_detection_features else None
        
        # Create empty safety features
        safety_features = np.zeros(5)  # All zeros for no detections
        
        # Combine features
        if self.output_mode == 'flattened':
            return self._create_flattened_observation(image_features, detection_features, safety_features)
        else:
            return self._create_dict_observation(observation, image_features, detection_features, safety_features)
    
    def _extract_image_features(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract features from image data.
        
        Args:
            image: Input image array
            
        Returns:
            Extracted image features or None
        """
        if image is None or not self.include_image_features:
            return None
        
        if self.image_feature_method == 'flatten':
            features = image.flatten().astype(np.float32)
        elif self.image_feature_method == 'encode':
            # Placeholder for CNN encoding - would need actual CNN model
            features = self._simple_image_encoding(image)
        else:
            return None
        
        # Normalize if requested
        if self.normalize_features:
            features = self._normalize_image_features(features)
        
        return features
    
    def _simple_image_encoding(self, image: np.ndarray) -> np.ndarray:
        """
        Simple image encoding using statistical features.
        
        Args:
            image: Input image
            
        Returns:
            Encoded feature vector
        """
        # Extract simple statistical features
        features = []
        
        # Global statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image)
        ])
        
        # Per-channel statistics
        if len(image.shape) == 3:
            for channel in range(image.shape[2]):
                channel_data = image[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data)
                ])
        
        # Spatial features (simple edge detection)
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        grad_x = np.abs(np.diff(gray, axis=1)).mean()
        grad_y = np.abs(np.diff(gray, axis=0)).mean()
        features.extend([grad_x, grad_y])
        
        # Pad to fixed size (512 features)
        features = np.array(features, dtype=np.float32)
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), mode='constant')
        else:
            features = features[:512]
        
        return features
    
    def _extract_detection_features(self, detections: np.ndarray) -> np.ndarray:
        """
        Extract and process detection features.
        
        Args:
            detections: Detection array from YOLO wrapper
            
        Returns:
            Processed detection features
        """
        # Flatten detection array
        features = detections.flatten().astype(np.float32)
        
        # Ensure correct size
        expected_size = self.max_detections * self.detection_feature_size
        if len(features) < expected_size:
            features = np.pad(features, (0, expected_size - len(features)), mode='constant')
        elif len(features) > expected_size:
            features = features[:expected_size]
        
        # Normalize if requested
        if self.normalize_features:
            features = self._normalize_detection_features(features)
        
        return features
    
    def _extract_safety_features(
        self, 
        detections: np.ndarray, 
        detection_count: int, 
        safety_critical: int, 
        inference_time: float
    ) -> np.ndarray:
        """
        Extract safety and metadata features.
        
        Args:
            detections: Detection array
            detection_count: Number of detections
            safety_critical: Safety critical flag
            inference_time: Inference processing time
            
        Returns:
            Safety feature vector
        """
        # Calculate additional safety metrics
        distances = []
        for i in range(min(detection_count, self.max_detections)):
            if i < len(detections):
                distance = detections[i, 8] if detections.shape[1] > 8 else 0.0  # Distance is 9th feature
                if distance > 0:
                    distances.append(distance)
        
        avg_distance = np.mean(distances) if distances else 0.0
        closest_distance = np.min(distances) if distances else 0.0
        
        # Create safety feature vector
        safety_features = np.array([
            detection_count / self.max_detections,  # Normalized detection count
            safety_critical * self.safety_feature_weight,  # Weighted safety flag
            np.clip(inference_time, 0, 1.0),  # Clipped inference time
            avg_distance / self.distance_normalization_factor,  # Normalized average distance
            closest_distance / self.distance_normalization_factor  # Normalized closest distance
        ], dtype=np.float32)
        
        return safety_features
    
    def _normalize_image_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize image features."""
        if self.feature_scaling_method == 'minmax':
            return (features - self._feature_stats['image_min']) / (
                self._feature_stats['image_max'] - self._feature_stats['image_min'] + 1e-8
            )
        elif self.feature_scaling_method == 'standard':
            return (features - self._feature_stats['image_mean']) / (
                self._feature_stats['image_std'] + 1e-8
            )
        else:
            return features
    
    def _normalize_detection_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize detection features."""
        if self.feature_scaling_method == 'minmax':
            return (features - self._feature_stats['detection_min']) / (
                self._feature_stats['detection_max'] - self._feature_stats['detection_min'] + 1e-8
            )
        elif self.feature_scaling_method == 'standard':
            return (features - self._feature_stats['detection_mean']) / (
                self._feature_stats['detection_std'] + 1e-8
            )
        else:
            return features
    
    def _create_flattened_observation(
        self, 
        image_features: Optional[np.ndarray], 
        detection_features: Optional[np.ndarray], 
        safety_features: np.ndarray
    ) -> np.ndarray:
        """
        Create flattened observation vector.
        
        Args:
            image_features: Processed image features
            detection_features: Processed detection features
            safety_features: Safety and metadata features
            
        Returns:
            Flattened observation vector
        """
        components = []
        
        if image_features is not None:
            components.append(image_features)
        
        if detection_features is not None:
            components.append(detection_features)
        
        components.append(safety_features)
        
        return np.concatenate(components)
    
    def _create_dict_observation(
        self, 
        image: Optional[np.ndarray],
        image_features: Optional[np.ndarray], 
        detection_features: Optional[np.ndarray], 
        safety_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create dictionary observation.
        
        Args:
            image: Original image (if available)
            image_features: Processed image features
            detection_features: Processed detection features
            safety_features: Safety and metadata features
            
        Returns:
            Dictionary observation
        """
        obs_dict = {}
        
        if self.include_image_features and image is not None:
            obs_dict['image'] = image
        
        if detection_features is not None:
            obs_dict['detection_features'] = detection_features
        
        obs_dict['safety_features'] = safety_features
        
        return obs_dict
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """
        Get feature extraction statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self._processing_stats.copy()
        
        if stats['total_observations'] > 0:
            stats['avg_feature_extraction_time'] = (
                stats['feature_extraction_time'] / stats['total_observations']
            )
        else:
            stats['avg_feature_extraction_time'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self._processing_stats = {
            'total_observations': 0,
            'feature_extraction_time': 0.0,
            'normalization_time': 0.0
        }
    
    def update_feature_stats(
        self, 
        detection_stats: Optional[Dict[str, np.ndarray]] = None,
        image_stats: Optional[Dict[str, float]] = None
    ):
        """
        Update feature normalization statistics.
        
        Args:
            detection_stats: Detection feature statistics
            image_stats: Image feature statistics
        """
        if detection_stats is not None:
            for key in ['detection_min', 'detection_max', 'detection_mean', 'detection_std']:
                if key in detection_stats:
                    self._feature_stats[key] = detection_stats[key]
        
        if image_stats is not None:
            for key in ['image_min', 'image_max', 'image_mean', 'image_std']:
                if key in image_stats:
                    self._feature_stats[key] = image_stats[key]
        
        logger.info("Feature normalization statistics updated")
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get information about the observation space and configuration.
        
        Returns:
            Dictionary with observation space information
        """
        return {
            'output_mode': self.output_mode,
            'observation_space_shape': self.observation_space.shape if hasattr(self.observation_space, 'shape') else 'Dict',
            'include_detection_features': self.include_detection_features,
            'include_image_features': self.include_image_features,
            'max_detections': self.max_detections,
            'detection_feature_size': self.detection_feature_size,
            'image_feature_method': self.image_feature_method,
            'normalize_features': self.normalize_features,
            'feature_scaling_method': self.feature_scaling_method
        }
    
    def reset(self, **kwargs):
        """Reset environment and clear processing history."""
        observation = self.env.reset(**kwargs)
        return self.observation(observation)