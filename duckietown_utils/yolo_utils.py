"""
YOLO v5 integration utilities for Duckietown RL environment.
Provides model loading, inference, and error handling capabilities.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    warnings.warn("ultralytics not available. YOLO functionality will be disabled.")

logger = logging.getLogger(__name__)


class YOLOModelLoader:
    """
    Utility class for loading and managing YOLO v5 models with robust error handling.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', confidence_threshold: float = 0.5):
        """
        Initialize YOLO model loader.
        
        Args:
            model_path: Path to YOLO model file (.pt)
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._model_loaded = False
        
    def _determine_device(self, device: str) -> str:
        """Determine the best available device for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                logger.info("CUDA not available, falling back to CPU")
                return 'cpu'
        return device
    
    def load_model(self) -> bool:
        """
        Load YOLO model with comprehensive error handling.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not YOLO_AVAILABLE:
            logger.error("ultralytics package not available. Cannot load YOLO model.")
            return False
            
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load model
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to specified device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # Test model with dummy input
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(dummy_input, verbose=False)
            
            self._model_loaded = True
            logger.info(f"YOLO model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None
            self._model_loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model_loaded and self.model is not None
    
    def get_model(self):
        """Get the loaded YOLO model instance."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def reload_model(self) -> bool:
        """Reload the model (useful for error recovery)."""
        self._model_loaded = False
        self.model = None
        return self.load_model()


class YOLOInferenceWrapper:
    """
    High-performance YOLO inference wrapper with optimization and error handling.
    """
    
    def __init__(self, model_loader: YOLOModelLoader, max_detections: int = 10):
        """
        Initialize inference wrapper.
        
        Args:
            model_loader: YOLOModelLoader instance
            max_detections: Maximum number of detections to return
        """
        self.model_loader = model_loader
        self.max_detections = max_detections
        self._inference_count = 0
        self._total_inference_time = 0.0
        
    def detect_objects(self, image: np.ndarray) -> Dict:
        """
        Perform object detection on input image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Dict containing detection results and metadata
        """
        if not self.model_loader.is_loaded():
            logger.warning("YOLO model not loaded, returning empty detections")
            return self._empty_detection_result()
        
        try:
            start_time = time.time()
            
            # Validate input image
            if not self._validate_image(image):
                return self._empty_detection_result()
            
            # Run inference
            results = self.model_loader.get_model()(
                image, 
                conf=self.model_loader.confidence_threshold,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results, image.shape)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self._update_metrics(inference_time)
            
            return {
                'detections': detections,
                'detection_count': len(detections),
                'inference_time': inference_time,
                'frame_shape': image.shape,
                'safety_critical': self._check_safety_critical(detections)
            }
            
        except Exception as e:
            logger.error(f"YOLO inference failed: {str(e)}")
            return self._empty_detection_result()
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate input image format and dimensions."""
        if image is None:
            logger.error("Input image is None")
            return False
        
        if len(image.shape) != 3:
            logger.error(f"Invalid image shape: {image.shape}. Expected (H, W, C)")
            return False
        
        if image.shape[2] != 3:
            logger.error(f"Invalid number of channels: {image.shape[2]}. Expected 3")
            return False
        
        return True
    
    def _process_results(self, results, image_shape: Tuple[int, int, int]) -> List[Dict]:
        """Process YOLO results into standardized detection format."""
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        # Get first result (batch size = 1)
        result = results[0]
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        # Get class names
        class_names = result.names if hasattr(result, 'names') else {}
        
        for i in range(min(len(boxes), self.max_detections)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = int(classes[i])
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # Calculate center and relative position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Normalize to image coordinates
            img_height, img_width = image_shape[:2]
            rel_x = (center_x - img_width / 2) / img_width
            rel_y = (center_y - img_height / 2) / img_height
            
            # Estimate distance (simple heuristic based on bounding box size)
            bbox_area = (x2 - x1) * (y2 - y1)
            normalized_area = bbox_area / (img_width * img_height)
            estimated_distance = max(0.1, 2.0 / (normalized_area + 0.1))  # Rough estimate
            
            detection = {
                'class': class_name,
                'confidence': float(confidence),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [float(center_x), float(center_y)],
                'relative_position': [float(rel_x), float(rel_y)],
                'distance': float(estimated_distance),
                'area': float(bbox_area)
            }
            
            detections.append(detection)
        
        return detections
    
    def _check_safety_critical(self, detections: List[Dict]) -> bool:
        """Check if any detections are safety critical (close to robot)."""
        safety_distance_threshold = 1.0  # meters
        
        for detection in detections:
            if detection['distance'] < safety_distance_threshold:
                # Also check if object is in front of robot (positive y in relative coordinates)
                if detection['relative_position'][1] > -0.2:  # Allow some tolerance
                    return True
        
        return False
    
    def _empty_detection_result(self) -> Dict:
        """Return empty detection result for error cases."""
        return {
            'detections': [],
            'detection_count': 0,
            'inference_time': 0.0,
            'frame_shape': None,
            'safety_critical': False
        }
    
    def _update_metrics(self, inference_time: float):
        """Update performance metrics."""
        self._inference_count += 1
        self._total_inference_time += inference_time
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if self._inference_count == 0:
            return {'avg_inference_time': 0.0, 'total_inferences': 0}
        
        return {
            'avg_inference_time': self._total_inference_time / self._inference_count,
            'total_inferences': self._inference_count,
            'total_time': self._total_inference_time
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._inference_count = 0
        self._total_inference_time = 0.0


def create_yolo_inference_system(
    model_path: str,
    device: str = 'auto',
    confidence_threshold: float = 0.5,
    max_detections: int = 10
) -> Optional[YOLOInferenceWrapper]:
    """
    Factory function to create a complete YOLO inference system.
    
    Args:
        model_path: Path to YOLO model file
        device: Device for inference ('cpu', 'cuda', 'auto')
        confidence_threshold: Minimum confidence for detections
        max_detections: Maximum number of detections to return
        
    Returns:
        YOLOInferenceWrapper instance if successful, None otherwise
    """
    try:
        # Create model loader
        model_loader = YOLOModelLoader(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold
        )
        
        # Load model
        if not model_loader.load_model():
            logger.error("Failed to load YOLO model")
            return None
        
        # Create inference wrapper
        inference_wrapper = YOLOInferenceWrapper(
            model_loader=model_loader,
            max_detections=max_detections
        )
        
        logger.info("YOLO inference system created successfully")
        return inference_wrapper
        
    except Exception as e:
        logger.error(f"Failed to create YOLO inference system: {str(e)}")
        return None