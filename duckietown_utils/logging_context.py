"""
Logging context managers and utilities for performance tracking.

This module provides context managers and utilities to simplify logging
of performance metrics and structured data in the enhanced Duckietown RL system.
"""

import time
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .enhanced_logger import get_logger


@dataclass
class TimingResult:
    """Result of timing measurement."""
    duration_ms: float
    start_time: float
    end_time: float


class PerformanceTracker:
    """Thread-safe performance tracking for logging."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._timings: Dict[str, float] = {}
        self._frame_id = 0
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        with self._lock:
            self._timings[f"{operation}_start"] = time.time()
    
    def end_timing(self, operation: str) -> float:
        """End timing an operation and return duration in ms."""
        end_time = time.time()
        with self._lock:
            start_key = f"{operation}_start"
            if start_key in self._timings:
                duration_ms = (end_time - self._timings[start_key]) * 1000
                del self._timings[start_key]
                return duration_ms
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, Optional[float]]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            
            gpu_memory_mb = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            return {
                "memory_usage_mb": memory_usage_mb,
                "gpu_memory_usage_mb": gpu_memory_mb
            }
        except Exception:
            return {"memory_usage_mb": None, "gpu_memory_usage_mb": None}
    
    def increment_frame(self) -> int:
        """Increment and return current frame ID."""
        with self._lock:
            self._frame_id += 1
            return self._frame_id
    
    def get_frame_id(self) -> int:
        """Get current frame ID."""
        with self._lock:
            return self._frame_id


# Global performance tracker
_performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker."""
    return _performance_tracker


@contextmanager
def log_timing(operation_name: str, logger=None, log_result: bool = True):
    """
    Context manager for timing operations and optionally logging results.
    
    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance to use (defaults to global logger)
        log_result: Whether to log the timing result
    
    Yields:
        TimingResult: Object containing timing information
    
    Example:
        with log_timing("object_detection") as timing:
            # Perform object detection
            detections = detect_objects(image)
        print(f"Detection took {timing.duration_ms:.2f}ms")
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.time()
    
    try:
        yield None  # Will be replaced with TimingResult after completion
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        result = TimingResult(
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time
        )
        
        if log_result:
            logger.log_info(f"{operation_name} completed", 
                           operation=operation_name,
                           duration_ms=duration_ms)


@contextmanager
def log_detection_timing(frame_id: int, logger=None):
    """
    Context manager specifically for object detection timing.
    
    Args:
        frame_id: Current frame identifier
        logger: Logger instance to use
    
    Yields:
        Callable: Function to call with detection results for logging
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.time()
    
    def log_detections(detections, confidence_threshold=0.5):
        """Log detection results with timing."""
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        logger.log_object_detection(
            frame_id=frame_id,
            detections=detections,
            processing_time_ms=processing_time_ms,
            confidence_threshold=confidence_threshold
        )
        
        return processing_time_ms
    
    yield log_detections


@contextmanager
def log_action_timing(frame_id: int, wrapper_source: str, logger=None):
    """
    Context manager for action processing timing.
    
    Args:
        frame_id: Current frame identifier
        wrapper_source: Name of the wrapper processing the action
        logger: Logger instance to use
    
    Yields:
        Callable: Function to call with action results for logging
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.time()
    
    def log_action(original_action, modified_action, action_type, reasoning, 
                   triggering_conditions, safety_checks):
        """Log action decision with timing."""
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        logger.log_action_decision(
            frame_id=frame_id,
            original_action=original_action,
            modified_action=modified_action,
            action_type=action_type,
            reasoning=reasoning,
            triggering_conditions=triggering_conditions,
            safety_checks=safety_checks,
            wrapper_source=wrapper_source
        )
        
        return processing_time_ms
    
    yield log_action


@contextmanager
def log_reward_timing(frame_id: int, episode_step: int, logger=None):
    """
    Context manager for reward calculation timing.
    
    Args:
        frame_id: Current frame identifier
        episode_step: Current step in episode
        logger: Logger instance to use
    
    Yields:
        Callable: Function to call with reward results for logging
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.time()
    
    def log_reward(total_reward, reward_components, reward_weights, cumulative_reward):
        """Log reward calculation with timing."""
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        logger.log_reward_components(
            frame_id=frame_id,
            total_reward=total_reward,
            reward_components=reward_components,
            reward_weights=reward_weights,
            episode_step=episode_step,
            cumulative_reward=cumulative_reward
        )
        
        return processing_time_ms
    
    yield log_reward


class LoggingMixin:
    """
    Mixin class to add logging capabilities to wrapper classes.
    
    This mixin provides common logging methods that can be used by
    wrapper classes to log their operations consistently.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger()
        self._performance_tracker = get_performance_tracker()
        self._wrapper_name = self.__class__.__name__
    
    def _log_wrapper_initialization(self, config: Dict[str, Any]) -> None:
        """Log wrapper initialization with configuration."""
        self._logger.log_info(f"{self._wrapper_name} initialized", 
                             wrapper=self._wrapper_name,
                             config=config)
    
    def _log_wrapper_error(self, operation: str, error: Exception, **context) -> None:
        """Log wrapper error with context."""
        self._logger.log_error(f"{self._wrapper_name} error in {operation}",
                              exception=error,
                              wrapper=self._wrapper_name,
                              operation=operation,
                              **context)
    
    def _log_wrapper_warning(self, message: str, **context) -> None:
        """Log wrapper warning with context."""
        self._logger.log_warning(f"{self._wrapper_name}: {message}",
                                wrapper=self._wrapper_name,
                                **context)
    
    def _get_frame_id(self) -> int:
        """Get current frame ID from performance tracker."""
        return self._performance_tracker.get_frame_id()
    
    def _increment_frame_id(self) -> int:
        """Increment and get frame ID."""
        return self._performance_tracker.increment_frame()


def log_performance_summary(logger=None, interval_frames: int = 100):
    """
    Decorator to log performance summary at regular intervals.
    
    Args:
        logger: Logger instance to use
        interval_frames: Number of frames between summary logs
    """
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        frame_count = 0
        
        def wrapper(*args, **kwargs):
            nonlocal frame_count
            frame_count += 1
            
            result = func(*args, **kwargs)
            
            if frame_count % interval_frames == 0:
                summary = logger.get_log_summary()
                logger.log_info("Performance summary", 
                               frames_processed=frame_count,
                               **summary)
            
            return result
        
        return wrapper
    return decorator


def create_structured_log_entry(
    log_type: str,
    frame_id: int,
    data: Dict[str, Any],
    timing_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create a structured log entry with standard fields.
    
    Args:
        log_type: Type of log entry
        frame_id: Current frame identifier
        data: Log-specific data
        timing_ms: Optional timing information
    
    Returns:
        Structured log entry dictionary
    """
    entry = {
        "timestamp": time.time(),
        "log_type": log_type,
        "frame_id": frame_id,
        **data
    }
    
    if timing_ms is not None:
        entry["processing_time_ms"] = timing_ms
    
    return entry