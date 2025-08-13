"""
Enhanced logging system for Duckietown RL with structured logging capabilities.

This module provides comprehensive logging for object detections, action decisions,
reward components, and performance metrics with JSON-structured output.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class ObjectDetectionLog:
    """Structured log entry for object detection results."""
    timestamp: float
    frame_id: int
    detections: List[Dict[str, Any]]
    processing_time_ms: float
    total_objects: int
    safety_critical: bool
    confidence_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ActionDecisionLog:
    """Structured log entry for action decisions and reasoning."""
    timestamp: float
    frame_id: int
    original_action: List[float]
    modified_action: List[float]
    action_type: str  # 'lane_following', 'object_avoidance', 'lane_changing'
    reasoning: str
    triggering_conditions: Dict[str, Any]
    safety_checks: Dict[str, bool]
    wrapper_source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RewardComponentLog:
    """Structured log entry for reward component analysis."""
    timestamp: float
    frame_id: int
    total_reward: float
    reward_components: Dict[str, float]
    reward_weights: Dict[str, float]
    episode_step: int
    cumulative_reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PerformanceMetricsLog:
    """Structured log entry for performance metrics."""
    timestamp: float
    frame_id: int
    fps: float
    detection_time_ms: float
    action_processing_time_ms: float
    reward_calculation_time_ms: float
    total_step_time_ms: float
    memory_usage_mb: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EnhancedLogger:
    """
    Comprehensive logging system for enhanced Duckietown RL.
    
    Provides structured logging for object detections, action decisions,
    reward components, and performance metrics with configurable output formats.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        log_detections: bool = True,
        log_actions: bool = True,
        log_rewards: bool = True,
        log_performance: bool = True,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize the enhanced logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_detections: Enable object detection logging
            log_actions: Enable action decision logging
            log_rewards: Enable reward component logging
            log_performance: Enable performance metrics logging
            console_output: Enable console output
            file_output: Enable file output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_detections = log_detections
        self.log_actions = log_actions
        self.log_rewards = log_rewards
        self.log_performance = log_performance
        
        # Setup main logger
        self.logger = logging.getLogger("enhanced_duckietown_rl")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self.json_formatter = JsonFormatter()
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(console_handler)
        
        if file_output:
            # Main log file
            main_log_file = self.log_dir / f"enhanced_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(main_log_file)
            file_handler.setFormatter(self.json_formatter)
            self.logger.addHandler(file_handler)
            
            # Separate structured log files
            if self.log_detections:
                self.detection_log_file = self.log_dir / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            if self.log_actions:
                self.action_log_file = self.log_dir / f"actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            if self.log_rewards:
                self.reward_log_file = self.log_dir / f"rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            if self.log_performance:
                self.performance_log_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        
        self.logger.info("Enhanced logger initialized", extra={
            "log_dir": str(self.log_dir),
            "log_level": log_level,
            "features": {
                "detections": log_detections,
                "actions": log_actions,
                "rewards": log_rewards,
                "performance": log_performance
            }
        })
    
    def log_object_detection(
        self,
        frame_id: int,
        detections: List[Dict[str, Any]],
        processing_time_ms: float,
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Log object detection results.
        
        Args:
            frame_id: Current frame identifier
            detections: List of detection dictionaries
            processing_time_ms: Time taken for detection processing
            confidence_threshold: Confidence threshold used
        """
        if not self.log_detections:
            return
        
        # Determine if any detections are safety critical
        safety_critical = any(
            det.get('distance', float('inf')) < 0.5 
            for det in detections
        )
        
        log_entry = ObjectDetectionLog(
            timestamp=time.time(),
            frame_id=frame_id,
            detections=detections,
            processing_time_ms=processing_time_ms,
            total_objects=len(detections),
            safety_critical=safety_critical,
            confidence_threshold=confidence_threshold
        )
        
        # Log to main logger
        self.logger.info("Object detection completed", extra={
            "type": "object_detection",
            "frame_id": frame_id,
            "total_objects": len(detections),
            "safety_critical": safety_critical,
            "processing_time_ms": processing_time_ms
        })
        
        # Write to structured log file
        if hasattr(self, 'detection_log_file'):
            self._write_jsonl(self.detection_log_file, log_entry.to_dict())
    
    def log_action_decision(
        self,
        frame_id: int,
        original_action: Union[np.ndarray, List[float]],
        modified_action: Union[np.ndarray, List[float]],
        action_type: str,
        reasoning: str,
        triggering_conditions: Dict[str, Any],
        safety_checks: Dict[str, bool],
        wrapper_source: str
    ) -> None:
        """
        Log action decision with reasoning and conditions.
        
        Args:
            frame_id: Current frame identifier
            original_action: Original action before modification
            modified_action: Action after wrapper modification
            action_type: Type of action (lane_following, object_avoidance, lane_changing)
            reasoning: Human-readable reasoning for the action
            triggering_conditions: Conditions that triggered this action
            safety_checks: Results of safety checks performed
            wrapper_source: Name of the wrapper that made this decision
        """
        if not self.log_actions:
            return
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(original_action, np.ndarray):
            original_action = original_action.tolist()
        if isinstance(modified_action, np.ndarray):
            modified_action = modified_action.tolist()
        
        log_entry = ActionDecisionLog(
            timestamp=time.time(),
            frame_id=frame_id,
            original_action=original_action,
            modified_action=modified_action,
            action_type=action_type,
            reasoning=reasoning,
            triggering_conditions=triggering_conditions,
            safety_checks=safety_checks,
            wrapper_source=wrapper_source
        )
        
        # Log to main logger
        self.logger.info("Action decision made", extra={
            "type": "action_decision",
            "frame_id": frame_id,
            "action_type": action_type,
            "wrapper_source": wrapper_source,
            "reasoning": reasoning,
            "safety_checks_passed": all(safety_checks.values())
        })
        
        # Write to structured log file
        if hasattr(self, 'action_log_file'):
            self._write_jsonl(self.action_log_file, log_entry.to_dict())
    
    def log_reward_components(
        self,
        frame_id: int,
        total_reward: float,
        reward_components: Dict[str, float],
        reward_weights: Dict[str, float],
        episode_step: int,
        cumulative_reward: float
    ) -> None:
        """
        Log reward component breakdown for analysis.
        
        Args:
            frame_id: Current frame identifier
            total_reward: Total calculated reward
            reward_components: Individual reward components
            reward_weights: Weights used for each component
            episode_step: Current step in episode
            cumulative_reward: Cumulative reward for episode
        """
        if not self.log_rewards:
            return
        
        log_entry = RewardComponentLog(
            timestamp=time.time(),
            frame_id=frame_id,
            total_reward=total_reward,
            reward_components=reward_components,
            reward_weights=reward_weights,
            episode_step=episode_step,
            cumulative_reward=cumulative_reward
        )
        
        # Log to main logger
        self.logger.info("Reward calculated", extra={
            "type": "reward_calculation",
            "frame_id": frame_id,
            "total_reward": total_reward,
            "episode_step": episode_step,
            "cumulative_reward": cumulative_reward,
            "dominant_component": max(reward_components.items(), key=lambda x: abs(x[1]))[0]
        })
        
        # Write to structured log file
        if hasattr(self, 'reward_log_file'):
            self._write_jsonl(self.reward_log_file, log_entry.to_dict())
    
    def log_performance_metrics(
        self,
        frame_id: int,
        detection_time_ms: float = 0.0,
        action_processing_time_ms: float = 0.0,
        reward_calculation_time_ms: float = 0.0,
        memory_usage_mb: Optional[float] = None,
        gpu_memory_usage_mb: Optional[float] = None
    ) -> None:
        """
        Log performance metrics for monitoring.
        
        Args:
            frame_id: Current frame identifier
            detection_time_ms: Time spent on object detection
            action_processing_time_ms: Time spent on action processing
            reward_calculation_time_ms: Time spent on reward calculation
            memory_usage_mb: Current memory usage in MB
            gpu_memory_usage_mb: Current GPU memory usage in MB
        """
        if not self.log_performance:
            return
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Maintain rolling window of frame times for FPS calculation
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:  # Keep last 30 frames
            self.frame_times.pop(0)
        
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0.0
        total_step_time_ms = (detection_time_ms + action_processing_time_ms + 
                             reward_calculation_time_ms)
        
        log_entry = PerformanceMetricsLog(
            timestamp=current_time,
            frame_id=frame_id,
            fps=fps,
            detection_time_ms=detection_time_ms,
            action_processing_time_ms=action_processing_time_ms,
            reward_calculation_time_ms=reward_calculation_time_ms,
            total_step_time_ms=total_step_time_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_usage_mb=gpu_memory_usage_mb
        )
        
        # Log to main logger (less frequent to avoid spam)
        if frame_id % 30 == 0:  # Log every 30 frames
            self.logger.info("Performance metrics", extra={
                "type": "performance_metrics",
                "frame_id": frame_id,
                "fps": round(fps, 2),
                "total_step_time_ms": round(total_step_time_ms, 2),
                "memory_usage_mb": memory_usage_mb
            })
        
        # Write to structured log file
        if hasattr(self, 'performance_log_file'):
            self._write_jsonl(self.performance_log_file, log_entry.to_dict())
    
    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error with additional context."""
        extra_data = {
            "type": "error",
            "timestamp": time.time(),
            **kwargs
        }
        
        if exception:
            extra_data["exception_type"] = type(exception).__name__
            extra_data["exception_message"] = str(exception)
        
        self.logger.error(message, extra=extra_data, exc_info=exception)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning with additional context."""
        extra_data = {
            "type": "warning",
            "timestamp": time.time(),
            **kwargs
        }
        self.logger.warning(message, extra=extra_data)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info with additional context."""
        extra_data = {
            "type": "info",
            "timestamp": time.time(),
            **kwargs
        }
        self.logger.info(message, extra=extra_data)
    
    def _write_jsonl(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write data to JSONL file."""
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(data, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write to log file {file_path}: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logging activity."""
        return {
            "log_dir": str(self.log_dir),
            "features_enabled": {
                "detections": self.log_detections,
                "actions": self.log_actions,
                "rewards": self.log_rewards,
                "performance": self.log_performance
            },
            "current_fps": 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0.0,
            "total_frames_processed": len(self.frame_times)
        }


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'getMessage', 'exc_info', 
                              'exc_text', 'stack_info']:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


# Global logger instance
_global_logger: Optional[EnhancedLogger] = None


def get_logger() -> EnhancedLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = EnhancedLogger()
    return _global_logger


def initialize_logger(**kwargs) -> EnhancedLogger:
    """Initialize the global logger with custom configuration."""
    global _global_logger
    _global_logger = EnhancedLogger(**kwargs)
    return _global_logger