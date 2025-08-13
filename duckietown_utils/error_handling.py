"""
Comprehensive Error Handling and Recovery System for Enhanced Duckietown RL.

This module provides robust error handling, graceful degradation, and recovery mechanisms
for all components of the enhanced Duckietown RL system, including YOLO inference,
action wrappers, and safety systems.
"""

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorizing different types of errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    component: str
    operation: str
    error_type: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    fallback_value: Any = None
    error_message: Optional[str] = None
    recovery_time: float = 0.0


class ErrorHandlingMixin:
    """
    Mixin class providing error handling capabilities to wrapper classes.
    
    This mixin provides standardized error handling, logging, and recovery
    mechanisms that can be used by all wrapper classes in the system.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error_handler = ErrorHandler(component_name=self.__class__.__name__)
        self._error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
    
    def _handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any = None
    ) -> RecoveryResult:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Error context information
            fallback_value: Default value to return if recovery fails
            
        Returns:
            RecoveryResult with recovery outcome
        """
        # Update error statistics
        self._update_error_stats(error, context)
        
        # Delegate to error handler
        return self._error_handler.handle_error(error, context, fallback_value)
    
    def _update_error_stats(self, error: Exception, context: ErrorContext):
        """Update error statistics for monitoring."""
        self._error_stats['total_errors'] += 1
        
        error_type = type(error).__name__
        self._error_stats['errors_by_type'][error_type] = (
            self._error_stats['errors_by_type'].get(error_type, 0) + 1
        )
        
        severity = context.severity.value
        self._error_stats['errors_by_severity'][severity] = (
            self._error_stats['errors_by_severity'].get(severity, 0) + 1
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return self._error_stats.copy()
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self._error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    This class provides comprehensive error handling with configurable recovery
    strategies, logging, and monitoring capabilities.
    """
    
    def __init__(self, component_name: str):
        """
        Initialize error handler.
        
        Args:
            component_name: Name of the component using this error handler
        """
        self.component_name = component_name
        self.recovery_strategies = self._setup_default_recovery_strategies()
        self.error_history = []
        self.max_history_size = 100
        
    def _setup_default_recovery_strategies(self) -> Dict[str, RecoveryStrategy]:
        """Setup default recovery strategies for common error types."""
        return {
            # YOLO and inference errors
            'RuntimeError': RecoveryStrategy.RETRY,
            'torch.cuda.OutOfMemoryError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'FileNotFoundError': RecoveryStrategy.FALLBACK,
            'ImportError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'ModuleNotFoundError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            
            # Action and safety errors
            'ValueError': RecoveryStrategy.FALLBACK,
            'IndexError': RecoveryStrategy.FALLBACK,
            'KeyError': RecoveryStrategy.FALLBACK,
            'TypeError': RecoveryStrategy.FALLBACK,
            
            # Critical safety errors
            'SafetyViolationError': RecoveryStrategy.EMERGENCY_STOP,
            'ActionValidationError': RecoveryStrategy.FALLBACK,
            
            # Network and I/O errors
            'ConnectionError': RecoveryStrategy.RETRY,
            'TimeoutError': RecoveryStrategy.RETRY,
            'IOError': RecoveryStrategy.RETRY,
            
            # Memory and resource errors
            'MemoryError': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'ResourceWarning': RecoveryStrategy.GRACEFUL_DEGRADATION,
            
            # Default fallback
            'Exception': RecoveryStrategy.FALLBACK
        }
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any = None
    ) -> RecoveryResult:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Error context information
            fallback_value: Default value to return if recovery fails
            
        Returns:
            RecoveryResult with recovery outcome
        """
        start_time = time.time()
        
        # Log the error
        self._log_error(error, context)
        
        # Add to error history
        self._add_to_history(error, context)
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error, context)
        context.recovery_strategy = strategy
        
        # Execute recovery strategy
        result = self._execute_recovery_strategy(error, context, fallback_value)
        result.recovery_time = time.time() - start_time
        
        return result
    
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with appropriate level based on severity."""
        error_msg = (
            f"Error in {context.component}.{context.operation}: "
            f"{type(error).__name__}: {str(error)}"
        )
        
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(error_msg, exc_info=True)
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(error_msg, exc_info=True)
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(error_msg)
        else:
            logger.debug(error_msg)
    
    def _add_to_history(self, error: Exception, context: ErrorContext):
        """Add error to history for pattern analysis."""
        error_record = {
            'timestamp': context.timestamp,
            'component': context.component,
            'operation': context.operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': context.severity.value,
            'retry_count': context.retry_count
        }
        
        self.error_history.append(error_record)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def _determine_recovery_strategy(
        self,
        error: Exception,
        context: ErrorContext
    ) -> RecoveryStrategy:
        """Determine the best recovery strategy for the error."""
        error_type = type(error).__name__
        
        # Check for explicit strategy in context
        if context.recovery_strategy != RecoveryStrategy.RETRY:  # Default value check
            return context.recovery_strategy
        
        # Use configured strategy for error type
        strategy = self.recovery_strategies.get(error_type)
        if strategy:
            return strategy
        
        # Check parent classes
        for error_class in type(error).__mro__[1:]:  # Skip the error itself
            parent_strategy = self.recovery_strategies.get(error_class.__name__)
            if parent_strategy:
                return parent_strategy
        
        # Default strategy based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.EMERGENCY_STOP
        elif context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.FALLBACK
        else:
            return RecoveryStrategy.RETRY
    
    def _execute_recovery_strategy(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any
    ) -> RecoveryResult:
        """Execute the determined recovery strategy."""
        strategy = context.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._handle_retry(error, context, fallback_value)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._handle_fallback(error, context, fallback_value)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._handle_graceful_degradation(error, context, fallback_value)
            
            elif strategy == RecoveryStrategy.EMERGENCY_STOP:
                return self._handle_emergency_stop(error, context)
            
            elif strategy == RecoveryStrategy.IGNORE:
                return self._handle_ignore(error, context, fallback_value)
            
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return self._handle_fallback(error, context, fallback_value)
                
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                fallback_value=fallback_value,
                error_message=f"Recovery failed: {str(recovery_error)}"
            )
    
    def _handle_retry(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any
    ) -> RecoveryResult:
        """Handle retry recovery strategy."""
        if context.retry_count >= context.max_retries:
            logger.warning(
                f"Max retries ({context.max_retries}) exceeded for "
                f"{context.component}.{context.operation}"
            )
            return self._handle_fallback(error, context, fallback_value)
        
        logger.info(
            f"Retrying {context.component}.{context.operation} "
            f"(attempt {context.retry_count + 1}/{context.max_retries})"
        )
        
        # For retry, we don't actually retry here - we just indicate that
        # the caller should retry the operation
        return RecoveryResult(
            success=False,  # Indicates retry needed
            strategy_used=RecoveryStrategy.RETRY,
            fallback_value=fallback_value,
            error_message="Retry required"
        )
    
    def _handle_fallback(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any
    ) -> RecoveryResult:
        """Handle fallback recovery strategy."""
        logger.info(
            f"Using fallback value for {context.component}.{context.operation}"
        )
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK,
            fallback_value=fallback_value,
            error_message=None
        )
    
    def _handle_graceful_degradation(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any
    ) -> RecoveryResult:
        """Handle graceful degradation recovery strategy."""
        logger.warning(
            f"Graceful degradation activated for {context.component}.{context.operation}"
        )
        
        # For graceful degradation, we might disable certain features
        # or use simplified alternatives
        degraded_value = self._get_degraded_fallback(context, fallback_value)
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_value=degraded_value,
            error_message="Operating in degraded mode"
        )
    
    def _handle_emergency_stop(
        self,
        error: Exception,
        context: ErrorContext
    ) -> RecoveryResult:
        """Handle emergency stop recovery strategy."""
        logger.critical(
            f"Emergency stop triggered by {context.component}.{context.operation}"
        )
        
        # Emergency stop - return safe action (zero velocity)
        emergency_action = np.array([0.0, 0.0])
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.EMERGENCY_STOP,
            fallback_value=emergency_action,
            error_message="Emergency stop activated"
        )
    
    def _handle_ignore(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_value: Any
    ) -> RecoveryResult:
        """Handle ignore recovery strategy."""
        logger.debug(
            f"Ignoring error in {context.component}.{context.operation}"
        )
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.IGNORE,
            fallback_value=fallback_value,
            error_message=None
        )
    
    def _get_degraded_fallback(self, context: ErrorContext, fallback_value: Any) -> Any:
        """Get appropriate fallback value for graceful degradation."""
        # Component-specific degraded fallbacks
        if context.component.lower().find('yolo') != -1:
            # For YOLO components, return empty detection result
            return {
                'detections': [],
                'detection_count': 0,
                'inference_time': 0.0,
                'frame_shape': None,
                'safety_critical': False
            }
        
        elif context.component.lower().find('action') != -1:
            # For action components, return safe action
            return np.array([0.1, 0.1])  # Slow forward movement
        
        else:
            return fallback_value
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get error history for analysis."""
        return self.error_history.copy()
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()


class SafetyOverrideSystem:
    """
    Safety override system for preventing dangerous actions and states.
    
    This system monitors actions and system state to prevent safety violations
    and provides emergency override capabilities.
    """
    
    def __init__(self):
        """Initialize safety override system."""
        self.safety_checks = {
            'action_bounds': self._check_action_bounds,
            'collision_risk': self._check_collision_risk,
            'system_health': self._check_system_health,
            'emergency_conditions': self._check_emergency_conditions
        }
        
        self.safety_violations = []
        self.emergency_stop_active = False
        self.max_violation_history = 50
        
    def validate_action(
        self,
        action: np.ndarray,
        observation: Any = None,
        system_state: Dict[str, Any] = None
    ) -> Tuple[bool, np.ndarray, List[str]]:
        """
        Validate action for safety and apply overrides if necessary.
        
        Args:
            action: Proposed action to validate
            observation: Current observation (optional)
            system_state: Current system state (optional)
            
        Returns:
            Tuple of (is_safe, safe_action, violation_messages)
        """
        violations = []
        safe_action = action.copy()
        
        # Run all safety checks
        for check_name, check_func in self.safety_checks.items():
            try:
                is_safe, override_action, message = check_func(
                    action, observation, system_state
                )
                
                if not is_safe:
                    violations.append(f"{check_name}: {message}")
                    if override_action is not None:
                        safe_action = override_action
                        
            except Exception as e:
                logger.error(f"Safety check {check_name} failed: {e}")
                violations.append(f"{check_name}: check failed")
        
        # Log violations
        if violations:
            self._log_safety_violations(violations, action, safe_action)
        
        is_safe = len(violations) == 0
        return is_safe, safe_action, violations
    
    def _check_action_bounds(
        self,
        action: np.ndarray,
        observation: Any,
        system_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Check if action is within valid bounds."""
        if action is None or len(action) == 0:
            return False, np.array([0.0, 0.0]), "Action is None or empty"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            return False, np.array([0.0, 0.0]), "Action contains NaN or infinite values"
        
        # Check bounds (assuming wheel velocities in [0, 1])
        if np.any(action < 0.0) or np.any(action > 1.0):
            clipped_action = np.clip(action, 0.0, 1.0)
            return False, clipped_action, f"Action out of bounds: {action}"
        
        return True, None, ""
    
    def _check_collision_risk(
        self,
        action: np.ndarray,
        observation: Any,
        system_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Check for collision risk based on observations."""
        if observation is None:
            return True, None, ""
        
        # Check for safety critical detections
        if isinstance(observation, dict):
            safety_critical = observation.get('safety_critical', [False])
            if isinstance(safety_critical, (list, np.ndarray)):
                safety_critical = safety_critical[0] if len(safety_critical) > 0 else False
            
            if safety_critical:
                # Check if action would increase collision risk
                forward_velocity = np.mean(action)
                if forward_velocity > 0.3:  # Threshold for high speed
                    # Reduce speed for safety
                    safe_action = action * 0.3  # Reduce to 30% speed
                    return False, safe_action, "High collision risk detected, reducing speed"
        
        return True, None, ""
    
    def _check_system_health(
        self,
        action: np.ndarray,
        observation: Any,
        system_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Check overall system health."""
        if system_state is None:
            return True, None, ""
        
        # Check for system errors or degraded performance
        error_count = system_state.get('error_count', 0)
        if error_count > 10:  # Threshold for too many errors
            # Reduce action magnitude for safety
            safe_action = action * 0.5
            return False, safe_action, f"High error count ({error_count}), reducing action magnitude"
        
        # Check for resource constraints
        memory_usage = system_state.get('memory_usage', 0.0)
        if memory_usage > 0.9:  # 90% memory usage
            # Conservative action to reduce computational load
            safe_action = action * 0.7
            return False, safe_action, f"High memory usage ({memory_usage:.1%}), reducing action"
        
        return True, None, ""
    
    def _check_emergency_conditions(
        self,
        action: np.ndarray,
        observation: Any,
        system_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Check for emergency conditions requiring immediate stop."""
        # Check if emergency stop is already active
        if self.emergency_stop_active:
            return False, np.array([0.0, 0.0]), "Emergency stop active"
        
        # Check for critical system failures
        if system_state and system_state.get('critical_failure', False):
            self.emergency_stop_active = True
            return False, np.array([0.0, 0.0]), "Critical system failure detected"
        
        return True, None, ""
    
    def _log_safety_violations(
        self,
        violations: List[str],
        original_action: np.ndarray,
        safe_action: np.ndarray
    ):
        """Log safety violations for monitoring."""
        violation_record = {
            'timestamp': time.time(),
            'violations': violations,
            'original_action': original_action.tolist(),
            'safe_action': safe_action.tolist()
        }
        
        self.safety_violations.append(violation_record)
        
        # Maintain history size limit
        if len(self.safety_violations) > self.max_violation_history:
            self.safety_violations.pop(0)
        
        # Log the violation
        logger.warning(f"Safety violations detected: {violations}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        self.emergency_stop_active = False
        logger.info("Emergency stop reset")
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety system statistics."""
        return {
            'total_violations': len(self.safety_violations),
            'emergency_stop_active': self.emergency_stop_active,
            'recent_violations': self.safety_violations[-10:] if self.safety_violations else []
        }


@contextmanager
def error_handling_context(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.FALLBACK,
    fallback_value: Any = None,
    max_retries: int = 3
):
    """
    Context manager for standardized error handling.
    
    Usage:
        with error_handling_context("YOLOWrapper", "detect_objects") as ctx:
            result = some_risky_operation()
            ctx.set_result(result)
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        error_type="",
        severity=severity,
        recovery_strategy=recovery_strategy,
        max_retries=max_retries
    )
    
    error_handler = ErrorHandler(component)
    result_container = {'result': fallback_value, 'error': None}
    
    class ContextManager:
        def set_result(self, result):
            result_container['result'] = result
        
        def get_result(self):
            return result_container['result']
    
    ctx_manager = ContextManager()
    
    try:
        yield ctx_manager
    except Exception as e:
        context.error_type = type(e).__name__
        recovery_result = error_handler.handle_error(e, context, fallback_value)
        
        if recovery_result.success:
            result_container['result'] = recovery_result.fallback_value
        else:
            result_container['error'] = e
            if recovery_result.strategy_used == RecoveryStrategy.RETRY:
                # Re-raise for retry logic
                raise
    
    return result_container['result']


# Custom exception classes for specific error types
class SafetyViolationError(Exception):
    """Raised when a safety violation is detected."""
    pass


class ActionValidationError(Exception):
    """Raised when action validation fails."""
    pass


class YOLOInferenceError(Exception):
    """Raised when YOLO inference fails."""
    pass


class GracefulDegradationError(Exception):
    """Raised when graceful degradation is needed."""
    pass