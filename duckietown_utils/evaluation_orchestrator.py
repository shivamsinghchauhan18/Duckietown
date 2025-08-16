#!/usr/bin/env python3
"""
🏆 EVALUATION ORCHESTRATOR 🏆
Core evaluation orchestrator infrastructure for rigorous model evaluation

This module implements the EvaluationOrchestrator class with model registry,
workflow coordination, and comprehensive evaluation management.
"""

import os
import sys
import time
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class EvaluationStatus(Enum):
    """Evaluation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PolicyMode(Enum):
    """Policy evaluation modes."""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"

@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_id: str
    model_path: str
    model_type: str  # 'checkpoint', 'onnx', 'pytorch', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    registration_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate model information."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

@dataclass
class EvaluationTask:
    """Represents a single evaluation task."""
    task_id: str
    model_id: str
    suite_name: str
    policy_mode: PolicyMode
    seeds: List[int]
    status: EvaluationStatus = EvaluationStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Generate task ID if not provided."""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

@dataclass
class EvaluationProgress:
    """Tracks evaluation progress across all tasks."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    running_tasks: int
    overall_progress: float
    estimated_time_remaining: Optional[float] = None
    
    @property
    def pending_tasks(self) -> int:
        return self.total_tasks - self.completed_tasks - self.failed_tasks - self.running_tasks

class ModelRegistry:
    """Registry for managing evaluation models."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, model_path: str, model_type: str = "checkpoint", 
                      model_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model for evaluation.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (checkpoint, onnx, pytorch, etc.)
            model_id: Optional custom model ID
            metadata: Optional metadata dictionary
            
        Returns:
            str: The assigned model ID
        """
        with self._lock:
            if model_id is None:
                model_id = f"model_{len(self.models):04d}_{int(time.time())}"
            
            if model_id in self.models:
                raise ValueError(f"Model ID already exists: {model_id}")
            
            model_info = ModelInfo(
                model_id=model_id,
                model_path=model_path,
                model_type=model_type,
                metadata=metadata or {}
            )
            
            self.models[model_id] = model_info
            self.logger.info(f"Registered model: {model_id} at {model_path}")
            return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        return self.models.get(model_id)
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self.models.values())
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        with self._lock:
            if model_id in self.models:
                del self.models[model_id]
                self.logger.info(f"Removed model: {model_id}")
                return True
            return False
    
    def clear(self):
        """Clear all registered models."""
        with self._lock:
            self.models.clear()
            self.logger.info("Cleared all registered models")

class SeedManager:
    """Manages seed generation and reproducibility for evaluations."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.seed_cache: Dict[str, List[int]] = {}
        self._lock = threading.Lock()
    
    def generate_seeds(self, suite_name: str, num_seeds: int, 
                      deterministic: bool = True) -> List[int]:
        """Generate seeds for a specific evaluation suite.
        
        Args:
            suite_name: Name of the evaluation suite
            num_seeds: Number of seeds to generate
            deterministic: Whether to use deterministic seed generation
            
        Returns:
            List[int]: Generated seeds
        """
        cache_key = f"{suite_name}_{num_seeds}_{deterministic}"
        
        with self._lock:
            if cache_key in self.seed_cache:
                return self.seed_cache[cache_key].copy()
            
            if deterministic:
                # Generate deterministic seeds based on suite name and base seed
                seeds = []
                suite_hash = hash(suite_name) % 10000
                for i in range(num_seeds):
                    seed = (self.base_seed + suite_hash + i * 17) % (2**31 - 1)
                    seeds.append(seed)
            else:
                # Generate random seeds
                import random
                random.seed(self.base_seed)
                seeds = [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]
            
            self.seed_cache[cache_key] = seeds.copy()
            return seeds
    
    def get_reproducible_seeds(self, suite_name: str, num_seeds: int) -> List[int]:
        """Get reproducible seeds for a suite (always deterministic)."""
        return self.generate_seeds(suite_name, num_seeds, deterministic=True)
    
    def clear_cache(self):
        """Clear the seed cache."""
        with self._lock:
            self.seed_cache.clear()

class EvaluationStateTracker:
    """Tracks evaluation state and progress."""
    
    def __init__(self):
        self.tasks: Dict[str, EvaluationTask] = {}
        self.task_history: List[EvaluationTask] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[EvaluationProgress], None]] = []
        self._last_progress_update = time.time()
    
    def add_task(self, task: EvaluationTask):
        """Add a new evaluation task."""
        with self._lock:
            self.tasks[task.task_id] = task
            self.logger.info(f"Added evaluation task: {task.task_id}")
    
    def update_task_status(self, task_id: str, status: EvaluationStatus, 
                          progress: Optional[float] = None, 
                          error_message: Optional[str] = None):
        """Update task status and progress."""
        with self._lock:
            if task_id not in self.tasks:
                self.logger.warning(f"Task not found: {task_id}")
                return
            
            task = self.tasks[task_id]
            old_status = task.status
            task.status = status
            
            if progress is not None:
                task.progress = progress
            
            if error_message is not None:
                task.error_message = error_message
            
            # Update timestamps
            if status == EvaluationStatus.RUNNING and old_status == EvaluationStatus.PENDING:
                task.start_time = datetime.now().isoformat()
            elif status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]:
                task.end_time = datetime.now().isoformat()
                # Note: Task remains in active tasks until explicitly cleared
                # This allows for immediate access to completed task results
            
            self.logger.info(f"Task {task_id} status: {old_status.value} -> {status.value}")
            
            # Trigger progress callbacks
            self._notify_progress_callbacks()
    
    def update_task_results(self, task_id: str, results: Dict[str, Any]):
        """Update task results."""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].results = results
                self.logger.info(f"Updated results for task: {task_id}")
    
    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: EvaluationStatus) -> List[EvaluationTask]:
        """Get all tasks with a specific status."""
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_progress(self) -> EvaluationProgress:
        """Get current evaluation progress."""
        with self._lock:
            # Count all tasks (active + history, but avoid double counting)
            all_tasks = list(self.tasks.values()) + self.task_history
            total_tasks = len(all_tasks)
            
            completed_tasks = len([t for t in all_tasks if t.status == EvaluationStatus.COMPLETED])
            failed_tasks = len([t for t in all_tasks if t.status == EvaluationStatus.FAILED])
            running_tasks = len([t for t in self.tasks.values() if t.status == EvaluationStatus.RUNNING])  # Only active tasks can be running
            
            overall_progress = completed_tasks / max(total_tasks, 1) * 100.0
            
            # Estimate time remaining based on completed tasks
            estimated_time = None
            if completed_tasks > 0 and running_tasks > 0:
                completed_with_times = [
                    t for t in self.task_history 
                    if t.status == EvaluationStatus.COMPLETED and t.start_time and t.end_time
                ]
                if completed_with_times:
                    avg_duration = sum([
                        (datetime.fromisoformat(t.end_time) - datetime.fromisoformat(t.start_time)).total_seconds()
                        for t in completed_with_times
                    ]) / len(completed_with_times)
                    
                    remaining_tasks = total_tasks - completed_tasks - failed_tasks
                    estimated_time = avg_duration * remaining_tasks
            
            return EvaluationProgress(
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                running_tasks=running_tasks,
                overall_progress=overall_progress,
                estimated_time_remaining=estimated_time
            )
    
    def add_progress_callback(self, callback: Callable[[EvaluationProgress], None]):
        """Add a progress callback function."""
        self._progress_callbacks.append(callback)
    
    def _notify_progress_callbacks(self):
        """Notify all progress callbacks."""
        current_time = time.time()
        # Throttle callbacks to avoid spam
        if current_time - self._last_progress_update > 1.0:  # Max 1 update per second
            progress = self.get_progress()
            for callback in self._progress_callbacks:
                try:
                    callback(progress)
                except Exception as e:
                    self.logger.error(f"Progress callback error: {e}")
            self._last_progress_update = current_time
    
    def clear_completed_tasks(self):
        """Clear completed tasks from active tracking."""
        with self._lock:
            completed_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]
            ]
            
            for task_id in completed_tasks:
                task = self.tasks.pop(task_id)
                if task not in self.task_history:
                    self.task_history.append(task)
            
            self.logger.info(f"Cleared {len(completed_tasks)} completed tasks")

class EvaluationOrchestrator:
    """Core evaluation orchestrator for managing comprehensive model evaluations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation orchestrator.
        
        Args:
            config: Configuration dictionary for the orchestrator
        """
        self.config = config or {}
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.seed_manager = SeedManager(base_seed=self.config.get('base_seed', 42))
        self.state_tracker = EvaluationStateTracker()
        
        # Configuration
        self.max_concurrent_evaluations = self.config.get('max_concurrent_evaluations', 4)
        self.default_seeds_per_suite = self.config.get('default_seeds_per_suite', 50)
        self.evaluation_timeout = self.config.get('evaluation_timeout', 3600)  # 1 hour
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Results storage
        self.results_dir = Path(self.config.get('results_dir', 'logs/evaluation_orchestrator'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for concurrent evaluations
        self._executor = None
        self._shutdown = False
        
        self.logger.info("🏆 Evaluation Orchestrator initialized")
        self.logger.info(f"📊 Max concurrent evaluations: {self.max_concurrent_evaluations}")
        self.logger.info(f"🎯 Default seeds per suite: {self.default_seeds_per_suite}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def register_model(self, model_path: str, model_type: str = "checkpoint", 
                      model_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model for evaluation.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (checkpoint, onnx, pytorch, etc.)
            model_id: Optional custom model ID
            metadata: Optional metadata dictionary
            
        Returns:
            str: The assigned model ID
        """
        return self.model_registry.register_model(model_path, model_type, model_id, metadata)
    
    def schedule_evaluation(self, model_ids: Union[str, List[str]], 
                          suite_names: Union[str, List[str]],
                          policy_modes: Optional[Union[PolicyMode, List[PolicyMode]]] = None,
                          seeds_per_suite: Optional[int] = None) -> List[str]:
        """Schedule evaluation tasks for models and suites.
        
        Args:
            model_ids: Model ID(s) to evaluate
            suite_names: Evaluation suite name(s)
            policy_modes: Policy mode(s) to use
            seeds_per_suite: Number of seeds per suite
            
        Returns:
            List[str]: List of scheduled task IDs
        """
        # Normalize inputs to lists
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if isinstance(suite_names, str):
            suite_names = [suite_names]
        if policy_modes is None:
            policy_modes = [PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC]
        elif isinstance(policy_modes, PolicyMode):
            policy_modes = [policy_modes]
        
        seeds_per_suite = seeds_per_suite or self.default_seeds_per_suite
        
        # Validate models exist
        for model_id in model_ids:
            if not self.model_registry.get_model(model_id):
                raise ValueError(f"Model not found: {model_id}")
        
        # Create evaluation tasks
        task_ids = []
        for model_id in model_ids:
            for suite_name in suite_names:
                for policy_mode in policy_modes:
                    # Generate seeds for this suite
                    seeds = self.seed_manager.get_reproducible_seeds(suite_name, seeds_per_suite)
                    
                    # Create task
                    task = EvaluationTask(
                        task_id=f"{model_id}_{suite_name}_{policy_mode.value}_{int(time.time())}",
                        model_id=model_id,
                        suite_name=suite_name,
                        policy_mode=policy_mode,
                        seeds=seeds
                    )
                    
                    self.state_tracker.add_task(task)
                    task_ids.append(task.task_id)
        
        self.logger.info(f"Scheduled {len(task_ids)} evaluation tasks")
        return task_ids
    
    def start_evaluation(self, task_ids: Optional[List[str]] = None) -> bool:
        """Start evaluation execution.
        
        Args:
            task_ids: Optional list of specific task IDs to run
            
        Returns:
            bool: True if evaluation started successfully
        """
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            if self._executor is not None:
                self.logger.warning("Evaluation already running")
                return False
            
            # Get tasks to run
            if task_ids is None:
                tasks_to_run = self.state_tracker.get_tasks_by_status(EvaluationStatus.PENDING)
            else:
                tasks_to_run = [
                    self.state_tracker.get_task(task_id) 
                    for task_id in task_ids
                    if self.state_tracker.get_task(task_id) is not None
                ]
            
            if not tasks_to_run:
                self.logger.info("No tasks to run")
                return False
            
            self.logger.info(f"Starting evaluation of {len(tasks_to_run)} tasks")
            
            # Start thread pool
            self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_evaluations)
            self._shutdown = False
            
            # Submit tasks
            for task in tasks_to_run:
                if not self._shutdown:
                    self._executor.submit(self._execute_task, task)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start evaluation: {e}")
            return False
    
    def _execute_task(self, task: EvaluationTask):
        """Execute a single evaluation task."""
        try:
            self.logger.info(f"Executing task: {task.task_id}")
            self.state_tracker.update_task_status(task.task_id, EvaluationStatus.RUNNING)
            
            # Get model info
            model_info = self.model_registry.get_model(task.model_id)
            if not model_info:
                raise ValueError(f"Model not found: {task.model_id}")
            
            # TODO: This is where the actual evaluation would happen
            # For now, we'll simulate the evaluation
            results = self._simulate_evaluation(task, model_info)
            
            # Update task with results
            self.state_tracker.update_task_results(task.task_id, results)
            self.state_tracker.update_task_status(task.task_id, EvaluationStatus.COMPLETED, progress=100.0)
            
            # Save results
            self._save_task_results(task, results)
            
            self.logger.info(f"Completed task: {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            self.state_tracker.update_task_status(
                task.task_id, 
                EvaluationStatus.FAILED, 
                error_message=str(e)
            )
    
    def _simulate_evaluation(self, task: EvaluationTask, model_info: ModelInfo) -> Dict[str, Any]:
        """Simulate evaluation execution (placeholder for actual evaluation)."""
        import random
        import time
        
        # Simulate evaluation time
        evaluation_time = random.uniform(10, 30)  # 10-30 seconds
        
        num_episodes = len(task.seeds)
        for i in range(num_episodes):
            if self._shutdown:
                break
            
            # Simulate episode execution
            time.sleep(evaluation_time / num_episodes)
            
            # Update progress
            progress = (i + 1) / num_episodes * 100.0
            self.state_tracker.update_task_status(task.task_id, EvaluationStatus.RUNNING, progress=progress)
        
        # Generate mock results
        results = {
            'model_id': task.model_id,
            'suite_name': task.suite_name,
            'policy_mode': task.policy_mode.value,
            'num_episodes': num_episodes,
            'success_rate': random.uniform(0.7, 0.95),
            'mean_reward': random.uniform(0.6, 0.9),
            'mean_episode_length': random.uniform(500, 800),
            'mean_lateral_deviation': random.uniform(0.05, 0.2),
            'mean_heading_error': random.uniform(2.0, 10.0),
            'mean_jerk': random.uniform(0.02, 0.1),
            'stability': random.uniform(0.8, 0.95),
            'evaluation_time': evaluation_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _save_task_results(self, task: EvaluationTask, results: Dict[str, Any]):
        """Save task results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task_{task.task_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Combine task info and results
        full_results = {
            'task_info': asdict(task),
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        self.logger.info(f"Saved results: {filepath}")
    
    def get_progress(self) -> EvaluationProgress:
        """Get current evaluation progress."""
        return self.state_tracker.get_progress()
    
    def add_progress_callback(self, callback: Callable[[EvaluationProgress], None]):
        """Add a progress callback function."""
        self.state_tracker.add_progress_callback(callback)
    
    def stop_evaluation(self):
        """Stop all running evaluations."""
        self.logger.info("Stopping evaluation...")
        self._shutdown = True
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # Update running tasks to cancelled
        running_tasks = self.state_tracker.get_tasks_by_status(EvaluationStatus.RUNNING)
        for task in running_tasks:
            self.state_tracker.update_task_status(task.task_id, EvaluationStatus.CANCELLED)
        
        self.logger.info("Evaluation stopped")
    
    def get_results(self, model_id: Optional[str] = None, 
                   suite_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation results with optional filtering.
        
        Args:
            model_id: Optional model ID filter
            suite_name: Optional suite name filter
            
        Returns:
            List[Dict[str, Any]]: List of evaluation results
        """
        results = []
        
        # Get completed tasks
        completed_tasks = self.state_tracker.get_tasks_by_status(EvaluationStatus.COMPLETED)
        completed_tasks.extend([
            t for t in self.state_tracker.task_history 
            if t.status == EvaluationStatus.COMPLETED
        ])
        
        for task in completed_tasks:
            # Apply filters
            if model_id and task.model_id != model_id:
                continue
            if suite_name and task.suite_name != suite_name:
                continue
            
            if task.results:
                results.append({
                    'task_id': task.task_id,
                    'model_id': task.model_id,
                    'suite_name': task.suite_name,
                    'policy_mode': task.policy_mode.value,
                    'results': task.results
                })
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_evaluation()
        self.state_tracker.clear_completed_tasks()
        self.model_registry.clear()
        self.seed_manager.clear_cache()
        self.logger.info("Orchestrator cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()