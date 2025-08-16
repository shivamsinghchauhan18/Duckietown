#!/usr/bin/env python3
"""
Unit tests for the Evaluation Orchestrator infrastructure.

Tests cover model registry, seed management, state tracking, and orchestrator coordination.
"""

import os
import sys
import time
import json
import tempfile
import threading
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.evaluation_orchestrator import (
    EvaluationOrchestrator, ModelRegistry, SeedManager, EvaluationStateTracker,
    ModelInfo, EvaluationTask, EvaluationProgress, EvaluationStatus, PolicyMode
)

class TestModelRegistry(TestCase):
    """Test cases for ModelRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create temporary model files
        self.model_file1 = Path(self.temp_dir) / "model1.pth"
        self.model_file2 = Path(self.temp_dir) / "model2.onnx"
        self.model_file1.touch()
        self.model_file2.touch()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_register_model_success(self):
        """Test successful model registration."""
        model_id = self.registry.register_model(
            model_path=str(self.model_file1),
            model_type="pytorch",
            metadata={"version": "1.0"}
        )
        
        self.assertIsNotNone(model_id)
        self.assertIn(model_id, self.registry.models)
        
        model_info = self.registry.get_model(model_id)
        self.assertEqual(model_info.model_path, str(self.model_file1))
        self.assertEqual(model_info.model_type, "pytorch")
        self.assertEqual(model_info.metadata["version"], "1.0")
    
    def test_register_model_custom_id(self):
        """Test model registration with custom ID."""
        custom_id = "my_custom_model"
        model_id = self.registry.register_model(
            model_path=str(self.model_file1),
            model_id=custom_id
        )
        
        self.assertEqual(model_id, custom_id)
        self.assertIn(custom_id, self.registry.models)
    
    def test_register_model_duplicate_id(self):
        """Test registration with duplicate ID raises error."""
        model_id = "duplicate_id"
        self.registry.register_model(str(self.model_file1), model_id=model_id)
        
        with self.assertRaises(ValueError):
            self.registry.register_model(str(self.model_file2), model_id=model_id)
    
    def test_register_model_nonexistent_file(self):
        """Test registration with nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.registry.register_model("/nonexistent/path/model.pth")
    
    def test_list_models(self):
        """Test listing all registered models."""
        id1 = self.registry.register_model(str(self.model_file1))
        id2 = self.registry.register_model(str(self.model_file2))
        
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)
        
        model_ids = [m.model_id for m in models]
        self.assertIn(id1, model_ids)
        self.assertIn(id2, model_ids)
    
    def test_remove_model(self):
        """Test model removal."""
        model_id = self.registry.register_model(str(self.model_file1))
        
        self.assertTrue(self.registry.remove_model(model_id))
        self.assertIsNone(self.registry.get_model(model_id))
        self.assertFalse(self.registry.remove_model(model_id))  # Already removed
    
    def test_clear_registry(self):
        """Test clearing all models."""
        self.registry.register_model(str(self.model_file1))
        self.registry.register_model(str(self.model_file2))
        
        self.assertEqual(len(self.registry.models), 2)
        
        self.registry.clear()
        self.assertEqual(len(self.registry.models), 0)
    
    def test_thread_safety(self):
        """Test thread safety of model registry."""
        results = []
        errors = []
        
        def register_models():
            try:
                for i in range(10):
                    model_id = self.registry.register_model(
                        str(self.model_file1),
                        model_id=f"thread_model_{threading.current_thread().ident}_{i}"
                    )
                    results.append(model_id)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=register_models)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 50)  # 5 threads * 10 models each
        self.assertEqual(len(set(results)), 50)  # All unique IDs

class TestSeedManager(TestCase):
    """Test cases for SeedManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed_manager = SeedManager(base_seed=42)
    
    def test_generate_deterministic_seeds(self):
        """Test deterministic seed generation."""
        seeds1 = self.seed_manager.generate_seeds("test_suite", 10, deterministic=True)
        seeds2 = self.seed_manager.generate_seeds("test_suite", 10, deterministic=True)
        
        self.assertEqual(len(seeds1), 10)
        self.assertEqual(len(seeds2), 10)
        self.assertEqual(seeds1, seeds2)  # Should be identical
    
    def test_generate_different_suite_seeds(self):
        """Test that different suites generate different seeds."""
        seeds1 = self.seed_manager.generate_seeds("suite1", 10, deterministic=True)
        seeds2 = self.seed_manager.generate_seeds("suite2", 10, deterministic=True)
        
        self.assertNotEqual(seeds1, seeds2)
    
    def test_generate_random_seeds(self):
        """Test random seed generation."""
        # Create separate seed managers to avoid caching
        seed_manager1 = SeedManager(base_seed=42)
        seed_manager2 = SeedManager(base_seed=123)
        
        seeds1 = seed_manager1.generate_seeds("test_suite", 10, deterministic=False)
        seeds2 = seed_manager2.generate_seeds("test_suite", 10, deterministic=False)
        
        self.assertEqual(len(seeds1), 10)
        self.assertEqual(len(seeds2), 10)
        # Random seeds should be different (with very high probability)
        self.assertNotEqual(seeds1, seeds2)
    
    def test_seed_caching(self):
        """Test seed caching functionality."""
        # First call should generate and cache
        seeds1 = self.seed_manager.generate_seeds("cached_suite", 5, deterministic=True)
        
        # Second call should return cached seeds
        seeds2 = self.seed_manager.generate_seeds("cached_suite", 5, deterministic=True)
        
        self.assertEqual(seeds1, seeds2)
        self.assertIn("cached_suite_5_True", self.seed_manager.seed_cache)
    
    def test_get_reproducible_seeds(self):
        """Test reproducible seed generation."""
        seeds1 = self.seed_manager.get_reproducible_seeds("repro_suite", 8)
        seeds2 = self.seed_manager.get_reproducible_seeds("repro_suite", 8)
        
        self.assertEqual(seeds1, seeds2)
        self.assertEqual(len(seeds1), 8)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        self.seed_manager.generate_seeds("test_suite", 5, deterministic=True)
        self.assertGreater(len(self.seed_manager.seed_cache), 0)
        
        self.seed_manager.clear_cache()
        self.assertEqual(len(self.seed_manager.seed_cache), 0)

class TestEvaluationStateTracker(TestCase):
    """Test cases for EvaluationStateTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = EvaluationStateTracker()
        self.test_task = EvaluationTask(
            task_id="test_task_1",
            model_id="test_model",
            suite_name="test_suite",
            policy_mode=PolicyMode.DETERMINISTIC,
            seeds=[1, 2, 3, 4, 5]
        )
    
    def test_add_task(self):
        """Test adding evaluation tasks."""
        self.tracker.add_task(self.test_task)
        
        self.assertIn(self.test_task.task_id, self.tracker.tasks)
        retrieved_task = self.tracker.get_task(self.test_task.task_id)
        self.assertEqual(retrieved_task.task_id, self.test_task.task_id)
    
    def test_update_task_status(self):
        """Test updating task status."""
        self.tracker.add_task(self.test_task)
        
        # Update to running
        self.tracker.update_task_status(
            self.test_task.task_id, 
            EvaluationStatus.RUNNING, 
            progress=50.0
        )
        
        task = self.tracker.get_task(self.test_task.task_id)
        self.assertEqual(task.status, EvaluationStatus.RUNNING)
        self.assertEqual(task.progress, 50.0)
        self.assertIsNotNone(task.start_time)
        
        # Update to completed
        self.tracker.update_task_status(
            self.test_task.task_id, 
            EvaluationStatus.COMPLETED, 
            progress=100.0
        )
        
        task = self.tracker.get_task(self.test_task.task_id)
        self.assertEqual(task.status, EvaluationStatus.COMPLETED)
        self.assertEqual(task.progress, 100.0)
        self.assertIsNotNone(task.end_time)
    
    def test_update_task_results(self):
        """Test updating task results."""
        self.tracker.add_task(self.test_task)
        
        results = {"success_rate": 0.85, "mean_reward": 0.72}
        self.tracker.update_task_results(self.test_task.task_id, results)
        
        task = self.tracker.get_task(self.test_task.task_id)
        self.assertEqual(task.results, results)
    
    def test_get_tasks_by_status(self):
        """Test filtering tasks by status."""
        # Add multiple tasks with different statuses
        task1 = EvaluationTask("task1", "model1", "suite1", PolicyMode.DETERMINISTIC, [1, 2])
        task2 = EvaluationTask("task2", "model2", "suite2", PolicyMode.STOCHASTIC, [3, 4])
        task3 = EvaluationTask("task3", "model3", "suite3", PolicyMode.DETERMINISTIC, [5, 6])
        
        self.tracker.add_task(task1)
        self.tracker.add_task(task2)
        self.tracker.add_task(task3)
        
        # Update statuses
        self.tracker.update_task_status(task1.task_id, EvaluationStatus.RUNNING)
        self.tracker.update_task_status(task2.task_id, EvaluationStatus.COMPLETED)
        # task3 remains PENDING
        
        # Test filtering
        pending_tasks = self.tracker.get_tasks_by_status(EvaluationStatus.PENDING)
        running_tasks = self.tracker.get_tasks_by_status(EvaluationStatus.RUNNING)
        completed_tasks = self.tracker.get_tasks_by_status(EvaluationStatus.COMPLETED)
        
        self.assertEqual(len(pending_tasks), 1)
        self.assertEqual(len(running_tasks), 1)
        self.assertEqual(len(completed_tasks), 1)
        
        self.assertEqual(pending_tasks[0].task_id, task3.task_id)
        self.assertEqual(running_tasks[0].task_id, task1.task_id)
        self.assertEqual(completed_tasks[0].task_id, task2.task_id)
    
    def test_get_progress(self):
        """Test progress calculation."""
        # Add tasks with different statuses
        tasks = []
        for i in range(10):
            task = EvaluationTask(f"task_{i}", f"model_{i}", "suite", PolicyMode.DETERMINISTIC, [1])
            tasks.append(task)
            self.tracker.add_task(task)
        
        # Set various statuses
        for i in range(3):
            self.tracker.update_task_status(tasks[i].task_id, EvaluationStatus.COMPLETED)
        for i in range(3, 5):
            self.tracker.update_task_status(tasks[i].task_id, EvaluationStatus.RUNNING)
        for i in range(5, 6):
            self.tracker.update_task_status(tasks[i].task_id, EvaluationStatus.FAILED)
        # Remaining tasks stay PENDING
        
        progress = self.tracker.get_progress()
        
        self.assertEqual(progress.total_tasks, 10)
        self.assertEqual(progress.completed_tasks, 3)
        self.assertEqual(progress.running_tasks, 2)
        self.assertEqual(progress.failed_tasks, 1)
        self.assertEqual(progress.pending_tasks, 4)
        self.assertEqual(progress.overall_progress, 30.0)  # 3/10 * 100
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def test_callback(progress):
            callback_calls.append(progress)
        
        self.tracker.add_progress_callback(test_callback)
        
        # Test callback registration
        self.assertEqual(len(self.tracker._progress_callbacks), 1)
        
        # Directly call _notify_progress_callbacks to test the mechanism
        # Set last update time to allow callback
        self.tracker._last_progress_update = 0.0
        
        # Add a task to have some progress to report
        task = EvaluationTask("callback_task", "model", "suite", PolicyMode.DETERMINISTIC, [1])
        self.tracker.add_task(task)
        
        # Directly trigger callback notification
        self.tracker._notify_progress_callbacks()
        
        # Should have received at least one callback
        self.assertGreater(len(callback_calls), 0)
        self.assertIsInstance(callback_calls[0], EvaluationProgress)
    
    def test_clear_completed_tasks(self):
        """Test clearing completed tasks."""
        # Add tasks
        task1 = EvaluationTask("task1", "model1", "suite", PolicyMode.DETERMINISTIC, [1])
        task2 = EvaluationTask("task2", "model2", "suite", PolicyMode.DETERMINISTIC, [2])
        
        self.tracker.add_task(task1)
        self.tracker.add_task(task2)
        
        # Complete one task
        self.tracker.update_task_status(task1.task_id, EvaluationStatus.COMPLETED)
        
        # Both tasks should still be in active tasks (completed tasks remain until cleared)
        self.assertEqual(len(self.tracker.tasks), 2)
        self.assertEqual(len(self.tracker.task_history), 0)
        
        # Clear completed tasks
        self.tracker.clear_completed_tasks()
        
        self.assertEqual(len(self.tracker.tasks), 1)  # Only pending task remains
        self.assertEqual(len(self.tracker.task_history), 1)  # Completed task moved to history

class TestEvaluationOrchestrator(TestCase):
    """Test cases for EvaluationOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        config = {
            'results_dir': self.temp_dir,
            'max_concurrent_evaluations': 2,
            'default_seeds_per_suite': 5,
            'base_seed': 123
        }
        
        self.orchestrator = EvaluationOrchestrator(config)
        
        # Create temporary model file
        self.model_file = Path(self.temp_dir) / "test_model.pth"
        self.model_file.touch()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.orchestrator.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_register_model(self):
        """Test model registration through orchestrator."""
        model_id = self.orchestrator.register_model(
            str(self.model_file),
            model_type="pytorch",
            metadata={"test": True}
        )
        
        self.assertIsNotNone(model_id)
        model_info = self.orchestrator.model_registry.get_model(model_id)
        self.assertEqual(model_info.model_type, "pytorch")
        self.assertTrue(model_info.metadata["test"])
    
    def test_schedule_evaluation_single_model(self):
        """Test scheduling evaluation for single model."""
        model_id = self.orchestrator.register_model(str(self.model_file))
        
        task_ids = self.orchestrator.schedule_evaluation(
            model_ids=model_id,
            suite_names="base",
            policy_modes=PolicyMode.DETERMINISTIC
        )
        
        self.assertEqual(len(task_ids), 1)
        
        task = self.orchestrator.state_tracker.get_task(task_ids[0])
        self.assertEqual(task.model_id, model_id)
        self.assertEqual(task.suite_name, "base")
        self.assertEqual(task.policy_mode, PolicyMode.DETERMINISTIC)
        self.assertEqual(len(task.seeds), 5)  # default_seeds_per_suite
    
    def test_schedule_evaluation_multiple_models_suites(self):
        """Test scheduling evaluation for multiple models and suites."""
        model_id1 = self.orchestrator.register_model(str(self.model_file), model_id="model1")
        model_id2 = self.orchestrator.register_model(str(self.model_file), model_id="model2")
        
        task_ids = self.orchestrator.schedule_evaluation(
            model_ids=[model_id1, model_id2],
            suite_names=["base", "hard_randomization"],
            policy_modes=[PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC]
        )
        
        # 2 models * 2 suites * 2 policy modes = 8 tasks
        self.assertEqual(len(task_ids), 8)
        
        # Check that all combinations are present
        model_ids = set()
        suite_names = set()
        policy_modes = set()
        
        for task_id in task_ids:
            task = self.orchestrator.state_tracker.get_task(task_id)
            model_ids.add(task.model_id)
            suite_names.add(task.suite_name)
            policy_modes.add(task.policy_mode)
        
        self.assertEqual(model_ids, {model_id1, model_id2})
        self.assertEqual(suite_names, {"base", "hard_randomization"})
        self.assertEqual(policy_modes, {PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC})
    
    def test_schedule_evaluation_invalid_model(self):
        """Test scheduling evaluation with invalid model ID."""
        with self.assertRaises(ValueError):
            self.orchestrator.schedule_evaluation(
                model_ids="nonexistent_model",
                suite_names="base"
            )
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_start_evaluation(self, mock_executor_class):
        """Test starting evaluation execution."""
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        
        # Register model and schedule tasks
        model_id = self.orchestrator.register_model(str(self.model_file))
        task_ids = self.orchestrator.schedule_evaluation(model_id, "base")
        
        # Start evaluation
        result = self.orchestrator.start_evaluation()
        
        self.assertTrue(result)
        mock_executor_class.assert_called_once_with(max_workers=2)
        self.assertEqual(mock_executor.submit.call_count, 2)  # 2 policy modes
    
    def test_get_progress(self):
        """Test getting evaluation progress."""
        # Register model and schedule tasks
        model_id = self.orchestrator.register_model(str(self.model_file))
        task_ids = self.orchestrator.schedule_evaluation(model_id, "base")
        
        progress = self.orchestrator.get_progress()
        
        self.assertEqual(progress.total_tasks, 2)  # 2 policy modes
        self.assertEqual(progress.pending_tasks, 2)
        self.assertEqual(progress.completed_tasks, 0)
        self.assertEqual(progress.overall_progress, 0.0)
    
    def test_get_results_empty(self):
        """Test getting results when no evaluations completed."""
        results = self.orchestrator.get_results()
        self.assertEqual(len(results), 0)
    
    def test_get_results_with_filter(self):
        """Test getting results with filtering."""
        # This would require completed tasks, which we'll mock
        model_id = self.orchestrator.register_model(str(self.model_file))
        task_ids = self.orchestrator.schedule_evaluation(model_id, ["base", "hard_randomization"])
        
        # Mock completed task - move it to history to avoid double counting
        task = self.orchestrator.state_tracker.get_task(task_ids[0])
        task.status = EvaluationStatus.COMPLETED
        task.results = {"success_rate": 0.85}
        
        # Move task to history and remove from active tasks
        self.orchestrator.state_tracker.task_history.append(task)
        del self.orchestrator.state_tracker.tasks[task.task_id]
        
        # Test filtering
        all_results = self.orchestrator.get_results()
        model_results = self.orchestrator.get_results(model_id=model_id)
        suite_results = self.orchestrator.get_results(suite_name="base")
        
        self.assertEqual(len(all_results), 1)
        self.assertEqual(len(model_results), 1)
        
        if task.suite_name == "base":
            self.assertEqual(len(suite_results), 1)
        else:
            self.assertEqual(len(suite_results), 0)
    
    def test_context_manager(self):
        """Test orchestrator as context manager."""
        with EvaluationOrchestrator() as orchestrator:
            self.assertIsInstance(orchestrator, EvaluationOrchestrator)
            model_id = orchestrator.register_model(str(self.model_file))
            self.assertIsNotNone(model_id)
        
        # After context exit, orchestrator should be cleaned up
        # (We can't easily test this without more complex mocking)

class TestIntegration(TestCase):
    """Integration tests for orchestrator components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_file = Path(self.temp_dir) / "integration_model.pth"
        self.model_file.touch()
        
        config = {
            'results_dir': self.temp_dir,
            'max_concurrent_evaluations': 1,
            'default_seeds_per_suite': 3,
            'base_seed': 456
        }
        
        self.orchestrator = EvaluationOrchestrator(config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.orchestrator.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow_simulation(self):
        """Test complete workflow from registration to results."""
        # Register model
        model_id = self.orchestrator.register_model(
            str(self.model_file),
            model_type="integration_test",
            metadata={"test_run": True}
        )
        
        # Schedule evaluation
        task_ids = self.orchestrator.schedule_evaluation(
            model_ids=model_id,
            suite_names="base",
            policy_modes=PolicyMode.DETERMINISTIC,
            seeds_per_suite=3
        )
        
        self.assertEqual(len(task_ids), 1)
        
        # Check initial progress
        progress = self.orchestrator.get_progress()
        self.assertEqual(progress.total_tasks, 1)
        self.assertEqual(progress.pending_tasks, 1)
        
        # Simulate task completion
        task = self.orchestrator.state_tracker.get_task(task_ids[0])
        self.orchestrator.state_tracker.update_task_status(
            task.task_id, 
            EvaluationStatus.COMPLETED,
            progress=100.0
        )
        
        mock_results = {
            "success_rate": 0.9,
            "mean_reward": 0.8,
            "episodes": 3
        }
        self.orchestrator.state_tracker.update_task_results(task.task_id, mock_results)
        
        # Check final progress
        progress = self.orchestrator.get_progress()
        self.assertEqual(progress.completed_tasks, 1)
        self.assertEqual(progress.overall_progress, 100.0)
        
        # Get results
        results = self.orchestrator.get_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["results"]["success_rate"], 0.9)
    
    def test_seed_reproducibility(self):
        """Test that seed generation is reproducible."""
        model_id = self.orchestrator.register_model(str(self.model_file))
        
        # Schedule same evaluation twice
        task_ids1 = self.orchestrator.schedule_evaluation(model_id, "base")
        task_ids2 = self.orchestrator.schedule_evaluation(model_id, "base")
        
        # Get seeds from both sets of tasks
        task1 = self.orchestrator.state_tracker.get_task(task_ids1[0])
        task2 = self.orchestrator.state_tracker.get_task(task_ids2[0])
        
        # Seeds should be identical for same suite
        self.assertEqual(task1.seeds, task2.seeds)
    
    def test_concurrent_task_management(self):
        """Test managing multiple concurrent tasks."""
        model_id = self.orchestrator.register_model(str(self.model_file))
        
        # Schedule multiple evaluations
        task_ids = self.orchestrator.schedule_evaluation(
            model_ids=model_id,
            suite_names=["base", "hard_randomization"],
            policy_modes=[PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC]
        )
        
        self.assertEqual(len(task_ids), 4)  # 2 suites * 2 modes
        
        # Simulate different completion states
        tasks = [self.orchestrator.state_tracker.get_task(tid) for tid in task_ids]
        
        self.orchestrator.state_tracker.update_task_status(tasks[0].task_id, EvaluationStatus.RUNNING)
        self.orchestrator.state_tracker.update_task_status(tasks[1].task_id, EvaluationStatus.COMPLETED)
        self.orchestrator.state_tracker.update_task_status(tasks[2].task_id, EvaluationStatus.FAILED)
        # tasks[3] remains PENDING
        
        # Check status distribution
        progress = self.orchestrator.get_progress()
        self.assertEqual(progress.running_tasks, 1)
        self.assertEqual(progress.completed_tasks, 1)
        self.assertEqual(progress.failed_tasks, 1)
        self.assertEqual(progress.pending_tasks, 1)

if __name__ == '__main__':
    import unittest
    unittest.main()