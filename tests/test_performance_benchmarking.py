#!/usr/bin/env python3
"""
🧪 PERFORMANCE BENCHMARKING TESTS 🧪
Performance benchmarking tests for evaluation throughput and scalability

This module tests the performance characteristics of the evaluation system,
including throughput benchmarks, memory usage validation, and scalability testing.

Requirements covered: 8.4, 13.3, 13.4
"""

import time
import psutil
import threading
import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.evaluation_orchestrator import (
    EvaluationOrchestrator, ModelRegistry, SeedManager, EvaluationStateTracker
)
from duckietown_utils.suite_manager import SuiteManager, SuiteResults, SuiteType, EpisodeResult
from duckietown_utils.metrics_calculator import MetricsCalculator
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer
from duckietown_utils.failure_analyzer import FailureAnalyzer
from duckietown_utils.robustness_analyzer import RobustnessAnalyzer


class PerformanceMonitor:
    """Utility class for monitoring performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if self.start_time is None:
            raise RuntimeError("Monitoring not started")
        
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        
        return {
            'elapsed_time': current_time - self.start_time,
            'memory_usage_mb': current_memory,
            'memory_increase_mb': current_memory - self.start_memory,
            'cpu_time': current_cpu_time - self.start_cpu_time,
            'peak_memory_mb': self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else current_memory
        }


class TestEvaluationThroughputBenchmarks:
    """Test suite for evaluation throughput benchmarks."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def performance_config(self, temp_workspace):
        """Create performance-optimized configuration."""
        return {
            'results_dir': temp_workspace,
            'max_concurrent_evaluations': 4,
            'default_seeds_per_suite': 20,
            'base_seed': 42,
            'bootstrap_samples': 1000,  # Reduced for performance testing
            'timeout_per_episode': 10.0,
            'save_artifacts': False,  # Disable for performance testing
            'generate_reports': False
        }
    
    @pytest.fixture
    def mock_models(self, temp_workspace):
        """Create mock models for performance testing."""
        models = {}
        for i in range(10):  # Create 10 models for scalability testing
            model_path = Path(temp_workspace) / f"model_{i}.pth"
            model_path.touch()
            models[f"model_{i}"] = {
                'path': str(model_path),
                'performance': 0.8 + (i % 3) * 0.05  # Varying performance
            }
        return models
    
    def test_model_registration_throughput(self, temp_workspace, mock_models):
        """Test throughput of model registration operations."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        registry = ModelRegistry()
        
        # Register models sequentially
        start_time = time.time()
        model_ids = []
        
        for model_name, model_info in mock_models.items():
            model_id = registry.register_model(
                model_info['path'],
                model_type="pytorch",
                model_id=model_name,
                metadata={'performance': model_info['performance']}
            )
            model_ids.append(model_id)
        
        registration_time = time.time() - start_time
        metrics = monitor.get_metrics()
        
        # Performance assertions
        models_per_second = len(mock_models) / registration_time
        assert models_per_second >= 50, f"Registration rate {models_per_second:.1f} models/sec too slow"
        assert metrics['memory_increase_mb'] < 10, f"Memory increase {metrics['memory_increase_mb']:.1f} MB too high"
        
        # Verify all models registered correctly
        assert len(model_ids) == len(mock_models)
        assert len(set(model_ids)) == len(model_ids)  # All unique
    
    def test_concurrent_model_registration(self, temp_workspace, mock_models):
        """Test concurrent model registration performance."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        registry = ModelRegistry()
        
        def register_model(model_name, model_info):
            return registry.register_model(
                model_info['path'],
                model_type="pytorch",
                model_id=model_name,
                metadata={'performance': model_info['performance']}
            )
        
        # Register models concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(register_model, name, info): name
                for name, info in mock_models.items()
            }
            
            model_ids = []
            for future in as_completed(futures):
                model_id = future.result()
                model_ids.append(model_id)
        
        concurrent_time = time.time() - start_time
        metrics = monitor.get_metrics()
        
        # Concurrent registration should be faster than sequential
        models_per_second = len(mock_models) / concurrent_time
        assert models_per_second >= 100, f"Concurrent registration rate {models_per_second:.1f} models/sec too slow"
        assert metrics['memory_increase_mb'] < 15, "Memory usage too high for concurrent registration"
        
        # All models should be registered
        assert len(model_ids) == len(mock_models)
        assert len(set(model_ids)) == len(model_ids)
    
    def test_task_scheduling_performance(self, performance_config, mock_models):
        """Test performance of evaluation task scheduling."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        orchestrator = EvaluationOrchestrator(performance_config)
        
        # Register models
        model_ids = []
        for model_name, model_info in mock_models.items():
            model_id = orchestrator.register_model(model_info['path'], model_id=model_name)
            model_ids.append(model_id)
        
        # Schedule large number of tasks
        start_time = time.time()
        
        task_ids = orchestrator.schedule_evaluation(
            model_ids=model_ids,
            suite_names=["base", "hard_randomization", "law_intersection"],
            policy_modes=["deterministic", "stochastic"],
            seeds_per_suite=20
        )
        
        scheduling_time = time.time() - start_time
        metrics = monitor.get_metrics()
        
        # Performance assertions
        expected_tasks = len(model_ids) * 3 * 2  # 10 models * 3 suites * 2 modes = 60 tasks
        assert len(task_ids) == expected_tasks
        
        tasks_per_second = len(task_ids) / scheduling_time
        assert tasks_per_second >= 500, f"Task scheduling rate {tasks_per_second:.1f} tasks/sec too slow"
        assert metrics['memory_increase_mb'] < 50, f"Memory increase {metrics['memory_increase_mb']:.1f} MB too high"
        
        orchestrator.cleanup()
    
    def test_suite_execution_throughput(self, performance_config):
        """Test throughput of suite execution."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        suite_manager = SuiteManager(performance_config)
        
        # Create mock model
        mock_model = Mock()
        mock_model.model_id = "performance_test_model"
        
        # Test with different episode counts
        episode_counts = [10, 20, 50, 100]
        throughput_results = []
        
        for episode_count in episode_counts:
            seeds = list(range(episode_count))
            
            with patch.object(suite_manager, '_simulate_episode') as mock_simulate:
                # Fast mock simulation
                def fast_simulation(episode_id, map_name, model, seed, suite_config, policy_mode):
                    return EpisodeResult(
                        episode_id=episode_id,
                        map_name=map_name,
                        seed=seed,
                        success=True,
                        reward=0.8,
                        episode_length=500,
                        lateral_deviation=0.1,
                        heading_error=5.0,
                        jerk=0.05,
                        stability=0.9
                    )
                
                mock_simulate.side_effect = fast_simulation
                
                # Measure execution time
                start_time = time.time()
                
                results = suite_manager.run_suite("base", mock_model, seeds)
                
                execution_time = time.time() - start_time
                episodes_per_second = episode_count / execution_time
                throughput_results.append(episodes_per_second)
                
                # Validate results
                assert results.total_episodes == episode_count
                assert len(results.episode_results) == episode_count
        
        # Throughput should be reasonable and scale appropriately
        min_throughput = min(throughput_results)
        assert min_throughput >= 100, f"Minimum throughput {min_throughput:.1f} episodes/sec too slow"
        
        metrics = monitor.get_metrics()
        assert metrics['memory_increase_mb'] < 100, "Memory usage too high for suite execution"
    
    def test_metrics_calculation_performance(self, performance_config):
        """Test performance of metrics calculation."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        calculator = MetricsCalculator(performance_config)
        
        # Create large dataset of episode results
        episode_counts = [100, 500, 1000, 2000]
        calculation_times = []
        
        for episode_count in episode_counts:
            # Generate episode results
            episodes = []
            for i in range(episode_count):
                episode = EpisodeResult(
                    episode_id=f"episode_{i}",
                    map_name=f"map_{i % 5}",
                    seed=i,
                    success=i % 4 != 0,  # 75% success rate
                    reward=0.8 if i % 4 != 0 else 0.3,
                    episode_length=500 + (i % 100),
                    lateral_deviation=0.1 + (i % 10) * 0.01,
                    heading_error=5.0 + (i % 5),
                    jerk=0.05 + (i % 8) * 0.005,
                    stability=0.9 - (i % 6) * 0.02
                )
                episodes.append(episode)
            
            # Measure calculation time
            start_time = time.time()
            
            metrics = calculator.aggregate_episode_metrics(episodes)
            
            calculation_time = time.time() - start_time
            calculation_times.append(calculation_time)
            
            # Validate metrics
            assert len(metrics) > 0
            assert 'success_rate' in metrics
            assert 'mean_reward' in metrics
            
            # Performance assertion
            episodes_per_second = episode_count / calculation_time
            assert episodes_per_second >= 1000, \
                f"Metrics calculation rate {episodes_per_second:.1f} episodes/sec too slow for {episode_count} episodes"
        
        # Calculation time should scale reasonably
        # Time complexity should be roughly linear
        for i in range(1, len(calculation_times)):
            time_ratio = calculation_times[i] / calculation_times[i-1]
            episode_ratio = episode_counts[i] / episode_counts[i-1]
            
            # Time ratio should not be much larger than episode ratio
            assert time_ratio <= episode_ratio * 1.5, \
                f"Metrics calculation doesn't scale linearly: time ratio {time_ratio:.2f} vs episode ratio {episode_ratio:.2f}"
        
        metrics = monitor.get_metrics()
        assert metrics['memory_increase_mb'] < 200, "Memory usage too high for metrics calculation"
    
    def test_statistical_analysis_performance(self, performance_config):
        """Test performance of statistical analysis operations."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        analyzer = StatisticalAnalyzer(performance_config)
        
        # Test confidence interval calculation performance
        sample_sizes = [100, 500, 1000, 2000]
        ci_times = []
        
        for sample_size in sample_sizes:
            np.random.seed(42)
            sample_data = np.random.normal(10, 2, sample_size)
            
            start_time = time.time()
            
            ci = analyzer.compute_confidence_intervals(sample_data, method="bootstrap")
            
            ci_time = time.time() - start_time
            ci_times.append(ci_time)
            
            # Validate CI
            assert ci.lower < ci.upper
            assert ci.confidence_level == 0.95
            
            # Performance assertion
            samples_per_second = sample_size / ci_time
            assert samples_per_second >= 10000, \
                f"CI calculation rate {samples_per_second:.1f} samples/sec too slow for {sample_size} samples"
        
        # Test comparison performance
        comparison_times = []
        
        for sample_size in sample_sizes:
            np.random.seed(42)
            sample_a = np.random.normal(10, 2, sample_size)
            sample_b = np.random.normal(11, 2, sample_size)
            
            start_time = time.time()
            
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                "model_a", "model_b", "test_metric"
            )
            
            comparison_time = time.time() - start_time
            comparison_times.append(comparison_time)
            
            # Validate comparison
            assert comparison.p_value is not None
            assert comparison.effect_size is not None
            
            # Performance assertion
            comparisons_per_second = 1 / comparison_time
            assert comparisons_per_second >= 10, \
                f"Comparison rate {comparisons_per_second:.1f} comparisons/sec too slow for {sample_size} samples"
        
        metrics = monitor.get_metrics()
        assert metrics['memory_increase_mb'] < 100, "Memory usage too high for statistical analysis"
    
    def test_concurrent_evaluation_throughput(self, performance_config, mock_models):
        """Test throughput of concurrent evaluations."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        orchestrator = EvaluationOrchestrator(performance_config)
        
        # Register subset of models for concurrent testing
        test_models = dict(list(mock_models.items())[:4])  # Use 4 models
        model_ids = []
        
        for model_name, model_info in test_models.items():
            model_id = orchestrator.register_model(model_info['path'], model_id=model_name)
            model_ids.append(model_id)
        
        # Schedule evaluations
        task_ids = orchestrator.schedule_evaluation(
            model_ids=model_ids,
            suite_names=["base", "hard_randomization"],
            policy_modes=["deterministic"],
            seeds_per_suite=10
        )
        
        # Mock fast suite execution
        with patch.object(orchestrator.suite_manager, 'run_suite') as mock_run_suite:
            def fast_suite_execution(suite_name, model, seeds, **kwargs):
                time.sleep(0.1)  # Simulate 100ms execution time
                return SuiteResults(
                    suite_name=suite_name,
                    suite_type=SuiteType.BASE,
                    model_id=model.model_id,
                    policy_mode="deterministic",
                    total_episodes=len(seeds),
                    successful_episodes=int(len(seeds) * 0.8),
                    episode_results=[]
                )
            
            mock_run_suite.side_effect = fast_suite_execution
            
            # Execute evaluations concurrently
            start_time = time.time()
            
            def execute_task(task_id):
                task = orchestrator.state_tracker.get_task(task_id)
                return orchestrator.suite_manager.run_suite(
                    task.suite_name,
                    Mock(model_id=task.model_id),
                    task.seeds
                )
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(execute_task, task_id) for task_id in task_ids]
                results = [future.result() for future in as_completed(futures)]
            
            execution_time = time.time() - start_time
        
        # Performance assertions
        total_tasks = len(task_ids)
        tasks_per_second = total_tasks / execution_time
        
        # With 4 workers and 100ms per task, should achieve ~40 tasks/sec theoretical max
        # Allow for overhead
        assert tasks_per_second >= 20, f"Concurrent execution rate {tasks_per_second:.1f} tasks/sec too slow"
        
        # All tasks should complete successfully
        assert len(results) == total_tasks
        
        metrics = monitor.get_metrics()
        assert metrics['memory_increase_mb'] < 150, "Memory usage too high for concurrent evaluation"
        
        orchestrator.cleanup()
    
    def test_memory_efficiency_large_datasets(self, performance_config):
        """Test memory efficiency with large datasets."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Test with progressively larger datasets
        dataset_sizes = [1000, 5000, 10000, 20000]
        memory_usage = []
        
        for dataset_size in dataset_sizes:
            # Create large episode dataset
            episodes = []
            for i in range(dataset_size):
                episode = EpisodeResult(
                    episode_id=f"episode_{i}",
                    map_name=f"map_{i % 10}",
                    seed=i,
                    success=i % 3 != 0,
                    reward=0.8 if i % 3 != 0 else 0.3,
                    episode_length=500,
                    lateral_deviation=0.1,
                    heading_error=5.0,
                    jerk=0.05,
                    stability=0.9
                )
                episodes.append(episode)
            
            # Measure memory usage
            current_memory = monitor.get_metrics()['memory_usage_mb']
            memory_usage.append(current_memory)
            
            # Process dataset
            calculator = MetricsCalculator(performance_config)
            metrics = calculator.aggregate_episode_metrics(episodes)
            
            # Validate processing
            assert len(metrics) > 0
            
            # Clean up
            del episodes
            del metrics
            del calculator
            import gc
            gc.collect()
        
        # Memory usage should scale reasonably
        max_memory_increase = max(memory_usage) - memory_usage[0]
        assert max_memory_increase < 500, f"Memory increase {max_memory_increase:.1f} MB too high for large datasets"
        
        # Memory usage per episode should be reasonable
        memory_per_episode = max_memory_increase / max(dataset_sizes)
        assert memory_per_episode < 0.05, f"Memory per episode {memory_per_episode:.3f} MB too high"
    
    def test_cpu_utilization_efficiency(self, performance_config):
        """Test CPU utilization efficiency."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Create computationally intensive task
        calculator = MetricsCalculator(performance_config)
        analyzer = StatisticalAnalyzer(performance_config)
        
        # Generate large dataset
        episodes = []
        for i in range(5000):
            episode = EpisodeResult(
                episode_id=f"episode_{i}",
                map_name=f"map_{i % 5}",
                seed=i,
                success=i % 4 != 0,
                reward=0.8 if i % 4 != 0 else 0.3,
                episode_length=500,
                lateral_deviation=0.1,
                heading_error=5.0,
                jerk=0.05,
                stability=0.9
            )
            episodes.append(episode)
        
        # Perform CPU-intensive operations
        start_time = time.time()
        
        # Metrics calculation
        metrics = calculator.aggregate_episode_metrics(episodes)
        
        # Statistical analysis
        success_data = [ep.reward for ep in episodes if ep.success]
        failure_data = [ep.reward for ep in episodes if not ep.success]
        
        comparison = analyzer.compare_models(
            success_data, failure_data,
            "success", "failure", "reward"
        )
        
        # Bootstrap confidence intervals
        ci = analyzer.compute_confidence_intervals(success_data, method="bootstrap")
        
        execution_time = time.time() - start_time
        final_metrics = monitor.get_metrics()
        
        # CPU efficiency assertions
        cpu_utilization = final_metrics['cpu_time'] / execution_time
        assert cpu_utilization >= 0.5, f"CPU utilization {cpu_utilization:.2f} too low"
        
        # Operations should complete in reasonable time
        operations_per_second = 3 / execution_time  # 3 major operations
        assert operations_per_second >= 1, f"Operation rate {operations_per_second:.2f} ops/sec too slow"


class TestScalabilityBenchmarks:
    """Test suite for scalability benchmarks."""
    
    def test_model_count_scalability(self, temp_workspace):
        """Test scalability with increasing number of models."""
        model_counts = [10, 25, 50, 100]
        registration_times = []
        memory_usage = []
        
        for model_count in model_counts:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            registry = ModelRegistry()
            
            # Create and register models
            start_time = time.time()
            
            for i in range(model_count):
                model_path = Path(temp_workspace) / f"model_{i}.pth"
                model_path.touch()
                
                registry.register_model(
                    str(model_path),
                    model_type="pytorch",
                    model_id=f"model_{i}"
                )
            
            registration_time = time.time() - start_time
            registration_times.append(registration_time)
            
            metrics = monitor.get_metrics()
            memory_usage.append(metrics['memory_usage_mb'])
            
            # Clean up
            registry.clear()
            del registry
        
        # Registration time should scale linearly
        for i in range(1, len(registration_times)):
            time_ratio = registration_times[i] / registration_times[i-1]
            model_ratio = model_counts[i] / model_counts[i-1]
            
            # Time scaling should be reasonable
            assert time_ratio <= model_ratio * 1.5, \
                f"Registration time doesn't scale linearly: {time_ratio:.2f} vs {model_ratio:.2f}"
        
        # Memory usage should scale reasonably
        memory_per_model = (memory_usage[-1] - memory_usage[0]) / (model_counts[-1] - model_counts[0])
        assert memory_per_model < 1.0, f"Memory per model {memory_per_model:.3f} MB too high"
    
    def test_episode_count_scalability(self, temp_workspace):
        """Test scalability with increasing episode counts."""
        episode_counts = [100, 500, 1000, 5000]
        processing_times = []
        
        config = {'results_dir': temp_workspace}
        calculator = MetricsCalculator(config)
        
        for episode_count in episode_counts:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Generate episodes
            episodes = []
            for i in range(episode_count):
                episode = EpisodeResult(
                    episode_id=f"episode_{i}",
                    map_name=f"map_{i % 3}",
                    seed=i,
                    success=i % 3 != 0,
                    reward=0.8 if i % 3 != 0 else 0.3,
                    episode_length=500,
                    lateral_deviation=0.1,
                    heading_error=5.0,
                    jerk=0.05,
                    stability=0.9
                )
                episodes.append(episode)
            
            # Process episodes
            start_time = time.time()
            
            metrics = calculator.aggregate_episode_metrics(episodes)
            per_map_metrics = calculator.calculate_per_map_metrics(episodes)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Validate results
            assert len(metrics) > 0
            assert len(per_map_metrics) == 3  # 3 maps
            
            # Performance assertion
            episodes_per_second = episode_count / processing_time
            assert episodes_per_second >= 500, \
                f"Processing rate {episodes_per_second:.1f} episodes/sec too slow for {episode_count} episodes"
        
        # Processing time should scale sub-quadratically
        for i in range(1, len(processing_times)):
            time_ratio = processing_times[i] / processing_times[i-1]
            episode_ratio = episode_counts[i] / episode_counts[i-1]
            
            # Should be roughly linear scaling
            assert time_ratio <= episode_ratio * 1.5, \
                f"Processing time scaling too poor: {time_ratio:.2f} vs {episode_ratio:.2f}"
    
    def test_concurrent_task_scalability(self, temp_workspace):
        """Test scalability of concurrent task execution."""
        worker_counts = [1, 2, 4, 8]
        task_count = 32  # Fixed number of tasks
        execution_times = []
        
        for worker_count in worker_counts:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            def mock_task(task_id):
                """Mock task that takes some time."""
                time.sleep(0.1)  # 100ms per task
                return f"result_{task_id}"
            
            # Execute tasks concurrently
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(mock_task, i) for i in range(task_count)]
                results = [future.result() for future in as_completed(futures)]
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Validate results
            assert len(results) == task_count
            
            metrics = monitor.get_metrics()
            
            # Performance assertions
            tasks_per_second = task_count / execution_time
            theoretical_max = worker_count * 10  # 10 tasks/sec per worker
            
            # Should achieve reasonable fraction of theoretical maximum
            efficiency = tasks_per_second / theoretical_max
            assert efficiency >= 0.7, \
                f"Concurrency efficiency {efficiency:.2f} too low with {worker_count} workers"
        
        # Execution time should decrease with more workers (up to a point)
        # First few increases should show improvement
        assert execution_times[1] < execution_times[0] * 0.8, "No speedup with 2 workers"
        assert execution_times[2] < execution_times[1] * 0.8, "No speedup with 4 workers"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])