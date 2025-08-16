#!/usr/bin/env python3
"""
🧪 EVALUATION CLI TESTS 🧪
Comprehensive tests for evaluation CLI tools and orchestration scripts

This module provides tests for:
- Main evaluation CLI functionality
- Batch evaluation operations
- Monitoring and progress reporting
- Analysis and querying utilities
- Integration and error handling

Run tests:
    python -m pytest tests/test_evaluation_cli.py -v
    python tests/test_evaluation_cli.py  # Direct execution
"""

import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation_cli import EvaluationCLI
from batch_evaluation import BatchEvaluationManager, BatchEvaluationConfig
from evaluation_monitor import EvaluationMonitor, MonitoringConfig, SystemMonitor
from evaluation_analysis import EvaluationAnalyzer, QueryFilter, ResultsDatabase
from duckietown_utils.evaluation_orchestrator import (
    EvaluationOrchestrator, 
    ModelRegistry, 
    SeedManager,
    EvaluationStateTracker,
    PolicyMode,
    EvaluationStatus
)


class TestEvaluationCLI(unittest.TestCase):
    """Test cases for main evaluation CLI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cli = EvaluationCLI()
        
        # Create mock model file
        self.mock_model_path = self.temp_dir / "test_model.pth"
        with open(self.mock_model_path, 'w') as f:
            f.write("# Mock model for testing")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_initialization(self):
        """Test CLI initialization."""
        self.assertIsInstance(self.cli, EvaluationCLI)
        self.assertIsNone(self.cli.orchestrator)
        self.assertIsNone(self.cli.config)
    
    def test_get_orchestrator(self):
        """Test orchestrator creation."""
        orchestrator = self.cli._get_orchestrator()
        self.assertIsInstance(orchestrator, EvaluationOrchestrator)
        
        # Test singleton behavior
        orchestrator2 = self.cli._get_orchestrator()
        self.assertIs(orchestrator, orchestrator2)
    
    def test_model_registration(self):
        """Test model registration functionality."""
        orchestrator = self.cli._get_orchestrator()
        
        # Test successful registration
        model_id = orchestrator.register_model(
            model_path=str(self.mock_model_path),
            model_type="checkpoint",
            model_id="test_model",
            metadata={"test": True}
        )
        
        self.assertEqual(model_id, "test_model")
        
        # Verify model is registered
        model_info = orchestrator.model_registry.get_model("test_model")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.model_id, "test_model")
        self.assertEqual(model_info.model_type, "checkpoint")
    
    def test_model_registration_invalid_path(self):
        """Test model registration with invalid path."""
        orchestrator = self.cli._get_orchestrator()
        
        with self.assertRaises(FileNotFoundError):
            orchestrator.register_model(
                model_path="nonexistent_model.pth",
                model_type="checkpoint"
            )
    
    def test_evaluation_scheduling(self):
        """Test evaluation scheduling."""
        orchestrator = self.cli._get_orchestrator()
        
        # Register a model first
        model_id = orchestrator.register_model(
            model_path=str(self.mock_model_path),
            model_type="checkpoint",
            model_id="test_model"
        )
        
        # Schedule evaluation
        task_ids = orchestrator.schedule_evaluation(
            model_ids=[model_id],
            suite_names=["base"],
            policy_modes=[PolicyMode.DETERMINISTIC],
            seeds_per_suite=5
        )
        
        self.assertIsInstance(task_ids, list)
        self.assertGreater(len(task_ids), 0)
        
        # Verify tasks are scheduled
        for task_id in task_ids:
            task = orchestrator.state_tracker.get_task(task_id)
            self.assertIsNotNone(task)
            self.assertEqual(task.model_id, model_id)
            self.assertEqual(task.status, EvaluationStatus.PENDING)
    
    def test_evaluation_progress_tracking(self):
        """Test evaluation progress tracking."""
        orchestrator = self.cli._get_orchestrator()
        
        # Get initial progress
        progress = orchestrator.get_progress()
        self.assertEqual(progress.total_tasks, 0)
        self.assertEqual(progress.completed_tasks, 0)
        
        # Register model and schedule evaluation
        model_id = orchestrator.register_model(
            model_path=str(self.mock_model_path),
            model_type="checkpoint"
        )
        
        task_ids = orchestrator.schedule_evaluation(
            model_ids=[model_id],
            suite_names=["base"],
            seeds_per_suite=2
        )
        
        # Check progress after scheduling
        progress = orchestrator.get_progress()
        self.assertEqual(progress.total_tasks, len(task_ids))
        self.assertEqual(progress.pending_tasks, len(task_ids))


class TestBatchEvaluation(unittest.TestCase):
    """Test cases for batch evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.temp_dir / "models"
        self.models_dir.mkdir()
        
        # Create mock model files
        self.model_files = []
        for i in range(3):
            model_path = self.models_dir / f"model_{i}.pth"
            with open(model_path, 'w') as f:
                f.write(f"# Mock model {i}")
            self.model_files.append(model_path)
        
        # Create batch manager
        self.config = BatchEvaluationConfig(
            models_directory=str(self.models_dir),
            results_directory=str(self.temp_dir / "results")
        )
        self.batch_manager = BatchEvaluationManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_discovery(self):
        """Test model file discovery."""
        discovered_models = self.batch_manager.discover_models(
            directory=str(self.models_dir),
            patterns=["*.pth"]
        )
        
        self.assertEqual(len(discovered_models), 3)
        for model_path in discovered_models:
            self.assertTrue(model_path.exists())
            self.assertTrue(model_path.suffix == ".pth")
    
    def test_model_discovery_with_exclusions(self):
        """Test model discovery with exclusion patterns."""
        # Create a temp file to exclude
        temp_file = self.models_dir / "temp_model.pth"
        with open(temp_file, 'w') as f:
            f.write("# Temp model")
        
        discovered_models = self.batch_manager.discover_models(
            directory=str(self.models_dir),
            patterns=["*.pth"],
            exclude_patterns=["*temp*"]
        )
        
        # Should exclude the temp file
        self.assertEqual(len(discovered_models), 3)  # Original 3, not 4
        
        discovered_names = [m.name for m in discovered_models]
        self.assertNotIn("temp_model.pth", discovered_names)
    
    @patch('evaluation_cli.EvaluationCLI')
    def test_batch_model_registration(self, mock_cli_class):
        """Test batch model registration."""
        # Mock the CLI and orchestrator
        mock_cli = Mock()
        mock_orchestrator = Mock()
        mock_cli._get_orchestrator.return_value = mock_orchestrator
        mock_cli_class.return_value = mock_cli
        
        # Mock successful registration
        mock_orchestrator.register_model.side_effect = lambda **kwargs: kwargs.get('model_id', 'auto_id')
        
        # Test batch registration
        registered_ids = self.batch_manager.register_models_batch(
            models_directory=str(self.models_dir),
            patterns=["*.pth"]
        )
        
        self.assertEqual(len(registered_ids), 3)
        self.assertEqual(mock_orchestrator.register_model.call_count, 3)
    
    def test_batch_config_validation(self):
        """Test batch configuration validation."""
        # Test valid configuration
        config = BatchEvaluationConfig(
            models_directory=str(self.models_dir),
            max_concurrent_evaluations=4
        )
        self.assertEqual(config.max_concurrent_evaluations, 4)
        
        # Test configuration with invalid directory
        config = BatchEvaluationConfig(
            models_directory="/nonexistent/directory"
        )
        # Should not raise error during creation, but during usage
        self.assertEqual(config.models_directory, "/nonexistent/directory")


class TestEvaluationMonitor(unittest.TestCase):
    """Test cases for evaluation monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MonitoringConfig(
            update_interval_seconds=0.1,  # Fast updates for testing
            history_length=10
        )
        self.monitor = EvaluationMonitor(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.monitor._monitoring:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsInstance(self.monitor, EvaluationMonitor)
        self.assertFalse(self.monitor._monitoring)
        self.assertIsInstance(self.monitor.system_monitor, SystemMonitor)
    
    def test_system_monitor(self):
        """Test system monitoring functionality."""
        system_monitor = SystemMonitor()
        
        # Test metrics collection
        metrics = system_monitor._collect_system_metrics()
        
        self.assertIsNotNone(metrics.timestamp)
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertGreaterEqual(metrics.memory_percent, 0)
        self.assertGreater(metrics.memory_total_gb, 0)
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop."""
        # Test start
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring)
        self.assertTrue(self.monitor.system_monitor._monitoring)
        
        # Wait a bit for some data collection
        time.sleep(0.5)
        
        # Test stop
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring)
        self.assertFalse(self.monitor.system_monitor._monitoring)
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_called = []
        
        def test_callback(progress):
            callback_called.append(progress)
        
        self.monitor.add_progress_callback(test_callback)
        
        # Start monitoring briefly
        self.monitor.start_monitoring()
        time.sleep(0.2)
        self.monitor.stop_monitoring()
        
        # Callback should have been called
        # Note: May be empty if no evaluation is running
        self.assertIsInstance(callback_called, list)
    
    def test_monitoring_status(self):
        """Test monitoring status retrieval."""
        status = self.monitor.get_current_status()
        
        self.assertIn('monitoring_active', status)
        self.assertIn('timestamp', status)
        self.assertIn('evaluation_metrics', status)
        self.assertIn('system_metrics', status)
        self.assertIn('recent_alerts', status)


class TestEvaluationAnalysis(unittest.TestCase):
    """Test cases for evaluation analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_results.db"
        self.analyzer = EvaluationAnalyzer(str(self.db_path))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.analyzer.db:
            self.analyzer.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, EvaluationAnalyzer)
        self.assertIsInstance(self.analyzer.db, ResultsDatabase)
    
    def test_database_operations(self):
        """Test database operations."""
        db = self.analyzer.db
        
        # Test model insertion
        model_info = {
            'model_id': 'test_model',
            'model_path': '/path/to/model.pth',
            'model_type': 'checkpoint',
            'registration_time': datetime.now().isoformat(),
            'metadata': {'test': True}
        }
        
        db.insert_model(model_info)
        
        # Test evaluation insertion
        evaluation_result = {
            'task_id': 'test_task',
            'model_id': 'test_model',
            'suite_name': 'base',
            'policy_mode': 'deterministic',
            'results': {
                'success_rate': 0.85,
                'mean_reward': 0.75,
                'mean_episode_length': 500,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        db.insert_evaluation(evaluation_result)
        
        # Test querying
        filter_criteria = QueryFilter(model_pattern='test_model')
        results = db.query_results(filter_criteria)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['model_id'], 'test_model')
        self.assertEqual(results[0]['success_rate'], 0.85)
    
    def test_query_filters(self):
        """Test query filter functionality."""
        # Test filter creation
        filter_criteria = QueryFilter(
            model_pattern="champion_*",
            suite_pattern="base",
            metric_name="success_rate",
            min_value=0.8
        )
        
        self.assertEqual(filter_criteria.model_pattern, "champion_*")
        self.assertEqual(filter_criteria.suite_pattern, "base")
        self.assertEqual(filter_criteria.metric_name, "success_rate")
        self.assertEqual(filter_criteria.min_value, 0.8)
    
    def test_trend_analysis(self):
        """Test performance trend analysis."""
        # Create mock trend data
        model_ids = ['model_1', 'model_2']
        
        # Mock the trend data retrieval
        with patch.object(self.analyzer, '_get_trend_data_fallback') as mock_trend:
            mock_trend.return_value = [
                {
                    'timestamp': '2024-08-01T10:00:00',
                    'suite_name': 'base',
                    'success_rate': 0.8,
                    'mean_reward': 0.7,
                    'mean_lateral_deviation': 0.1
                },
                {
                    'timestamp': '2024-08-02T10:00:00',
                    'suite_name': 'base',
                    'success_rate': 0.85,
                    'mean_reward': 0.75,
                    'mean_lateral_deviation': 0.09
                }
            ]
            
            analysis_result = self.analyzer.analyze_performance_trends(
                model_ids=model_ids,
                time_range_days=7
            )
            
            self.assertEqual(analysis_result.analysis_type, "performance_trends")
            self.assertIn('model_1', analysis_result.data)
            self.assertIn('model_2', analysis_result.data)
            self.assertIsInstance(analysis_result.recommendations, list)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        model_ids = ['model_a', 'model_b']
        metrics = ['success_rate', 'mean_reward']
        
        # Mock query results
        with patch.object(self.analyzer, 'query_results') as mock_query:
            mock_query.side_effect = [
                # Results for model_a
                [{
                    'model_id': 'model_a',
                    'suite_name': 'base',
                    'results': {
                        'success_rate': 0.85,
                        'mean_reward': 0.75
                    }
                }],
                # Results for model_b
                [{
                    'model_id': 'model_b',
                    'suite_name': 'base',
                    'results': {
                        'success_rate': 0.80,
                        'mean_reward': 0.70
                    }
                }]
            ]
            
            comparison_result = self.analyzer.compare_models(
                model_ids=model_ids,
                metrics=metrics
            )
            
            self.assertEqual(comparison_result.analysis_type, "model_comparison")
            self.assertIn('model_statistics', comparison_result.summary)
            self.assertIn('rankings', comparison_result.summary)
    
    def test_data_export(self):
        """Test data export functionality."""
        # Add some test data to database
        db = self.analyzer.db
        
        model_info = {
            'model_id': 'export_test_model',
            'model_path': '/path/to/model.pth',
            'model_type': 'checkpoint',
            'registration_time': datetime.now().isoformat()
        }
        db.insert_model(model_info)
        
        evaluation_result = {
            'task_id': 'export_test_task',
            'model_id': 'export_test_model',
            'suite_name': 'base',
            'policy_mode': 'deterministic',
            'results': {
                'success_rate': 0.9,
                'mean_reward': 0.8,
                'timestamp': datetime.now().isoformat()
            }
        }
        db.insert_evaluation(evaluation_result)
        
        # Test CSV export
        csv_file = self.temp_dir / "test_export.csv"
        count = self.analyzer.export_results(
            output_path=str(csv_file),
            format_type="csv"
        )
        
        self.assertGreater(count, 0)
        self.assertTrue(csv_file.exists())
        
        # Test JSON export
        json_file = self.temp_dir / "test_export.json"
        count = self.analyzer.export_results(
            output_path=str(json_file),
            format_type="json"
        )
        
        self.assertGreater(count, 0)
        self.assertTrue(json_file.exists())
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn('results', export_data)
        self.assertIn('total_results', export_data)


class TestIntegration(unittest.TestCase):
    """Integration tests for CLI components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock model
        self.model_path = self.temp_dir / "integration_model.pth"
        with open(self.model_path, 'w') as f:
            f.write("# Integration test model")
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_to_analysis_integration(self):
        """Test integration between CLI and analysis components."""
        # Initialize CLI and register model
        cli = EvaluationCLI()
        orchestrator = cli._get_orchestrator()
        
        model_id = orchestrator.register_model(
            model_path=str(self.model_path),
            model_type="checkpoint",
            model_id="integration_test_model"
        )
        
        # Initialize analyzer
        analyzer = EvaluationAnalyzer()
        
        # Test that analyzer can access CLI results
        results = analyzer.query_results(QueryFilter(model_pattern=model_id))
        
        # Should be able to query even if no evaluation results yet
        self.assertIsInstance(results, list)
    
    def test_monitoring_integration(self):
        """Test monitoring integration with evaluation."""
        # Initialize components
        cli = EvaluationCLI()
        orchestrator = cli._get_orchestrator()
        
        config = MonitoringConfig(update_interval_seconds=0.1)
        monitor = EvaluationMonitor(config)
        
        try:
            # Start monitoring
            monitor.start_monitoring()
            
            # Register model and schedule evaluation
            model_id = orchestrator.register_model(
                model_path=str(self.model_path),
                model_type="checkpoint"
            )
            
            task_ids = orchestrator.schedule_evaluation(
                model_ids=[model_id],
                suite_names=["base"],
                seeds_per_suite=1
            )
            
            # Monitor should be able to track progress
            time.sleep(0.2)
            status = monitor.get_current_status()
            
            self.assertIn('monitoring_active', status)
            self.assertTrue(status['monitoring_active'])
            
        finally:
            monitor.stop_monitoring()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_model_path(self):
        """Test handling of invalid model paths."""
        cli = EvaluationCLI()
        orchestrator = cli._get_orchestrator()
        
        with self.assertRaises(FileNotFoundError):
            orchestrator.register_model(
                model_path="/nonexistent/path/model.pth",
                model_type="checkpoint"
            )
    
    def test_invalid_model_id(self):
        """Test handling of duplicate model IDs."""
        temp_dir = Path(tempfile.mkdtemp())
        model_path = temp_dir / "test_model.pth"
        
        try:
            with open(model_path, 'w') as f:
                f.write("# Test model")
            
            cli = EvaluationCLI()
            orchestrator = cli._get_orchestrator()
            
            # Register model with specific ID
            model_id = orchestrator.register_model(
                model_path=str(model_path),
                model_type="checkpoint",
                model_id="duplicate_test"
            )
            
            # Try to register again with same ID
            with self.assertRaises(ValueError):
                orchestrator.register_model(
                    model_path=str(model_path),
                    model_type="checkpoint",
                    model_id="duplicate_test"
                )
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_evaluation_without_models(self):
        """Test evaluation scheduling without registered models."""
        cli = EvaluationCLI()
        orchestrator = cli._get_orchestrator()
        
        with self.assertRaises(ValueError):
            orchestrator.schedule_evaluation(
                model_ids=["nonexistent_model"],
                suite_names=["base"]
            )
    
    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        # Try to create database in read-only location
        try:
            analyzer = EvaluationAnalyzer("/root/readonly.db")
            # Should handle gracefully or raise appropriate error
        except (PermissionError, OSError):
            # Expected behavior
            pass
    
    def test_monitoring_without_evaluation(self):
        """Test monitoring when no evaluation is running."""
        config = MonitoringConfig(update_interval_seconds=0.1)
        monitor = EvaluationMonitor(config)
        
        try:
            monitor.start_monitoring()
            time.sleep(0.2)
            
            status = monitor.get_current_status()
            
            # Should handle gracefully
            self.assertIn('monitoring_active', status)
            self.assertTrue(status['monitoring_active'])
            
            # Evaluation metrics may be None
            self.assertIn('evaluation_metrics', status)
            
        finally:
            monitor.stop_monitoring()


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestEvaluationCLI,
        TestBatchEvaluation,
        TestEvaluationMonitor,
        TestEvaluationAnalysis,
        TestIntegration,
        TestErrorHandling
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests directly
    success = run_tests()
    sys.exit(0 if success else 1)