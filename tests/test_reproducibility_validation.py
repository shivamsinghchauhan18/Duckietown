#!/usr/bin/env python3
"""
🧪 REPRODUCIBILITY VALIDATION TESTS 🧪
Tests for reproducibility with fixed seeds and configurations

This module validates that evaluation results are reproducible across runs
with identical seeds and configurations, ensuring scientific rigor.

Requirements covered: 8.4, 13.3, 13.4
"""

import os
import sys
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.evaluation_orchestrator import (
    EvaluationOrchestrator, SeedManager, PolicyMode
)
from duckietown_utils.suite_manager import (
    SuiteManager, SuiteResults, SuiteType, EpisodeResult
)
from duckietown_utils.metrics_calculator import MetricsCalculator
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer
from duckietown_utils.failure_analyzer import FailureAnalyzer
from duckietown_utils.robustness_analyzer import RobustnessAnalyzer


class TestSeedReproducibility:
    """Test suite for seed-based reproducibility."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def reproducible_config(self, temp_workspace):
        """Create configuration optimized for reproducibility."""
        return {
            'results_dir': temp_workspace,
            'base_seed': 42,
            'random_seed': 42,
            'bootstrap_samples': 1000,
            'confidence_level': 0.95,
            'default_seeds_per_suite': 10,
            'deterministic_mode': True,
            'save_artifacts': True
        }
    
    def test_seed_manager_reproducibility(self, reproducible_config):
        """Test that SeedManager produces identical seeds across runs."""
        # Create two seed managers with same configuration
        seed_manager1 = SeedManager(base_seed=reproducible_config['base_seed'])
        seed_manager2 = SeedManager(base_seed=reproducible_config['base_seed'])
        
        # Generate seeds for same suite
        suite_name = "test_suite"
        num_seeds = 20
        
        seeds1 = seed_manager1.generate_seeds(suite_name, num_seeds, deterministic=True)
        seeds2 = seed_manager2.generate_seeds(suite_name, num_seeds, deterministic=True)
        
        # Seeds should be identical
        assert seeds1 == seeds2, "Seed managers with same base seed produce different results"
        assert len(seeds1) == num_seeds
        assert len(set(seeds1)) == num_seeds, "Generated seeds are not unique"
        
        # Test with different suite names should produce different seeds
        seeds3 = seed_manager1.generate_seeds("different_suite", num_seeds, deterministic=True)
        assert seeds1 != seeds3, "Different suite names should produce different seeds"
        
        # Test reproducibility across multiple calls
        seeds4 = seed_manager1.generate_seeds(suite_name, num_seeds, deterministic=True)
        assert seeds1 == seeds4, "Multiple calls with same parameters should be identical"
    
    def test_episode_simulation_reproducibility(self, temp_workspace, reproducible_config):
        """Test that episode simulation is reproducible with same seeds."""
        suite_manager = SuiteManager(reproducible_config)
        
        # Create mock model
        mock_model = Mock()
        mock_model.model_id = "test_model"
        
        # Fixed seeds for reproducibility
        seeds = [1, 2, 3, 4, 5]
        
        # Mock deterministic episode simulation
        with patch.object(suite_manager, '_simulate_episode') as mock_simulate:
            def deterministic_simulation(episode_id, map_name, model, seed, suite_config, policy_mode):
                # Use seed to generate deterministic results
                np.random.seed(seed)
                success = np.random.random() < 0.8
                reward = 0.8 if success else 0.3
                
                return EpisodeResult(
                    episode_id=episode_id,
                    map_name=map_name,
                    seed=seed,
                    success=success,
                    reward=reward,
                    episode_length=500 + int(np.random.normal(0, 50)),
                    lateral_deviation=0.1 + np.random.uniform(0, 0.05),
                    heading_error=5.0 + np.random.normal(0, 2),
                    jerk=0.05 + np.random.uniform(0, 0.02),
                    stability=0.9 - np.random.uniform(0, 0.1)
                )
            
            mock_simulate.side_effect = deterministic_simulation
            
            # Run same evaluation twice
            results1 = suite_manager.run_suite("base", mock_model, seeds, "deterministic")
            results2 = suite_manager.run_suite("base", mock_model, seeds, "deterministic")
            
            # Results should be identical
            assert results1.success_rate == results2.success_rate
            assert results1.mean_reward == results2.mean_reward
            assert results1.total_episodes == results2.total_episodes
            
            # Episode-level results should match
            for ep1, ep2 in zip(results1.episode_results, results2.episode_results):
                assert ep1.seed == ep2.seed
                assert ep1.success == ep2.success
                assert ep1.reward == ep2.reward
                assert ep1.episode_length == ep2.episode_length
                assert abs(ep1.lateral_deviation - ep2.lateral_deviation) < 1e-10
    
    def test_metrics_calculation_reproducibility(self, reproducible_config):
        """Test that metrics calculations are reproducible."""
        calculator = MetricsCalculator(reproducible_config)
        
        # Create identical episode datasets
        def create_episode_dataset():
            episodes = []
            np.random.seed(42)  # Fixed seed for dataset generation
            
            for i in range(100):
                episode = EpisodeResult(
                    episode_id=f"episode_{i}",
                    map_name=f"map_{i % 3}",
                    seed=i,
                    success=np.random.random() < 0.75,
                    reward=np.random.uniform(0.3, 0.9),
                    episode_length=int(np.random.normal(500, 50)),
                    lateral_deviation=np.random.uniform(0.05, 0.2),
                    heading_error=np.random.uniform(2, 10),
                    jerk=np.random.uniform(0.02, 0.1),
                    stability=np.random.uniform(0.7, 0.95)
                )
                episodes.append(episode)
            
            return episodes
        
        # Calculate metrics multiple times
        episodes1 = create_episode_dataset()
        episodes2 = create_episode_dataset()
        
        metrics1 = calculator.aggregate_episode_metrics(episodes1)
        metrics2 = calculator.aggregate_episode_metrics(episodes2)
        
        # Metrics should be identical
        for metric_name in metrics1:
            assert metric_name in metrics2
            assert abs(metrics1[metric_name].value - metrics2[metric_name].value) < 1e-10
            assert metrics1[metric_name].sample_size == metrics2[metric_name].sample_size
        
        # Per-map metrics should also be identical
        per_map1 = calculator.calculate_per_map_metrics(episodes1)
        per_map2 = calculator.calculate_per_map_metrics(episodes2)
        
        assert set(per_map1.keys()) == set(per_map2.keys())
        
        for map_name in per_map1:
            for metric_name in per_map1[map_name]:
                assert abs(per_map1[map_name][metric_name].value - 
                          per_map2[map_name][metric_name].value) < 1e-10
    
    def test_statistical_analysis_reproducibility(self, reproducible_config):
        """Test that statistical analysis is reproducible."""
        analyzer = StatisticalAnalyzer(reproducible_config)
        
        # Create test data
        np.random.seed(42)
        sample_a = np.random.normal(10, 2, 100)
        sample_b = np.random.normal(11, 2, 100)
        
        # Perform analysis multiple times
        results = []
        for _ in range(5):
            np.random.seed(42)  # Reset seed for bootstrap
            
            comparison = analyzer.compare_models(
                sample_a, sample_b,
                "model_a", "model_b", "test_metric"
            )
            
            ci = analyzer.compute_confidence_intervals(sample_a, method="bootstrap")
            
            results.append({
                'p_value': comparison.p_value,
                'effect_size': comparison.effect_size,
                'ci_lower': ci.lower,
                'ci_upper': ci.upper
            })
        
        # All results should be identical
        for i in range(1, len(results)):
            assert abs(results[i]['p_value'] - results[0]['p_value']) < 1e-10
            assert abs(results[i]['effect_size'] - results[0]['effect_size']) < 1e-10
            assert abs(results[i]['ci_lower'] - results[0]['ci_lower']) < 1e-10
            assert abs(results[i]['ci_upper'] - results[0]['ci_upper']) < 1e-10
    
    def test_evaluation_orchestrator_reproducibility(self, temp_workspace, reproducible_config):
        """Test end-to-end evaluation reproducibility."""
        # Create two orchestrators with same configuration
        orchestrator1 = EvaluationOrchestrator(reproducible_config)
        orchestrator2 = EvaluationOrchestrator(reproducible_config)
        
        # Create mock model file
        model_path = Path(temp_workspace) / "test_model.pth"
        model_path.touch()
        
        # Register same model in both orchestrators
        model_id1 = orchestrator1.register_model(str(model_path), model_id="test_model")
        model_id2 = orchestrator2.register_model(str(model_path), model_id="test_model")
        
        assert model_id1 == model_id2
        
        # Schedule identical evaluations
        task_ids1 = orchestrator1.schedule_evaluation(
            model_ids=model_id1,
            suite_names="base",
            policy_modes=PolicyMode.DETERMINISTIC,
            seeds_per_suite=5
        )
        
        task_ids2 = orchestrator2.schedule_evaluation(
            model_ids=model_id2,
            suite_names="base",
            policy_modes=PolicyMode.DETERMINISTIC,
            seeds_per_suite=5
        )
        
        # Tasks should have identical seeds
        task1 = orchestrator1.state_tracker.get_task(task_ids1[0])
        task2 = orchestrator2.state_tracker.get_task(task_ids2[0])
        
        assert task1.seeds == task2.seeds
        assert task1.suite_name == task2.suite_name
        assert task1.policy_mode == task2.policy_mode
        
        # Clean up
        orchestrator1.cleanup()
        orchestrator2.cleanup()
    
    def test_configuration_hash_reproducibility(self, reproducible_config):
        """Test that configuration hashing is reproducible."""
        # Create multiple instances with same config
        configs = []
        for _ in range(5):
            config_copy = reproducible_config.copy()
            configs.append(config_copy)
        
        # Hash configurations
        hashes = []
        for config in configs:
            config_str = json.dumps(config, sort_keys=True)
            import hashlib
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            hashes.append(config_hash)
        
        # All hashes should be identical
        for i in range(1, len(hashes)):
            assert hashes[i] == hashes[0], "Configuration hashing not reproducible"
        
        # Different config should produce different hash
        modified_config = reproducible_config.copy()
        modified_config['base_seed'] = 123
        
        modified_str = json.dumps(modified_config, sort_keys=True)
        modified_hash = hashlib.md5(modified_str.encode()).hexdigest()
        
        assert modified_hash != hashes[0], "Different configurations produce same hash"


class TestEnvironmentReproducibility:
    """Test suite for environment-level reproducibility."""
    
    @pytest.fixture
    def environment_config(self, temp_workspace):
        """Create environment configuration."""
        return {
            'results_dir': temp_workspace,
            'log_environment_info': True,
            'capture_system_state': True,
            'validate_reproducibility': True
        }
    
    def test_system_state_capture(self, environment_config):
        """Test capture of system state for reproducibility."""
        # Mock system information
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'numpy_version': np.__version__,
            'random_state': np.random.get_state()[1][0],  # First element of state
            'environment_variables': {
                'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED', 'not_set'),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')
            }
        }
        
        # Save system state
        state_file = Path(environment_config['results_dir']) / 'system_state.json'
        with open(state_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        # Verify state file exists and is readable
        assert state_file.exists()
        
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state['python_version'] == sys.version
        assert loaded_state['platform'] == sys.platform
        assert 'numpy_version' in loaded_state
    
    def test_git_commit_tracking(self, environment_config):
        """Test tracking of git commit for reproducibility."""
        try:
            import subprocess
            
            # Try to get git commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                
                # Save git information
                git_info = {
                    'commit_hash': commit_hash,
                    'branch': subprocess.run(
                        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent.parent
                    ).stdout.strip(),
                    'dirty': subprocess.run(
                        ['git', 'diff', '--quiet'],
                        cwd=Path(__file__).parent.parent
                    ).returncode != 0
                }
                
                git_file = Path(environment_config['results_dir']) / 'git_info.json'
                with open(git_file, 'w') as f:
                    json.dump(git_info, f, indent=2)
                
                assert git_file.exists()
                assert len(commit_hash) == 40  # Full SHA-1 hash
                
        except (subprocess.SubprocessError, FileNotFoundError):
            # Git not available or not in git repository
            pytest.skip("Git not available for commit tracking test")
    
    def test_dependency_version_tracking(self, environment_config):
        """Test tracking of dependency versions."""
        # Get versions of key dependencies
        dependencies = {}
        
        try:
            dependencies['numpy'] = np.__version__
        except:
            dependencies['numpy'] = 'unknown'
        
        try:
            import scipy
            dependencies['scipy'] = scipy.__version__
        except ImportError:
            dependencies['scipy'] = 'not_installed'
        
        try:
            import pytest
            dependencies['pytest'] = pytest.__version__
        except ImportError:
            dependencies['pytest'] = 'not_installed'
        
        # Save dependency information
        deps_file = Path(environment_config['results_dir']) / 'dependencies.json'
        with open(deps_file, 'w') as f:
            json.dump(dependencies, f, indent=2)
        
        assert deps_file.exists()
        
        # Verify content
        with open(deps_file, 'r') as f:
            loaded_deps = json.load(f)
        
        assert 'numpy' in loaded_deps
        assert loaded_deps['numpy'] != 'unknown'
    
    def test_random_state_management(self, environment_config):
        """Test proper random state management for reproducibility."""
        # Test numpy random state
        initial_state = np.random.get_state()
        
        # Set known seed
        np.random.seed(42)
        state_after_seed = np.random.get_state()
        
        # Generate some random numbers
        random_numbers1 = np.random.random(10)
        
        # Reset to same seed
        np.random.seed(42)
        random_numbers2 = np.random.random(10)
        
        # Should be identical
        np.testing.assert_array_equal(random_numbers1, random_numbers2)
        
        # Restore initial state
        np.random.set_state(initial_state)
        
        # Test Python random state
        import random
        
        random.seed(42)
        python_random1 = [random.random() for _ in range(10)]
        
        random.seed(42)
        python_random2 = [random.random() for _ in range(10)]
        
        assert python_random1 == python_random2
    
    def test_configuration_serialization_reproducibility(self, environment_config):
        """Test that configuration serialization is reproducible."""
        # Create complex configuration
        complex_config = {
            'nested_dict': {
                'param1': 1.5,
                'param2': [1, 2, 3],
                'param3': {'a': 'value_a', 'b': 'value_b'}
            },
            'list_param': [3.14, 2.71, 1.41],
            'string_param': 'test_value',
            'bool_param': True,
            'none_param': None
        }
        
        # Serialize multiple times
        serializations = []
        for _ in range(5):
            serialized = json.dumps(complex_config, sort_keys=True, indent=2)
            serializations.append(serialized)
        
        # All serializations should be identical
        for i in range(1, len(serializations)):
            assert serializations[i] == serializations[0]
        
        # Test deserialization
        for serialized in serializations:
            deserialized = json.loads(serialized)
            assert deserialized == complex_config


class TestCrossRunReproducibility:
    """Test suite for reproducibility across different runs."""
    
    @pytest.fixture
    def cross_run_config(self, temp_workspace):
        """Create configuration for cross-run testing."""
        return {
            'results_dir': temp_workspace,
            'base_seed': 12345,
            'random_seed': 12345,
            'bootstrap_samples': 500,  # Reduced for faster testing
            'deterministic_mode': True,
            'save_intermediate_results': True
        }
    
    def test_multiple_run_consistency(self, cross_run_config):
        """Test consistency across multiple evaluation runs."""
        results = []
        
        # Run same evaluation multiple times
        for run_id in range(3):
            # Create fresh components for each run
            calculator = MetricsCalculator(cross_run_config)
            analyzer = StatisticalAnalyzer(cross_run_config)
            
            # Generate identical test data
            np.random.seed(cross_run_config['base_seed'])
            episodes = []
            
            for i in range(50):
                episode = EpisodeResult(
                    episode_id=f"episode_{i}",
                    map_name=f"map_{i % 2}",
                    seed=i,
                    success=np.random.random() < 0.8,
                    reward=np.random.uniform(0.5, 0.9),
                    episode_length=int(np.random.normal(500, 30)),
                    lateral_deviation=np.random.uniform(0.05, 0.15),
                    heading_error=np.random.uniform(3, 8),
                    jerk=np.random.uniform(0.03, 0.08),
                    stability=np.random.uniform(0.8, 0.95)
                )
                episodes.append(episode)
            
            # Calculate metrics
            metrics = calculator.aggregate_episode_metrics(episodes)
            
            # Statistical analysis
            success_rewards = [ep.reward for ep in episodes if ep.success]
            np.random.seed(cross_run_config['random_seed'])
            ci = analyzer.compute_confidence_intervals(success_rewards, method="bootstrap")
            
            run_results = {
                'run_id': run_id,
                'success_rate': metrics['success_rate'].value,
                'mean_reward': metrics['mean_reward'].value,
                'ci_lower': ci.lower,
                'ci_upper': ci.upper,
                'total_episodes': len(episodes)
            }
            
            results.append(run_results)
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            assert abs(results[i]['success_rate'] - results[0]['success_rate']) < 1e-10
            assert abs(results[i]['mean_reward'] - results[0]['mean_reward']) < 1e-10
            assert abs(results[i]['ci_lower'] - results[0]['ci_lower']) < 1e-10
            assert abs(results[i]['ci_upper'] - results[0]['ci_upper']) < 1e-10
            assert results[i]['total_episodes'] == results[0]['total_episodes']
    
    def test_intermediate_result_consistency(self, cross_run_config):
        """Test that intermediate results are consistent across runs."""
        intermediate_results = []
        
        for run_id in range(2):
            calculator = MetricsCalculator(cross_run_config)
            
            # Process episodes in batches to create intermediate results
            all_episodes = []
            batch_results = []
            
            np.random.seed(cross_run_config['base_seed'])
            
            for batch in range(3):
                batch_episodes = []
                for i in range(10):
                    episode_id = batch * 10 + i
                    episode = EpisodeResult(
                        episode_id=f"episode_{episode_id}",
                        map_name=f"map_{episode_id % 2}",
                        seed=episode_id,
                        success=np.random.random() < 0.75,
                        reward=np.random.uniform(0.4, 0.9),
                        episode_length=int(np.random.normal(500, 25)),
                        lateral_deviation=np.random.uniform(0.08, 0.18),
                        heading_error=np.random.uniform(4, 9),
                        jerk=np.random.uniform(0.04, 0.09),
                        stability=np.random.uniform(0.75, 0.95)
                    )
                    batch_episodes.append(episode)
                
                all_episodes.extend(batch_episodes)
                
                # Calculate intermediate metrics
                batch_metrics = calculator.aggregate_episode_metrics(all_episodes)
                batch_results.append({
                    'batch': batch,
                    'episodes_processed': len(all_episodes),
                    'success_rate': batch_metrics['success_rate'].value,
                    'mean_reward': batch_metrics['mean_reward'].value
                })
            
            intermediate_results.append({
                'run_id': run_id,
                'batch_results': batch_results
            })
        
        # Compare intermediate results across runs
        run1_batches = intermediate_results[0]['batch_results']
        run2_batches = intermediate_results[1]['batch_results']
        
        assert len(run1_batches) == len(run2_batches)
        
        for batch1, batch2 in zip(run1_batches, run2_batches):
            assert batch1['batch'] == batch2['batch']
            assert batch1['episodes_processed'] == batch2['episodes_processed']
            assert abs(batch1['success_rate'] - batch2['success_rate']) < 1e-10
            assert abs(batch1['mean_reward'] - batch2['mean_reward']) < 1e-10
    
    def test_artifact_reproducibility(self, cross_run_config):
        """Test that saved artifacts are reproducible."""
        artifact_hashes = []
        
        for run_id in range(2):
            # Create test results
            episodes = []
            np.random.seed(cross_run_config['base_seed'])
            
            for i in range(20):
                episode = EpisodeResult(
                    episode_id=f"episode_{i}",
                    map_name=f"map_{i % 2}",
                    seed=i,
                    success=np.random.random() < 0.8,
                    reward=np.random.uniform(0.6, 0.9),
                    episode_length=int(np.random.normal(500, 20)),
                    lateral_deviation=np.random.uniform(0.1, 0.2),
                    heading_error=np.random.uniform(5, 10),
                    jerk=np.random.uniform(0.05, 0.1),
                    stability=np.random.uniform(0.8, 0.9)
                )
                episodes.append(episode)
            
            # Save results to file
            results_file = Path(cross_run_config['results_dir']) / f'results_run_{run_id}.json'
            
            # Convert episodes to serializable format
            serializable_episodes = []
            for ep in episodes:
                ep_dict = {
                    'episode_id': ep.episode_id,
                    'map_name': ep.map_name,
                    'seed': ep.seed,
                    'success': ep.success,
                    'reward': ep.reward,
                    'episode_length': ep.episode_length,
                    'lateral_deviation': ep.lateral_deviation,
                    'heading_error': ep.heading_error,
                    'jerk': ep.jerk,
                    'stability': ep.stability
                }
                serializable_episodes.append(ep_dict)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_episodes, f, sort_keys=True, indent=2)
            
            # Calculate file hash
            import hashlib
            with open(results_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            artifact_hashes.append(file_hash)
        
        # File hashes should be identical
        assert artifact_hashes[0] == artifact_hashes[1], "Artifact files are not reproducible"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])