#!/usr/bin/env python3
"""
Unit tests for the Suite Manager.

Tests cover suite configuration, suite execution, and results management.
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.suite_manager import (
    SuiteManager, SuiteConfig, SuiteType, EpisodeResult, SuiteResults
)

class TestSuiteConfig(TestCase):
    """Test cases for SuiteConfig dataclass."""
    
    def test_valid_suite_config(self):
        """Test creating valid suite configuration."""
        config = SuiteConfig(
            suite_name="test_suite",
            suite_type=SuiteType.BASE,
            description="Test suite",
            maps=["map1", "map2"],
            episodes_per_map=10
        )
        
        self.assertEqual(config.suite_name, "test_suite")
        self.assertEqual(config.suite_type, SuiteType.BASE)
        self.assertEqual(len(config.maps), 2)
        self.assertEqual(config.episodes_per_map, 10)
    
    def test_invalid_suite_config_no_maps(self):
        """Test that empty maps list raises error."""
        with self.assertRaises(ValueError):
            SuiteConfig(
                suite_name="invalid_suite",
                suite_type=SuiteType.BASE,
                description="Invalid suite",
                maps=[],  # Empty maps
                episodes_per_map=10
            )
    
    def test_invalid_suite_config_zero_episodes(self):
        """Test that zero episodes raises error."""
        with self.assertRaises(ValueError):
            SuiteConfig(
                suite_name="invalid_suite",
                suite_type=SuiteType.BASE,
                description="Invalid suite",
                maps=["map1"],
                episodes_per_map=0  # Zero episodes
            )

class TestEpisodeResult(TestCase):
    """Test cases for EpisodeResult dataclass."""
    
    def test_episode_result_creation(self):
        """Test creating episode result."""
        result = EpisodeResult(
            episode_id="test_episode",
            map_name="test_map",
            seed=42,
            success=True,
            reward=0.85,
            episode_length=500,
            lateral_deviation=0.1,
            heading_error=5.0,
            jerk=0.05,
            stability=0.9
        )
        
        self.assertEqual(result.episode_id, "test_episode")
        self.assertTrue(result.success)
        self.assertEqual(result.reward, 0.85)
        self.assertFalse(result.collision)  # Default value
        self.assertIsNotNone(result.timestamp)

class TestSuiteResults(TestCase):
    """Test cases for SuiteResults dataclass."""
    
    def test_suite_results_aggregation(self):
        """Test automatic aggregation of episode results."""
        # Create sample episode results
        episodes = [
            EpisodeResult("ep1", "map1", 1, True, 0.8, 400, 0.1, 3.0, 0.03, 0.9),
            EpisodeResult("ep2", "map1", 2, True, 0.9, 450, 0.05, 2.0, 0.02, 0.95),
            EpisodeResult("ep3", "map1", 3, False, 0.3, 200, 0.3, 10.0, 0.1, 0.5, collision=True),
        ]
        
        results = SuiteResults(
            suite_name="test_suite",
            suite_type=SuiteType.BASE,
            model_id="test_model",
            policy_mode="deterministic",
            total_episodes=3,
            successful_episodes=2,
            episode_results=episodes
        )
        
        # Check aggregated metrics
        self.assertAlmostEqual(results.success_rate, 2/3, places=3)
        self.assertAlmostEqual(results.mean_reward, (0.8 + 0.9 + 0.3) / 3, places=3)
        self.assertAlmostEqual(results.mean_episode_length, (400 + 450 + 200) / 3, places=1)
        self.assertAlmostEqual(results.collision_rate, 1/3, places=3)
        self.assertGreater(results.stability, 0.0)
    
    def test_suite_results_empty_episodes(self):
        """Test suite results with no episodes."""
        results = SuiteResults(
            suite_name="empty_suite",
            suite_type=SuiteType.BASE,
            model_id="test_model",
            policy_mode="deterministic",
            total_episodes=0,
            successful_episodes=0,
            episode_results=[]
        )
        
        # Should not crash and should have default values
        self.assertEqual(results.success_rate, 0.0)
        self.assertEqual(results.mean_reward, 0.0)

class TestSuiteManager(TestCase):
    """Test cases for SuiteManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        config = {
            'results_dir': self.temp_dir,
            'base_config': {'episodes_per_map': 5},
            'hard_randomization_config': {'episodes_per_map': 4},
            'law_intersection_config': {'episodes_per_map': 3},
            'out_of_distribution_config': {'episodes_per_map': 3},
            'stress_adversarial_config': {'episodes_per_map': 2}
        }
        
        self.suite_manager = SuiteManager(config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_default_suites(self):
        """Test that default suites are initialized."""
        suites = self.suite_manager.list_suites()
        
        expected_suites = ["base", "hard_randomization", "law_intersection", 
                          "out_of_distribution", "stress_adversarial"]
        
        for suite in expected_suites:
            self.assertIn(suite, suites)
    
    def test_get_suite_config(self):
        """Test getting suite configuration."""
        base_config = self.suite_manager.get_suite_config("base")
        
        self.assertIsNotNone(base_config)
        self.assertEqual(base_config.suite_name, "base")
        self.assertEqual(base_config.suite_type, SuiteType.BASE)
        self.assertEqual(base_config.episodes_per_map, 5)  # From config
    
    def test_get_suite_info(self):
        """Test getting detailed suite information."""
        info = self.suite_manager.get_suite_info("base")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "base")
        self.assertEqual(info["type"], "base")
        self.assertIn("description", info)
        self.assertIn("maps", info)
        self.assertIn("total_episodes", info)
        self.assertGreater(info["total_episodes"], 0)
    
    def test_get_nonexistent_suite(self):
        """Test getting nonexistent suite returns None."""
        config = self.suite_manager.get_suite_config("nonexistent")
        info = self.suite_manager.get_suite_info("nonexistent")
        
        self.assertIsNone(config)
        self.assertIsNone(info)
    
    def test_custom_suite_registration(self):
        """Test registering custom suite through config."""
        custom_config = {
            'custom_suites': {
                'my_custom_suite': {
                    'suite_type': 'base',
                    'description': 'My custom test suite',
                    'maps': ['custom_map1', 'custom_map2'],
                    'episodes_per_map': 15,
                    'environment_config': {'custom_param': True}
                }
            }
        }
        
        custom_manager = SuiteManager(custom_config)
        
        self.assertIn('my_custom_suite', custom_manager.list_suites())
        
        config = custom_manager.get_suite_config('my_custom_suite')
        self.assertEqual(config.episodes_per_map, 15)
        self.assertEqual(config.maps, ['custom_map1', 'custom_map2'])
        self.assertTrue(config.environment_config['custom_param'])
    
    def test_suite_difficulty_mapping(self):
        """Test suite difficulty calculation."""
        # Test different suite types have different difficulties
        base_difficulty = self.suite_manager._get_suite_difficulty(SuiteType.BASE)
        hard_difficulty = self.suite_manager._get_suite_difficulty(SuiteType.HARD_RANDOMIZATION)
        stress_difficulty = self.suite_manager._get_suite_difficulty(SuiteType.STRESS_ADVERSARIAL)
        
        self.assertEqual(base_difficulty, 0.0)
        self.assertGreater(hard_difficulty, base_difficulty)
        self.assertGreater(stress_difficulty, hard_difficulty)
    
    @patch('time.sleep')  # Speed up tests by mocking sleep
    def test_run_suite_simulation(self, mock_sleep):
        """Test running a suite (simulation mode)."""
        mock_model = Mock()
        mock_model.model_id = "test_model"
        
        seeds = list(range(1, 26))  # Provide enough seeds for all episodes
        
        # Mock progress callback
        progress_updates = []
        def progress_callback(progress):
            progress_updates.append(progress)
        
        # Run base suite
        results = self.suite_manager.run_suite(
            suite_name="base",
            model=mock_model,
            seeds=seeds,
            policy_mode="deterministic",
            progress_callback=progress_callback
        )
        
        # Check results structure
        self.assertIsInstance(results, SuiteResults)
        self.assertEqual(results.suite_name, "base")
        self.assertEqual(results.suite_type, SuiteType.BASE)
        self.assertEqual(results.model_id, "test_model")
        self.assertEqual(results.policy_mode, "deterministic")
        
        # Check that episodes were generated
        self.assertGreater(results.total_episodes, 0)
        self.assertEqual(len(results.episode_results), results.total_episodes)
        
        # Check that progress callbacks were called
        self.assertGreater(len(progress_updates), 0)
        self.assertEqual(progress_updates[-1], 100.0)  # Final progress should be 100%
        
        # Check aggregated metrics are reasonable
        self.assertGreaterEqual(results.success_rate, 0.0)
        self.assertLessEqual(results.success_rate, 1.0)
        self.assertGreaterEqual(results.mean_reward, 0.0)
        self.assertLessEqual(results.mean_reward, 1.0)
    
    def test_run_nonexistent_suite(self):
        """Test running nonexistent suite raises error."""
        mock_model = Mock()
        
        with self.assertRaises(ValueError):
            self.suite_manager.run_suite(
                suite_name="nonexistent_suite",
                model=mock_model,
                seeds=[1, 2, 3]
            )
    
    @patch('time.sleep')
    def test_episode_simulation_consistency(self, mock_sleep):
        """Test that episode simulation produces consistent results."""
        mock_model = Mock()
        
        # Run same episode multiple times with same seed
        episode_results = []
        for _ in range(3):
            result = self.suite_manager._simulate_episode(
                episode_id="test_episode",
                map_name="test_map",
                model=mock_model,
                seed=42,  # Same seed
                suite_config=self.suite_manager.get_suite_config("base"),
                policy_mode="deterministic"
            )
            episode_results.append(result)
        
        # Results should be identical due to same seed
        for i in range(1, len(episode_results)):
            self.assertEqual(episode_results[0].success, episode_results[i].success)
            self.assertAlmostEqual(episode_results[0].reward, episode_results[i].reward, places=5)
            self.assertEqual(episode_results[0].episode_length, episode_results[i].episode_length)
    
    def test_suite_statistics(self):
        """Test getting suite statistics."""
        stats = self.suite_manager.get_suite_statistics("base")
        
        self.assertEqual(stats["suite_name"], "base")
        self.assertEqual(stats["suite_type"], "base")
        self.assertIn("total_maps", stats)
        self.assertIn("episodes_per_map", stats)
        self.assertIn("total_episodes", stats)
        self.assertIn("estimated_runtime_minutes", stats)
        
        # Check calculations
        config = self.suite_manager.get_suite_config("base")
        expected_total = len(config.maps) * config.episodes_per_map
        self.assertEqual(stats["total_episodes"], expected_total)
    
    def test_validate_suite_config(self):
        """Test suite configuration validation."""
        # Valid suite
        validation = self.suite_manager.validate_suite_config("base")
        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["errors"]), 0)
        
        # Invalid suite
        validation = self.suite_manager.validate_suite_config("nonexistent")
        self.assertFalse(validation["valid"])
        self.assertGreater(len(validation["errors"]), 0)
    
    def test_save_and_load_results(self):
        """Test saving and loading suite results."""
        # Create mock results
        episodes = [
            EpisodeResult("ep1", "map1", 1, True, 0.8, 400, 0.1, 3.0, 0.03, 0.9),
            EpisodeResult("ep2", "map1", 2, False, 0.3, 200, 0.3, 10.0, 0.1, 0.5)
        ]
        
        results = SuiteResults(
            suite_name="test_suite",
            suite_type=SuiteType.BASE,
            model_id="test_model",
            policy_mode="deterministic",
            total_episodes=2,
            successful_episodes=1,
            episode_results=episodes
        )
        
        # Save results
        self.suite_manager._save_suite_results(results)
        
        # Check that file was created
        result_files = list(Path(self.temp_dir).glob("suite_*.json"))
        self.assertEqual(len(result_files), 1)
        
        # Load and verify
        with open(result_files[0], 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data["suite_name"], "test_suite")
        self.assertEqual(loaded_data["total_episodes"], 2)
        self.assertEqual(len(loaded_data["episode_results"]), 2)

class TestSuiteIntegration(TestCase):
    """Integration tests for suite manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.suite_manager = SuiteManager({'results_dir': self.temp_dir})
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('time.sleep')
    def test_full_suite_workflow(self, mock_sleep):
        """Test complete suite evaluation workflow."""
        mock_model = Mock()
        mock_model.model_id = "integration_test_model"
        
        # Get suite info
        suite_info = self.suite_manager.get_suite_info("base")
        self.assertIsNotNone(suite_info)
        
        # Validate suite
        validation = self.suite_manager.validate_suite_config("base")
        self.assertTrue(validation["valid"])
        
        # Run suite
        seeds = list(range(1, 11))  # 10 seeds
        results = self.suite_manager.run_suite(
            suite_name="base",
            model=mock_model,
            seeds=seeds,
            policy_mode="deterministic"
        )
        
        # Verify results
        self.assertIsInstance(results, SuiteResults)
        self.assertEqual(results.model_id, "integration_test_model")
        self.assertGreater(results.total_episodes, 0)
        self.assertGreater(results.execution_time, 0)
        
        # Check that results file was saved
        result_files = list(Path(self.temp_dir).glob("suite_*.json"))
        self.assertEqual(len(result_files), 1)
    
    @patch('time.sleep')
    def test_multiple_suite_comparison(self, mock_sleep):
        """Test running multiple suites for comparison."""
        mock_model = Mock()
        mock_model.model_id = "comparison_model"
        
        seeds = list(range(1, 26))  # Provide enough seeds for all episodes
        suite_names = ["base", "hard_randomization"]
        
        results = {}
        for suite_name in suite_names:
            suite_results = self.suite_manager.run_suite(
                suite_name=suite_name,
                model=mock_model,
                seeds=seeds
            )
            results[suite_name] = suite_results
        
        # Compare results
        base_results = results["base"]
        hard_results = results["hard_randomization"]
        
        # Hard randomization should generally be more difficult
        # (though this is probabilistic in simulation)
        self.assertIsInstance(base_results.success_rate, float)
        self.assertIsInstance(hard_results.success_rate, float)
        
        # Both should have reasonable number of episodes
        self.assertGreater(base_results.total_episodes, 0)
        self.assertGreater(hard_results.total_episodes, 0)
        
        # Check that different suite types are recorded
        self.assertEqual(base_results.suite_type, SuiteType.BASE)
        self.assertEqual(hard_results.suite_type, SuiteType.HARD_RANDOMIZATION)
    
    def test_suite_configuration_edge_cases(self):
        """Test edge cases in suite configuration."""
        # Test all suite types exist and are valid
        for suite_type in SuiteType:
            suites_of_type = [
                name for name, config in self.suite_manager.suite_configs.items()
                if config.suite_type == suite_type
            ]
            self.assertGreater(len(suites_of_type), 0, f"No suites found for type {suite_type}")
        
        # Test that all suites have reasonable configurations
        for suite_name in self.suite_manager.list_suites():
            config = self.suite_manager.get_suite_config(suite_name)
            
            # Basic validation
            self.assertGreater(len(config.maps), 0)
            self.assertGreater(config.episodes_per_map, 0)
            self.assertGreater(config.timeout_per_episode, 0)
            
            # Validate suite info
            info = self.suite_manager.get_suite_info(suite_name)
            self.assertEqual(info["name"], suite_name)
            self.assertEqual(info["total_episodes"], 
                           len(config.maps) * config.episodes_per_map)

if __name__ == '__main__':
    import unittest
    unittest.main()