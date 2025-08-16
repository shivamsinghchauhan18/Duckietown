#!/usr/bin/env python3
"""
Tests for evaluation test suites implementation.
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.test_suites import (
    BaseTestSuite, BaseSuite, HardRandomizationSuite, LawIntersectionSuite,
    OutOfDistributionSuite, StressAdversarialSuite, TestSuiteFactory,
    create_all_suite_configs
)
from duckietown_utils.suite_manager import SuiteType, SuiteConfig


class TestBaseTestSuite:
    """Test the abstract base test suite class."""
    
    def test_abstract_methods(self):
        """Test that BaseTestSuite cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTestSuite({})


class TestBaseSuite:
    """Test the Base evaluation suite."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'episodes_per_map': 50,
            'timeout_per_episode': 120.0
        }
        self.suite = BaseSuite(self.config)
    
    def test_initialization(self):
        """Test suite initialization."""
        assert self.suite.suite_type == SuiteType.BASE
        assert self.suite.suite_name == "base"
        assert "Clean environmental conditions" in self.suite.get_description()
    
    def test_maps(self):
        """Test map configuration."""
        maps = self.suite.get_maps()
        assert isinstance(maps, list)
        assert len(maps) > 0
        assert 'LF-norm-loop' in maps
        assert 'LF-norm-small_loop' in maps
    
    def test_environment_config(self):
        """Test environment configuration."""
        env_config = self.suite.get_environment_config()
        
        # Check clean conditions
        assert env_config['lighting_variation'] == 0.0
        assert env_config['texture_variation'] == 0.0
        assert env_config['camera_noise'] == 0.0
        assert env_config['friction_variation'] == 0.0
        assert env_config['traffic_density'] == 0.0
        
        # Check disabled features
        assert env_config['static_obstacles'] is False
        assert env_config['dynamic_obstacles'] is False
        assert env_config['weather_effects'] is False
    
    def test_evaluation_config(self):
        """Test evaluation configuration."""
        eval_config = self.suite.get_evaluation_config()
        
        assert eval_config['deterministic_spawn'] is True
        assert eval_config['fixed_weather'] is True
        assert eval_config['clean_textures'] is True
        assert eval_config['disable_randomization'] is True
    
    def test_suite_config_creation(self):
        """Test SuiteConfig creation."""
        suite_config = self.suite.create_suite_config()
        
        assert isinstance(suite_config, SuiteConfig)
        assert suite_config.suite_name == "base"
        assert suite_config.suite_type == SuiteType.BASE
        assert suite_config.episodes_per_map == 50
        assert suite_config.timeout_per_episode == 120.0
    
    def test_custom_config(self):
        """Test custom configuration override."""
        custom_config = {
            'episodes_per_map': 25,
            'timeout_per_episode': 90.0,
            'maps': ['custom_map_1', 'custom_map_2']
        }
        
        suite = BaseSuite(custom_config)
        assert suite.get_episodes_per_map() == 25
        assert suite.get_timeout_per_episode() == 90.0
        assert suite.get_maps() == ['custom_map_1', 'custom_map_2']


class TestHardRandomizationSuite:
    """Test the Hard Randomization evaluation suite."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {'episodes_per_map': 40}
        self.suite = HardRandomizationSuite(self.config)
    
    def test_initialization(self):
        """Test suite initialization."""
        assert self.suite.suite_type == SuiteType.HARD_RANDOMIZATION
        assert self.suite.suite_name == "hard_randomization"
        assert "Heavy environmental noise" in self.suite.get_description()
    
    def test_environment_config(self):
        """Test environment configuration for heavy randomization."""
        env_config = self.suite.get_environment_config()
        
        # Check high variation values
        assert env_config['lighting_variation'] == 0.8
        assert env_config['texture_variation'] == 0.7
        assert env_config['camera_noise'] == 0.6
        assert env_config['friction_variation'] == 0.5
        assert env_config['traffic_density'] == 0.4
        
        # Check enabled features
        assert env_config['static_obstacles'] is True
        assert env_config['dynamic_obstacles'] is True
        assert env_config['weather_effects'] is True
        assert env_config['texture_domain_shift'] is True
    
    def test_evaluation_config(self):
        """Test evaluation configuration."""
        eval_config = self.suite.get_evaluation_config()
        
        assert eval_config['deterministic_spawn'] is False
        assert eval_config['weather_variation'] is True
        assert eval_config['texture_randomization'] is True
        assert eval_config['physics_randomization'] is True
    
    def test_maps(self):
        """Test map selection includes larger maps."""
        maps = self.suite.get_maps()
        assert 'huge_loop' in maps
        assert 'multi_track' in maps


class TestLawIntersectionSuite:
    """Test the Law/Intersection evaluation suite."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {'episodes_per_map': 30}
        self.suite = LawIntersectionSuite(self.config)
    
    def test_initialization(self):
        """Test suite initialization."""
        assert self.suite.suite_type == SuiteType.LAW_INTERSECTION
        assert self.suite.suite_name == "law_intersection"
        assert "Traffic rule compliance" in self.suite.get_description()
    
    def test_environment_config(self):
        """Test environment configuration for traffic rules."""
        env_config = self.suite.get_environment_config()
        
        # Check traffic rule elements
        assert env_config['stop_signs'] is True
        assert env_config['traffic_lights'] is True
        assert env_config['yield_signs'] is True
        assert env_config['right_of_way_scenarios'] is True
        assert env_config['intersection_complexity'] == 0.7
        
        # Check traffic participants
        assert env_config['other_vehicles'] is True
        assert env_config['pedestrians'] is True
        assert env_config['cyclists'] is True
    
    def test_evaluation_config(self):
        """Test evaluation configuration for law compliance."""
        eval_config = self.suite.get_evaluation_config()
        
        assert eval_config['track_violations'] is True
        assert eval_config['intersection_testing'] is True
        assert eval_config['traffic_rule_enforcement'] is True
        assert eval_config['compliance_scoring'] is True
        
        # Check violation categories
        violation_categories = eval_config['violation_categories']
        assert 'stop_sign_violations' in violation_categories
        assert 'traffic_light_violations' in violation_categories
        assert 'speed_violations' in violation_categories
        
        # Check violation penalties
        penalties = eval_config['violation_penalties']
        assert penalties['stop_sign'] == -10.0
        assert penalties['traffic_light'] == -15.0
    
    def test_timeout(self):
        """Test longer timeout for complex scenarios."""
        assert self.suite.get_timeout_per_episode() == 150.0


class TestOutOfDistributionSuite:
    """Test the Out-of-Distribution evaluation suite."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {'episodes_per_map': 35}
        self.suite = OutOfDistributionSuite(self.config)
    
    def test_initialization(self):
        """Test suite initialization."""
        assert self.suite.suite_type == SuiteType.OUT_OF_DISTRIBUTION
        assert self.suite.suite_name == "out_of_distribution"
        assert "Unseen environmental conditions" in self.suite.get_description()
    
    def test_environment_config(self):
        """Test environment configuration for OOD conditions."""
        env_config = self.suite.get_environment_config()
        
        # Check novel conditions
        assert env_config['unseen_textures'] is True
        assert env_config['night_conditions'] is True
        assert env_config['rain_simulation'] is True
        assert env_config['novel_obstacles'] is True
        assert env_config['texture_domain_shift'] is True
        
        # Check sensor degradation
        assert env_config['sensor_noise'] == 0.8
        assert env_config['sensor_degradation'] is True
        assert env_config['motion_blur'] == 0.6
    
    def test_evaluation_config(self):
        """Test evaluation configuration for OOD testing."""
        eval_config = self.suite.get_evaluation_config()
        
        assert eval_config['domain_shift_testing'] is True
        assert eval_config['robustness_evaluation'] is True
        assert eval_config['generalization_testing'] is True
        assert eval_config['novelty_detection'] is True
        assert eval_config['ood_specific_metrics'] is True
    
    def test_maps(self):
        """Test map selection includes custom/novel maps."""
        maps = self.suite.get_maps()
        assert '_custom_technical_floor' in maps
        assert '_huge_C_floor' in maps
        assert '_huge_V_floor' in maps


class TestStressAdversarialSuite:
    """Test the Stress/Adversarial evaluation suite."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {'episodes_per_map': 25}
        self.suite = StressAdversarialSuite(self.config)
    
    def test_initialization(self):
        """Test suite initialization."""
        assert self.suite.suite_type == SuiteType.STRESS_ADVERSARIAL
        assert self.suite.suite_name == "stress_adversarial"
        assert "Extreme stress testing" in self.suite.get_description()
    
    def test_environment_config(self):
        """Test environment configuration for stress testing."""
        env_config = self.suite.get_environment_config()
        
        # Check sensor failures
        assert env_config['sensor_dropouts'] == 0.3
        assert env_config['camera_blackouts'] is True
        assert env_config['sensor_calibration_drift'] is True
        
        # Check actuator problems
        assert env_config['wheel_bias'] == 0.4
        assert env_config['steering_lag'] == 0.3
        assert env_config['brake_degradation'] == 0.3
        
        # Check adversarial conditions
        assert env_config['moving_obstacles'] is True
        assert env_config['aggressive_agents'] is True
        assert env_config['adversarial_conditions'] is True
        assert env_config['extreme_lighting'] is True
        
        # Check emergency scenarios
        assert env_config['emergency_stops'] is True
        assert env_config['collision_avoidance_scenarios'] is True
    
    def test_evaluation_config(self):
        """Test evaluation configuration for stress testing."""
        eval_config = self.suite.get_evaluation_config()
        
        assert eval_config['failure_mode_testing'] is True
        assert eval_config['recovery_testing'] is True
        assert eval_config['safety_validation'] is True
        assert eval_config['stress_response_analysis'] is True
        assert eval_config['emergency_response_testing'] is True
        assert eval_config['fault_tolerance_evaluation'] is True
    
    def test_timeout(self):
        """Test longer timeout for recovery scenarios."""
        assert self.suite.get_timeout_per_episode() == 180.0
    
    def test_episodes_per_map(self):
        """Test fewer episodes due to extreme difficulty."""
        assert self.suite.get_episodes_per_map() == 25


class TestTestSuiteFactory:
    """Test the test suite factory."""
    
    def test_create_suite(self):
        """Test suite creation."""
        config = {'episodes_per_map': 30}
        
        # Test all available suites
        for suite_name in TestSuiteFactory.get_available_suites():
            suite = TestSuiteFactory.create_suite(suite_name, config)
            assert isinstance(suite, BaseTestSuite)
            assert suite.suite_name == suite_name
    
    def test_unknown_suite(self):
        """Test error handling for unknown suite."""
        with pytest.raises(ValueError, match="Unknown suite"):
            TestSuiteFactory.create_suite('unknown_suite', {})
    
    def test_get_available_suites(self):
        """Test getting available suite names."""
        suites = TestSuiteFactory.get_available_suites()
        
        expected_suites = [
            'base', 'hard_randomization', 'law_intersection',
            'out_of_distribution', 'stress_adversarial'
        ]
        
        for expected in expected_suites:
            assert expected in suites
    
    def test_register_custom_suite(self):
        """Test registering a custom suite."""
        
        class CustomSuite(BaseTestSuite):
            def _get_suite_type(self):
                return SuiteType.BASE
            
            def _get_suite_name(self):
                return "custom"
            
            def get_environment_config(self):
                return {}
            
            def get_evaluation_config(self):
                return {}
            
            def get_maps(self):
                return ['custom_map']
            
            def get_description(self):
                return "Custom test suite"
        
        # Register the custom suite
        TestSuiteFactory.register_suite('custom', CustomSuite)
        
        # Test it can be created
        suite = TestSuiteFactory.create_suite('custom', {})
        assert isinstance(suite, CustomSuite)
        assert suite.suite_name == "custom"
        
        # Clean up
        del TestSuiteFactory._suite_classes['custom']
    
    def test_register_invalid_suite(self):
        """Test error handling for invalid suite class."""
        
        class InvalidSuite:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseTestSuite"):
            TestSuiteFactory.register_suite('invalid', InvalidSuite)


class TestCreateAllSuiteConfigs:
    """Test the create_all_suite_configs function."""
    
    def test_create_all_configs(self):
        """Test creating all suite configurations."""
        config = {
            'base_config': {'episodes_per_map': 50},
            'hard_randomization_config': {'episodes_per_map': 40},
            'law_intersection_config': {'episodes_per_map': 30},
            'out_of_distribution_config': {'episodes_per_map': 35},
            'stress_adversarial_config': {'episodes_per_map': 25}
        }
        
        suite_configs = create_all_suite_configs(config)
        
        # Check all suites are created
        expected_suites = [
            'base', 'hard_randomization', 'law_intersection',
            'out_of_distribution', 'stress_adversarial'
        ]
        
        for suite_name in expected_suites:
            assert suite_name in suite_configs
            suite_config = suite_configs[suite_name]
            assert isinstance(suite_config, SuiteConfig)
            assert suite_config.suite_name == suite_name
    
    def test_empty_config(self):
        """Test with empty configuration."""
        suite_configs = create_all_suite_configs({})
        
        # Should still create all suites with defaults
        assert len(suite_configs) == 5
        
        for suite_config in suite_configs.values():
            assert isinstance(suite_config, SuiteConfig)
            assert suite_config.episodes_per_map > 0
            assert len(suite_config.maps) > 0


class TestSuiteIntegration:
    """Integration tests for test suites."""
    
    def test_suite_config_serialization(self):
        """Test that suite configs can be serialized to JSON."""
        config = {'episodes_per_map': 30}
        suite = BaseSuite(config)
        suite_config = suite.create_suite_config()
        
        # Convert to dict for JSON serialization
        config_dict = {
            'suite_name': suite_config.suite_name,
            'suite_type': suite_config.suite_type.value,
            'description': suite_config.description,
            'maps': suite_config.maps,
            'episodes_per_map': suite_config.episodes_per_map,
            'environment_config': suite_config.environment_config,
            'evaluation_config': suite_config.evaluation_config,
            'timeout_per_episode': suite_config.timeout_per_episode
        }
        
        # Should be JSON serializable
        json_str = json.dumps(config_dict, indent=2)
        assert len(json_str) > 0
        
        # Should be deserializable
        loaded_config = json.loads(json_str)
        assert loaded_config['suite_name'] == 'base'
        assert loaded_config['suite_type'] == 'base'
    
    def test_all_suites_have_required_attributes(self):
        """Test that all suites have required attributes."""
        config = {}
        
        for suite_name in TestSuiteFactory.get_available_suites():
            suite = TestSuiteFactory.create_suite(suite_name, config)
            
            # Check required methods return valid values
            assert isinstance(suite.get_maps(), list)
            assert len(suite.get_maps()) > 0
            assert isinstance(suite.get_episodes_per_map(), int)
            assert suite.get_episodes_per_map() > 0
            assert isinstance(suite.get_timeout_per_episode(), (int, float))
            assert suite.get_timeout_per_episode() > 0
            assert isinstance(suite.get_description(), str)
            assert len(suite.get_description()) > 0
            assert isinstance(suite.get_environment_config(), dict)
            assert isinstance(suite.get_evaluation_config(), dict)
    
    def test_suite_difficulty_progression(self):
        """Test that suites have appropriate difficulty progression."""
        config = {}
        
        # Create all suites
        suites = {}
        for suite_name in TestSuiteFactory.get_available_suites():
            suites[suite_name] = TestSuiteFactory.create_suite(suite_name, config)
        
        # Base suite should have minimal variation
        base_env = suites['base'].get_environment_config()
        assert base_env.get('lighting_variation', 0) == 0.0
        assert base_env.get('texture_variation', 0) == 0.0
        assert base_env.get('camera_noise', 0) == 0.0
        
        # Hard randomization should have high variation
        hard_env = suites['hard_randomization'].get_environment_config()
        assert hard_env.get('lighting_variation', 0) > 0.5
        assert hard_env.get('texture_variation', 0) > 0.5
        assert hard_env.get('camera_noise', 0) > 0.5
        
        # Stress suite should have the longest timeout
        stress_timeout = suites['stress_adversarial'].get_timeout_per_episode()
        base_timeout = suites['base'].get_timeout_per_episode()
        assert stress_timeout > base_timeout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])