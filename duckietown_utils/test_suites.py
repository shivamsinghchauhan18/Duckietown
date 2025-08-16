#!/usr/bin/env python3
"""
🧪 EVALUATION TEST SUITES 🧪
Concrete implementations of evaluation test suites for comprehensive model testing

This module implements the actual test suite classes that define specific
environmental conditions, parameters, and evaluation protocols for each
suite type: Base, Hard Randomization, Law/Intersection, Out-of-Distribution,
and Stress/Adversarial.
"""

import os
import sys
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.suite_manager import SuiteType, SuiteConfig, EpisodeResult

class BaseTestSuite(ABC):
    """Abstract base class for evaluation test suites."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test suite.
        
        Args:
            config: Configuration dictionary for the suite
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.suite_type = self._get_suite_type()
        self.suite_name = self._get_suite_name()
        
    @abstractmethod
    def _get_suite_type(self) -> SuiteType:
        """Get the suite type."""
        pass
    
    @abstractmethod
    def _get_suite_name(self) -> str:
        """Get the suite name."""
        pass
    
    @abstractmethod
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for this suite."""
        pass
    
    @abstractmethod
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration for this suite."""
        pass
    
    @abstractmethod
    def get_maps(self) -> List[str]:
        """Get list of maps for this suite."""
        pass
    
    def get_episodes_per_map(self) -> int:
        """Get number of episodes per map."""
        return self.config.get('episodes_per_map', 50)
    
    def get_timeout_per_episode(self) -> float:
        """Get timeout per episode in seconds."""
        return self.config.get('timeout_per_episode', 120.0)
    
    def create_suite_config(self) -> SuiteConfig:
        """Create a SuiteConfig for this test suite."""
        return SuiteConfig(
            suite_name=self.suite_name,
            suite_type=self.suite_type,
            description=self.get_description(),
            maps=self.get_maps(),
            episodes_per_map=self.get_episodes_per_map(),
            environment_config=self.get_environment_config(),
            evaluation_config=self.get_evaluation_config(),
            timeout_per_episode=self.get_timeout_per_episode()
        )
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of this test suite."""
        pass
    
    def setup_environment(self, env, seed: int) -> Any:
        """Setup environment for this suite.
        
        Args:
            env: The environment to configure
            seed: Random seed for reproducibility
            
        Returns:
            Configured environment
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Apply environment configuration
        env_config = self.get_environment_config()
        return self._apply_environment_config(env, env_config)
    
    def _apply_environment_config(self, env, config: Dict[str, Any]) -> Any:
        """Apply environment configuration to the environment."""
        # This is a placeholder - in practice, you'd configure the actual environment
        # based on the specific environment implementation
        return env
    
    def validate_episode_result(self, result: EpisodeResult) -> bool:
        """Validate an episode result for this suite."""
        # Basic validation - can be overridden by specific suites
        return (
            result.episode_id is not None and
            result.map_name is not None and
            result.seed is not None and
            isinstance(result.success, bool) and
            isinstance(result.reward, (int, float)) and
            isinstance(result.episode_length, int) and
            result.episode_length >= 0
        )


class BaseSuite(BaseTestSuite):
    """Base evaluation suite with clean environmental conditions."""
    
    def _get_suite_type(self) -> SuiteType:
        return SuiteType.BASE
    
    def _get_suite_name(self) -> str:
        return "base"
    
    def get_description(self) -> str:
        return "Clean environmental conditions with default parameters for baseline evaluation"
    
    def get_maps(self) -> List[str]:
        """Get maps for base suite - standard, well-tested maps."""
        return self.config.get('maps', [
            'LF-norm-loop',
            'LF-norm-small_loop', 
            'LF-norm-zigzag',
            'LF-norm-techtrack'
        ])
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Clean environment configuration."""
        return {
            # Lighting conditions
            'lighting_variation': 0.0,
            'ambient_light': 1.0,
            'directional_light': 1.0,
            'shadow_intensity': 0.5,
            
            # Texture and visual
            'texture_variation': 0.0,
            'texture_quality': 'high',
            'road_texture': 'default',
            'lane_marking_quality': 'high',
            
            # Camera settings
            'camera_noise': 0.0,
            'camera_blur': 0.0,
            'camera_distortion': 0.0,
            'image_compression': 0.0,
            
            # Physics
            'friction_variation': 0.0,
            'wheel_friction': 1.0,
            'air_resistance': 1.0,
            'gravity': 9.81,
            
            # Traffic and obstacles
            'traffic_density': 0.0,
            'static_obstacles': False,
            'dynamic_obstacles': False,
            'pedestrians': False,
            
            # Weather
            'weather_effects': False,
            'wind_speed': 0.0,
            'rain_intensity': 0.0,
            'fog_density': 0.0,
            
            # Sensor configuration
            'sensor_noise': 0.0,
            'sensor_dropouts': 0.0,
            'sensor_delay': 0.0
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Evaluation configuration for base suite."""
        return {
            'deterministic_spawn': True,
            'fixed_spawn_poses': True,
            'spawn_pose_variation': 0.0,
            'fixed_weather': True,
            'clean_textures': True,
            'consistent_lighting': True,
            'disable_randomization': True,
            'record_trajectories': True,
            'save_observations': False,
            'detailed_logging': True
        }


class HardRandomizationSuite(BaseTestSuite):
    """Hard randomization suite with heavy environmental noise and traffic."""
    
    def _get_suite_type(self) -> SuiteType:
        return SuiteType.HARD_RANDOMIZATION
    
    def _get_suite_name(self) -> str:
        return "hard_randomization"
    
    def get_description(self) -> str:
        return "Heavy environmental noise, texture variation, and moderate traffic for robustness testing"
    
    def get_maps(self) -> List[str]:
        """Get maps for hard randomization - includes larger, more complex maps."""
        return self.config.get('maps', [
            'LF-norm-loop',
            'LF-norm-zigzag', 
            'huge_loop',
            'huge_loop2',
            'multi_track'
        ])
    
    def get_episodes_per_map(self) -> int:
        """Fewer episodes per map due to increased difficulty."""
        return self.config.get('episodes_per_map', 40)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Heavy randomization environment configuration."""
        return {
            # Heavy lighting variation
            'lighting_variation': 0.8,
            'ambient_light_range': (0.3, 1.5),
            'directional_light_range': (0.2, 1.8),
            'shadow_intensity_range': (0.1, 0.9),
            'lighting_color_shift': 0.6,
            
            # Significant texture variation
            'texture_variation': 0.7,
            'texture_quality_range': ('low', 'high'),
            'road_texture_randomization': True,
            'lane_marking_degradation': 0.5,
            'texture_domain_shift': True,
            
            # Camera noise and distortion
            'camera_noise': 0.6,
            'camera_blur_range': (0.0, 0.4),
            'camera_distortion_range': (0.0, 0.3),
            'image_compression_range': (0.1, 0.5),
            'camera_angle_variation': 0.3,
            
            # Physics variation
            'friction_variation': 0.5,
            'wheel_friction_range': (0.6, 1.4),
            'air_resistance_range': (0.8, 1.2),
            'gravity_variation': 0.1,
            'wheel_slip_probability': 0.2,
            
            # Moderate traffic
            'traffic_density': 0.4,
            'static_obstacles': True,
            'static_obstacle_density': 0.3,
            'dynamic_obstacles': True,
            'dynamic_obstacle_density': 0.2,
            'pedestrians': True,
            'pedestrian_density': 0.1,
            
            # Weather effects
            'weather_effects': True,
            'wind_speed_range': (0.0, 5.0),
            'rain_intensity_range': (0.0, 0.4),
            'fog_density_range': (0.0, 0.3),
            'weather_change_probability': 0.3,
            
            # Sensor noise
            'sensor_noise': 0.4,
            'sensor_dropouts': 0.1,
            'sensor_delay_range': (0.0, 0.05)
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Evaluation configuration for hard randomization."""
        return {
            'deterministic_spawn': False,
            'spawn_pose_variation': 0.5,
            'randomize_spawn_time': True,
            'weather_variation': True,
            'texture_randomization': True,
            'lighting_randomization': True,
            'physics_randomization': True,
            'traffic_randomization': True,
            'record_trajectories': True,
            'save_observations': True,
            'detailed_logging': True,
            'failure_analysis': True
        }


class LawIntersectionSuite(BaseTestSuite):
    """Law/Intersection suite for traffic rule compliance testing."""
    
    def _get_suite_type(self) -> SuiteType:
        return SuiteType.LAW_INTERSECTION
    
    def _get_suite_name(self) -> str:
        return "law_intersection"
    
    def get_description(self) -> str:
        return "Traffic rule compliance testing with intersections, stop signs, and right-of-way scenarios"
    
    def get_maps(self) -> List[str]:
        """Get maps with intersections and traffic rules."""
        return self.config.get('maps', [
            'ETHZ_autolab_technical_track',
            'multi_track',
            'multi_track2',
            '_custom_technical_floor'
        ])
    
    def get_episodes_per_map(self) -> int:
        """Moderate episode count for focused testing."""
        return self.config.get('episodes_per_map', 30)
    
    def get_timeout_per_episode(self) -> float:
        """Longer timeout for complex intersection scenarios."""
        return self.config.get('timeout_per_episode', 150.0)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Environment configuration focused on traffic rules."""
        return {
            # Moderate environmental variation
            'lighting_variation': 0.3,
            'texture_variation': 0.2,
            'camera_noise': 0.2,
            'friction_variation': 0.1,
            
            # Traffic rule elements
            'stop_signs': True,
            'stop_sign_density': 0.8,
            'traffic_lights': True,
            'traffic_light_density': 0.6,
            'yield_signs': True,
            'speed_limit_signs': True,
            
            # Intersection complexity
            'intersection_complexity': 0.7,
            'four_way_intersections': True,
            'three_way_intersections': True,
            'roundabouts': True,
            'crosswalks': True,
            
            # Right-of-way scenarios
            'right_of_way_scenarios': True,
            'priority_road_markings': True,
            'merge_scenarios': True,
            'lane_change_zones': True,
            
            # Traffic participants
            'other_vehicles': True,
            'other_vehicle_density': 0.5,
            'pedestrians': True,
            'pedestrian_density': 0.3,
            'cyclists': True,
            'cyclist_density': 0.2,
            
            # Rule enforcement
            'speed_monitoring': True,
            'stop_line_detection': True,
            'traffic_light_compliance': True,
            'lane_discipline': True
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Evaluation configuration for law/intersection testing."""
        return {
            'track_violations': True,
            'violation_categories': [
                'stop_sign_violations',
                'traffic_light_violations', 
                'speed_violations',
                'right_of_way_violations',
                'lane_violations',
                'crosswalk_violations'
            ],
            'intersection_testing': True,
            'traffic_rule_enforcement': True,
            'compliance_scoring': True,
            'violation_penalties': {
                'stop_sign': -10.0,
                'traffic_light': -15.0,
                'speed': -5.0,
                'right_of_way': -12.0,
                'lane': -3.0,
                'crosswalk': -8.0
            },
            'record_trajectories': True,
            'save_violation_videos': True,
            'detailed_logging': True,
            'legal_compliance_analysis': True
        }


class OutOfDistributionSuite(BaseTestSuite):
    """Out-of-Distribution suite with unseen conditions and sensor noise."""
    
    def _get_suite_type(self) -> SuiteType:
        return SuiteType.OUT_OF_DISTRIBUTION
    
    def _get_suite_name(self) -> str:
        return "out_of_distribution"
    
    def get_description(self) -> str:
        return "Unseen environmental conditions, novel textures, and sensor noise for generalization testing"
    
    def get_maps(self) -> List[str]:
        """Get maps with novel/unseen characteristics."""
        return self.config.get('maps', [
            '_custom_technical_floor',
            '_huge_C_floor',
            '_huge_V_floor',
            '_plus_floor',
            '_myTestA'
        ])
    
    def get_episodes_per_map(self) -> int:
        """Moderate episode count for OOD testing."""
        return self.config.get('episodes_per_map', 35)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Out-of-distribution environment configuration."""
        return {
            # Novel visual conditions
            'unseen_textures': True,
            'texture_domain_shift': True,
            'novel_road_materials': True,
            'unusual_lane_markings': True,
            'non_standard_colors': True,
            
            # Extreme lighting conditions
            'night_conditions': True,
            'low_light_probability': 0.4,
            'extreme_brightness': True,
            'high_contrast_lighting': True,
            'unusual_shadow_patterns': True,
            
            # Weather extremes
            'rain_simulation': True,
            'heavy_rain_probability': 0.3,
            'snow_conditions': True,
            'fog_conditions': True,
            'extreme_weather_probability': 0.2,
            
            # Sensor degradation
            'sensor_noise': 0.8,
            'sensor_degradation': True,
            'camera_lens_effects': True,
            'motion_blur': 0.6,
            'compression_artifacts': 0.5,
            
            # Novel obstacles and objects
            'novel_obstacles': True,
            'unusual_object_shapes': True,
            'non_standard_vehicles': True,
            'construction_zones': True,
            'temporary_signage': True,
            
            # Environmental shifts
            'seasonal_variations': True,
            'urban_vs_rural_shift': True,
            'infrastructure_age_variation': True,
            'cultural_road_differences': True,
            
            # Physics variations
            'unusual_friction_surfaces': True,
            'elevation_changes': True,
            'banking_variations': True,
            'surface_irregularities': True
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Evaluation configuration for OOD testing."""
        return {
            'domain_shift_testing': True,
            'robustness_evaluation': True,
            'generalization_testing': True,
            'adaptation_analysis': True,
            'novelty_detection': True,
            'performance_degradation_tracking': True,
            'failure_mode_analysis': True,
            'recovery_capability_testing': True,
            'record_trajectories': True,
            'save_failure_cases': True,
            'detailed_logging': True,
            'ood_specific_metrics': True
        }


class StressAdversarialSuite(BaseTestSuite):
    """Stress/Adversarial suite with sensor failures and extreme conditions."""
    
    def _get_suite_type(self) -> SuiteType:
        return SuiteType.STRESS_ADVERSARIAL
    
    def _get_suite_name(self) -> str:
        return "stress_adversarial"
    
    def get_description(self) -> str:
        return "Extreme stress testing with sensor failures, adversarial conditions, and safety-critical scenarios"
    
    def get_maps(self) -> List[str]:
        """Get challenging maps for stress testing."""
        return self.config.get('maps', [
            'huge_loop2',
            'multi_track2',
            '_loop_dyn_duckiebots',
            '_loop_duckies',
            'ETHZ_autolab_technical_track'
        ])
    
    def get_episodes_per_map(self) -> int:
        """Fewer episodes due to extreme difficulty."""
        return self.config.get('episodes_per_map', 25)
    
    def get_timeout_per_episode(self) -> float:
        """Longer timeout for recovery scenarios."""
        return self.config.get('timeout_per_episode', 180.0)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Extreme stress environment configuration."""
        return {
            # Sensor failures
            'sensor_dropouts': 0.3,
            'sensor_failure_duration': (1.0, 5.0),
            'camera_blackouts': True,
            'intermittent_sensor_loss': True,
            'sensor_calibration_drift': True,
            
            # Actuator problems
            'wheel_bias': 0.4,
            'steering_lag': 0.3,
            'throttle_response_delay': 0.2,
            'brake_degradation': 0.3,
            'actuator_noise': 0.5,
            
            # Moving obstacles and adversarial agents
            'moving_obstacles': True,
            'aggressive_agents': True,
            'unpredictable_behavior': True,
            'obstacle_density': 0.6,
            'dynamic_obstacle_speed': (0.5, 2.0),
            
            # Extreme environmental conditions
            'extreme_lighting': True,
            'strobe_lighting': True,
            'complete_darkness_periods': True,
            'blinding_light_sources': True,
            'rapid_lighting_changes': True,
            
            # Adversarial conditions
            'adversarial_conditions': True,
            'adversarial_textures': True,
            'adversarial_patterns': True,
            'optical_illusions': True,
            'misleading_lane_markings': True,
            
            # Physical stress
            'extreme_friction_variations': True,
            'slippery_surfaces': 0.4,
            'rough_terrain': True,
            'surface_damage': True,
            'debris_on_road': True,
            
            # Communication and system stress
            'network_latency': 0.5,
            'packet_loss': 0.2,
            'system_overload': True,
            'memory_pressure': True,
            'cpu_throttling': 0.3,
            
            # Emergency scenarios
            'emergency_stops': True,
            'collision_avoidance_scenarios': True,
            'near_miss_situations': True,
            'recovery_scenarios': True
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Evaluation configuration for stress/adversarial testing."""
        return {
            'failure_mode_testing': True,
            'recovery_testing': True,
            'safety_validation': True,
            'stress_response_analysis': True,
            'adversarial_robustness': True,
            'emergency_response_testing': True,
            'fault_tolerance_evaluation': True,
            'graceful_degradation_testing': True,
            'safety_critical_scenarios': True,
            'record_all_episodes': True,
            'save_failure_videos': True,
            'detailed_sensor_logging': True,
            'system_performance_monitoring': True,
            'safety_metric_tracking': True,
            'recovery_time_measurement': True
        }


class TestSuiteFactory:
    """Factory for creating test suite instances."""
    
    _suite_classes = {
        'base': BaseSuite,
        'hard_randomization': HardRandomizationSuite,
        'law_intersection': LawIntersectionSuite,
        'out_of_distribution': OutOfDistributionSuite,
        'stress_adversarial': StressAdversarialSuite
    }
    
    @classmethod
    def create_suite(cls, suite_name: str, config: Dict[str, Any]) -> BaseTestSuite:
        """Create a test suite instance.
        
        Args:
            suite_name: Name of the suite to create
            config: Configuration for the suite
            
        Returns:
            BaseTestSuite: Instance of the requested suite
            
        Raises:
            ValueError: If suite_name is not recognized
        """
        if suite_name not in cls._suite_classes:
            available = list(cls._suite_classes.keys())
            raise ValueError(f"Unknown suite: {suite_name}. Available: {available}")
        
        suite_class = cls._suite_classes[suite_name]
        return suite_class(config)
    
    @classmethod
    def get_available_suites(cls) -> List[str]:
        """Get list of available suite names."""
        return list(cls._suite_classes.keys())
    
    @classmethod
    def register_suite(cls, suite_name: str, suite_class: type):
        """Register a custom suite class.
        
        Args:
            suite_name: Name for the suite
            suite_class: Class implementing BaseTestSuite
        """
        if not issubclass(suite_class, BaseTestSuite):
            raise ValueError("Suite class must inherit from BaseTestSuite")
        
        cls._suite_classes[suite_name] = suite_class


def create_all_suite_configs(config: Dict[str, Any]) -> Dict[str, SuiteConfig]:
    """Create SuiteConfig objects for all available test suites.
    
    Args:
        config: Global configuration dictionary
        
    Returns:
        Dict mapping suite names to SuiteConfig objects
    """
    suite_configs = {}
    
    for suite_name in TestSuiteFactory.get_available_suites():
        suite_config = config.get(f'{suite_name}_config', {})
        suite = TestSuiteFactory.create_suite(suite_name, suite_config)
        suite_configs[suite_name] = suite.create_suite_config()
    
    return suite_configs


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create all suites with default configuration
    config = {
        'base_config': {'episodes_per_map': 50},
        'hard_randomization_config': {'episodes_per_map': 40},
        'law_intersection_config': {'episodes_per_map': 30},
        'out_of_distribution_config': {'episodes_per_map': 35},
        'stress_adversarial_config': {'episodes_per_map': 25}
    }
    
    suite_configs = create_all_suite_configs(config)
    
    print("🧪 Available Test Suites:")
    for name, suite_config in suite_configs.items():
        print(f"\n📋 {name.upper()}:")
        print(f"   Type: {suite_config.suite_type.value}")
        print(f"   Description: {suite_config.description}")
        print(f"   Maps: {len(suite_config.maps)}")
        print(f"   Episodes per map: {suite_config.episodes_per_map}")
        print(f"   Total episodes: {len(suite_config.maps) * suite_config.episodes_per_map}")
        print(f"   Timeout: {suite_config.timeout_per_episode}s")