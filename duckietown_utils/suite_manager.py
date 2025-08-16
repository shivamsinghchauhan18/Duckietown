#!/usr/bin/env python3
"""
🧪 SUITE MANAGER 🧪
Manages different evaluation test suites for comprehensive model testing

This module implements the SuiteManager class for coordinating different
evaluation suites including Base, Hard Randomization, Law/Intersection,
Out-of-Distribution, and Stress/Adversarial suites.
"""

import os
import sys
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class SuiteType(Enum):
    """Evaluation suite types."""
    BASE = "base"
    HARD_RANDOMIZATION = "hard_randomization"
    LAW_INTERSECTION = "law_intersection"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    STRESS_ADVERSARIAL = "stress_adversarial"

@dataclass
class SuiteConfig:
    """Configuration for an evaluation suite."""
    suite_name: str
    suite_type: SuiteType
    description: str
    maps: List[str]
    episodes_per_map: int
    environment_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    timeout_per_episode: float = 120.0  # seconds
    
    def __post_init__(self):
        """Validate suite configuration."""
        if not self.maps:
            raise ValueError("Suite must specify at least one map")
        if self.episodes_per_map <= 0:
            raise ValueError("Episodes per map must be positive")

@dataclass
class EpisodeResult:
    """Results from a single episode evaluation."""
    episode_id: str
    map_name: str
    seed: int
    success: bool
    reward: float
    episode_length: int
    lateral_deviation: float
    heading_error: float
    jerk: float
    stability: float
    collision: bool = False
    off_lane: bool = False
    violations: Dict[str, int] = field(default_factory=dict)
    lap_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class SuiteResults:
    """Results from running a complete evaluation suite."""
    suite_name: str
    suite_type: SuiteType
    model_id: str
    policy_mode: str
    total_episodes: int
    successful_episodes: int
    episode_results: List[EpisodeResult]
    
    # Aggregated metrics
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_episode_length: float = 0.0
    mean_lateral_deviation: float = 0.0
    mean_heading_error: float = 0.0
    mean_jerk: float = 0.0
    stability: float = 0.0
    
    # Safety metrics
    collision_rate: float = 0.0
    off_lane_rate: float = 0.0
    violation_rate: float = 0.0
    
    # Execution metadata
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Calculate aggregated metrics from episode results."""
        if not self.episode_results:
            return
        
        # Basic aggregations
        self.success_rate = self.successful_episodes / max(self.total_episodes, 1)
        self.mean_reward = np.mean([ep.reward for ep in self.episode_results])
        self.mean_episode_length = np.mean([ep.episode_length for ep in self.episode_results])
        self.mean_lateral_deviation = np.mean([ep.lateral_deviation for ep in self.episode_results])
        self.mean_heading_error = np.mean([ep.heading_error for ep in self.episode_results])
        self.mean_jerk = np.mean([ep.jerk for ep in self.episode_results])
        
        # Stability calculation
        rewards = [ep.reward for ep in self.episode_results]
        if rewards:
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            self.stability = 1.0 - (reward_std / max(reward_mean, 1.0)) if reward_mean > 0 else 0.0
            self.stability = max(self.stability, 0.0)
        
        # Safety metrics
        self.collision_rate = np.mean([1.0 if ep.collision else 0.0 for ep in self.episode_results])
        self.off_lane_rate = np.mean([1.0 if ep.off_lane else 0.0 for ep in self.episode_results])
        
        # Violation rate (normalized)
        total_violations = sum([sum(ep.violations.values()) for ep in self.episode_results])
        self.violation_rate = total_violations / max(self.total_episodes, 1) / 10.0  # Normalize

class SuiteManager:
    """Manages different evaluation test suites."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the suite manager.
        
        Args:
            config: Configuration dictionary for the suite manager
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize suite configurations
        self.suite_configs: Dict[str, SuiteConfig] = {}
        self._initialize_default_suites()
        
        # Load custom suites from config
        custom_suites = self.config.get('custom_suites', {})
        for suite_name, suite_config in custom_suites.items():
            self._register_custom_suite(suite_name, suite_config)
        
        # Results storage
        self.results_dir = Path(self.config.get('results_dir', 'logs/suite_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🧪 Suite Manager initialized with {len(self.suite_configs)} suites")
    
    def _initialize_default_suites(self):
        """Initialize default evaluation suites using concrete test suite implementations."""
        from duckietown_utils.test_suites import TestSuiteFactory
        
        # Create suite configurations using the concrete test suite classes
        suite_names = ['base', 'hard_randomization', 'law_intersection', 
                      'out_of_distribution', 'stress_adversarial']
        
        for suite_name in suite_names:
            try:
                # Get suite-specific config from main config
                suite_config = self.config.get(f'{suite_name}_config', {})
                
                # Create the test suite instance
                test_suite = TestSuiteFactory.create_suite(suite_name, suite_config)
                
                # Create the SuiteConfig and register it
                self.suite_configs[suite_name] = test_suite.create_suite_config()
                
                self.logger.info(f"Initialized {suite_name} suite with {len(test_suite.get_maps())} maps")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {suite_name} suite: {e}")
                # Continue with other suites even if one fails
    
    def _register_custom_suite(self, suite_name: str, suite_config: Dict[str, Any]):
        """Register a custom evaluation suite."""
        try:
            config = SuiteConfig(
                suite_name=suite_name,
                suite_type=SuiteType(suite_config.get('suite_type', 'base')),
                description=suite_config.get('description', f'Custom suite: {suite_name}'),
                maps=suite_config.get('maps', []),
                episodes_per_map=suite_config.get('episodes_per_map', 30),
                environment_config=suite_config.get('environment_config', {}),
                evaluation_config=suite_config.get('evaluation_config', {}),
                timeout_per_episode=suite_config.get('timeout_per_episode', 120.0)
            )
            
            self.suite_configs[suite_name] = config
            self.logger.info(f"Registered custom suite: {suite_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register custom suite {suite_name}: {e}")
    
    def get_suite_config(self, suite_name: str) -> Optional[SuiteConfig]:
        """Get configuration for a specific suite."""
        return self.suite_configs.get(suite_name)
    
    def list_suites(self) -> List[str]:
        """List all available suite names."""
        return list(self.suite_configs.keys())
    
    def get_suite_info(self, suite_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a suite."""
        config = self.get_suite_config(suite_name)
        if not config:
            return None
        
        return {
            'name': config.suite_name,
            'type': config.suite_type.value,
            'description': config.description,
            'maps': config.maps,
            'episodes_per_map': config.episodes_per_map,
            'total_episodes': len(config.maps) * config.episodes_per_map,
            'timeout_per_episode': config.timeout_per_episode,
            'environment_config': config.environment_config,
            'evaluation_config': config.evaluation_config
        }
    
    def run_suite(self, suite_name: str, model, seeds: List[int], 
                  policy_mode: str = "deterministic",
                  progress_callback: Optional[Callable[[float], None]] = None) -> SuiteResults:
        """Run a complete evaluation suite.
        
        Args:
            suite_name: Name of the suite to run
            model: The model to evaluate
            seeds: List of seeds to use for evaluation
            policy_mode: Policy mode ('deterministic' or 'stochastic')
            progress_callback: Optional callback for progress updates
            
        Returns:
            SuiteResults: Complete results from the suite evaluation
        """
        suite_config = self.get_suite_config(suite_name)
        if not suite_config:
            raise ValueError(f"Suite not found: {suite_name}")
        
        self.logger.info(f"🧪 Running suite: {suite_name}")
        self.logger.info(f"📍 Maps: {suite_config.maps}")
        self.logger.info(f"🎲 Seeds per map: {suite_config.episodes_per_map}")
        
        start_time = time.time()
        episode_results = []
        successful_episodes = 0
        total_episodes = 0
        
        # Calculate total episodes for progress tracking
        total_planned_episodes = len(suite_config.maps) * suite_config.episodes_per_map
        
        # Run evaluation on each map
        for map_idx, map_name in enumerate(suite_config.maps):
            self.logger.info(f"📍 Evaluating on map: {map_name} ({map_idx + 1}/{len(suite_config.maps)})")
            
            # Get seeds for this map
            map_seeds = seeds[:suite_config.episodes_per_map]
            
            # Run episodes on this map
            for episode_idx, seed in enumerate(map_seeds):
                episode_id = f"{suite_name}_{map_name}_{seed}_{int(time.time())}"
                
                try:
                    # Run single episode
                    episode_result = self._run_episode(
                        episode_id=episode_id,
                        map_name=map_name,
                        model=model,
                        seed=seed,
                        suite_config=suite_config,
                        policy_mode=policy_mode
                    )
                    
                    episode_results.append(episode_result)
                    total_episodes += 1
                    
                    if episode_result.success:
                        successful_episodes += 1
                    
                    # Update progress
                    if progress_callback:
                        progress = (total_episodes / total_planned_episodes) * 100.0
                        progress_callback(progress)
                    
                    if episode_idx % 10 == 0:
                        self.logger.info(f"  Episode {episode_idx + 1}/{len(map_seeds)} completed")
                
                except Exception as e:
                    self.logger.error(f"Episode {episode_id} failed: {e}")
                    # Create failed episode result
                    failed_result = EpisodeResult(
                        episode_id=episode_id,
                        map_name=map_name,
                        seed=seed,
                        success=False,
                        reward=0.0,
                        episode_length=0,
                        lateral_deviation=999.0,
                        heading_error=999.0,
                        jerk=999.0,
                        stability=0.0,
                        collision=True,
                        metadata={'error': str(e)}
                    )
                    episode_results.append(failed_result)
                    total_episodes += 1
        
        # Create suite results
        execution_time = time.time() - start_time
        
        suite_results = SuiteResults(
            suite_name=suite_name,
            suite_type=suite_config.suite_type,
            model_id=getattr(model, 'model_id', 'unknown'),
            policy_mode=policy_mode,
            total_episodes=total_episodes,
            successful_episodes=successful_episodes,
            episode_results=episode_results,
            execution_time=execution_time
        )
        
        # Save results
        self._save_suite_results(suite_results)
        
        self.logger.info(f"✅ Suite {suite_name} completed in {execution_time/60:.1f} minutes")
        self.logger.info(f"📊 Success rate: {suite_results.success_rate:.3f}")
        self.logger.info(f"🏆 Mean reward: {suite_results.mean_reward:.3f}")
        
        return suite_results
    
    def _run_episode(self, episode_id: str, map_name: str, model, seed: int,
                    suite_config: SuiteConfig, policy_mode: str) -> EpisodeResult:
        """Run a single episode evaluation."""
        
        # For now, simulate episode execution
        # In a real implementation, this would create the environment and run the episode
        episode_result = self._simulate_episode(
            episode_id=episode_id,
            map_name=map_name,
            model=model,
            seed=seed,
            suite_config=suite_config,
            policy_mode=policy_mode
        )
        
        return episode_result
    
    def _simulate_episode(self, episode_id: str, map_name: str, model, seed: int,
                         suite_config: SuiteConfig, policy_mode: str) -> EpisodeResult:
        """Simulate episode execution (placeholder for actual evaluation)."""
        
        # Set random seed for reproducible simulation
        random.seed(seed)
        np.random.seed(seed)
        
        # Simulate episode execution time
        time.sleep(0.1)  # Brief simulation delay
        
        # Generate results based on suite type and configuration
        suite_difficulty = self._get_suite_difficulty(suite_config.suite_type)
        
        # Base success probability adjusted by suite difficulty
        base_success_prob = 0.85 - (suite_difficulty * 0.3)
        success = random.random() < base_success_prob
        
        # Generate metrics with noise based on suite difficulty
        reward = random.uniform(0.4, 0.9) * (1.0 - suite_difficulty * 0.3)
        episode_length = int(random.uniform(400, 900) * (1.0 + suite_difficulty * 0.5))
        
        lateral_deviation = random.uniform(0.02, 0.25) * (1.0 + suite_difficulty * 2.0)
        heading_error = random.uniform(1.0, 15.0) * (1.0 + suite_difficulty * 1.5)
        jerk = random.uniform(0.01, 0.15) * (1.0 + suite_difficulty * 2.0)
        stability = random.uniform(0.6, 0.95) * (1.0 - suite_difficulty * 0.2)
        
        # Safety events based on suite difficulty
        collision = random.random() < (suite_difficulty * 0.15)
        off_lane = random.random() < (suite_difficulty * 0.2)
        
        # Traffic violations for law/intersection suite
        violations = {}
        if suite_config.suite_type == SuiteType.LAW_INTERSECTION:
            violations = {
                'stop_sign_violations': random.randint(0, 2),
                'speed_violations': random.randint(0, 3),
                'right_of_way_violations': random.randint(0, 1)
            }
        
        # Lap time if successful
        lap_time = random.uniform(25.0, 45.0) if success else None
        
        # Adjust success based on safety events
        if collision or off_lane:
            success = False
        
        return EpisodeResult(
            episode_id=episode_id,
            map_name=map_name,
            seed=seed,
            success=success,
            reward=reward,
            episode_length=episode_length,
            lateral_deviation=lateral_deviation,
            heading_error=heading_error,
            jerk=jerk,
            stability=stability,
            collision=collision,
            off_lane=off_lane,
            violations=violations,
            lap_time=lap_time,
            metadata={
                'suite_type': suite_config.suite_type.value,
                'policy_mode': policy_mode,
                'difficulty': suite_difficulty
            }
        )
    
    def _get_suite_difficulty(self, suite_type: SuiteType) -> float:
        """Get difficulty multiplier for suite type."""
        difficulty_map = {
            SuiteType.BASE: 0.0,
            SuiteType.HARD_RANDOMIZATION: 0.3,
            SuiteType.LAW_INTERSECTION: 0.4,
            SuiteType.OUT_OF_DISTRIBUTION: 0.6,
            SuiteType.STRESS_ADVERSARIAL: 0.8
        }
        return difficulty_map.get(suite_type, 0.5)
    
    def _save_suite_results(self, suite_results: SuiteResults):
        """Save suite results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"suite_{suite_results.suite_name}_{suite_results.model_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to dictionary for JSON serialization
        results_dict = asdict(suite_results)
        
        # Convert enum to string for JSON serialization
        results_dict['suite_type'] = suite_results.suite_type.value
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Saved suite results: {filepath}")
    
    def load_suite_results(self, filepath: Union[str, Path]) -> Optional[SuiteResults]:
        """Load suite results from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert back to dataclass
            # Note: This is a simplified conversion - in practice you'd need proper deserialization
            suite_results = SuiteResults(**data)
            return suite_results
            
        except Exception as e:
            self.logger.error(f"Failed to load suite results from {filepath}: {e}")
            return None
    
    def get_suite_statistics(self, suite_name: str) -> Dict[str, Any]:
        """Get statistics about a suite's historical results."""
        # This would analyze historical results files
        # For now, return basic suite information
        config = self.get_suite_config(suite_name)
        if not config:
            return {}
        
        return {
            'suite_name': suite_name,
            'suite_type': config.suite_type.value,
            'total_maps': len(config.maps),
            'episodes_per_map': config.episodes_per_map,
            'total_episodes': len(config.maps) * config.episodes_per_map,
            'estimated_runtime_minutes': (len(config.maps) * config.episodes_per_map * 
                                        config.timeout_per_episode) / 60.0
        }
    
    def validate_suite_config(self, suite_name: str) -> Dict[str, Any]:
        """Validate a suite configuration."""
        config = self.get_suite_config(suite_name)
        if not config:
            return {'valid': False, 'errors': [f'Suite not found: {suite_name}']}
        
        errors = []
        warnings = []
        
        # Check maps exist (simplified check)
        for map_name in config.maps:
            # In practice, you'd check if the map file exists
            if not map_name:
                errors.append(f"Empty map name in suite {suite_name}")
        
        # Check reasonable episode counts
        if config.episodes_per_map > 100:
            warnings.append(f"High episode count ({config.episodes_per_map}) may take long time")
        
        # Check timeout values
        if config.timeout_per_episode < 30.0:
            warnings.append(f"Short timeout ({config.timeout_per_episode}s) may cause premature termination")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'estimated_runtime_minutes': (len(config.maps) * config.episodes_per_map * 
                                        config.timeout_per_episode) / 60.0
        }