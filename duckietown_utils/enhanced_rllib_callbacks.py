"""
Enhanced RLLib callbacks for multi-objective rewards, curriculum learning, and advanced monitoring.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

import ray.rllib as rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.tune.result import TRAINING_ITERATION, TIMESTEPS_TOTAL

from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.enhanced_logger import EnhancedLogger

logger = logging.getLogger(__name__)


class EnhancedRLLibCallbacks(DefaultCallbacks):
    """Enhanced callbacks for multi-objective RL training with comprehensive monitoring."""
    
    def __init__(self, enhanced_config: EnhancedRLConfig, 
                 enhanced_logger: EnhancedLogger,
                 curriculum_callback: Optional['CurriculumLearningCallback'] = None,
                 checkpoint_callback: Optional['ModelCheckpointCallback'] = None):
        """
        Initialize enhanced callbacks.
        
        Args:
            enhanced_config: Enhanced configuration
            enhanced_logger: Enhanced logger instance
            curriculum_callback: Curriculum learning callback
            checkpoint_callback: Model checkpoint callback
        """
        super().__init__()
        self.enhanced_config = enhanced_config
        self.enhanced_logger = enhanced_logger
        self.curriculum_callback = curriculum_callback
        self.checkpoint_callback = checkpoint_callback
        
        # Metrics tracking
        self.episode_metrics = {}
        self.training_metrics = {}
        
        logger.info("Enhanced RLLib callbacks initialized")
    
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy], episode: MultiAgentEpisode,
                        env_index: int, **kwargs):
        """Initialize episode tracking with enhanced metrics."""
        # Standard metrics from original callbacks
        episode.user_data['robot_speed'] = []
        episode.user_data['robot_cur_pos'] = []
        episode.user_data['deviation_centerline'] = []
        episode.user_data['deviation_heading'] = []
        episode.user_data['distance_travelled'] = []
        episode.user_data['distance_travelled_any'] = []
        episode.user_data['proximity_penalty'] = []
        episode.user_data['collision_risk_step_cnt'] = 0
        
        # Enhanced multi-objective reward tracking
        episode.user_data['reward_components'] = {
            'lane_following': [],
            'object_avoidance': [],
            'lane_change': [],
            'efficiency': [],
            'safety_penalty': [],
            'total': []
        }
        
        # Object detection and avoidance metrics
        episode.user_data['object_detections'] = []
        episode.user_data['avoidance_actions'] = []
        episode.user_data['safety_violations'] = []
        
        # Lane changing metrics
        episode.user_data['lane_changes'] = []
        episode.user_data['lane_change_attempts'] = 0
        episode.user_data['lane_change_successes'] = 0
        
        # Performance metrics
        episode.user_data['processing_times'] = {
            'detection': [],
            'action': [],
            'total': []
        }
        
        # Curriculum learning metrics
        episode.user_data['curriculum_stage'] = getattr(self.curriculum_callback, 'current_stage', 0)
        episode.user_data['scenario_difficulty'] = 0.0
        
        # Custom histogram data
        episode.hist_data['sampled_actions'] = []
        episode.hist_data['reward_components'] = []
        episode.hist_data['detection_confidences'] = []
        episode.hist_data['avoidance_forces'] = []
        episode.hist_data['_robot_coordinates'] = []
        
        # Log episode start
        if self.enhanced_config.logging.log_actions:
            self.enhanced_logger.log_episode_start(episode.episode_id, env_index)
    
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
        """Track step-level metrics with enhanced monitoring."""
        # Get environment info
        env_info = episode.last_info_for()
        if env_info is None:
            return
        
        # Standard metrics tracking (from original callbacks)
        self._track_standard_metrics(episode, env_info)
        
        # Enhanced reward component tracking
        self._track_reward_components(episode, env_info)
        
        # Object detection and avoidance tracking
        self._track_object_detection(episode, env_info)
        
        # Lane changing tracking
        self._track_lane_changing(episode, env_info)
        
        # Performance tracking
        self._track_performance_metrics(episode, env_info)
        
        # Action tracking with enhanced details
        action = episode.last_action_for()
        if action is not None:
            episode.hist_data['sampled_actions'].append(np.clip(action, -1.0, 1.0))
            
            # Log action details
            if self.enhanced_config.logging.log_actions:
                self.enhanced_logger.log_action(
                    episode.episode_id,
                    episode.length,
                    action,
                    env_info.get('action_reasoning', {}),
                    env_info.get('safety_checks', {})
                )
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                      policies: Dict[str, Policy], episode: MultiAgentEpisode,
                      env_index: int, **kwargs):
        """Calculate episode-level metrics and log results."""
        # Standard metrics (from original callbacks)
        episode.custom_metrics['mean_robot_speed'] = np.mean(episode.user_data['robot_speed'])
        episode.custom_metrics['deviation_centerline'] = np.mean(episode.user_data['deviation_centerline'])
        episode.custom_metrics['deviation_heading'] = np.mean(episode.user_data['deviation_heading'])
        episode.custom_metrics['distance_travelled'] = np.sum(episode.user_data['distance_travelled'])
        episode.custom_metrics['distance_travelled_any'] = np.sum(episode.user_data['distance_travelled_any'])
        episode.custom_metrics['proximity_penalty'] = np.sum(episode.user_data['proximity_penalty'])
        episode.custom_metrics['collision_risk_step_cnt'] = episode.user_data['collision_risk_step_cnt']
        
        # Enhanced multi-objective reward metrics
        for component, values in episode.user_data['reward_components'].items():
            if values:
                episode.custom_metrics[f'{component}_reward_mean'] = np.mean(values)
                episode.custom_metrics[f'{component}_reward_total'] = np.sum(values)
        
        # Object detection metrics
        detections = episode.user_data['object_detections']
        if detections:
            episode.custom_metrics['objects_detected_total'] = len(detections)
            episode.custom_metrics['avg_detection_confidence'] = np.mean([d['confidence'] for d in detections])
            episode.custom_metrics['safety_critical_detections'] = sum(1 for d in detections if d.get('safety_critical', False))
        
        # Avoidance metrics
        avoidance_actions = episode.user_data['avoidance_actions']
        if avoidance_actions:
            episode.custom_metrics['avoidance_actions_total'] = len(avoidance_actions)
            episode.custom_metrics['avg_avoidance_force'] = np.mean([a['force_magnitude'] for a in avoidance_actions])
        
        # Lane changing metrics
        episode.custom_metrics['lane_change_attempts'] = episode.user_data['lane_change_attempts']
        episode.custom_metrics['lane_change_successes'] = episode.user_data['lane_change_successes']
        episode.custom_metrics['lane_change_success_rate'] = (
            episode.user_data['lane_change_successes'] / max(1, episode.user_data['lane_change_attempts'])
        )
        
        # Safety metrics
        safety_violations = episode.user_data['safety_violations']
        episode.custom_metrics['safety_violations_total'] = len(safety_violations)
        episode.custom_metrics['safety_violations_mean'] = len(safety_violations) / max(1, episode.length)
        
        # Performance metrics
        processing_times = episode.user_data['processing_times']
        for metric_name, times in processing_times.items():
            if times:
                episode.custom_metrics[f'processing_time_{metric_name}_mean'] = np.mean(times)
                episode.custom_metrics[f'processing_time_{metric_name}_max'] = np.max(times)
        
        # Curriculum learning metrics
        episode.custom_metrics['curriculum_stage'] = episode.user_data['curriculum_stage']
        episode.custom_metrics['scenario_difficulty'] = episode.user_data['scenario_difficulty']
        
        # Calculate composite scores
        episode.custom_metrics['overall_performance_score'] = self._calculate_performance_score(episode)
        episode.custom_metrics['safety_score'] = self._calculate_safety_score(episode)
        episode.custom_metrics['efficiency_score'] = self._calculate_efficiency_score(episode)
        
        # Store robot coordinates for trajectory plotting
        episode.hist_data['_robot_coordinates'].append(episode.user_data['robot_cur_pos'])
        
        # Log episode completion
        if self.enhanced_config.logging.log_rewards:
            self.enhanced_logger.log_episode_end(
                episode.episode_id,
                episode.total_reward,
                episode.length,
                episode.custom_metrics
            )
    
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Process training results with enhanced metrics and curriculum learning."""
        # Clean up histogram data (from original callbacks)
        episodes_this_iter = result['episodes_this_iter']
        timesteps_this_iter = result['timesteps_this_iter']
        
        # Clean custom histograms
        hist_stats = result.get('hist_stats', {})
        for key in ['sampled_actions', 'reward_components', 'detection_confidences', 
                   'avoidance_forces', '_robot_coordinates']:
            if key in hist_stats:
                if key == '_robot_coordinates':
                    hist_stats[key] = hist_stats[key][:episodes_this_iter]
                else:
                    hist_stats[key] = hist_stats[key][:timesteps_this_iter]
        
        # Clean built-in histograms
        for key in ['episode_lengths', 'episode_reward']:
            if key in hist_stats:
                hist_stats[key] = hist_stats[key][:episodes_this_iter]
        
        # Apply curriculum learning updates
        if self.curriculum_callback:
            self.curriculum_callback.on_train_result(trainer, result)
        
        # Apply checkpoint callback
        if self.checkpoint_callback:
            self.checkpoint_callback.on_train_result(trainer, result)
        
        # Log training metrics
        if self.enhanced_config.logging.log_performance:
            self.enhanced_logger.log_training_result(
                result.get(TRAINING_ITERATION, 0),
                result.get(TIMESTEPS_TOTAL, 0),
                result.get('episode_reward_mean', 0.0),
                result.get('custom_metrics', {})
            )
    
    def _track_standard_metrics(self, episode: MultiAgentEpisode, env_info: Dict[str, Any]):
        """Track standard metrics from original callbacks."""
        simulator_info = env_info.get('Simulator', {})
        
        episode.user_data['robot_speed'].append(simulator_info.get('robot_speed', 0.0))
        episode.user_data['proximity_penalty'].append(simulator_info.get('proximity_penalty', 0.0))
        
        if simulator_info.get('proximity_penalty', 0.0) < 0.0:
            episode.user_data['collision_risk_step_cnt'] += 1
        
        # Lane position tracking
        if 'lane_position' in simulator_info:
            lane_pos = simulator_info['lane_position']
            episode.user_data['deviation_centerline'].append(abs(lane_pos.get('dist', 0.0)))
            episode.user_data['deviation_heading'].append(abs(lane_pos.get('angle_deg', 0.0)))
        
        # Position tracking
        cur_pos = simulator_info.get('cur_pos', [0, 0, 0])
        episode.user_data['robot_cur_pos'].append(cur_pos)
        
        # Distance calculation
        if len(episode.user_data['robot_cur_pos']) > 1:
            prev_pos = episode.user_data['robot_cur_pos'][-2]
            dist_travelled_any = np.linalg.norm(np.array(cur_pos) - np.array(prev_pos))
            episode.user_data['distance_travelled_any'].append(dist_travelled_any)
            
            # Distance in correct lane
            if 'lane_position' in simulator_info and simulator_info['lane_position'].get('dist', -1) > -0.1:
                episode.user_data['distance_travelled'].append(dist_travelled_any)
            else:
                episode.user_data['distance_travelled'].append(0.0)
        else:
            episode.user_data['distance_travelled_any'].append(0.0)
            episode.user_data['distance_travelled'].append(0.0)
    
    def _track_reward_components(self, episode: MultiAgentEpisode, env_info: Dict[str, Any]):
        """Track multi-objective reward components."""
        reward_info = env_info.get('reward_components', {})
        
        for component in episode.user_data['reward_components']:
            value = reward_info.get(component, 0.0)
            episode.user_data['reward_components'][component].append(value)
            episode.hist_data['reward_components'].append(value)
    
    def _track_object_detection(self, episode: MultiAgentEpisode, env_info: Dict[str, Any]):
        """Track object detection and avoidance metrics."""
        detection_info = env_info.get('object_detection', {})
        
        if detection_info:
            detections = detection_info.get('detections', [])
            for detection in detections:
                episode.user_data['object_detections'].append(detection)
                episode.hist_data['detection_confidences'].append(detection.get('confidence', 0.0))
                
                # Log detection if enabled
                if self.enhanced_config.logging.log_detections:
                    self.enhanced_logger.log_detection(
                        episode.episode_id,
                        episode.length,
                        detection
                    )
        
        # Track avoidance actions
        avoidance_info = env_info.get('object_avoidance', {})
        if avoidance_info:
            episode.user_data['avoidance_actions'].append(avoidance_info)
            episode.hist_data['avoidance_forces'].append(avoidance_info.get('force_magnitude', 0.0))
    
    def _track_lane_changing(self, episode: MultiAgentEpisode, env_info: Dict[str, Any]):
        """Track lane changing metrics."""
        lane_change_info = env_info.get('lane_changing', {})
        
        if lane_change_info:
            if lane_change_info.get('attempt_started', False):
                episode.user_data['lane_change_attempts'] += 1
            
            if lane_change_info.get('change_completed', False):
                episode.user_data['lane_change_successes'] += 1
                episode.user_data['lane_changes'].append({
                    'step': episode.length,
                    'duration': lane_change_info.get('duration', 0.0),
                    'success': True
                })
    
    def _track_performance_metrics(self, episode: MultiAgentEpisode, env_info: Dict[str, Any]):
        """Track performance and timing metrics."""
        performance_info = env_info.get('performance', {})
        
        for metric_name in ['detection', 'action', 'total']:
            if metric_name in performance_info:
                episode.user_data['processing_times'][metric_name].append(
                    performance_info[metric_name]
                )
        
        # Track safety violations
        safety_info = env_info.get('safety', {})
        if safety_info.get('violation', False):
            episode.user_data['safety_violations'].append({
                'step': episode.length,
                'type': safety_info.get('violation_type', 'unknown'),
                'severity': safety_info.get('severity', 1.0)
            })
    
    def _calculate_performance_score(self, episode: MultiAgentEpisode) -> float:
        """Calculate overall performance score."""
        # Weighted combination of different performance aspects
        weights = {
            'distance': 0.3,
            'lane_following': 0.25,
            'safety': 0.25,
            'efficiency': 0.2
        }
        
        # Distance score
        distance_score = min(1.0, episode.custom_metrics.get('distance_travelled', 0.0) / 10.0)
        
        # Lane following score
        deviation = episode.custom_metrics.get('deviation_centerline', 1.0)
        lane_score = max(0.0, 1.0 - deviation / 0.5)  # Normalize by 0.5m max deviation
        
        # Safety score
        safety_score = max(0.0, 1.0 - episode.custom_metrics.get('safety_violations_mean', 0.0))
        
        # Efficiency score (based on speed and smoothness)
        speed_score = min(1.0, episode.custom_metrics.get('mean_robot_speed', 0.0) / 0.5)
        
        return (weights['distance'] * distance_score +
                weights['lane_following'] * lane_score +
                weights['safety'] * safety_score +
                weights['efficiency'] * speed_score)
    
    def _calculate_safety_score(self, episode: MultiAgentEpisode) -> float:
        """Calculate safety score based on violations and collisions."""
        violations = episode.custom_metrics.get('safety_violations_total', 0)
        collision_risk = episode.custom_metrics.get('collision_risk_step_cnt', 0)
        episode_length = max(1, episode.length)
        
        # Penalty for violations and collision risks
        violation_penalty = violations / episode_length
        collision_penalty = collision_risk / episode_length
        
        return max(0.0, 1.0 - violation_penalty - collision_penalty)
    
    def _calculate_efficiency_score(self, episode: MultiAgentEpisode) -> float:
        """Calculate efficiency score based on speed and path optimality."""
        distance_travelled = episode.custom_metrics.get('distance_travelled', 0.0)
        distance_any = episode.custom_metrics.get('distance_travelled_any', 0.0)
        mean_speed = episode.custom_metrics.get('mean_robot_speed', 0.0)
        
        # Path efficiency (staying in lane)
        path_efficiency = distance_travelled / max(0.1, distance_any) if distance_any > 0 else 0.0
        
        # Speed efficiency
        speed_efficiency = min(1.0, mean_speed / 0.5)  # Normalize by target speed
        
        return 0.6 * path_efficiency + 0.4 * speed_efficiency


class CurriculumLearningCallback:
    """Callback for implementing curriculum learning in enhanced RL training."""
    
    def __init__(self, enhanced_config: EnhancedRLConfig):
        """
        Initialize curriculum learning callback.
        
        Args:
            enhanced_config: Enhanced configuration with curriculum settings
        """
        self.enhanced_config = enhanced_config
        self.current_stage = 0
        self.stage_start_timestep = 0
        self.stage_performance_history = []
        
        # Default curriculum stages if not specified in config
        self.stages = getattr(enhanced_config, 'curriculum_stages', [
            {
                'name': 'basic_lane_following',
                'timesteps': 200000,
                'criteria': {'episode_reward_mean': 0.5},
                'env_config': {'spawn_obstacles': False, 'domain_rand': False}
            },
            {
                'name': 'static_obstacles',
                'timesteps': 300000,
                'criteria': {'episode_reward_mean': 0.7, 'safety_score': 0.8},
                'env_config': {'spawn_obstacles': True, 'obstacles': {'duckie': {'density': 0.3, 'static': True}}}
            },
            {
                'name': 'dynamic_obstacles',
                'timesteps': 500000,
                'criteria': {'episode_reward_mean': 0.8, 'safety_score': 0.9},
                'env_config': {'spawn_obstacles': True, 'obstacles': {'duckie': {'density': 0.5, 'static': False}}}
            }
        ])
        
        logger.info(f"Curriculum learning initialized with {len(self.stages)} stages")
    
    def on_train_result(self, trainer, result: dict):
        """Update curriculum based on training progress."""
        timesteps_total = result.get(TIMESTEPS_TOTAL, 0)
        
        # Check if we should advance to next stage
        if self._should_advance_stage(result, timesteps_total):
            self._advance_to_next_stage(trainer, timesteps_total)
        
        # Update stage performance history
        self.stage_performance_history.append({
            'timestep': timesteps_total,
            'stage': self.current_stage,
            'episode_reward_mean': result.get('episode_reward_mean', 0.0),
            'custom_metrics': result.get('custom_metrics', {})
        })
    
    def _should_advance_stage(self, result: dict, timesteps_total: int) -> bool:
        """Check if criteria are met to advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False  # Already at final stage
        
        current_stage_config = self.stages[self.current_stage]
        
        # Check minimum timesteps requirement
        timesteps_in_stage = timesteps_total - self.stage_start_timestep
        if timesteps_in_stage < current_stage_config.get('min_timesteps', 50000):
            return False
        
        # Check performance criteria
        criteria = current_stage_config.get('criteria', {})
        for metric, threshold in criteria.items():
            if metric in result:
                if result[metric] < threshold:
                    return False
            elif metric in result.get('custom_metrics', {}):
                if result['custom_metrics'][metric] < threshold:
                    return False
            else:
                logger.warning(f"Curriculum metric {metric} not found in results")
                return False
        
        return True
    
    def _advance_to_next_stage(self, trainer, timesteps_total: int):
        """Advance to the next curriculum stage."""
        self.current_stage += 1
        self.stage_start_timestep = timesteps_total
        
        if self.current_stage < len(self.stages):
            stage_config = self.stages[self.current_stage]
            logger.info(f"Advancing to curriculum stage {self.current_stage}: {stage_config['name']}")
            
            # Update environment configuration
            env_config_updates = stage_config.get('env_config', {})
            if env_config_updates:
                # Apply environment configuration updates to all workers
                def update_env_config(worker):
                    for env in worker.async_env.envs:
                        for key, value in env_config_updates.items():
                            setattr(env.unwrapped, key, value)
                
                trainer.workers.foreach_worker(update_env_config)
                logger.info(f"Applied environment updates: {env_config_updates}")


class ModelCheckpointCallback:
    """Callback for advanced model checkpointing and evaluation."""
    
    def __init__(self, paths, enhanced_config: EnhancedRLConfig):
        """
        Initialize model checkpoint callback.
        
        Args:
            paths: Artifact paths for saving checkpoints
            enhanced_config: Enhanced configuration
        """
        self.paths = paths
        self.enhanced_config = enhanced_config
        self.best_performance = -float('inf')
        self.checkpoint_history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(paths.experiment_base_path) / 'enhanced_checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Model checkpoint callback initialized: {self.checkpoint_dir}")
    
    def on_train_result(self, trainer, result: dict):
        """Handle training result and create checkpoints as needed."""
        timesteps_total = result.get(TIMESTEPS_TOTAL, 0)
        episode_reward_mean = result.get('episode_reward_mean', 0.0)
        
        # Calculate composite performance score
        performance_score = self._calculate_composite_score(result)
        
        # Save checkpoint if performance improved
        if performance_score > self.best_performance:
            self.best_performance = performance_score
            checkpoint_path = self._save_enhanced_checkpoint(trainer, result, 'best')
            logger.info(f"New best model saved: {checkpoint_path} (score: {performance_score:.4f})")
        
        # Save periodic checkpoints
        if timesteps_total % 100000 == 0:  # Every 100k timesteps
            checkpoint_path = self._save_enhanced_checkpoint(trainer, result, f'timestep_{timesteps_total}')
            logger.info(f"Periodic checkpoint saved: {checkpoint_path}")
        
        # Update checkpoint history
        self.checkpoint_history.append({
            'timestep': timesteps_total,
            'performance_score': performance_score,
            'episode_reward_mean': episode_reward_mean,
            'custom_metrics': result.get('custom_metrics', {})
        })
    
    def _calculate_composite_score(self, result: dict) -> float:
        """Calculate composite performance score for checkpoint decisions."""
        weights = {
            'reward': 0.4,
            'safety': 0.3,
            'efficiency': 0.2,
            'stability': 0.1
        }
        
        # Reward component
        reward_score = result.get('episode_reward_mean', 0.0)
        
        # Safety component
        custom_metrics = result.get('custom_metrics', {})
        safety_score = custom_metrics.get('safety_score', 0.0)
        
        # Efficiency component
        efficiency_score = custom_metrics.get('efficiency_score', 0.0)
        
        # Stability component (based on reward variance)
        reward_std = result.get('episode_reward_std', 1.0)
        stability_score = max(0.0, 1.0 - reward_std / 2.0)
        
        return (weights['reward'] * reward_score +
                weights['safety'] * safety_score +
                weights['efficiency'] * efficiency_score +
                weights['stability'] * stability_score)
    
    def _save_enhanced_checkpoint(self, trainer, result: dict, suffix: str) -> str:
        """Save enhanced checkpoint with additional metadata."""
        # Save standard RLLib checkpoint
        checkpoint_path = trainer.save(str(self.checkpoint_dir))
        
        # Save enhanced metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'timesteps_total': result.get(TIMESTEPS_TOTAL, 0),
            'episode_reward_mean': result.get('episode_reward_mean', 0.0),
            'custom_metrics': result.get('custom_metrics', {}),
            'performance_score': self._calculate_composite_score(result),
            'enhanced_config': self.enhanced_config.to_dict(),
            'suffix': suffix
        }
        
        metadata_path = Path(checkpoint_path) / 'enhanced_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return checkpoint_path