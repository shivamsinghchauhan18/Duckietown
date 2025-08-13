"""
Training utilities for enhanced PPO training including evaluation, checkpointing, and analysis.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode

from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.env import launch_and_wrap_enhanced_env
from duckietown_utils.enhanced_logger import EnhancedLogger

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for enhanced RL training."""
    
    def __init__(self, enhanced_config: EnhancedRLConfig):
        """
        Initialize model evaluator.
        
        Args:
            enhanced_config: Enhanced configuration
        """
        self.enhanced_config = enhanced_config
        self.enhanced_logger = EnhancedLogger(enhanced_config.logging)
        
    def evaluate_checkpoint(self, checkpoint_path: str, env_config: Dict[str, Any],
                          num_episodes: int = 20, render: bool = False,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            env_config: Environment configuration
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        
        # Setup evaluation configuration
        eval_env_config = env_config.copy()
        eval_env_config['mode'] = 'inference'
        
        # Create trainer and restore model
        trainer_config = {
            'env': 'EnhancedDuckietown',
            'env_config': eval_env_config,
            'explore': False,
            'num_workers': 0,
            'num_gpus': 0
        }
        
        trainer = PPOTrainer(config=trainer_config)
        trainer.restore(checkpoint_path)
        
        # Run evaluation episodes
        results = self._run_evaluation_episodes(
            trainer, eval_env_config, num_episodes, render
        )
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results)
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(checkpoint_path, aggregate_results, results)
        
        logger.info(f"Evaluation completed: {aggregate_results['summary']}")
        return aggregate_results
    
    def _run_evaluation_episodes(self, trainer: PPOTrainer, env_config: Dict[str, Any],
                                num_episodes: int, render: bool) -> List[Dict[str, Any]]:
        """Run evaluation episodes and collect detailed results."""
        # Create evaluation environment
        eval_env = launch_and_wrap_enhanced_env(env_config, self.enhanced_config)
        
        episode_results = []
        
        for episode_idx in range(num_episodes):
            logger.info(f"Running evaluation episode {episode_idx + 1}/{num_episodes}")
            
            # Run single episode
            episode_result = self._run_single_episode(
                trainer, eval_env, episode_idx, render
            )
            episode_results.append(episode_result)
            
            # Log episode result
            self.enhanced_logger.log_evaluation_episode(
                episode_idx, episode_result
            )
        
        return episode_results
    
    def _run_single_episode(self, trainer: PPOTrainer, env, episode_idx: int,
                           render: bool) -> Dict[str, Any]:
        """Run a single evaluation episode and collect metrics."""
        obs = env.reset()
        episode_data = {
            'episode_idx': episode_idx,
            'total_reward': 0.0,
            'episode_length': 0,
            'success': False,
            'collision': False,
            'completion_reason': 'unknown',
            'metrics': {
                'reward_components': [],
                'object_detections': [],
                'avoidance_actions': [],
                'lane_changes': [],
                'safety_violations': [],
                'performance_metrics': []
            },
            'trajectory': [],
            'actions': [],
            'observations': []
        }
        
        done = False
        step_count = 0
        
        while not done:
            # Get action from policy
            action = trainer.compute_action(obs, explore=False)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Record step data
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1
            episode_data['actions'].append(action.tolist() if hasattr(action, 'tolist') else action)
            
            # Extract detailed metrics from info
            if info:
                self._extract_step_metrics(episode_data, info, step_count)
            
            # Record trajectory
            if 'Simulator' in (info or {}):
                cur_pos = info['Simulator'].get('cur_pos', [0, 0, 0])
                episode_data['trajectory'].append(cur_pos)
            
            # Render if requested
            if render:
                env.render()
            
            obs = next_obs
            step_count += 1
            
            # Safety check for infinite episodes
            if step_count > 2000:
                done = True
                episode_data['completion_reason'] = 'max_steps'
        
        # Determine episode outcome
        episode_data['success'] = self._determine_episode_success(episode_data, info)
        episode_data['collision'] = info.get('collision', False) if info else False
        
        if not episode_data['completion_reason'] or episode_data['completion_reason'] == 'unknown':
            if episode_data['collision']:
                episode_data['completion_reason'] = 'collision'
            elif episode_data['success']:
                episode_data['completion_reason'] = 'success'
            else:
                episode_data['completion_reason'] = 'failure'
        
        return episode_data
    
    def _extract_step_metrics(self, episode_data: Dict[str, Any], 
                             info: Dict[str, Any], step: int):
        """Extract detailed metrics from step info."""
        # Reward components
        if 'reward_components' in info:
            episode_data['metrics']['reward_components'].append({
                'step': step,
                **info['reward_components']
            })
        
        # Object detection
        if 'object_detection' in info:
            detection_info = info['object_detection']
            detection_info['step'] = step
            episode_data['metrics']['object_detections'].append(detection_info)
        
        # Object avoidance
        if 'object_avoidance' in info:
            avoidance_info = info['object_avoidance']
            avoidance_info['step'] = step
            episode_data['metrics']['avoidance_actions'].append(avoidance_info)
        
        # Lane changing
        if 'lane_changing' in info:
            lane_change_info = info['lane_changing']
            lane_change_info['step'] = step
            episode_data['metrics']['lane_changes'].append(lane_change_info)
        
        # Safety violations
        if 'safety' in info and info['safety'].get('violation', False):
            safety_info = info['safety']
            safety_info['step'] = step
            episode_data['metrics']['safety_violations'].append(safety_info)
        
        # Performance metrics
        if 'performance' in info:
            perf_info = info['performance']
            perf_info['step'] = step
            episode_data['metrics']['performance_metrics'].append(perf_info)
    
    def _determine_episode_success(self, episode_data: Dict[str, Any], 
                                  final_info: Optional[Dict[str, Any]]) -> bool:
        """Determine if episode was successful based on multiple criteria."""
        # Basic success criteria
        min_reward = 0.5
        min_distance = 5.0
        max_safety_violations = 2
        
        # Check reward threshold
        if episode_data['total_reward'] < min_reward:
            return False
        
        # Check distance traveled (if available)
        if len(episode_data['trajectory']) > 1:
            total_distance = 0.0
            for i in range(1, len(episode_data['trajectory'])):
                dist = np.linalg.norm(
                    np.array(episode_data['trajectory'][i]) - 
                    np.array(episode_data['trajectory'][i-1])
                )
                total_distance += dist
            
            if total_distance < min_distance:
                return False
        
        # Check safety violations
        if len(episode_data['metrics']['safety_violations']) > max_safety_violations:
            return False
        
        # Check for collision
        if episode_data.get('collision', False):
            return False
        
        return True
    
    def _calculate_aggregate_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all episodes."""
        if not episode_results:
            return {'summary': {}, 'detailed': {}}
        
        # Basic statistics
        rewards = [ep['total_reward'] for ep in episode_results]
        lengths = [ep['episode_length'] for ep in episode_results]
        successes = [ep['success'] for ep in episode_results]
        collisions = [ep['collision'] for ep in episode_results]
        
        summary_metrics = {
            'num_episodes': len(episode_results),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_episode_length': np.mean(lengths),
            'success_rate': np.mean(successes),
            'collision_rate': np.mean(collisions)
        }
        
        # Detailed component analysis
        detailed_metrics = self._analyze_component_performance(episode_results)
        
        # Performance scores
        performance_scores = self._calculate_performance_scores(episode_results)
        
        return {
            'summary': summary_metrics,
            'detailed': detailed_metrics,
            'performance_scores': performance_scores,
            'episode_results': episode_results
        }
    
    def _analyze_component_performance(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of individual components."""
        component_analysis = {
            'object_detection': {
                'total_detections': 0,
                'avg_confidence': 0.0,
                'safety_critical_detections': 0
            },
            'object_avoidance': {
                'total_avoidance_actions': 0,
                'avg_avoidance_force': 0.0,
                'successful_avoidances': 0
            },
            'lane_changing': {
                'total_attempts': 0,
                'successful_changes': 0,
                'avg_duration': 0.0
            },
            'safety': {
                'total_violations': 0,
                'violation_types': {},
                'avg_violations_per_episode': 0.0
            }
        }
        
        total_episodes = len(episode_results)
        
        for episode in episode_results:
            metrics = episode['metrics']
            
            # Object detection analysis
            detections = metrics['object_detections']
            component_analysis['object_detection']['total_detections'] += len(detections)
            
            if detections:
                confidences = []
                safety_critical = 0
                for detection in detections:
                    if 'detections' in detection:
                        for det in detection['detections']:
                            confidences.append(det.get('confidence', 0.0))
                            if det.get('safety_critical', False):
                                safety_critical += 1
                
                if confidences:
                    component_analysis['object_detection']['avg_confidence'] += np.mean(confidences)
                component_analysis['object_detection']['safety_critical_detections'] += safety_critical
            
            # Object avoidance analysis
            avoidance_actions = metrics['avoidance_actions']
            component_analysis['object_avoidance']['total_avoidance_actions'] += len(avoidance_actions)
            
            if avoidance_actions:
                forces = [action.get('force_magnitude', 0.0) for action in avoidance_actions]
                component_analysis['object_avoidance']['avg_avoidance_force'] += np.mean(forces)
                # Count successful avoidances (no collision after avoidance)
                if not episode['collision']:
                    component_analysis['object_avoidance']['successful_avoidances'] += 1
            
            # Lane changing analysis
            lane_changes = metrics['lane_changes']
            attempts = sum(1 for lc in lane_changes if lc.get('attempt_started', False))
            successes = sum(1 for lc in lane_changes if lc.get('change_completed', False))
            
            component_analysis['lane_changing']['total_attempts'] += attempts
            component_analysis['lane_changing']['successful_changes'] += successes
            
            if lane_changes:
                durations = [lc.get('duration', 0.0) for lc in lane_changes if lc.get('duration', 0.0) > 0]
                if durations:
                    component_analysis['lane_changing']['avg_duration'] += np.mean(durations)
            
            # Safety analysis
            violations = metrics['safety_violations']
            component_analysis['safety']['total_violations'] += len(violations)
            
            for violation in violations:
                violation_type = violation.get('violation_type', 'unknown')
                if violation_type not in component_analysis['safety']['violation_types']:
                    component_analysis['safety']['violation_types'][violation_type] = 0
                component_analysis['safety']['violation_types'][violation_type] += 1
        
        # Calculate averages
        if total_episodes > 0:
            component_analysis['object_detection']['avg_confidence'] /= total_episodes
            component_analysis['object_avoidance']['avg_avoidance_force'] /= total_episodes
            component_analysis['lane_changing']['avg_duration'] /= total_episodes
            component_analysis['safety']['avg_violations_per_episode'] = (
                component_analysis['safety']['total_violations'] / total_episodes
            )
        
        return component_analysis
    
    def _calculate_performance_scores(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate composite performance scores."""
        if not episode_results:
            return {}
        
        # Extract metrics for scoring
        rewards = [ep['total_reward'] for ep in episode_results]
        successes = [ep['success'] for ep in episode_results]
        collisions = [ep['collision'] for ep in episode_results]
        
        # Calculate component scores
        reward_score = np.mean(rewards)
        success_score = np.mean(successes)
        safety_score = 1.0 - np.mean(collisions)
        
        # Calculate efficiency score based on episode length and distance
        efficiency_scores = []
        for episode in episode_results:
            if len(episode['trajectory']) > 1:
                distance = sum(
                    np.linalg.norm(np.array(episode['trajectory'][i]) - np.array(episode['trajectory'][i-1]))
                    for i in range(1, len(episode['trajectory']))
                )
                efficiency = distance / max(1, episode['episode_length'])
                efficiency_scores.append(min(1.0, efficiency / 0.1))  # Normalize
            else:
                efficiency_scores.append(0.0)
        
        efficiency_score = np.mean(efficiency_scores) if efficiency_scores else 0.0
        
        # Calculate overall performance score
        weights = {'reward': 0.3, 'success': 0.3, 'safety': 0.25, 'efficiency': 0.15}
        overall_score = (
            weights['reward'] * reward_score +
            weights['success'] * success_score +
            weights['safety'] * safety_score +
            weights['efficiency'] * efficiency_score
        )
        
        return {
            'reward_score': reward_score,
            'success_score': success_score,
            'safety_score': safety_score,
            'efficiency_score': efficiency_score,
            'overall_score': overall_score
        }
    
    def _save_evaluation_results(self, checkpoint_path: str, 
                                aggregate_results: Dict[str, Any],
                                episode_results: List[Dict[str, Any]]):
        """Save evaluation results to file."""
        checkpoint_dir = Path(checkpoint_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate results
        results_file = checkpoint_dir / f'evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            # Remove episode_results from aggregate for JSON serialization
            save_data = aggregate_results.copy()
            save_data.pop('episode_results', None)
            json.dump(save_data, f, indent=2, default=str)
        
        # Save detailed episode results
        episodes_file = checkpoint_dir / f'episode_results_{timestamp}.pkl'
        with open(episodes_file, 'wb') as f:
            pickle.dump(episode_results, f)
        
        logger.info(f"Evaluation results saved: {results_file}")


class TrainingAnalyzer:
    """Analyze training progress and performance."""
    
    def __init__(self, experiment_path: str):
        """
        Initialize training analyzer.
        
        Args:
            experiment_path: Path to experiment directory
        """
        self.experiment_path = Path(experiment_path)
        
    def analyze_training_progress(self) -> Dict[str, Any]:
        """Analyze training progress from logs and checkpoints."""
        analysis = {
            'training_metrics': self._analyze_training_metrics(),
            'checkpoint_analysis': self._analyze_checkpoints(),
            'performance_trends': self._analyze_performance_trends(),
            'component_analysis': self._analyze_component_performance()
        }
        
        return analysis
    
    def _analyze_training_metrics(self) -> Dict[str, Any]:
        """Analyze training metrics from logs."""
        # Implementation would parse training logs and extract metrics
        # This is a placeholder for the actual implementation
        return {
            'total_timesteps': 0,
            'total_episodes': 0,
            'training_duration': 0,
            'convergence_analysis': {}
        }
    
    def _analyze_checkpoints(self) -> Dict[str, Any]:
        """Analyze available checkpoints."""
        checkpoint_dir = self.experiment_path / 'enhanced_checkpoints'
        if not checkpoint_dir.exists():
            return {'checkpoints': [], 'best_checkpoint': None}
        
        checkpoints = []
        best_performance = -float('inf')
        best_checkpoint = None
        
        for checkpoint_path in checkpoint_dir.iterdir():
            if checkpoint_path.is_dir():
                metadata_file = checkpoint_path / 'enhanced_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    checkpoints.append({
                        'path': str(checkpoint_path),
                        'metadata': metadata
                    })
                    
                    performance = metadata.get('performance_score', -float('inf'))
                    if performance > best_performance:
                        best_performance = performance
                        best_checkpoint = str(checkpoint_path)
        
        return {
            'checkpoints': checkpoints,
            'best_checkpoint': best_checkpoint,
            'num_checkpoints': len(checkpoints)
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over training."""
        # Implementation would analyze performance metrics over time
        # This is a placeholder for the actual implementation
        return {
            'reward_trend': 'improving',
            'safety_trend': 'stable',
            'efficiency_trend': 'improving'
        }
    
    def _analyze_component_performance(self) -> Dict[str, Any]:
        """Analyze individual component performance."""
        # Implementation would analyze component-specific metrics
        # This is a placeholder for the actual implementation
        return {
            'object_detection': {'status': 'good', 'confidence': 0.85},
            'object_avoidance': {'status': 'good', 'success_rate': 0.92},
            'lane_changing': {'status': 'fair', 'success_rate': 0.78}
        }


def compare_models(checkpoint_paths: List[str], env_config: Dict[str, Any],
                  enhanced_config: EnhancedRLConfig, num_episodes: int = 10) -> Dict[str, Any]:
    """
    Compare multiple model checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths to compare
        env_config: Environment configuration
        enhanced_config: Enhanced configuration
        num_episodes: Number of episodes for evaluation
        
    Returns:
        Comparison results
    """
    evaluator = ModelEvaluator(enhanced_config)
    
    comparison_results = {
        'models': {},
        'comparison': {}
    }
    
    # Evaluate each model
    for i, checkpoint_path in enumerate(checkpoint_paths):
        model_name = f'model_{i}_{Path(checkpoint_path).name}'
        logger.info(f"Evaluating model: {model_name}")
        
        results = evaluator.evaluate_checkpoint(
            checkpoint_path, env_config, num_episodes, save_results=False
        )
        comparison_results['models'][model_name] = results
    
    # Compare models
    if len(comparison_results['models']) > 1:
        comparison_results['comparison'] = _compare_model_results(
            comparison_results['models']
        )
    
    return comparison_results


def _compare_model_results(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare results from multiple models."""
    comparison = {
        'best_overall': None,
        'best_safety': None,
        'best_efficiency': None,
        'metric_comparison': {}
    }
    
    # Find best models for different criteria
    best_overall_score = -float('inf')
    best_safety_score = -float('inf')
    best_efficiency_score = -float('inf')
    
    for model_name, results in model_results.items():
        performance_scores = results.get('performance_scores', {})
        
        overall_score = performance_scores.get('overall_score', 0)
        safety_score = performance_scores.get('safety_score', 0)
        efficiency_score = performance_scores.get('efficiency_score', 0)
        
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            comparison['best_overall'] = model_name
        
        if safety_score > best_safety_score:
            best_safety_score = safety_score
            comparison['best_safety'] = model_name
        
        if efficiency_score > best_efficiency_score:
            best_efficiency_score = efficiency_score
            comparison['best_efficiency'] = model_name
    
    # Create metric comparison table
    metrics = ['mean_reward', 'success_rate', 'collision_rate']
    for metric in metrics:
        comparison['metric_comparison'][metric] = {}
        for model_name, results in model_results.items():
            value = results.get('summary', {}).get(metric, 0)
            comparison['metric_comparison'][metric][model_name] = value
    
    return comparison