"""
Enhanced RLLib loggers for multi-objective RL training with advanced visualization.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import numpy as np

import ray.tune.logger
from ray.tune.result import NODE_IP, TRAINING_ITERATION, TIMESTEPS_TOTAL
from ray.tune.utils import flatten_dict

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Install with: pip install wandb")

from duckietown_utils.trajectory_plot import plot_trajectories

logger = logging.getLogger(__name__)


class EnhancedTensorboardLogger(ray.tune.logger.Logger):
    """Enhanced Tensorboard logger with multi-objective reward visualization."""
    
    def __init__(self, config, logdir, trial):
        super().__init__(config, logdir, trial)
        self._writer = SummaryWriter(logdir=logdir, filename_suffix="_enhanced")
        self.config = config
        
        # Setup visualization parameters
        self.plot_config = {
            'figsize': (12, 8),
            'dpi': 100,
            'style': 'seaborn-v0_8',
            'color_palette': 'husl'
        }
        
        logger.info(f"Enhanced Tensorboard logger initialized: {logdir}")
    
    def on_result(self, result):
        """Log enhanced metrics and visualizations to Tensorboard."""
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        
        # Log standard metrics
        self._log_scalar_metrics(result, step)
        
        # Log multi-objective reward components
        self._log_reward_components(result, step)
        
        # Log object detection metrics
        self._log_detection_metrics(result, step)
        
        # Log performance metrics
        self._log_performance_metrics(result, step)
        
        # Log curriculum learning progress
        self._log_curriculum_metrics(result, step)
        
        # Create and log visualizations
        self._log_trajectory_visualization(result, step)
        self._log_reward_composition_plot(result, step)
        self._log_performance_dashboard(result, step)
        
        # Log histograms
        self._log_histograms(result, step)
        
        self.flush()
    
    def _log_scalar_metrics(self, result: dict, step: int):
        """Log scalar metrics to Tensorboard."""
        # Standard RL metrics
        for metric in ['episode_reward_mean', 'episode_reward_min', 'episode_reward_max',
                      'episode_len_mean', 'episodes_this_iter', 'timesteps_this_iter']:
            if metric in result:
                self._writer.add_scalar(f'Training/{metric}', result[metric], step)
        
        # Custom metrics
        custom_metrics = result.get('custom_metrics', {})
        for metric_name, value in custom_metrics.items():
            if isinstance(value, (int, float)):
                category = self._categorize_metric(metric_name)
                self._writer.add_scalar(f'{category}/{metric_name}', value, step)
    
    def _log_reward_components(self, result: dict, step: int):
        """Log multi-objective reward components."""
        custom_metrics = result.get('custom_metrics', {})
        
        # Reward components
        reward_components = {}
        for metric_name, value in custom_metrics.items():
            if 'reward' in metric_name and 'mean' in metric_name:
                component_name = metric_name.replace('_reward_mean', '')
                reward_components[component_name] = value
                self._writer.add_scalar(f'Rewards/{component_name}', value, step)
        
        # Log reward composition as stacked bar chart data
        if reward_components:
            for i, (component, value) in enumerate(reward_components.items()):
                self._writer.add_scalar(f'RewardComposition/{component}', value, step)
    
    def _log_detection_metrics(self, result: dict, step: int):
        """Log object detection and avoidance metrics."""
        custom_metrics = result.get('custom_metrics', {})
        
        detection_metrics = [
            'objects_detected_total', 'avg_detection_confidence', 'safety_critical_detections',
            'avoidance_actions_total', 'avg_avoidance_force'
        ]
        
        for metric in detection_metrics:
            if metric in custom_metrics:
                self._writer.add_scalar(f'ObjectDetection/{metric}', custom_metrics[metric], step)
    
    def _log_performance_metrics(self, result: dict, step: int):
        """Log performance and timing metrics."""
        custom_metrics = result.get('custom_metrics', {})
        
        # Performance scores
        performance_metrics = [
            'overall_performance_score', 'safety_score', 'efficiency_score',
            'lane_change_success_rate', 'safety_violations_mean'
        ]
        
        for metric in performance_metrics:
            if metric in custom_metrics:
                self._writer.add_scalar(f'Performance/{metric}', custom_metrics[metric], step)
        
        # Processing times
        timing_metrics = [k for k in custom_metrics.keys() if 'processing_time' in k]
        for metric in timing_metrics:
            self._writer.add_scalar(f'Timing/{metric}', custom_metrics[metric], step)
    
    def _log_curriculum_metrics(self, result: dict, step: int):
        """Log curriculum learning progress."""
        custom_metrics = result.get('custom_metrics', {})
        
        if 'curriculum_stage' in custom_metrics:
            self._writer.add_scalar('Curriculum/stage', custom_metrics['curriculum_stage'], step)
        
        if 'scenario_difficulty' in custom_metrics:
            self._writer.add_scalar('Curriculum/difficulty', custom_metrics['scenario_difficulty'], step)
    
    def _log_trajectory_visualization(self, result: dict, step: int):
        """Create and log trajectory visualization."""
        try:
            robot_coordinates = result.get('hist_stats', {}).get('_robot_coordinates', [])
            if robot_coordinates:
                plt.style.use(self.plot_config['style'])
                traj_fig = plot_trajectories(robot_coordinates)
                traj_fig.set_size_inches(self.plot_config['figsize'])
                traj_fig.set_dpi(self.plot_config['dpi'])
                
                self._writer.add_figure("Visualizations/TrainingTrajectories", traj_fig, global_step=step)
                plt.close(traj_fig)
        except Exception as e:
            logger.warning(f"Failed to create trajectory visualization: {e}")
    
    def _log_reward_composition_plot(self, result: dict, step: int):
        """Create and log reward composition visualization."""
        try:
            custom_metrics = result.get('custom_metrics', {})
            
            # Extract reward components
            reward_data = {}
            for metric_name, value in custom_metrics.items():
                if 'reward_mean' in metric_name and value != 0:
                    component_name = metric_name.replace('_reward_mean', '').replace('_', ' ').title()
                    reward_data[component_name] = value
            
            if reward_data:
                plt.style.use(self.plot_config['style'])
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create horizontal bar chart
                components = list(reward_data.keys())
                values = list(reward_data.values())
                colors = sns.color_palette(self.plot_config['color_palette'], len(components))
                
                bars = ax.barh(components, values, color=colors)
                ax.set_xlabel('Reward Value')
                ax.set_title(f'Reward Components (Step {step})')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    ax.text(width + 0.01 if width >= 0 else width - 0.01, 
                           bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
                
                plt.tight_layout()
                self._writer.add_figure("Visualizations/RewardComposition", fig, global_step=step)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create reward composition plot: {e}")
    
    def _log_performance_dashboard(self, result: dict, step: int):
        """Create and log performance dashboard."""
        try:
            custom_metrics = result.get('custom_metrics', {})
            
            # Key performance indicators
            kpis = {
                'Overall Performance': custom_metrics.get('overall_performance_score', 0),
                'Safety Score': custom_metrics.get('safety_score', 0),
                'Efficiency Score': custom_metrics.get('efficiency_score', 0),
                'Lane Following': 1.0 - custom_metrics.get('deviation_centerline', 1.0),
                'Object Avoidance': min(1.0, custom_metrics.get('avg_detection_confidence', 0)),
                'Lane Change Success': custom_metrics.get('lane_change_success_rate', 0)
            }
            
            plt.style.use(self.plot_config['style'])
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # KPI radar chart
            self._create_radar_chart(ax1, kpis, 'Performance Dashboard')
            
            # Episode reward trend (if available)
            episode_rewards = result.get('hist_stats', {}).get('episode_reward', [])
            if episode_rewards:
                ax2.plot(episode_rewards[-100:])  # Last 100 episodes
                ax2.set_title('Recent Episode Rewards')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Reward')
                ax2.grid(True, alpha=0.3)
            
            # Safety metrics
            safety_metrics = {
                'Safety Violations': custom_metrics.get('safety_violations_mean', 0),
                'Collision Risk': custom_metrics.get('collision_risk_step_cnt', 0) / 500,  # Normalize
                'Proximity Penalty': abs(custom_metrics.get('proximity_penalty', 0)) / 10  # Normalize
            }
            ax3.bar(safety_metrics.keys(), safety_metrics.values(), color='red', alpha=0.7)
            ax3.set_title('Safety Metrics')
            ax3.set_ylabel('Normalized Value')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Processing time distribution
            timing_data = []
            timing_labels = []
            for metric_name, value in custom_metrics.items():
                if 'processing_time' in metric_name and 'mean' in metric_name:
                    timing_data.append(value * 1000)  # Convert to ms
                    timing_labels.append(metric_name.replace('processing_time_', '').replace('_mean', ''))
            
            if timing_data:
                ax4.pie(timing_data, labels=timing_labels, autopct='%1.1f%%')
                ax4.set_title('Processing Time Distribution (ms)')
            
            plt.tight_layout()
            self._writer.add_figure("Visualizations/PerformanceDashboard", fig, global_step=step)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to create performance dashboard: {e}")
    
    def _log_histograms(self, result: dict, step: int):
        """Log histogram data to Tensorboard."""
        hist_stats = result.get('hist_stats', {})
        
        for hist_name, hist_data in hist_stats.items():
            if hist_name.startswith('_'):  # Skip internal data
                continue
            
            try:
                if isinstance(hist_data, list) and hist_data:
                    # Convert to numpy array and flatten if needed
                    data = np.array(hist_data)
                    if data.ndim > 1:
                        data = data.flatten()
                    
                    # Remove any non-finite values
                    data = data[np.isfinite(data)]
                    
                    if len(data) > 0:
                        self._writer.add_histogram(f'Histograms/{hist_name}', data, step)
            except Exception as e:
                logger.warning(f"Failed to log histogram {hist_name}: {e}")
    
    def _create_radar_chart(self, ax, data: Dict[str, float], title: str):
        """Create a radar chart for performance metrics."""
        categories = list(data.keys())
        values = list(data.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values for completing the circle
        values += values[:1]
        
        # Plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=12, weight='bold')
        ax.grid(True)
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric for better organization in Tensorboard."""
        if 'reward' in metric_name:
            return 'Rewards'
        elif 'detection' in metric_name or 'avoidance' in metric_name:
            return 'ObjectDetection'
        elif 'lane' in metric_name:
            return 'LaneFollowing'
        elif 'safety' in metric_name or 'collision' in metric_name:
            return 'Safety'
        elif 'performance' in metric_name or 'efficiency' in metric_name:
            return 'Performance'
        elif 'processing_time' in metric_name:
            return 'Timing'
        elif 'curriculum' in metric_name:
            return 'Curriculum'
        else:
            return 'Custom'
    
    def flush(self):
        """Flush the writer."""
        if self._writer is not None:
            self._writer.flush()


class EnhancedWeightsAndBiasesLogger(ray.tune.logger.Logger):
    """Enhanced Weights & Biases logger with comprehensive experiment tracking."""
    
    def __init__(self, config, logdir, trial):
        super().__init__(config, logdir, trial)
        
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases not available. Skipping W&B logging.")
            self.wandb_run = None
            return
        
        self.trial = trial
        self.experiment_tag = trial.experiment_tag
        self.config = config
        
        # Initialize W&B run
        project_name = config.get('env_config', {}).get('wandb', {}).get('project', 'enhanced-duckietown-rl')
        run_name = f"{config['env_config']['experiment_name']}_{trial.experiment_tag}"
        
        self.wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            reinit=True,
            tags=['enhanced-rl', 'multi-objective', 'duckietown']
        )
        
        # Log configuration
        valid_config = self._prepare_config_for_logging(config)
        self.wandb_run.config.update(valid_config, allow_val_change=True)
        
        logger.info(f"Enhanced W&B logger initialized: {project_name}/{run_name}")
    
    def on_result(self, result):
        """Log comprehensive results to Weights & Biases."""
        if self.wandb_run is None:
            return
        
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        
        # Log scalar metrics
        self._log_scalar_metrics(result, step)
        
        # Log histograms
        self._log_histograms(result, step)
        
        # Log visualizations
        self._log_visualizations(result, step)
        
        # Log custom tables
        self._log_performance_table(result, step)
    
    def _log_scalar_metrics(self, result: dict, step: int):
        """Log scalar metrics to W&B."""
        # Prepare metrics for logging
        logged_results = [
            'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'episode_len_mean', 'custom_metrics', 'sampler_perf', 'info', 'perf'
        ]
        
        result_copy = {k: v for k, v in result.items() if k in logged_results}
        flat_result = flatten_dict(result_copy, delimiter="/")
        
        # Add step information
        flat_result['training/timesteps_total'] = step
        flat_result['training/iteration'] = result.get(TRAINING_ITERATION, 0)
        
        self.wandb_run.log(flat_result, step=step, sync=False)
    
    def _log_histograms(self, result: dict, step: int):
        """Log histogram data to W&B."""
        hist_stats = result.get('hist_stats', {})
        
        for key, val in hist_stats.items():
            if key.startswith('_'):  # Skip internal data like robot coordinates
                continue
            
            try:
                if isinstance(val, list) and val:
                    # Convert to numpy array for histogram
                    data = np.array(val)
                    if data.ndim > 1:
                        data = data.flatten()
                    
                    # Remove non-finite values
                    data = data[np.isfinite(data)]
                    
                    if len(data) > 0:
                        self.wandb_run.log({
                            f"Histograms/{key}": wandb.Histogram(data)
                        }, step=step, sync=False)
            except Exception as e:
                logger.warning(f"Unable to log histogram for {key}: {e}")
    
    def _log_visualizations(self, result: dict, step: int):
        """Log visualizations to W&B."""
        try:
            # Trajectory plot
            robot_coordinates = result.get('hist_stats', {}).get('_robot_coordinates', [])
            if robot_coordinates:
                traj_fig = plot_trajectories(robot_coordinates)
                self.wandb_run.log({
                    'Visualizations/Episode_Trajectories': wandb.Image(traj_fig)
                }, step=step, sync=False)
                plt.close(traj_fig)
            
            # Reward composition plot
            custom_metrics = result.get('custom_metrics', {})
            reward_data = {}
            for metric_name, value in custom_metrics.items():
                if 'reward_mean' in metric_name and value != 0:
                    component_name = metric_name.replace('_reward_mean', '')
                    reward_data[component_name] = value
            
            if reward_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                components = list(reward_data.keys())
                values = list(reward_data.values())
                
                bars = ax.bar(components, values, color=sns.color_palette('husl', len(components)))
                ax.set_ylabel('Reward Value')
                ax.set_title(f'Reward Components (Step {step})')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
                
                plt.tight_layout()
                self.wandb_run.log({
                    'Visualizations/Reward_Composition': wandb.Image(fig)
                }, step=step, sync=False)
                plt.close(fig)
                
        except Exception as e:
            logger.warning(f"Failed to create W&B visualizations: {e}")
    
    def _log_performance_table(self, result: dict, step: int):
        """Log performance metrics as a table to W&B."""
        try:
            custom_metrics = result.get('custom_metrics', {})
            
            # Create performance summary table
            performance_data = []
            
            # Key metrics to track
            key_metrics = [
                ('Episode Reward', 'episode_reward_mean', result.get('episode_reward_mean', 0)),
                ('Overall Performance', 'overall_performance_score', custom_metrics.get('overall_performance_score', 0)),
                ('Safety Score', 'safety_score', custom_metrics.get('safety_score', 0)),
                ('Efficiency Score', 'efficiency_score', custom_metrics.get('efficiency_score', 0)),
                ('Lane Following', 'deviation_centerline', 1.0 - custom_metrics.get('deviation_centerline', 1.0)),
                ('Object Detection', 'avg_detection_confidence', custom_metrics.get('avg_detection_confidence', 0)),
                ('Lane Change Success', 'lane_change_success_rate', custom_metrics.get('lane_change_success_rate', 0)),
                ('Safety Violations', 'safety_violations_mean', custom_metrics.get('safety_violations_mean', 0))
            ]
            
            for metric_name, key, value in key_metrics:
                performance_data.append([step, metric_name, key, value])
            
            # Create W&B table
            table = wandb.Table(
                columns=['Step', 'Metric', 'Key', 'Value'],
                data=performance_data
            )
            
            self.wandb_run.log({
                'Performance/Summary_Table': table
            }, step=step, sync=False)
            
        except Exception as e:
            logger.warning(f"Failed to create performance table: {e}")
    
    def _prepare_config_for_logging(self, config: dict) -> dict:
        """Prepare configuration for W&B logging by removing non-serializable items."""
        valid_config = config.copy()
        
        # Remove callbacks (not serializable)
        if 'callbacks' in valid_config:
            del valid_config['callbacks']
        
        # Flatten nested dictionaries
        valid_config = flatten_dict(valid_config, delimiter="/")
        
        return valid_config
    
    def close(self):
        """Close W&B run."""
        if self.wandb_run is not None:
            wandb.join()


def create_enhanced_loggers(config: dict) -> List[ray.tune.logger.Logger]:
    """
    Create list of enhanced loggers based on configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        List of logger classes
    """
    loggers = [
        ray.tune.logger.CSVLogger,
        ray.tune.logger.TBXLogger,
        EnhancedTensorboardLogger
    ]
    
    # Add W&B logger if available and configured
    if WANDB_AVAILABLE:
        wandb_config = config.get('env_config', {}).get('wandb', {})
        if wandb_config.get('enabled', True):
            loggers.append(EnhancedWeightsAndBiasesLogger)
    
    return loggers