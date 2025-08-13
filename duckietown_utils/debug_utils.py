"""
Debug utilities for log analysis and debugging support.

This module provides utilities for:
- Log file parsing and analysis
- Performance profiling
- Debug data extraction
- Automated debugging reports
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import defaultdict, Counter
import logging
import time


class LogAnalyzer:
    """Analyzer for structured log files from the enhanced RL system."""
    
    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
        self.parsed_logs = {}
        self.analysis_results = {}
        
    def parse_log_files(self) -> Dict[str, pd.DataFrame]:
        """Parse all log files in the directory."""
        log_files = {
            'actions': list(self.log_directory.glob('actions_*.jsonl')),
            'detections': list(self.log_directory.glob('detections_*.jsonl')),
            'rewards': list(self.log_directory.glob('rewards_*.jsonl')),
            'performance': list(self.log_directory.glob('performance_*.jsonl')),
            'main': list(self.log_directory.glob('enhanced_rl_*.log'))
        }
        
        for log_type, files in log_files.items():
            if not files:
                continue
                
            if log_type == 'main':
                self.parsed_logs[log_type] = self._parse_main_log(files[0])
            else:
                self.parsed_logs[log_type] = self._parse_jsonl_files(files)
        
        return self.parsed_logs
    
    def _parse_jsonl_files(self, files: List[Path]) -> pd.DataFrame:
        """Parse JSONL files into DataFrame."""
        all_data = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            all_data.append(data)
            except Exception as e:
                logging.warning(f"Failed to parse {file_path}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_data)
    
    def _parse_main_log(self, file_path: Path) -> pd.DataFrame:
        """Parse main log file into structured format."""
        log_entries = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = self._parse_log_line(line.strip())
                        if entry:
                            log_entries.append(entry)
        except Exception as e:
            logging.warning(f"Failed to parse main log {file_path}: {e}")
        
        return pd.DataFrame(log_entries)
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line."""
        # Pattern for structured logs: TIMESTAMP - LEVEL - MESSAGE
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, level, message = match.groups()
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                return {
                    'timestamp': timestamp,
                    'level': level,
                    'message': message
                }
            except ValueError:
                pass
        
        return None
    
    def analyze_detection_performance(self) -> Dict[str, Any]:
        """Analyze object detection performance."""
        if 'detections' not in self.parsed_logs:
            return {}
        
        df = self.parsed_logs['detections']
        if df.empty:
            return {}
        
        # Handle nested detection format - flatten detections
        all_detections = []
        for _, row in df.iterrows():
            if 'detections' in row and isinstance(row['detections'], list):
                for detection in row['detections']:
                    detection_record = {
                        'timestamp': row['timestamp'],
                        'frame_id': row.get('frame_id', 0),
                        'processing_time': row.get('processing_time_ms', row.get('processing_time', 0)),
                        **detection
                    }
                    all_detections.append(detection_record)
        
        if not all_detections:
            # Fallback to original format if no nested detections found
            if 'class' in df.columns:
                all_detections = df.to_dict('records')
            else:
                return {}
        
        # Convert to DataFrame for analysis
        detection_df = pd.DataFrame(all_detections)
        
        analysis = {
            'total_detections': len(detection_df),
            'total_frames': len(df),
            'detection_rate': len(detection_df) / (df['timestamp'].max() - df['timestamp'].min()) if len(df) > 1 else 0,
            'class_distribution': detection_df['class'].value_counts().to_dict() if 'class' in detection_df.columns else {},
            'confidence_stats': {
                'mean': detection_df['confidence'].mean(),
                'std': detection_df['confidence'].std(),
                'min': detection_df['confidence'].min(),
                'max': detection_df['confidence'].max()
            } if 'confidence' in detection_df.columns else None,
            'processing_time_stats': {
                'mean': df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else df['processing_time'].mean() if 'processing_time' in df.columns else 0,
                'std': df['processing_time_ms'].std() if 'processing_time_ms' in df.columns else df['processing_time'].std() if 'processing_time' in df.columns else 0,
                'min': df['processing_time_ms'].min() if 'processing_time_ms' in df.columns else df['processing_time'].min() if 'processing_time' in df.columns else 0,
                'max': df['processing_time_ms'].max() if 'processing_time_ms' in df.columns else df['processing_time'].max() if 'processing_time' in df.columns else 0
            } if 'processing_time_ms' in df.columns or 'processing_time' in df.columns else None
        }
        
        self.analysis_results['detection_performance'] = analysis
        return analysis
    
    def analyze_action_patterns(self) -> Dict[str, Any]:
        """Analyze action decision patterns."""
        if 'actions' not in self.parsed_logs:
            return {}
        
        df = self.parsed_logs['actions']
        if df.empty:
            return {}
        
        analysis = {
            'total_actions': len(df),
            'action_type_distribution': df['action_type'].value_counts().to_dict(),
            'safety_critical_rate': df['safety_critical'].mean() if 'safety_critical' in df.columns else 0,
            'action_frequency': len(df) / (df['timestamp'].max() - df['timestamp'].min()) if len(df) > 1 else 0
        }
        
        # Analyze action sequences
        action_sequences = []
        for i in range(len(df) - 1):
            current_action = df.iloc[i]['action_type']
            next_action = df.iloc[i + 1]['action_type']
            action_sequences.append((current_action, next_action))
        
        analysis['common_sequences'] = Counter(action_sequences).most_common(10)
        
        self.analysis_results['action_patterns'] = analysis
        return analysis
    
    def analyze_reward_trends(self) -> Dict[str, Any]:
        """Analyze reward component trends."""
        if 'rewards' not in self.parsed_logs:
            return {}
        
        df = self.parsed_logs['rewards']
        if df.empty:
            return {}
        
        # Handle nested reward components format
        component_data = []
        for _, row in df.iterrows():
            if 'reward_components' in row and isinstance(row['reward_components'], dict):
                # Flatten reward components
                record = {
                    'timestamp': row['timestamp'],
                    'total_reward': row['total_reward'],
                    'episode': row.get('episode', row.get('episode_step', 1)),
                    **row['reward_components']
                }
                component_data.append(record)
            else:
                # Use original format
                component_data.append(row.to_dict())
        
        if not component_data:
            return {}
        
        # Convert to DataFrame for analysis
        reward_df = pd.DataFrame(component_data)
        
        # Extract reward components (exclude metadata columns)
        component_columns = [col for col in reward_df.columns 
                           if col not in ['timestamp', 'total_reward', 'episode', 'frame_id', 'episode_step', 'cumulative_reward']]
        
        analysis = {
            'total_episodes': reward_df['episode'].nunique() if 'episode' in reward_df.columns else 1,
            'total_steps': len(reward_df),
            'total_reward_stats': {
                'mean': reward_df['total_reward'].mean(),
                'std': reward_df['total_reward'].std(),
                'min': reward_df['total_reward'].min(),
                'max': reward_df['total_reward'].max()
            },
            'component_stats': {}
        }
        
        for component in component_columns:
            if component in reward_df.columns:
                component_values = reward_df[component]
                analysis['component_stats'][component] = {
                    'mean': component_values.mean(),
                    'std': component_values.std(),
                    'contribution': abs(component_values.mean()) / abs(reward_df['total_reward'].mean()) if reward_df['total_reward'].mean() != 0 else 0
                }
        
        # Analyze reward trends over time
        if 'episode' in reward_df.columns and reward_df['episode'].nunique() > 1:
            episode_rewards = reward_df.groupby('episode')['total_reward'].sum()
            analysis['episode_trend'] = {
                'improving': episode_rewards.corr(pd.Series(range(len(episode_rewards)))) > 0.1,
                'correlation': episode_rewards.corr(pd.Series(range(len(episode_rewards))))
            }
        else:
            # Single episode - analyze step-by-step trend
            if len(reward_df) > 1:
                step_correlation = reward_df['total_reward'].corr(pd.Series(range(len(reward_df))))
                analysis['step_trend'] = {
                    'improving': step_correlation > 0.1,
                    'correlation': step_correlation
                }
        
        self.analysis_results['reward_trends'] = analysis
        return analysis
    
    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze system performance metrics."""
        if 'performance' not in self.parsed_logs:
            return {}
        
        df = self.parsed_logs['performance']
        if df.empty:
            return {}
        
        analysis = {
            'fps_stats': {
                'mean': df['fps'].mean(),
                'std': df['fps'].std(),
                'min': df['fps'].min(),
                'max': df['fps'].max(),
                'below_threshold': (df['fps'] < 10).sum() / len(df)
            } if 'fps' in df.columns else None,
            'detection_time_stats': {
                'mean': df['detection_time'].mean(),
                'std': df['detection_time'].std(),
                'min': df['detection_time'].min(),
                'max': df['detection_time'].max(),
                'above_threshold': (df['detection_time'] > 50).sum() / len(df)
            } if 'detection_time' in df.columns else None,
            'memory_stats': {
                'mean': df['memory_usage'].mean(),
                'std': df['memory_usage'].std(),
                'min': df['memory_usage'].min(),
                'max': df['memory_usage'].max(),
                'above_threshold': (df['memory_usage'] > 2048).sum() / len(df)
            } if 'memory_usage' in df.columns else None
        }
        
        self.analysis_results['performance_metrics'] = analysis
        return analysis
    
    def generate_debug_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive debug report."""
        # Run all analyses
        self.parse_log_files()
        detection_analysis = self.analyze_detection_performance()
        action_analysis = self.analyze_action_patterns()
        reward_analysis = self.analyze_reward_trends()
        performance_analysis = self.analyze_performance_metrics()
        
        # Generate report
        report = []
        report.append("# Enhanced Duckietown RL Debug Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Log Directory: {self.log_directory}")
        report.append("")
        
        # Detection Performance
        if detection_analysis:
            report.append("## Object Detection Performance")
            report.append(f"- Total Detections: {detection_analysis['total_detections']}")
            report.append(f"- Detection Rate: {detection_analysis['detection_rate']:.2f} detections/sec")
            report.append("- Class Distribution:")
            for class_name, count in detection_analysis['class_distribution'].items():
                report.append(f"  - {class_name}: {count}")
            report.append(f"- Average Confidence: {detection_analysis['confidence_stats']['mean']:.3f}")
            if detection_analysis['processing_time_stats']:
                report.append(f"- Average Processing Time: {detection_analysis['processing_time_stats']['mean']:.2f}ms")
            report.append("")
        
        # Action Patterns
        if action_analysis:
            report.append("## Action Decision Patterns")
            report.append(f"- Total Actions: {action_analysis['total_actions']}")
            report.append(f"- Safety Critical Rate: {action_analysis['safety_critical_rate']:.3f}")
            report.append("- Action Type Distribution:")
            for action_type, count in action_analysis['action_type_distribution'].items():
                report.append(f"  - {action_type}: {count}")
            report.append("- Common Action Sequences:")
            for (action1, action2), count in action_analysis['common_sequences'][:5]:
                report.append(f"  - {action1} → {action2}: {count}")
            report.append("")
        
        # Reward Analysis
        if reward_analysis:
            report.append("## Reward Analysis")
            report.append(f"- Total Episodes: {reward_analysis['total_episodes']}")
            report.append(f"- Average Total Reward: {reward_analysis['total_reward_stats']['mean']:.3f}")
            report.append("- Component Contributions:")
            for component, stats in reward_analysis['component_stats'].items():
                report.append(f"  - {component}: {stats['mean']:.3f} ({stats['contribution']:.1%})")
            if 'episode_trend' in reward_analysis:
                trend = "improving" if reward_analysis['episode_trend']['improving'] else "declining"
                report.append(f"- Episode Trend: {trend} (correlation: {reward_analysis['episode_trend']['correlation']:.3f})")
            report.append("")
        
        # Performance Metrics
        if performance_analysis:
            report.append("## Performance Metrics")
            if performance_analysis['fps_stats']:
                fps_stats = performance_analysis['fps_stats']
                report.append(f"- Average FPS: {fps_stats['mean']:.1f} (min: {fps_stats['min']:.1f}, max: {fps_stats['max']:.1f})")
                report.append(f"- Below 10 FPS: {fps_stats['below_threshold']:.1%}")
            
            if performance_analysis['detection_time_stats']:
                det_stats = performance_analysis['detection_time_stats']
                report.append(f"- Average Detection Time: {det_stats['mean']:.1f}ms")
                report.append(f"- Above 50ms: {det_stats['above_threshold']:.1%}")
            
            if performance_analysis['memory_stats']:
                mem_stats = performance_analysis['memory_stats']
                report.append(f"- Average Memory Usage: {mem_stats['mean']:.0f}MB")
                report.append(f"- Above 2GB: {mem_stats['above_threshold']:.1%}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def _generate_recommendations(self) -> List[str]:
        """Generate debugging recommendations based on analysis."""
        recommendations = []
        
        # Performance recommendations
        if 'performance_metrics' in self.analysis_results:
            perf = self.analysis_results['performance_metrics']
            
            if perf and perf.get('fps_stats') and perf['fps_stats'].get('below_threshold', 0) > 0.1:
                recommendations.append("Consider optimizing processing pipeline - FPS frequently below 10")
            
            if perf and perf.get('detection_time_stats') and perf['detection_time_stats'].get('above_threshold', 0) > 0.1:
                recommendations.append("YOLO detection time often exceeds 50ms - consider model optimization")
            
            if perf and perf.get('memory_stats') and perf['memory_stats'].get('above_threshold', 0) > 0.05:
                recommendations.append("Memory usage occasionally exceeds 2GB - monitor for memory leaks")
        
        # Detection recommendations
        if 'detection_performance' in self.analysis_results:
            det = self.analysis_results['detection_performance']
            
            if det and det.get('confidence_stats') and det['confidence_stats'].get('mean', 1.0) < 0.7:
                recommendations.append("Low average detection confidence - consider retraining or adjusting threshold")
            
            if det and det.get('detection_rate', 1.0) < 1.0:
                recommendations.append("Low detection rate - check if objects are being missed")
        
        # Action recommendations
        if 'action_patterns' in self.analysis_results:
            action = self.analysis_results['action_patterns']
            
            if action and action.get('safety_critical_rate', 0) > 0.2:
                recommendations.append("High safety critical action rate - review safety thresholds")
        
        # Reward recommendations
        if 'reward_trends' in self.analysis_results:
            reward = self.analysis_results['reward_trends']
            
            if reward and reward.get('episode_trend', {}).get('correlation', 0) < 0:
                recommendations.append("Reward trend is declining - check training stability")
            elif reward and reward.get('step_trend', {}).get('correlation', 0) < 0:
                recommendations.append("Step-by-step reward trend is declining - check training stability")
        
        if not recommendations:
            recommendations.append("System appears to be performing within expected parameters")
        
        return recommendations
    
    def create_visualization_plots(self, output_dir: Optional[str] = None):
        """Create visualization plots for the analysis."""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = self.log_directory / "analysis_plots"
            output_path.mkdir(exist_ok=True)
        
        # Detection performance plots
        if 'detections' in self.parsed_logs and not self.parsed_logs['detections'].empty:
            self._plot_detection_analysis(output_path)
        
        # Action pattern plots
        if 'actions' in self.parsed_logs and not self.parsed_logs['actions'].empty:
            self._plot_action_analysis(output_path)
        
        # Reward trend plots
        if 'rewards' in self.parsed_logs and not self.parsed_logs['rewards'].empty:
            self._plot_reward_analysis(output_path)
        
        # Performance plots
        if 'performance' in self.parsed_logs and not self.parsed_logs['performance'].empty:
            self._plot_performance_analysis(output_path)
    
    def _plot_detection_analysis(self, output_path: Path):
        """Create detection analysis plots."""
        df = self.parsed_logs['detections']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Object Detection Analysis')
        
        # Flatten detections for plotting
        all_detections = []
        for _, row in df.iterrows():
            if 'detections' in row and isinstance(row['detections'], list):
                for detection in row['detections']:
                    detection_record = {
                        'timestamp': row['timestamp'],
                        'frame_id': row.get('frame_id', 0),
                        'processing_time': row.get('processing_time_ms', row.get('processing_time', 0)),
                        **detection
                    }
                    all_detections.append(detection_record)
        
        if not all_detections and 'class' in df.columns:
            # Fallback to original format
            all_detections = df.to_dict('records')
        
        if all_detections:
            detection_df = pd.DataFrame(all_detections)
            
            # Class distribution
            if 'class' in detection_df.columns:
                class_counts = detection_df['class'].value_counts()
                axes[0, 0].bar(class_counts.index, class_counts.values)
                axes[0, 0].set_title('Detection Class Distribution')
                axes[0, 0].set_ylabel('Count')
                plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Confidence distribution
            if 'confidence' in detection_df.columns:
                axes[0, 1].hist(detection_df['confidence'], bins=20, alpha=0.7)
                axes[0, 1].set_title('Confidence Score Distribution')
                axes[0, 1].set_xlabel('Confidence')
                axes[0, 1].set_ylabel('Frequency')
        
        # Detections over time (use original df for frame-based analysis)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        if 'total_objects' in df.columns:
            axes[1, 0].plot(df['timestamp'], df['total_objects'])
            axes[1, 0].set_title('Objects per Frame Over Time')
            axes[1, 0].set_ylabel('Objects per Frame')
        else:
            detection_counts = df.set_index('timestamp').resample('1S').size()
            axes[1, 0].plot(detection_counts.index, detection_counts.values)
            axes[1, 0].set_title('Frames per Second')
            axes[1, 0].set_ylabel('Frames per Second')
        
        # Processing time (if available)
        if 'processing_time_ms' in df.columns:
            axes[1, 1].plot(df['timestamp'], df['processing_time_ms'])
            axes[1, 1].set_title('Detection Processing Time')
            axes[1, 1].set_ylabel('Time (ms)')
        elif 'processing_time' in df.columns:
            axes[1, 1].plot(df['timestamp'], df['processing_time'])
            axes[1, 1].set_title('Detection Processing Time')
            axes[1, 1].set_ylabel('Time (ms)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Processing time\nnot available',
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Processing Time')
        
        plt.tight_layout()
        plt.savefig(output_path / 'detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_analysis(self, output_path: Path):
        """Create action analysis plots."""
        df = self.parsed_logs['actions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Action Decision Analysis')
        
        # Action type distribution
        action_counts = df['action_type'].value_counts()
        axes[0, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Action Type Distribution')
        
        # Safety critical actions over time
        if 'safety_critical' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            safety_over_time = df.set_index('timestamp')['safety_critical'].resample('10S').mean()
            axes[0, 1].plot(safety_over_time.index, safety_over_time.values)
            axes[0, 1].set_title('Safety Critical Action Rate')
            axes[0, 1].set_ylabel('Rate')
        
        # Action values distribution (if available)
        if 'action_values' in df.columns:
            # Assuming action_values is a list/array
            try:
                action_values = np.array([eval(av) if isinstance(av, str) else av 
                                        for av in df['action_values']])
                if action_values.ndim == 2:
                    for i in range(action_values.shape[1]):
                        axes[1, 0].hist(action_values[:, i], alpha=0.5, label=f'Action {i}')
                    axes[1, 0].set_title('Action Values Distribution')
                    axes[1, 0].legend()
            except:
                axes[1, 0].text(0.5, 0.5, 'Action values\nnot plottable',
                              ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Action frequency over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        action_freq = df.set_index('timestamp').resample('1S').size()
        axes[1, 1].plot(action_freq.index, action_freq.values)
        axes[1, 1].set_title('Action Frequency Over Time')
        axes[1, 1].set_ylabel('Actions per Second')
        
        plt.tight_layout()
        plt.savefig(output_path / 'action_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reward_analysis(self, output_path: Path):
        """Create reward analysis plots."""
        df = self.parsed_logs['rewards']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Reward Component Analysis')
        
        # Handle nested reward components format
        component_data = []
        for _, row in df.iterrows():
            if 'reward_components' in row and isinstance(row['reward_components'], dict):
                record = {
                    'timestamp': row['timestamp'],
                    'total_reward': row['total_reward'],
                    'episode': row.get('episode', row.get('episode_step', 1)),
                    **row['reward_components']
                }
                component_data.append(record)
            else:
                component_data.append(row.to_dict())
        
        if component_data:
            reward_df = pd.DataFrame(component_data)
            reward_df['timestamp'] = pd.to_datetime(reward_df['timestamp'], unit='s')
            
            # Total reward over time
            axes[0, 0].plot(reward_df['timestamp'], reward_df['total_reward'])
            axes[0, 0].set_title('Total Reward Over Time')
            axes[0, 0].set_ylabel('Reward')
            
            # Component contributions
            component_cols = [col for col in reward_df.columns 
                             if col not in ['timestamp', 'total_reward', 'episode', 'frame_id', 'episode_step', 'cumulative_reward']]
            if component_cols:
                component_means = reward_df[component_cols].mean()
                axes[0, 1].bar(component_means.index, component_means.values)
                axes[0, 1].set_title('Average Component Rewards')
                axes[0, 1].set_ylabel('Average Reward')
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # Episode/step rewards
            if 'episode' in reward_df.columns and reward_df['episode'].nunique() > 1:
                episode_rewards = reward_df.groupby('episode')['total_reward'].sum()
                axes[1, 0].plot(episode_rewards.index, episode_rewards.values)
                axes[1, 0].set_title('Episode Total Rewards')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Total Reward')
            else:
                # Plot step-by-step rewards
                axes[1, 0].plot(range(len(reward_df)), reward_df['total_reward'])
                axes[1, 0].set_title('Step-by-Step Rewards')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Reward')
            
            # Reward distribution
            axes[1, 1].hist(reward_df['total_reward'], bins=30, alpha=0.7)
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path / 'reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_analysis(self, output_path: Path):
        """Create performance analysis plots."""
        df = self.parsed_logs['performance']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Metrics Analysis')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # FPS over time
        if 'fps' in df.columns:
            axes[0, 0].plot(df['timestamp'], df['fps'])
            axes[0, 0].axhline(y=10, color='r', linestyle='--', label='Min Target')
            axes[0, 0].set_title('FPS Over Time')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].legend()
        
        # Detection time
        if 'detection_time' in df.columns:
            axes[0, 1].plot(df['timestamp'], df['detection_time'])
            axes[0, 1].axhline(y=50, color='r', linestyle='--', label='Max Target')
            axes[0, 1].set_title('Detection Time Over Time')
            axes[0, 1].set_ylabel('Time (ms)')
            axes[0, 1].legend()
        
        # Memory usage
        if 'memory_usage' in df.columns:
            axes[1, 0].plot(df['timestamp'], df['memory_usage'])
            axes[1, 0].axhline(y=2048, color='r', linestyle='--', label='Max Target')
            axes[1, 0].set_title('Memory Usage Over Time')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].legend()
        
        # Performance summary
        perf_metrics = ['fps', 'detection_time', 'memory_usage']
        available_metrics = [m for m in perf_metrics if m in df.columns]
        
        if available_metrics:
            perf_data = df[available_metrics].describe()
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=perf_data.round(2).values,
                                   rowLabels=perf_data.index,
                                   colLabels=perf_data.columns,
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            axes[1, 1].set_title('Performance Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


class DebugProfiler:
    """Performance profiler for debugging bottlenecks."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.active_timers = {}
        
    def start_timer(self, name: str):
        """Start timing a code section."""
        self.active_timers[name] = time.time()
        
    def end_timer(self, name: str):
        """End timing and record duration."""
        if name in self.active_timers:
            duration = time.time() - self.active_timers[name]
            self.timings[name].append(duration * 1000)  # Convert to ms
            del self.active_timers[name]
            return duration
        return None
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'count': len(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times)
                }
        return stats
    
    def print_stats(self):
        """Print timing statistics."""
        stats = self.get_stats()
        print("\n=== Performance Profile ===")
        print(f"{'Section':<20} {'Count':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Total':<10}")
        print("-" * 80)
        
        for name, stat in stats.items():
            print(f"{name:<20} {stat['count']:<8} {stat['mean']:<10.2f} {stat['std']:<10.2f} "
                  f"{stat['min']:<10.2f} {stat['max']:<10.2f} {stat['total']:<10.2f}")
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.active_timers.clear()


# Context manager for easy profiling
class ProfileSection:
    """Context manager for profiling code sections."""
    
    def __init__(self, profiler: DebugProfiler, name: str):
        self.profiler = profiler
        self.name = name
        
    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.name)


def create_debug_session(log_directory: str) -> Tuple[LogAnalyzer, str]:
    """Create a debug session with analysis and report."""
    analyzer = LogAnalyzer(log_directory)
    report = analyzer.generate_debug_report()
    analyzer.create_visualization_plots()
    
    return analyzer, report