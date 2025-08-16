#!/usr/bin/env python3
"""
🔍 FAILURE ANALYZER 🔍
Comprehensive failure analysis and diagnostic system

This module implements the FailureAnalyzer class with comprehensive failure classification,
episode trace capture and state analysis, action histogram generation, lane deviation tracking,
video recording system for worst-performing episodes, and spatial heatmap generation.
"""

import os
import sys
import numpy as np
import json
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import required components
from duckietown_utils.suite_manager import EpisodeResult, SuiteResults
from duckietown_utils.enhanced_logger import EnhancedLogger

class FailureType(Enum):
    """Types of failures that can occur during episodes."""
    COLLISION_STATIC = "collision_static"
    COLLISION_DYNAMIC = "collision_dynamic"
    OFF_LANE_LEFT = "off_lane_left"
    OFF_LANE_RIGHT = "off_lane_right"
    STUCK = "stuck"
    OSCILLATION = "oscillation"
    OVER_SPEED = "over_speed"
    MISSED_STOP = "missed_stop"
    SENSOR_GLITCH = "sensor_glitch"
    SLIP_OVERSTEER = "slip_oversteer"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class FailureSeverity(Enum):
    """Severity levels for failures."""
    CRITICAL = "critical"  # Safety-critical failures
    HIGH = "high"         # Performance-degrading failures
    MEDIUM = "medium"     # Minor issues
    LOW = "low"          # Negligible issues

@dataclass
class StateTrace:
    """State information captured during episode execution."""
    timestamp: float
    position: Tuple[float, float, float]  # x, y, theta
    velocity: Tuple[float, float]         # linear, angular
    lane_position: float                  # lateral position in lane
    heading_error: float                  # heading deviation from lane
    action: Tuple[float, float]          # steering, throttle
    reward: float
    observations: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class FailureEvent:
    """Information about a specific failure event."""
    failure_type: FailureType
    severity: FailureSeverity
    timestamp: float
    position: Tuple[float, float, float]
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class EpisodeTrace:
    """Complete trace information for an episode."""
    episode_id: str
    model_id: str
    map_name: str
    suite: str
    seed: int
    success: bool
    failure_events: List[FailureEvent]
    state_trace: List[StateTrace]
    action_histogram: Dict[str, Any]
    lane_deviation_timeline: List[float]
    performance_metrics: Dict[str, float]
    video_path: Optional[str] = None
    
@dataclass
class FailureAnalysisConfig:
    """Configuration for failure analysis."""
    # Failure detection thresholds
    stuck_threshold: float = 0.1          # m/s minimum velocity
    stuck_duration: float = 2.0           # seconds
    oscillation_threshold: float = 0.5    # steering change threshold
    oscillation_window: int = 10          # steps to check
    overspeed_threshold: float = 2.0      # m/s maximum velocity
    lane_deviation_threshold: float = 0.3  # meters from lane center
    
    # Video recording settings
    record_worst_k: int = 5
    video_fps: int = 30
    video_quality: str = "high"           # 'low', 'medium', 'high'
    
    # Heatmap generation settings
    heatmap_resolution: int = 100         # pixels per meter
    heatmap_smoothing: float = 0.1        # gaussian smoothing sigma
    
    # Analysis settings
    min_trace_length: int = 10            # minimum steps for analysis
    confidence_threshold: float = 0.95    # for statistical analysis

class FailureAnalyzer:
    """
    Comprehensive failure analysis and diagnostic system.
    
    This class provides detailed failure classification, episode trace analysis,
    action pattern analysis, video recording, and spatial pattern visualization.
    """
    
    def __init__(self, config: Optional[FailureAnalysisConfig] = None):
        """
        Initialize the FailureAnalyzer.
        
        Args:
            config: Configuration for failure analysis
        """
        self.config = config or FailureAnalysisConfig()
        self.logger = EnhancedLogger("FailureAnalyzer")
        
        # Initialize analysis storage
        self.episode_traces: Dict[str, EpisodeTrace] = {}
        self.failure_statistics: Dict[str, Any] = {}
        self.spatial_patterns: Dict[str, Any] = {}
        
        # Create output directories
        self.output_dir = Path("logs/failure_analysis")
        self.video_dir = self.output_dir / "videos"
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.trace_dir = self.output_dir / "traces"
        
        for dir_path in [self.output_dir, self.video_dir, self.heatmap_dir, self.trace_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger.logger.info(f"FailureAnalyzer initialized with output directory: {self.output_dir}")
    
    def analyze_episode(self, episode_result: EpisodeResult, 
                       state_trace: List[StateTrace],
                       video_frames: Optional[List[np.ndarray]] = None) -> EpisodeTrace:
        """
        Analyze a single episode for failures and patterns.
        
        Args:
            episode_result: Episode result data
            state_trace: Complete state trace for the episode
            video_frames: Optional video frames for recording
            
        Returns:
            Complete episode trace with failure analysis
        """
        # Extract model_id from metadata if available, otherwise use episode_id
        model_id = episode_result.metadata.get('model_id', 'unknown_model')
        episode_id = f"{model_id}_{episode_result.map_name}_{episode_result.seed}"
        
        self.logger.logger.info(f"Analyzing episode: {episode_id}")
        
        # Classify failures
        failure_events = self._classify_failures(episode_result, state_trace)
        
        # Generate action histogram
        action_histogram = self._generate_action_histogram(state_trace)
        
        # Extract lane deviation timeline
        lane_deviation_timeline = self._extract_lane_deviation_timeline(state_trace)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(state_trace)
        
        # Record video if this is a worst-performing episode
        video_path = None
        if video_frames and self._should_record_video(episode_result, failure_events):
            video_path = self._record_episode_video(episode_id, video_frames)
        
        # Create episode trace
        episode_trace = EpisodeTrace(
            episode_id=episode_id,
            model_id=model_id,
            map_name=episode_result.map_name,
            suite=episode_result.metadata.get('suite', 'unknown_suite'),
            seed=episode_result.seed,
            success=episode_result.success,
            failure_events=failure_events,
            state_trace=state_trace,
            action_histogram=action_histogram,
            lane_deviation_timeline=lane_deviation_timeline,
            performance_metrics=performance_metrics,
            video_path=video_path
        )
        
        # Store episode trace
        self.episode_traces[episode_id] = episode_trace
        
        # Save trace to disk
        self._save_episode_trace(episode_trace)
        
        self.logger.logger.info(f"Episode analysis complete: {len(failure_events)} failures detected")
        
        return episode_trace
    
    def _classify_failures(self, episode_result: EpisodeResult, 
                          state_trace: List[StateTrace]) -> List[FailureEvent]:
        """
        Classify failures based on episode result and state trace.
        
        Args:
            episode_result: Episode result data
            state_trace: State trace for analysis
            
        Returns:
            List of classified failure events
        """
        failure_events = []
        
        if len(state_trace) < self.config.min_trace_length:
            return failure_events
        
        # Check for collision failures
        if episode_result.collision:
            # Determine if collision was with static or dynamic obstacle
            # This would require additional context from the environment
            failure_type = FailureType.COLLISION_STATIC  # Default assumption
            failure_events.append(FailureEvent(
                failure_type=failure_type,
                severity=FailureSeverity.CRITICAL,
                timestamp=state_trace[-1].timestamp,
                position=state_trace[-1].position,
                description="Collision detected during episode"
            ))
        
        # Check for off-lane failures
        if episode_result.off_lane:
            # Determine which side based on final lane position
            final_lane_pos = state_trace[-1].lane_position
            failure_type = FailureType.OFF_LANE_LEFT if final_lane_pos < 0 else FailureType.OFF_LANE_RIGHT
            failure_events.append(FailureEvent(
                failure_type=failure_type,
                severity=FailureSeverity.HIGH,
                timestamp=state_trace[-1].timestamp,
                position=state_trace[-1].position,
                description=f"Off-lane departure: {final_lane_pos:.3f}m from center"
            ))
        
        # Check for stuck behavior
        stuck_events = self._detect_stuck_behavior(state_trace)
        failure_events.extend(stuck_events)
        
        # Check for oscillation
        oscillation_events = self._detect_oscillation(state_trace)
        failure_events.extend(oscillation_events)
        
        # Check for over-speed violations
        overspeed_events = self._detect_overspeed(state_trace)
        failure_events.extend(overspeed_events)
        
        # Check for sensor glitches (based on observation inconsistencies)
        sensor_events = self._detect_sensor_glitches(state_trace)
        failure_events.extend(sensor_events)
        
        # Check for slip/oversteer events
        slip_events = self._detect_slip_oversteer(state_trace)
        failure_events.extend(slip_events)
        
        return failure_events
    
    def _detect_stuck_behavior(self, state_trace: List[StateTrace]) -> List[FailureEvent]:
        """Detect stuck behavior based on velocity thresholds."""
        failure_events = []
        stuck_start = None
        
        for i, state in enumerate(state_trace):
            velocity_magnitude = np.sqrt(state.velocity[0]**2 + state.velocity[1]**2)
            
            if velocity_magnitude < self.config.stuck_threshold:
                if stuck_start is None:
                    stuck_start = i
            else:
                if stuck_start is not None:
                    stuck_duration = state_trace[i-1].timestamp - state_trace[stuck_start].timestamp
                    if stuck_duration >= self.config.stuck_duration:
                        failure_events.append(FailureEvent(
                            failure_type=FailureType.STUCK,
                            severity=FailureSeverity.HIGH,
                            timestamp=state_trace[stuck_start].timestamp,
                            position=state_trace[stuck_start].position,
                            description=f"Stuck for {stuck_duration:.2f}s",
                            context={"duration": stuck_duration, "start_index": stuck_start}
                        ))
                    stuck_start = None
        
        return failure_events
    
    def _detect_oscillation(self, state_trace: List[StateTrace]) -> List[FailureEvent]:
        """Detect oscillatory behavior in steering actions."""
        failure_events = []
        
        if len(state_trace) < self.config.oscillation_window:
            return failure_events
        
        for i in range(self.config.oscillation_window, len(state_trace)):
            window = state_trace[i-self.config.oscillation_window:i]
            steering_actions = [state.action[0] for state in window]
            
            # Calculate steering changes
            steering_changes = np.abs(np.diff(steering_actions))
            mean_change = np.mean(steering_changes)
            
            if mean_change > self.config.oscillation_threshold:
                failure_events.append(FailureEvent(
                    failure_type=FailureType.OSCILLATION,
                    severity=FailureSeverity.MEDIUM,
                    timestamp=state_trace[i].timestamp,
                    position=state_trace[i].position,
                    description=f"Oscillatory steering: {mean_change:.3f} mean change",
                    context={"mean_steering_change": mean_change, "window_size": self.config.oscillation_window}
                ))
        
        return failure_events
    
    def _detect_overspeed(self, state_trace: List[StateTrace]) -> List[FailureEvent]:
        """Detect over-speed violations."""
        failure_events = []
        
        for state in state_trace:
            velocity_magnitude = np.sqrt(state.velocity[0]**2 + state.velocity[1]**2)
            
            if velocity_magnitude > self.config.overspeed_threshold:
                failure_events.append(FailureEvent(
                    failure_type=FailureType.OVER_SPEED,
                    severity=FailureSeverity.MEDIUM,
                    timestamp=state.timestamp,
                    position=state.position,
                    description=f"Over-speed: {velocity_magnitude:.2f} m/s",
                    context={"velocity": velocity_magnitude}
                ))
        
        return failure_events
    
    def _detect_sensor_glitches(self, state_trace: List[StateTrace]) -> List[FailureEvent]:
        """Detect sensor glitches based on observation inconsistencies."""
        failure_events = []
        
        # This would require more sophisticated analysis of observation data
        # For now, we'll implement a basic check for missing or invalid observations
        for state in state_trace:
            if not state.observations or len(state.observations) == 0:
                failure_events.append(FailureEvent(
                    failure_type=FailureType.SENSOR_GLITCH,
                    severity=FailureSeverity.HIGH,
                    timestamp=state.timestamp,
                    position=state.position,
                    description="Missing or empty observations",
                    context={"observation_keys": list(state.observations.keys())}
                ))
        
        return failure_events
    
    def _detect_slip_oversteer(self, state_trace: List[StateTrace]) -> List[FailureEvent]:
        """Detect slip/oversteer events based on motion patterns."""
        failure_events = []
        
        for i in range(1, len(state_trace)):
            prev_state = state_trace[i-1]
            curr_state = state_trace[i]
            
            # Calculate expected vs actual heading change
            steering_action = prev_state.action[0]
            expected_heading_change = steering_action * 0.1  # Simplified model
            actual_heading_change = curr_state.position[2] - prev_state.position[2]
            
            # Normalize angle difference
            heading_diff = np.abs(actual_heading_change - expected_heading_change)
            heading_diff = min(heading_diff, 2*np.pi - heading_diff)
            
            if heading_diff > 0.5:  # Threshold for slip detection
                failure_events.append(FailureEvent(
                    failure_type=FailureType.SLIP_OVERSTEER,
                    severity=FailureSeverity.MEDIUM,
                    timestamp=curr_state.timestamp,
                    position=curr_state.position,
                    description=f"Slip/oversteer detected: {heading_diff:.3f} rad difference",
                    context={"heading_difference": heading_diff, "steering_action": steering_action}
                ))
        
        return failure_events    

    def _generate_action_histogram(self, state_trace: List[StateTrace]) -> Dict[str, Any]:
        """
        Generate action histogram for the episode.
        
        Args:
            state_trace: State trace for analysis
            
        Returns:
            Action histogram data
        """
        if not state_trace:
            return {}
        
        # Extract actions
        steering_actions = [state.action[0] for state in state_trace]
        throttle_actions = [state.action[1] for state in state_trace]
        
        # Create histograms
        steering_hist, steering_bins = np.histogram(steering_actions, bins=20, range=(-1, 1))
        throttle_hist, throttle_bins = np.histogram(throttle_actions, bins=20, range=(-1, 1))
        
        # Calculate statistics
        steering_stats = {
            'mean': np.mean(steering_actions),
            'std': np.std(steering_actions),
            'min': np.min(steering_actions),
            'max': np.max(steering_actions),
            'median': np.median(steering_actions)
        }
        
        throttle_stats = {
            'mean': np.mean(throttle_actions),
            'std': np.std(throttle_actions),
            'min': np.min(throttle_actions),
            'max': np.max(throttle_actions),
            'median': np.median(throttle_actions)
        }
        
        return {
            'steering': {
                'histogram': steering_hist.tolist(),
                'bins': steering_bins.tolist(),
                'statistics': steering_stats
            },
            'throttle': {
                'histogram': throttle_hist.tolist(),
                'bins': throttle_bins.tolist(),
                'statistics': throttle_stats
            },
            'total_actions': len(state_trace)
        }
    
    def _extract_lane_deviation_timeline(self, state_trace: List[StateTrace]) -> List[float]:
        """
        Extract lane deviation timeline from state trace.
        
        Args:
            state_trace: State trace for analysis
            
        Returns:
            List of lane deviation values over time
        """
        return [abs(state.lane_position) for state in state_trace]
    
    def _calculate_performance_metrics(self, state_trace: List[StateTrace]) -> Dict[str, float]:
        """
        Calculate performance metrics from state trace.
        
        Args:
            state_trace: State trace for analysis
            
        Returns:
            Dictionary of performance metrics
        """
        if not state_trace:
            return {}
        
        # Extract data
        lane_positions = [abs(state.lane_position) for state in state_trace]
        heading_errors = [abs(state.heading_error) for state in state_trace]
        rewards = [state.reward for state in state_trace]
        steering_actions = [state.action[0] for state in state_trace]
        
        # Calculate metrics
        metrics = {
            'mean_lane_deviation': np.mean(lane_positions),
            'max_lane_deviation': np.max(lane_positions),
            'std_lane_deviation': np.std(lane_positions),
            'mean_heading_error': np.mean(heading_errors),
            'max_heading_error': np.max(heading_errors),
            'mean_reward': np.mean(rewards),
            'total_reward': np.sum(rewards),
            'reward_std': np.std(rewards),
            'steering_smoothness': np.mean(np.abs(np.diff(steering_actions))),
            'episode_length': len(state_trace)
        }
        
        return metrics
    
    def _should_record_video(self, episode_result: EpisodeResult, 
                           failure_events: List[FailureEvent]) -> bool:
        """
        Determine if this episode should be recorded as a video.
        
        Args:
            episode_result: Episode result data
            failure_events: List of failure events
            
        Returns:
            True if video should be recorded
        """
        # Record if episode failed
        if not episode_result.success:
            return True
        
        # Record if there are critical failures
        critical_failures = [f for f in failure_events if f.severity == FailureSeverity.CRITICAL]
        if critical_failures:
            return True
        
        # Record if there are multiple failures
        if len(failure_events) >= 3:
            return True
        
        return False
    
    def _record_episode_video(self, episode_id: str, 
                            video_frames: List[np.ndarray]) -> str:
        """
        Record episode video from frames.
        
        Args:
            episode_id: Unique episode identifier
            video_frames: List of video frames
            
        Returns:
            Path to recorded video file
        """
        if not video_frames:
            return None
        
        # Create video filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{episode_id}_{timestamp}.mp4"
        video_path = self.video_dir / video_filename
        
        # Get frame dimensions
        height, width = video_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path), 
            fourcc, 
            self.config.video_fps, 
            (width, height)
        )
        
        try:
            # Write frames
            for frame in video_frames:
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                video_writer.write(frame_bgr)
            
            self.logger.logger.info(f"Video recorded: {video_path}")
            return str(video_path)
            
        except Exception as e:
            self.logger.logger.error(f"Error recording video: {e}")
            return None
        finally:
            video_writer.release()
    
    def _save_episode_trace(self, episode_trace: EpisodeTrace):
        """
        Save episode trace to disk.
        
        Args:
            episode_trace: Episode trace to save
        """
        trace_filename = f"{episode_trace.episode_id}_trace.json"
        trace_path = self.trace_dir / trace_filename
        
        try:
            # Convert to serializable format
            serializable_events = []
            for event in episode_trace.failure_events:
                event_dict = asdict(event)
                # Convert enum to string
                event_dict['failure_type'] = event.failure_type.value
                event_dict['severity'] = event.severity.value
                serializable_events.append(event_dict)
            
            trace_data = {
                'episode_id': episode_trace.episode_id,
                'model_id': episode_trace.model_id,
                'map_name': episode_trace.map_name,
                'suite': episode_trace.suite,
                'seed': episode_trace.seed,
                'success': episode_trace.success,
                'failure_events': serializable_events,
                'action_histogram': episode_trace.action_histogram,
                'lane_deviation_timeline': episode_trace.lane_deviation_timeline,
                'performance_metrics': episode_trace.performance_metrics,
                'video_path': episode_trace.video_path,
                'state_trace_length': len(episode_trace.state_trace)
            }
            
            with open(trace_path, 'w') as f:
                json.dump(trace_data, f, indent=2)
                
            self.logger.logger.debug(f"Episode trace saved: {trace_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Error saving episode trace: {e}")
    
    def generate_failure_statistics(self, model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive failure statistics across episodes.
        
        Args:
            model_ids: Optional list of model IDs to analyze
            
        Returns:
            Dictionary of failure statistics
        """
        # Filter episodes by model IDs if specified
        episodes = self.episode_traces.values()
        if model_ids:
            episodes = [ep for ep in episodes if ep.model_id in model_ids]
        
        if not episodes:
            return {}
        
        # Collect failure data
        failure_counts = Counter()
        failure_by_model = defaultdict(Counter)
        failure_by_map = defaultdict(Counter)
        failure_by_suite = defaultdict(Counter)
        severity_counts = Counter()
        
        for episode in episodes:
            for failure in episode.failure_events:
                failure_counts[failure.failure_type.value] += 1
                failure_by_model[episode.model_id][failure.failure_type.value] += 1
                failure_by_map[episode.map_name][failure.failure_type.value] += 1
                failure_by_suite[episode.suite][failure.failure_type.value] += 1
                severity_counts[failure.severity.value] += 1
        
        # Calculate statistics
        total_episodes = len(episodes)
        failed_episodes = len([ep for ep in episodes if not ep.success])
        success_rate = (total_episodes - failed_episodes) / total_episodes if total_episodes > 0 else 0
        
        statistics = {
            'summary': {
                'total_episodes': total_episodes,
                'failed_episodes': failed_episodes,
                'success_rate': success_rate,
                'total_failures': sum(failure_counts.values())
            },
            'failure_types': dict(failure_counts),
            'failure_by_model': {k: dict(v) for k, v in failure_by_model.items()},
            'failure_by_map': {k: dict(v) for k, v in failure_by_map.items()},
            'failure_by_suite': {k: dict(v) for k, v in failure_by_suite.items()},
            'severity_distribution': dict(severity_counts),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save statistics
        stats_path = self.output_dir / "failure_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        self.failure_statistics = statistics
        self.logger.logger.info(f"Failure statistics generated for {total_episodes} episodes")
        
        return statistics
    
    def generate_spatial_heatmaps(self, map_name: str, 
                                model_ids: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate spatial heatmaps for failure patterns and lane deviations.
        
        Args:
            map_name: Name of the map to analyze
            model_ids: Optional list of model IDs to include
            
        Returns:
            Dictionary mapping heatmap type to file path
        """
        # Filter episodes for the specified map
        episodes = [ep for ep in self.episode_traces.values() if ep.map_name == map_name]
        if model_ids:
            episodes = [ep for ep in episodes if ep.model_id in model_ids]
        
        if not episodes:
            self.logger.logger.warning(f"No episodes found for map: {map_name}")
            return {}
        
        heatmap_paths = {}
        
        # Generate lane deviation heatmap
        deviation_heatmap_path = self._generate_lane_deviation_heatmap(episodes, map_name)
        if deviation_heatmap_path:
            heatmap_paths['lane_deviation'] = deviation_heatmap_path
        
        # Generate failure location heatmap
        failure_heatmap_path = self._generate_failure_location_heatmap(episodes, map_name)
        if failure_heatmap_path:
            heatmap_paths['failure_locations'] = failure_heatmap_path
        
        # Generate contact point heatmap
        contact_heatmap_path = self._generate_contact_point_heatmap(episodes, map_name)
        if contact_heatmap_path:
            heatmap_paths['contact_points'] = contact_heatmap_path
        
        self.logger.logger.info(f"Generated {len(heatmap_paths)} heatmaps for map: {map_name}")
        
        return heatmap_paths
    
    def _generate_lane_deviation_heatmap(self, episodes: List[EpisodeTrace], 
                                       map_name: str) -> Optional[str]:
        """Generate heatmap of lane deviations across the track."""
        try:
            # Collect position and deviation data
            positions = []
            deviations = []
            
            for episode in episodes:
                for i, state in enumerate(episode.state_trace):
                    positions.append((state.position[0], state.position[1]))
                    deviations.append(abs(state.lane_position))
            
            if not positions:
                return None
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract x, y coordinates
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Create 2D histogram with deviation values
            heatmap, xedges, yedges = np.histogram2d(
                x_coords, y_coords, 
                bins=50, 
                weights=deviations
            )
            
            # Normalize by counts to get average deviation
            counts, _, _ = np.histogram2d(x_coords, y_coords, bins=50)
            heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts!=0)
            
            # Plot heatmap
            im = ax.imshow(heatmap.T, origin='lower', 
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap='YlOrRd', aspect='equal')
            
            ax.set_title(f'Lane Deviation Heatmap - {map_name}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average Lane Deviation (m)')
            
            # Save heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lane_deviation_heatmap_{map_name}_{timestamp}.png"
            filepath = self.heatmap_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            self.logger.logger.error(f"Error generating lane deviation heatmap: {e}")
            return None
    
    def _generate_failure_location_heatmap(self, episodes: List[EpisodeTrace], 
                                         map_name: str) -> Optional[str]:
        """Generate heatmap of failure locations."""
        try:
            # Collect failure positions
            failure_positions = []
            failure_types = []
            
            for episode in episodes:
                for failure in episode.failure_events:
                    failure_positions.append((failure.position[0], failure.position[1]))
                    failure_types.append(failure.failure_type.value)
            
            if not failure_positions:
                return None
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract coordinates
            x_coords = [pos[0] for pos in failure_positions]
            y_coords = [pos[1] for pos in failure_positions]
            
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=30)
            
            # Plot heatmap
            im = ax.imshow(heatmap.T, origin='lower',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap='Reds', aspect='equal')
            
            ax.set_title(f'Failure Location Heatmap - {map_name}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Failure Count')
            
            # Save heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failure_location_heatmap_{map_name}_{timestamp}.png"
            filepath = self.heatmap_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            self.logger.logger.error(f"Error generating failure location heatmap: {e}")
            return None
    
    def _generate_contact_point_heatmap(self, episodes: List[EpisodeTrace], 
                                      map_name: str) -> Optional[str]:
        """Generate heatmap of collision contact points."""
        try:
            # Collect collision positions
            collision_positions = []
            
            for episode in episodes:
                for failure in episode.failure_events:
                    if failure.failure_type in [FailureType.COLLISION_STATIC, FailureType.COLLISION_DYNAMIC]:
                        collision_positions.append((failure.position[0], failure.position[1]))
            
            if not collision_positions:
                return None
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract coordinates
            x_coords = [pos[0] for pos in collision_positions]
            y_coords = [pos[1] for pos in collision_positions]
            
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=25)
            
            # Plot heatmap
            im = ax.imshow(heatmap.T, origin='lower',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap='OrRd', aspect='equal')
            
            ax.set_title(f'Collision Contact Points - {map_name}')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Collision Count')
            
            # Save heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contact_point_heatmap_{map_name}_{timestamp}.png"
            filepath = self.heatmap_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            self.logger.logger.error(f"Error generating contact point heatmap: {e}")
            return None
    
    def generate_failure_report(self, model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive failure analysis report.
        
        Args:
            model_ids: Optional list of model IDs to include in report
            
        Returns:
            Comprehensive failure analysis report
        """
        # Generate failure statistics
        statistics = self.generate_failure_statistics(model_ids)
        
        # Get unique maps for heatmap generation
        episodes = self.episode_traces.values()
        if model_ids:
            episodes = [ep for ep in episodes if ep.model_id in model_ids]
        
        unique_maps = list(set(ep.map_name for ep in episodes))
        
        # Generate heatmaps for each map
        heatmaps = {}
        for map_name in unique_maps:
            map_heatmaps = self.generate_spatial_heatmaps(map_name, model_ids)
            if map_heatmaps:
                heatmaps[map_name] = map_heatmaps
        
        # Compile comprehensive report
        report = {
            'report_metadata': {
                'generation_time': datetime.now().isoformat(),
                'analyzer_config': asdict(self.config),
                'analyzed_models': model_ids or list(set(ep.model_id for ep in episodes)),
                'analyzed_maps': unique_maps
            },
            'failure_statistics': statistics,
            'spatial_analysis': {
                'heatmaps_generated': heatmaps,
                'total_heatmaps': sum(len(maps) for maps in heatmaps.values())
            },
            'episode_summaries': self._generate_episode_summaries(model_ids),
            'recommendations': self._generate_recommendations(statistics)
        }
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"failure_analysis_report_{timestamp}.json"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.logger.info(f"Comprehensive failure analysis report generated: {report_path}")
        
        return report
    
    def _generate_episode_summaries(self, model_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate summaries for worst-performing episodes."""
        episodes = list(self.episode_traces.values())
        if model_ids:
            episodes = [ep for ep in episodes if ep.model_id in model_ids]
        
        # Sort by number of failures (descending) and success (failed first)
        episodes.sort(key=lambda ep: (not ep.success, len(ep.failure_events)), reverse=True)
        
        # Take worst episodes
        worst_episodes = episodes[:self.config.record_worst_k]
        
        summaries = []
        for episode in worst_episodes:
            summary = {
                'episode_id': episode.episode_id,
                'model_id': episode.model_id,
                'map_name': episode.map_name,
                'suite': episode.suite,
                'success': episode.success,
                'failure_count': len(episode.failure_events),
                'failure_types': [f.failure_type.value for f in episode.failure_events],
                'performance_metrics': episode.performance_metrics,
                'video_available': episode.video_path is not None,
                'video_path': episode.video_path
            }
            summaries.append(summary)
        
        return summaries
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on failure analysis."""
        recommendations = []
        
        if not statistics or 'failure_types' not in statistics:
            return recommendations
        
        failure_types = statistics['failure_types']
        total_failures = sum(failure_types.values())
        
        if total_failures == 0:
            recommendations.append("No failures detected - model performance appears stable")
            return recommendations
        
        # Analyze most common failures
        sorted_failures = sorted(failure_types.items(), key=lambda x: x[1], reverse=True)
        
        for failure_type, count in sorted_failures[:3]:  # Top 3 failure types
            percentage = (count / total_failures) * 100
            
            if failure_type == 'collision_static':
                recommendations.append(
                    f"High collision rate with static obstacles ({percentage:.1f}% of failures). "
                    "Consider improving object detection sensitivity or avoidance algorithms."
                )
            elif failure_type == 'off_lane_left' or failure_type == 'off_lane_right':
                recommendations.append(
                    f"Frequent lane departures ({percentage:.1f}% of failures). "
                    "Consider tuning lane-following reward weights or improving lane detection."
                )
            elif failure_type == 'stuck':
                recommendations.append(
                    f"Stuck behavior detected ({percentage:.1f}% of failures). "
                    "Consider adding exploration bonuses or improving action space coverage."
                )
            elif failure_type == 'oscillation':
                recommendations.append(
                    f"Oscillatory behavior detected ({percentage:.1f}% of failures). "
                    "Consider adding action smoothing or reducing learning rate."
                )
        
        # Check success rate
        success_rate = statistics.get('summary', {}).get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}). Consider extending training time "
                "or adjusting reward function balance."
            )
        
        return recommendations
    
    def analyze_failures(self, episode_results: List[EpisodeResult]) -> Dict[str, Any]:
        """
        Analyze failure patterns across episode results.
        
        This method provides the expected API interface as documented in the 
        Evaluation Orchestrator API Documentation. It processes episode results
        and returns comprehensive failure analysis.
        
        Args:
            episode_results: List of episode results to analyze
            
        Returns:
            Comprehensive failure analysis including failure types, patterns,
            and statistics
        """
        # Process each episode result
        for episode_result in episode_results:
            self.analyze_episode(episode_result)
        
        # Generate comprehensive failure statistics
        failure_analysis = self.generate_failure_statistics()
        
        # Add additional analysis specific to this method call
        failure_analysis['episode_count'] = len(episode_results)
        failure_analysis['analysis_timestamp'] = datetime.now().isoformat()
        
        return failure_analysis

# Export main classes and functions
__all__ = [
    'FailureAnalyzer',
    'FailureType',
    'FailureSeverity',
    'StateTrace',
    'FailureEvent',
    'EpisodeTrace',
    'FailureAnalysisConfig'
]