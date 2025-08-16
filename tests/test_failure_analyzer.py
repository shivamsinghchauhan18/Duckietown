#!/usr/bin/env python3
"""
🧪 FAILURE ANALYZER TESTS 🧪
Comprehensive unit tests for the FailureAnalyzer class

This module tests all aspects of failure analysis including failure classification,
episode trace capture, action histogram generation, video recording, and spatial heatmaps.
"""

import os
import sys
import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.failure_analyzer import (
    FailureAnalyzer, FailureType, FailureSeverity, StateTrace, 
    FailureEvent, EpisodeTrace, FailureAnalysisConfig
)
from duckietown_utils.suite_manager import EpisodeResult

def create_test_episode_result(episode_id="test_episode", success=True, collision=False, off_lane=False, **kwargs):
    """Helper function to create test episode results with correct structure."""
    defaults = {
        "episode_id": episode_id,
        "map_name": "test_map",
        "seed": 42,
        "success": success,
        "reward": 0.8 if success else 0.3,
        "episode_length": 300 if success else 150,
        "lateral_deviation": 0.1 if success else 0.4,
        "heading_error": 2.0 if success else 8.0,
        "jerk": 0.05 if success else 0.15,
        "stability": 5.0 if success else 2.0,
        "collision": collision,
        "off_lane": off_lane,
        "violations": {},
        "lap_time": 30.0 if success else 15.0,
        "metadata": {"model_id": "test_model", "suite": "base"},
        "timestamp": "2024-01-01T00:00:00"
    }
    defaults.update(kwargs)
    return EpisodeResult(**defaults)

class TestFailureAnalyzer:
    """Test suite for FailureAnalyzer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FailureAnalysisConfig(
            stuck_threshold=0.1,
            stuck_duration=2.0,
            oscillation_threshold=0.5,
            oscillation_window=10,
            overspeed_threshold=2.0,
            lane_deviation_threshold=0.3,
            record_worst_k=3,
            video_fps=30,
            min_trace_length=5
        )
    
    @pytest.fixture
    def sample_state_trace(self):
        """Create sample state trace for testing."""
        trace = []
        for i in range(20):
            state = StateTrace(
                timestamp=i * 0.1,
                position=(i * 0.1, 0.0, 0.0),
                velocity=(1.0, 0.0),
                lane_position=0.1 * np.sin(i * 0.2),  # Slight oscillation
                heading_error=0.05 * np.sin(i * 0.3),
                action=(0.1 * np.sin(i * 0.4), 0.8),  # Varying steering
                reward=1.0,
                observations={'camera': np.zeros((64, 64, 3))}
            )
            trace.append(state)
        return trace
    
    @pytest.fixture
    def sample_episode_result(self):
        """Create sample episode result."""
        return create_test_episode_result("test_episode_001")
    
    @patch('duckietown_utils.failure_analyzer.Path.mkdir')
    def test_initialization(self, mock_mkdir, config, temp_dir):
        """Test FailureAnalyzer initialization."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            assert analyzer.config == config
            assert analyzer.episode_traces == {}
            assert analyzer.failure_statistics == {}
            assert analyzer.spatial_patterns == {}
            
            # Check that directories were created
            assert mock_mkdir.call_count >= 4  # output, video, heatmap, trace dirs
    
    def test_failure_classification_collision(self, config, sample_state_trace, temp_dir):
        """Test collision failure classification."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Create episode result with collision
            episode_result = create_test_episode_result(
                "test_episode_collision", success=False, collision=True
            )
            
            failures = analyzer._classify_failures(episode_result, sample_state_trace)
            
            # Should detect collision failure
            collision_failures = [f for f in failures if f.failure_type == FailureType.COLLISION_STATIC]
            assert len(collision_failures) == 1
            assert collision_failures[0].severity == FailureSeverity.CRITICAL
    
    def test_failure_classification_off_lane(self, config, temp_dir):
        """Test off-lane failure classification."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Create state trace with off-lane position
            state_trace = [
                StateTrace(
                    timestamp=1.0, position=(1.0, 0.0, 0.0), velocity=(1.0, 0.0),
                    lane_position=-0.5, heading_error=0.0, action=(0.0, 0.8),
                    reward=0.0, observations={}
                )
            ]
            
            episode_result = create_test_episode_result(
                "test_episode_off_lane", success=False, off_lane=True
            )
            
            failures = analyzer._classify_failures(episode_result, state_trace)
            
            # Should detect off-lane failure
            off_lane_failures = [f for f in failures if f.failure_type == FailureType.OFF_LANE_LEFT]
            assert len(off_lane_failures) == 1
            assert off_lane_failures[0].severity == FailureSeverity.HIGH
    
    def test_stuck_behavior_detection(self, config, temp_dir):
        """Test stuck behavior detection."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Create state trace with stuck behavior
            state_trace = []
            for i in range(30):
                velocity = (0.05, 0.0) if 10 <= i <= 25 else (1.0, 0.0)  # Stuck in middle
                state = StateTrace(
                    timestamp=i * 0.1, position=(i * 0.01, 0.0, 0.0),
                    velocity=velocity, lane_position=0.0, heading_error=0.0,
                    action=(0.0, 0.8), reward=1.0, observations={}
                )
                state_trace.append(state)
            
            stuck_events = analyzer._detect_stuck_behavior(state_trace)
            
            # Should detect stuck behavior
            assert len(stuck_events) >= 1
            assert stuck_events[0].failure_type == FailureType.STUCK
            assert stuck_events[0].severity == FailureSeverity.HIGH
    
    def test_oscillation_detection(self, config, temp_dir):
        """Test oscillation detection."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Create state trace with oscillatory steering
            state_trace = []
            for i in range(20):
                steering = 0.8 * np.sin(i * 2.0)  # High frequency oscillation
                state = StateTrace(
                    timestamp=i * 0.1, position=(i * 0.1, 0.0, 0.0),
                    velocity=(1.0, 0.0), lane_position=0.0, heading_error=0.0,
                    action=(steering, 0.8), reward=1.0, observations={}
                )
                state_trace.append(state)
            
            oscillation_events = analyzer._detect_oscillation(state_trace)
            
            # Should detect oscillation
            assert len(oscillation_events) > 0
            assert oscillation_events[0].failure_type == FailureType.OSCILLATION
            assert oscillation_events[0].severity == FailureSeverity.MEDIUM
    
    def test_overspeed_detection(self, config, temp_dir):
        """Test overspeed detection."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Create state trace with overspeed
            state_trace = [
                StateTrace(
                    timestamp=1.0, position=(1.0, 0.0, 0.0),
                    velocity=(3.0, 0.0),  # Above threshold
                    lane_position=0.0, heading_error=0.0,
                    action=(0.0, 1.0), reward=1.0, observations={}
                )
            ]
            
            overspeed_events = analyzer._detect_overspeed(state_trace)
            
            # Should detect overspeed
            assert len(overspeed_events) == 1
            assert overspeed_events[0].failure_type == FailureType.OVER_SPEED
            assert overspeed_events[0].severity == FailureSeverity.MEDIUM
    
    def test_action_histogram_generation(self, config, sample_state_trace, temp_dir):
        """Test action histogram generation."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            histogram = analyzer._generate_action_histogram(sample_state_trace)
            
            # Check histogram structure
            assert 'steering' in histogram
            assert 'throttle' in histogram
            assert 'total_actions' in histogram
            
            # Check steering histogram
            steering_data = histogram['steering']
            assert 'histogram' in steering_data
            assert 'bins' in steering_data
            assert 'statistics' in steering_data
            assert len(steering_data['histogram']) == 20  # 20 bins
            assert len(steering_data['bins']) == 21  # 21 bin edges
            
            # Check statistics
            stats = steering_data['statistics']
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'median' in stats
            
            assert histogram['total_actions'] == len(sample_state_trace)
    
    def test_lane_deviation_timeline(self, config, sample_state_trace, temp_dir):
        """Test lane deviation timeline extraction."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            timeline = analyzer._extract_lane_deviation_timeline(sample_state_trace)
            
            # Check timeline
            assert len(timeline) == len(sample_state_trace)
            assert all(isinstance(dev, float) for dev in timeline)
            assert all(dev >= 0 for dev in timeline)  # Should be absolute values
    
    def test_performance_metrics_calculation(self, config, sample_state_trace, temp_dir):
        """Test performance metrics calculation."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            metrics = analyzer._calculate_performance_metrics(sample_state_trace)
            
            # Check required metrics
            expected_metrics = [
                'mean_lane_deviation', 'max_lane_deviation', 'std_lane_deviation',
                'mean_heading_error', 'max_heading_error', 'mean_reward',
                'total_reward', 'reward_std', 'steering_smoothness', 'episode_length'
            ]
            
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))
            
            assert metrics['episode_length'] == len(sample_state_trace)
    
    @patch('cv2.VideoWriter')
    def test_video_recording(self, mock_video_writer, config, temp_dir):
        """Test video recording functionality."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Create mock video frames
            video_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
            
            # Mock video writer
            mock_writer_instance = MagicMock()
            mock_video_writer.return_value = mock_writer_instance
            
            video_path = analyzer._record_episode_video("test_episode", video_frames)
            
            # Check that video writer was called
            assert mock_video_writer.called
            assert mock_writer_instance.write.call_count == len(video_frames)
            assert mock_writer_instance.release.called
            assert video_path is not None
    
    def test_should_record_video(self, config, temp_dir):
        """Test video recording decision logic."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Test failed episode
            failed_result = EpisodeResult(
                run_id="test", model_id="test", mode="deterministic", suite="base",
                map_name="test", seed=42, success=False, collision=True, off_lane=False,
                violations={}, reward_mean=0.5, lap_time_s=15.0, deviation_m=0.2,
                heading_deg=5.0, jerk_mean=0.1, stability_mu_over_sigma=2.0,
                episode_len_steps=150, video_path=None, trace_path=None,
                config_hash="hash", env_build="build", timestamp="2024-01-01T00:00:00"
            )
            
            assert analyzer._should_record_video(failed_result, [])
            
            # Test successful episode with critical failures
            success_result = EpisodeResult(
                run_id="test", model_id="test", mode="deterministic", suite="base",
                map_name="test", seed=42, success=True, collision=False, off_lane=False,
                violations={}, reward_mean=0.8, lap_time_s=30.0, deviation_m=0.1,
                heading_deg=2.0, jerk_mean=0.05, stability_mu_over_sigma=5.0,
                episode_len_steps=300, video_path=None, trace_path=None,
                config_hash="hash", env_build="build", timestamp="2024-01-01T00:00:00"
            )
            
            critical_failure = FailureEvent(
                failure_type=FailureType.COLLISION_STATIC,
                severity=FailureSeverity.CRITICAL,
                timestamp=1.0,
                position=(1.0, 0.0, 0.0),
                description="Test collision"
            )
            
            assert analyzer._should_record_video(success_result, [critical_failure])
            
            # Test successful episode with no significant failures
            assert not analyzer._should_record_video(success_result, [])
    
    def test_episode_analysis_integration(self, config, sample_episode_result, sample_state_trace, temp_dir):
        """Test complete episode analysis integration."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Mock file operations
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = MagicMock()
                
                episode_trace = analyzer.analyze_episode(
                    sample_episode_result, 
                    sample_state_trace
                )
                
                # Check episode trace structure
                assert episode_trace.episode_id is not None
                assert episode_trace.model_id == sample_episode_result.model_id
                assert episode_trace.map_name == sample_episode_result.map_name
                assert episode_trace.success == sample_episode_result.success
                assert isinstance(episode_trace.failure_events, list)
                assert isinstance(episode_trace.action_histogram, dict)
                assert isinstance(episode_trace.lane_deviation_timeline, list)
                assert isinstance(episode_trace.performance_metrics, dict)
                
                # Check that episode was stored
                assert episode_trace.episode_id in analyzer.episode_traces
    
    def test_failure_statistics_generation(self, config, temp_dir):
        """Test failure statistics generation."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Add some mock episode traces
            for i in range(3):
                episode_trace = EpisodeTrace(
                    episode_id=f"episode_{i}",
                    model_id="test_model",
                    map_name="test_map",
                    suite="base",
                    seed=i,
                    success=i > 0,  # First episode fails
                    failure_events=[
                        FailureEvent(
                            failure_type=FailureType.COLLISION_STATIC,
                            severity=FailureSeverity.CRITICAL,
                            timestamp=1.0,
                            position=(1.0, 0.0, 0.0),
                            description="Test failure"
                        )
                    ] if i == 0 else [],
                    state_trace=[],
                    action_histogram={},
                    lane_deviation_timeline=[],
                    performance_metrics={}
                )
                analyzer.episode_traces[episode_trace.episode_id] = episode_trace
            
            # Mock file operations
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = MagicMock()
                
                statistics = analyzer.generate_failure_statistics()
                
                # Check statistics structure
                assert 'summary' in statistics
                assert 'failure_types' in statistics
                assert 'failure_by_model' in statistics
                assert 'failure_by_map' in statistics
                assert 'failure_by_suite' in statistics
                assert 'severity_distribution' in statistics
                
                # Check summary
                summary = statistics['summary']
                assert summary['total_episodes'] == 3
                assert summary['failed_episodes'] == 1
                assert summary['success_rate'] == 2/3
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_heatmap_generation(self, mock_close, mock_savefig, config, temp_dir):
        """Test spatial heatmap generation."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Add mock episode trace with state data
            state_trace = [
                StateTrace(
                    timestamp=i * 0.1,
                    position=(i * 0.1, i * 0.05, 0.0),
                    velocity=(1.0, 0.0),
                    lane_position=0.1 * np.sin(i * 0.2),
                    heading_error=0.0,
                    action=(0.0, 0.8),
                    reward=1.0,
                    observations={}
                ) for i in range(20)
            ]
            
            episode_trace = EpisodeTrace(
                episode_id="test_episode",
                model_id="test_model",
                map_name="test_map",
                suite="base",
                seed=42,
                success=False,
                failure_events=[
                    FailureEvent(
                        failure_type=FailureType.COLLISION_STATIC,
                        severity=FailureSeverity.CRITICAL,
                        timestamp=1.0,
                        position=(1.0, 0.5, 0.0),
                        description="Test collision"
                    )
                ],
                state_trace=state_trace,
                action_histogram={},
                lane_deviation_timeline=[],
                performance_metrics={}
            )
            
            analyzer.episode_traces["test_episode"] = episode_trace
            
            heatmaps = analyzer.generate_spatial_heatmaps("test_map")
            
            # Check that heatmaps were generated
            assert isinstance(heatmaps, dict)
            # At least one heatmap should be generated
            assert len(heatmaps) > 0
            
            # Check that matplotlib functions were called
            assert mock_savefig.called
            assert mock_close.called
    
    def test_comprehensive_report_generation(self, config, temp_dir):
        """Test comprehensive failure report generation."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Add mock episode trace
            episode_trace = EpisodeTrace(
                episode_id="test_episode",
                model_id="test_model",
                map_name="test_map",
                suite="base",
                seed=42,
                success=False,
                failure_events=[
                    FailureEvent(
                        failure_type=FailureType.COLLISION_STATIC,
                        severity=FailureSeverity.CRITICAL,
                        timestamp=1.0,
                        position=(1.0, 0.0, 0.0),
                        description="Test failure"
                    )
                ],
                state_trace=[],
                action_histogram={},
                lane_deviation_timeline=[],
                performance_metrics={}
            )
            
            analyzer.episode_traces["test_episode"] = episode_trace
            
            # Mock file operations and matplotlib
            with patch('builtins.open', create=True) as mock_open, \
                 patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'):
                
                mock_open.return_value.__enter__.return_value = MagicMock()
                
                report = analyzer.generate_failure_report()
                
                # Check report structure
                assert 'report_metadata' in report
                assert 'failure_statistics' in report
                assert 'spatial_analysis' in report
                assert 'episode_summaries' in report
                assert 'recommendations' in report
                
                # Check metadata
                metadata = report['report_metadata']
                assert 'generation_time' in metadata
                assert 'analyzer_config' in metadata
                assert 'analyzed_models' in metadata
                assert 'analyzed_maps' in metadata
    
    def test_empty_trace_handling(self, config, temp_dir):
        """Test handling of empty or invalid traces."""
        with patch('duckietown_utils.failure_analyzer.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            analyzer = FailureAnalyzer(config)
            
            # Test empty state trace
            empty_trace = []
            histogram = analyzer._generate_action_histogram(empty_trace)
            assert histogram == {}
            
            timeline = analyzer._extract_lane_deviation_timeline(empty_trace)
            assert timeline == []
            
            metrics = analyzer._calculate_performance_metrics(empty_trace)
            assert metrics == {}
            
            # Test short trace (below minimum length)
            short_trace = [
                StateTrace(
                    timestamp=0.0, position=(0.0, 0.0, 0.0), velocity=(1.0, 0.0),
                    lane_position=0.0, heading_error=0.0, action=(0.0, 0.8),
                    reward=1.0, observations={}
                )
            ]
            
            episode_result = EpisodeResult(
                run_id="test", model_id="test", mode="deterministic", suite="base",
                map_name="test", seed=42, success=True, collision=False, off_lane=False,
                violations={}, reward_mean=0.8, lap_time_s=1.0, deviation_m=0.0,
                heading_deg=0.0, jerk_mean=0.0, stability_mu_over_sigma=1.0,
                episode_len_steps=1, video_path=None, trace_path=None,
                config_hash="hash", env_build="build", timestamp="2024-01-01T00:00:00"
            )
            
            failures = analyzer._classify_failures(episode_result, short_trace)
            assert failures == []  # Should return empty list for short traces

if __name__ == "__main__":
    pytest.main([__file__])