#!/usr/bin/env python3
"""
🔍 FAILURE ANALYZER EXAMPLE 🔍
Example usage of the FailureAnalyzer for comprehensive failure analysis

This example demonstrates how to use the FailureAnalyzer to analyze episode failures,
generate statistics, create spatial heatmaps, and produce comprehensive reports.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.failure_analyzer import (
    FailureAnalyzer, FailureAnalysisConfig, StateTrace
)
from duckietown_utils.suite_manager import EpisodeResult
from duckietown_utils.enhanced_logger import EnhancedLogger

def create_sample_episode_data():
    """Create sample episode data for demonstration."""
    
    # Create sample episode results
    episode_results = []
    state_traces = []
    
    for episode_idx in range(5):
        # Create episode result
        success = episode_idx > 1  # First two episodes fail
        collision = episode_idx == 0
        off_lane = episode_idx == 1
        
        episode_result = EpisodeResult(
            episode_id=f"demo_episode_{episode_idx:03d}",
            map_name="loop_empty",
            seed=42 + episode_idx,
            success=success,
            reward=0.8 if success else 0.3,
            episode_length=300 if success else 150,
            lateral_deviation=0.1 if success else 0.4,
            heading_error=2.0 if success else 8.0,
            jerk=0.05 if success else 0.15,
            stability=5.0 if success else 2.0,
            collision=collision,
            off_lane=off_lane,
            violations={},
            lap_time=30.0 if success else 15.0,
            metadata={
                "model_id": "demo_model_v1",
                "suite": "base",
                "mode": "deterministic"
            },
            timestamp=datetime.now().isoformat()
        )
        episode_results.append(episode_result)
        
        # Create corresponding state trace
        state_trace = create_sample_state_trace(episode_idx, success)
        state_traces.append(state_trace)
    
    return episode_results, state_traces

def create_sample_state_trace(episode_idx: int, success: bool):
    """Create a sample state trace for an episode."""
    
    trace_length = 300 if success else 150
    state_trace = []
    
    for step in range(trace_length):
        t = step * 0.1
        
        # Simulate different behaviors based on episode type
        if episode_idx == 0:  # Collision episode - stuck behavior
            if 50 <= step <= 100:
                velocity = (0.05, 0.0)  # Stuck
                position = (5.0, 0.0, 0.0)  # Static position
            else:
                velocity = (1.0, 0.0)
                position = (step * 0.02, 0.0, 0.0)
        elif episode_idx == 1:  # Off-lane episode - increasing deviation
            velocity = (1.2, 0.0)  # Slightly fast
            lane_pos = min(0.5, step * 0.01)  # Increasing deviation
            position = (step * 0.02, lane_pos, 0.0)
        else:  # Successful episodes
            velocity = (1.0, 0.0)
            lane_pos = 0.1 * np.sin(step * 0.1)  # Normal lane following
            position = (step * 0.02, lane_pos * 0.1, step * 0.01)
        
        # Generate actions
        if episode_idx == 0 and 50 <= step <= 100:
            action = (0.0, 0.0)  # No action when stuck
        elif episode_idx == 2:  # Oscillatory behavior
            action = (0.6 * np.sin(step * 0.5), 0.8)  # High frequency steering
        else:
            action = (0.2 * np.sin(step * 0.1), 0.8)  # Normal steering
        
        # Calculate other values
        lane_position = lane_pos if 'lane_pos' in locals() else 0.1 * np.sin(step * 0.1)
        heading_error = 0.05 * np.sin(step * 0.2)
        reward = 1.0 if success else max(0.0, 1.0 - step * 0.01)
        
        state = StateTrace(
            timestamp=t,
            position=position,
            velocity=velocity,
            lane_position=lane_position,
            heading_error=heading_error,
            action=action,
            reward=reward,
            observations={
                'camera': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                'speed': velocity[0],
                'steering': action[0]
            }
        )
        state_trace.append(state)
    
    return state_trace

def create_sample_video_frames(num_frames: int = 100):
    """Create sample video frames for demonstration."""
    frames = []
    for i in range(num_frames):
        # Create a simple gradient frame with some variation
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 2) % 255  # Red channel varies
        frame[:, :, 1] = 128  # Green constant
        frame[:, :, 2] = 255 - ((i * 2) % 255)  # Blue inverse of red
        frames.append(frame)
    return frames

def demonstrate_failure_analysis():
    """Demonstrate comprehensive failure analysis workflow."""
    
    logger = EnhancedLogger("FailureAnalyzerDemo")
    logger.logger.info("🔍 Starting Failure Analyzer Demonstration")
    
    # Create configuration
    config = FailureAnalysisConfig(
        stuck_threshold=0.1,
        stuck_duration=2.0,
        oscillation_threshold=0.5,
        oscillation_window=10,
        overspeed_threshold=2.0,
        lane_deviation_threshold=0.3,
        record_worst_k=3,
        video_fps=30,
        video_quality="high",
        heatmap_resolution=100,
        min_trace_length=10
    )
    
    # Initialize analyzer
    analyzer = FailureAnalyzer(config)
    logger.logger.info("✅ FailureAnalyzer initialized")
    
    # Create sample data
    logger.logger.info("📊 Creating sample episode data...")
    episode_results, state_traces = create_sample_episode_data()
    
    # Analyze each episode
    logger.logger.info("🔍 Analyzing episodes...")
    episode_traces = []
    
    for i, (episode_result, state_trace) in enumerate(zip(episode_results, state_traces)):
        logger.logger.info(f"Analyzing episode {i+1}/5: {episode_result.episode_id}")
        
        # Create video frames for failed episodes
        video_frames = None
        if not episode_result.success:
            video_frames = create_sample_video_frames(len(state_trace))
        
        # Analyze episode
        episode_trace = analyzer.analyze_episode(
            episode_result, 
            state_trace, 
            video_frames
        )
        episode_traces.append(episode_trace)
        
        # Print episode summary
        logger.logger.info(f"  Success: {episode_trace.success}")
        logger.logger.info(f"  Failures detected: {len(episode_trace.failure_events)}")
        if episode_trace.failure_events:
            failure_types = [f.failure_type.value for f in episode_trace.failure_events]
            logger.logger.info(f"  Failure types: {failure_types}")
        logger.logger.info(f"  Video recorded: {episode_trace.video_path is not None}")
    
    # Generate failure statistics
    logger.logger.info("📈 Generating failure statistics...")
    statistics = analyzer.generate_failure_statistics()
    
    logger.logger.info("📊 Failure Statistics Summary:")
    summary = statistics['summary']
    logger.logger.info(f"  Total episodes: {summary['total_episodes']}")
    logger.logger.info(f"  Failed episodes: {summary['failed_episodes']}")
    logger.logger.info(f"  Success rate: {summary['success_rate']:.1%}")
    logger.logger.info(f"  Total failures: {summary['total_failures']}")
    
    if statistics['failure_types']:
        logger.logger.info("  Failure type distribution:")
        for failure_type, count in statistics['failure_types'].items():
            logger.logger.info(f"    {failure_type}: {count}")
    
    # Generate spatial heatmaps
    logger.logger.info("🗺️ Generating spatial heatmaps...")
    heatmaps = analyzer.generate_spatial_heatmaps("loop_empty")
    
    if heatmaps:
        logger.logger.info(f"Generated {len(heatmaps)} heatmaps:")
        for heatmap_type, path in heatmaps.items():
            logger.logger.info(f"  {heatmap_type}: {path}")
    else:
        logger.logger.info("No heatmaps generated (insufficient data)")
    
    # Generate comprehensive report
    logger.logger.info("📋 Generating comprehensive failure report...")
    report = analyzer.generate_failure_report()
    
    logger.logger.info("📋 Report Summary:")
    metadata = report['report_metadata']
    logger.logger.info(f"  Generated at: {metadata['generation_time']}")
    logger.logger.info(f"  Analyzed models: {metadata['analyzed_models']}")
    logger.logger.info(f"  Analyzed maps: {metadata['analyzed_maps']}")
    
    spatial_analysis = report['spatial_analysis']
    logger.logger.info(f"  Total heatmaps: {spatial_analysis['total_heatmaps']}")
    
    if 'recommendations' in report and report['recommendations']:
        logger.logger.info("🎯 Recommendations:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            logger.logger.info(f"  {i}. {recommendation}")
    
    # Demonstrate episode summaries
    if 'episode_summaries' in report:
        logger.logger.info("📝 Worst Episode Summaries:")
        for summary in report['episode_summaries'][:3]:  # Show top 3
            logger.logger.info(f"  Episode: {summary['episode_id']}")
            logger.logger.info(f"    Success: {summary['success']}")
            logger.logger.info(f"    Failures: {summary['failure_count']}")
            logger.logger.info(f"    Types: {summary['failure_types']}")
            logger.logger.info(f"    Video: {summary['video_available']}")
    
    # Show output directory structure
    logger.logger.info("📁 Output Directory Structure:")
    output_dir = Path("logs/failure_analysis")
    if output_dir.exists():
        for subdir in ['videos', 'heatmaps', 'traces']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob('*')))
                logger.logger.info(f"  {subdir}/: {file_count} files")
    
    logger.logger.info("✅ Failure Analysis Demonstration Complete!")
    logger.logger.info(f"📂 Check output directory: {output_dir}")
    
    return analyzer, report

def demonstrate_advanced_analysis():
    """Demonstrate advanced failure analysis features."""
    
    logger = EnhancedLogger("AdvancedFailureAnalysis")
    logger.logger.info("🚀 Advanced Failure Analysis Features")
    
    # Create analyzer with custom configuration
    config = FailureAnalysisConfig(
        stuck_threshold=0.05,  # More sensitive
        oscillation_threshold=0.3,  # Lower threshold
        record_worst_k=5,  # Record more videos
        heatmap_resolution=150,  # Higher resolution
        confidence_threshold=0.99  # Higher confidence
    )
    
    analyzer = FailureAnalyzer(config)
    
    # Create more complex episode data
    logger.logger.info("Creating complex failure scenarios...")
    
    # Scenario 1: Multiple failure types in one episode
    complex_trace = []
    for step in range(200):
        t = step * 0.1
        
        # Different phases of the episode
        if step < 50:  # Normal driving
            velocity = (1.0, 0.0)
            action = (0.1 * np.sin(step * 0.1), 0.8)
            lane_pos = 0.05 * np.sin(step * 0.1)
        elif step < 100:  # Oscillatory behavior
            velocity = (0.8, 0.0)
            action = (0.8 * np.sin(step * 1.0), 0.6)  # High frequency
            lane_pos = 0.2 * np.sin(step * 0.5)
        elif step < 150:  # Stuck behavior
            velocity = (0.02, 0.0)  # Very slow
            action = (0.0, 0.1)  # Minimal throttle
            lane_pos = 0.3  # Off to the side
        else:  # Recovery attempt with overspeed
            velocity = (2.5, 0.0)  # Too fast
            action = (0.5, 1.0)  # Full throttle
            lane_pos = 0.4  # Still off-lane
        
        state = StateTrace(
            timestamp=t,
            position=(step * 0.02, lane_pos, step * 0.01),
            velocity=velocity,
            lane_position=lane_pos,
            heading_error=0.1 * np.sin(step * 0.3),
            action=action,
            reward=max(0.0, 1.0 - abs(lane_pos) - abs(velocity[0] - 1.0)),
            observations={'complex_scenario': True}
        )
        complex_trace.append(state)
    
    # Create complex episode result
    complex_result = EpisodeResult(
        episode_id="complex_failure_001",
        map_name="complex_track",
        seed=123,
        success=False,
        reward=0.2,
        episode_length=200,
        lateral_deviation=0.35,
        heading_error=15.0,
        jerk=0.25,
        stability=1.2,
        collision=True,
        off_lane=True,
        violations={'overspeed': 1, 'oscillation': 1},
        lap_time=20.0,
        metadata={
            "model_id": "complex_model",
            "suite": "stress",
            "mode": "stochastic"
        },
        timestamp=datetime.now().isoformat()
    )
    
    # Analyze complex episode
    logger.logger.info("Analyzing complex failure episode...")
    complex_episode_trace = analyzer.analyze_episode(
        complex_result, 
        complex_trace,
        create_sample_video_frames(200)
    )
    
    logger.logger.info(f"Complex episode analysis results:")
    logger.logger.info(f"  Total failures detected: {len(complex_episode_trace.failure_events)}")
    
    failure_summary = {}
    for failure in complex_episode_trace.failure_events:
        failure_type = failure.failure_type.value
        failure_summary[failure_type] = failure_summary.get(failure_type, 0) + 1
    
    for failure_type, count in failure_summary.items():
        logger.logger.info(f"    {failure_type}: {count}")
    
    # Analyze action patterns
    action_hist = complex_episode_trace.action_histogram
    logger.logger.info("Action pattern analysis:")
    if 'steering' in action_hist:
        steering_stats = action_hist['steering']['statistics']
        logger.logger.info(f"  Steering - Mean: {steering_stats['mean']:.3f}, Std: {steering_stats['std']:.3f}")
    
    # Analyze performance metrics
    perf_metrics = complex_episode_trace.performance_metrics
    logger.logger.info("Performance metrics:")
    logger.logger.info(f"  Mean lane deviation: {perf_metrics.get('mean_lane_deviation', 0):.3f}m")
    logger.logger.info(f"  Steering smoothness: {perf_metrics.get('steering_smoothness', 0):.3f}")
    logger.logger.info(f"  Mean reward: {perf_metrics.get('mean_reward', 0):.3f}")
    
    logger.logger.info("✅ Advanced Analysis Complete!")

if __name__ == "__main__":
    print("🔍 Failure Analyzer Example")
    print("=" * 50)
    
    try:
        # Run basic demonstration
        analyzer, report = demonstrate_failure_analysis()
        
        print("\n" + "=" * 50)
        
        # Run advanced demonstration
        demonstrate_advanced_analysis()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("📂 Check the logs/failure_analysis directory for generated outputs")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()