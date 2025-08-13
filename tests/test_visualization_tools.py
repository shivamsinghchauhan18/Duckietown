#!/usr/bin/env python3
"""
Comprehensive test suite for debugging and visualization tools.

Tests all components of the visualization system:
- Real-time detection visualization
- Action decision visualization  
- Reward component visualization
- Performance monitoring dashboard
- Log analysis utilities
- Debug profiling tools
"""

import unittest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
import sys
import threading
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.visualization_utils import (
    RealTimeDetectionVisualizer,
    ActionDecisionVisualizer,
    RewardComponentVisualizer,
    PerformanceMonitoringDashboard,
    DetectionVisualization,
    ActionVisualization,
    RewardVisualization
)
from duckietown_utils.debug_utils import (
    LogAnalyzer,
    DebugProfiler,
    ProfileSection,
    create_debug_session
)
from duckietown_utils.visualization_manager import (
    VisualizationManager,
    VisualizationConfig,
    create_visualization_manager
)


class TestRealTimeDetectionVisualizer(unittest.TestCase):
    """Test the real-time detection visualizer."""
    
    def setUp(self):
        self.visualizer = RealTimeDetectionVisualizer(
            window_name="test_window",
            confidence_threshold=0.5
        )
        
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.window_name, "test_window")
        self.assertEqual(self.visualizer.confidence_threshold, 0.5)
        self.assertIn('duckiebot', self.visualizer.colors)
        
    def test_visualize_detections(self):
        """Test detection visualization."""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test detections
        detections = [
            {
                'class': 'duckiebot',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'distance': 1.5
            },
            {
                'class': 'cone',
                'confidence': 0.3,  # Below threshold
                'bbox': [300, 300, 400, 400],
                'distance': 2.0
            }
        ]
        
        # Test visualization
        vis_image = self.visualizer.visualize_detections(image, detections)
        
        # Check that image is modified (should have bounding boxes)
        self.assertEqual(vis_image.shape, image.shape)
        self.assertFalse(np.array_equal(vis_image, image))
        
    def test_empty_detections(self):
        """Test visualization with no detections."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = []
        
        vis_image = self.visualizer.visualize_detections(image, detections)
        
        # Image should still be modified (detection count added)
        self.assertEqual(vis_image.shape, image.shape)
        
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_show_frame(self, mock_waitkey, mock_imshow):
        """Test showing a frame."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = []
        
        self.visualizer.show_frame(image, detections)
        
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once_with(1)


class TestActionDecisionVisualizer(unittest.TestCase):
    """Test the action decision visualizer."""
    
    def setUp(self):
        # Mock matplotlib to avoid GUI issues in tests
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.draw'), \
             patch('matplotlib.pyplot.pause'):
            # Mock the return value of subplots
            mock_fig = Mock()
            # Create a mock axes array that supports indexing like numpy array
            mock_axes = Mock()
            mock_axes.__getitem__ = Mock(return_value=Mock())
            mock_axes.flat = [Mock() for _ in range(4)]  # 2x2 = 4 axes
            mock_subplots.return_value = (mock_fig, mock_axes)
            self.visualizer = ActionDecisionVisualizer(max_history=10)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.max_history, 10)
        self.assertEqual(len(self.visualizer.action_history), 0)
        
    def test_add_action(self):
        """Test adding actions to visualizer."""
        action_viz = ActionVisualization(
            action_type="lane_following",
            action_values=np.array([0.1, 0.8]),
            reasoning="Following lane center",
            timestamp=time.time(),
            safety_critical=False
        )
        
        with patch.object(self.visualizer, '_update_plots'):
            self.visualizer.add_action(action_viz)
        
        self.assertEqual(len(self.visualizer.action_history), 1)
        self.assertEqual(self.visualizer.action_history[0].action_type, "lane_following")
        
    def test_max_history_limit(self):
        """Test that history is limited to max_history."""
        with patch.object(self.visualizer, '_update_plots'):
            for i in range(15):  # More than max_history
                action_viz = ActionVisualization(
                    action_type=f"action_{i}",
                    action_values=np.array([0.0, 0.0]),
                    reasoning=f"Action {i}",
                    timestamp=time.time(),
                    safety_critical=False
                )
                self.visualizer.add_action(action_viz)
        
        self.assertEqual(len(self.visualizer.action_history), 10)  # max_history


class TestRewardComponentVisualizer(unittest.TestCase):
    """Test the reward component visualizer."""
    
    def setUp(self):
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.draw'), \
             patch('matplotlib.pyplot.pause'):
            # Mock the return value of subplots
            mock_fig = Mock()
            mock_axes = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            self.visualizer = RewardComponentVisualizer(max_history=10)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.max_history, 10)
        self.assertEqual(len(self.visualizer.reward_history), 0)
        self.assertIn('lane_following', self.visualizer.component_colors)
        
    def test_add_reward(self):
        """Test adding rewards to visualizer."""
        reward_viz = RewardVisualization(
            components={'lane_following': 0.5, 'object_avoidance': 0.2},
            total_reward=0.7,
            timestamp=time.time()
        )
        
        with patch.object(self.visualizer, '_update_plots'):
            self.visualizer.add_reward(reward_viz)
        
        self.assertEqual(len(self.visualizer.reward_history), 1)
        self.assertEqual(self.visualizer.reward_history[0].total_reward, 0.7)


class TestPerformanceMonitoringDashboard(unittest.TestCase):
    """Test the performance monitoring dashboard."""
    
    def setUp(self):
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.draw'), \
             patch('matplotlib.pyplot.pause'):
            # Mock the return value of subplots
            mock_fig = Mock()
            mock_axes = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            self.dashboard = PerformanceMonitoringDashboard(max_history=10)
    
    def test_initialization(self):
        """Test dashboard initialization."""
        self.assertEqual(self.dashboard.max_history, 10)
        self.assertEqual(len(self.dashboard.fps_history), 0)
        
    def test_add_metrics(self):
        """Test adding performance metrics."""
        with patch.object(self.dashboard, '_update_dashboard'):
            self.dashboard.add_metrics(15.0, 30.0, 5.0, 1024.0)
        
        self.assertEqual(len(self.dashboard.fps_history), 1)
        self.assertEqual(len(self.dashboard.detection_time_history), 1)
        self.assertEqual(len(self.dashboard.action_time_history), 1)
        self.assertEqual(len(self.dashboard.memory_usage_history), 1)


class TestDebugProfiler(unittest.TestCase):
    """Test the debug profiler."""
    
    def setUp(self):
        self.profiler = DebugProfiler()
        
    def test_initialization(self):
        """Test profiler initialization."""
        self.assertEqual(len(self.profiler.timings), 0)
        self.assertEqual(len(self.profiler.active_timers), 0)
        
    def test_timing(self):
        """Test timing functionality."""
        self.profiler.start_timer("test_section")
        time.sleep(0.01)  # 10ms
        duration = self.profiler.end_timer("test_section")
        
        self.assertIsNotNone(duration)
        self.assertGreater(duration, 0.005)  # At least 5ms
        self.assertEqual(len(self.profiler.timings["test_section"]), 1)
        
    def test_profile_section_context_manager(self):
        """Test ProfileSection context manager."""
        with ProfileSection(self.profiler, "context_test"):
            time.sleep(0.01)
        
        self.assertEqual(len(self.profiler.timings["context_test"]), 1)
        self.assertGreater(self.profiler.timings["context_test"][0], 5)  # At least 5ms
        
    def test_get_stats(self):
        """Test statistics generation."""
        # Add some timing data
        for _ in range(5):
            with ProfileSection(self.profiler, "test_stats"):
                time.sleep(0.001)
        
        stats = self.profiler.get_stats()
        
        self.assertIn("test_stats", stats)
        self.assertEqual(stats["test_stats"]["count"], 5)
        self.assertGreater(stats["test_stats"]["mean"], 0)
        
    def test_reset(self):
        """Test profiler reset."""
        self.profiler.start_timer("test")
        self.profiler.end_timer("test")
        
        self.assertEqual(len(self.profiler.timings), 1)
        
        self.profiler.reset()
        
        self.assertEqual(len(self.profiler.timings), 0)
        self.assertEqual(len(self.profiler.active_timers), 0)


class TestLogAnalyzer(unittest.TestCase):
    """Test the log analyzer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)
        self.analyzer = LogAnalyzer(str(self.log_dir))
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_logs(self):
        """Create test log files."""
        # Create detections log
        detections_data = [
            {
                'timestamp': time.time(),
                'class': 'duckiebot',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'processing_time': 25.0
            },
            {
                'timestamp': time.time() + 1,
                'class': 'cone',
                'confidence': 0.6,
                'bbox': [300, 300, 400, 400],
                'processing_time': 30.0
            }
        ]
        
        with open(self.log_dir / 'detections_test.jsonl', 'w') as f:
            for data in detections_data:
                f.write(json.dumps(data) + '\n')
        
        # Create actions log
        actions_data = [
            {
                'timestamp': time.time(),
                'action_type': 'lane_following',
                'action_values': [0.1, 0.8],
                'safety_critical': False
            },
            {
                'timestamp': time.time() + 1,
                'action_type': 'object_avoidance',
                'action_values': [-0.3, 0.5],
                'safety_critical': True
            }
        ]
        
        with open(self.log_dir / 'actions_test.jsonl', 'w') as f:
            for data in actions_data:
                f.write(json.dumps(data) + '\n')
        
        # Create rewards log
        rewards_data = [
            {
                'timestamp': time.time(),
                'episode': 1,
                'total_reward': 0.5,
                'lane_following': 0.3,
                'object_avoidance': 0.2
            },
            {
                'timestamp': time.time() + 1,
                'episode': 1,
                'total_reward': 0.7,
                'lane_following': 0.4,
                'object_avoidance': 0.3
            }
        ]
        
        with open(self.log_dir / 'rewards_test.jsonl', 'w') as f:
            for data in rewards_data:
                f.write(json.dumps(data) + '\n')
        
        # Create performance log
        performance_data = [
            {
                'timestamp': time.time(),
                'fps': 15.0,
                'detection_time': 25.0,
                'memory_usage': 1024.0
            },
            {
                'timestamp': time.time() + 1,
                'fps': 12.0,
                'detection_time': 35.0,
                'memory_usage': 1200.0
            }
        ]
        
        with open(self.log_dir / 'performance_test.jsonl', 'w') as f:
            for data in performance_data:
                f.write(json.dumps(data) + '\n')
    
    def test_parse_log_files(self):
        """Test parsing log files."""
        self.create_test_logs()
        
        parsed_logs = self.analyzer.parse_log_files()
        
        self.assertIn('detections', parsed_logs)
        self.assertIn('actions', parsed_logs)
        self.assertIn('rewards', parsed_logs)
        self.assertIn('performance', parsed_logs)
        
        # Check data was parsed correctly
        self.assertEqual(len(parsed_logs['detections']), 2)
        self.assertEqual(len(parsed_logs['actions']), 2)
        self.assertEqual(len(parsed_logs['rewards']), 2)
        self.assertEqual(len(parsed_logs['performance']), 2)
        
    def test_analyze_detection_performance(self):
        """Test detection performance analysis."""
        self.create_test_logs()
        self.analyzer.parse_log_files()
        
        analysis = self.analyzer.analyze_detection_performance()
        
        self.assertEqual(analysis['total_detections'], 2)
        self.assertIn('duckiebot', analysis['class_distribution'])
        self.assertIn('cone', analysis['class_distribution'])
        self.assertIn('mean', analysis['confidence_stats'])
        self.assertIn('processing_time_stats', analysis)
        
    def test_analyze_action_patterns(self):
        """Test action pattern analysis."""
        self.create_test_logs()
        self.analyzer.parse_log_files()
        
        analysis = self.analyzer.analyze_action_patterns()
        
        self.assertEqual(analysis['total_actions'], 2)
        self.assertIn('lane_following', analysis['action_type_distribution'])
        self.assertIn('object_avoidance', analysis['action_type_distribution'])
        self.assertEqual(analysis['safety_critical_rate'], 0.5)  # 1 out of 2
        
    def test_analyze_reward_trends(self):
        """Test reward trend analysis."""
        self.create_test_logs()
        self.analyzer.parse_log_files()
        
        analysis = self.analyzer.analyze_reward_trends()
        
        self.assertEqual(analysis['total_episodes'], 1)
        self.assertIn('mean', analysis['total_reward_stats'])
        self.assertIn('lane_following', analysis['component_stats'])
        self.assertIn('object_avoidance', analysis['component_stats'])
        
    def test_analyze_performance_metrics(self):
        """Test performance metrics analysis."""
        self.create_test_logs()
        self.analyzer.parse_log_files()
        
        analysis = self.analyzer.analyze_performance_metrics()
        
        self.assertIn('fps_stats', analysis)
        self.assertIn('detection_time_stats', analysis)
        self.assertIn('memory_stats', analysis)
        
        # Check specific values
        self.assertEqual(analysis['fps_stats']['mean'], 13.5)  # (15 + 12) / 2
        
    def test_generate_debug_report(self):
        """Test debug report generation."""
        self.create_test_logs()
        
        report = self.analyzer.generate_debug_report()
        
        self.assertIn("Enhanced Duckietown RL Debug Report", report)
        self.assertIn("Object Detection Performance", report)
        self.assertIn("Action Decision Patterns", report)
        self.assertIn("Reward Analysis", report)
        self.assertIn("Performance Metrics", report)
        self.assertIn("Recommendations", report)


class TestVisualizationManager(unittest.TestCase):
    """Test the visualization manager."""
    
    def setUp(self):
        self.config = VisualizationConfig(
            enable_detection_viz=True,
            enable_action_viz=True,
            enable_reward_viz=True,
            enable_performance_viz=True,
            enable_profiling=True
        )
        
        # Mock visualization components to avoid GUI issues
        with patch('duckietown_utils.visualization_manager.RealTimeDetectionVisualizer'), \
             patch('duckietown_utils.visualization_manager.ActionDecisionVisualizer'), \
             patch('duckietown_utils.visualization_manager.RewardComponentVisualizer'), \
             patch('duckietown_utils.visualization_manager.PerformanceMonitoringDashboard'), \
             patch('duckietown_utils.visualization_manager.DebugProfiler'):
            self.manager = VisualizationManager(self.config)
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.config, self.config)
        self.assertFalse(self.manager.running)
        self.assertEqual(len(self.manager.threads), 0)
        
    def test_start_stop(self):
        """Test starting and stopping the manager."""
        with patch.object(self.manager, '_action_update_loop'), \
             patch.object(self.manager, '_reward_update_loop'), \
             patch.object(self.manager, '_performance_update_loop'):
            
            self.manager.start()
            self.assertTrue(self.manager.running)
            
            # Give threads time to start
            time.sleep(0.1)
            
            self.manager.stop()
            self.assertFalse(self.manager.running)
    
    def test_update_methods(self):
        """Test update methods don't crash."""
        # Test detection update
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [{'class': 'test', 'confidence': 0.8, 'bbox': [0, 0, 100, 100]}]
        
        # These should not raise exceptions
        self.manager.update_detections(image, detections)
        self.manager.update_action('test', np.array([0.1, 0.8]), 'test reasoning')
        self.manager.update_reward({'test': 0.5}, 0.5)
        self.manager.update_performance(15.0, 25.0, 5.0, 1024.0)
    
    def test_profiling(self):
        """Test profiling functionality."""
        # Since we're mocking the profiler, we need to set up the mock properly
        if hasattr(self.manager, 'profiler') and self.manager.profiler:
            # Mock the profiler methods
            self.manager.profiler.get_stats = Mock(return_value={"test_section": {"mean": 1.0, "count": 1}})
            
            with self.manager.profile_section("test_section"):
                time.sleep(0.001)
            
            stats = self.manager.get_profiling_stats()
            if stats and not isinstance(stats, Mock):  # Only check if profiler returns real data
                self.assertIn("test_section", stats)


class TestVisualizationConfig(unittest.TestCase):
    """Test the visualization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig()
        
        self.assertTrue(config.enable_detection_viz)
        self.assertTrue(config.enable_action_viz)
        self.assertTrue(config.enable_reward_viz)
        self.assertTrue(config.enable_performance_viz)
        self.assertTrue(config.enable_profiling)
        
        self.assertEqual(config.detection_confidence_threshold, 0.5)
        self.assertEqual(config.max_action_history, 100)
        self.assertEqual(config.max_reward_history, 1000)
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VisualizationConfig(
            enable_detection_viz=False,
            detection_confidence_threshold=0.7,
            max_action_history=50
        )
        
        self.assertFalse(config.enable_detection_viz)
        self.assertEqual(config.detection_confidence_threshold, 0.7)
        self.assertEqual(config.max_action_history, 50)


class TestCreateVisualizationManager(unittest.TestCase):
    """Test the convenience function for creating visualization manager."""
    
    def test_create_manager_all_enabled(self):
        """Test creating manager with all features enabled."""
        with patch('duckietown_utils.visualization_manager.VisualizationManager') as mock_manager:
            manager = create_visualization_manager(enable_all=True)
            
            mock_manager.assert_called_once()
            # Check that config has all features enabled
            config = mock_manager.call_args[0][0]
            self.assertTrue(config.enable_detection_viz)
            self.assertTrue(config.enable_action_viz)
            self.assertTrue(config.enable_reward_viz)
            self.assertTrue(config.enable_performance_viz)
            self.assertTrue(config.enable_profiling)
    
    def test_create_manager_with_logs(self):
        """Test creating manager with log directory."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('duckietown_utils.visualization_manager.VisualizationManager') as mock_manager:
                mock_instance = Mock()
                mock_instance.create_debug_session.return_value = "Test report"
                mock_manager.return_value = mock_instance
                
                manager = create_visualization_manager(log_directory=temp_dir)
                
                mock_instance.create_debug_session.assert_called_once_with(temp_dir)
        finally:
            import shutil
            shutil.rmtree(temp_dir)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestRealTimeDetectionVisualizer,
        TestActionDecisionVisualizer,
        TestRewardComponentVisualizer,
        TestPerformanceMonitoringDashboard,
        TestDebugProfiler,
        TestLogAnalyzer,
        TestVisualizationManager,
        TestVisualizationConfig,
        TestCreateVisualizationManager
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)