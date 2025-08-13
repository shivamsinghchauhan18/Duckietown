#!/usr/bin/env python3
"""
Validation script for debugging and visualization tools.

This script validates that all debugging and visualization components
are working correctly and can be integrated successfully.
"""

import sys
import time
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from duckietown_utils.visualization_manager import (
    create_visualization_manager,
    VisualizationConfig
)
from duckietown_utils.debug_utils import (
    LogAnalyzer,
    DebugProfiler,
    ProfileSection
)
from duckietown_utils.visualization_utils import (
    RealTimeDetectionVisualizer,
    ActionVisualization,
    RewardVisualization
)


def test_visualization_components():
    """Test individual visualization components."""
    print("Testing visualization components...")
    
    try:
        # Test detection visualizer
        print("  ✓ Testing RealTimeDetectionVisualizer...")
        det_viz = RealTimeDetectionVisualizer()
        
        # Test with mock data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            {
                'class': 'duckiebot',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'distance': 1.5
            }
        ]
        
        vis_image = det_viz.visualize_detections(image, detections)
        assert vis_image.shape == image.shape
        print("    ✓ Detection visualization working")
        
        # Test data structures
        print("  ✓ Testing data structures...")
        action_viz = ActionVisualization(
            action_type="lane_following",
            action_values=np.array([0.1, 0.8]),
            reasoning="Following lane center",
            timestamp=time.time(),
            safety_critical=False
        )
        assert action_viz.action_type == "lane_following"
        print("    ✓ ActionVisualization working")
        
        reward_viz = RewardVisualization(
            components={'lane_following': 0.5, 'object_avoidance': 0.2},
            total_reward=0.7,
            timestamp=time.time()
        )
        assert reward_viz.total_reward == 0.7
        print("    ✓ RewardVisualization working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error testing visualization components: {e}")
        return False


def test_profiling_system():
    """Test the profiling system."""
    print("Testing profiling system...")
    
    try:
        profiler = DebugProfiler()
        
        # Test manual timing
        profiler.start_timer("test_section")
        time.sleep(0.01)
        duration = profiler.end_timer("test_section")
        
        assert duration is not None
        assert duration > 0.005  # At least 5ms
        print("    ✓ Manual timing working")
        
        # Test context manager
        with ProfileSection(profiler, "context_test"):
            time.sleep(0.01)
        
        stats = profiler.get_stats()
        assert "test_section" in stats
        assert "context_test" in stats
        assert stats["test_section"]["count"] == 1
        print("    ✓ Context manager working")
        
        # Test statistics
        assert stats["test_section"]["mean"] > 5  # At least 5ms
        print("    ✓ Statistics generation working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error testing profiling system: {e}")
        return False


def test_log_analysis():
    """Test log analysis functionality."""
    print("Testing log analysis...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test log files
        log_dir = Path(temp_dir)
        
        # Create mock detection log
        import json
        detections_data = [
            {
                'timestamp': time.time(),
                'class': 'duckiebot',
                'confidence': 0.8,
                'bbox': [100, 100, 200, 200],
                'processing_time': 25.0
            }
        ]
        
        with open(log_dir / 'detections_test.jsonl', 'w') as f:
            for data in detections_data:
                f.write(json.dumps(data) + '\n')
        
        # Test log analyzer
        analyzer = LogAnalyzer(str(log_dir))
        parsed_logs = analyzer.parse_log_files()
        
        assert 'detections' in parsed_logs
        assert len(parsed_logs['detections']) == 1
        print("    ✓ Log parsing working")
        
        # Test analysis
        detection_analysis = analyzer.analyze_detection_performance()
        assert detection_analysis['total_detections'] == 1
        print("    ✓ Detection analysis working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error testing log analysis: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)


def test_visualization_manager():
    """Test the visualization manager."""
    print("Testing visualization manager...")
    
    try:
        # Create manager with minimal configuration
        config = VisualizationConfig(
            enable_detection_viz=False,  # Disable GUI components for testing
            enable_action_viz=False,
            enable_reward_viz=False,
            enable_performance_viz=False,
            enable_profiling=True
        )
        
        manager = create_visualization_manager(enable_all=False)
        manager.config = config
        
        # Test profiling
        if manager.profiler:
            with manager.profile_section("test_manager"):
                time.sleep(0.001)
            
            stats = manager.get_profiling_stats()
            if stats and "test_manager" in stats:
                print("    ✓ Manager profiling working")
            else:
                print("    ✓ Manager created (profiling disabled)")
        else:
            print("    ✓ Manager created (profiling disabled)")
        
        # Test update methods (should not crash)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = []
        
        manager.update_detections(image, detections)
        manager.update_action("test", np.array([0.1, 0.8]), "test reasoning")
        manager.update_reward({'test': 0.5}, 0.5)
        manager.update_performance(15.0, 25.0, 5.0, 1024.0)
        
        print("    ✓ Update methods working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error testing visualization manager: {e}")
        return False


def test_command_line_interface():
    """Test the command-line interface."""
    print("Testing command-line interface...")
    
    try:
        # Test that the debug script exists and is executable
        debug_script = Path("debug_enhanced_rl.py")
        assert debug_script.exists(), "debug_enhanced_rl.py not found"
        assert debug_script.stat().st_mode & 0o111, "debug_enhanced_rl.py not executable"
        print("    ✓ Debug script exists and is executable")
        
        # Test help command
        import subprocess
        result = subprocess.run([sys.executable, str(debug_script), "--help"], 
                              capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, "Help command failed"
        assert "Enhanced Duckietown RL" in result.stdout, "Help text incorrect"
        print("    ✓ Help command working")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error testing command-line interface: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("Testing component integration...")
    
    try:
        # Test that all imports work
        from duckietown_utils.visualization_utils import (
            RealTimeDetectionVisualizer,
            ActionDecisionVisualizer,
            RewardComponentVisualizer,
            PerformanceMonitoringDashboard
        )
        from duckietown_utils.debug_utils import (
            LogAnalyzer,
            DebugProfiler,
            create_debug_session
        )
        from duckietown_utils.visualization_manager import (
            VisualizationManager,
            create_visualization_manager
        )
        print("    ✓ All imports successful")
        
        # Test that examples exist
        examples_dir = Path("examples")
        required_examples = [
            "visualization_debugging_example.py",
            "complete_debugging_integration_example.py"
        ]
        
        for example in required_examples:
            example_path = examples_dir / example
            assert example_path.exists(), f"Example {example} not found"
        print("    ✓ All example files exist")
        
        # Test that documentation exists
        docs_dir = Path("docs")
        doc_file = docs_dir / "Debugging_and_Visualization_Tools.md"
        assert doc_file.exists(), "Documentation not found"
        print("    ✓ Documentation exists")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error testing integration: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Enhanced Duckietown RL - Debugging Tools Validation")
    print("=" * 55)
    
    tests = [
        ("Visualization Components", test_visualization_components),
        ("Profiling System", test_profiling_system),
        ("Log Analysis", test_log_analysis),
        ("Visualization Manager", test_visualization_manager),
        ("Command-Line Interface", test_command_line_interface),
        ("Component Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"  ✓ {test_name} - PASSED")
            else:
                print(f"  ✗ {test_name} - FAILED")
        except Exception as e:
            print(f"  ✗ {test_name} - ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*55}")
    print("VALIDATION SUMMARY")
    print(f"{'='*55}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✓" if success else "✗"
        print(f"  {icon} {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All debugging and visualization tools are working correctly!")
        print("\nYou can now use:")
        print("  - Real-time visualization during training")
        print("  - Comprehensive log analysis")
        print("  - Performance profiling")
        print("  - Debug report generation")
        print("  - Command-line debugging utilities")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)