#!/usr/bin/env python3
"""
Enhanced Duckietown RL Debugging and Visualization Utility

This script provides a command-line interface for debugging and visualizing
the enhanced Duckietown RL system. It can be used to:
- Analyze log files
- Generate debug reports
- Run real-time visualization
- Profile performance
- Create visualization plots

Usage:
    python debug_enhanced_rl.py analyze logs/enhanced_logging_demo
    python debug_enhanced_rl.py visualize --real-time
    python debug_enhanced_rl.py profile logs/enhanced_logging_demo
    python debug_enhanced_rl.py report logs/enhanced_logging_demo --output debug_report.md
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from duckietown_utils.visualization_manager import (
    create_visualization_manager,
    VisualizationConfig
)
from duckietown_utils.debug_utils import (
    LogAnalyzer,
    create_debug_session,
    DebugProfiler
)


def analyze_logs(log_directory: str, output_dir: str = None):
    """Analyze log files and generate comprehensive analysis."""
    print(f"Analyzing logs in: {log_directory}")
    
    log_path = Path(log_directory)
    if not log_path.exists():
        print(f"Error: Log directory {log_directory} does not exist")
        return False
    
    try:
        analyzer = LogAnalyzer(log_directory)
        
        # Parse all log files
        print("Parsing log files...")
        parsed_logs = analyzer.parse_log_files()
        
        if not parsed_logs:
            print("No log files found or all files are empty")
            return False
        
        print(f"Found log types: {list(parsed_logs.keys())}")
        
        # Run analyses
        print("Running detection performance analysis...")
        detection_analysis = analyzer.analyze_detection_performance()
        
        print("Running action pattern analysis...")
        action_analysis = analyzer.analyze_action_patterns()
        
        print("Running reward trend analysis...")
        reward_analysis = analyzer.analyze_reward_trends()
        
        print("Running performance metrics analysis...")
        performance_analysis = analyzer.analyze_performance_metrics()
        
        # Print summary
        print("\n=== Analysis Summary ===")
        
        if detection_analysis:
            print(f"Detections: {detection_analysis['total_detections']} total")
            print(f"Detection rate: {detection_analysis['detection_rate']:.2f}/sec")
            print(f"Avg confidence: {detection_analysis['confidence_stats']['mean']:.3f}")
        
        if action_analysis:
            print(f"Actions: {action_analysis['total_actions']} total")
            print(f"Safety critical rate: {action_analysis['safety_critical_rate']:.3f}")
        
        if reward_analysis:
            print(f"Episodes: {reward_analysis['total_episodes']}")
            print(f"Avg total reward: {reward_analysis['total_reward_stats']['mean']:.3f}")
        
        if performance_analysis and performance_analysis['fps_stats']:
            fps_stats = performance_analysis['fps_stats']
            print(f"Avg FPS: {fps_stats['mean']:.1f} (min: {fps_stats['min']:.1f})")
        
        # Create visualization plots
        print("\nGenerating visualization plots...")
        if output_dir:
            analyzer.create_visualization_plots(output_dir)
            print(f"Plots saved to: {output_dir}")
        else:
            analyzer.create_visualization_plots()
            print(f"Plots saved to: {log_path}/analysis_plots")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing logs: {e}")
        return False


def generate_report(log_directory: str, output_file: str = None):
    """Generate a comprehensive debug report."""
    print(f"Generating debug report for: {log_directory}")
    
    try:
        analyzer, report = create_debug_session(log_directory)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Debug report saved to: {output_file}")
        else:
            print("\n" + "="*60)
            print(report)
            print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return False


def run_real_time_visualization(duration: int = 60):
    """Run real-time visualization demo."""
    print(f"Starting real-time visualization for {duration} seconds...")
    print("Close visualization windows to stop early.")
    
    try:
        # Import demo function
        from examples.visualization_debugging_example import (
            simulate_detection_data,
            simulate_action_data,
            simulate_reward_data,
            simulate_performance_data
        )
        
        # Create visualization manager
        config = VisualizationConfig(
            enable_detection_viz=True,
            enable_action_viz=True,
            enable_reward_viz=True,
            enable_performance_viz=True,
            enable_profiling=True
        )
        
        viz_manager = create_visualization_manager(enable_all=True)
        viz_manager.start()
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            step += 1
            
            # Simulate data
            image, detections = simulate_detection_data()
            viz_manager.update_detections(image, detections)
            
            if step % 3 == 0:
                action_type, action_values, reasoning, safety_critical = simulate_action_data()
                viz_manager.update_action(action_type, action_values, reasoning, safety_critical)
            
            if step % 5 == 0:
                components, total_reward = simulate_reward_data()
                viz_manager.update_reward(components, total_reward)
            
            fps, detection_time, action_time, memory_usage = simulate_performance_data()
            viz_manager.update_performance(fps, detection_time, action_time, memory_usage)
            
            time.sleep(0.1)
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Visualization running... {elapsed:.1f}s elapsed")
        
        print("Stopping visualization...")
        viz_manager.stop()
        
        # Show profiling results
        stats = viz_manager.get_profiling_stats()
        if stats:
            print("\nProfiling Results:")
            for section, stat in stats.items():
                print(f"  {section}: {stat['mean']:.2f}ms avg ({stat['count']} calls)")
        
        return True
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        return True
    except Exception as e:
        print(f"Error running visualization: {e}")
        return False


def profile_system(log_directory: str = None, duration: int = 30):
    """Profile system performance."""
    print(f"Profiling system performance for {duration} seconds...")
    
    try:
        profiler = DebugProfiler()
        
        if log_directory:
            # Profile log analysis
            print("Profiling log analysis...")
            
            from duckietown_utils.debug_utils import ProfileSection
            
            with ProfileSection(profiler, "log_parsing"):
                analyzer = LogAnalyzer(log_directory)
                analyzer.parse_log_files()
            
            with ProfileSection(profiler, "detection_analysis"):
                analyzer.analyze_detection_performance()
            
            with ProfileSection(profiler, "action_analysis"):
                analyzer.analyze_action_patterns()
            
            with ProfileSection(profiler, "reward_analysis"):
                analyzer.analyze_reward_trends()
            
            with ProfileSection(profiler, "performance_analysis"):
                analyzer.analyze_performance_metrics()
            
            with ProfileSection(profiler, "report_generation"):
                analyzer.generate_debug_report()
        
        else:
            # Profile simulation
            print("Profiling simulation...")
            
            from examples.visualization_debugging_example import (
                simulate_detection_data,
                simulate_action_data,
                simulate_reward_data,
                simulate_performance_data
            )
            
            start_time = time.time()
            
            from duckietown_utils.debug_utils import ProfileSection
            
            while time.time() - start_time < duration:
                with ProfileSection(profiler, "detection_simulation"):
                    simulate_detection_data()
                
                with ProfileSection(profiler, "action_simulation"):
                    simulate_action_data()
                
                with ProfileSection(profiler, "reward_simulation"):
                    simulate_reward_data()
                
                with ProfileSection(profiler, "performance_simulation"):
                    simulate_performance_data()
                
                time.sleep(0.01)
        
        # Print results
        print("\n=== Profiling Results ===")
        profiler.print_stats()
        
        return True
        
    except Exception as e:
        print(f"Error profiling system: {e}")
        return False


def list_available_logs():
    """List available log directories."""
    print("Available log directories:")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("  No logs directory found")
        return
    
    log_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    if not log_dirs:
        print("  No log directories found")
        return
    
    for log_dir in sorted(log_dirs):
        # Check for log files
        log_files = list(log_dir.glob("*.jsonl")) + list(log_dir.glob("*.log"))
        print(f"  {log_dir.name} ({len(log_files)} files)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Duckietown RL Debugging and Visualization Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze logs/enhanced_logging_demo
  %(prog)s visualize --real-time --duration 120
  %(prog)s report logs/enhanced_logging_demo --output debug_report.md
  %(prog)s profile --log-dir logs/enhanced_logging_demo
  %(prog)s list-logs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze log files')
    analyze_parser.add_argument('log_directory', help='Path to log directory')
    analyze_parser.add_argument('--output-dir', help='Output directory for plots')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Run visualization')
    viz_parser.add_argument('--real-time', action='store_true', 
                           help='Run real-time visualization demo')
    viz_parser.add_argument('--duration', type=int, default=60,
                           help='Duration in seconds (default: 60)')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate debug report')
    report_parser.add_argument('log_directory', help='Path to log directory')
    report_parser.add_argument('--output', help='Output file for report')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile system performance')
    profile_parser.add_argument('--log-dir', help='Log directory to profile')
    profile_parser.add_argument('--duration', type=int, default=30,
                               help='Duration in seconds (default: 30)')
    
    # List logs command
    subparsers.add_parser('list-logs', help='List available log directories')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = False
    
    try:
        if args.command == 'analyze':
            success = analyze_logs(args.log_directory, args.output_dir)
        
        elif args.command == 'visualize':
            if args.real_time:
                success = run_real_time_visualization(args.duration)
            else:
                print("Please specify --real-time for visualization")
                success = False
        
        elif args.command == 'report':
            success = generate_report(args.log_directory, args.output)
        
        elif args.command == 'profile':
            success = profile_system(args.log_dir, args.duration)
        
        elif args.command == 'list-logs':
            list_available_logs()
            success = True
        
        else:
            print(f"Unknown command: {args.command}")
            success = False
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        success = True
    except Exception as e:
        print(f"Unexpected error: {e}")
        success = False
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())