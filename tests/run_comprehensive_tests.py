#!/usr/bin/env python3
"""
Comprehensive test runner for the Enhanced Duckietown RL system.
Runs all test suites and generates a comprehensive report.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


class ComprehensiveTestRunner:
    """Runner for comprehensive test suite."""
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(self, test_file, markers=None, timeout=300):
        """Run a specific test suite."""
        print(f"\n{'='*60}")
        print(f"Running {test_file}")
        print(f"{'='*60}")
        
        cmd = ["python", "-m", "pytest", f"tests/{test_file}", "-v", "--tb=short"]
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add output options
        result_file = self.output_dir / f"{test_file.replace('.py', '_results.xml')}"
        cmd.extend(["--junitxml", str(result_file)])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            self.results[test_file] = {
                'success': success,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'markers': markers or [],
                'result_file': str(result_file)
            }
            
            print(f"✅ {test_file} completed in {duration:.2f}s" if success else 
                  f"❌ {test_file} failed in {duration:.2f}s")
            
            if not success:
                print("STDERR:", result.stderr[-500:] if result.stderr else "None")
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} timed out after {timeout}s")
            self.results[test_file] = {
                'success': False,
                'duration': timeout,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test timed out after {timeout}s',
                'markers': markers or [],
                'result_file': str(result_file)
            }
            return False
        
        except Exception as e:
            print(f"💥 {test_file} crashed: {e}")
            self.results[test_file] = {
                'success': False,
                'duration': 0,
                'returncode': -2,
                'stdout': '',
                'stderr': str(e),
                'markers': markers or [],
                'result_file': str(result_file)
            }
            return False
    
    def run_all_tests(self, test_categories=None):
        """Run all comprehensive tests."""
        self.start_time = datetime.now()
        
        # Define test suites
        test_suites = {
            'unit': {
                'file': 'test_comprehensive_unit_tests.py',
                'markers': None,
                'timeout': 180,
                'description': 'Unit tests for all wrapper classes'
            },
            'integration': {
                'file': 'test_integration_pipeline.py',
                'markers': ['integration'],
                'timeout': 300,
                'description': 'Integration tests for complete pipeline'
            },
            'performance': {
                'file': 'test_performance_benchmarks.py',
                'markers': ['performance'],
                'timeout': 600,
                'description': 'Performance benchmarking tests'
            },
            'scenario': {
                'file': 'test_scenario_based_tests.py',
                'markers': ['scenario'],
                'timeout': 400,
                'description': 'Scenario-based obstacle avoidance tests'
            },
            'safety': {
                'file': 'test_safety_validation.py',
                'markers': ['safety'],
                'timeout': 500,
                'description': 'Safety validation tests'
            }
        }
        
        # Filter test categories if specified
        if test_categories:
            test_suites = {k: v for k, v in test_suites.items() if k in test_categories}
        
        print(f"🚀 Starting comprehensive test run at {self.start_time}")
        print(f"📁 Results will be saved to: {self.output_dir}")
        print(f"🧪 Running {len(test_suites)} test suites")
        
        # Run each test suite
        for category, suite_info in test_suites.items():
            print(f"\n📋 {category.upper()}: {suite_info['description']}")
            
            success = self.run_test_suite(
                suite_info['file'],
                suite_info['markers'],
                suite_info['timeout']
            )
            
            if not success:
                print(f"⚠️  {category} tests failed, continuing with other suites...")
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        self.generate_report()
        
        return self.get_overall_success()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate summary statistics
        total_suites = len(self.results)
        successful_suites = sum(1 for r in self.results.values() if r['success'])
        failed_suites = total_suites - successful_suites
        
        # Create summary report
        report = {
            'test_run_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'total_suites': total_suites,
                'successful_suites': successful_suites,
                'failed_suites': failed_suites,
                'success_rate': successful_suites / total_suites if total_suites > 0 else 0
            },
            'suite_results': self.results,
            'recommendations': self.generate_recommendations()
        }
        
        # Save JSON report
        report_file = self.output_dir / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable report
        self.generate_human_readable_report(report, report_file)
        
        print(f"\n📊 Comprehensive report saved to: {report_file}")
    
    def generate_human_readable_report(self, report, report_file):
        """Generate human-readable test report."""
        report_txt = report_file.with_suffix('.txt')
        
        with open(report_txt, 'w') as f:
            f.write("ENHANCED DUCKIETOWN RL - COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            info = report['test_run_info']
            f.write(f"Test Run Summary:\n")
            f.write(f"  Start Time: {info['start_time']}\n")
            f.write(f"  End Time: {info['end_time']}\n")
            f.write(f"  Total Duration: {info['total_duration_seconds']:.2f} seconds\n")
            f.write(f"  Total Suites: {info['total_suites']}\n")
            f.write(f"  Successful: {info['successful_suites']}\n")
            f.write(f"  Failed: {info['failed_suites']}\n")
            f.write(f"  Success Rate: {info['success_rate']:.1%}\n\n")
            
            # Individual suite results
            f.write("Individual Suite Results:\n")
            f.write("-" * 40 + "\n")
            
            for suite_name, result in report['suite_results'].items():
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                f.write(f"{suite_name}: {status} ({result['duration']:.2f}s)\n")
                
                if not result['success']:
                    f.write(f"  Error: {result['stderr'][:200]}...\n")
                f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 20 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"📄 Human-readable report saved to: {report_txt}")
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_suites = [name for name, result in self.results.items() if not result['success']]
        
        if not failed_suites:
            recommendations.append("All test suites passed! The system appears to be working correctly.")
        else:
            recommendations.append(f"{len(failed_suites)} test suite(s) failed. Review the detailed logs.")
            
            # Specific recommendations based on failed suites
            if 'test_comprehensive_unit_tests.py' in failed_suites:
                recommendations.append("Unit tests failed - check individual wrapper implementations.")
            
            if 'test_integration_pipeline.py' in failed_suites:
                recommendations.append("Integration tests failed - check environment setup and wrapper composition.")
            
            if 'test_performance_benchmarks.py' in failed_suites:
                recommendations.append("Performance tests failed - system may not meet real-time requirements.")
            
            if 'test_scenario_based_tests.py' in failed_suites:
                recommendations.append("Scenario tests failed - obstacle avoidance behavior needs improvement.")
            
            if 'test_safety_validation.py' in failed_suites:
                recommendations.append("Safety tests failed - CRITICAL: review safety mechanisms immediately.")
        
        # Performance recommendations
        slow_suites = [name for name, result in self.results.items() 
                      if result['duration'] > 300 and result['success']]
        if slow_suites:
            recommendations.append(f"Slow test suites detected: {', '.join(slow_suites)}. Consider optimization.")
        
        return recommendations
    
    def get_overall_success(self):
        """Get overall success status."""
        return all(result['success'] for result in self.results.values())


def main():
    """Main function for running comprehensive tests."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for Enhanced Duckietown RL')
    parser.add_argument('--categories', nargs='+', 
                       choices=['unit', 'integration', 'performance', 'scenario', 'safety'],
                       help='Test categories to run (default: all)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory for test results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (unit and basic integration)')
    
    args = parser.parse_args()
    
    # Set test categories
    if args.quick:
        test_categories = ['unit', 'integration']
    else:
        test_categories = args.categories
    
    # Run tests
    runner = ComprehensiveTestRunner(args.output_dir)
    success = runner.run_all_tests(test_categories)
    
    # Print final status
    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Review the report and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()