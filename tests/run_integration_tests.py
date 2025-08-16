#!/usr/bin/env python3
"""
🧪 INTEGRATION TEST RUNNER 🧪
Comprehensive test runner for evaluation integration tests

This script runs all integration tests for the evaluation system,
including end-to-end pipeline tests, statistical validation,
performance benchmarking, and reproducibility validation.

Requirements covered: 8.4, 9.1-9.5, 13.3, 13.4
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    """Test result information."""
    test_file: str
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    error_message: Optional[str] = None
    output: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Test suite result summary."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    test_results: List[TestResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests


class IntegrationTestRunner:
    """Runner for integration tests."""
    
    def __init__(self, verbose: bool = False, save_results: bool = True):
        self.verbose = verbose
        self.save_results = save_results
        self.results_dir = Path(__file__).parent / 'integration_test_results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Test suites to run
        self.test_suites = {
            'evaluation_integration': {
                'file': 'test_evaluation_integration.py',
                'description': 'End-to-end evaluation pipeline integration tests',
                'timeout': 300  # 5 minutes
            },
            'statistical_validation': {
                'file': 'test_statistical_validation.py',
                'description': 'Statistical validation and confidence interval tests',
                'timeout': 180  # 3 minutes
            },
            'performance_benchmarking': {
                'file': 'test_performance_benchmarking.py',
                'description': 'Performance and throughput benchmarking tests',
                'timeout': 240  # 4 minutes
            },
            'reproducibility_validation': {
                'file': 'test_reproducibility_validation.py',
                'description': 'Reproducibility and seed validation tests',
                'timeout': 120  # 2 minutes
            }
        }
    
    def run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> TestSuiteResult:
        """Run a single test suite."""
        print(f"\n🧪 Running {suite_name}: {suite_config['description']}")
        print("=" * 80)
        
        test_file = Path(__file__).parent / suite_config['file']
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                error_tests=0,
                total_duration=0.0,
                test_results=[TestResult(
                    test_file=str(test_file),
                    test_name="file_not_found",
                    status="error",
                    duration=0.0,
                    error_message=f"Test file not found: {test_file}"
                )]
            )
        
        # Run pytest with JSON output
        start_time = time.time()
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_file),
            '--json-report',
            '--json-report-file', str(self.results_dir / f'{suite_name}_results.json'),
            '-v' if self.verbose else '-q',
            '--tb=short',
            f'--timeout={suite_config["timeout"]}'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config['timeout'] + 30  # Extra buffer
            )
            
            duration = time.time() - start_time
            
            # Parse JSON results if available
            json_file = self.results_dir / f'{suite_name}_results.json'
            test_results = []
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    
                    for test in json_data.get('tests', []):
                        test_result = TestResult(
                            test_file=suite_config['file'],
                            test_name=test.get('nodeid', 'unknown'),
                            status=test.get('outcome', 'unknown'),
                            duration=test.get('duration', 0.0),
                            error_message=test.get('call', {}).get('longrepr') if test.get('outcome') == 'failed' else None,
                            output=test.get('call', {}).get('stdout')
                        )
                        test_results.append(test_result)
                    
                    # Summary from JSON
                    summary = json_data.get('summary', {})
                    total_tests = summary.get('total', 0)
                    passed_tests = summary.get('passed', 0)
                    failed_tests = summary.get('failed', 0)
                    skipped_tests = summary.get('skipped', 0)
                    error_tests = summary.get('error', 0)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️  Failed to parse JSON results: {e}")
                    # Fall back to parsing stdout
                    total_tests, passed_tests, failed_tests, skipped_tests, error_tests = self._parse_pytest_output(result.stdout)
            else:
                # Parse stdout if JSON not available
                total_tests, passed_tests, failed_tests, skipped_tests, error_tests = self._parse_pytest_output(result.stdout)
            
            # Print results
            if result.returncode == 0:
                print(f"✅ {suite_name} completed successfully")
            else:
                print(f"❌ {suite_name} failed with return code {result.returncode}")
            
            print(f"📊 Results: {passed_tests} passed, {failed_tests} failed, {skipped_tests} skipped, {error_tests} errors")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            
            if self.verbose and result.stdout:
                print("\n📝 Output:")
                print(result.stdout)
            
            if result.stderr:
                print("\n⚠️  Errors:")
                print(result.stderr)
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                error_tests=error_tests,
                total_duration=duration,
                test_results=test_results
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {suite_name} timed out after {duration:.2f} seconds")
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=1,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration=duration,
                test_results=[TestResult(
                    test_file=suite_config['file'],
                    test_name="timeout",
                    status="error",
                    duration=duration,
                    error_message=f"Test suite timed out after {suite_config['timeout']} seconds"
                )]
            )
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {suite_name} crashed: {e}")
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=1,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration=duration,
                test_results=[TestResult(
                    test_file=suite_config['file'],
                    test_name="exception",
                    status="error",
                    duration=duration,
                    error_message=str(e)
                )]
            )
    
    def _parse_pytest_output(self, output: str) -> tuple:
        """Parse pytest output to extract test counts."""
        lines = output.split('\n')
        
        # Look for summary line like "5 passed, 2 failed, 1 skipped in 10.5s"
        for line in reversed(lines):
            if 'passed' in line or 'failed' in line or 'error' in line:
                # Extract numbers
                import re
                
                passed_match = re.search(r'(\d+) passed', line)
                failed_match = re.search(r'(\d+) failed', line)
                skipped_match = re.search(r'(\d+) skipped', line)
                error_match = re.search(r'(\d+) error', line)
                
                passed = int(passed_match.group(1)) if passed_match else 0
                failed = int(failed_match.group(1)) if failed_match else 0
                skipped = int(skipped_match.group(1)) if skipped_match else 0
                errors = int(error_match.group(1)) if error_match else 0
                
                total = passed + failed + skipped + errors
                
                return total, passed, failed, skipped, errors
        
        # Fallback if no summary found
        return 0, 0, 0, 0, 1
    
    def run_all_suites(self, suite_filter: Optional[List[str]] = None) -> Dict[str, TestSuiteResult]:
        """Run all test suites."""
        print("🚀 Starting Integration Test Suite")
        print("=" * 80)
        
        overall_start_time = time.time()
        results = {}
        
        # Filter suites if specified
        suites_to_run = self.test_suites
        if suite_filter:
            suites_to_run = {k: v for k, v in self.test_suites.items() if k in suite_filter}
        
        # Run each test suite
        for suite_name, suite_config in suites_to_run.items():
            result = self.run_test_suite(suite_name, suite_config)
            results[suite_name] = result
        
        overall_duration = time.time() - overall_start_time
        
        # Print overall summary
        self._print_summary(results, overall_duration)
        
        # Save results if requested
        if self.save_results:
            self._save_results(results, overall_duration)
        
        return results
    
    def _print_summary(self, results: Dict[str, TestSuiteResult], overall_duration: float):
        """Print overall test summary."""
        print("\n" + "=" * 80)
        print("📋 INTEGRATION TEST SUMMARY")
        print("=" * 80)
        
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed_tests for r in results.values())
        total_failed = sum(r.failed_tests for r in results.values())
        total_skipped = sum(r.skipped_tests for r in results.values())
        total_errors = sum(r.error_tests for r in results.values())
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print(f"   💥 Errors: {total_errors}")
        print(f"   📈 Success Rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "   📈 Success Rate: N/A")
        print(f"   ⏱️  Total Duration: {overall_duration:.2f} seconds")
        
        print(f"\n📋 Suite Breakdown:")
        for suite_name, result in results.items():
            status_icon = "✅" if result.failed_tests == 0 and result.error_tests == 0 else "❌"
            print(f"   {status_icon} {suite_name}: {result.passed_tests}/{result.total_tests} passed ({result.success_rate*100:.1f}%) in {result.total_duration:.2f}s")
        
        # Highlight failures
        failed_suites = [name for name, result in results.items() if result.failed_tests > 0 or result.error_tests > 0]
        if failed_suites:
            print(f"\n⚠️  Failed Suites: {', '.join(failed_suites)}")
        
        # Overall status
        if total_failed == 0 and total_errors == 0:
            print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        else:
            print(f"\n💔 SOME INTEGRATION TESTS FAILED")
        
        print("=" * 80)
    
    def _save_results(self, results: Dict[str, TestSuiteResult], overall_duration: float):
        """Save test results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f'integration_test_summary_{timestamp}.json'
        
        # Convert results to serializable format
        serializable_results = {}
        for suite_name, result in results.items():
            serializable_results[suite_name] = asdict(result)
        
        summary_data = {
            'timestamp': timestamp,
            'overall_duration': overall_duration,
            'total_suites': len(results),
            'suite_results': serializable_results,
            'summary': {
                'total_tests': sum(r.total_tests for r in results.values()),
                'passed_tests': sum(r.passed_tests for r in results.values()),
                'failed_tests': sum(r.failed_tests for r in results.values()),
                'skipped_tests': sum(r.skipped_tests for r in results.values()),
                'error_tests': sum(r.error_tests for r in results.values())
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def check_dependencies(self) -> bool:
        """Check that required dependencies are available."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'pytest',
            'pytest-json-report',
            'pytest-timeout',
            'numpy',
            'psutil'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies available")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run integration tests for evaluation system')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to file')
    parser.add_argument('--suites', nargs='+', help='Specific test suites to run', 
                       choices=['evaluation_integration', 'statistical_validation', 
                               'performance_benchmarking', 'reproducibility_validation'])
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    runner = IntegrationTestRunner(
        verbose=args.verbose,
        save_results=not args.no_save
    )
    
    # Check dependencies if requested
    if args.check_deps:
        if runner.check_dependencies():
            print("✅ All dependencies satisfied")
            return 0
        else:
            print("❌ Missing dependencies")
            return 1
    
    # Check dependencies before running tests
    if not runner.check_dependencies():
        print("❌ Cannot run tests due to missing dependencies")
        return 1
    
    # Run tests
    try:
        results = runner.run_all_suites(args.suites)
        
        # Return appropriate exit code
        total_failed = sum(r.failed_tests for r in results.values())
        total_errors = sum(r.error_tests for r in results.values())
        
        if total_failed == 0 and total_errors == 0:
            return 0  # Success
        else:
            return 1  # Failure
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())