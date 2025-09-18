#!/usr/bin/env python3
"""
Integration test runner for the AI Trading Platform.

This script runs comprehensive integration tests including:
- End-to-end workflow tests
- Performance benchmarks
- Chaos engineering tests
- Multi-exchange integration tests

Usage:
    python tests/run_integration_tests.py [options]

Options:
    --fast          Run only fast tests (exclude slow performance tests)
    --performance   Run only performance benchmarks
    --chaos         Run only chaos engineering tests
    --workflows     Run only end-to-end workflow tests
    --all           Run all integration tests (default)
    --verbose       Enable verbose output
    --parallel      Run tests in parallel where possible
    --report        Generate detailed test report
"""

import sys
import os
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
import json


class IntegrationTestRunner:
    """Runner for comprehensive integration tests."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(self, test_type: str = "all", **kwargs) -> dict:
        """Run specified test suite."""
        
        self.start_time = time.time()
        
        print(f"🚀 Starting {test_type} integration tests...")
        print(f"📁 Test directory: {self.test_dir}")
        print(f"📁 Project root: {self.project_root}")
        print("-" * 60)
        
        # Change to project root directory
        os.chdir(self.project_root)
        
        if test_type == "all":
            self._run_all_tests(**kwargs)
        elif test_type == "workflows":
            self._run_workflow_tests(**kwargs)
        elif test_type == "performance":
            self._run_performance_tests(**kwargs)
        elif test_type == "chaos":
            self._run_chaos_tests(**kwargs)
        elif test_type == "fast":
            self._run_fast_tests(**kwargs)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        self.end_time = time.time()
        
        return self._generate_summary()
    
    def _run_all_tests(self, **kwargs):
        """Run all integration tests."""
        
        test_suites = [
            ("End-to-End Workflows", "tests/test_end_to_end_workflows.py"),
            ("Performance Benchmarks", "tests/test_performance_benchmarks.py"),
            ("Chaos Engineering", "tests/test_chaos_engineering.py"),
            ("Comprehensive Integration", "tests/test_comprehensive_integration.py"),
            ("Data Persistence", "tests/test_data_persistence_consistency.py"),
            ("API Performance", "tests/test_api_performance.py"),
            ("Backtesting Integration", "tests/test_backtesting_integration.py")
        ]
        
        for suite_name, test_file in test_suites:
            print(f"\n📋 Running {suite_name} tests...")
            result = self._run_pytest(test_file, **kwargs)
            self.results[suite_name] = result
    
    def _run_workflow_tests(self, **kwargs):
        """Run end-to-end workflow tests."""
        
        print("🔄 Running end-to-end workflow tests...")
        
        test_files = [
            "tests/test_end_to_end_workflows.py",
            "tests/test_comprehensive_integration.py::TestEndToEndTradingWorkflows"
        ]
        
        for test_file in test_files:
            result = self._run_pytest(test_file, **kwargs)
            self.results[f"Workflows - {test_file}"] = result
    
    def _run_performance_tests(self, **kwargs):
        """Run performance benchmark tests."""
        
        print("⚡ Running performance benchmark tests...")
        
        # Performance tests with specific markers
        test_commands = [
            ("Data Processing Performance", "tests/test_performance_benchmarks.py::TestDataProcessingPerformance"),
            ("ML Model Performance", "tests/test_performance_benchmarks.py::TestMLModelPerformance"),
            ("System Resource Usage", "tests/test_performance_benchmarks.py::TestSystemResourceUsage"),
            ("API Performance", "tests/test_api_performance.py"),
            ("Scalability Benchmarks", "tests/test_performance_benchmarks.py::TestScalabilityBenchmarks")
        ]
        
        for test_name, test_path in test_commands:
            print(f"\n📊 Running {test_name}...")
            result = self._run_pytest(test_path, **kwargs)
            self.results[test_name] = result
    
    def _run_chaos_tests(self, **kwargs):
        """Run chaos engineering tests."""
        
        print("🌪️  Running chaos engineering tests...")
        
        test_commands = [
            ("Network Failures", "tests/test_chaos_engineering.py::TestNetworkFailures"),
            ("Service Failures", "tests/test_chaos_engineering.py::TestServiceFailures"),
            ("Resource Exhaustion", "tests/test_chaos_engineering.py::TestResourceExhaustion"),
            ("Multi-Exchange Resilience", "tests/test_comprehensive_integration.py::TestChaosEngineering")
        ]
        
        for test_name, test_path in test_commands:
            print(f"\n💥 Running {test_name}...")
            result = self._run_pytest(test_path, **kwargs)
            self.results[test_name] = result
    
    def _run_fast_tests(self, **kwargs):
        """Run only fast tests (exclude slow performance tests)."""
        
        print("🏃 Running fast integration tests...")
        
        # Run all tests but exclude slow ones
        pytest_args = ["-m", "not slow"]
        
        test_files = [
            "tests/test_end_to_end_workflows.py",
            "tests/test_comprehensive_integration.py",
            "tests/test_data_persistence_consistency.py"
        ]
        
        for test_file in test_files:
            result = self._run_pytest(test_file, extra_args=pytest_args, **kwargs)
            self.results[f"Fast - {test_file}"] = result
    
    def _run_pytest(self, test_path: str, verbose: bool = False, 
                   parallel: bool = False, extra_args: list = None) -> dict:
        """Run pytest with specified parameters."""
        
        cmd = ["python", "-m", "pytest", test_path]
        
        # Add common arguments
        cmd.extend(["-v" if verbose else "-q"])
        cmd.extend(["--tb=short"])
        cmd.extend(["--durations=10"])  # Show 10 slowest tests
        
        # Add parallel execution if requested
        if parallel:
            cmd.extend(["-n", "auto"])  # Requires pytest-xdist
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            end_time = time.time()
            
            return {
                'success': result.returncode == 0,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'duration': 1800,
                'stdout': '',
                'stderr': 'Test timed out after 30 minutes',
                'returncode': -1
            }
        
        except Exception as e:
            return {
                'success': False,
                'duration': 0,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
    
    def _generate_summary(self) -> dict:
        """Generate test execution summary."""
        
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        summary = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'total_duration': total_duration,
            'test_suites': len(self.results),
            'successful_suites': sum(1 for r in self.results.values() if r['success']),
            'failed_suites': sum(1 for r in self.results.values() if not r['success']),
            'results': self.results
        }
        
        return summary
    
    def print_summary(self, summary: dict):
        """Print test execution summary."""
        
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"📋 Test Suites: {summary['test_suites']}")
        print(f"✅ Successful: {summary['successful_suites']}")
        print(f"❌ Failed: {summary['failed_suites']}")
        
        if summary['successful_suites'] == summary['test_suites']:
            print("\n🎉 All integration tests passed!")
        else:
            print(f"\n⚠️  {summary['failed_suites']} test suite(s) failed")
        
        print("\n📋 Detailed Results:")
        print("-" * 40)
        
        for suite_name, result in summary['results'].items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            
            print(f"{status} {suite_name} ({duration:.2f}s)")
            
            if not result['success']:
                print(f"   Error: {result['stderr'][:100]}...")
        
        print("\n" + "=" * 60)
    
    def save_report(self, summary: dict, filename: str = None):
        """Save detailed test report to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integration_test_report_{timestamp}.json"
        
        report_path = self.test_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_path}")


def main():
    """Main entry point for test runner."""
    
    parser = argparse.ArgumentParser(
        description="Run AI Trading Platform integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", default=True,
                           help="Run all integration tests (default)")
    test_group.add_argument("--fast", action="store_true",
                           help="Run only fast tests (exclude slow performance tests)")
    test_group.add_argument("--performance", action="store_true",
                           help="Run only performance benchmarks")
    test_group.add_argument("--chaos", action="store_true",
                           help="Run only chaos engineering tests")
    test_group.add_argument("--workflows", action="store_true",
                           help="Run only end-to-end workflow tests")
    
    # Execution options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Run tests in parallel where possible")
    parser.add_argument("--report", "-r", action="store_true",
                       help="Generate detailed test report")
    
    args = parser.parse_args()
    
    # Determine test type
    if args.fast:
        test_type = "fast"
    elif args.performance:
        test_type = "performance"
    elif args.chaos:
        test_type = "chaos"
    elif args.workflows:
        test_type = "workflows"
    else:
        test_type = "all"
    
    # Run tests
    runner = IntegrationTestRunner()
    
    try:
        summary = runner.run_test_suite(
            test_type=test_type,
            verbose=args.verbose,
            parallel=args.parallel
        )
        
        runner.print_summary(summary)
        
        if args.report:
            runner.save_report(summary)
        
        # Exit with appropriate code
        if summary['failed_suites'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()