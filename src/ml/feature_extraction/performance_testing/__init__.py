"""Performance testing framework for feature extraction module.

This module provides a comprehensive performance testing framework specifically
designed to validate the <100ms feature extraction requirement for the AI
Trading Platform's CNN+LSTM feature extraction system.
"""

from .framework import (
    FeatureExtractionPerformanceTester,
    PerformanceTestResult,
    PerformanceTestConfig
)

from .load_testing import (
    LoadTester,
    LoadTestConfig,
    LoadTestResult
)

from .stress_testing import (
    StressTester,
    StressTestConfig,
    StressTestResult
)

from .metrics import (
    PerformanceMetricsCollector,
    PerformanceReport,
    LatencyStats,
    ResourceStats
)

from .reporting import (
    PerformanceReporter,
    HTMLReportGenerator,
    JSONReportGenerator,
    generate_comparison_report
)

from .integration import (
    PerformanceTestingIntegration,
    create_performance_test_suite
)

__all__ = [
    'FeatureExtractionPerformanceTester',
    'PerformanceTestResult',
    'PerformanceTestConfig',
    'LoadTester',
    'LoadTestConfig',
    'LoadTestResult',
    'StressTester',
    'StressTestConfig',
    'StressTestResult',
    'PerformanceMetricsCollector',
    'PerformanceReport',
    'LatencyStats',
    'ResourceStats',
    'PerformanceReporter',
    'HTMLReportGenerator',
    'JSONReportGenerator',
    'generate_comparison_report',
    'PerformanceTestingIntegration',
    'create_performance_test_suite'
]