"""Integration of performance testing with existing monitoring components.

This module provides integration between the feature extraction performance
testing framework and existing monitoring systems.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from src.ml.feature_extraction.monitoring import (
    FeatureExtractionPerformanceMonitor,
    FeatureExtractionMetrics
)
from src.ml.feature_extraction.enhanced_metrics import EnhancedMetricsCollector
from src.ml.feature_extraction.alerting import FeatureExtractionAlertingSystem

from .framework import PerformanceTestResult
from .load_testing import LoadTestResult
from .stress_testing import StressTestResult
from .metrics import PerformanceMetricsCollector, PerformanceReport

logger = logging.getLogger(__name__)


class PerformanceTestingIntegration:
    """Integration between performance testing and monitoring systems."""
    
    def __init__(self):
        """Initialize integration components."""
        # Initialize monitoring components
        self.performance_monitor = FeatureExtractionPerformanceMonitor()
        self.enhanced_metrics_collector = EnhancedMetricsCollector()
        self.alerting_system = FeatureExtractionAlertingSystem(self.enhanced_metrics_collector)
        
        # Initialize metrics collector
        self.metrics_collector = PerformanceMetricsCollector()
        
        logger.info("Performance testing integration initialized")
    
    def integrate_performance_test_result(self, result: PerformanceTestResult) -> None:
        """Integrate performance test result with monitoring systems.
        
        Args:
            result: Performance test result to integrate
        """
        logger.info(f"Integrating performance test result: {result.test_name}")
        
        # Convert to performance report
        report = self.metrics_collector.collect_from_performance_test(result)
        
        # Add metrics to enhanced collector
        if result.latencies_ms:
            # Create feature extraction metrics for each latency measurement
            for latency_ms in result.latencies_ms[:10]:  # Limit to first 10 for demo
                metrics = FeatureExtractionMetrics(
                    timestamp=datetime.now(),
                    duration_ms=latency_ms,
                    used_cache=False,  # Simplified for demo
                    used_fallback=False, # Simplified for demo
                    had_error=False,  # Simplified for demo
                    cpu_percent=np.random.uniform(10, 50),  # Simulated CPU usage
                    memory_mb=np.random.uniform(100, 300),  # Simulated memory usage
                    batch_size=1
                )
                self.performance_monitor.end_extraction(
                    duration_ms=latency_ms,
                    used_cache=False,
                    used_fallback=False,
                    had_error=False,
                    cpu_percent=metrics.cpu_percent,
                    memory_mb=metrics.memory_mb,
                    batch_size=metrics.batch_size
                )
                self.enhanced_metrics_collector.add_feature_extraction_metrics(metrics)
        
        logger.info(f"Integrated {len(result.latencies_ms)} latency measurements")
    
    def integrate_load_test_result(self, result: LoadTestResult) -> None:
        """Integrate load test result with monitoring systems.
        
        Args:
            result: Load test result to integrate
        """
        logger.info(f"Integrating load test result: {result.test_name}")
        
        # Convert to performance report
        report = self.metrics_collector.collect_from_load_test(result)
        
        # Add metrics to enhanced collector
        if result.request_latencies_ms:
            # Create feature extraction metrics for sample latency measurements
            sample_size = min(50, len(result.request_latencies_ms)) # Limit for performance
            sample_indices = np.random.choice(len(result.request_latencies_ms), sample_size, replace=False)
            
            for i in sample_indices:
                latency_ms = result.request_latencies_ms[i]
                metrics = FeatureExtractionMetrics(
                    timestamp=datetime.now(),
                    duration_ms=latency_ms,
                    used_cache=i % 3 == 0,  # Simulate cache usage
                    used_fallback=i % 10 == 0,  # Simulate fallback usage
                    had_error=i % 20 == 0,  # Simulate occasional errors
                    cpu_percent=np.random.uniform(20, 80),  # Simulated CPU usage
                    memory_mb=np.random.uniform(150, 500),  # Simulated memory usage
                    batch_size=1
                )
                self.performance_monitor.end_extraction(
                    duration_ms=latency_ms,
                    used_cache=metrics.used_cache,
                    used_fallback=metrics.used_fallback,
                    had_error=metrics.had_error,
                    cpu_percent=metrics.cpu_percent,
                    memory_mb=metrics.memory_mb,
                    batch_size=metrics.batch_size
                )
                self.enhanced_metrics_collector.add_feature_extraction_metrics(metrics)
        
        logger.info(f"Integrated {min(50, len(result.request_latencies_ms))} sample latency measurements")
    
    def integrate_stress_test_result(self, result: StressTestResult) -> None:
        """Integrate stress test result with monitoring systems.
        
        Args:
            result: Stress test result to integrate
        """
        logger.info(f"Integrating stress test result: {result.test_name}")
        
        # Convert to performance report
        report = self.metrics_collector.collect_from_stress_test(result)
        
        # Add metrics to enhanced collector
        if result.request_latencies_ms:
            # Create feature extraction metrics for sample latency measurements
            sample_size = min(30, len(result.request_latencies_ms))  # Limit for performance
            sample_indices = np.random.choice(len(result.request_latencies_ms), sample_size, replace=False)
            
            for i in sample_indices:
                latency_ms = result.request_latencies_ms[i]
                # Simulate higher resource usage in stress testing
                metrics = FeatureExtractionMetrics(
                    timestamp=datetime.now(),
                    duration_ms=latency_ms,
                    used_cache=i % 4 == 0,  # Less cache usage under stress
                    used_fallback=i % 5 == 0,  # More fallback usage under stress
                    had_error=i % 10 == 0,  # More errors under stress
                    cpu_percent=np.random.uniform(50, 95),  # High CPU usage
                    memory_mb=np.random.uniform(300, 1000),  # High memory usage
                    batch_size=1
                )
                self.performance_monitor.end_extraction(
                    duration_ms=latency_ms,
                    used_cache=metrics.used_cache,
                    used_fallback=metrics.used_fallback,
                    had_error=metrics.had_error,
                    cpu_percent=metrics.cpu_percent,
                    memory_mb=metrics.memory_mb,
                    batch_size=metrics.batch_size
                )
                self.enhanced_metrics_collector.add_feature_extraction_metrics(metrics)
        
        logger.info(f"Integrated {min(30, len(result.request_latencies_ms))} sample latency measurements")
    
    def get_unified_performance_stats(self) -> Dict[str, Any]:
        """Get unified performance statistics from all integrated systems.
        
        Returns:
            Dictionary with unified performance statistics
        """
        # Get stats from each system
        feature_stats = self.enhanced_metrics_collector.get_performance_summary()
        monitor_stats = self.performance_monitor.get_performance_stats()
        requirements_status = self.performance_monitor.get_performance_requirements_status()
        
        # Get alert summary
        alert_summary = self.alerting_system.get_alert_summary()
        
        return {
            'feature_extraction': feature_stats,
            'performance_monitor': monitor_stats,
            'performance_requirements': requirements_status,
            'alerts': alert_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_performance_alerts(self) -> Dict[str, Any]:
        """Check for performance alerts.
        
        Returns:
            Dictionary with alert information
        """
        try:
            alerts = self.alerting_system.check_performance_alerts()
            return {
                'alert_count': len(alerts),
                'alerts': [str(alert) for alert in alerts],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
            return {
                'alert_count': 0,
                'alerts': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_100ms_requirement(self) -> Dict[str, Any]:
        """Validate the <100ms feature extraction requirement.
        
        Returns:
            Dictionary with validation results
        """
        # Get performance summary
        performance_summary = self.enhanced_metrics_collector.get_performance_summary()
        requirements_status = self.enhanced_metrics_collector.meets_performance_requirements()
        
        # Get detailed latency stats
        latency_stats = self.performance_monitor.get_performance_stats()
        
        # Check if requirement is met
        p95_latency = performance_summary.get('p95_latency_ms', float('inf'))
        meets_requirement = p95_latency < 100.0
        
        return {
            'meets_100ms_requirement': meets_requirement,
            'current_p95_latency_ms': p95_latency,
            'target_latency_ms': 100.0,
            'latency_margin_ms': 100.0 - p95_latency if p95_latency != float('inf') else 0.0,
            'performance_summary': performance_summary,
            'requirements_status': requirements_status,
            'detailed_stats': latency_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_comprehensive_report(self, test_results) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            test_results: List of test results to include in report
            
        Returns:
            Dictionary with comprehensive report data
        """
        # Convert test results to performance reports
        reports = []
        for result in test_results:
            if isinstance(result, PerformanceTestResult):
                report = self.metrics_collector.collect_from_performance_test(result)
                reports.append(report)
            elif isinstance(result, LoadTestResult):
                report = self.metrics_collector.collect_from_load_test(result)
                reports.append(report)
            elif isinstance(result, StressTestResult):
                report = self.metrics_collector.collect_from_stress_test(result)
                reports.append(report)
        
        # Get unified stats
        unified_stats = self.get_unified_performance_stats()
        
        # Validate requirements
        requirement_validation = self.validate_100ms_requirement()
        
        # Check alerts
        alerts = self.check_performance_alerts()
        
        return {
            'reports': [report.get_summary() for report in reports],
            'unified_stats': unified_stats,
            'requirement_validation': requirement_validation,
            'alerts': alerts,
            'generated_at': datetime.now().isoformat()
        }
    
    def reset_monitoring_systems(self) -> None:
        """Reset all monitoring systems."""
        try:
            self.performance_monitor.reset_statistics()
            self.enhanced_metrics_collector.reset_metrics()
            self.alerting_system.clear_alert_history()
            logger.info("All monitoring systems reset successfully")
        except Exception as e:
            logger.error(f"Error resetting monitoring systems: {e}")


def create_performance_test_suite():
    """Create a comprehensive performance test suite.
    
    Returns:
        Dictionary with test suite configuration
    """
    return {
        'single_extraction_test': {
            'name': 'Single Feature Extraction Latency Test',
            'description': 'Validates <100ms requirement for single feature extraction',
            'config': {
                'iterations': 1000,
                'warmup_iterations': 100,
                'target_latency_ms': 100.0
            }
        },
        'cached_extraction_test': {
            'name': 'Cached Feature Extraction Performance Test',
            'description': 'Validates cached extraction performance',
            'config': {
                'iterations': 500,
                'warmup_iterations': 50,
                'target_latency_ms': 50.0
            }
        },
        'concurrent_load_test': {
            'name': 'Concurrent Load Performance Test',
            'description': 'Validates performance under concurrent load',
            'config': {
                'concurrent_users': 10,
                'requests_per_user': 100,
                'target_response_time_ms': 100.0,
                'target_throughput_rps': 50.0
            }
        },
        'stress_test': {
            'name': 'Stress Test',
            'description': 'Validates performance under extreme conditions',
            'config': {
                'max_concurrent_users': 50,
                'max_requests_per_user': 200,
                'test_duration_seconds': 300,
                'max_acceptable_latency_ms': 1000.0
            }
        }
    }