"""Demo script showing how to use the feature extraction performance testing framework.

This script demonstrates how to use the performance testing framework to validate
feature extraction performance and integrate with existing monitoring systems.
"""

import numpy as np
from unittest.mock import Mock
import logging
import time

from src.ml.feature_extraction.cnn_lstm_extractor import CNNLSTMExtractor
from src.ml.feature_extraction.factory import FeatureExtractorFactory

# Performance testing modules
from src.ml.feature_extraction.performance_testing.framework import (
    FeatureExtractionPerformanceTester,
    PerformanceTestConfig
)
from src.ml.feature_extraction.performance_testing.load_testing import (
    LoadTester,
    LoadTestConfig
)
from src.ml.feature_extraction.performance_testing.stress_testing import (
    StressTester,
    StressTestConfig
)
from src.ml.feature_extraction.performance_testing.metrics import PerformanceMetricsCollector
from src.ml.feature_extraction.performance_testing.reporting import (
    JSONReportGenerator,
    HTMLReportGenerator
)
from src.ml.feature_extraction.performance_testing.integration import PerformanceTestingIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_model():
    """Create a mock model for demonstration purposes."""
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = None
    
    def mock_forward(input_tensor, return_features=True, use_ensemble=True):
        # Simulate realistic processing time (10-50ms)
        processing_time = 0.01 + np.random.random() * 0.04
        time.sleep(processing_time)
        
        # Return realistic outputs
        batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 0 else 1
        return {
            'fused_features': np.random.randn(batch_size, 10, 256).astype(np.float32),
            'cnn_features': np.random.randn(batch_size, 10, 128).astype(np.float32),
            'lstm_features': np.random.randn(batch_size, 10, 128).astype(np.float32),
            'classification_probs': np.random.rand(batch_size, 3).astype(np.float32),
            'regression_uncertainty': np.random.rand(batch_size, 1).astype(np.float32)
        }
    
    model.forward = mock_forward
    return model


def demo_single_extraction_performance():
    """Demonstrate single extraction performance testing."""
    logger.info("=== Single Extraction Performance Test ===")
    
    # Create extractor
    mock_model = create_mock_model()
    extractor = FeatureExtractorFactory.create_basic_extractor(mock_model)
    
    # Configure performance test
    config = PerformanceTestConfig(
        num_iterations=100,
        warmup_iterations=10,
        target_latency_ms=100.0
    )
    
    # Run performance test
    tester = FeatureExtractionPerformanceTester(config)
    result = tester.run_performance_test(extractor)
    
    # Display results
    stats = result.get_latency_stats()
    logger.info(f"Performance Test Results:")
    logger.info(f"  Test Name: {result.test_name}")
    logger.info(f"  Total Iterations: {len(result.latencies_ms)}")
    logger.info(f"  Average Latency: {stats['avg_latency_ms']:.2f}ms")
    logger.info(f"  P95 Latency: {stats['p95_latency_ms']:.2f}ms")
    logger.info(f"  P99 Latency: {stats['p99_latency_ms']:.2f}ms")
    logger.info(f"  Success Rate: {result.get_success_rate():.2%}")
    logger.info(f"  Meets Requirements: {result.meets_performance_requirements()}")
    
    return result


def demo_load_testing():
    """Demonstrate load testing."""
    logger.info("\n=== Load Testing ===")
    
    # Create extractor
    mock_model = create_mock_model()
    extractor = FeatureExtractorFactory.create_basic_extractor(mock_model)
    
    # Configure load test
    config = LoadTestConfig(
        concurrent_users=5,
        requests_per_user=20,
        target_response_time_ms=100.0,
        target_throughput_rps=20.0
    )
    
    # Run load test
    load_tester = LoadTester(config)
    result = load_tester.run_concurrent_users_test(extractor)
    
    # Display results
    throughput = result.get_throughput_rps()
    avg_latency = result.get_avg_latency_ms()
    error_rate = result.get_error_rate()
    percentiles = result.get_latency_percentiles()
    
    logger.info(f"Load Test Results:")
    logger.info(f"  Test Name: {result.test_name}")
    logger.info(f"  Total Requests: {result.total_requests}")
    logger.info(f"  Successful Requests: {result.successful_requests}")
    logger.info(f"  Failed Requests: {result.failed_requests}")
    logger.info(f" Success Rate: {1 - error_rate:.2%}")
    logger.info(f"  Throughput: {throughput:.2f} RPS")
    logger.info(f"  Average Latency: {avg_latency:.2f}ms")
    logger.info(f"  P95 Latency: {percentiles['p95']:.2f}ms")
    logger.info(f"  P99 Latency: {percentiles['p99']:.2f}ms")
    
    return result


def demo_stress_testing():
    """Demonstrate stress testing."""
    logger.info("\n=== Stress Testing ===")
    
    # Create extractor
    mock_model = create_mock_model()
    extractor = FeatureExtractorFactory.create_basic_extractor(mock_model)
    
    # Configure stress test
    config = StressTestConfig(
        max_concurrent_users=3,
        max_requests_per_user=10,
        test_duration_seconds=30,
        max_acceptable_error_rate=0.20
    )
    
    # Run stress test
    stress_tester = StressTester(config)
    result = stress_tester.run_stress_test(extractor)
    
    # Display results
    throughput = result.get_throughput_rps()
    avg_latency = result.get_avg_latency_ms()
    error_rate = result.get_error_rate()
    percentiles = result.get_latency_percentiles()
    
    logger.info(f"Stress Test Results:")
    logger.info(f"  Test Name: {result.test_name}")
    logger.info(f"  Duration: {result.end_time - result.start_time:.2f}s")
    logger.info(f"  Total Requests: {result.total_requests}")
    logger.info(f"  Successful Requests: {result.successful_requests}")
    logger.info(f"  Failed Requests: {result.failed_requests}")
    logger.info(f"  Success Rate: {1 - error_rate:.2%}")
    logger.info(f"  Throughput: {throughput:.2f} RPS")
    logger.info(f"  Average Latency: {avg_latency:.2f}ms")
    logger.info(f"  P95 Latency: {percentiles['p95']:.2f}ms")
    logger.info(f"  Peak Memory: {result.peak_memory_mb:.2f}MB")
    
    return result


def demo_metrics_collection_and_reporting(test_results):
    """Demonstrate metrics collection and reporting."""
    logger.info("\n=== Metrics Collection and Reporting ===")
    
    # Collect metrics
    collector = PerformanceMetricsCollector()
    reports = []
    
    for result in test_results:
        if hasattr(result, 'latencies_ms'):  # PerformanceTestResult
            report = collector.collect_from_performance_test(result)
            reports.append(report)
        elif hasattr(result, 'request_latencies_ms'):  # LoadTestResult or StressTestResult
            if hasattr(result, 'user_metrics'):  # LoadTestResult
                report = collector.collect_from_load_test(result)
            else:  # StressTestResult
                report = collector.collect_from_stress_test(result)
            reports.append(report)
    
    # Generate reports
    json_generator = JSONReportGenerator()
    html_generator = HTMLReportGenerator()
    
    # Generate JSON report
    json_report = json_generator.generate_report(reports)
    logger.info(f"JSON Report Generated: {len(json_report)} characters")
    
    # Generate HTML report
    html_report = html_generator.generate_report(reports)
    logger.info(f"HTML Report Generated: {len(html_report)} characters")
    
    # Save reports to files
    try:
        with open('performance_test_report.json', 'w') as f:
            f.write(json_report)
        logger.info("JSON report saved to performance_test_report.json")
        
        with open('performance_test_report.html', 'w') as f:
            f.write(html_report)
        logger.info("HTML report saved to performance_test_report.html")
    except Exception as e:
        logger.error(f"Error saving reports: {e}")
    
    return reports


def demo_integration_with_monitoring():
    """Demonstrate integration with existing monitoring systems."""
    logger.info("\n=== Integration with Monitoring Systems ===")
    
    # Create performance testing integration
    integration = PerformanceTestingIntegration()
    
    # Create extractor
    mock_model = create_mock_model()
    extractor = FeatureExtractorFactory.create_basic_extractor(mock_model)
    
    # Run a small performance test
    config = PerformanceTestConfig(
        num_iterations=50,
        warmup_iterations=5
    )
    
    tester = FeatureExtractionPerformanceTester(config)
    result = tester.run_performance_test(extractor)
    
    # Integrate with monitoring systems
    integration.integrate_performance_test_result(result)
    
    # Get unified performance stats
    unified_stats = integration.get_unified_performance_stats()
    
    # Validate <100ms requirement
    requirement_validation = integration.validate_100ms_requirement()
    
    # Check for alerts
    alerts = integration.check_performance_alerts()
    
    # Display results
    logger.info("Unified Performance Stats:")
    logger.info(f"  Feature Extraction Tests: {len(unified_stats.get('feature_extraction', {}))}")
    logger.info(f"  Performance Monitor Available: {'performance_monitor' in unified_stats}")
    
    logger.info("\nRequirement Validation:")
    logger.info(f"  Meets <100ms Requirement: {requirement_validation.get('meets_100ms_requirement', False)}")
    logger.info(f"  Current P95 Latency: {requirement_validation.get('current_p95_latency_ms', 0):.2f}ms")
    logger.info(f"  Latency Margin: {requirement_validation.get('latency_margin_ms', 0):.2f}ms")
    
    logger.info("\nAlerts:")
    logger.info(f"  Alert Count: {alerts.get('alert_count', 0)}")
    
    return integration


def main():
    """Main demo function."""
    logger.info("Feature Extraction Performance Testing Framework Demo")
    logger.info("=" * 50)
    
    # Run demonstrations
    performance_result = demo_single_extraction_performance()
    load_result = demo_load_testing()
    stress_result = demo_stress_testing()
    
    # Collect and report metrics
    test_results = [performance_result, load_result, stress_result]
    reports = demo_metrics_collection_and_reporting(test_results)
    
    # Demonstrate integration
    integration = demo_integration_with_monitoring()
    
    logger.info("\n=== Demo Completed ===")
    logger.info("Check the generated performance_test_report.json and performance_test_report.html files for detailed reports.")


if __name__ == "__main__":
    main()