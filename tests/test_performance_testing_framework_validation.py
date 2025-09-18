"""Validation tests for the performance testing framework.

This module contains tests to validate that the performance testing framework
is working correctly and can properly measure feature extraction performance.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock
import logging

from src.ml.feature_extraction.base import FeatureExtractor
from src.ml.feature_extraction.cnn_lstm_extractor import CNNLSTMExtractor
from src.ml.feature_extraction.config import FeatureExtractionConfig

# Performance testing modules
from src.ml.feature_extraction.performance_testing.framework import (
    FeatureExtractionPerformanceTester,
    PerformanceTestConfig,
    PerformanceTestResult
)
from src.ml.feature_extraction.performance_testing.load_testing import (
    LoadTester,
    LoadTestConfig,
    LoadTestResult
)
from src.ml.feature_extraction.performance_testing.stress_testing import (
    StressTester,
    StressTestConfig,
    StressTestResult
)
from src.ml.feature_extraction.performance_testing.metrics import (
    PerformanceMetricsCollector,
    PerformanceReport
)
from src.ml.feature_extraction.performance_testing.integration import PerformanceTestingIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFrameworkValidation:
    """Validate that the performance testing framework works correctly."""
    
    @pytest.fixture
    def mock_model_fast(self):
        """Create a mock model that processes quickly."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            # Fast processing (1-5ms)
            time.sleep(0.001 + np.random.random() * 0.004)
            
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
    
    @pytest.fixture
    def mock_model_slow(self):
        """Create a mock model that processes slowly."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            # Slow processing (50-100ms)
            time.sleep(0.05 + np.random.random() * 0.05)
            
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
    
    def test_performance_tester_validation(self, mock_model_fast):
        """Validate that the performance tester can measure performance correctly."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model_fast, device='cpu')
        
        # Configure performance test
        config = PerformanceTestConfig(
            num_iterations=50,  # Small number for testing
            warmup_iterations=10,
            target_latency_ms=100.0
        )
        
        tester = FeatureExtractionPerformanceTester(config)
        
        # Run performance test
        result = tester.run_performance_test(extractor)
        
        # Validate result structure
        assert isinstance(result, PerformanceTestResult)
        assert result.test_name == config.test_name
        assert len(result.latencies_ms) == config.num_iterations
        assert result.successful_extractions == config.num_iterations
        assert result.failed_extractions == 0
        
        # Validate latency measurements
        assert all(latency >= 0 for latency in result.latencies_ms)
        avg_latency = np.mean(result.latencies_ms)
        assert avg_latency > 0
        
        # Validate statistics calculation
        stats = result.get_latency_stats()
        assert 'avg_latency_ms' in stats
        assert 'p95_latency_ms' in stats
        assert stats['avg_latency_ms'] > 0
        assert stats['p95_latency_ms'] > 0
        
        logger.info("Performance tester validation passed")
        logger.info(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
        logger.info(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
    
    def test_load_tester_validation(self, mock_model_fast):
        """Validate that the load tester can measure concurrent performance."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model_fast, device='cpu')
        
        # Configure load test
        config = LoadTestConfig(
            concurrent_users=3,  # Small number for testing
            requests_per_user=10,  # Small number for testing
            target_response_time_ms=100.0,
            target_throughput_rps=10.0
        )
        
        load_tester = LoadTester(config)
        
        # Run load test
        result = load_tester.run_concurrent_users_test(extractor)
        
        # Validate result structure
        assert isinstance(result, LoadTestResult)
        assert result.test_name == config.test_name
        assert result.total_requests == config.concurrent_users * config.requests_per_user
        assert result.successful_requests <= result.total_requests
        assert result.failed_requests >= 0
        assert result.end_time >= result.start_time
        
        # Validate metrics calculation
        throughput = result.get_throughput_rps()
        assert throughput >= 0
        
        avg_latency = result.get_avg_latency_ms()
        assert avg_latency >= 0
        
        error_rate = result.get_error_rate()
        assert 0.0 <= error_rate <= 1.0
        
        percentiles = result.get_latency_percentiles()
        assert 'p50' in percentiles
        assert 'p95' in percentiles
        
        logger.info("Load tester validation passed")
        logger.info(f"Total requests: {result.total_requests}")
        logger.info(f"Throughput: {throughput:.2f} RPS")
        logger.info(f"Average latency: {avg_latency:.2f}ms")
        logger.info(f"Error rate: {error_rate:.2%}")
    
    def test_stress_tester_validation(self, mock_model_fast):
        """Validate that the stress tester can measure performance under stress."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model_fast, device='cpu')
        
        # Configure stress test
        config = StressTestConfig(
            max_concurrent_users=2,  # Small number for testing
            max_requests_per_user=5,  # Small number for testing
            test_duration_seconds=15, # Short duration for testing
            max_acceptable_error_rate=0.30  # Higher tolerance for testing
        )
        
        stress_tester = StressTester(config)
        
        # Run stress test
        result = stress_tester.run_stress_test(extractor)
        
        # Validate result structure
        assert isinstance(result, StressTestResult)
        assert result.test_name == config.test_name
        assert result.total_requests >= 0
        assert result.successful_requests >= 0
        assert result.failed_requests >= 0
        assert result.end_time >= result.start_time
        
        # Validate metrics calculation
        throughput = result.get_throughput_rps()
        assert throughput >= 0
        
        avg_latency = result.get_avg_latency_ms()
        assert avg_latency >= 0
        
        error_rate = result.get_error_rate()
        assert 0.0 <= error_rate <= 1.0
        
        percentiles = result.get_latency_percentiles()
        assert 'p50' in percentiles
        assert 'p95' in percentiles
        
        logger.info("Stress tester validation passed")
        logger.info(f"Total requests: {result.total_requests}")
        logger.info(f"Duration: {result.end_time - result.start_time:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} RPS")
        logger.info(f"Average latency: {avg_latency:.2f}ms")
        logger.info(f"Error rate: {error_rate:.2%}")
    
    def test_metrics_collection_validation(self, mock_model_fast):
        """Validate that metrics collection works correctly."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model_fast, device='cpu')
        
        # Run a small performance test
        config = PerformanceTestConfig(
            num_iterations=20,  # Small number for testing
            warmup_iterations=5
        )
        
        tester = FeatureExtractionPerformanceTester(config)
        perf_result = tester.run_performance_test(extractor)
        
        # Collect metrics
        collector = PerformanceMetricsCollector()
        report = collector.collect_from_performance_test(perf_result)
        
        # Validate report structure
        assert isinstance(report, PerformanceReport)
        assert report.test_name == perf_result.test_name
        assert report.test_type == "performance"
        assert report.total_requests == perf_result.successful_extractions + perf_result.failed_extractions
        assert 0.0 <= report.success_rate <= 1.0
        
        # Validate latency stats
        assert report.latency_stats.avg_ms > 0
        assert report.latency_stats.p95_ms > 0
        
        # Validate throughput
        assert report.throughput_rps >= 0
        
        # Validate requirements checking
        assert isinstance(report.meets_latency_requirement, bool)
        assert isinstance(report.meets_throughput_requirement, bool)
        assert isinstance(report.meets_resource_requirement, bool)
        
        logger.info("Metrics collection validation passed")
        logger.info(f"Success rate: {report.success_rate:.2%}")
        logger.info(f"Average latency: {report.latency_stats.avg_ms:.2f}ms")
        logger.info(f"P95 latency: {report.latency_stats.p95_ms:.2f}ms")
        logger.info(f"Throughput: {report.throughput_rps:.2f} RPS")
    
    def test_framework_integration_validation(self, mock_model_fast):
        """Validate integration between framework components."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model_fast, device='cpu')
        
        # Create performance testing integration
        integration = PerformanceTestingIntegration()
        
        # Run performance test
        perf_config = PerformanceTestConfig(
            num_iterations=30,  # Small number for testing
            warmup_iterations=5
        )
        
        tester = FeatureExtractionPerformanceTester(perf_config)
        perf_result = tester.run_performance_test(extractor)
        
        # Integrate with monitoring systems
        integration.integrate_performance_test_result(perf_result)
        
        # Validate integration
        unified_stats = integration.get_unified_performance_stats()
        assert isinstance(unified_stats, dict)
        assert 'feature_extraction' in unified_stats
        assert 'performance_monitor' in unified_stats
        assert 'performance_requirements' in unified_stats
        assert 'alerts' in unified_stats
        
        # Validate requirement validation
        requirement_validation = integration.validate_100ms_requirement()
        assert isinstance(requirement_validation, dict)
        assert 'meets_100ms_requirement' in requirement_validation
        assert 'current_p95_latency_ms' in requirement_validation
        assert 'target_latency_ms' in requirement_validation
        assert requirement_validation['target_latency_ms'] == 100.0
        
        # Validate alert checking
        alerts = integration.check_performance_alerts()
        assert isinstance(alerts, dict)
        assert 'alert_count' in alerts
        assert 'alerts' in alerts
        
        logger.info("Framework integration validation passed")
        logger.info(f"Unified stats keys: {list(unified_stats.keys())}")
        logger.info(f"Requirement validation: {requirement_validation['meets_100ms_requirement']}")
        logger.info(f"Alert count: {alerts['alert_count']}")
    
    def test_performance_comparison_validation(self, mock_model_fast, mock_model_slow):
        """Validate that the framework can detect performance differences."""
        # Create extractors
        fast_extractor = CNNLSTMExtractor(mock_model_fast, device='cpu')
        slow_extractor = CNNLSTMExtractor(mock_model_slow, device='cpu')
        
        # Configure performance test
        config = PerformanceTestConfig(
            num_iterations=30,  # Small number for testing
            warmup_iterations=5
        )
        
        tester = FeatureExtractionPerformanceTester(config)
        
        # Run tests on both extractors
        fast_result = tester.run_performance_test(fast_extractor)
        slow_result = tester.run_performance_test(slow_extractor)
        
        # Compare results
        fast_stats = fast_result.get_latency_stats()
        slow_stats = slow_result.get_latency_stats()
        
        # Slow model should have higher latency
        assert slow_stats['avg_latency_ms'] > fast_stats['avg_latency_ms']
        assert slow_stats['p95_latency_ms'] > fast_stats['p95_latency_ms']
        
        logger.info("Performance comparison validation passed")
        logger.info(f"Fast model average latency: {fast_stats['avg_latency_ms']:.2f}ms")
        logger.info(f"Slow model average latency: {slow_stats['avg_latency_ms']:.2f}ms")
        logger.info(f"Performance difference: {slow_stats['avg_latency_ms'] - fast_stats['avg_latency_ms']:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])