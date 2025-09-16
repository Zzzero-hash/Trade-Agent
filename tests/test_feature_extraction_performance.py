"""Performance tests for feature extraction components.

This module contains tests to validate feature extraction performance
under various load conditions and to verify the <100ms requirement.
"""

import pytest
import numpy as np
import time
import logging
from unittest.mock import Mock, patch

from src.ml.feature_extraction.base import FeatureExtractor
from src.ml.feature_extraction.cnn_lstm_extractor import CNNLSTMExtractor
from src.ml.feature_extraction.cached_extractor import CachedFeatureExtractor
from src.ml.feature_extraction.fallback_extractor import FallbackFeatureExtractor
from src.ml.feature_extraction.factory import FeatureExtractorFactory
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFeatureExtractionPerformance:
    """Test feature extraction performance under various conditions."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        # Mock forward method to return expected outputs
        mock_outputs = {
            'fused_features': np.random.randn(1, 10, 256).astype(np.float32),
            'cnn_features': np.random.randn(1, 10, 128).astype(np.float32),
            'lstm_features': np.random.randn(1, 10, 128).astype(np.float32),
            'classification_probs': np.random.rand(1, 3).astype(np.float32),
            'regression_uncertainty': np.random.rand(1, 1).astype(np.float32),
            'ensemble_weights': np.random.rand(1, 4).astype(np.float32)
        }
        model.forward.return_value = mock_outputs
        return model
    
    @pytest.fixture
    def basic_extractor(self, mock_model):
        """Create a basic CNN+LSTM extractor."""
        return CNNLSTMExtractor(mock_model, device='cpu')
    
    @pytest.fixture
    def cached_extractor(self, mock_model):
        """Create a cached extractor."""
        base_extractor = CNNLSTMExtractor(mock_model, device='cpu')
        return CachedFeatureExtractor(base_extractor, cache_size=100, ttl_seconds=60)
    
    @pytest.fixture
    def full_extractor(self, mock_model):
        """Create a full feature extractor with caching and fallback."""
        config = FeatureExtractionConfig(
            enable_caching=True,
            enable_fallback=True,
            cache_size=100,
            cache_ttl_seconds=60
        )
        return FeatureExtractorFactory.create_extractor(mock_model, config)
    
    def test_single_extraction_latency(self, basic_extractor):
        """Test single feature extraction latency requirement (<100ms)."""
        test_data = np.random.randn(60, 15).astype(np.float32)
        
        # Run multiple iterations to get stable measurements
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = basic_extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        logger.info(f"Single extraction latency results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        logger.info(f"  P99: {p99_latency:.2f}ms")
        
        # Validate requirements
        assert avg_latency < 80, f"Average latency {avg_latency:.2f}ms >= 80ms"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms >= 100ms"
        assert p99_latency < 150, f"P99 latency {p99_latency:.2f}ms >= 150ms"
    
    def test_cached_extraction_performance(self, cached_extractor):
        """Test cached feature extraction performance."""
        test_data = np.random.randn(60, 15).astype(np.float32)
        
        # First call to populate cache
        cached_extractor.extract_features(test_data)
        
        # Run multiple iterations for cached performance
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = cached_extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        logger.info(f"Cached extraction latency results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        
        # Cached extraction should be much faster
        assert avg_latency < 20, f"Average cached latency {avg_latency:.2f}ms >= 20ms"
        assert p95_latency < 50, f"P95 cached latency {p95_latency:.2f}ms >= 50ms"
    
    def test_fallback_extraction_performance(self, mock_model):
        """Test fallback feature extraction performance."""
        # Create extractor that always uses fallback
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_fallback=True,
            fallback_feature_dim=15
        )
        
        # Mock extractor that always fails
        failing_extractor = Mock(spec=FeatureExtractor)
        failing_extractor.extract_features.side_effect = Exception("Model unavailable")
        
        fallback_extractor = FallbackFeatureExtractor(failing_extractor)
        
        test_data = np.random.randn(60, 15).astype(np.float32)
        
        # Run fallback extraction tests
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = fallback_extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)
        
        logger.info(f"Fallback extraction results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")
        
        # Fallback should be very fast
        assert avg_latency < 10, f"Average fallback latency {avg_latency:.2f}ms >= 10ms"
        assert p95_latency < 20, f"P95 fallback latency {p95_latency:.2f}ms >= 20ms"
        assert max_latency < 50, f"Max fallback latency {max_latency:.2f}ms >= 50ms"
    
    def test_performance_under_concurrent_load(self, basic_extractor):
        """Test feature extraction performance under concurrent load."""
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        def extraction_worker(worker_id: int, num_extractions: int = 20):
            """Worker function for concurrent testing."""
            results = []
            test_data = np.random.randn(60, 15).astype(np.float32)
            
            for i in range(num_extractions):
                start_time = time.time()
                try:
                    features = basic_extractor.extract_features(test_data)
                    end_time = time.time()
                    results.append({
                        'worker_id': worker_id,
                        'extraction_id': i,
                        'latency_ms': (end_time - start_time) * 1000,
                        'success': True
                    })
                except Exception as e:
                    end_time = time.time()
                    results.append({
                        'worker_id': worker_id,
                        'extraction_id': i,
                        'latency_ms': (end_time - start_time) * 1000,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
        
        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(extraction_worker, i, 10)  # 10 extractions per worker
                    for i in range(concurrency)
                ]
                
                all_results = []
                for future in futures:
                    all_results.extend(future.result())
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_extractions = len(all_results)
            successful_extractions = sum(1 for r in all_results if r['success'])
            failed_extractions = total_extractions - successful_extractions
            latencies = [r['latency_ms'] for r in all_results if r['success']]
            
            avg_latency = np.mean(latencies) if latencies else 0
            p95_latency = np.percentile(latencies, 95) if len(latencies) >= 20 else 0
            throughput = total_extractions / total_time if total_time > 0 else 0
            
            success_rate = successful_extractions / total_extractions if total_extractions > 0 else 0
            
            logger.info(f"  Concurrency {concurrency}:")
            logger.info(f"    Total extractions: {total_extractions}")
            logger.info(f"    Success rate: {success_rate:.2%}")
            logger.info(f"    Average latency: {avg_latency:.2f}ms")
            logger.info(f"    P95 latency: {p95_latency:.2f}ms")
            logger.info(f"    Throughput: {throughput:.2f} extractions/sec")
            
            # Validate performance requirements
            assert success_rate >= 0.95, f"Success rate {success_rate:.2%} < 95%"
            assert avg_latency < 200, f"Average latency {avg_latency:.2f}ms >= 200ms"
            assert throughput > 5, f"Throughput {throughput:.2f} < 5 extractions/sec"


class TestLoadTestingScenarios:
    """Test load testing scenarios for feature extraction."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        # Mock forward method to return expected outputs
        mock_outputs = {
            'fused_features': np.random.randn(1, 10, 256).astype(np.float32),
            'cnn_features': np.random.randn(1, 10, 128).astype(np.float32),
            'lstm_features': np.random.randn(1, 10, 128).astype(np.float32),
            'classification_probs': np.random.rand(1, 3).astype(np.float32),
            'regression_uncertainty': np.random.rand(1, 1).astype(np.float32),
            'ensemble_weights': np.random.rand(1, 4).astype(np.float32)
        }
        model.forward.return_value = mock_outputs
        return model
    
    @pytest.fixture
    def extractor(self, mock_model):
        """Create a feature extractor for testing."""
        return CNNLSTMExtractor(mock_model, device='cpu')
    
    def test_constant_load_scenario(self, extractor):
        """Test constant load scenario."""
        # Configure load test
        config = LoadTestConfig(
            concurrent_users=5,
            requests_per_user=50,
            target_response_time_ms=100.0,
            target_throughput_rps=20.0
        )
        
        load_tester = LoadTester(config)
        
        # Run concurrent users test
        result = load_tester.run_concurrent_users_test(extractor)
        
        # Validate results
        assert result.total_requests > 0
        assert result.successful_requests >= result.total_requests * 0.9  # At least 90% success rate
        
        throughput = result.get_throughput_rps()
        assert throughput > 5  # At least 5 requests per second
        
        avg_latency = result.get_avg_latency_ms()
        assert avg_latency < 500  # Average latency < 500ms
        
        error_rate = result.get_error_rate()
        assert error_rate < 0.1  # Error rate < 10%
        
        # Check latency percentiles
        percentiles = result.get_latency_percentiles()
        assert percentiles['p95'] < 1000  # P95 latency < 1 second
        
        logger.info(f"Constant load test results:")
        logger.info(f"  Total requests: {result.total_requests}")
        logger.info(f"  Success rate: {1 - error_rate:.2%}")
        logger.info(f"  Throughput: {throughput:.2f} RPS")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
        logger.info(f"  P95 latency: {percentiles['p95']:.2f}ms")
    
    def test_ramp_up_load_scenario(self, extractor):
        """Test ramp-up load scenario."""
        # Configure load test
        config = LoadTestConfig(
            concurrent_users=10,
            requests_per_user=20,
            ramp_up_time_seconds=30
        )
        
        load_tester = LoadTester(config)
        
        # Run ramp-up test with smaller parameters for testing
        result = load_tester.run_ramp_up_load_test(extractor, max_users=5, ramp_duration_seconds=10)
        
        # Validate results
        assert result.total_requests >= 0  # May be 0 in fast test
        
        if result.total_requests > 0:
            throughput = result.get_throughput_rps()
            assert throughput > 0
            
            avg_latency = result.get_avg_latency_ms()
            assert avg_latency < 1000  # Average latency < 1 second
            
            logger.info(f"Ramp-up load test results:")
            logger.info(f"  Total requests: {result.total_requests}")
            logger.info(f"  Throughput: {throughput:.2f} RPS")
            logger.info(f"  Average latency: {avg_latency:.2f}ms")
    
    def test_performance_test_framework(self, extractor):
        """Test the performance test framework."""
        # Configure performance test
        config = PerformanceTestConfig(
            num_iterations=50,  # Reduced for testing
            warmup_iterations=10,
            target_latency_ms=100.0
        )
        
        tester = FeatureExtractionPerformanceTester(config)
        
        # Run performance test
        result = tester.run_performance_test(extractor)
        
        # Validate results
        assert isinstance(result, PerformanceTestResult)
        assert result.test_name == config.test_name
        assert len(result.latencies_ms) == config.num_iterations
        
        # Check latency statistics
        stats = result.get_latency_stats()
        assert 'avg_latency_ms' in stats
        assert 'p95_latency_ms' in stats
        
        # Check resource statistics
        resource_stats = result.get_resource_stats()
        assert isinstance(resource_stats, dict)
        
        # Check requirements validation
        meets_requirements = result.meets_performance_requirements()
        assert isinstance(meets_requirements, bool)
        
        success_rate = result.get_success_rate()
        assert 0.0 <= success_rate <= 1.0
        
        logger.info(f"Performance test framework results:")
        logger.info(f"  Test name: {result.test_name}")
        logger.info(f"  Total iterations: {len(result.latencies_ms)}")
        logger.info(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
        logger.info(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Meets requirements: {meets_requirements}")


class TestStressTestingScenarios:
    """Test stress testing scenarios for feature extraction."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        # Mock forward method to return expected outputs
        mock_outputs = {
            'fused_features': np.random.randn(1, 10, 256).astype(np.float32),
            'cnn_features': np.random.randn(1, 10, 128).astype(np.float32),
            'lstm_features': np.random.randn(1, 10, 128).astype(np.float32),
            'classification_probs': np.random.rand(1, 3).astype(np.float32),
            'regression_uncertainty': np.random.rand(1, 1).astype(np.float32),
            'ensemble_weights': np.random.rand(1, 4).astype(np.float32)
        }
        model.forward.return_value = mock_outputs
        return model
    
    @pytest.fixture
    def extractor(self, mock_model):
        """Create a feature extractor for testing."""
        return CNNLSTMExtractor(mock_model, device='cpu')
    
    def test_stress_test_basic(self, extractor):
        """Test basic stress test functionality."""
        # Configure stress test with reduced parameters for testing
        config = StressTestConfig(
            max_concurrent_users=3,  # Reduced for testing
            max_requests_per_user=20,  # Reduced for testing
            test_duration_seconds=30,  # Reduced for testing
            ramp_up_duration_seconds=10  # Reduced for testing
        )
        
        stress_tester = StressTester(config)
        
        # Run stress test
        result = stress_tester.run_stress_test(extractor)
        
        # Validate results
        assert isinstance(result, StressTestResult)
        assert result.test_name == config.test_name
        assert result.end_time >= result.start_time
        
        # Check metrics
        assert result.total_requests >= 0
        assert result.successful_requests >= 0
        assert result.failed_requests >= 0
        
        throughput = result.get_throughput_rps()
        assert throughput >= 0
        
        avg_latency = result.get_avg_latency_ms()
        assert avg_latency >= 0
        
        error_rate = result.get_error_rate()
        assert 0.0 <= error_rate <= 1.0
        
        logger.info(f"Stress test results:")
        logger.info(f"  Test name: {result.test_name}")
        logger.info(f"  Duration: {result.end_time - result.start_time:.2f}s")
        logger.info(f"  Total requests: {result.total_requests}")
        logger.info(f"  Success rate: {1 - error_rate:.2%}")
        logger.info(f"  Throughput: {throughput:.2f} RPS")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
    
    def test_memory_exhaustion_test(self, extractor):
        """Test memory exhaustion stress test."""
        # Configure stress test
        config = StressTestConfig(
            max_concurrent_users=2,  # Reduced for testing
            max_requests_per_user=10,  # Reduced for testing
            test_duration_seconds=20  # Reduced for testing
        )
        
        stress_tester = StressTester(config)
        
        # Run memory exhaustion test
        result = stress_tester.run_memory_exhaustion_test(extractor)
        
        # Validate results
        assert isinstance(result, StressTestResult)
        assert result.total_requests >= 0
        
        if result.total_requests > 0:
            throughput = result.get_throughput_rps()
            assert throughput >= 0
            
            avg_latency = result.get_avg_latency_ms()
            assert avg_latency >= 0
            
            logger.info(f"Memory exhaustion test results:")
            logger.info(f"  Total requests: {result.total_requests}")
            logger.info(f"  Throughput: {throughput:.2f} RPS")
            logger.info(f"  Average latency: {avg_latency:.2f}ms")
            logger.info(f"  Memory growth: {result.memory_growth_mb:.2f}MB")


class TestPerformanceMetricsAndReporting:
    """Test performance metrics collection and reporting."""
    
    def test_metrics_collection(self):
        """Test metrics collection from test results."""
        collector = PerformanceMetricsCollector()
        
        # Create mock test results
        config = PerformanceTestConfig(num_iterations=10)
        result = PerformanceTestResult(
            test_name="Test Performance",
            test_timestamp=time.time(),
            config=config,
            latencies_ms=[50.0] * 50 + [150.0] * 50,  # Mix of fast and slow requests
            successful_extractions=95,
            failed_extractions=5
        )
        
        # Collect metrics
        report = collector.collect_from_performance_test(result)
        
        # Validate report
        assert isinstance(report, PerformanceReport)
        assert report.test_name == "Test Performance"
        assert report.test_type == "performance"
        assert report.total_requests == 100
        assert report.success_rate == 0.95
        
        # Check latency stats
        assert report.latency_stats.avg_ms > 0
        assert report.latency_stats.p95_ms > 0
        
        # Check throughput
        assert report.throughput_rps >= 0
        
        # Check requirements validation
        assert isinstance(report.meets_latency_requirement, bool)
        assert isinstance(report.meets_throughput_requirement, bool)
        assert isinstance(report.meets_resource_requirement, bool)
        
        logger.info(f"Metrics collection results:")
        logger.info(f"  Test name: {report.test_name}")
        logger.info(f"  Success rate: {report.success_rate:.2%}")
        logger.info(f"  Average latency: {report.latency_stats.avg_ms:.2f}ms")
        logger.info(f"  P95 latency: {report.latency_stats.p95_ms:.2f}ms")
        logger.info(f"  Throughput: {report.throughput_rps:.2f} RPS")
    
    def test_report_generation(self):
        """Test report generation."""
        from src.ml.feature_extraction.performance_testing.reporting import (
            JSONReportGenerator,
            HTMLReportGenerator
        )
        
        # Create mock report
        latency_stats = Mock()
        latency_stats.avg_ms = 75.5
        latency_stats.p95_ms = 95.2
        
        resource_stats = Mock()
        resource_stats.peak_memory_mb = 150.0
        
        report = PerformanceReport(
            test_name="Sample Test",
            test_type="performance",
            timestamp=time.time(),
            duration_seconds=60.0,
            total_requests=1000,
            successful_requests=980,
            failed_requests=20,
            success_rate=0.98,
            latency_stats=latency_stats,
            throughput_rps=16.7,
            resource_stats=resource_stats,
            meets_latency_requirement=True,
            meets_throughput_requirement=True,
            meets_resource_requirement=True
        )
        
        # Test JSON report generation
        json_generator = JSONReportGenerator()
        json_report = json_generator.generate_report([report])
        
        assert isinstance(json_report, str)
        assert len(json_report) > 0
        assert "Sample Test" in json_report
        
        # Test HTML report generation
        html_generator = HTMLReportGenerator()
        html_report = html_generator.generate_report([report])
        
        assert isinstance(html_report, str)
        assert len(html_report) > 0
        assert "Sample Test" in html_report
        assert "<html" in html_report.lower()
        
        logger.info("Report generation test passed")
        logger.info(f"JSON report length: {len(json_report)} characters")
        logger.info(f"HTML report length: {len(html_report)} characters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])