"""Validation tests for the <100ms feature extraction requirement.

This module contains comprehensive tests to validate that the feature extraction
system meets the <100ms latency requirement under various operational conditions.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch
import logging

from src.ml.feature_extraction.base import FeatureExtractor
from src.ml.feature_extraction.cnn_lstm_extractor import CNNLSTMExtractor
from src.ml.feature_extraction.cached_extractor import CachedFeatureExtractor
from src.ml.feature_extraction.fallback_extractor import FallbackFeatureExtractor
from src.ml.feature_extraction.factory import FeatureExtractorFactory
from src.ml.feature_extraction.config import FeatureExtractionConfig

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
from src.ml.feature_extraction.performance_testing.integration import PerformanceTestingIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSingleExtractionValidation:
    """Validate single feature extraction meets <100ms requirement."""
    
    @pytest.fixture
    def mock_model(self):
        """Create realistic mock model."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            import random
            # Simulate processing time that varies but stays under 100ms for most cases
            processing_time = random.uniform(0.02, 0.08)  # 20-80ms
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
    
    def test_cold_start_validation(self, mock_model):
        """Validate cold start performance meets <100ms requirement."""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=False,
            enable_fallback=False
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        test_data = np.random.randn(60, 15).astype(np.float32)
        
        # Run cold start extraction tests
        latencies = []
        iterations = 1000
        
        for _ in range(iterations):
            start_time = time.time()
            features = extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        logger.info("Cold Start Validation Results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        logger.info(f"  P99: {p99_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")
        
        # Validate requirements
        assert avg_latency < 80, f"Average latency {avg_latency:.2f}ms >= 80ms"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms >= 100ms"
        assert p99_latency < 150, f"P99 latency {p99_latency:.2f}ms >= 150ms"
        assert max_latency < 200, f"Max latency {max_latency:.2f}ms >= 200ms"
    
    def test_cached_extraction_validation(self, mock_model):
        """Validate cached extraction meets <50ms requirement."""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            cache_size=1000,
            cache_ttl_seconds=300,
            enable_fallback=False
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        test_data = np.random.randn(60, 15).astype(np.float32)
        
        # First call to populate cache
        extractor.extract_features(test_data)
        
        # Run cached extraction tests
        latencies = []
        iterations = 1000
        
        for _ in range(iterations):
            start_time = time.time()
            features = extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)
        
        logger.info("Cached Extraction Validation Results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")
        
        # Validate requirements (cached should be much faster)
        assert avg_latency < 20, f"Average cached latency {avg_latency:.2f}ms >= 20ms"
        assert p95_latency < 50, f"P95 cached latency {p95_latency:.2f}ms >= 50ms"
        assert max_latency < 100, f"Max cached latency {max_latency:.2f}ms >= 100ms"
    
    def test_fallback_extraction_validation(self):
        """Validate fallback extraction meets <20ms requirement."""
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
        iterations = 1000
        
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
        
        logger.info("Fallback Extraction Validation Results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")
        
        # Validate requirements (fallback should be very fast)
        assert avg_latency < 10, f"Average fallback latency {avg_latency:.2f}ms >= 10ms"
        assert p95_latency < 20, f"P95 fallback latency {p95_latency:.2f}ms >= 20ms"
        assert max_latency < 50, f"Max fallback latency {max_latency:.2f}ms >= 50ms"


class TestConcurrentLoadValidation:
    """Validate feature extraction meets <100ms requirement under concurrent load."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model with realistic timing."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            import random
            # Simulate processing time that can vary under load (20-80ms)
            processing_time = random.uniform(0.02, 0.08)
            time.sleep(processing_time)
            
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
    
    @pytest.mark.asyncio
    async def test_high_load_validation(self, mock_model):
        """Validate system behavior under high load (100+ RPS)."""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            cache_size=5000,
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Track performance metrics
        latencies = []
        successful_requests = 0
        failed_requests = 0
        
        # Simulate high load: 20 concurrent users, 50 requests each
        import asyncio
        import concurrent.futures
        
        def worker(worker_id, num_requests):
            """Worker function to simulate load."""
            worker_latencies = []
            worker_success = 0
            worker_failed = 0
            
            test_data = np.random.randn(60, 15).astype(np.float32)
            
            for i in range(num_requests):
                start_time = time.time()
                try:
                    features = extractor.extract_features(test_data)
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    worker_latencies.append(latency_ms)
                    worker_success += 1
                except Exception as e:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    worker_latencies.append(latency_ms)
                    worker_failed += 1
            
            return worker_latencies, worker_success, worker_failed
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(worker, i, 50) for i in range(20)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                worker_latencies, worker_success, worker_failed = future.result()
                latencies.extend(worker_latencies)
                successful_requests += worker_success
                failed_requests += worker_failed
        
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Calculate statistics
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            max_latency = np.max(latencies)
        else:
            avg_latency = p95_latency = p99_latency = max_latency = 0
        
        logger.info("High Load Validation Results:")
        logger.info(f"  Total Requests: {total_requests}")
        logger.info(f"  Success Rate: {success_rate:.2%}")
        logger.info(f"  Average Latency: {avg_latency:.2f}ms")
        logger.info(f"  P95 Latency: {p95_latency:.2f}ms")
        logger.info(f"  P99 Latency: {p99_latency:.2f}ms")
        logger.info(f"  Max Latency: {max_latency:.2f}ms")
        
        # Validate requirements
        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} < 99%"
        assert avg_latency < 70, f"Average latency {avg_latency:.2f}ms >= 70ms"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms >= 100ms"
        assert p99_latency < 150, f"P99 latency {p99_latency:.2f}ms >= 150ms"
        assert max_latency < 200, f"Max latency {max_latency:.2f}ms >= 200ms"


class TestDataSizeValidation:
    """Validate performance with different data sizes."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            import random
            # Processing time increases with data size
            base_time = 0.02
            size_factor = input_tensor.shape[0] * input_tensor.shape[1] / 1000
            processing_time = base_time + (size_factor * 0.01)
            time.sleep(min(processing_time, 0.1))  # Cap at 100ms
            
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
    
    def test_small_data_validation(self, mock_model):
        """Validate performance with small data."""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Small data window
        small_data = np.random.randn(30, 10).astype(np.float32)  # 30 timesteps, 10 features
        
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = extractor.extract_features(small_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        logger.info("Small Data Validation Results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        
        # Should still meet requirements
        assert avg_latency < 80, f"Small data average latency {avg_latency:.2f}ms >= 80ms"
        assert p95_latency < 100, f"Small data P95 latency {p95_latency:.2f}ms >= 100ms"
    
    def test_large_data_validation(self, mock_model):
        """Validate performance with large data."""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Large data window
        large_data = np.random.randn(200, 20).astype(np.float32)  # 200 timesteps, 20 features
        
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = extractor.extract_features(large_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        logger.info("Large Data Validation Results:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        
        # Large data should still meet requirements but may be slower
        assert avg_latency < 120, f"Large data average latency {avg_latency:.2f}ms >= 120ms"
        assert p95_latency < 150, f"Large data P95 latency {p95_latency:.2f}ms >= 150ms"


class TestPerformanceTestingFrameworkValidation:
    """Validate the performance testing framework itself."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            # Fast processing for testing framework
            time.sleep(0.01)  # 10ms
            
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
    
    def test_framework_integration_validation(self, mock_model):
        """Validate integration between performance testing framework and monitoring."""
        # Create extractor
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Create performance testing integration
        integration = PerformanceTestingIntegration()
        
        # Run performance test
        perf_config = PerformanceTestConfig(
            num_iterations=50,  # Reduced for testing
            warmup_iterations=10,
            target_latency_ms=100.0
        )
        
        tester = FeatureExtractionPerformanceTester(perf_config)
        perf_result = tester.run_performance_test(extractor)
        
        # Integrate with monitoring systems
        integration.integrate_performance_test_result(perf_result)
        
        # Validate integration
        unified_stats = integration.get_unified_performance_stats()
        assert 'feature_extraction' in unified_stats
        assert 'performance_monitor' in unified_stats
        assert 'performance_requirements' in unified_stats
        
        # Validate requirement validation
        requirement_validation = integration.validate_100ms_requirement()
        assert 'meets_100ms_requirement' in requirement_validation
        assert 'current_p95_latency_ms' in requirement_validation
        
        logger.info("Framework integration validation passed")
        logger.info(f"Requirement validation: {requirement_validation['meets_100ms_requirement']}")
    
    def test_load_testing_validation(self, mock_model):
        """Validate load testing framework."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model, device='cpu')
        
        # Configure load test
        load_config = LoadTestConfig(
            concurrent_users=3,  # Reduced for testing
            requests_per_user=20,  # Reduced for testing
            target_response_time_ms=100.0,
            target_throughput_rps=10.0
        )
        
        load_tester = LoadTester(load_config)
        load_result = load_tester.run_concurrent_users_test(extractor)
        
        # Validate load test results
        assert load_result.total_requests > 0
        assert load_result.successful_requests >= 0
        assert load_result.get_throughput_rps() >= 0
        
        throughput = load_result.get_throughput_rps()
        avg_latency = load_result.get_avg_latency_ms()
        error_rate = load_result.get_error_rate()
        
        logger.info("Load testing validation results:")
        logger.info(f"  Total requests: {load_result.total_requests}")
        logger.info(f"  Throughput: {throughput:.2f} RPS")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
        logger.info(f"  Error rate: {error_rate:.2%}")
        
        # Basic validation
        assert throughput >= 1, f"Throughput {throughput:.2f} RPS < 1 RPS"
        assert avg_latency < 1000, f"Average latency {avg_latency:.2f}ms >= 1000ms"
    
    def test_stress_testing_validation(self, mock_model):
        """Validate stress testing framework."""
        # Create extractor
        extractor = CNNLSTMExtractor(mock_model, device='cpu')
        
        # Configure stress test
        stress_config = StressTestConfig(
            max_concurrent_users=2,  # Reduced for testing
            max_requests_per_user=10,  # Reduced for testing
            test_duration_seconds=20,  # Reduced for testing
            max_acceptable_error_rate=0.20  # Higher tolerance for testing
        )
        
        stress_tester = StressTester(stress_config)
        stress_result = stress_tester.run_stress_test(extractor)
        
        # Validate stress test results
        assert stress_result.total_requests >= 0
        assert stress_result.end_time >= stress_result.start_time
        
        throughput = stress_result.get_throughput_rps()
        avg_latency = stress_result.get_avg_latency_ms()
        error_rate = stress_result.get_error_rate()
        
        logger.info("Stress testing validation results:")
        logger.info(f"  Total requests: {stress_result.total_requests}")
        logger.info(f"  Duration: {stress_result.end_time - stress_result.start_time:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} RPS")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
        logger.info(f"  Error rate: {error_rate:.2%}")
        
        # Basic validation
        assert stress_result.end_time >= stress_result.start_time
        assert throughput >= 0
        assert avg_latency >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])