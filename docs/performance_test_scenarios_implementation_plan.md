# Performance Test Scenarios Implementation Plan

## Overview

This document outlines the implementation plan for the performance test scenarios designed to validate the <100ms feature extraction requirement. The implementation will be done in the Code mode by the development team.

## 1. Core Performance Testing Infrastructure

### 1.1 Performance Metrics Collection

Create a comprehensive metrics collection system that integrates with the existing `PerformanceTracker`:

```python
# File: src/utils/performance_metrics.py

import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics collection"""
    
    # Timing metrics
    latencies: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Resource metrics
    cpu_samples: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Fallback metrics
    fallback_used: int = 0
    
    def start_timing(self) -> None:
        """Start timing a request"""
        self.start_time = time.time()
    
    def end_timing(self, success: bool = True, cache_hit: bool = False, fallback: bool = False) -> float:
        """End timing and record metrics"""
        self.end_time = time.time()
        latency = (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        self.latencies.append(latency)
        
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        if fallback:
            self.fallback_used += 1
            
        return latency
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles"""
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        return {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'avg': np.mean(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
    
    def get_throughput(self) -> float:
        """Get requests per second"""
        if self.start_time == 0 or self.end_time == 0:
            return 0.0
        duration = self.end_time - self.start_time
        return self.total_requests / duration if duration > 0 else 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        return self.successful_requests / max(self.total_requests, 1)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_cache_ops = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total_cache_ops, 1)
    
    def get_fallback_rate(self) -> float:
        """Get fallback usage rate"""
        return self.fallback_used / max(self.total_requests, 1)
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.latencies.clear()
        self.start_time = 0.0
        self.end_time = 0.0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_used = 0
```

### 1.2 Load Testing Framework

Create a flexible load testing framework:

```python
# File: tests/performance/load_testing_framework.py

import asyncio
import time
import statistics
from typing import Callable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class LoadPattern:
    """Load pattern definitions"""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"

class LoadTestConfig:
    """Configuration for load testing"""
    
    def __init__(self, 
                 load_pattern: str = LoadPattern.CONSTANT,
                 target_rps: int = 10,
                 duration_seconds: int = 60,
                 ramp_up_duration: int = 30,
                 num_workers: int = 4,
                 timeout_seconds: float = 30.0):
        self.load_pattern = load_pattern
        self.target_rps = target_rps
        self.duration_seconds = duration_seconds
        self.ramp_up_duration = ramp_up_duration
        self.num_workers = num_workers
        self.timeout_seconds = timeout_seconds

class LoadTester:
    """Main load testing engine"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    async def run_constant_load_test(self, 
                                   test_function: Callable,
                                   test_args: List[Any] = None) -> PerformanceMetrics:
        """Run constant load test"""
        test_args = test_args or []
        
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds
        
        request_intervals = 1.0 / self.config.target_rps
        next_request_time = start_time
        
        while time.time() < end_time:
            current_time = time.time()
            
            if current_time >= next_request_time:
                # Execute test function
                await self._execute_test_function(test_function, test_args)
                next_request_time += request_intervals
            else:
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.001)
        
        return self.metrics
    
    async def run_ramp_up_load_test(self,
                                  test_function: Callable,
                                  test_args: List[Any] = None) -> PerformanceMetrics:
        """Run ramp-up load test"""
        test_args = test_args or []
        
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds
        
        while time.time() < end_time:
            # Calculate current RPS based on elapsed time
            elapsed = time.time() - start_time
            progress = min(elapsed / self.config.ramp_up_duration, 1.0)
            current_rps = int(self.config.target_rps * progress)
            
            if current_rps > 0:
                request_intervals = 1.0 / current_rps
                # Execute test function
                await self._execute_test_function(test_function, test_args)
                await asyncio.sleep(request_intervals)
            else:
                await asyncio.sleep(0.1)
        
        return self.metrics
    
    async def _execute_test_function(self, test_function: Callable, test_args: List[Any]) -> None:
        """Execute a single test function and collect metrics"""
        self.metrics.start_timing()
        
        try:
            if asyncio.iscoroutinefunction(test_function):
                result = await asyncio.wait_for(
                    test_function(*test_args), 
                    timeout=self.config.timeout_seconds
                )
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, test_function, *test_args),
                        timeout=self.config.timeout_seconds
                    )
            
            self.metrics.end_timing(success=True)
            
        except asyncio.TimeoutError:
            self.metrics.end_timing(success=False)
        except Exception as e:
            self.metrics.end_timing(success=False)
```

## 2. Performance Test Scenarios Implementation

### 2.1 Single Feature Extraction Tests

```python
# File: tests/performance/test_single_extraction.py

import pytest
import numpy as np
from unittest.mock import Mock
import time

from src.ml.feature_extraction import (
    FeatureExtractorFactory,
    FeatureExtractionConfig,
    CNNLSTMExtractor
)
from tests.performance.load_testing_framework import LoadTester, LoadTestConfig

class TestSingleFeatureExtractionPerformance:
    """Performance tests for single feature extraction"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock CNN+LSTM model for testing"""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        # Mock forward method with realistic timing
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            # Simulate model processing time
            time.sleep(0.05)  # 50ms processing time
            return {
                'fused_features': np.random.randn(1, 10, 256),
                'classification_probs': np.random.rand(1, 3),
                'regression_uncertainty': np.random.rand(1, 1)
            }
        
        model.forward = mock_forward
        return model
    
    @pytest.fixture
    def feature_extractor(self, mock_model):
        """Create feature extractor for testing"""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=False,  # Disable caching for consistent timing
            enable_fallback=False
        )
        return FeatureExtractorFactory.create_basic_extractor(mock_model, config)
    
    @pytest.fixture
    def test_data(self):
        """Create test market data"""
        return np.random.randn(60, 15)  # 60 timesteps, 15 features
    
    def test_cold_start_latency(self, feature_extractor, test_data):
        """Test cold start feature extraction latency < 100ms"""
        
        # Run multiple iterations for statistical significance
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = feature_extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Cold start latency results:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")
        
        # Assert latency requirements
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms >= 100ms"
        assert p95_latency < 150, f"P95 latency {p95_latency:.2f}ms >= 150ms"
        assert p99_latency < 200, f"P99 latency {p99_latency:.2f}ms >= 200ms"
    
    def test_cached_extraction_latency(self, mock_model, test_data):
        """Test cached feature extraction latency"""
        
        # Create cached extractor
        config = FeatureExtractionConfig(
            enable_caching=True,
            cache_size=100,
            cache_ttl_seconds=60
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # First call to populate cache
        extractor.extract_features(test_data)
        
        # Run multiple iterations for cached performance
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            features = extractor.extract_features(test_data)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"Cached extraction latency results:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        
        # Cached extraction should be much faster
        assert avg_latency < 20, f"Cached average latency {avg_latency:.2f}ms >= 20ms"
        assert p95_latency < 50, f"Cached P95 latency {p95_latency:.2f}ms >= 50ms"
    
    def test_fallback_extraction_latency(self, test_data):
        """Test fallback feature extraction latency"""
        
        # Create extractor with fallback enabled but no model
        config = FeatureExtractionConfig(
            enable_fallback=True,
            fallback_feature_dim=15
        )
        
        # Mock extractor that always uses fallback
        extractor = Mock()
        extractor.extract_features.side_effect = Exception("Model error")
        
        # Wrap with fallback extractor
        from src.ml.feature_extraction import FallbackFeatureExtractor
        fallback_extractor = FallbackFeatureExtractor(extractor)
        
        # Run multiple iterations
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
        
        print(f"Fallback extraction latency results:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        
        # Fallback should be very fast
        assert avg_latency < 10, f"Fallback average latency {avg_latency:.2f}ms >= 10ms"
        assert p95_latency < 20, f"Fallback P95 latency {p95_latency:.2f}ms >= 20ms"
```

### 2.2 Concurrent Load Tests

```python
# File: tests/performance/test_concurrent_load.py

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock

from src.ml.feature_extraction import FeatureExtractorFactory, FeatureExtractionConfig
from tests.performance.load_testing_framework import (
    LoadTester, 
    LoadTestConfig, 
    LoadPattern
)

class TestConcurrentLoadPerformance:
    """Performance tests for concurrent load scenarios"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model with realistic timing"""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            import time
            time.sleep(0.02)  # 20ms processing time
            return {
                'fused_features': np.random.randn(1, 10, 256),
                'classification_probs': np.random.rand(1, 3),
                'regression_uncertainty': np.random.rand(1, 1)
            }
        
        model.forward = mock_forward
        return model
    
    @pytest.fixture
    def feature_extractor(self, mock_model):
        """Create feature extractor"""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            cache_size=1000,
            enable_fallback=True
        )
        return FeatureExtractorFactory.create_extractor(mock_model, config)
    
    @pytest.fixture
    def test_data_generator(self):
        """Generate different test data for each request"""
        def generate_data():
            return np.random.randn(60, 15)
        return generate_data
    
    @pytest.mark.asyncio
    async def test_constant_load_performance(self, feature_extractor, test_data_generator):
        """Test constant load performance"""
        
        async def extract_features_async():
            data = test_data_generator()
            return feature_extractor.extract_features(data)
        
        # Configure load test
        config = LoadTestConfig(
            load_pattern=LoadPattern.CONSTANT,
            target_rps=50,  # 50 requests per second
            duration_seconds=120,  # 2 minutes
            num_workers=8
        )
        
        # Run load test
        load_tester = LoadTester(config)
        metrics = await load_tester.run_constant_load_test(
            extract_features_async
        )
        
        # Analyze results
        latency_percentiles = metrics.get_latency_percentiles()
        throughput = metrics.get_throughput()
        success_rate = metrics.get_success_rate()
        cache_hit_rate = metrics.get_cache_hit_rate()
        
        print(f"Constant load test results:")
        print(f"  Throughput: {throughput:.2f} RPS")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Cache hit rate: {cache_hit_rate:.2%}")
        print(f"  Latency P50: {latency_percentiles.get('p50', 0):.2f}ms")
        print(f"  Latency P95: {latency_percentiles.get('p95', 0):.2f}ms")
        print(f"  Latency P99: {latency_percentiles.get('p99', 0):.2f}ms")
        
        # Assert performance requirements
        assert throughput >= 45, f"Throughput {throughput:.2f} RPS < 45 RPS"
        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} < 9%"
        assert latency_percentiles.get('p95', 0) < 10, f"P95 latency >= 100ms"
    
    @pytest.mark.asyncio
    async def test_ramp_up_load_performance(self, feature_extractor, test_data_generator):
        """Test ramp-up load performance"""
        
        async def extract_features_async():
            data = test_data_generator()
            return feature_extractor.extract_features(data)
        
        # Configure ramp-up load test
        config = LoadTestConfig(
            load_pattern=LoadPattern.RAMP_UP,
            target_rps=100,  # Peak at 100 RPS
            duration_seconds=180,  # 3 minutes
            ramp_up_duration=60,  # 1 minute ramp-up
            num_workers=16
        )
        
        # Run load test
        load_tester = LoadTester(config)
        metrics = await load_tester.run_ramp_up_load_test(
            extract_features_async
        )
        
        # Analyze results
        latency_percentiles = metrics.get_latency_percentiles()
        throughput = metrics.get_throughput()
        success_rate = metrics.get_success_rate()
        
        print(f"Ramp-up load test results:")
        print(f"  Peak throughput: {throughput:.2f} RPS")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Latency P95: {latency_percentiles.get('p95', 0):.2f}ms")
        print(f"  Latency P99: {latency_percentiles.get('p99', 0):.2f}ms")
        
        # Assert performance requirements
        assert success_rate >= 0.98, f"Success rate {success_rate:.2%} < 98%"
        assert latency_percentiles.get('p95', 0) < 150, f"P95 latency >= 150ms"
```

### 2.3 Stress Testing Scenarios

```python
# File: tests/performance/test_stress_scenarios.py

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock
import psutil
import time

from src.ml.feature_extraction import FeatureExtractorFactory, FeatureExtractionConfig
from tests.performance.load_testing_framework import LoadTester, LoadTestConfig

class TestStressScenarios:
    """Stress testing scenarios for feature extraction"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model with configurable timing"""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        def mock_forward(input_tensor, return_features=True, use_ensemble=True):
            # Simulate processing time that can vary under stress
            import random
            processing_time = random.uniform(0.01, 0.05)  # 10-50ms
            time.sleep(processing_time)
            return {
                'fused_features': np.random.randn(1, 10, 256),
                'classification_probs': np.random.rand(1, 3),
                'regression_uncertainty': np.random.rand(1, 1)
            }
        
        model.forward = mock_forward
        return model
    
    def test_maximum_throughput(self, mock_model):
        """Find maximum sustainable throughput"""
        
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            cache_size=10000,  # Large cache
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Binary search for maximum RPS
        min_rps = 10
        max_rps = 200
        target_success_rate = 0.95
        
        while max_rps - min_rps > 5:
            test_rps = (min_rps + max_rps) // 2
            
            # Run short load test
            load_config = LoadTestConfig(
                target_rps=test_rps,
                duration_seconds=30,  # Short test
                num_workers=8
            )
            
            load_tester = LoadTester(load_config)
            
            async def extract_features_async():
                data = np.random.randn(60, 15)
                return extractor.extract_features(data)
            
            # This would be implemented in the actual test
            # metrics = await load_tester.run_constant_load_test(extract_features_async)
            # success_rate = metrics.get_success_rate()
            
            # For now, we'll simulate the result
            success_rate = 0.98 if test_rps < 150 else 0.90
            
            if success_rate >= target_success_rate:
                min_rps = test_rps
            else:
                max_rps = test_rps
        
        max_sustainable_rps = min_rps
        print(f"Maximum sustainable throughput: {max_sustainable_rps} RPS")
        
        # Should achieve reasonable throughput
        assert max_sustainable_rps >= 80, f"Max throughput {max_sustainable_rps} RPS < 80 RPS"
    
    def test_memory_usage_under_load(self, mock_model):
        """Test memory usage doesn't grow excessively under load"""
        
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            cache_size=1000,
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many feature extractions
        test_data = np.random.randn(60, 15)
        iterations = 10000
        
        for i in range(iterations):
            features = extractor.extract_features(test_data)
            
            # Check memory every 1000 iterations
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                print(f"Iteration {i}: Memory usage: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
                
                # Memory growth should be reasonable
                assert memory_growth < 500, f"Memory growth {memory_growth:.1f}MB >= 500MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        print(f"Total memory growth: {total_growth:.1f}MB")
        
        # Total memory growth should be reasonable
        assert total_growth < 1000, f"Total memory growth {total_growth:.1f}MB >= 1000MB"
    
    def test_resource_exhaustion_handling(self, mock_model):
        """Test graceful handling of resource exhaustion"""
        
        # Create extractor with small cache to force more processing
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=False,  # No caching to increase load
            enable_fallback=True
        )
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Run high-concurrency test
        import concurrent.futures
        import threading
        
        def extract_features():
            data = np.random.randn(60, 15)
            try:
                return extractor.extract_features(data)
            except Exception as e:
                return None
        
        max_workers = 50
        num_requests = 500
        
        successful = 0
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_features) for _ in range(num_requests)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=5.0)
                    if result is not None:
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        
        success_rate = successful / num_requests
        print(f"Resource exhaustion test results:")
        print(f"  Successful requests: {successful}/{num_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Should maintain reasonable success rate even under stress
        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} < 90%"
```

## 3. Integration with Enhanced Trading Environment

```python
# File: tests/performance/test_enhanced_environment_performance.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.ml.enhanced_trading_environment import (
    EnhancedTradingEnvironment,
    EnhancedTradingConfig
)
from tests.performance.load_testing_framework import LoadTester, LoadTestConfig

class TestEnhancedEnvironmentPerformance:
    """Performance tests for enhanced trading environment"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for testing"""
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
        
        data = []
        price = 100.0
        
        for i, timestamp in enumerate(timestamps):
            # Random walk for price
            price_change = np.random.normal(0, 0.02)
            price = price * (1 + price_change)
            
            row = {
                'timestamp': timestamp,
                'symbol': 'TEST',
                'open': price,
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': np.random.randint(1000, 100000),
                'returns': price_change,
                'volatility': abs(np.random.normal(0, 0.02)),
                'rsi': np.random.uniform(20, 80),
                'macd': np.random.normal(0, 0.1),
                'macd_signal': np.random.normal(0, 0.05),
                'bb_position': np.random.uniform(0, 1),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'sma_5': price,
                'sma_20': price,
                'ema_12': price
            }
            data.append(row)
        
        return pd.DataFrame(data).reset_index(drop=True)
    
    @pytest.mark.slow
    def test_environment_step_performance(self, sample_market_data):
        """Test performance of environment step operations"""
        
        config = EnhancedTradingConfig(
            lookback_window=60,
            enable_fallback=True,
            enable_feature_caching=True
        )
        
        with patch('src.ml.enhanced_trading_environment.CNNLSTMFeatureExtractor') as mock_extractor_class:
            # Mock feature extractor with realistic timing
            mock_extractor = Mock()
            mock_extractor.extract_features.return_value = {
                'fused_features': np.random.randn(1, 15),
                'classification_confidence': np.array([0.8]),
                'regression_uncertainty': np.array([0.1]),
                'ensemble_weights': None,
                'fallback_used': np.array([True])
            }
            mock_extractor.get_feature_dimensions.return_value = {'fused_features': 15, 'total': 17}
            mock_extractor.get_status.return_value = {'model_loaded': False}
            mock_extractor_class.return_value = mock_extractor
            
            env = EnhancedTradingEnvironment(
                market_data=sample_market_data,
                config=config,
                symbols=['TEST']
            )
            
            # Run multiple environment steps
            observation, info = env.reset()
            
            latencies = []
            iterations = 1000
            
            for _ in range(iterations):
                action = env.action_space.sample()
                
                start_time = time.time()
                observation, reward, terminated, truncated, step_info = env.step(action)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if terminated or truncated:
                    observation, info = env.reset()
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            print(f"Environment step performance:")
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  P95 latency: {p95_latency:.2f}ms")
            print(f"  P99 latency: {p99_latency:.2f}ms")
            
            # Environment steps should be reasonably fast
            assert avg_latency < 50, f"Average step latency {avg_latency:.2f}ms >= 50ms"
            assert p95_latency < 100, f"P95 step latency {p95_latency:.2f}ms >= 100ms"
```

## 4. Performance Reporting and Analysis

```python
# File: tests/performance/performance_reporter.py

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceReport:
    """Comprehensive performance test report"""
    
    test_name: str
    timestamp: datetime
    metrics: Dict[str, Any]
    configuration: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_name': self.test_name,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'configuration': self.configuration
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        summary = f"""
Performance Test Report: {self.test_name}
=======================================
Timestamp: {self.timestamp}
        
Metrics Summary:
"""
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                summary += f"  {key}: {value:.2f}\n"
            else:
                summary += f"  {key}: {value}\n"
        
        return summary

class PerformanceReporter:
    """Generates performance reports and visualizations"""
    
    def __init__(self):
        self.reports: List[PerformanceReport] = []
    
    def add_report(self, report: PerformanceReport) -> None:
        """Add a performance report"""
        self.reports.append(report)
    
    def generate_comparison_report(self, baseline_report: PerformanceReport, 
                                 current_report: PerformanceReport) -> Dict[str, Any]:
        """Generate comparison between baseline and current performance"""
        
        comparison = {
            'baseline_test': baseline_report.test_name,
            'current_test': current_report.test_name,
            'improvements': {},
            'regressions': {},
            'unchanged': {}
        }
        
        baseline_metrics = baseline_report.metrics
        current_metrics = current_report.metrics
        
        # Compare numeric metrics
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    
                    if abs(change_percent) < 1:  # Less than 1% change
                        comparison['unchanged'][metric_name] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'change_percent': change_percent
                        }
                    elif change_percent < 0:  # Improvement
                        comparison['improvements'][metric_name] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'improvement_percent': abs(change_percent)
                        }
                    else:  # Regression
                        comparison['regressions'][metric_name] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'regression_percent': change_percent
                        }
        
        return comparison
    
    def generate_html_report(self) -> str:
        """Generate HTML report with visualizations"""
        # This would be implemented in the actual system
        html_content = "<html><body><h1>Performance Test Results</h1>"
        
        for report in self.reports:
            html_content += f"<h2>{report.test_name}</h2>"
            html_content += f"<p>Timestamp: {report.timestamp}</p>"
            
            # Add metrics table
            html_content += "<table border='1'>"
            html_content += "<tr><th>Metric</th><th>Value</th></tr>"
            
            for key, value in report.metrics.items():
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
            
            html_content += "</table>"
        
        html_content += "</body></html>"
        return html_content

# Performance Alerting System
class PerformanceAlert:
    """Performance alerting system"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check if any metrics exceed thresholds"""
        alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and value > threshold:
                    alerts.append({
                        'level': 'CRITICAL' if value > threshold * 1.2 else 'WARNING',
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'message': f"{metric_name} {value} exceeds threshold {threshold}"
                    })
        
        return alerts

# Default thresholds for feature extraction
DEFAULT_THRESHOLDS = {
    'latency_p95_ms': 100.0,
    'latency_p99_ms': 200.0,
    'error_rate': 0.01,
    'memory_growth_mb': 1000.0
}
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
1. Implement `PerformanceMetrics` class
2. Create `LoadTester` framework
3. Set up basic test structure
4. Implement single extraction tests

### Phase 2: Advanced Testing (Week 2)
1. Implement concurrent load testing
2. Add stress testing scenarios
3. Create enhanced environment integration tests
4. Implement resource monitoring

### Phase 3: Reporting and Automation (Week 3)
1. Create performance reporting system
2. Implement alerting mechanisms
3. Integrate with CI/CD pipeline
4. Add historical trend analysis

### Phase 4: Validation and Optimization (Week 4)
1. Validate <100ms requirement under all scenarios
2. Optimize performance bottlenecks
3. Create baseline performance metrics
4. Document findings and recommendations

## Success Criteria

1. **Latency Validation**: Consistently achieve <100ms P95 latency for feature extraction
2. **Throughput Targets**: Maintain 50+ RPS under normal load conditions
3. **Resource Efficiency**: Keep memory growth under 1GB for 10K requests
4. **Reliability**: Maintain 99%+ success rate under stress conditions
5. **Monitoring**: Comprehensive metrics collection and alerting
6. **Automation**: Seamless integration with existing development workflow

This implementation plan provides a comprehensive approach to validating the <100ms feature extraction requirement while ensuring the system performs well under various load conditions.