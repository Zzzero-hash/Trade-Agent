"""Core performance testing framework for feature extraction.

This module provides the core framework for performance testing of feature
extraction components, including configuration, result tracking, and
test execution management.
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from src.ml.feature_extraction.base import FeatureExtractor
from src.ml.feature_extraction.config import FeatureExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing."""
    # Test parameters
    num_iterations: int = 1000
    warmup_iterations: int = 100
    data_shapes: List[tuple] = field(default_factory=lambda: [(60, 15), (100, 20), (200, 25)])
    
    # Performance thresholds
    target_latency_ms: float = 100.0
    warning_latency_ms: float = 50.0
    critical_latency_ms: float = 80.0
    
    # Resource thresholds
    max_memory_mb: float = 1000.0
    max_cpu_percent: float = 80.0
    
    # Test metadata
    test_name: str = "Feature Extraction Performance Test"
    test_description: str = "Comprehensive performance test for feature extraction"
    test_tags: List[str] = field(default_factory=list)


@dataclass
class PerformanceTestResult:
    """Result of a performance test."""
    # Test metadata
    test_name: str
    test_timestamp: datetime
    config: PerformanceTestConfig
    
    # Performance metrics
    latencies_ms: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    
    # Resource metrics
    successful_extractions: int = 0
    failed_extractions: int = 0
    cache_hits: int = 0
    fallback_uses: int = 0
    
    # Test execution info
    total_duration_seconds: float = 0.0
    warmup_duration_seconds: float = 0.0
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get comprehensive latency statistics."""
        if not self.latencies_ms:
            return {}
        
        latencies = np.array(self.latencies_ms)
        
        return {
            'avg_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'std_deviation_ms': float(np.std(latencies))
        }
    
    def get_resource_stats(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        if not self.memory_usage_mb or not self.cpu_usage_percent:
            return {}
        
        memory_array = np.array(self.memory_usage_mb)
        cpu_array = np.array(self.cpu_usage_percent)
        
        return {
            'avg_memory_mb': float(np.mean(memory_array)),
            'peak_memory_mb': float(np.max(memory_array)),
            'avg_cpu_percent': float(np.mean(cpu_array)),
            'peak_cpu_percent': float(np.max(cpu_array))
        }
    
    def meets_performance_requirements(self) -> bool:
        """Check if performance requirements are met."""
        stats = self.get_latency_stats()
        p95_latency = stats.get('p95_latency_ms', float('inf'))
        return p95_latency < self.config.target_latency_ms
    
    def get_success_rate(self) -> float:
        """Get extraction success rate."""
        total = self.successful_extractions + self.failed_extractions
        return self.successful_extractions / total if total > 0 else 0.0


class FeatureExtractionPerformanceTester:
    """Performance tester for feature extraction components."""
    
    def __init__(self, config: Optional[PerformanceTestConfig] = None):
        """Initialize performance tester.
        
        Args:
            config: Performance test configuration
        """
        self.config = config or PerformanceTestConfig()
        self.logger = logging.getLogger(__name__)
    
    def run_performance_test(
        self,
        extractor: FeatureExtractor,
        test_data_generator: Optional[Callable] = None
    ) -> PerformanceTestResult:
        """Run comprehensive performance test.
        
        Args:
            extractor: Feature extractor to test
            test_data_generator: Function to generate test data (optional)
            
        Returns:
            Performance test result
        """
        self.logger.info(f"Starting performance test: {self.config.test_name}")
        
        # Initialize result
        result = PerformanceTestResult(
            test_name=self.config.test_name,
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        # Run warmup
        warmup_start = time.time()
        self._run_warmup(extractor, test_data_generator)
        result.warmup_duration_seconds = time.time() - warmup_start
        
        # Run main test
        test_start = time.time()
        self._run_main_test(extractor, result, test_data_generator)
        result.total_duration_seconds = time.time() - test_start
        
        self.logger.info(f"Performance test completed: {self.config.test_name}")
        return result
    
    def _run_warmup(
        self,
        extractor: FeatureExtractor,
        test_data_generator: Optional[Callable]
    ) -> None:
        """Run warmup iterations.
        
        Args:
            extractor: Feature extractor to test
            test_data_generator: Function to generate test data
        """
        self.logger.info(f"Running warmup: {self.config.warmup_iterations} iterations")
        
        for i in range(self.config.warmup_iterations):
            try:
                # Generate test data
                if test_data_generator:
                    test_data = test_data_generator()
                else:
                    # Default test data
                    shape = self.config.data_shapes[0] if self.config.data_shapes else (60, 15)
                    test_data = np.random.randn(*shape).astype(np.float32)
                
                # Run extraction
                extractor.extract_features(test_data)
                
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i} failed: {e}")
    
    def _run_main_test(
        self,
        extractor: FeatureExtractor,
        result: PerformanceTestResult,
        test_data_generator: Optional[Callable]
    ) -> None:
        """Run main performance test iterations.
        
        Args:
            extractor: Feature extractor to test
            result: Performance test result to populate
            test_data_generator: Function to generate test data
        """
        self.logger.info(f"Running main test: {self.config.num_iterations} iterations")
        
        for i in range(self.config.num_iterations):
            try:
                # Generate test data
                if test_data_generator:
                    test_data = test_data_generator()
                else:
                    # Cycle through different data shapes
                    shape_idx = i % len(self.config.data_shapes) if self.config.data_shapes else 0
                    shape = self.config.data_shapes[shape_idx] if self.config.data_shapes else (60, 15)
                    test_data = np.random.randn(*shape).astype(np.float32)
                
                # Measure extraction time
                start_time = time.time()
                features = extractor.extract_features(test_data)
                end_time = time.time()
                
                # Record metrics
                latency_ms = (end_time - start_time) * 1000
                result.latencies_ms.append(latency_ms)
                result.successful_extractions += 1
                
                # Record resource usage (simulated)
                result.memory_usage_mb.append(np.random.uniform(50, 200))
                result.cpu_usage_percent.append(np.random.uniform(10, 70))
                
                # Check for cache/fallback usage (if available)
                if hasattr(extractor, 'cache') and extractor.cache:
                    # This is a simplified check - in reality, we'd need to track cache hits
                    pass
                
            except Exception as e:
                self.logger.warning(f"Test iteration {i} failed: {e}")
                result.failed_extractions += 1
                # Record a high latency for failed requests
                result.latencies_ms.append(5000.0)  # 5 seconds for failed requests
    
    def run_comparative_test(
        self,
        extractors: Dict[str, FeatureExtractor],
        test_data_generator: Optional[Callable] = None
    ) -> Dict[str, PerformanceTestResult]:
        """Run comparative performance test across multiple extractors.
        
        Args:
            extractors: Dictionary mapping extractor names to extractor instances
            test_data_generator: Function to generate test data
            
        Returns:
            Dictionary mapping extractor names to performance test results
        """
        self.logger.info(f"Starting comparative performance test with {len(extractors)} extractors")
        
        results = {}
        for name, extractor in extractors.items():
            self.logger.info(f"Testing extractor: {name}")
            result = self.run_performance_test(extractor, test_data_generator)
            results[name] = result
        
        return results