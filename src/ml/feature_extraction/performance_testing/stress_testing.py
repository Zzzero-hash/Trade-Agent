"""Stress testing framework for feature extraction.

This module provides stress testing capabilities to validate feature extraction
performance under extreme load conditions and resource constraints.
"""

import time
import logging
import threading
import gc
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import psutil
import torch

from src.ml.feature_extraction.base import FeatureExtractor
from .load_testing import LoadTestConfig, LoadTestResult

logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    # Stress parameters
    max_concurrent_users: int = 100
    max_requests_per_user: int = 1000
    resource_exhaustion_test: bool = True
    memory_pressure_test: bool = True
    cpu_saturation_test: bool = True
    
    # Test duration
    test_duration_seconds: int = 600  # 10 minutes
    ramp_up_duration_seconds: int = 120 # 2 minutes
    
    # Failure thresholds
    max_acceptable_error_rate: float = 0.10  # 10%
    max_acceptable_latency_ms: float = 5000.0  # 5 seconds
    min_required_throughput_rps: float = 10.0
    
    # Resource thresholds
    critical_memory_mb: float = 3000.0
    critical_cpu_percent: float = 95.0
    
    # Test metadata
    test_name: str = "Feature Extraction Stress Test"
    test_description: str = "Stress test for feature extraction under extreme conditions"
    test_tags: List[str] = field(default_factory=list)


@dataclass
class StressTestResult:
    """Result of a stress test."""
    # Test metadata
    test_name: str
    test_timestamp: datetime
    config: StressTestConfig
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    
    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Detailed metrics
    request_latencies_ms: List[float] = field(default_factory=list)
    resource_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    memory_growth_mb: float = 0.0
    
    # Failure points
    failure_points: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_throughput_rps(self) -> float:
        """Get requests per second throughput."""
        if self.end_time > self.start_time:
            return self.total_requests / (self.end_time - self.start_time)
        return 0.0
    
    def get_avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if self.total_requests > 0:
            return self.total_latency_ms / self.total_requests
        return 0.0
    
    def get_error_rate(self) -> float:
        """Get error rate."""
        if self.total_requests > 0:
            return self.failed_requests / self.total_requests
        return 0.0
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles."""
        if not self.request_latencies_ms:
            return {}
        
        latencies = np.array(self.request_latencies_ms)
        
        return {
            'p50': float(np.percentile(latencies, 50)),
            'p90': float(np.percentile(latencies, 90)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies))
        }
    
    def meets_minimum_requirements(self) -> bool:
        """Check if stress test meets minimum requirements."""
        throughput = self.get_throughput_rps()
        error_rate = self.get_error_rate()
        avg_latency = self.get_avg_latency_ms()
        
        return (
            throughput >= self.config.min_required_throughput_rps and
            error_rate <= self.config.max_acceptable_error_rate and
            avg_latency <= self.config.max_acceptable_latency_ms
        )


class StressTester:
    """Stress tester for feature extraction components."""
    
    def __init__(self, config: Optional[StressTestConfig] = None):
        """Initialize stress tester.
        
        Args:
            config: Stress test configuration
        """
        self.config = config or StressTestConfig()
        self.logger = logging.getLogger(__name__)
    
    def run_stress_test(
        self,
        extractor: FeatureExtractor,
        test_data_generator: Optional[Callable] = None
    ) -> StressTestResult:
        """Run comprehensive stress test.
        
        Args:
            extractor: Feature extractor to test
            test_data_generator: Function to generate test data
            
        Returns:
            Stress test result
        """
        self.logger.info(f"Starting stress test: {self.config.test_name}")
        
        result = StressTestResult(
            test_name=self.config.test_name,
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        result.start_time = time.time()
        
        # Track initial resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Active users and metrics collection
        active_users = []
        resource_monitor_stop = threading.Event()
        
        def resource_monitor():
            """Monitor system resources during test."""
            while not resource_monitor_stop.is_set():
                try:
                    # Collect resource metrics
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_percent = process.cpu_percent()
                    
                    # System-wide metrics
                    system_cpu = psutil.cpu_percent()
                    system_memory = psutil.virtual_memory().percent
                    
                    resource_metrics = {
                        'timestamp': time.time(),
                        'process_memory_mb': current_memory,
                        'process_cpu_percent': cpu_percent,
                        'system_cpu_percent': system_cpu,
                        'system_memory_percent': system_memory
                    }
                    
                    result.resource_metrics.append(resource_metrics)
                    
                    # Check for critical resource usage
                    if current_memory > self.config.critical_memory_mb:
                        self.logger.warning(f"Critical memory usage: {current_memory:.1f}MB")
                        result.failure_points.append({
                            'type': 'memory_exceeded',
                            'timestamp': time.time(),
                            'memory_mb': current_memory
                        })
                    
                    if system_cpu > self.config.critical_cpu_percent:
                        self.logger.warning(f"Critical CPU usage: {system_cpu:.1f}%")
                        result.failure_points.append({
                            'type': 'cpu_saturated',
                            'timestamp': time.time(),
                            'cpu_percent': system_cpu
                        })
                    
                    # Update peak metrics
                    result.peak_memory_mb = max(result.peak_memory_mb, current_memory)
                    result.peak_cpu_percent = max(result.peak_cpu_percent, system_cpu)
                    
                    time.sleep(1)  # Check every second
                    
                except Exception as e:
                    self.logger.warning(f"Resource monitoring error: {e}")
        
        # Start resource monitoring
        resource_thread = threading.Thread(target=resource_monitor)
        resource_thread.start()
        
        def stress_user_worker(user_id: int) -> Dict[str, Any]:
            """Worker function for a stress test user."""
            user_metrics = {
                'user_id': user_id,
                'requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_latency_ms': 0.0,
                'latencies_ms': []
            }
            
            for i in range(self.config.max_requests_per_user):
                try:
                    # Generate test data
                    if test_data_generator:
                        test_data = test_data_generator()
                    else:
                        # Vary data size to increase stress
                        data_size = 60 + (user_id % 50)  # Vary window size
                        num_features = 15 + (user_id % 10)  # Vary feature count
                        test_data = np.random.randn(data_size, num_features).astype(np.float32)
                    
                    # Measure extraction time
                    start_time = time.time()
                    _ = extractor.extract_features(test_data)  # We don't use the features
                    end_time = time.time()
                    
                    # Record metrics
                    latency_ms = (end_time - start_time) * 1000
                    user_metrics['latencies_ms'].append(latency_ms)
                    user_metrics['total_latency_ms'] += latency_ms
                    user_metrics['successful_requests'] += 1
                    
                    # Periodic garbage collection to simulate real-world conditions
                    if i % 100 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                except Exception as e:
                    self.logger.warning(f"Stress user {user_id}, request {i} failed: {e}")
                    user_metrics['failed_requests'] += 1
                    # Record high latency for failed requests
                    user_metrics['latencies_ms'].append(5000.0)
                    user_metrics['total_latency_ms'] += 5000.0
                
                user_metrics['requests'] += 1
                
                # Check if test duration exceeded
                if time.time() - result.start_time > self.config.test_duration_seconds:
                    break
            
            return user_metrics
        
        try:
            # Ramp up users
            ramp_up_interval = self.config.ramp_up_duration_seconds / self.config.max_concurrent_users
            user_results = []
            
            for user_id in range(self.config.max_concurrent_users):
                # Start user worker
                user_thread = threading.Thread(
                    target=lambda uid=user_id: user_results.append(stress_user_worker(uid))
                )
                user_thread.start()
                active_users.append(user_thread)
                
                # Check if test duration exceeded
                if time.time() - result.start_time > self.config.test_duration_seconds:
                    self.logger.info("Test duration exceeded, stopping user ramp-up")
                    break
                
                # Wait for next user
                time.sleep(ramp_up_interval)
            
            # Wait for all users to complete or timeout
            test_end_time = result.start_time + self.config.test_duration_seconds
            for user_thread in active_users:
                remaining_time = max(0, test_end_time - time.time())
                user_thread.join(timeout=remaining_time)
            
        except Exception as e:
            self.logger.error(f"Stress test execution error: {e}")
        finally:
            # Stop resource monitoring
            resource_monitor_stop.set()
            resource_thread.join()
        
        # Aggregate results
        for user_result in user_results:
            result.total_requests += user_result['requests']
            result.successful_requests += user_result['successful_requests']
            result.failed_requests += user_result['failed_requests']
            result.total_latency_ms += user_result['total_latency_ms']
            result.request_latencies_ms.extend(user_result['latencies_ms'])
        
        result.end_time = time.time()
        
        # Calculate final resource metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        result.memory_growth_mb = final_memory - initial_memory
        
        self.logger.info(f"Stress test completed: {result.total_requests} requests")
        return result
    
    def run_memory_exhaustion_test(
        self,
        extractor: FeatureExtractor
    ) -> StressTestResult:
        """Run memory exhaustion stress test.
        
        Args:
            extractor: Feature extractor to test
            
        Returns:
            Stress test result
        """
        self.logger.info("Starting memory exhaustion stress test")
        
        result = StressTestResult(
            test_name="Memory Exhaustion Stress Test",
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        result.start_time = time.time()
        
        # Track initial resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run extraction with increasingly large data
        data_multiplier = 1
        max_multiplier = 20
        requests_made = 0
        
        while data_multiplier <= max_multiplier and time.time() - result.start_time < self.config.test_duration_seconds:
            try:
                # Generate increasingly large test data
                data_size = 60 * data_multiplier
                num_features = 15 * data_multiplier
                test_data = np.random.randn(data_size, num_features).astype(np.float32)
                
                # Measure extraction time
                start_time = time.time()
                _ = extractor.extract_features(test_data)  # We don't use the features
                end_time = time.time()
                
                # Record metrics
                latency_ms = (end_time - start_time) * 1000
                result.request_latencies_ms.append(latency_ms)
                result.total_latency_ms += latency_ms
                result.successful_requests += 1
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                result.peak_memory_mb = max(result.peak_memory_mb, current_memory)
                
                # Record resource metrics
                resource_metrics = {
                    'timestamp': time.time(),
                    'process_memory_mb': current_memory,
                    'data_multiplier': data_multiplier
                }
                result.resource_metrics.append(resource_metrics)
                
                requests_made += 1
                data_multiplier += 1
                
            except Exception as e:
                self.logger.warning(f"Memory exhaustion test failed at multiplier {data_multiplier}: {e}")
                result.failed_requests += 1
                result.request_latencies_ms.append(5000.0)  # High latency for failures
                result.total_latency_ms += 5000.0
                break  # Stop on first failure
        
        result.total_requests = requests_made + result.failed_requests
        result.end_time = time.time()
        
        # Calculate final resource metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        result.memory_growth_mb = final_memory - initial_memory
        
        self.logger.info(f"Memory exhaustion test completed: {result.total_requests} requests")
        return result
    
    def run_cpu_saturation_test(
        self,
        extractor: FeatureExtractor
    ) -> StressTestResult:
        """Run CPU saturation stress test.
        
        Args:
            extractor: Feature extractor to test
            
        Returns:
            Stress test result
        """
        self.logger.info("Starting CPU saturation stress test")
        
        result = StressTestResult(
            test_name="CPU Saturation Stress Test",
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        result.start_time = time.time()
        
        # Run many concurrent extractions to saturate CPU
        concurrent_workers = min(50, self.config.max_concurrent_users)  # Limit to reasonable number
        requests_per_worker = 100
        
        def cpu_stress_worker(worker_id: int) -> Dict[str, Any]:
            """Worker function for CPU stress test."""
            worker_metrics = {
                'requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_latency_ms': 0.0,
                'latencies_ms': []
            }
            
            for i in range(requests_per_worker):
                try:
                    # Generate complex test data
                    test_data = np.random.randn(100, 20).astype(np.float32)
                    
                    # Measure extraction time
                    start_time = time.time()
                    _ = extractor.extract_features(test_data)  # We don't use the features
                    end_time = time.time()
                    
                    # Record metrics
                    latency_ms = (end_time - start_time) * 1000
                    worker_metrics['latencies_ms'].append(latency_ms)
                    worker_metrics['total_latency_ms'] += latency_ms
                    worker_metrics['successful_requests'] += 1
                    
                except Exception as e:
                    self.logger.warning(f"CPU stress worker {worker_id}, request {i} failed: {e}")
                    worker_metrics['failed_requests'] += 1
                    worker_metrics['latencies_ms'].append(5000.0)
                    worker_metrics['total_latency_ms'] += 5000.0
                
                worker_metrics['requests'] += 1
            
            return worker_metrics
        
        # Run concurrent workers
        worker_results = []
        worker_threads = []
        
        for worker_id in range(concurrent_workers):
            worker_thread = threading.Thread(
                target=lambda wid=worker_id: worker_results.append(cpu_stress_worker(wid))
            )
            worker_thread.start()
            worker_threads.append(worker_thread)
        
        # Wait for all workers to complete
        for worker_thread in worker_threads:
            worker_thread.join()
        
        # Aggregate results
        for worker_result in worker_results:
            result.total_requests += worker_result['requests']
            result.successful_requests += worker_result['successful_requests']
            result.failed_requests += worker_result['failed_requests']
            result.total_latency_ms += worker_result['total_latency_ms']
            result.request_latencies_ms.extend(worker_result['latencies_ms'])
        
        result.end_time = time.time()
        
        self.logger.info(f"CPU saturation test completed: {result.total_requests} requests")
        return result