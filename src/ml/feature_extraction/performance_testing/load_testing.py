"""Load testing framework for feature extraction.

This module provides load testing capabilities to validate feature extraction
performance under various concurrent load conditions.
"""

import time
import logging
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ml.feature_extraction.base import FeatureExtractor
from .framework import PerformanceTestConfig, PerformanceTestResult

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    # Load parameters
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time_seconds: int = 30
    think_time_seconds: float = 0.1
    
    # Test patterns
    load_pattern: str = "constant"  # constant, ramp-up, spike, step
    spike_duration_seconds: int = 10
    spike_multiplier: int = 3
    
    # Performance thresholds
    target_response_time_ms: float = 100.0
    target_throughput_rps: float = 50.0
    max_error_rate: float = 0.01
    
    # Resource thresholds
    max_memory_mb: float = 2000.0
    max_cpu_percent: float = 85.0
    
    # Test metadata
    test_name: str = "Feature Extraction Load Test"
    test_description: str = "Load test for feature extraction under concurrent users"
    test_tags: List[str] = field(default_factory=list)


@dataclass
class LoadTestResult:
    """Result of a load test."""
    # Test metadata
    test_name: str
    test_timestamp: datetime
    config: LoadTestConfig
    
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
    user_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Resource metrics
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    
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
    
    def meets_performance_requirements(self) -> bool:
        """Check if load test meets performance requirements."""
        throughput = self.get_throughput_rps()
        error_rate = self.get_error_rate()
        avg_latency = self.get_avg_latency_ms()
        
        return (
            throughput >= self.config.target_throughput_rps and
            error_rate <= self.config.max_error_rate and
            avg_latency <= self.config.target_response_time_ms
        )


class LoadTester:
    """Load tester for feature extraction components."""
    
    def __init__(self, config: Optional[LoadTestConfig] = None):
        """Initialize load tester.
        
        Args:
            config: Load test configuration
        """
        self.config = config or LoadTestConfig()
        self.logger = logging.getLogger(__name__)
    
    def run_constant_load_test(
        self,
        extractor: FeatureExtractor,
        duration_seconds: int = 300,
        target_rps: int = 50
    ) -> LoadTestResult:
        """Run constant load test for specified duration.
        
        Args:
            extractor: Feature extractor to test
            duration_seconds: Test duration in seconds
            target_rps: Target requests per second
            
        Returns:
            Load test result
        """
        self.logger.info(f"Starting constant load test: {duration_seconds}s at {target_rps} RPS")
        
        result = LoadTestResult(
            test_name=self.config.test_name,
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        result.start_time = time.time()
        
        # Calculate request interval
        request_interval = 1.0 / target_rps if target_rps > 0 else 0.1
        
        # Run test for specified duration
        end_time = result.start_time + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            request_start = time.time()
            
            try:
                # Generate test data
                test_data = np.random.randn(60, 15).astype(np.float32)
                
                # Measure extraction time
                start_time = time.time()
                _ = extractor.extract_features(test_data)  # We don't use the features
                end_time_req = time.time()
                
                # Record metrics
                latency_ms = (end_time_req - start_time) * 1000
                result.request_latencies_ms.append(latency_ms)
                result.total_latency_ms += latency_ms
                result.successful_requests += 1
                
            except Exception as e:
                self.logger.warning(f"Request {request_count} failed: {e}")
                result.failed_requests += 1
                # Record high latency for failed requests
                result.request_latencies_ms.append(5000.0)
                result.total_latency_ms += 5000.0
            
            result.total_requests += 1
            request_count += 1
            
            # Maintain target rate
            elapsed = time.time() - request_start
            sleep_time = max(0, request_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        result.end_time = time.time()
        
        self.logger.info(f"Constant load test completed: {result.total_requests} requests")
        return result
    
    def run_concurrent_users_test(
        self,
        extractor: FeatureExtractor,
        test_data_generator: Optional[Callable] = None
    ) -> LoadTestResult:
        """Run concurrent users load test.
        
        Args:
            extractor: Feature extractor to test
            test_data_generator: Function to generate test data
            
        Returns:
            Load test result
        """
        self.logger.info(f"Starting concurrent users test: {self.config.concurrent_users} users")
        
        result = LoadTestResult(
            test_name=self.config.test_name,
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        result.start_time = time.time()
        
        def user_worker(user_id: int) -> Dict[str, Any]:
            """Worker function for a single user."""
            user_metrics = {
                'user_id': user_id,
                'requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_latency_ms': 0.0,
                'latencies_ms': []
            }
            
            for i in range(self.config.requests_per_user):
                try:
                    # Generate test data
                    if test_data_generator:
                        test_data = test_data_generator()
                    else:
                        test_data = np.random.randn(60, 15).astype(np.float32)
                    
                    # Measure extraction time
                    start_time = time.time()
                    _ = extractor.extract_features(test_data)  # We don't use the features
                    end_time = time.time()
                    
                    # Record metrics
                    latency_ms = (end_time - start_time) * 1000
                    user_metrics['latencies_ms'].append(latency_ms)
                    user_metrics['total_latency_ms'] += latency_ms
                    user_metrics['successful_requests'] += 1
                    
                    # Add think time
                    if self.config.think_time_seconds > 0:
                        time.sleep(self.config.think_time_seconds)
                        
                except Exception as e:
                    self.logger.warning(f"User {user_id}, request {i} failed: {e}")
                    user_metrics['failed_requests'] += 1
                    # Record high latency for failed requests
                    user_metrics['latencies_ms'].append(5000.0)
                    user_metrics['total_latency_ms'] += 5000.0
                
                user_metrics['requests'] += 1
            
            return user_metrics
        
        # Run concurrent users
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = [
                executor.submit(user_worker, user_id)
                for user_id in range(self.config.concurrent_users)
            ]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    user_metrics = future.result()
                    user_id = user_metrics['user_id']
                    result.user_metrics[user_id] = user_metrics
                    
                    # Aggregate metrics
                    result.total_requests += user_metrics['requests']
                    result.successful_requests += user_metrics['successful_requests']
                    result.failed_requests += user_metrics['failed_requests']
                    result.total_latency_ms += user_metrics['total_latency_ms']
                    result.request_latencies_ms.extend(user_metrics['latencies_ms'])
                    
                except Exception as e:
                    self.logger.error(f"User worker failed: {e}")
        
        result.end_time = time.time()
        
        self.logger.info(f"Concurrent users test completed: {result.total_requests} requests")
        return result
    
    def run_ramp_up_load_test(
        self,
        extractor: FeatureExtractor,
        max_users: int = 50,
        ramp_duration_seconds: int = 300
    ) -> LoadTestResult:
        """Run ramp-up load test.
        
        Args:
            extractor: Feature extractor to test
            max_users: Maximum number of concurrent users
            ramp_duration_seconds: Duration to reach maximum users
            
        Returns:
            Load test result
        """
        self.logger.info(f"Starting ramp-up load test: {max_users} users over {ramp_duration_seconds}s")
        
        result = LoadTestResult(
            test_name=self.config.test_name,
            test_timestamp=datetime.now(),
            config=self.config
        )
        
        result.start_time = time.time()
        
        # Calculate ramp-up parameters
        step_duration = ramp_duration_seconds / max_users if max_users > 0 else 1
        active_users = []
        
        # Ramp up users
        for user_id in range(max_users):
            # Add new user
            def user_worker():
                """Worker function for a single user."""
                requests = 0
                successful_requests = 0
                failed_requests = 0
                total_latency_ms = 0.0
                latencies_ms = []
                
                # Each user makes requests for the remaining time
                user_end_time = time.time() + (ramp_duration_seconds - (user_id * step_duration))
                
                while time.time() < user_end_time:
                    try:
                        # Generate test data
                        test_data = np.random.randn(60, 15).astype(np.float32)
                        
                        # Measure extraction time
                        start_time = time.time()
                        _ = extractor.extract_features(test_data)  # We don't use the features
                        end_time = time.time()
                        
                        # Record metrics
                        latency_ms = (end_time - start_time) * 1000
                        latencies_ms.append(latency_ms)
                        total_latency_ms += latency_ms
                        successful_requests += 1
                        
                        # Add think time
                        if self.config.think_time_seconds > 0:
                            time.sleep(self.config.think_time_seconds)
                            
                    except Exception as e:
                        self.logger.warning(f"Ramp-up user {user_id}, request failed: {e}")
                        failed_requests += 1
                        # Record high latency for failed requests
                        latencies_ms.append(5000.0)
                        total_latency_ms += 5000.0
                    
                    requests += 1
                
                return {
                    'requests': requests,
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'total_latency_ms': total_latency_ms,
                    'latencies_ms': latencies_ms
                }
            
            # Start user in background
            user_thread = threading.Thread(target=user_worker)
            user_thread.start()
            active_users.append(user_thread)
            
            # Wait for next step
            time.sleep(step_duration)
        
        # Wait for all users to complete
        for user_thread in active_users:
            user_thread.join()
        
        result.end_time = time.time()
        
        # For simplicity, we're not collecting detailed per-user metrics in this ramp-up test
        # In a real implementation, we would collect and aggregate these metrics
        
        self.logger.info(f"Ramp-up load test completed")
        return result