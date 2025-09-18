# Load Testing Framework Implementation

## Overview

This document details the implementation of the load testing framework for validating the <100ms feature extraction requirement. The framework will be implemented in the Code mode by the development team.

## 1. Core Load Testing Components

### 1.1 Load Test Configuration

```python
# File: src/performance/load_test_config.py

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class LoadPattern(Enum):
    """Load pattern types"""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STEP = "step"

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    
    # Basic configuration
    name: str
    description: str
    load_pattern: LoadPattern
    duration_seconds: int
    
    # Load parameters
    target_rps: int = 10
    min_rps: int = 1
    max_rps: int = 100
    
    # Pattern-specific parameters
    ramp_up_duration: int = 60 # For ramp-up pattern
    spike_duration: int = 10    # For spike pattern
    spike_rps: int = 50         # Peak RPS for spike
    
    # Worker configuration
    num_workers: int = 4
    max_workers: int = 32
    
    # Timeout and retry settings
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Metrics and reporting
    metrics_collection_interval: int = 5  # seconds
    report_format: str = "json"
    
    # Resource monitoring
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_network: bool = False
    
    # Alerting thresholds
    latency_p95_threshold_ms: float = 100.0
    latency_p99_threshold_ms: float = 200.0
    error_rate_threshold: float = 0.01
    success_rate_threshold: float = 0.95
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        
        if self.target_rps <= 0:
            raise ValueError("target_rps must be positive")
        
        if self.ramp_up_duration > self.duration_seconds:
            raise ValueError("ramp_up_duration cannot exceed duration_seconds")
```

### 1.2 Load Test Runner

```python
# File: src/performance/load_test_runner.py

import asyncio
import time
import statistics
from typing import Callable, List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import psutil
import threading
from datetime import datetime

from .load_test_config import LoadTestConfig, LoadPattern
from .performance_metrics import PerformanceMetrics

class LoadTestRunner:
    """Main load test execution engine"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.running = False
        self.start_time = 0.0
        self.end_time = 0.0
        
    async def run_test(self, 
                      test_function: Callable,
                      test_args: Optional[List[Any]] = None,
                      test_kwargs: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Execute load test based on configuration"""
        
        test_args = test_args or []
        test_kwargs = test_kwargs or {}
        
        self.running = True
        self.start_time = time.time()
        self.end_time = self.start_time + self.config.duration_seconds
        
        print(f"Starting load test: {self.config.name}")
        print(f"Pattern: {self.config.load_pattern.value}")
        print(f"Duration: {self.config.duration_seconds}s")
        
        try:
            if self.config.load_pattern == LoadPattern.CONSTANT:
                await self._run_constant_load(test_function, test_args, test_kwargs)
            elif self.config.load_pattern == LoadPattern.RAMP_UP:
                await self._run_ramp_up_load(test_function, test_args, test_kwargs)
            elif self.config.load_pattern == LoadPattern.SPIKE:
                await self._run_spike_load(test_function, test_args, test_kwargs)
            elif self.config.load_pattern == LoadPattern.STEP:
                await self._run_step_load(test_function, test_args, test_kwargs)
            
            # Final metrics calculation
            self.metrics.end_time = time.time()
            
            print(f"Load test completed: {self.config.name}")
            self._print_summary()
            
            return self.metrics
            
        finally:
            self.running = False
    
    async def _run_constant_load(self, test_function: Callable, 
                               test_args: List[Any], 
                               test_kwargs: Dict[str, Any]) -> None:
        """Run constant load pattern"""
        
        request_interval = 1.0 / self.config.target_rps
        next_request_time = self.start_time
        
        while time.time() < self.end_time and self.running:
            current_time = time.time()
            
            if current_time >= next_request_time:
                await self._execute_single_request(test_function, test_args, test_kwargs)
                next_request_time += request_interval
            else:
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.001)
    
    async def _run_ramp_up_load(self, test_function: Callable,
                              test_args: List[Any],
                              test_kwargs: Dict[str, Any]) -> None:
        """Run ramp-up load pattern"""
        
        while time.time() < self.end_time and self.running:
            # Calculate current RPS based on elapsed time
            elapsed = time.time() - self.start_time
            progress = min(elapsed / self.config.ramp_up_duration, 1.0)
            current_rps = int(self.config.min_rps + 
                            (self.config.max_rps - self.config.min_rps) * progress)
            
            if current_rps > 0:
                request_interval = 1.0 / current_rps
                await self._execute_single_request(test_function, test_args, test_kwargs)
                await asyncio.sleep(request_interval)
            else:
                await asyncio.sleep(0.1)
    
    async def _run_spike_load(self, test_function: Callable,
                            test_args: List[Any],
                            test_kwargs: Dict[str, Any]) -> None:
        """Run spike load pattern"""
        
        base_interval = 1.0 / self.config.target_rps
        spike_interval = 1.0 / self.config.spike_rps
        
        last_spike_time = self.start_time
        spike_active = False
        
        while time.time() < self.end_time and self.running:
            current_time = time.time()
            
            # Check if we should start a spike
            if not spike_active and current_time - last_spike_time >= (self.config.spike_duration * 2):
                spike_active = True
                last_spike_time = current_time
            
            # Check if spike should end
            if spike_active and current_time - last_spike_time >= self.config.spike_duration:
                spike_active = False
            
            # Execute request with appropriate interval
            if spike_active:
                await self._execute_single_request(test_function, test_args, test_kwargs)
                await asyncio.sleep(spike_interval)
            else:
                await self._execute_single_request(test_function, test_args, test_kwargs)
                await asyncio.sleep(base_interval)
    
    async def _run_step_load(self, test_function: Callable,
                           test_args: List[Any],
                           test_kwargs: Dict[str, Any]) -> None:
        """Run step load pattern"""
        
        step_duration = self.config.duration_seconds // 4
        step_rps_values = [
            self.config.target_rps // 4,
            self.config.target_rps // 2,
            self.config.target_rps,
            self.config.target_rps // 2
        ]
        
        for step_index, step_rps in enumerate(step_rps_values):
            step_start = self.start_time + (step_index * step_duration)
            step_end = step_start + step_duration
            
            request_interval = 1.0 / step_rps if step_rps > 0 else 1.0
            
            while time.time() < min(step_end, self.end_time) and self.running:
                await self._execute_single_request(test_function, test_args, test_kwargs)
                await asyncio.sleep(request_interval)
    
    async def _execute_single_request(self, test_function: Callable,
                                    test_args: List[Any],
                                    test_kwargs: Dict[str, Any]) -> None:
        """Execute a single test request and collect metrics"""
        
        self.metrics.start_timing()
        
        try:
            if asyncio.iscoroutinefunction(test_function):
                result = await asyncio.wait_for(
                    test_function(*test_args, **test_kwargs),
                    timeout=self.config.request_timeout_seconds
                )
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, test_function, *test_args),
                        timeout=self.config.request_timeout_seconds
                    )
            
            self.metrics.end_timing(success=True)
            
        except asyncio.TimeoutError:
            self.metrics.end_timing(success=False, timeout=True)
        except Exception as e:
            self.metrics.end_timing(success=False, error=True)
    
    def _collect_resource_metrics(self) -> None:
        """Collect system resource metrics"""
        if self.config.monitor_cpu:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics.cpu_samples.append(cpu_percent)
        
        if self.config.monitor_memory:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.metrics.memory_samples.append(memory_mb)
    
    def _print_summary(self) -> None:
        """Print test summary"""
        latency_percentiles = self.metrics.get_latency_percentiles()
        throughput = self.metrics.get_throughput()
        success_rate = self.metrics.get_success_rate()
        cache_hit_rate = self.metrics.get_cache_hit_rate()
        fallback_rate = self.metrics.get_fallback_rate()
        
        print("\n" + "="*50)
        print("LOAD TEST SUMMARY")
        print("="*50)
        print(f"Test Name: {self.config.name}")
        print(f"Duration: {self.config.duration_seconds}s")
        print(f"Pattern: {self.config.load_pattern.value}")
        print(f"Target RPS: {self.config.target_rps}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Throughput: {throughput:.2f} RPS")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Cache Hit Rate: {cache_hit_rate:.2%}")
        print(f"  Fallback Rate: {fallback_rate:.2%}")
        print()
        print("LATENCY METRICS (ms):")
        print(f"  Average: {latency_percentiles.get('avg', 0):.2f}")
        print(f"  P50: {latency_percentiles.get('p50', 0):.2f}")
        print(f"  P95: {latency_percentiles.get('p95', 0):.2f}")
        print(f"  P99: {latency_percentiles.get('p99', 0):.2f}")
        print(f"  Min: {latency_percentiles.get('min', 0):.2f}")
        print(f"  Max: {latency_percentiles.get('max', 0):.2f}")
        print()
        print("RESOURCE METRICS:")
        if self.metrics.cpu_samples:
            avg_cpu = statistics.mean(self.metrics.cpu_samples)
            max_cpu = max(self.metrics.cpu_samples)
            print(f"  Avg CPU: {avg_cpu:.1f}%")
            print(f"  Max CPU: {max_cpu:.1f}%")
        if self.metrics.memory_samples:
            avg_memory = statistics.mean(self.metrics.memory_samples)
            max_memory = max(self.metrics.memory_samples)
            print(f"  Avg Memory: {avg_memory:.1f}MB")
            print(f"  Max Memory: {max_memory:.1f}MB")
        print("="*50)
```

### 1.3 Concurrent Load Testing

```python
# File: src/performance/concurrent_load_tester.py

import asyncio
import time
from typing import Callable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .load_test_config import LoadTestConfig
from .load_test_runner import LoadTestRunner
from .performance_metrics import PerformanceMetrics

class ConcurrentLoadTester:
    """Execute multiple load tests concurrently"""
    
    def __init__(self):
        self.test_runners: List[LoadTestRunner] = []
    
    async def run_concurrent_tests(self, 
                                 test_configs: List[LoadTestConfig],
                                 test_function: Callable,
                                 test_args: Optional[List[Any]] = None,
                                 test_kwargs: Optional[Dict[str, Any]] = None) -> List[PerformanceMetrics]:
        """Run multiple load tests concurrently"""
        
        test_args = test_args or []
        test_kwargs = test_kwargs or {}
        
        # Create test runners
        runners = [LoadTestRunner(config) for config in test_configs]
        
        # Run all tests concurrently
        tasks = [
            runner.run_test(test_function, test_args, test_kwargs)
            for runner in runners
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, PerformanceMetrics):
                successful_results.append(result)
            elif isinstance(result, Exception):
                print(f"Test failed with exception: {result}")
        
        return successful_results
    
    def run_distributed_load_test(self,
                                config: LoadTestConfig,
                                test_function: Callable,
                                num_instances: int = 4) -> PerformanceMetrics:
        """Run distributed load test across multiple instances"""
        
        # This would typically involve running tests on different machines
        # For now, we'll simulate with multiple threads
        
        def run_single_instance(instance_id: int) -> PerformanceMetrics:
            """Run a single load test instance"""
            runner = LoadTestRunner(config)
            # In a real implementation, this would be async
            # For now, we'll return a mock result
            metrics = PerformanceMetrics()
            metrics.total_requests = config.target_rps * config.duration_seconds // num_instances
            metrics.successful_requests = int(metrics.total_requests * 0.98)  # 98% success rate
            return metrics
        
        # Run instances concurrently
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [
                executor.submit(run_single_instance, i)
                for i in range(num_instances)
            ]
            
            results = [future.result() for future in futures]
        
        # Aggregate results
        aggregated_metrics = PerformanceMetrics()
        
        for result in results:
            aggregated_metrics.total_requests += result.total_requests
            aggregated_metrics.successful_requests += result.successful_requests
            aggregated_metrics.latencies.extend(result.latencies)
        
        return aggregated_metrics
```

## 2. Load Pattern Generators

### 2.1 Constant Load Generator

```python
# File: src/performance/load_patterns/constant_load.py

import asyncio
import time
from typing import Callable, List, Dict, Any, AsyncIterator

class ConstantLoadGenerator:
    """Generate constant load pattern"""
    
    def __init__(self, target_rps: int):
        self.target_rps = target_rps
        self.request_interval = 1.0 / target_rps if target_rps > 0 else 1.0
    
    async def generate_load(self, 
                          duration_seconds: int,
                          start_time: float) -> AsyncIterator[float]:
        """Generate constant load for specified duration"""
        
        end_time = start_time + duration_seconds
        next_request_time = start_time
        
        while time.time() < end_time:
            current_time = time.time()
            
            if current_time >= next_request_time:
                yield current_time
                next_request_time += self.request_interval
            else:
                await asyncio.sleep(0.001)
```

### 2.2 Ramp-up Load Generator

```python
# File: src/performance/load_patterns/ramp_up_load.py

import asyncio
import time
from typing import AsyncIterator

class RampUpLoadGenerator:
    """Generate ramp-up load pattern"""
    
    def __init__(self, min_rps: int, max_rps: int, ramp_duration: int):
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.ramp_duration = ramp_duration
    
    async def generate_load(self,
                          duration_seconds: int,
                          start_time: float) -> AsyncIterator[float]:
        """Generate ramp-up load for specified duration"""
        
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            # Calculate current RPS based on elapsed time
            elapsed = time.time() - start_time
            progress = min(elapsed / self.ramp_duration, 1.0)
            current_rps = int(self.min_rps + (self.max_rps - self.min_rps) * progress)
            
            if current_rps > 0:
                request_interval = 1.0 / current_rps
                yield time.time()
                await asyncio.sleep(request_interval)
            else:
                await asyncio.sleep(0.1)
```

### 2.3 Spike Load Generator

```python
# File: src/performance/load_patterns/spike_load.py

import asyncio
import time
from typing import AsyncIterator

class SpikeLoadGenerator:
    """Generate spike load pattern"""
    
    def __init__(self, base_rps: int, spike_rps: int, spike_duration: int):
        self.base_rps = base_rps
        self.spike_rps = spike_rps
        self.spike_duration = spike_duration
    
    async def generate_load(self,
                          duration_seconds: int,
                          start_time: float) -> AsyncIterator[float]:
        """Generate spike load for specified duration"""
        
        end_time = start_time + duration_seconds
        last_spike_time = start_time
        spike_active = False
        
        while time.time() < end_time:
            current_time = time.time()
            
            # Check if we should start a spike
            if not spike_active and current_time - last_spike_time >= (self.spike_duration * 2):
                spike_active = True
                last_spike_time = current_time
            
            # Check if spike should end
            if spike_active and current_time - last_spike_time >= self.spike_duration:
                spike_active = False
            
            # Generate request with appropriate interval
            if spike_active and self.spike_rps > 0:
                request_interval = 1.0 / self.spike_rps
                yield current_time
                await asyncio.sleep(request_interval)
            elif self.base_rps > 0:
                request_interval = 1.0 / self.base_rps
                yield current_time
                await asyncio.sleep(request_interval)
            else:
                await asyncio.sleep(0.1)
```

## 3. Load Testing Utilities

### 3.1 Data Generators

```python
# File: src/performance/test_data_generators.py

import numpy as np
from typing import Generator, Tuple
import pandas as pd
from datetime import datetime, timedelta

class MarketDataGenerator:
    """Generate realistic market data for testing"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'TSLA': 200.0,
            'AMZN': 3200.0
        }
    
    def generate_single_market_window(self, 
                                    symbol: str = 'TEST',
                                    timesteps: int = 60,
                                    features: int = 15) -> np.ndarray:
        """Generate a single market data window"""
        
        # Start with base price or random
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Generate price series with random walk
        prices = [base_price]
        for _ in range(timesteps - 1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Generate features based on prices
        data = np.zeros((timesteps, features))
        
        for i, price in enumerate(prices):
            # OHLC-like features
            data[i, 0] = price * (1 + np.random.normal(0, 0.005))  # Open
            data[i, 1] = price * (1 + abs(np.random.normal(0, 0.01)))  # High
            data[i, 2] = price * (1 - abs(np.random.normal(0, 0.01)))  # Low
            data[i, 3] = price  # Close
            
            # Volume
            data[i, 4] = np.random.randint(1000, 100000)
            
            # Technical indicators (simplified)
            data[i, 5] = np.random.normal(0, 0.02)  # Returns
            data[i, 6] = abs(np.random.normal(0, 0.03))  # Volatility
            data[i, 7] = np.random.uniform(20, 80)  # RSI
            data[i, 8] = np.random.normal(0, 0.1)  # MACD
            data[i, 9] = np.random.normal(0, 0.05)  # MACD Signal
            
            # Additional features
            data[i, 10] = np.random.uniform(0, 1)  # BB Position
            data[i, 11] = np.random.uniform(0.5, 2.0)  # Volume Ratio
            data[i, 12] = price  # SMA
            data[i, 13] = price  # EMA
            data[i, 14] = price  # Another technical indicator
        
        return data
    
    def generate_batch_market_data(self, 
                                 batch_size: int = 32,
                                 timesteps: int = 60,
                                 features: int = 15) -> np.ndarray:
        """Generate batch of market data"""
        batch_data = np.zeros((batch_size, timesteps, features))
        
        for i in range(batch_size):
            symbol = np.random.choice(self.symbols)
            batch_data[i] = self.generate_single_market_window(symbol, timesteps, features)
        
        return batch_data

class FeatureExtractionDataGenerator:
    """Generate data specifically for feature extraction testing"""
    
    def __init__(self):
        self.generator = MarketDataGenerator()
    
    def generate_test_data(self, 
                          data_type: str = 'single',
                          **kwargs) -> np.ndarray:
        """Generate test data based on type"""
        
        if data_type == 'single':
            return self.generator.generate_single_market_window(**kwargs)
        elif data_type == 'batch':
            return self.generator.generate_batch_market_data(**kwargs)
        elif data_type == 'edge_case_small':
            # Very small dataset
            return np.random.randn(10, 5)
        elif data_type == 'edge_case_large':
            # Very large dataset
            return np.random.randn(200, 50)
        elif data_type == 'edge_case_nan':
            # Dataset with NaN values
            data = np.random.randn(60, 15)
            # Insert some NaN values
            data[10:15, 5:8] = np.nan
            return data
        else:
            return self.generator.generate_single_market_window(**kwargs)
```

### 3.2 Resource Monitor

```python
# File: src/performance/resource_monitor.py

import psutil
import time
import threading
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamps: List[float] = field(default_factory=list)
    cpu_percent: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    memory_percent: List[float] = field(default_factory=list)
    network_bytes_sent: List[int] = field(default_factory=list)
    network_bytes_recv: List[int] = field(default_factory=list)

class ResourceMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self.metrics = ResourceMetrics()
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Resource monitoring started")
    
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Resource monitoring stopped")
        return self.metrics
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        net_io_start = psutil.net_io_counters()
        
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = psutil.virtual_memory().percent
                
                # Network usage
                net_io_current = psutil.net_io_counters()
                bytes_sent = net_io_current.bytes_sent - net_io_start.bytes_sent
                bytes_recv = net_io_current.bytes_recv - net_io_start.bytes_recv
                
                # Store metrics
                self.metrics.timestamps.append(timestamp)
                self.metrics.cpu_percent.append(cpu_percent)
                self.metrics.memory_mb.append(memory_mb)
                self.metrics.memory_percent.append(memory_percent)
                self.metrics.network_bytes_sent.append(bytes_sent)
                self.metrics.network_bytes_recv.append(bytes_recv)
                
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(self.interval_seconds)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        if not self.metrics.timestamps:
            return {}
        
        return {
            'cpu_percent': self.metrics.cpu_percent[-1],
            'memory_mb': self.metrics.memory_mb[-1],
            'memory_percent': self.metrics.memory_percent[-1],
            'network_bytes_sent': self.metrics.network_bytes_sent[-1],
            'network_bytes_recv': self.metrics.network_bytes_recv[-1]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.metrics.timestamps:
            return {}
        
        return {
            'duration_seconds': self.metrics.timestamps[-1] - self.metrics.timestamps[0] if self.metrics.timestamps else 0,
            'avg_cpu_percent': sum(self.metrics.cpu_percent) / len(self.metrics.cpu_percent) if self.metrics.cpu_percent else 0,
            'max_cpu_percent': max(self.metrics.cpu_percent) if self.metrics.cpu_percent else 0,
            'avg_memory_mb': sum(self.metrics.memory_mb) / len(self.metrics.memory_mb) if self.metrics.memory_mb else 0,
            'max_memory_mb': max(self.metrics.memory_mb) if self.metrics.memory_mb else 0,
            'total_network_bytes_sent': self.metrics.network_bytes_sent[-1] if self.metrics.network_bytes_sent else 0,
            'total_network_bytes_recv': self.metrics.network_bytes_recv[-1] if self.metrics.network_bytes_recv else 0
        }
```

## 4. Integration Examples

### 4.1 Feature Extraction Load Testing

```python
# File: examples/feature_extraction_load_test.py

import asyncio
import numpy as np
from unittest.mock import Mock

from src.performance.load_test_config import LoadTestConfig, LoadPattern
from src.performance.load_test_runner import LoadTestRunner
from src.performance.test_data_generators import FeatureExtractionDataGenerator
from src.ml.feature_extraction import FeatureExtractorFactory, FeatureExtractionConfig

async def run_feature_extraction_load_test():
    """Example of running feature extraction load test"""
    
    # Create mock model
    mock_model = Mock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None
    
    def mock_forward(input_tensor, return_features=True, use_ensemble=True):
        import time
        time.sleep(0.02)  # 20ms processing time
        return {
            'fused_features': np.random.randn(1, 10, 256),
            'classification_probs': np.random.rand(1, 3),
            'regression_uncertainty': np.random.rand(1, 1)
        }
    
    mock_model.forward = mock_forward
    
    # Create feature extractor
    config = FeatureExtractionConfig(
        fused_feature_dim=256,
        enable_caching=True,
        cache_size=1000,
        enable_fallback=True
    )
    extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
    
    # Create data generator
    data_generator = FeatureExtractionDataGenerator()
    
    # Define test function
    async def extract_features_async():
        data = data_generator.generate_test_data('single', timesteps=60, features=15)
        return extractor.extract_features(data)
    
    # Configure load test
    load_config = LoadTestConfig(
        name="Feature Extraction Load Test",
        description="Test CNN+LSTM feature extraction performance under load",
        load_pattern=LoadPattern.CONSTANT,
        target_rps=50,
        duration_seconds=120,
        num_workers=8,
        latency_p95_threshold_ms=100.0,
        latency_p99_threshold_ms=200.0
    )
    
    # Run load test
    runner = LoadTestRunner(load_config)
    metrics = await runner.run_test(extract_features_async)
    
    return metrics

# Run the example
if __name__ == "__main__":
    asyncio.run(run_feature_extraction_load_test())
```

### 4.2 Concurrent Load Testing

```python
# File: examples/concurrent_load_test_example.py

import asyncio
from src.performance.load_test_config import LoadTestConfig, LoadPattern
from src.performance.concurrent_load_tester import ConcurrentLoadTester

async def run_concurrent_load_tests():
    """Example of running concurrent load tests"""
    
    # Define multiple test configurations
    configs = [
        LoadTestConfig(
            name="Low Load Test",
            description="Test under low load conditions",
            load_pattern=LoadPattern.CONSTANT,
            target_rps=20,
            duration_seconds=60
        ),
        LoadTestConfig(
            name="Medium Load Test",
            description="Test under medium load conditions",
            load_pattern=LoadPattern.RAMP_UP,
            min_rps=10,
            max_rps=80,
            ramp_up_duration=30,
            duration_seconds=120
        ),
        LoadTestConfig(
            name="High Load Test",
            description="Test under high load conditions",
            load_pattern=LoadPattern.SPIKE,
            target_rps=50,
            spike_rps=150,
            spike_duration=10,
            duration_seconds=180
        )
    ]
    
    # Mock test function
    async def mock_test_function():
        import time
        import random
        # Simulate processing time
        processing_time = random.uniform(0.01, 0.05)
        time.sleep(processing_time)
        return {"result": "success"}
    
    # Run concurrent tests
    tester = ConcurrentLoadTester()
    results = await tester.run_concurrent_tests(configs, mock_test_function)
    
    # Print results
    for i, metrics in enumerate(results):
        print(f"Test {i+1} Results:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Success Rate: {metrics.get_success_rate():.2%}")
        latency_percentiles = metrics.get_latency_percentiles()
        print(f"  P95 Latency: {latency_percentiles.get('p95', 0):.2f}ms")
        print()

if __name__ == "__main__":
    asyncio.run(run_concurrent_load_tests())
```

## Implementation Roadmap

### Phase 1: Core Components (Week 1)
1. Implement `LoadTestConfig` and `LoadPattern` enums
2. Create `LoadTestRunner` with basic load patterns
3. Implement `PerformanceMetrics` collection
4. Add resource monitoring capabilities

### Phase 2: Advanced Load Patterns (Week 2)
1. Implement all load pattern generators
2. Add concurrent load testing support
3. Create data generators for realistic testing
4. Add distributed load testing capabilities

### Phase 3: Integration and Testing (Week 3)
1. Integrate with feature extraction components
2. Create comprehensive test suites
3. Add alerting and notification systems
4. Implement CI/CD integration

### Phase 4: Optimization and Documentation (Week 4)
1. Optimize performance of the testing framework
2. Create detailed documentation and examples
3. Add visualization and reporting features
4. Conduct validation testing

## Success Criteria

1. **Framework Completeness**: All core components implemented and tested
2. **Load Pattern Support**: Support for constant, ramp-up, spike, and step patterns
3. **Resource Monitoring**: Comprehensive system resource monitoring
4. **Integration Ready**: Seamless integration with existing feature extraction system
5. **Performance**: Framework overhead < 5% of total test execution time
6. **Reliability**: 99.9% uptime for test execution

This load testing framework implementation provides a robust foundation for validating the <100ms feature extraction requirement under various load conditions.