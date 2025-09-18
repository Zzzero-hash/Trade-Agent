"""
Performance benchmarking tests for latency and throughput.

Comprehensive performance tests to validate system performance under
various load conditions and identify bottlenecks.

Requirements: 6.6
"""

import pytest
import asyncio
import time
import statistics
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Callable
import numpy as np
import pandas as pd
import psutil
import os
from unittest.mock import Mock, AsyncMock, patch

from src.services.data_aggregator import DataAggregator
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService
from src.ml.cnn_lstm_hybrid_model import CNNLSTMHybridModel
from src.models.market_data import MarketData, ExchangeType


class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class TestDataProcessingPerformance:
    """Test data processing performance benchmarks."""
    
    @pytest.fixture
    def performance_data(self):
        """Generate large dataset for performance testing."""
        
        # Generate 1 million data points
        num_records = 1_000_000
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'] * (num_records // 5)
        
        data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=365)
        
        for i in range(num_records):
            timestamp = base_time + timedelta(minutes=i)
            symbol = symbols[i]
            
            # Generate realistic price data
            base_price = 150.0 if 'AAPL' in symbol else 2500.0
            price_change = np.random.normal(0, 0.02)
            price = base_price * (1 + price_change)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=np.random.randint(100000, 1000000),
                exchange=ExchangeType.ROBINHOOD
            )
            data.append(market_data)
        
        return data
    
    def test_data_aggregation_throughput(self, performance_data):
        """Test data aggregation throughput with large datasets."""
        
        print(f"Testing data aggregation with {len(performance_data):,} records...")
        
        # Test different batch sizes
        batch_sizes = [1000, 5000, 10000, 50000, 100000]
        results = []
        
        for batch_size in batch_sizes:
            batch_data = performance_data[:batch_size]
            
            with PerformanceTimer(f"Batch size {batch_size}") as timer:
                # Convert to DataFrame (simulating aggregation)
                df = pd.DataFrame([md.__dict__ for md in batch_data])
                
                # Perform typical aggregation operations
                grouped = df.groupby('symbol').agg({
                    'close': ['mean', 'std', 'min', 'max'],
                    'volume': ['sum', 'mean'],
                    'timestamp': ['min', 'max']
                })
                
                # Calculate technical indicators
                df['sma_20'] = df.groupby('symbol')['close'].rolling(20).mean().reset_index(0, drop=True)
                df['volatility'] = df.groupby('symbol')['close'].rolling(20).std().reset_index(0, drop=True)
            
            throughput = batch_size / (timer.elapsed_ms / 1000)  # records per second
            
            results.append({
                'batch_size': batch_size,
                'latency_ms': timer.elapsed_ms,
                'throughput_rps': throughput,
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            })
            
            print(f"  Batch {batch_size:,}: {timer.elapsed_ms:.2f}ms, "
                  f"{throughput:,.0f} records/sec, "
                  f"{results[-1]['memory_mb']:.1f}MB")
        
        # Verify performance requirements
        for result in results:
            assert result['throughput_rps'] > 1000  # At least 1K records/sec
            assert result['latency_ms'] < 30000     # Less than 30 seconds
        
        # Throughput should generally increase with batch size (up to a point)
        small_batch_throughput = results[0]['throughput_rps']
        large_batch_throughput = results[-1]['throughput_rps']
        
        print(f"Throughput improvement: {large_batch_throughput/small_batch_throughput:.2f}x")
    
    def test_feature_engineering_performance(self, performance_data):
        """Test feature engineering performance."""
        
        # Create sample data for feature engineering
        sample_size = 10000
        sample_data = performance_data[:sample_size]
        
        df = pd.DataFrame([md.__dict__ for md in sample_data])
        
        print(f"Testing feature engineering with {sample_size:,} records...")
        
        feature_operations = {
            'Technical Indicators': lambda df: self._calculate_technical_indicators(df),
            'Rolling Statistics': lambda df: self._calculate_rolling_stats(df),
            'Price Transformations': lambda df: self._calculate_price_transforms(df),
            'Volume Features': lambda df: self._calculate_volume_features(df),
            'Time Features': lambda df: self._calculate_time_features(df)
        }
        
        results = {}
        
        for operation_name, operation_func in feature_operations.items():
            with PerformanceTimer(operation_name) as timer:
                result_df = operation_func(df.copy())
            
            throughput = sample_size / (timer.elapsed_ms / 1000)
            
            results[operation_name] = {
                'latency_ms': timer.elapsed_ms,
                'throughput_rps': throughput,
                'features_added': len(result_df.columns) - len(df.columns)
            }
            
            print(f"  {operation_name}: {timer.elapsed_ms:.2f}ms, "
                  f"{throughput:,.0f} records/sec, "
                  f"{results[operation_name]['features_added']} features")
        
        # Verify all operations complete within reasonable time
        for operation, result in results.items():
            assert result['latency_ms'] < 5000  # Less than 5 seconds
            assert result['throughput_rps'] > 500  # At least 500 records/sec
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        
        # Simple moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
        
        # Exponential moving averages
        for span in [12, 26]:
            df[f'ema_{span}'] = df.groupby('symbol')['close'].ewm(span=span).mean().reset_index(0, drop=True)
        
        # RSI
        df['rsi'] = df.groupby('symbol')['close'].apply(self._calculate_rsi).reset_index(0, drop=True)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df.groupby('symbol')['macd'].ewm(span=9).mean().reset_index(0, drop=True)
        
        return df
    
    def _calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling statistics."""
        
        windows = [10, 20, 50]
        
        for window in windows:
            # Price statistics
            df[f'volatility_{window}'] = df.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
            df[f'min_{window}'] = df.groupby('symbol')['close'].rolling(window).min().reset_index(0, drop=True)
            df[f'max_{window}'] = df.groupby('symbol')['close'].rolling(window).max().reset_index(0, drop=True)
            
            # Volume statistics
            df[f'volume_mean_{window}'] = df.groupby('symbol')['volume'].rolling(window).mean().reset_index(0, drop=True)
            df[f'volume_std_{window}'] = df.groupby('symbol')['volume'].rolling(window).std().reset_index(0, drop=True)
        
        return df
    
    def _calculate_price_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price transformations."""
        
        # Returns
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df.groupby('symbol')['close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Normalized prices
        df['price_normalized'] = df.groupby('symbol')['close'].apply(
            lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
        ).reset_index(0, drop=True)
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        
        # Volume ratios
        df['volume_ratio'] = df['volume'] / df.groupby('symbol')['volume'].rolling(20).mean().reset_index(0, drop=True)
        
        # Volume-price features
        df['vwap'] = (df['volume'] * df['close']).groupby(df['symbol']).cumsum() / df['volume'].groupby(df['symbol']).cumsum()
        df['volume_price_trend'] = df['volume'] * np.sign(df['returns'])
        
        return df
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class TestMLModelPerformance:
    """Test ML model inference performance."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock ML model for performance testing."""
        
        model = Mock(spec=CNNLSTMHybridModel)
        
        # Mock inference with realistic timing
        def mock_predict(data, batch_size=32):
            # Simulate model inference time based on batch size
            inference_time = 0.01 + (len(data) / batch_size) * 0.005  # Base + per-batch time
            time.sleep(inference_time)
            
            # Return mock predictions
            num_samples = len(data)
            return {
                'predictions': np.random.random((num_samples, 3)),  # 3-class predictions
                'confidence': np.random.uniform(0.6, 0.95, num_samples),
                'uncertainty': np.random.uniform(0.05, 0.2, num_samples)
            }
        
        model.predict = mock_predict
        return model
    
    def test_single_inference_latency(self, mock_model):
        """Test single inference latency."""
        
        print("Testing single inference latency...")
        
        # Test different input sizes
        input_sizes = [1, 10, 50, 100, 500]
        results = []
        
        for size in input_sizes:
            # Generate mock input data
            input_data = np.random.random((size, 50))  # 50 features
            
            latencies = []
            
            # Run multiple inferences to get stable measurements
            for _ in range(10):
                with PerformanceTimer() as timer:
                    predictions = mock_model.predict(input_data)
                
                latencies.append(timer.elapsed_ms)
            
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            throughput = size / (avg_latency / 1000)  # samples per second
            
            results.append({
                'input_size': size,
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'throughput_sps': throughput
            })
            
            print(f"  Size {size}: {avg_latency:.2f}ms avg, {p95_latency:.2f}ms P95, "
                  f"{throughput:.0f} samples/sec")
        
        # Verify latency requirements
        single_sample_latency = results[0]['avg_latency_ms']
        assert single_sample_latency < 100  # Single inference < 100ms
        
        # Batch processing should be more efficient
        batch_efficiency = results[-1]['throughput_sps'] / results[0]['throughput_sps']
        assert batch_efficiency > 2  # At least 2x improvement with batching
        
        print(f"Batch processing efficiency: {batch_efficiency:.2f}x")
    
    def test_concurrent_inference_performance(self, mock_model):
        """Test concurrent inference performance."""
        
        print("Testing concurrent inference performance...")
        
        def run_inference(thread_id: int, num_inferences: int = 20):
            """Run multiple inferences in a thread."""
            
            results = []
            input_data = np.random.random((10, 50))  # 10 samples, 50 features
            
            for i in range(num_inferences):
                with PerformanceTimer() as timer:
                    predictions = mock_model.predict(input_data)
                
                results.append({
                    'thread_id': thread_id,
                    'inference_id': i,
                    'latency_ms': timer.elapsed_ms
                })
            
            return results
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16]
        
        for concurrency in concurrency_levels:
            print(f"  Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(run_inference, i, 10)  # 10 inferences per thread
                    for i in range(concurrency)
                ]
                
                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_inferences = len(all_results)
            avg_latency = statistics.mean([r['latency_ms'] for r in all_results])
            throughput = total_inferences / total_time
            
            print(f"    Concurrency {concurrency}: {throughput:.1f} inferences/sec, "
                  f"avg latency: {avg_latency:.2f}ms")
            
            # Verify reasonable performance
            assert throughput > concurrency * 5  # At least 5 inferences/sec per thread
            assert avg_latency < 1000  # Average latency < 1 second
    
    @pytest.mark.slow
    def test_sustained_inference_load(self, mock_model):
        """Test sustained inference load over time."""
        
        print("Testing sustained inference load...")
        
        # Run sustained load for 60 seconds
        duration_seconds = 60
        target_rps = 50  # Target 50 requests per second
        
        results = []
        start_time = time.time()
        request_interval = 1.0 / target_rps
        
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            
            # Generate input data
            input_data = np.random.random((5, 50))  # 5 samples
            
            with PerformanceTimer() as timer:
                predictions = mock_model.predict(input_data)
            
            results.append({
                'timestamp': request_start,
                'latency_ms': timer.elapsed_ms,
                'throughput_sps': 5 / (timer.elapsed_ms / 1000)  # 5 samples processed
            })
            
            # Maintain target rate
            elapsed = time.time() - request_start
            sleep_time = max(0, request_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Analyze sustained performance
        total_requests = len(results)
        avg_latency = statistics.mean([r['latency_ms'] for r in results])
        p95_latency = np.percentile([r['latency_ms'] for r in results], 95)
        p99_latency = np.percentile([r['latency_ms'] for r in results], 99)
        
        actual_rps = total_requests / duration_seconds
        
        print(f"Sustained load results:")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Total requests: {total_requests}")
        print(f"  Actual RPS: {actual_rps:.1f}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  P99 latency: {p99_latency:.2f}ms")
        
        # Verify sustained performance requirements
        assert actual_rps >= target_rps * 0.9  # Within 10% of target
        assert avg_latency < 200  # Average < 200ms
        assert p95_latency < 500  # P95 < 500ms
        assert p99_latency < 1000  # P99 < 1 second


class TestSystemResourceUsage:
    """Test system resource usage under load."""
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns under different loads."""
        
        print("Testing memory usage patterns...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_measurements = []
        
        # Test different data processing loads
        data_sizes = [1000, 5000, 10000, 50000, 100000]
        
        for size in data_sizes:
            # Generate data
            data = []
            for i in range(size):
                data.append({
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': f'STOCK{i % 100}',
                    'price': 100.0 + np.random.random(),
                    'volume': np.random.randint(1000, 10000)
                })
            
            # Process data
            df = pd.DataFrame(data)
            
            # Perform operations that use memory
            df['sma_20'] = df.groupby('symbol')['price'].rolling(20).mean()
            df['volatility'] = df.groupby('symbol')['price'].rolling(20).std()
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            memory_measurements.append({
                'data_size': size,
                'memory_mb': current_memory,
                'memory_growth_mb': memory_growth,
                'memory_per_record_kb': (memory_growth * 1024) / size if size > 0 else 0
            })
            
            print(f"  Size {size:,}: {current_memory:.1f}MB total, "
                  f"{memory_growth:.1f}MB growth, "
                  f"{memory_measurements[-1]['memory_per_record_kb']:.2f}KB/record")
            
            # Clean up
            del data, df
        
        # Verify memory usage is reasonable
        max_memory_growth = max(m['memory_growth_mb'] for m in memory_measurements)
        assert max_memory_growth < 1000  # Less than 1GB growth
        
        # Memory per record should be reasonable
        avg_memory_per_record = statistics.mean([
            m['memory_per_record_kb'] for m in memory_measurements if m['data_size'] > 0
        ])
        assert avg_memory_per_record < 10  # Less than 10KB per record
        
        print(f"Maximum memory growth: {max_memory_growth:.1f}MB")
        print(f"Average memory per record: {avg_memory_per_record:.2f}KB")
    
    def test_cpu_usage_under_load(self):
        """Test CPU usage under computational load."""
        
        print("Testing CPU usage under load...")
        
        def cpu_intensive_task(duration_seconds: int = 5):
            """Run CPU-intensive task for specified duration."""
            
            start_time = time.time()
            cpu_measurements = []
            
            # Simulate feature engineering workload
            while time.time() - start_time < duration_seconds:
                # Generate random data
                data = np.random.random((1000, 50))
                
                # Perform CPU-intensive operations
                result = np.dot(data, data.T)  # Matrix multiplication
                result = np.linalg.svd(result)  # SVD decomposition
                
                # Measure CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_measurements.append(cpu_percent)
            
            return cpu_measurements
        
        # Test different workload intensities
        workload_durations = [5, 10, 15]  # seconds
        
        for duration in workload_durations:
            print(f"  Running {duration}s CPU-intensive workload...")
            
            cpu_usage = cpu_intensive_task(duration)
            
            avg_cpu = statistics.mean(cpu_usage)
            max_cpu = max(cpu_usage)
            
            print(f"    Average CPU: {avg_cpu:.1f}%")
            print(f"    Maximum CPU: {max_cpu:.1f}%")
            
            # Verify CPU usage is within reasonable bounds
            assert avg_cpu < 90  # Average CPU < 90%
            assert max_cpu <= 100  # Max CPU <= 100%
    
    def test_disk_io_performance(self):
        """Test disk I/O performance for data storage operations."""
        
        print("Testing disk I/O performance...")
        
        import tempfile
        import shutil
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test different file sizes
            file_sizes = [1, 10, 100, 1000]  # MB
            
            for size_mb in file_sizes:
                print(f"  Testing {size_mb}MB file I/O...")
                
                # Generate test data
                num_records = size_mb * 1000  # ~1KB per record
                data = []
                
                for i in range(num_records):
                    data.append({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'symbol': f'STOCK{i % 100}',
                        'price': 100.0 + np.random.random(),
                        'volume': np.random.randint(1000, 10000),
                        'features': list(np.random.random(10))
                    })
                
                df = pd.DataFrame(data)
                
                # Test write performance
                file_path = os.path.join(temp_dir, f'test_{size_mb}mb.parquet')
                
                with PerformanceTimer("Write") as write_timer:
                    df.to_parquet(file_path, compression='snappy')
                
                # Test read performance
                with PerformanceTimer("Read") as read_timer:
                    read_df = pd.read_parquet(file_path)
                
                # Calculate metrics
                file_size_actual = os.path.getsize(file_path) / 1024 / 1024  # MB
                write_throughput = file_size_actual / (write_timer.elapsed_ms / 1000)  # MB/s
                read_throughput = file_size_actual / (read_timer.elapsed_ms / 1000)  # MB/s
                
                print(f"    File size: {file_size_actual:.1f}MB")
                print(f"    Write: {write_timer.elapsed_ms:.2f}ms ({write_throughput:.1f}MB/s)")
                print(f"    Read: {read_timer.elapsed_ms:.2f}ms ({read_throughput:.1f}MB/s)")
                
                # Verify I/O performance
                assert write_throughput > 10  # At least 10 MB/s write
                assert read_throughput > 50   # At least 50 MB/s read
                assert len(read_df) == len(df)  # Data integrity
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestScalabilityBenchmarks:
    """Test system scalability under increasing load."""
    
    def test_horizontal_scaling_simulation(self):
        """Simulate horizontal scaling performance."""
        
        print("Testing horizontal scaling simulation...")
        
        def process_batch(batch_id: int, batch_size: int):
            """Process a batch of data."""
            
            # Generate batch data
            data = np.random.random((batch_size, 50))
            
            # Simulate processing time
            processing_time = 0.01 + batch_size * 0.0001  # Base + per-item time
            time.sleep(processing_time)
            
            return {
                'batch_id': batch_id,
                'batch_size': batch_size,
                'processing_time': processing_time,
                'throughput': batch_size / processing_time
            }
        
        # Test different numbers of workers
        worker_counts = [1, 2, 4, 8, 16]
        batch_size = 1000
        batches_per_worker = 10
        
        scaling_results = []
        
        for num_workers in worker_counts:
            print(f"  Testing with {num_workers} workers...")
            
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                
                for worker_id in range(num_workers):
                    for batch_num in range(batches_per_worker):
                        batch_id = worker_id * batches_per_worker + batch_num
                        future = executor.submit(process_batch, batch_id, batch_size)
                        futures.append(future)
                
                results = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_batches = len(results)
            total_items = total_batches * batch_size
            overall_throughput = total_items / total_time
            
            scaling_results.append({
                'workers': num_workers,
                'total_time': total_time,
                'total_batches': total_batches,
                'total_items': total_items,
                'throughput': overall_throughput,
                'efficiency': overall_throughput / num_workers  # Items per worker per second
            })
            
            print(f"    {num_workers} workers: {overall_throughput:.0f} items/sec, "
                  f"efficiency: {scaling_results[-1]['efficiency']:.0f} items/worker/sec")
        
        # Analyze scaling efficiency
        baseline_throughput = scaling_results[0]['throughput']
        
        for result in scaling_results[1:]:
            scaling_factor = result['throughput'] / baseline_throughput
            ideal_scaling = result['workers']
            efficiency = scaling_factor / ideal_scaling
            
            print(f"  {result['workers']} workers: {scaling_factor:.2f}x speedup "
                  f"({efficiency:.1%} efficiency)")
            
            # Verify reasonable scaling efficiency
            assert efficiency > 0.5  # At least 50% scaling efficiency
    
    def test_memory_scaling_patterns(self):
        """Test memory usage scaling patterns."""
        
        print("Testing memory scaling patterns...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory usage with increasing data sizes
        data_multipliers = [1, 2, 4, 8, 16]
        base_size = 10000
        
        memory_scaling = []
        
        for multiplier in data_multipliers:
            data_size = base_size * multiplier
            
            # Generate and process data
            data_list = []
            
            for i in range(data_size):
                data_list.append({
                    'id': i,
                    'values': list(np.random.random(20)),
                    'metadata': f'item_{i}'
                })
            
            # Convert to DataFrame and process
            df = pd.DataFrame(data_list)
            
            # Perform memory-intensive operations
            df['sum_values'] = df['values'].apply(sum)
            df['mean_values'] = df['values'].apply(np.mean)
            df['std_values'] = df['values'].apply(np.std)
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - initial_memory
            
            memory_scaling.append({
                'multiplier': multiplier,
                'data_size': data_size,
                'memory_mb': memory_used,
                'memory_per_item_kb': (memory_used * 1024) / data_size
            })
            
            print(f"  {multiplier}x data ({data_size:,} items): {memory_used:.1f}MB, "
                  f"{memory_scaling[-1]['memory_per_item_kb']:.2f}KB/item")
            
            # Clean up
            del data_list, df
        
        # Analyze memory scaling
        for i in range(1, len(memory_scaling)):
            prev_result = memory_scaling[i-1]
            curr_result = memory_scaling[i]
            
            data_ratio = curr_result['data_size'] / prev_result['data_size']
            memory_ratio = curr_result['memory_mb'] / prev_result['memory_mb']
            
            scaling_efficiency = data_ratio / memory_ratio if memory_ratio > 0 else 0
            
            print(f"  {curr_result['multiplier']}x scaling efficiency: {scaling_efficiency:.2f}")
            
            # Memory should scale reasonably with data size
            assert 0.5 <= scaling_efficiency <= 2.0  # Within reasonable bounds


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v", 
        "-s",
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])