"""
Chaos engineering tests for system resilience.

Tests system behavior under various failure conditions to validate
fault tolerance and recovery mechanisms.

Requirements: 6.6
"""

import pytest
import asyncio
import time
import random
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Callable, Optional
import numpy as np
import pandas as pd

from src.services.data_aggregator import DataAggregator
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.risk_management_service import RiskManagementService
from src.exchanges.robinhood import RobinhoodConnector
from src.exchanges.oanda import OANDAConnector
from src.exchanges.coinbase import CoinbaseConnector


class ChaosMonkey:
    """Chaos engineering utility for introducing controlled failures."""
    
    def __init__(self):
        self.active_failures = {}
        self.failure_history = []
    
    def introduce_failure(self, component: str, failure_type: str, 
                         duration: float = None, probability: float = 1.0):
        """Introduce a failure in a system component."""
        
        failure_id = f"{component}_{failure_type}_{time.time()}"
        
        failure_config = {
            'id': failure_id,
            'component': component,
            'type': failure_type,
            'start_time': time.time(),
            'duration': duration,
            'probability': probability,
            'active': True
        }
        
        self.active_failures[failure_id] = failure_config
        self.failure_history.append(failure_config)
        
        return failure_id
    
    def should_fail(self, component: str, failure_type: str = None) -> bool:
        """Check if a component should fail based on active failures."""
        
        current_time = time.time()
        
        for failure_id, config in list(self.active_failures.items()):
            if not config['active']:
                continue
            
            # Check if failure has expired
            if config['duration'] and (current_time - config['start_time']) > config['duration']:
                config['active'] = False
                del self.active_failures[failure_id]
                continue
            
            # Check if this failure applies
            if config['component'] == component:
                if failure_type is None or config['type'] == failure_type:
                    # Check probability
                    if random.random() < config['probability']:
                        return True
        
        return False
    
    def recover_component(self, component: str, failure_type: str = None):
        """Recover a component from failures."""
        
        for failure_id, config in list(self.active_failures.items()):
            if config['component'] == component:
                if failure_type is None or config['type'] == failure_type:
                    config['active'] = False
                    del self.active_failures[failure_id]
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get statistics about failures introduced."""
        
        total_failures = len(self.failure_history)
        active_failures = len(self.active_failures)
        
        failure_types = {}
        for failure in self.failure_history:
            failure_type = failure['type']
            if failure_type not in failure_types:
                failure_types[failure_type] = 0
            failure_types[failure_type] += 1
        
        return {
            'total_failures': total_failures,
            'active_failures': active_failures,
            'failure_types': failure_types
        }


class TestNetworkFailures:
    """Test resilience to network failures."""
    
    @pytest.fixture
    def chaos_monkey(self):
        """Create chaos monkey for failure injection."""
        return ChaosMonkey()
    
    @pytest.fixture
    def resilient_services(self):
        """Create services with resilience features."""
        
        # Mock services with failure simulation
        services = {
            'data_aggregator': Mock(spec=DataAggregator),
            'decision_engine': Mock(spec=TradingDecisionEngine),
            'robinhood': Mock(spec=RobinhoodConnector),
            'oanda': Mock(spec=OANDAConnector),
            'coinbase': Mock(spec=CoinbaseConnector)
        }
        
        # Add async methods
        for service in services.values():
            service.get_historical_data = AsyncMock()
            service.get_real_time_data = AsyncMock()
            service.place_order = AsyncMock()
        
        return services
    
    @pytest.mark.asyncio
    async def test_exchange_connection_timeouts(self, chaos_monkey, resilient_services):
        """Test resilience to exchange connection timeouts."""
        
        print("Testing exchange connection timeout resilience...")
        
        # Introduce timeout failures
        chaos_monkey.introduce_failure('robinhood', 'timeout', duration=10.0, probability=0.5)
        chaos_monkey.introduce_failure('oanda', 'timeout', duration=15.0, probability=0.3)
        
        # Mock timeout behavior
        def mock_get_data_with_timeout(exchange_name):
            async def mock_method(*args, **kwargs):
                if chaos_monkey.should_fail(exchange_name, 'timeout'):
                    raise asyncio.TimeoutError(f"{exchange_name} connection timeout")
                
                # Return successful data
                return pd.DataFrame({
                    'timestamp': [datetime.now(timezone.utc)],
                    'symbol': ['AAPL'],
                    'close': [150.0]
                })
            
            return mock_method
        
        # Apply timeout behavior to exchanges
        resilient_services['robinhood'].get_historical_data = mock_get_data_with_timeout('robinhood')
        resilient_services['oanda'].get_historical_data = mock_get_data_with_timeout('oanda')
        resilient_services['coinbase'].get_historical_data = mock_get_data_with_timeout('coinbase')
        
        # Test data aggregator with failover
        data_aggregator = resilient_services['data_aggregator']
        
        async def get_data_with_failover(symbol, preferred_exchange, fallback_exchanges):
            """Get data with automatic failover."""
            
            exchanges = [preferred_exchange] + fallback_exchanges
            
            for exchange_name in exchanges:
                try:
                    if exchange_name in resilient_services:
                        exchange = resilient_services[exchange_name]
                        data = await exchange.get_historical_data(symbol)
                        print(f"✓ Successfully retrieved data from {exchange_name}")
                        return data
                
                except asyncio.TimeoutError:
                    print(f"✗ Timeout from {exchange_name}, trying next...")
                    continue
                except Exception as e:
                    print(f"✗ Error from {exchange_name}: {e}")
                    continue
            
            raise Exception("All exchanges failed")
        
        # Test failover mechanism
        attempts = 20
        successful_attempts = 0
        
        for i in range(attempts):
            try:
                data = await get_data_with_failover(
                    'AAPL', 
                    'robinhood', 
                    ['oanda', 'coinbase']
                )
                
                if not data.empty:
                    successful_attempts += 1
            
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
        
        success_rate = successful_attempts / attempts
        print(f"Success rate with failover: {success_rate:.1%}")
        
        # Should achieve reasonable success rate despite failures
        assert success_rate > 0.7  # At least 70% success rate
        
        # Verify failures were introduced
        stats = chaos_monkey.get_failure_stats()
        assert stats['total_failures'] > 0
        print(f"Chaos monkey stats: {stats}")
    
    @pytest.mark.asyncio
    async def test_network_partitions(self, chaos_monkey, resilient_services):
        """Test resilience to network partitions."""
        
        print("Testing network partition resilience...")
        
        # Simulate network partition affecting multiple exchanges
        chaos_monkey.introduce_failure('robinhood', 'network_partition', duration=20.0)
        chaos_monkey.introduce_failure('oanda', 'network_partition', duration=20.0)
        
        # Mock partition behavior
        def create_partition_mock(exchange_name):
            async def mock_method(*args, **kwargs):
                if chaos_monkey.should_fail(exchange_name, 'network_partition'):
                    raise ConnectionError(f"Network partition: {exchange_name} unreachable")
                
                return pd.DataFrame({'symbol': ['TEST'], 'close': [100.0]})
            
            return mock_method
        
        # Apply partition behavior
        for exchange_name in ['robinhood', 'oanda']:
            exchange = resilient_services[exchange_name]
            exchange.get_historical_data = create_partition_mock(exchange_name)
        
        # Coinbase remains available
        resilient_services['coinbase'].get_historical_data = AsyncMock(
            return_value=pd.DataFrame({'symbol': ['TEST'], 'close': [100.0]})
        )
        
        # Test system behavior during partition
        class PartitionAwareSystem:
            def __init__(self, services):
                self.services = services
                self.degraded_mode = False
                self.available_exchanges = []
            
            async def check_exchange_availability(self):
                """Check which exchanges are available."""
                
                available = []
                
                for exchange_name, exchange in self.services.items():
                    if exchange_name in ['robinhood', 'oanda', 'coinbase']:
                        try:
                            await exchange.get_historical_data('TEST')
                            available.append(exchange_name)
                        except Exception:
                            pass
                
                self.available_exchanges = available
                
                # Enter degraded mode if less than 2 exchanges available
                if len(available) < 2:
                    self.degraded_mode = True
                    print(f"Entering degraded mode. Available exchanges: {available}")
                else:
                    self.degraded_mode = False
                    print(f"Normal mode. Available exchanges: {available}")
            
            async def get_market_data(self, symbol):
                """Get market data with partition awareness."""
                
                await self.check_exchange_availability()
                
                if not self.available_exchanges:
                    raise Exception("No exchanges available")
                
                # Use any available exchange
                exchange_name = self.available_exchanges[0]
                exchange = self.services[exchange_name]
                
                return await exchange.get_historical_data(symbol)
        
        system = PartitionAwareSystem(resilient_services)
        
        # Test system behavior over time
        test_duration = 30  # seconds
        start_time = time.time()
        
        results = []
        
        while time.time() - start_time < test_duration:
            try:
                data = await system.get_market_data('AAPL')
                
                results.append({
                    'timestamp': time.time(),
                    'success': True,
                    'degraded_mode': system.degraded_mode,
                    'available_exchanges': len(system.available_exchanges)
                })
                
            except Exception as e:
                results.append({
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e),
                    'degraded_mode': system.degraded_mode,
                    'available_exchanges': len(system.available_exchanges)
                })
            
            await asyncio.sleep(1)  # Check every second
        
        # Analyze results
        successful_requests = sum(1 for r in results if r['success'])
        total_requests = len(results)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        degraded_periods = sum(1 for r in results if r['degraded_mode'])
        degraded_ratio = degraded_periods / total_requests if total_requests > 0 else 0
        
        print(f"Partition test results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Degraded mode ratio: {degraded_ratio:.1%}")
        
        # System should maintain some level of service
        assert success_rate > 0.5  # At least 50% success rate
        assert degraded_ratio < 1.0  # Not always in degraded mode
    
    @pytest.mark.asyncio
    async def test_intermittent_connectivity(self, chaos_monkey, resilient_services):
        """Test resilience to intermittent connectivity issues."""
        
        print("Testing intermittent connectivity resilience...")
        
        # Introduce intermittent failures with varying probability
        chaos_monkey.introduce_failure('robinhood', 'intermittent', duration=60.0, probability=0.3)
        
        # Mock intermittent behavior
        async def intermittent_mock(*args, **kwargs):
            if chaos_monkey.should_fail('robinhood', 'intermittent'):
                raise ConnectionError("Intermittent connection failure")
            
            return pd.DataFrame({'symbol': ['AAPL'], 'close': [150.0]})
        
        resilient_services['robinhood'].get_historical_data = intermittent_mock
        
        # Test retry mechanism
        class RetryableService:
            def __init__(self, service):
                self.service = service
                self.max_retries = 3
                self.retry_delay = 0.1  # 100ms
            
            async def get_data_with_retry(self, symbol):
                """Get data with exponential backoff retry."""
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return await self.service.get_historical_data(symbol)
                    
                    except ConnectionError as e:
                        if attempt == self.max_retries:
                            raise e
                        
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)
                
                raise Exception("Max retries exceeded")
        
        retryable_service = RetryableService(resilient_services['robinhood'])
        
        # Test multiple requests with intermittent failures
        num_requests = 50
        successful_requests = 0
        retry_counts = []
        
        for i in range(num_requests):
            try:
                # Track retry attempts
                original_max_retries = retryable_service.max_retries
                attempt_count = 0
                
                # Monkey patch to count attempts
                original_method = retryable_service.service.get_historical_data
                
                async def counting_method(*args, **kwargs):
                    nonlocal attempt_count
                    attempt_count += 1
                    return await original_method(*args, **kwargs)
                
                retryable_service.service.get_historical_data = counting_method
                
                data = await retryable_service.get_data_with_retry('AAPL')
                
                if not data.empty:
                    successful_requests += 1
                
                retry_counts.append(attempt_count - 1)  # Subtract 1 for initial attempt
                
                # Restore original method
                retryable_service.service.get_historical_data = original_method
            
            except Exception as e:
                retry_counts.append(retryable_service.max_retries)
                print(f"Request {i+1} failed after retries: {e}")
        
        success_rate = successful_requests / num_requests
        avg_retries = sum(retry_counts) / len(retry_counts)
        
        print(f"Intermittent connectivity test results:")
        print(f"  Requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average retries: {avg_retries:.2f}")
        
        # Retry mechanism should improve success rate
        assert success_rate > 0.8  # At least 80% success with retries
        assert avg_retries > 0     # Some retries should have occurred


class TestServiceFailures:
    """Test resilience to internal service failures."""
    
    @pytest.fixture
    def chaos_monkey(self):
        return ChaosMonkey()
    
    @pytest.mark.asyncio
    async def test_ml_model_failures(self, chaos_monkey):
        """Test resilience to ML model failures."""
        
        print("Testing ML model failure resilience...")
        
        # Mock ML model with failure injection
        class FailureProneModel:
            def __init__(self, chaos_monkey):
                self.chaos_monkey = chaos_monkey
                self.fallback_predictions = {
                    'classification': [0, 1, 0],  # HOLD, BUY, SELL
                    'regression': 150.0,
                    'confidence': 0.5
                }
            
            async def predict(self, data):
                """Predict with potential failures."""
                
                if self.chaos_monkey.should_fail('ml_model', 'inference_error'):
                    raise RuntimeError("Model inference failed")
                
                if self.chaos_monkey.should_fail('ml_model', 'cuda_oom'):
                    raise RuntimeError("CUDA out of memory")
                
                if self.chaos_monkey.should_fail('ml_model', 'model_corruption'):
                    raise ValueError("Model weights corrupted")
                
                # Normal prediction
                return {
                    'classification': np.random.choice([0, 1, 2]),
                    'regression': 150.0 + np.random.normal(0, 5),
                    'confidence': np.random.uniform(0.7, 0.95)
                }
            
            async def predict_with_fallback(self, data):
                """Predict with fallback mechanism."""
                
                try:
                    return await self.predict(data)
                
                except Exception as e:
                    print(f"Model prediction failed: {e}, using fallback")
                    return self.fallback_predictions
        
        # Introduce various ML model failures
        chaos_monkey.introduce_failure('ml_model', 'inference_error', probability=0.2)
        chaos_monkey.introduce_failure('ml_model', 'cuda_oom', probability=0.1)
        chaos_monkey.introduce_failure('ml_model', 'model_corruption', probability=0.05)
        
        model = FailureProneModel(chaos_monkey)
        
        # Test predictions with failures
        num_predictions = 100
        successful_predictions = 0
        fallback_predictions = 0
        
        for i in range(num_predictions):
            try:
                result = await model.predict_with_fallback(np.random.random((10, 50)))
                
                if result == model.fallback_predictions:
                    fallback_predictions += 1
                
                successful_predictions += 1
            
            except Exception as e:
                print(f"Prediction {i+1} failed completely: {e}")
        
        success_rate = successful_predictions / num_predictions
        fallback_rate = fallback_predictions / num_predictions
        
        print(f"ML model failure test results:")
        print(f"  Total predictions: {num_predictions}")
        print(f"  Successful predictions: {successful_predictions}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Fallback rate: {fallback_rate:.1%}")
        
        # Should maintain high success rate with fallbacks
        assert success_rate > 0.95  # At least 95% success with fallbacks
        assert fallback_rate > 0    # Some fallbacks should have occurred
    
    @pytest.mark.asyncio
    async def test_database_failures(self, chaos_monkey):
        """Test resilience to database failures."""
        
        print("Testing database failure resilience...")
        
        # Mock database with failure injection
        class FailureProneDatabase:
            def __init__(self, chaos_monkey):
                self.chaos_monkey = chaos_monkey
                self.cache = {}  # Simple in-memory cache
            
            async def query(self, sql, params=None):
                """Execute query with potential failures."""
                
                if self.chaos_monkey.should_fail('database', 'connection_timeout'):
                    raise ConnectionError("Database connection timeout")
                
                if self.chaos_monkey.should_fail('database', 'query_timeout'):
                    raise TimeoutError("Query execution timeout")
                
                if self.chaos_monkey.should_fail('database', 'deadlock'):
                    raise Exception("Database deadlock detected")
                
                # Simulate successful query
                await asyncio.sleep(0.01)  # Simulate query time
                return [{'id': 1, 'data': 'test'}]
            
            async def query_with_cache(self, sql, params=None):
                """Query with cache fallback."""
                
                cache_key = f"{sql}_{params}"
                
                try:
                    result = await self.query(sql, params)
                    self.cache[cache_key] = result  # Cache successful result
                    return result
                
                except Exception as e:
                    print(f"Database query failed: {e}")
                    
                    # Try cache
                    if cache_key in self.cache:
                        print("Using cached result")
                        return self.cache[cache_key]
                    
                    # No cache available
                    raise e
        
        # Introduce database failures
        chaos_monkey.introduce_failure('database', 'connection_timeout', probability=0.15)
        chaos_monkey.introduce_failure('database', 'query_timeout', probability=0.1)
        chaos_monkey.introduce_failure('database', 'deadlock', probability=0.05)
        
        db = FailureProneDatabase(chaos_monkey)
        
        # Test queries with failures and caching
        num_queries = 100
        successful_queries = 0
        cached_queries = 0
        
        # Pre-populate cache with some queries
        for i in range(5):
            try:
                await db.query_with_cache(f"SELECT * FROM table_{i}")
            except Exception:
                pass
        
        for i in range(num_queries):
            query_sql = f"SELECT * FROM table_{i % 10}"  # Repeat some queries for cache hits
            
            try:
                # Check if result came from cache
                cache_key = f"{query_sql}_None"
                had_cache = cache_key in db.cache
                
                result = await db.query_with_cache(query_sql)
                
                if had_cache and len(result) > 0:
                    cached_queries += 1
                
                successful_queries += 1
            
            except Exception as e:
                print(f"Query {i+1} failed completely: {e}")
        
        success_rate = successful_queries / num_queries
        cache_hit_rate = cached_queries / successful_queries if successful_queries > 0 else 0
        
        print(f"Database failure test results:")
        print(f"  Total queries: {num_queries}")
        print(f"  Successful queries: {successful_queries}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Cache hit rate: {cache_hit_rate:.1%}")
        
        # Should maintain reasonable success rate with caching
        assert success_rate > 0.8   # At least 80% success with cache
        assert cache_hit_rate > 0   # Some cache hits should occur
    
    @pytest.mark.asyncio
    async def test_cascading_failures(self, chaos_monkey):
        """Test resilience to cascading failures."""
        
        print("Testing cascading failure resilience...")
        
        # Mock system with interdependent services
        class InterdependentSystem:
            def __init__(self, chaos_monkey):
                self.chaos_monkey = chaos_monkey
                self.circuit_breakers = {}
                self.service_health = {
                    'data_service': True,
                    'ml_service': True,
                    'trading_service': True
                }
            
            def get_circuit_breaker(self, service_name):
                """Get or create circuit breaker for service."""
                
                if service_name not in self.circuit_breakers:
                    self.circuit_breakers[service_name] = {
                        'state': 'closed',  # closed, open, half_open
                        'failure_count': 0,
                        'last_failure_time': 0,
                        'failure_threshold': 5,
                        'timeout': 10  # seconds
                    }
                
                return self.circuit_breakers[service_name]
            
            async def call_service(self, service_name, operation):
                """Call service with circuit breaker protection."""
                
                cb = self.get_circuit_breaker(service_name)
                current_time = time.time()
                
                # Check circuit breaker state
                if cb['state'] == 'open':
                    if current_time - cb['last_failure_time'] > cb['timeout']:
                        cb['state'] = 'half_open'
                        print(f"Circuit breaker for {service_name} is half-open")
                    else:
                        raise Exception(f"Circuit breaker open for {service_name}")
                
                try:
                    # Simulate service call with potential failure
                    if self.chaos_monkey.should_fail(service_name, 'service_error'):
                        raise Exception(f"{service_name} operation failed")
                    
                    # Successful call
                    if cb['state'] == 'half_open':
                        cb['state'] = 'closed'
                        cb['failure_count'] = 0
                        print(f"Circuit breaker for {service_name} closed")
                    
                    return f"{service_name}_{operation}_result"
                
                except Exception as e:
                    cb['failure_count'] += 1
                    cb['last_failure_time'] = current_time
                    
                    if cb['failure_count'] >= cb['failure_threshold']:
                        cb['state'] = 'open'
                        print(f"Circuit breaker for {service_name} opened")
                    
                    raise e
            
            async def execute_trading_workflow(self):
                """Execute complete trading workflow."""
                
                try:
                    # Step 1: Get market data
                    data = await self.call_service('data_service', 'get_market_data')
                    
                    # Step 2: Generate ML prediction
                    prediction = await self.call_service('ml_service', 'predict')
                    
                    # Step 3: Execute trade
                    trade_result = await self.call_service('trading_service', 'execute_trade')
                    
                    return {
                        'success': True,
                        'data': data,
                        'prediction': prediction,
                        'trade': trade_result
                    }
                
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e)
                    }
        
        # Introduce cascading failures
        chaos_monkey.introduce_failure('data_service', 'service_error', probability=0.2)
        chaos_monkey.introduce_failure('ml_service', 'service_error', probability=0.15)
        chaos_monkey.introduce_failure('trading_service', 'service_error', probability=0.1)
        
        system = InterdependentSystem(chaos_monkey)
        
        # Test workflow execution with cascading failures
        num_workflows = 200
        successful_workflows = 0
        circuit_breaker_activations = 0
        
        for i in range(num_workflows):
            result = await system.execute_trading_workflow()
            
            if result['success']:
                successful_workflows += 1
            
            # Check for circuit breaker activations
            for service_name, cb in system.circuit_breakers.items():
                if cb['state'] == 'open':
                    circuit_breaker_activations += 1
                    break
            
            # Small delay between workflows
            await asyncio.sleep(0.01)
        
        success_rate = successful_workflows / num_workflows
        cb_activation_rate = circuit_breaker_activations / num_workflows
        
        print(f"Cascading failure test results:")
        print(f"  Total workflows: {num_workflows}")
        print(f"  Successful workflows: {successful_workflows}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Circuit breaker activations: {circuit_breaker_activations}")
        print(f"  CB activation rate: {cb_activation_rate:.1%}")
        
        # Circuit breakers should prevent complete system failure
        assert success_rate > 0.4  # At least 40% success despite cascading failures
        assert cb_activation_rate > 0  # Circuit breakers should activate


class TestResourceExhaustion:
    """Test resilience to resource exhaustion scenarios."""
    
    @pytest.fixture
    def chaos_monkey(self):
        return ChaosMonkey()
    
    def test_memory_pressure_handling(self, chaos_monkey):
        """Test handling of memory pressure scenarios."""
        
        print("Testing memory pressure handling...")
        
        import psutil
        
        # Mock memory monitor
        class MemoryMonitor:
            def __init__(self, chaos_monkey):
                self.chaos_monkey = chaos_monkey
                self.pressure_threshold = 0.85  # 85% memory usage
            
            def get_memory_usage(self):
                """Get current memory usage percentage."""
                
                if self.chaos_monkey.should_fail('system', 'memory_pressure'):
                    return 0.95  # Simulate high memory usage
                
                return psutil.virtual_memory().percent / 100
            
            def is_under_pressure(self):
                """Check if system is under memory pressure."""
                return self.get_memory_usage() > self.pressure_threshold
        
        # Mock resource manager
        class ResourceManager:
            def __init__(self, memory_monitor):
                self.memory_monitor = memory_monitor
                self.cache_size = 1000
                self.cache = {}
            
            def handle_memory_pressure(self):
                """Handle memory pressure by clearing caches."""
                
                if self.memory_monitor.is_under_pressure():
                    print("Memory pressure detected, clearing caches...")
                    
                    # Clear half of the cache
                    items_to_remove = len(self.cache) // 2
                    keys_to_remove = list(self.cache.keys())[:items_to_remove]
                    
                    for key in keys_to_remove:
                        del self.cache[key]
                    
                    return True
                
                return False
            
            def process_data(self, data_id, data_size=1000):
                """Process data with memory management."""
                
                # Check memory before processing
                if self.memory_monitor.is_under_pressure():
                    self.handle_memory_pressure()
                
                # Simulate data processing
                processed_data = list(range(data_size))
                
                # Cache result if memory allows
                if not self.memory_monitor.is_under_pressure():
                    self.cache[data_id] = processed_data
                
                return len(processed_data)
        
        # Introduce memory pressure
        chaos_monkey.introduce_failure('system', 'memory_pressure', duration=30.0, probability=0.3)
        
        monitor = MemoryMonitor(chaos_monkey)
        manager = ResourceManager(monitor)
        
        # Test data processing under memory pressure
        num_operations = 100
        successful_operations = 0
        cache_clears = 0
        
        for i in range(num_operations):
            try:
                # Check if memory pressure handling occurred
                initial_cache_size = len(manager.cache)
                
                result = manager.process_data(f"data_{i}", data_size=500)
                
                if len(manager.cache) < initial_cache_size:
                    cache_clears += 1
                
                if result > 0:
                    successful_operations += 1
            
            except Exception as e:
                print(f"Operation {i+1} failed: {e}")
        
        success_rate = successful_operations / num_operations
        cache_clear_rate = cache_clears / num_operations
        
        print(f"Memory pressure test results:")
        print(f"  Total operations: {num_operations}")
        print(f"  Successful operations: {successful_operations}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Cache clears: {cache_clears}")
        print(f"  Cache clear rate: {cache_clear_rate:.1%}")
        
        # System should handle memory pressure gracefully
        assert success_rate > 0.9  # At least 90% success
        assert cache_clear_rate > 0  # Some cache clearing should occur
    
    def test_cpu_saturation_handling(self, chaos_monkey):
        """Test handling of CPU saturation scenarios."""
        
        print("Testing CPU saturation handling...")
        
        # Mock CPU monitor
        class CPUMonitor:
            def __init__(self, chaos_monkey):
                self.chaos_monkey = chaos_monkey
                self.saturation_threshold = 0.9  # 90% CPU usage
            
            def get_cpu_usage(self):
                """Get current CPU usage percentage."""
                
                if self.chaos_monkey.should_fail('system', 'cpu_saturation'):
                    return 0.98  # Simulate high CPU usage
                
                return psutil.cpu_percent(interval=0.1) / 100
            
            def is_saturated(self):
                """Check if CPU is saturated."""
                return self.get_cpu_usage() > self.saturation_threshold
        
        # Mock task scheduler with load balancing
        class TaskScheduler:
            def __init__(self, cpu_monitor):
                self.cpu_monitor = cpu_monitor
                self.task_queue = []
                self.max_concurrent_tasks = 4
                self.current_tasks = 0
            
            async def execute_task(self, task_id, complexity=1):
                """Execute task with CPU load management."""
                
                # Check CPU saturation
                if self.cpu_monitor.is_saturated():
                    print(f"CPU saturated, queuing task {task_id}")
                    self.task_queue.append((task_id, complexity))
                    return None
                
                # Check concurrent task limit
                if self.current_tasks >= self.max_concurrent_tasks:
                    print(f"Max concurrent tasks reached, queuing task {task_id}")
                    self.task_queue.append((task_id, complexity))
                    return None
                
                # Execute task
                self.current_tasks += 1
                
                try:
                    # Simulate CPU-intensive work
                    work_time = 0.01 * complexity
                    await asyncio.sleep(work_time)
                    
                    result = f"task_{task_id}_completed"
                    
                    return result
                
                finally:
                    self.current_tasks -= 1
                    
                    # Process queued tasks if CPU allows
                    await self.process_queue()
            
            async def process_queue(self):
                """Process queued tasks when resources become available."""
                
                while (self.task_queue and 
                       not self.cpu_monitor.is_saturated() and 
                       self.current_tasks < self.max_concurrent_tasks):
                    
                    task_id, complexity = self.task_queue.pop(0)
                    print(f"Processing queued task {task_id}")
                    
                    # Execute queued task
                    await self.execute_task(task_id, complexity)
        
        # Introduce CPU saturation
        chaos_monkey.introduce_failure('system', 'cpu_saturation', duration=20.0, probability=0.4)
        
        monitor = CPUMonitor(chaos_monkey)
        scheduler = TaskScheduler(monitor)
        
        # Test task execution under CPU saturation
        num_tasks = 50
        completed_tasks = 0
        queued_tasks = 0
        
        tasks = []
        
        for i in range(num_tasks):
            task = asyncio.create_task(scheduler.execute_task(i, complexity=random.randint(1, 3)))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, str) and 'completed' in result:
                completed_tasks += 1
            elif result is None:
                queued_tasks += 1
        
        completion_rate = completed_tasks / num_tasks
        queue_rate = queued_tasks / num_tasks
        
        print(f"CPU saturation test results:")
        print(f"  Total tasks: {num_tasks}")
        print(f"  Completed tasks: {completed_tasks}")
        print(f"  Queued tasks: {queued_tasks}")
        print(f"  Completion rate: {completion_rate:.1%}")
        print(f"  Queue rate: {queue_rate:.1%}")
        
        # System should handle CPU saturation by queuing tasks
        assert completion_rate > 0.7  # At least 70% completion
        assert queue_rate > 0  # Some tasks should be queued during saturation


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "-s", 
        "--tb=short"
    ])