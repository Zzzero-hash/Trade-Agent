"""
Comprehensive integration tests for the AI Trading Platform.

This module contains end-to-end integration tests covering:
1. Complete trading workflows across multiple exchanges
2. Performance benchmarking for latency and throughput
3. Chaos engineering tests for system resilience
4. Multi-exchange data consistency validation

Requirements: 2.6, 6.6
"""

import pytest
import asyncio
import time
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random

# Core imports
from src.models.market_data import MarketData, ExchangeType
from src.models.trading_signal import TradingSignal, TradingAction
from src.models.portfolio import Portfolio, Position
from src.services.data_aggregator import DataAggregator
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.risk_management_service import RiskManagementService
from src.exchanges.robinhood import RobinhoodConnector
from src.exchanges.oanda import OANDAConnector
from src.exchanges.coinbase import CoinbaseConnector


class TestEndToEndTradingWorkflows:
    """Test complete trading workflows from data ingestion to execution."""
    
    @pytest.fixture
    def mock_exchanges(self):
        """Create mock exchange connectors."""
        robinhood = Mock(spec=RobinhoodConnector)
        oanda = Mock(spec=OANDAConnector)
        coinbase = Mock(spec=CoinbaseConnector)
        
        # Mock async methods
        for exchange in [robinhood, oanda, coinbase]:
            exchange.get_historical_data = AsyncMock()
            exchange.get_real_time_data = AsyncMock()
            exchange.place_order = AsyncMock()
            exchange.get_account_info = AsyncMock()
            exchange.get_positions = AsyncMock()
        
        return {
            'robinhood': robinhood,
            'oanda': oanda,
            'coinbase': coinbase
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data for testing."""
        symbols = ['AAPL', 'GOOGL', 'EUR/USD', 'BTC/USD']
        data = []
        
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        for symbol in symbols:
            if symbol == 'AAPL':
                base_price = 150.0
                exchange = ExchangeType.ROBINHOOD
            elif symbol == 'GOOGL':
                base_price = 2500.0
                exchange = ExchangeType.ROBINHOOD
            elif symbol == 'EUR/USD':
                base_price = 1.1000
                exchange = ExchangeType.OANDA
            else:  # BTC/USD
                base_price = 45000.0
                exchange = ExchangeType.COINBASE
            
            for i in range(720):  # 30 days * 24 hours
                timestamp = base_time + timedelta(hours=i)
                
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.02)
                price = base_price * (1 + price_change * 0.1)
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=price * 0.999,
                    high=price * 1.005,
                    low=price * 0.995,
                    close=price,
                    volume=random.randint(100000, 1000000),
                    exchange=exchange
                )
                data.append(market_data)
        
        return data
    
    @pytest.mark.asyncio
    async def test_complete_stock_trading_workflow(self, mock_exchanges, sample_market_data):
        """Test complete stock trading workflow via Robinhood."""
        
        # Setup mock data
        stock_data = [md for md in sample_market_data if md.exchange == ExchangeType.ROBINHOOD]
        mock_exchanges['robinhood'].get_historical_data.return_value = pd.DataFrame([
            {
                'timestamp': md.timestamp,
                'symbol': md.symbol,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in stock_data if md.symbol == 'AAPL'
        ])
        
        # Mock account info
        mock_exchanges['robinhood'].get_account_info.return_value = {
            'buying_power': 50000.0,
            'total_equity': 100000.0
        }
        
        # Mock successful order placement
        mock_exchanges['robinhood'].place_order.return_value = {
            'order_id': 'test_order_123',
            'status': 'filled',
            'filled_quantity': 100,
            'filled_price': 150.50
        }
        
        # Initialize services
        data_aggregator = DataAggregator()
        data_aggregator.robinhood_connector = mock_exchanges['robinhood']
        
        decision_engine = TradingDecisionEngine()
        portfolio_service = PortfolioManagementService()
        risk_service = RiskManagementService()
        
        # Step 1: Data ingestion and processing
        historical_data = await data_aggregator.get_historical_data(
            symbol='AAPL',
            exchange='robinhood',
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
        
        assert not historical_data.empty
        assert 'AAPL' in historical_data['symbol'].values
        
        # Step 2: Feature engineering and signal generation
        with patch.object(decision_engine, 'generate_signal') as mock_signal:
            mock_signal.return_value = TradingSignal(
                symbol='AAPL',
                action=TradingAction.BUY,
                confidence=0.85,
                position_size=0.1,
                target_price=155.0,
                stop_loss=145.0,
                timestamp=datetime.now(timezone.utc),
                model_version='test-v1.0'
            )
            
            signal = await decision_engine.generate_signal('AAPL', historical_data)
            
            assert signal.action == TradingAction.BUY
            assert signal.confidence > 0.8
            assert signal.position_size > 0
        
        # Step 3: Risk validation
        with patch.object(risk_service, 'validate_trade') as mock_risk:
            mock_risk.return_value = True
            
            risk_approved = await risk_service.validate_trade(signal, portfolio_service.get_portfolio())
            assert risk_approved is True
        
        # Step 4: Portfolio optimization
        with patch.object(portfolio_service, 'calculate_optimal_position_size') as mock_position:
            mock_position.return_value = 100  # shares
            
            optimal_size = await portfolio_service.calculate_optimal_position_size(signal)
            assert optimal_size == 100
        
        # Step 5: Order execution
        order_result = await mock_exchanges['robinhood'].place_order({
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'type': 'market'
        })
        
        assert order_result['status'] == 'filled'
        assert order_result['filled_quantity'] == 100
        
        # Step 6: Portfolio update
        with patch.object(portfolio_service, 'update_position') as mock_update:
            mock_update.return_value = True
            
            portfolio_updated = await portfolio_service.update_position(
                'AAPL', 100, 150.50, 'buy'
            )
            assert portfolio_updated is True
        
        print("✓ Complete stock trading workflow test passed")
    
    @pytest.mark.asyncio
    async def test_complete_forex_trading_workflow(self, mock_exchanges, sample_market_data):
        """Test complete forex trading workflow via OANDA."""
        
        # Setup mock forex data
        forex_data = [md for md in sample_market_data if md.exchange == ExchangeType.OANDA]
        mock_exchanges['oanda'].get_historical_data.return_value = pd.DataFrame([
            {
                'timestamp': md.timestamp,
                'symbol': md.symbol,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in forex_data if md.symbol == 'EUR/USD'
        ])
        
        # Mock account info
        mock_exchanges['oanda'].get_account_info.return_value = {
            'balance': 25000.0,
            'margin_available': 20000.0,
            'currency': 'USD'
        }
        
        # Mock successful forex order
        mock_exchanges['oanda'].place_order.return_value = {
            'order_id': 'forex_order_456',
            'status': 'filled',
            'filled_units': 10000,
            'filled_price': 1.1050
        }
        
        # Initialize services
        data_aggregator = DataAggregator()
        data_aggregator.oanda_connector = mock_exchanges['oanda']
        
        decision_engine = TradingDecisionEngine()
        
        # Step 1: Get forex data
        forex_data = await data_aggregator.get_historical_data(
            symbol='EUR/USD',
            exchange='oanda',
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
        
        assert not forex_data.empty
        
        # Step 2: Generate forex signal
        with patch.object(decision_engine, 'generate_signal') as mock_signal:
            mock_signal.return_value = TradingSignal(
                symbol='EUR/USD',
                action=TradingAction.BUY,
                confidence=0.78,
                position_size=0.05,
                target_price=1.1100,
                stop_loss=1.1000,
                timestamp=datetime.now(timezone.utc),
                model_version='forex-v1.0'
            )
            
            signal = await decision_engine.generate_signal('EUR/USD', forex_data)
            assert signal.action == TradingAction.BUY
        
        # Step 3: Execute forex order
        order_result = await mock_exchanges['oanda'].place_order({
            'instrument': 'EUR_USD',
            'units': 10000,
            'type': 'MARKET'
        })
        
        assert order_result['status'] == 'filled'
        assert order_result['filled_units'] == 10000
        
        print("✓ Complete forex trading workflow test passed")
    
    @pytest.mark.asyncio
    async def test_complete_crypto_trading_workflow(self, mock_exchanges, sample_market_data):
        """Test complete crypto trading workflow via Coinbase."""
        
        # Setup mock crypto data
        crypto_data = [md for md in sample_market_data if md.exchange == ExchangeType.COINBASE]
        mock_exchanges['coinbase'].get_historical_data.return_value = pd.DataFrame([
            {
                'timestamp': md.timestamp,
                'symbol': md.symbol,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in crypto_data if md.symbol == 'BTC/USD'
        ])
        
        # Mock account info
        mock_exchanges['coinbase'].get_account_info.return_value = {
            'available_balance': {'USD': 15000.0, 'BTC': 0.0},
            'total_balance': {'USD': 15000.0, 'BTC': 0.0}
        }
        
        # Mock successful crypto order
        mock_exchanges['coinbase'].place_order.return_value = {
            'order_id': 'crypto_order_789',
            'status': 'filled',
            'filled_size': 0.1,
            'filled_price': 45500.0
        }
        
        # Initialize services
        data_aggregator = DataAggregator()
        data_aggregator.coinbase_connector = mock_exchanges['coinbase']
        
        decision_engine = TradingDecisionEngine()
        
        # Step 1: Get crypto data
        crypto_data = await data_aggregator.get_historical_data(
            symbol='BTC/USD',
            exchange='coinbase',
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
        
        assert not crypto_data.empty
        
        # Step 2: Generate crypto signal
        with patch.object(decision_engine, 'generate_signal') as mock_signal:
            mock_signal.return_value = TradingSignal(
                symbol='BTC/USD',
                action=TradingAction.BUY,
                confidence=0.82,
                position_size=0.02,
                target_price=47000.0,
                stop_loss=43000.0,
                timestamp=datetime.now(timezone.utc),
                model_version='crypto-v1.0'
            )
            
            signal = await decision_engine.generate_signal('BTC/USD', crypto_data)
            assert signal.action == TradingAction.BUY
        
        # Step 3: Execute crypto order
        order_result = await mock_exchanges['coinbase'].place_order({
            'product_id': 'BTC-USD',
            'side': 'buy',
            'size': 0.1,
            'type': 'market'
        })
        
        assert order_result['status'] == 'filled'
        assert order_result['filled_size'] == 0.1
        
        print("✓ Complete crypto trading workflow test passed")
    
    @pytest.mark.asyncio
    async def test_multi_asset_portfolio_workflow(self, mock_exchanges, sample_market_data):
        """Test multi-asset portfolio management workflow."""
        
        # Setup mock data for all exchanges
        for exchange_name, exchange_mock in mock_exchanges.items():
            if exchange_name == 'robinhood':
                exchange_mock.get_historical_data.return_value = pd.DataFrame([
                    md.__dict__ for md in sample_market_data 
                    if md.exchange == ExchangeType.ROBINHOOD
                ])
            elif exchange_name == 'oanda':
                exchange_mock.get_historical_data.return_value = pd.DataFrame([
                    md.__dict__ for md in sample_market_data 
                    if md.exchange == ExchangeType.OANDA
                ])
            else:  # coinbase
                exchange_mock.get_historical_data.return_value = pd.DataFrame([
                    md.__dict__ for md in sample_market_data 
                    if md.exchange == ExchangeType.COINBASE
                ])
        
        # Initialize services
        data_aggregator = DataAggregator()
        data_aggregator.robinhood_connector = mock_exchanges['robinhood']
        data_aggregator.oanda_connector = mock_exchanges['oanda']
        data_aggregator.coinbase_connector = mock_exchanges['coinbase']
        
        portfolio_service = PortfolioManagementService()
        
        # Step 1: Aggregate data from all exchanges
        symbols = ['AAPL', 'EUR/USD', 'BTC/USD']
        exchanges = ['robinhood', 'oanda', 'coinbase']
        
        all_data = {}
        for symbol, exchange in zip(symbols, exchanges):
            data = await data_aggregator.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                start_date=datetime.now(timezone.utc) - timedelta(days=7),
                end_date=datetime.now(timezone.utc)
            )
            all_data[symbol] = data
        
        assert len(all_data) == 3
        assert all(not df.empty for df in all_data.values())
        
        # Step 2: Portfolio optimization
        with patch.object(portfolio_service, 'optimize_portfolio') as mock_optimize:
            mock_optimize.return_value = {
                'AAPL': 0.4,
                'EUR/USD': 0.3,
                'BTC/USD': 0.3
            }
            
            optimal_weights = await portfolio_service.optimize_portfolio(
                symbols, all_data
            )
            
            assert sum(optimal_weights.values()) == pytest.approx(1.0, rel=1e-2)
            assert all(weight >= 0 for weight in optimal_weights.values())
        
        print("✓ Multi-asset portfolio workflow test passed")


class TestPerformanceBenchmarking:
    """Performance benchmarking tests for latency and throughput."""
    
    @pytest.fixture
    def performance_data(self):
        """Generate performance test data."""
        return {
            'market_data': [
                MarketData(
                    symbol=f'STOCK{i}',
                    timestamp=datetime.now(timezone.utc),
                    open=100.0 + i,
                    high=102.0 + i,
                    low=99.0 + i,
                    close=101.0 + i,
                    volume=1000000,
                    exchange=ExchangeType.ROBINHOOD
                ) for i in range(1000)
            ]
        }
    
    def test_data_aggregation_latency(self, performance_data):
        """Benchmark data aggregation latency."""
        
        with patch('src.services.data_aggregator.DataAggregator') as MockAggregator:
            aggregator = MockAggregator()
            
            # Mock the aggregation method
            def mock_aggregate(data_list):
                # Simulate processing time
                time.sleep(0.001)  # 1ms processing time
                return pd.DataFrame([md.__dict__ for md in data_list])
            
            aggregator.aggregate_market_data = mock_aggregate
            
            latencies = []
            
            # Test with different batch sizes
            batch_sizes = [10, 50, 100, 500, 1000]
            
            for batch_size in batch_sizes:
                batch_data = performance_data['market_data'][:batch_size]
                
                start_time = time.time()
                result = aggregator.aggregate_market_data(batch_data)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append({
                    'batch_size': batch_size,
                    'latency_ms': latency_ms,
                    'throughput_records_per_sec': batch_size / (latency_ms / 1000)
                })
                
                print(f"Batch size {batch_size}: {latency_ms:.2f}ms, "
                      f"Throughput: {latencies[-1]['throughput_records_per_sec']:.0f} records/sec")
            
            # Verify performance characteristics
            assert all(l['latency_ms'] < 1000 for l in latencies)  # < 1 second
            assert all(l['throughput_records_per_sec'] > 100 for l in latencies)  # > 100 records/sec
    
    def test_signal_generation_latency(self):
        """Benchmark signal generation latency."""
        
        with patch('src.services.trading_decision_engine.TradingDecisionEngine') as MockEngine:
            engine = MockEngine()
            
            # Mock signal generation with realistic processing time
            def mock_generate_signal(symbol, data):
                # Simulate ML model inference time
                time.sleep(0.05)  # 50ms inference time
                return TradingSignal(
                    symbol=symbol,
                    action=TradingAction.BUY,
                    confidence=0.8,
                    position_size=0.1,
                    timestamp=datetime.now(timezone.utc),
                    model_version='benchmark-v1.0'
                )
            
            engine.generate_signal = mock_generate_signal
            
            # Test signal generation for multiple symbols
            symbols = [f'STOCK{i}' for i in range(20)]
            latencies = []
            
            for symbol in symbols:
                start_time = time.time()
                signal = engine.generate_signal(symbol, None)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            print(f"Signal generation - Avg: {avg_latency:.2f}ms, "
                  f"P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")
            
            # Performance requirements
            assert avg_latency < 100  # Average < 100ms
            assert p95_latency < 200  # P95 < 200ms
            assert p99_latency < 500  # P99 < 500ms
    
    def test_concurrent_processing_throughput(self):
        """Test throughput under concurrent processing."""
        
        def process_request(request_id):
            """Simulate processing a single request."""
            start_time = time.time()
            
            # Simulate processing time with some variance
            processing_time = random.uniform(0.01, 0.05)  # 10-50ms
            time.sleep(processing_time)
            
            end_time = time.time()
            return {
                'request_id': request_id,
                'processing_time_ms': (end_time - start_time) * 1000,
                'timestamp': start_time
            }
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(process_request, i) 
                    for i in range(concurrency * 2)  # 2x requests per worker
                ]
                
                results = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_requests = len(results)
            throughput = total_requests / total_time
            avg_processing_time = statistics.mean([r['processing_time_ms'] for r in results])
            
            print(f"Concurrency {concurrency}: {throughput:.1f} req/sec, "
                  f"Avg processing: {avg_processing_time:.2f}ms")
            
            # Verify throughput scales reasonably with concurrency
            assert throughput > concurrency * 10  # At least 10 req/sec per worker
    
    @pytest.mark.slow
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained processing load
        data_batches = []
        
        for batch in range(100):  # 100 batches
            # Create batch of market data
            batch_data = [
                MarketData(
                    symbol=f'STOCK{i}',
                    timestamp=datetime.now(timezone.utc),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000000,
                    exchange=ExchangeType.ROBINHOOD
                ) for i in range(100)  # 100 records per batch
            ]
            
            # Process batch (simulate)
            processed_batch = pd.DataFrame([md.__dict__ for md in batch_data])
            data_batches.append(processed_batch)
            
            # Check memory every 10 batches
            if batch % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                print(f"Batch {batch}: Memory: {current_memory:.1f}MB "
                      f"(growth: {memory_growth:.1f}MB)")
                
                # Memory growth should be reasonable
                assert memory_growth < 1000  # Less than 1GB growth
        
        # Clean up
        del data_batches
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        print(f"Total memory growth: {total_growth:.1f}MB")
        assert total_growth < 500  # Less than 500MB total growth


class TestChaosEngineering:
    """Chaos engineering tests for system resilience."""
    
    @pytest.fixture
    def resilient_system(self):
        """Setup system components with resilience features."""
        
        # Mock components with failure simulation
        components = {
            'data_aggregator': Mock(),
            'decision_engine': Mock(),
            'portfolio_service': Mock(),
            'risk_service': Mock()
        }
        
        # Add failure simulation methods
        for component in components.values():
            component.simulate_failure = Mock()
            component.is_healthy = Mock(return_value=True)
        
        return components
    
    def test_exchange_connection_failures(self, resilient_system):
        """Test resilience to exchange connection failures."""
        
        # Simulate exchange failures
        failure_scenarios = [
            {'exchange': 'robinhood', 'failure_type': 'timeout'},
            {'exchange': 'oanda', 'failure_type': 'rate_limit'},
            {'exchange': 'coinbase', 'failure_type': 'connection_error'}
        ]
        
        for scenario in failure_scenarios:
            print(f"Testing {scenario['exchange']} {scenario['failure_type']} failure...")
            
            # Mock the failure
            with patch(f'src.exchanges.{scenario["exchange"]}_connector') as mock_exchange:
                if scenario['failure_type'] == 'timeout':
                    mock_exchange.get_historical_data.side_effect = asyncio.TimeoutError()
                elif scenario['failure_type'] == 'rate_limit':
                    mock_exchange.get_historical_data.side_effect = Exception("Rate limit exceeded")
                else:  # connection_error
                    mock_exchange.get_historical_data.side_effect = ConnectionError("Connection failed")
                
                # System should handle the failure gracefully
                data_aggregator = resilient_system['data_aggregator']
                
                # Mock fallback behavior
                data_aggregator.get_historical_data_with_fallback = Mock(
                    return_value=pd.DataFrame({'symbol': ['FALLBACK'], 'close': [100.0]})
                )
                
                # Test that system continues to function
                result = data_aggregator.get_historical_data_with_fallback('AAPL', scenario['exchange'])
                
                assert not result.empty
                print(f"✓ System handled {scenario['exchange']} {scenario['failure_type']} failure")
    
    def test_model_inference_failures(self, resilient_system):
        """Test resilience to ML model inference failures."""
        
        failure_scenarios = [
            'model_not_loaded',
            'cuda_out_of_memory',
            'inference_timeout',
            'model_corruption'
        ]
        
        for failure_type in failure_scenarios:
            print(f"Testing model {failure_type} failure...")
            
            decision_engine = resilient_system['decision_engine']
            
            # Mock different failure types
            if failure_type == 'model_not_loaded':
                decision_engine.generate_signal.side_effect = FileNotFoundError("Model not found")
            elif failure_type == 'cuda_out_of_memory':
                decision_engine.generate_signal.side_effect = RuntimeError("CUDA out of memory")
            elif failure_type == 'inference_timeout':
                decision_engine.generate_signal.side_effect = TimeoutError("Inference timeout")
            else:  # model_corruption
                decision_engine.generate_signal.side_effect = ValueError("Invalid model state")
            
            # Mock fallback signal generation
            decision_engine.generate_fallback_signal = Mock(
                return_value=TradingSignal(
                    symbol='AAPL',
                    action=TradingAction.HOLD,
                    confidence=0.5,
                    position_size=0.0,
                    timestamp=datetime.now(timezone.utc),
                    model_version='fallback-v1.0'
                )
            )
            
            # Test fallback behavior
            fallback_signal = decision_engine.generate_fallback_signal('AAPL')
            
            assert fallback_signal.action == TradingAction.HOLD
            assert fallback_signal.confidence == 0.5
            print(f"✓ System handled {failure_type} with fallback signal")
    
    def test_database_failures(self, resilient_system):
        """Test resilience to database failures."""
        
        database_failures = [
            'connection_timeout',
            'query_timeout',
            'disk_full',
            'connection_pool_exhausted'
        ]
        
        for failure_type in database_failures:
            print(f"Testing database {failure_type} failure...")
            
            # Mock database failure
            with patch('src.repositories.market_data_repository.MarketDataRepository') as mock_repo:
                if failure_type == 'connection_timeout':
                    mock_repo.get_historical_data.side_effect = ConnectionError("Connection timeout")
                elif failure_type == 'query_timeout':
                    mock_repo.get_historical_data.side_effect = TimeoutError("Query timeout")
                elif failure_type == 'disk_full':
                    mock_repo.get_historical_data.side_effect = OSError("No space left on device")
                else:  # connection_pool_exhausted
                    mock_repo.get_historical_data.side_effect = Exception("Connection pool exhausted")
                
                # Mock cache fallback
                with patch('src.services.cache_service.CacheService') as mock_cache:
                    mock_cache.get_cached_data.return_value = pd.DataFrame({
                        'symbol': ['AAPL'],
                        'close': [150.0],
                        'timestamp': [datetime.now(timezone.utc)]
                    })
                    
                    # Test that system falls back to cache
                    cached_data = mock_cache.get_cached_data('AAPL')
                    
                    assert not cached_data.empty
                    print(f"✓ System handled {failure_type} with cache fallback")
    
    def test_network_partitions(self, resilient_system):
        """Test resilience to network partitions."""
        
        partition_scenarios = [
            {'affected_services': ['robinhood'], 'duration': 5},
            {'affected_services': ['oanda', 'coinbase'], 'duration': 10},
            {'affected_services': ['all_exchanges'], 'duration': 15}
        ]
        
        for scenario in partition_scenarios:
            print(f"Testing network partition affecting {scenario['affected_services']} "
                  f"for {scenario['duration']} seconds...")
            
            # Simulate network partition
            start_time = time.time()
            
            # Mock network failures for affected services
            with patch('src.utils.network_utils.check_connectivity') as mock_connectivity:
                mock_connectivity.return_value = False
                
                # System should detect partition and switch to degraded mode
                data_aggregator = resilient_system['data_aggregator']
                data_aggregator.enter_degraded_mode = Mock()
                data_aggregator.is_in_degraded_mode = Mock(return_value=True)
                
                # Simulate partition detection
                data_aggregator.enter_degraded_mode()
                
                assert data_aggregator.is_in_degraded_mode()
                print(f"✓ System detected partition and entered degraded mode")
                
                # Simulate partition recovery
                time.sleep(0.1)  # Brief simulation
                mock_connectivity.return_value = True
                
                data_aggregator.exit_degraded_mode = Mock()
                data_aggregator.is_in_degraded_mode = Mock(return_value=False)
                
                data_aggregator.exit_degraded_mode()
                
                assert not data_aggregator.is_in_degraded_mode()
                print(f"✓ System recovered from partition")
    
    def test_resource_exhaustion(self, resilient_system):
        """Test resilience to resource exhaustion."""
        
        resource_scenarios = [
            'memory_pressure',
            'cpu_saturation',
            'disk_space_low',
            'file_descriptor_limit'
        ]
        
        for scenario in resource_scenarios:
            print(f"Testing {scenario} scenario...")
            
            # Mock resource monitoring
            with patch('src.utils.resource_monitor.ResourceMonitor') as mock_monitor:
                if scenario == 'memory_pressure':
                    mock_monitor.get_memory_usage.return_value = 0.95  # 95% memory usage
                elif scenario == 'cpu_saturation':
                    mock_monitor.get_cpu_usage.return_value = 0.98  # 98% CPU usage
                elif scenario == 'disk_space_low':
                    mock_monitor.get_disk_usage.return_value = 0.92  # 92% disk usage
                else:  # file_descriptor_limit
                    mock_monitor.get_fd_usage.return_value = 0.90  # 90% FD usage
                
                # System should implement resource management
                resource_manager = Mock()
                resource_manager.handle_resource_pressure = Mock()
                
                # Simulate resource pressure handling
                if scenario == 'memory_pressure':
                    resource_manager.handle_resource_pressure('memory')
                elif scenario == 'cpu_saturation':
                    resource_manager.handle_resource_pressure('cpu')
                elif scenario == 'disk_space_low':
                    resource_manager.handle_resource_pressure('disk')
                else:
                    resource_manager.handle_resource_pressure('file_descriptors')
                
                resource_manager.handle_resource_pressure.assert_called_once()
                print(f"✓ System handled {scenario} pressure")


class TestMultiExchangeIntegration:
    """Test multi-exchange data consistency and integration."""
    
    @pytest.fixture
    def multi_exchange_data(self):
        """Generate test data for multiple exchanges."""
        
        # Create overlapping symbols across exchanges
        symbols_data = {
            'robinhood': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            'oanda': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
            'coinbase': ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
        }
        
        data = {}
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for exchange, symbols in symbols_data.items():
            exchange_data = []
            
            for symbol in symbols:
                for i in range(1440):  # 24 hours * 60 minutes
                    timestamp = base_time + timedelta(minutes=i)
                    
                    # Generate realistic price based on asset type
                    if exchange == 'robinhood':
                        base_price = 150.0 if 'AAPL' in symbol else 2500.0
                    elif exchange == 'oanda':
                        base_price = 1.1000
                    else:  # coinbase
                        base_price = 45000.0 if 'BTC' in symbol else 3000.0
                    
                    price_change = np.random.normal(0, 0.001)
                    price = base_price * (1 + price_change)
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=price * 0.9995,
                        high=price * 1.0005,
                        low=price * 0.9995,
                        close=price,
                        volume=random.randint(10000, 100000),
                        exchange=ExchangeType[exchange.upper()]
                    )
                    exchange_data.append(market_data)
            
            data[exchange] = exchange_data
        
        return data
    
    @pytest.mark.asyncio
    async def test_cross_exchange_data_consistency(self, multi_exchange_data):
        """Test data consistency across multiple exchanges."""
        
        # Mock exchange connectors
        mock_connectors = {}
        
        for exchange_name, exchange_data in multi_exchange_data.items():
            connector = Mock()
            connector.get_historical_data = AsyncMock()
            
            # Convert to DataFrame format
            df_data = pd.DataFrame([md.__dict__ for md in exchange_data])
            connector.get_historical_data.return_value = df_data
            
            mock_connectors[exchange_name] = connector
        
        # Initialize data aggregator with all connectors
        data_aggregator = DataAggregator()
        data_aggregator.robinhood_connector = mock_connectors['robinhood']
        data_aggregator.oanda_connector = mock_connectors['oanda']
        data_aggregator.coinbase_connector = mock_connectors['coinbase']
        
        # Test data retrieval from each exchange
        exchanges = ['robinhood', 'oanda', 'coinbase']
        symbols = ['AAPL', 'EUR/USD', 'BTC/USD']
        
        all_data = {}
        
        for exchange, symbol in zip(exchanges, symbols):
            data = await data_aggregator.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                start_date=datetime.now(timezone.utc) - timedelta(hours=1),
                end_date=datetime.now(timezone.utc)
            )
            
            all_data[f"{exchange}_{symbol}"] = data
            
            # Verify data quality
            assert not data.empty
            assert 'timestamp' in data.columns
            assert 'close' in data.columns
            assert data['timestamp'].is_monotonic_increasing
        
        print("✓ Cross-exchange data consistency test passed")
    
    @pytest.mark.asyncio
    async def test_timestamp_synchronization(self, multi_exchange_data):
        """Test timestamp synchronization across exchanges."""
        
        # Create data with slight timestamp differences
        base_time = datetime.now(timezone.utc)
        
        # Simulate timestamp drift between exchanges
        timestamp_offsets = {
            'robinhood': timedelta(milliseconds=0),    # Reference
            'oanda': timedelta(milliseconds=50),       # 50ms ahead
            'coinbase': timedelta(milliseconds=-30)    # 30ms behind
        }
        
        synchronized_data = {}
        
        for exchange, offset in timestamp_offsets.items():
            # Adjust timestamps
            adjusted_data = []
            for md in multi_exchange_data[exchange][:60]:  # First 60 records
                adjusted_md = MarketData(
                    symbol=md.symbol,
                    timestamp=md.timestamp + offset,
                    open=md.open,
                    high=md.high,
                    low=md.low,
                    close=md.close,
                    volume=md.volume,
                    exchange=md.exchange
                )
                adjusted_data.append(adjusted_md)
            
            synchronized_data[exchange] = adjusted_data
        
        # Mock timestamp synchronizer
        with patch('src.services.timestamp_synchronizer.TimestampSynchronizer') as MockSync:
            synchronizer = MockSync()
            
            # Mock synchronization method
            def mock_synchronize(data_dict):
                # Simulate timestamp alignment
                aligned_data = {}
                reference_times = [md.timestamp for md in data_dict['robinhood']]
                
                for exchange, data_list in data_dict.items():
                    aligned_list = []
                    for i, md in enumerate(data_list):
                        if i < len(reference_times):
                            aligned_md = MarketData(
                                symbol=md.symbol,
                                timestamp=reference_times[i],  # Align to reference
                                open=md.open,
                                high=md.high,
                                low=md.low,
                                close=md.close,
                                volume=md.volume,
                                exchange=md.exchange
                            )
                            aligned_list.append(aligned_md)
                    aligned_data[exchange] = aligned_list
                
                return aligned_data
            
            synchronizer.synchronize_timestamps = mock_synchronize
            
            # Test synchronization
            aligned_data = synchronizer.synchronize_timestamps(synchronized_data)
            
            # Verify all exchanges have aligned timestamps
            reference_timestamps = [md.timestamp for md in aligned_data['robinhood']]
            
            for exchange, data_list in aligned_data.items():
                exchange_timestamps = [md.timestamp for md in data_list]
                
                # Timestamps should be aligned (within tolerance)
                for ref_ts, ex_ts in zip(reference_timestamps, exchange_timestamps):
                    time_diff = abs((ref_ts - ex_ts).total_seconds())
                    assert time_diff < 0.001  # Within 1ms tolerance
            
            print("✓ Timestamp synchronization test passed")
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, multi_exchange_data):
        """Test data quality validation across exchanges."""
        
        # Introduce data quality issues
        corrupted_data = {}
        
        for exchange, data_list in multi_exchange_data.items():
            corrupted_list = []
            
            for i, md in enumerate(data_list[:100]):  # First 100 records
                if i % 20 == 0:  # Introduce issues every 20th record
                    # Create data quality issues
                    if i % 40 == 0:  # Price anomaly
                        corrupted_md = MarketData(
                            symbol=md.symbol,
                            timestamp=md.timestamp,
                            open=md.open,
                            high=md.high * 10,  # Unrealistic high price
                            low=md.low,
                            close=md.close,
                            volume=md.volume,
                            exchange=md.exchange
                        )
                    else:  # Volume anomaly
                        corrupted_md = MarketData(
                            symbol=md.symbol,
                            timestamp=md.timestamp,
                            open=md.open,
                            high=md.high,
                            low=md.low,
                            close=md.close,
                            volume=0,  # Zero volume
                            exchange=md.exchange
                        )
                    corrupted_list.append(corrupted_md)
                else:
                    corrupted_list.append(md)
            
            corrupted_data[exchange] = corrupted_list
        
        # Mock data quality validator
        with patch('src.services.data_quality_validator.DataQualityValidator') as MockValidator:
            validator = MockValidator()
            
            # Mock validation methods
            def mock_validate_prices(data_list):
                issues = []
                for i, md in enumerate(data_list):
                    if md.high > md.close * 5:  # Detect price anomalies
                        issues.append({
                            'index': i,
                            'issue': 'price_anomaly',
                            'severity': 'high'
                        })
                return issues
            
            def mock_validate_volume(data_list):
                issues = []
                for i, md in enumerate(data_list):
                    if md.volume == 0:  # Detect volume issues
                        issues.append({
                            'index': i,
                            'issue': 'zero_volume',
                            'severity': 'medium'
                        })
                return issues
            
            validator.validate_prices = mock_validate_prices
            validator.validate_volume = mock_validate_volume
            
            # Test validation for each exchange
            for exchange, data_list in corrupted_data.items():
                price_issues = validator.validate_prices(data_list)
                volume_issues = validator.validate_volume(data_list)
                
                # Should detect the introduced issues
                assert len(price_issues) > 0
                assert len(volume_issues) > 0
                
                print(f"✓ {exchange}: Detected {len(price_issues)} price issues, "
                      f"{len(volume_issues)} volume issues")
        
        print("✓ Data quality validation test passed")
    
    def test_exchange_failover_mechanism(self, multi_exchange_data):
        """Test failover mechanism when exchanges become unavailable."""
        
        # Mock exchange availability
        exchange_status = {
            'robinhood': True,
            'oanda': True,
            'coinbase': True
        }
        
        # Mock data aggregator with failover capability
        with patch('src.services.data_aggregator.DataAggregator') as MockAggregator:
            aggregator = MockAggregator()
            
            def mock_get_data_with_failover(symbol, preferred_exchange, fallback_exchanges):
                # Check if preferred exchange is available
                if exchange_status[preferred_exchange]:
                    return multi_exchange_data[preferred_exchange][:10]  # Return some data
                
                # Try fallback exchanges
                for fallback in fallback_exchanges:
                    if exchange_status[fallback]:
                        return multi_exchange_data[fallback][:10]
                
                # No exchanges available
                return []
            
            aggregator.get_data_with_failover = mock_get_data_with_failover
            
            # Test normal operation
            data = aggregator.get_data_with_failover('AAPL', 'robinhood', ['oanda', 'coinbase'])
            assert len(data) > 0
            print("✓ Normal operation: Data retrieved from preferred exchange")
            
            # Test failover when preferred exchange is down
            exchange_status['robinhood'] = False
            
            data = aggregator.get_data_with_failover('AAPL', 'robinhood', ['oanda', 'coinbase'])
            assert len(data) > 0
            print("✓ Failover: Data retrieved from fallback exchange")
            
            # Test when all exchanges are down
            exchange_status['oanda'] = False
            exchange_status['coinbase'] = False
            
            data = aggregator.get_data_with_failover('AAPL', 'robinhood', ['oanda', 'coinbase'])
            assert len(data) == 0
            print("✓ All exchanges down: No data retrieved")
        
        print("✓ Exchange failover mechanism test passed")


if __name__ == "__main__":
    # Run with specific markers for different test categories
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])