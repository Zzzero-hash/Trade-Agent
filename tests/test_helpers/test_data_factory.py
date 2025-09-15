"""
Factory classes for generating test data.
Reduces code duplication and improves maintainability.
"""

from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
import random
import numpy as np

from src.models.market_data import MarketData, ExchangeType


class MarketDataFactory:
    """Factory for creating realistic market data for testing."""
    
    SYMBOL_CONFIG = {
        'AAPL': {'base_price': 150.0, 'exchange': ExchangeType.ROBINHOOD},
        'GOOGL': {'base_price': 2500.0, 'exchange': ExchangeType.ROBINHOOD},
        'EUR/USD': {'base_price': 1.1000, 'exchange': ExchangeType.OANDA},
        'BTC/USD': {'base_price': 45000.0, 'exchange': ExchangeType.COINBASE},
        'ETH/USD': {'base_price': 3000.0, 'exchange': ExchangeType.COINBASE},
    }
    
    @classmethod
    def create_market_data_series(cls, symbol: str, hours: int = 720, 
                                 base_time: datetime = None) -> List[MarketData]:
        """Create a series of market data for a given symbol."""
        if base_time is None:
            base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        config = cls.SYMBOL_CONFIG.get(symbol)
        if not config:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        data = []
        base_price = config['base_price']
        exchange = config['exchange']
        
        for i in range(hours):
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
    
    @classmethod
    def create_multi_symbol_data(cls, symbols: List[str] = None, 
                                hours: int = 720) -> List[MarketData]:
        """Create market data for multiple symbols."""
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'EUR/USD', 'BTC/USD']
        
        all_data = []
        for symbol in symbols:
            all_data.extend(cls.create_market_data_series(symbol, hours))
        
        return all_data
    
    @classmethod
    def create_performance_test_data(cls, count: int = 1000) -> List[MarketData]:
        """Create market data optimized for performance testing."""
        return [
            MarketData(
                symbol=f'STOCK{i}',
                timestamp=datetime.now(timezone.utc),
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000000,
                exchange=ExchangeType.ROBINHOOD
            ) for i in range(count)
        ]


class MockExchangeFactory:
    """Factory for creating mock exchange connectors."""
    
    @staticmethod
    def create_mock_exchanges() -> Dict[str, Any]:
        """Create mock exchange connectors with common setup."""
        from unittest.mock import Mock, AsyncMock
        from src.exchanges.robinhood_connector import RobinhoodConnector
        from src.exchanges.oanda_connector import OANDAConnector
        from src.exchanges.coinbase_connector import CoinbaseConnector
        
        robinhood = Mock(spec=RobinhoodConnector)
        oanda = Mock(spec=OANDAConnector)
        coinbase = Mock(spec=CoinbaseConnector)
        
        # Mock async methods for all exchanges
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


class TestScenarioFactory:
    """Factory for creating test scenarios."""
    
    @staticmethod
    def create_failure_scenarios() -> List[Dict[str, str]]:
        """Create failure scenarios for chaos engineering tests."""
        return [
            {'exchange': 'robinhood', 'failure_type': 'timeout'},
            {'exchange': 'oanda', 'failure_type': 'rate_limit'},
            {'exchange': 'coinbase', 'failure_type': 'connection_error'}
        ]
    
    @staticmethod
    def create_model_failure_scenarios() -> List[str]:
        """Create model failure scenarios."""
        return [
            'model_not_loaded',
            'cuda_out_of_memory',
            'inference_timeout',
            'model_corruption'
        ]
    
    @staticmethod
    def create_database_failure_scenarios() -> List[str]:
        """Create database failure scenarios."""
        return [
            'connection_timeout',
            'query_timeout',
            'disk_full',
            'connection_pool_exhausted'
        ]