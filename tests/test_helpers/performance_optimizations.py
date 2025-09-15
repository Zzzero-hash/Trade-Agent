"""
Performance optimizations for test data generation and execution.
"""

import functools
from typing import List, Dict, Any, Generator
from datetime import datetime, timezone, timedelta
import numpy as np

from src.models.market_data import MarketData, ExchangeType


class LazyMarketDataGenerator:
    """Lazy generator for market data to reduce memory usage."""
    
    def __init__(self, symbol: str, base_price: float, exchange: ExchangeType, 
                 hours: int = 720):
        self.symbol = symbol
        self.base_price = base_price
        self.exchange = exchange
        self.hours = hours
        self.base_time = datetime.now(timezone.utc) - timedelta(days=30)
    
    def __iter__(self) -> Generator[MarketData, None, None]:
        """Generate market data on demand."""
        for i in range(self.hours):
            timestamp = self.base_time + timedelta(hours=i)
            
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)
            price = self.base_price * (1 + price_change * 0.1)
            
            yield MarketData(
                symbol=self.symbol,
                timestamp=timestamp,
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=np.random.randint(100000, 1000000),
                exchange=self.exchange
            )
    
    def take(self, n: int) -> List[MarketData]:
        """Take first n items from the generator."""
        return list(self.__iter__())[:n]


@functools.lru_cache(maxsize=128)
def cached_market_data_generation(symbol: str, hours: int, seed: int = 42) -> List[MarketData]:
    """Cached market data generation to avoid regenerating same data."""
    np.random.seed(seed)  # For reproducible tests
    
    symbol_config = {
        'AAPL': {'base_price': 150.0, 'exchange': ExchangeType.ROBINHOOD},
        'GOOGL': {'base_price': 2500.0, 'exchange': ExchangeType.ROBINHOOD},
        'EUR/USD': {'base_price': 1.1000, 'exchange': ExchangeType.OANDA},
        'BTC/USD': {'base_price': 45000.0, 'exchange': ExchangeType.COINBASE},
    }
    
    config = symbol_config.get(symbol, {'base_price': 100.0, 'exchange': ExchangeType.ROBINHOOD})
    generator = LazyMarketDataGenerator(
        symbol, config['base_price'], config['exchange'], hours
    )
    
    return generator.take(hours)


class PerformanceTestHelper:
    """Helper class for performance testing optimizations."""
    
    @staticmethod
    def create_batch_processor(batch_size: int = 100):
        """Create a batch processor for efficient data processing."""
        def process_batch(data_list: List[Any], processor_func) -> List[Any]:
            results = []
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_results = processor_func(batch)
                results.extend(batch_results)
            return results
        return process_batch
    
    @staticmethod
    def memory_efficient_data_frame_creation(data_list: List[MarketData]) -> Dict[str, List]:
        """Create DataFrame data more efficiently."""
        # Pre-allocate lists for better performance
        timestamps = []
        symbols = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        for md in data_list:
            timestamps.append(md.timestamp)
            symbols.append(md.symbol)
            opens.append(md.open)
            highs.append(md.high)
            lows.append(md.low)
            closes.append(md.close)
            volumes.append(md.volume)
        
        return {
            'timestamp': timestamps,
            'symbol': symbols,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }


class TestExecutionTimer:
    """Context manager for timing test execution."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"⏱️  {self.test_name} completed in {duration:.3f}s")
    
    @property
    def duration(self) -> float:
        """Get the duration of the test execution."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0