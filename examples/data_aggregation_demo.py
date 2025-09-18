"""
Demo script showing the unified data aggregation system in action.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.data_aggregator import DataAggregator
from src.models.market_data import MarketData, ExchangeType
from src.exchanges.robinhood import RobinhoodConnector
from src.exchanges.oanda import OANDAConnector
from src.exchanges.coinbase import CoinbaseConnector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDataGenerator:
    """Generate mock market data for demonstration."""
    
    def __init__(self, exchange_type: ExchangeType, base_price: float = 100.0):
        self.exchange_type = exchange_type
        self.base_price = base_price
        self.current_price = base_price
        self.data_count = 0
    
    async def generate_market_data(self, symbol: str) -> MarketData:
        """Generate realistic market data."""
        import random
        
        # Simulate price movement
        price_change = random.uniform(-0.5, 0.5)
        self.current_price += price_change
        
        # Ensure positive prices
        self.current_price = max(self.current_price, 1.0)
        
        # Generate OHLC data with proper relationships
        open_price = self.current_price
        
        # Generate close price first
        close_change = random.uniform(-0.5, 0.5)
        close_price = open_price + close_change
        
        # Ensure positive prices
        close_price = max(close_price, 1.0)
        
        # Generate high and low based on open and close
        min_price = min(open_price, close_price)
        max_price = max(open_price, close_price)
        
        # High should be at least the max of open/close
        high_price = max_price + random.uniform(0, 1.0)
        
        # Low should be at most the min of open/close  
        low_price = min_price - random.uniform(0, 1.0)
        low_price = max(low_price, 0.1)  # Ensure positive
        
        volume = random.uniform(1000, 10000)
        
        # Add some exchange-specific variations while maintaining relationships
        if self.exchange_type == ExchangeType.COINBASE:
            # Crypto tends to be more volatile
            extra_high = random.uniform(0, 1.0)
            extra_low = random.uniform(0, 0.5)
            high_price += extra_high
            low_price = max(low_price - extra_low, 0.1)
            volume *= 2
        elif self.exchange_type == ExchangeType.OANDA:
            # Forex has smaller spreads - keep prices closer
            spread = random.uniform(0.01, 0.05)
            high_price = max_price + spread/2
            low_price = max(min_price - spread/2, 0.1)
        
        self.data_count += 1
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            open=round(open_price, 4),
            high=round(high_price, 4),
            low=round(low_price, 4),
            close=round(close_price, 4),
            volume=round(volume, 2),
            exchange=self.exchange_type
        )


class MockExchange:
    """Mock exchange for demonstration."""
    
    def __init__(self, name: str, exchange_type: ExchangeType):
        self.name = name
        self.exchange_type = exchange_type
        self.is_connected = True
        self.data_generator = MockDataGenerator(exchange_type)
        self.logger = logging.getLogger(f"MockExchange.{name}")
    
    async def get_real_time_data(self, symbols: List[str]):
        """Generate mock real-time data stream."""
        self.logger.info(f"Starting real-time data stream for {symbols}")
        
        while self.is_connected:
            for symbol in symbols:
                try:
                    data = await self.data_generator.generate_market_data(symbol)
                    yield data
                    await asyncio.sleep(0.5)  # Generate data every 500ms
                except Exception as e:
                    self.logger.error(f"Error generating data for {symbol}: {e}")
    
    async def get_historical_data(self, symbol, timeframe, start, end):
        """Generate mock historical data."""
        import pandas as pd
        
        # Generate hourly data for the time range
        dates = pd.date_range(start=start, end=end, freq='1h')
        data = []
        
        base_price = self.data_generator.base_price
        for i, date in enumerate(dates):
            price = base_price + i * 0.1 + (i % 10) * 0.05  # Trending with noise
            
            data.append({
                'timestamp': date,
                'open': round(price, 4),
                'high': round(price + 0.5, 4),
                'low': round(price - 0.5, 4),
                'close': round(price + 0.2, 4),
                'volume': 1000 + i * 10,
                'symbol': symbol
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df


async def demonstrate_real_time_aggregation():
    """Demonstrate real-time data aggregation."""
    logger.info("=== Real-Time Data Aggregation Demo ===")
    
    # Create mock exchanges
    exchanges = [
        MockExchange("MockCoinbase", ExchangeType.COINBASE),
        MockExchange("MockRobinhood", ExchangeType.ROBINHOOD),
        MockExchange("MockOANDA", ExchangeType.OANDA)
    ]
    
    # Create data aggregator
    aggregator = DataAggregator(exchanges)
    
    # Symbols to track
    symbols = ["BTCUSD", "ETHUSD"]
    
    logger.info(f"Starting aggregation for symbols: {symbols}")
    logger.info("Collecting data for 30 seconds...")
    
    # Collect aggregated data for a short period
    start_time = datetime.now()
    aggregated_count = 0
    
    try:
        async for aggregated_data in aggregator.start_aggregation(symbols):
            aggregated_count += 1
            
            logger.info(f"Aggregated Data #{aggregated_count}:")
            logger.info(f"  Symbol: {aggregated_data.symbol}")
            logger.info(f"  Timestamp: {aggregated_data.timestamp}")
            logger.info(f"  Price: {aggregated_data.close:.4f}")
            logger.info(f"  Volume: {aggregated_data.volume:.2f}")
            logger.info(f"  Sources: {aggregated_data.source_count} ({', '.join(aggregated_data.exchanges)})")
            logger.info(f"  Confidence: {aggregated_data.confidence_score:.2f}")
            
            if aggregated_data.quality_issues:
                logger.warning(f"  Quality Issues: {len(aggregated_data.quality_issues)}")
                for issue in aggregated_data.quality_issues[:3]:  # Show first 3 issues
                    logger.warning(f"    - {issue.severity.upper()}: {issue.description}")
            
            logger.info("")
            
            # Stop after 30 seconds or 10 aggregated data points
            if (datetime.now() - start_time).seconds > 30 or aggregated_count >= 10:
                break
                
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        # Stop exchanges
        for exchange in exchanges:
            exchange.is_connected = False
    
    logger.info(f"Demo completed. Processed {aggregated_count} aggregated data points.")


async def demonstrate_historical_aggregation():
    """Demonstrate historical data aggregation."""
    logger.info("=== Historical Data Aggregation Demo ===")
    
    # Create mock exchanges
    exchanges = [
        MockExchange("MockCoinbase", ExchangeType.COINBASE),
        MockExchange("MockRobinhood", ExchangeType.ROBINHOOD),
        MockExchange("MockOANDA", ExchangeType.OANDA)
    ]
    
    # Create data aggregator
    aggregator = DataAggregator(exchanges)
    
    # Get historical data
    symbol = "BTCUSD"
    start_time = datetime.now(timezone.utc) - timedelta(days=1)
    end_time = datetime.now(timezone.utc)
    
    logger.info(f"Fetching historical data for {symbol}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    df = await aggregator.get_historical_aggregated_data(
        symbol, "1h", start_time, end_time
    )
    
    if not df.empty:
        logger.info(f"Retrieved {len(df)} historical data points")
        logger.info("\nSample data (first 5 rows):")
        logger.info(df.head().to_string())
        
        logger.info(f"\nData summary:")
        logger.info(f"  Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
        logger.info(f"  Average volume: {df['volume'].mean():.2f}")
        logger.info(f"  Average sources per point: {df['source_count'].mean():.1f}")
    else:
        logger.warning("No historical data retrieved")


async def demonstrate_data_quality_monitoring():
    """Demonstrate data quality monitoring."""
    logger.info("=== Data Quality Monitoring Demo ===")
    
    # Create mock exchanges with some problematic data
    exchanges = [
        MockExchange("MockCoinbase", ExchangeType.COINBASE),
        MockExchange("MockRobinhood", ExchangeType.ROBINHOOD)
    ]
    
    # Create data aggregator
    aggregator = DataAggregator(exchanges)
    
    # Simulate some data collection with quality issues
    logger.info("Simulating data collection with quality issues...")
    
    # Add some test data with issues to the buffer
    from src.services.data_aggregator import DataQualityReport, DataQualityIssue
    
    test_data = MarketData(
        symbol="BTCUSD",
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        exchange=ExchangeType.COINBASE
    )
    
    # Create some quality issues
    quality_issues = [
        DataQualityReport(
            symbol="BTCUSD",
            exchange="COINBASE",
            timestamp=datetime.now(timezone.utc),
            issue_type=DataQualityIssue.PRICE_ANOMALY,
            severity="medium",
            description="Price spike detected"
        ),
        DataQualityReport(
            symbol="BTCUSD",
            exchange="ROBINHOOD",
            timestamp=datetime.now(timezone.utc),
            issue_type=DataQualityIssue.VOLUME_ANOMALY,
            severity="low",
            description="Unusually high volume"
        )
    ]
    
    # Add to buffer
    for i in range(10):
        aggregator.raw_data_buffer.append({
            'data': test_data,
            'quality_issues': quality_issues if i % 3 == 0 else [],  # Issues in 1/3 of data
            'received_at': datetime.now(timezone.utc) - timedelta(minutes=i)
        })
    
    # Get quality summary
    summary = aggregator.get_data_quality_summary(hours=1)
    
    logger.info("Data Quality Summary:")
    logger.info(f"  Total data points: {summary['total_data_points']}")
    logger.info(f"  Total quality issues: {summary['total_quality_issues']}")
    logger.info(f"  Issue rate: {summary['issue_rate']:.2%}")
    
    if summary['issue_types']:
        logger.info("  Issue types:")
        for issue_type, count in summary['issue_types'].items():
            logger.info(f"    - {issue_type}: {count}")
    
    if summary['severity_distribution']:
        logger.info("  Severity distribution:")
        for severity, count in summary['severity_distribution'].items():
            logger.info(f"    - {severity}: {count}")
    
    if summary['exchange_statistics']:
        logger.info("  Exchange statistics:")
        for exchange, stats in summary['exchange_statistics'].items():
            issue_rate = stats['issues'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(f"    - {exchange}: {stats['count']} points, {issue_rate:.2%} issue rate")


async def main():
    """Run all demonstrations."""
    logger.info("Starting Data Aggregation System Demo")
    logger.info("=" * 50)
    
    try:
        # Run demonstrations
        await demonstrate_historical_aggregation()
        await asyncio.sleep(2)
        
        await demonstrate_data_quality_monitoring()
        await asyncio.sleep(2)
        
        await demonstrate_real_time_aggregation()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    logger.info("=" * 50)
    logger.info("Data Aggregation System Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())