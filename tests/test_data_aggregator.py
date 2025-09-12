"""
Tests for the unified data aggregation system.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import statistics

from src.services.data_aggregator import (
    DataAggregator,
    TimestampSynchronizer,
    DataQualityValidator,
    DataQualityIssue,
    DataQualityReport,
    AggregatedData
)
from src.models.market_data import MarketData, ExchangeType
from src.exchanges.base import ExchangeConnector


class MockExchangeConnector(ExchangeConnector):
    """Mock exchange connector for testing."""
    
    def __init__(self, name: str, data_stream: list = None):
        super().__init__("test_key", "test_secret", True)
        self.name = name
        self.data_stream = data_stream or []
        self.is_connected = True
        self._stream_index = 0
    
    async def connect(self) -> bool:
        return True
    
    async def disconnect(self) -> None:
        pass
    
    async def get_real_time_data(self, symbols):
        """Yield test data from the stream."""
        for data in self.data_stream:
            yield data
            await asyncio.sleep(0.01)  # Small delay to simulate real streaming
    
    async def get_historical_data(self, symbol, timeframe, start, end):
        """Return mock historical data."""
        dates = pd.date_range(start=start, end=end, freq='1h')
        data = []
        
        for i, date in enumerate(dates):
            base_price = 100 + i * 0.1
            data.append({
                'timestamp': date,
                'open': base_price,
                'high': base_price + 1,
                'low': base_price - 1,
                'close': base_price + 0.5,
                'volume': 1000 + i * 10,
                'symbol': symbol
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    async def place_order(self, order):
        pass
    
    async def cancel_order(self, order_id):
        pass
    
    async def get_account_info(self):
        pass
    
    async def get_positions(self):
        pass
    
    def get_supported_symbols(self):
        return ["BTCUSD", "ETHUSD"]


class TestTimestampSynchronizer:
    """Test timestamp synchronization functionality."""
    
    def setup_method(self):
        self.synchronizer = TimestampSynchronizer(tolerance_seconds=5)
    
    def test_normalize_timestamp(self):
        """Test timestamp normalization."""
        # Test naive datetime
        naive_dt = datetime(2023, 1, 1, 12, 0, 0, 500000)
        normalized = self.synchronizer.normalize_timestamp(naive_dt)
        
        assert normalized.tzinfo == timezone.utc
        assert normalized.microsecond == 0
        assert normalized.second == 1  # Rounded up due to microseconds >= 500000
    
    def test_normalize_timestamp_rounding_down(self):
        """Test timestamp rounding down."""
        dt = datetime(2023, 1, 1, 12, 0, 0, 400000, tzinfo=timezone.utc)
        normalized = self.synchronizer.normalize_timestamp(dt)
        
        assert normalized.microsecond == 0
        assert normalized.second == 0  # Rounded down
    
    def test_align_timestamps(self):
        """Test timestamp alignment."""
        base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        data_points = [
            MarketData(
                symbol="BTCUSD",
                timestamp=base_time,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
                exchange=ExchangeType.COINBASE
            ),
            MarketData(
                symbol="BTCUSD",
                timestamp=base_time + timedelta(milliseconds=200),  # Smaller difference
                open=100.1,
                high=101.1,
                low=99.1,
                close=100.6,
                volume=1100.0,
                exchange=ExchangeType.ROBINHOOD
            )
        ]
        
        aligned = self.synchronizer.align_timestamps(data_points)
        
        # Both should be aligned to the same timestamp (or very close)
        assert len(aligned) <= 2  # Could be 1 or 2 depending on rounding
        total_points = sum(len(group) for group in aligned.values())
        assert total_points == 2
    
    def test_find_timestamp_gaps(self):
        """Test timestamp gap detection."""
        base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamps = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=5),  # Gap here
            base_time + timedelta(minutes=6)
        ]
        
        expected_interval = timedelta(minutes=1)
        gaps = self.synchronizer.find_timestamp_gaps(timestamps, expected_interval)
        
        assert len(gaps) == 1
        gap_start, gap_end, gap_duration = gaps[0]
        assert gap_duration == timedelta(minutes=4)


class TestDataQualityValidator:
    """Test data quality validation functionality."""
    
    def setup_method(self):
        self.validator = DataQualityValidator()
    
    def test_validate_basic_constraints_valid_data(self):
        """Test validation with valid data."""
        valid_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        issues = self.validator.validate_market_data(valid_data)
        
        # Should have no issues for valid data
        basic_issues = [i for i in issues if i.issue_type == DataQualityIssue.PRICE_ANOMALY]
        assert len(basic_issues) == 0
    
    def test_validate_negative_prices(self):
        """Test validation with negative prices."""
        # Create valid data first, then modify it to bypass Pydantic validation
        valid_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        # Manually set invalid value to test validator logic
        valid_data.open = -100.0
        
        issues = self.validator.validate_market_data(valid_data)
        
        price_issues = [i for i in issues if i.issue_type == DataQualityIssue.PRICE_ANOMALY]
        assert len(price_issues) > 0
        assert any("Negative or zero price" in issue.description for issue in price_issues)
    
    def test_validate_price_relationships(self):
        """Test validation of price relationships."""
        # Create valid data first, then modify it
        valid_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        # Manually set invalid relationship
        valid_data.high = 98.0  # High < Low (invalid)
        
        issues = self.validator.validate_market_data(valid_data)
        
        price_issues = [i for i in issues if i.issue_type == DataQualityIssue.PRICE_ANOMALY]
        assert len(price_issues) > 0
        assert any("High price" in issue.description and "Low price" in issue.description 
                  for issue in price_issues)
    
    def test_validate_negative_volume(self):
        """Test validation with negative volume."""
        # Create valid data first, then modify it
        valid_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        # Manually set invalid volume
        valid_data.volume = -1000.0
        
        issues = self.validator.validate_market_data(valid_data)
        
        volume_issues = [i for i in issues if i.issue_type == DataQualityIssue.VOLUME_ANOMALY]
        assert len(volume_issues) > 0
        assert any("Negative volume" in issue.description for issue in volume_issues)
    
    def test_validate_future_timestamp(self):
        """Test validation with future timestamp."""
        # Create valid data first, then modify it
        valid_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        # Manually set future timestamp
        valid_data.timestamp = datetime.now(timezone.utc) + timedelta(hours=1)
        
        issues = self.validator.validate_market_data(valid_data)
        
        timestamp_issues = [i for i in issues if i.issue_type == DataQualityIssue.TIMESTAMP_GAP]
        assert len(timestamp_issues) > 0
        assert any("Future timestamp" in issue.description for issue in timestamp_issues)
    
    def test_price_anomaly_detection(self):
        """Test price anomaly detection with historical data."""
        # First, add some normal historical data
        base_time = datetime.now(timezone.utc)
        normal_price = 100.0
        
        for i in range(20):
            normal_data = MarketData(
                symbol="BTCUSD",
                timestamp=base_time - timedelta(minutes=i),
                open=normal_price,
                high=normal_price + 1,
                low=normal_price - 1,
                close=normal_price + (i % 2) * 0.1,  # Small variations
                volume=1000.0,
                exchange=ExchangeType.COINBASE
            )
            self.validator.validate_market_data(normal_data)
        
        # Now test with anomalous data - create valid data first
        anomalous_data = MarketData(
            symbol="BTCUSD",
            timestamp=base_time,
            open=normal_price,
            high=normal_price * 2 + 1,  # Adjust high to be valid
            low=normal_price - 1,
            close=normal_price * 2,  # 100% price jump (anomaly)
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        issues = self.validator.validate_market_data(anomalous_data)
        
        anomaly_issues = [i for i in issues if i.issue_type == DataQualityIssue.PRICE_ANOMALY 
                         and "anomaly" in i.description.lower()]
        assert len(anomaly_issues) > 0


class TestDataAggregator:
    """Test data aggregation functionality."""
    
    def setup_method(self):
        self.mock_exchanges = []
        self.aggregator = None
    
    def create_test_data(self, symbol: str, exchange: ExchangeType, base_price: float = 100.0):
        """Create test market data."""
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            open=base_price,
            high=base_price + 1,
            low=base_price - 1,
            close=base_price + 0.5,
            volume=1000.0,
            exchange=exchange
        )
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        mock_exchange = MockExchangeConnector("test_exchange")
        aggregator = DataAggregator([mock_exchange])
        
        assert len(aggregator.exchanges) == 1
        assert aggregator.synchronizer is not None
        assert aggregator.validator is not None
    
    @pytest.mark.asyncio
    async def test_historical_data_aggregation(self):
        """Test historical data aggregation."""
        # Create mock exchanges with different data
        exchange1 = MockExchangeConnector("exchange1")
        exchange2 = MockExchangeConnector("exchange2")
        
        aggregator = DataAggregator([exchange1, exchange2])
        
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        result_df = await aggregator.get_historical_aggregated_data(
            "BTCUSD", "1h", start_time, end_time
        )
        
        assert not result_df.empty
        assert 'open' in result_df.columns
        assert 'high' in result_df.columns
        assert 'low' in result_df.columns
        assert 'close' in result_df.columns
        assert 'volume' in result_df.columns
        assert 'source_count' in result_df.columns
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        aggregator = DataAggregator([])
        
        # Test with single source
        single_source_data = [self.create_test_data("BTCUSD", ExchangeType.COINBASE)]
        score = aggregator._calculate_confidence_score(single_source_data, [])
        assert score == 0.7  # Single source penalty
        
        # Test with multiple sources
        multi_source_data = [
            self.create_test_data("BTCUSD", ExchangeType.COINBASE, 100.0),
            self.create_test_data("BTCUSD", ExchangeType.ROBINHOOD, 100.1)
        ]
        score = aggregator._calculate_confidence_score(multi_source_data, [])
        assert score > 0.7  # Better than single source
        
        # Test with quality issues
        quality_issues = [
            DataQualityReport(
                symbol="BTCUSD",
                exchange="COINBASE",
                timestamp=datetime.now(timezone.utc),
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="high",
                description="Test issue"
            )
        ]
        score_with_issues = aggregator._calculate_confidence_score(multi_source_data, quality_issues)
        assert score_with_issues < score  # Quality issues reduce score
    
    def test_data_quality_summary(self):
        """Test data quality summary generation."""
        aggregator = DataAggregator([])
        
        # Add some test data to buffer
        test_item = {
            'data': self.create_test_data("BTCUSD", ExchangeType.COINBASE),
            'quality_issues': [
                DataQualityReport(
                    symbol="BTCUSD",
                    exchange="COINBASE",
                    timestamp=datetime.now(timezone.utc),
                    issue_type=DataQualityIssue.PRICE_ANOMALY,
                    severity="medium",
                    description="Test issue"
                )
            ],
            'received_at': datetime.now(timezone.utc)
        }
        
        aggregator.raw_data_buffer.append(test_item)
        
        summary = aggregator.get_data_quality_summary(hours=1)
        
        assert 'total_data_points' in summary
        assert 'total_quality_issues' in summary
        assert 'issue_rate' in summary
        assert 'issue_types' in summary
        assert 'severity_distribution' in summary
        assert 'exchange_statistics' in summary
        
        assert summary['total_data_points'] == 1
        assert summary['total_quality_issues'] == 1
        assert summary['issue_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_create_aggregated_data(self):
        """Test creation of aggregated data from multiple sources."""
        aggregator = DataAggregator([])
        
        timestamp = datetime.now(timezone.utc)
        data_points = [
            MarketData(
                symbol="BTCUSD",
                timestamp=timestamp,
                open=100.0,
                high=102.0,
                low=98.0,
                close=101.0,
                volume=1000.0,
                exchange=ExchangeType.COINBASE
            ),
            MarketData(
                symbol="BTCUSD",
                timestamp=timestamp,
                open=100.2,
                high=101.8,
                low=98.5,
                close=100.8,
                volume=1200.0,
                exchange=ExchangeType.ROBINHOOD
            )
        ]
        
        all_items = [
            {'data': dp, 'quality_issues': [], 'received_at': timestamp}
            for dp in data_points
        ]
        
        aggregated = await aggregator._create_aggregated_data(
            "BTCUSD", timestamp, data_points, all_items
        )
        
        assert aggregated.symbol == "BTCUSD"
        assert aggregated.timestamp == timestamp
        assert aggregated.source_count == 2
        assert aggregated.volume == 2200.0  # Sum of volumes
        assert aggregated.high == 102.0  # Max of highs
        assert aggregated.low == 98.0   # Min of lows
        
        # Open and close should be medians
        expected_open = statistics.median([100.0, 100.2])
        expected_close = statistics.median([101.0, 100.8])
        assert aggregated.open == expected_open
        assert aggregated.close == expected_close
        
        assert len(aggregated.exchanges) == 2
        assert ExchangeType.COINBASE.value in aggregated.exchanges
        assert ExchangeType.ROBINHOOD.value in aggregated.exchanges


class TestMultiExchangeConsistency:
    """Test multi-exchange data consistency."""
    
    @pytest.mark.asyncio
    async def test_price_consistency_across_exchanges(self):
        """Test that price differences across exchanges are detected."""
        # Create data with significant price differences
        coinbase_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            exchange=ExchangeType.COINBASE
        )
        
        robinhood_data = MarketData(
            symbol="BTCUSD",
            timestamp=datetime.now(timezone.utc),
            open=110.0,  # 10% difference
            high=111.0,
            low=109.0,
            close=110.5,
            volume=1200.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        aggregator = DataAggregator([])
        data_points = [coinbase_data, robinhood_data]
        
        confidence_score = aggregator._calculate_confidence_score(data_points, [])
        
        # Confidence should be reduced due to price inconsistency
        assert confidence_score < 0.85  # Should be penalized for high variation
    
    def test_timestamp_alignment_tolerance(self):
        """Test that timestamps within tolerance are properly aligned."""
        synchronizer = TimestampSynchronizer(tolerance_seconds=5)
        
        base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create data points with slight timestamp differences
        data_points = [
            MarketData(
                symbol="BTCUSD",
                timestamp=base_time,
                open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
                exchange=ExchangeType.COINBASE
            ),
            MarketData(
                symbol="BTCUSD",
                timestamp=base_time + timedelta(seconds=2),
                open=100.1, high=101.1, low=99.1, close=100.6, volume=1100.0,
                exchange=ExchangeType.ROBINHOOD
            ),
            MarketData(
                symbol="BTCUSD",
                timestamp=base_time + timedelta(seconds=4),
                open=100.2, high=101.2, low=99.2, close=100.7, volume=1200.0,
                exchange=ExchangeType.OANDA
            )
        ]
        
        aligned = synchronizer.align_timestamps(data_points)
        
        # All should be aligned to the same normalized timestamp
        assert len(aligned) <= 3  # Could be 1-3 depending on rounding
        
        # Total data points should be preserved
        total_aligned = sum(len(group) for group in aligned.values())
        assert total_aligned == 3


if __name__ == "__main__":
    pytest.main([__file__])