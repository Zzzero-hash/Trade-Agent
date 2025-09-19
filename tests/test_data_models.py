"""
Unit tests for advanced market data models.

Tests comprehensive functionality including:
- MarketData validation and calculations
- OrderBook microstructure metrics
- Trade execution quality metrics
- Statistical outlier detection
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data.models import (
    MarketData, OrderBook, Trade, OrderBookLevel,
    MarketMetadata, TechnicalIndicators, DataValidator,
    AssetClass, TradeSide, MarketRegime,
    create_market_data, create_order_book, create_trade
)


class TestMarketData:
    """Test cases for MarketData class."""
    
    def test_market_data_creation(self):
        """Test basic MarketData creation and validation."""
        timestamp = datetime.now()
        data = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            timeframe="1m",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        assert data.symbol == "AAPL"
        assert data.timeframe == "1m"
        assert data.open == 150.0
        assert data.high == 152.0
        assert data.low == 149.0
        assert data.close == 151.0
        assert data.volume == 1000000
        assert data.vwap is not None  # Should be calculated
        assert data.typical_price == (152.0 + 149.0 + 151.0) / 3
    
    def test_market_data_validation_errors(self):
        """Test MarketData validation catches errors."""
        timestamp = datetime.now()
        
        # Test negative prices
        with pytest.raises(ValueError, match="All OHLCV prices must be positive"):
            MarketData(
                timestamp=timestamp,
                symbol="AAPL",
                timeframe="1m",
                open=-150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000
            )
        
        # Test inconsistent OHLC
        with pytest.raises(ValueError, match="OHLC prices are inconsistent"):
            MarketData(
                timestamp=timestamp,
                symbol="AAPL",
                timeframe="1m",
                open=150.0,
                high=148.0,  # High less than open
                low=149.0,
                close=151.0,
                volume=1000000
            )
        
        # Test negative volume
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            MarketData(
                timestamp=timestamp,
                symbol="AAPL",
                timeframe="1m",
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=-1000
            )
    
    def test_outlier_detection(self):
        """Test statistical outlier detection."""
        timestamp = datetime.now()
        
        # Create data with extreme price range (should be flagged as outlier)
        data = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            timeframe="1m",
            open=100.0,
            high=130.0,  # 30% range
            low=100.0,
            close=125.0,
            volume=1000000
        )
        
        assert data.is_outlier
        assert "extreme_price_range" in data.outlier_reasons
        
        # Create data with zero volume (should be flagged)
        data_zero_vol = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            timeframe="1m",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=0
        )
        
        assert data_zero_vol.is_outlier
        assert "zero_volume" in data_zero_vol.outlier_reasons
    
    def test_multi_timeframe_data(self):
        """Test multi-timeframe data functionality."""
        timestamp = datetime.now()
        data = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            timeframe="1m",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        # Add 5-minute timeframe data
        data.add_timeframe_data("5m", {
            "open": 148.0,
            "high": 153.0,
            "low": 147.0,
            "close": 151.0,
            "volume": 5000000
        })
        
        tf_data = data.get_timeframe_data("5m")
        assert tf_data is not None
        assert tf_data["open"] == 148.0
        assert tf_data["volume"] == 5000000
    
    def test_serialization(self):
        """Test MarketData serialization and deserialization."""
        timestamp = datetime.now()
        original = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            timeframe="1m",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        # Convert to dict and back
        data_dict = original.to_dict()
        reconstructed = MarketData.from_dict(data_dict)
        
        assert reconstructed.symbol == original.symbol
        assert reconstructed.open == original.open
        assert reconstructed.high == original.high
        assert reconstructed.low == original.low
        assert reconstructed.close == original.close
        assert reconstructed.volume == original.volume


class TestOrderBook:
    """Test cases for OrderBook class."""
    
    def test_order_book_creation(self):
        """Test OrderBook creation and basic properties."""
        timestamp = datetime.now()
        
        bids = [
            OrderBookLevel(100.0, 1000),
            OrderBookLevel(99.5, 2000),
            OrderBookLevel(99.0, 1500)
        ]
        asks = [
            OrderBookLevel(100.5, 800),
            OrderBookLevel(101.0, 1200),
            OrderBookLevel(101.5, 900)
        ]
        
        book = OrderBook(
            timestamp=timestamp,
            symbol="AAPL",
            bids=bids,
            asks=asks
        )
        
        assert book.best_bid == 100.0
        assert book.best_ask == 100.5
        assert book.mid_price == 100.25
        assert book.spread == 0.5
        assert abs(book.spread_bps - 49.88) < 0.1  # Approximately 50 bps
    
    def test_order_book_sorting(self):
        """Test that order book levels are properly sorted."""
        timestamp = datetime.now()
        
        # Create unsorted bids and asks
        bids = [
            OrderBookLevel(99.0, 1500),
            OrderBookLevel(100.0, 1000),  # Should be first
            OrderBookLevel(99.5, 2000)
        ]
        asks = [
            OrderBookLevel(101.0, 1200),
            OrderBookLevel(100.5, 800),   # Should be first
            OrderBookLevel(101.5, 900)
        ]
        
        book = OrderBook(
            timestamp=timestamp,
            symbol="AAPL",
            bids=bids,
            asks=asks
        )
        
        # Check that bids are sorted descending
        assert book.bids[0].price == 100.0
        assert book.bids[1].price == 99.5
        assert book.bids[2].price == 99.0
        
        # Check that asks are sorted ascending
        assert book.asks[0].price == 100.5
        assert book.asks[1].price == 101.0
        assert book.asks[2].price == 101.5
    
    def test_order_book_depth_analysis(self):
        """Test order book depth analysis methods."""
        timestamp = datetime.now()
        
        bids = [
            OrderBookLevel(100.0, 1000),
            OrderBookLevel(99.5, 2000),
            OrderBookLevel(99.0, 1500)
        ]
        asks = [
            OrderBookLevel(100.5, 800),
            OrderBookLevel(101.0, 1200),
            OrderBookLevel(101.5, 900)
        ]
        
        book = OrderBook(
            timestamp=timestamp,
            symbol="AAPL",
            bids=bids,
            asks=asks
        )
        
        # Test depth retrieval
        bid_depth = book.get_depth('bid', 2)
        assert len(bid_depth) == 2
        assert bid_depth[0].price == 100.0
        assert bid_depth[1].price == 99.5
        
        # Test total size calculation
        bid_size = book.get_total_size('bid', 2)
        assert bid_size == 3000  # 1000 + 2000
        
        ask_size = book.get_total_size('ask', 2)
        assert ask_size == 2000  # 800 + 1200
        
        # Test weighted price calculation
        bid_weighted = book.get_weighted_price('bid', 2)
        expected_bid_weighted = (100.0 * 1000 + 99.5 * 2000) / 3000
        assert abs(bid_weighted - expected_bid_weighted) < 0.001
        
        # Test imbalance calculation
        imbalance = book.calculate_imbalance(2)
        expected_imbalance = (3000 - 2000) / (3000 + 2000)  # 0.2
        assert abs(imbalance - expected_imbalance) < 0.001
    
    def test_order_book_validation_errors(self):
        """Test OrderBook validation catches errors."""
        timestamp = datetime.now()
        
        # Test invalid price
        with pytest.raises(ValueError, match="Price must be positive"):
            OrderBookLevel(-100.0, 1000)
        
        # Test invalid size
        with pytest.raises(ValueError, match="Size must be positive"):
            OrderBookLevel(100.0, -1000)


class TestTrade:
    """Test cases for Trade class."""
    
    def test_trade_creation(self):
        """Test Trade creation and basic properties."""
        timestamp = datetime.now()
        
        trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            price=100.25,
            size=1000,
            side=TradeSide.BUY,
            bid_price=100.0,
            ask_price=100.5
        )
        
        assert trade.symbol == "AAPL"
        assert trade.price == 100.25
        assert trade.size == 1000
        assert trade.side == TradeSide.BUY
        assert trade.notional_value == 100250.0
        assert trade.mid_price == 100.25  # (100.0 + 100.5) / 2
        assert trade.is_aggressive is True  # Buy at mid price (aggressive)
        assert trade.liquidity_flag == 'taker'
    
    def test_trade_execution_quality_metrics(self):
        """Test trade execution quality calculations."""
        timestamp = datetime.now()
        
        # Test aggressive buy trade
        trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            price=100.5,  # At ask price
            size=1000,
            side=TradeSide.BUY,
            bid_price=100.0,
            ask_price=100.5
        )
        
        assert trade.mid_price == 100.25
        assert trade.effective_spread == 0.5  # 2 * |100.5 - 100.25|
        assert trade.is_aggressive is True
        assert trade.liquidity_flag == 'taker'
        
        # Test passive sell trade
        passive_trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            price=100.3,  # Above mid price (100.25) for sell = passive
            size=500,
            side=TradeSide.SELL,
            bid_price=100.0,
            ask_price=100.5
        )
        
        assert passive_trade.is_aggressive is False
        assert passive_trade.liquidity_flag == 'maker'
    
    def test_market_impact_calculation(self):
        """Test market impact calculation."""
        timestamp = datetime.now()
        
        buy_trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            price=100.5,
            size=1000,
            side=TradeSide.BUY
        )
        
        # Calculate impact relative to reference price
        reference_price = 100.0
        impact = buy_trade.calculate_market_impact(reference_price)
        expected_impact = (100.5 - 100.0) / 100.0  # 0.005 or 0.5%
        assert abs(impact - expected_impact) < 0.001
        
        sell_trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            price=99.5,
            size=1000,
            side=TradeSide.SELL
        )
        
        sell_impact = sell_trade.calculate_market_impact(reference_price)
        expected_sell_impact = (100.0 - 99.5) / 100.0  # 0.005 or 0.5%
        assert abs(sell_impact - expected_sell_impact) < 0.001
    
    def test_trade_validation_errors(self):
        """Test Trade validation catches errors."""
        timestamp = datetime.now()
        
        # Test negative price
        with pytest.raises(ValueError, match="Trade price must be positive"):
            Trade(
                timestamp=timestamp,
                symbol="AAPL",
                price=-100.0,
                size=1000,
                side=TradeSide.BUY
            )
        
        # Test negative size
        with pytest.raises(ValueError, match="Trade size must be positive"):
            Trade(
                timestamp=timestamp,
                symbol="AAPL",
                price=100.0,
                size=-1000,
                side=TradeSide.BUY
            )


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_price_series_validation(self):
        """Test price series validation and outlier detection."""
        # Create normal price series
        normal_prices = [100.0, 100.5, 101.0, 100.8, 101.2, 100.9, 101.1]
        outlier_flags, reasons = DataValidator.validate_price_series(normal_prices)
        
        # Should not detect outliers in normal series
        assert not any(outlier_flags)
        
        # Create series with outlier
        outlier_prices = [100.0, 100.5, 200.0, 100.8, 101.2]  # 200.0 is clear outlier
        outlier_flags, reasons = DataValidator.validate_price_series(outlier_prices, z_threshold=1.5)
        
        # Should detect the outlier
        assert any(outlier_flags)
    
    def test_volume_series_validation(self):
        """Test volume series validation."""
        # Create normal volume series
        normal_volumes = [1000000, 1200000, 800000, 1100000, 900000]
        outlier_flags, reasons = DataValidator.validate_volume_series(normal_volumes)
        
        # Should not detect outliers in normal series
        assert not any(outlier_flags)
        
        # Create series with zero volume
        zero_vol_series = [1000000, 1200000, 0, 1100000, 900000]
        outlier_flags, reasons = DataValidator.validate_volume_series(zero_vol_series)
        
        # Should detect zero volume
        assert outlier_flags[2]  # Third element (index 2) should be flagged
        assert any("zero_volume" in reason for reason in reasons[2])
        
        # Create series with volume spike
        spike_series = [1000000, 1200000, 15000000, 1100000, 900000]  # 15M is spike
        outlier_flags, reasons = DataValidator.validate_volume_series(spike_series, multiplier_threshold=5.0)
        
        # Should detect volume spike
        assert outlier_flags[2]  # Third element should be flagged


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_market_data(self):
        """Test create_market_data factory function."""
        timestamp = datetime.now()
        metadata = MarketMetadata(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            lot_size=100,
            trading_hours={"monday": ("09:30", "16:00")},
            timezone="US/Eastern"
        )
        
        data = create_market_data(
            symbol="AAPL",
            timestamp=timestamp,
            timeframe="1m",
            ohlcv=(150.0, 152.0, 149.0, 151.0, 1000000),
            metadata=metadata
        )
        
        assert data.symbol == "AAPL"
        assert data.open == 150.0
        assert data.metadata == metadata
    
    def test_create_order_book(self):
        """Test create_order_book factory function."""
        timestamp = datetime.now()
        
        book = create_order_book(
            symbol="AAPL",
            timestamp=timestamp,
            bids=[(100.0, 1000), (99.5, 2000)],
            asks=[(100.5, 800), (101.0, 1200)]
        )
        
        assert book.symbol == "AAPL"
        assert book.best_bid == 100.0
        assert book.best_ask == 100.5
        assert len(book.bids) == 2
        assert len(book.asks) == 2
    
    def test_create_trade(self):
        """Test create_trade factory function."""
        timestamp = datetime.now()
        
        trade = create_trade(
            symbol="AAPL",
            timestamp=timestamp,
            price=100.25,
            size=1000,
            side="buy",
            bid_price=100.0,
            ask_price=100.5
        )
        
        assert trade.symbol == "AAPL"
        assert trade.price == 100.25
        assert trade.side == TradeSide.BUY
        assert trade.bid_price == 100.0


if __name__ == "__main__":
    pytest.main([__file__])