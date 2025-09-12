"""
Comprehensive unit tests for data models.
"""
import pytest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError

from src.models import (
    MarketData, ExchangeType,
    TradingSignal, TradingAction,
    Portfolio, Position
)


class TestMarketData:
    """Test cases for MarketData model."""
    
    def test_valid_market_data(self):
        """Test creation of valid market data."""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime(2023, 12, 1, 15, 30),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        assert data.symbol == "AAPL"
        assert data.open == 150.0
        assert data.high == 155.0
        assert data.low == 149.0
        assert data.close == 154.0
        assert data.volume == 1000000.0
        assert data.exchange == ExchangeType.ROBINHOOD
    
    def test_symbol_validation(self):
        """Test symbol validation and normalization."""
        # Test empty symbol
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            MarketData(
                symbol="",
                timestamp=datetime.now(),
                open=150.0, high=155.0, low=149.0, close=154.0,
                volume=1000.0, exchange=ExchangeType.ROBINHOOD
            )
        
        # Test whitespace handling
        data = MarketData(
            symbol="  aapl  ",
            timestamp=datetime.now(),
            open=150.0, high=155.0, low=149.0, close=154.0,
            volume=1000.0, exchange=ExchangeType.ROBINHOOD
        )
        assert data.symbol == "AAPL"
    
    def test_price_validation(self):
        """Test price validation rules."""
        base_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(),
            "volume": 1000.0,
            "exchange": ExchangeType.ROBINHOOD
        }
        
        # Test negative prices
        with pytest.raises(ValidationError):
            MarketData(open=-150.0, high=155.0, low=149.0, close=154.0, **base_data)
        
        # Test zero prices
        with pytest.raises(ValidationError):
            MarketData(open=0.0, high=155.0, low=149.0, close=154.0, **base_data)
        
        # Test high < low
        with pytest.raises(ValidationError, match="High price must be >= low price"):
            MarketData(open=150.0, high=148.0, low=149.0, close=154.0, **base_data)
        
        # Test high < open
        with pytest.raises(ValidationError, match="High price must be >= open price"):
            MarketData(open=150.0, high=149.0, low=148.0, close=149.5, **base_data)
        
        # Test low > high
        with pytest.raises(ValidationError, match="High price must be >= low price"):
            MarketData(open=150.0, high=155.0, low=156.0, close=154.0, **base_data)
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        
        with pytest.raises(ValidationError, match="Timestamp cannot be in the future"):
            MarketData(
                symbol="AAPL",
                timestamp=future_time,
                open=150.0, high=155.0, low=149.0, close=154.0,
                volume=1000.0, exchange=ExchangeType.ROBINHOOD
            )
    
    def test_volume_validation(self):
        """Test volume validation."""
        # Negative volume should fail
        with pytest.raises(ValidationError):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=150.0, high=155.0, low=149.0, close=154.0,
                volume=-1000.0, exchange=ExchangeType.ROBINHOOD
            )
        
        # Zero volume should be allowed
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0, high=155.0, low=149.0, close=154.0,
            volume=0.0, exchange=ExchangeType.ROBINHOOD
        )
        assert data.volume == 0.0


class TestTradingSignal:
    """Test cases for TradingSignal model."""
    
    def test_valid_trading_signal(self):
        """Test creation of valid trading signal."""
        signal = TradingSignal(
            symbol="AAPL",
            action=TradingAction.BUY,
            confidence=0.85,
            position_size=0.1,
            target_price=155.0,
            stop_loss=145.0,
            timestamp=datetime.now(),
            model_version="v1.0.0"
        )
        
        assert signal.symbol == "AAPL"
        assert signal.action == TradingAction.BUY
        assert signal.confidence == 0.85
        assert signal.position_size == 0.1
        assert signal.target_price == 155.0
        assert signal.stop_loss == 145.0
        assert signal.model_version == "v1.0.0"
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        base_data = {
            "symbol": "AAPL",
            "action": TradingAction.BUY,
            "position_size": 0.1,
            "timestamp": datetime.now(),
            "model_version": "v1.0.0"
        }
        
        # Test confidence > 1
        with pytest.raises(ValidationError):
            TradingSignal(confidence=1.5, **base_data)
        
        # Test confidence < 0
        with pytest.raises(ValidationError):
            TradingSignal(confidence=-0.1, **base_data)
        
        # Test valid boundary values
        signal1 = TradingSignal(confidence=0.0, **base_data)
        assert signal1.confidence == 0.0
        
        signal2 = TradingSignal(confidence=1.0, **base_data)
        assert signal2.confidence == 1.0
    
    def test_position_size_validation(self):
        """Test position size validation."""
        base_data = {
            "symbol": "AAPL",
            "action": TradingAction.BUY,
            "confidence": 0.8,
            "timestamp": datetime.now(),
            "model_version": "v1.0.0"
        }
        
        # Test position_size > 1
        with pytest.raises(ValidationError):
            TradingSignal(position_size=1.5, **base_data)
        
        # Test position_size < 0
        with pytest.raises(ValidationError):
            TradingSignal(position_size=-0.1, **base_data)
    
    def test_stop_loss_validation(self):
        """Test stop loss validation relative to target price and action."""
        base_data = {
            "symbol": "AAPL",
            "confidence": 0.8,
            "position_size": 0.1,
            "timestamp": datetime.now(),
            "model_version": "v1.0.0"
        }
        
        # BUY signal: stop loss should be below target price
        with pytest.raises(ValidationError, match="Stop loss must be below target price for BUY signals"):
            TradingSignal(
                action=TradingAction.BUY,
                target_price=150.0,
                stop_loss=155.0,
                **base_data
            )
        
        # SELL signal: stop loss should be above target price
        with pytest.raises(ValidationError, match="Stop loss must be above target price for SELL signals"):
            TradingSignal(
                action=TradingAction.SELL,
                target_price=150.0,
                stop_loss=145.0,
                **base_data
            )
        
        # Valid BUY signal
        signal = TradingSignal(
            action=TradingAction.BUY,
            target_price=150.0,
            stop_loss=145.0,
            **base_data
        )
        assert signal.stop_loss == 145.0
    
    def test_is_actionable(self):
        """Test actionable signal detection."""
        # High confidence BUY signal should be actionable
        signal1 = TradingSignal(
            symbol="AAPL",
            action=TradingAction.BUY,
            confidence=0.8,
            position_size=0.1,
            timestamp=datetime.now(),
            model_version="v1.0.0"
        )
        assert signal1.is_actionable() is True
        
        # Low confidence BUY signal should not be actionable
        signal2 = TradingSignal(
            symbol="AAPL",
            action=TradingAction.BUY,
            confidence=0.3,
            position_size=0.1,
            timestamp=datetime.now(),
            model_version="v1.0.0"
        )
        assert signal2.is_actionable() is False
        
        # HOLD signal should not be actionable regardless of confidence
        signal3 = TradingSignal(
            symbol="AAPL",
            action=TradingAction.HOLD,
            confidence=0.9,
            position_size=0.0,
            timestamp=datetime.now(),
            model_version="v1.0.0"
        )
        assert signal3.is_actionable() is False


class TestPosition:
    """Test cases for Position model."""
    
    def test_valid_position(self):
        """Test creation of valid position."""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100.0
        assert position.avg_cost == 150.0
        assert position.current_price == 155.0
        assert position.unrealized_pnl == 500.0
        assert position.realized_pnl == 0.0
    
    def test_zero_quantity_validation(self):
        """Test that zero quantity is not allowed."""
        with pytest.raises(ValidationError, match="Position quantity cannot be zero"):
            Position(
                symbol="AAPL",
                quantity=0.0,
                avg_cost=150.0,
                current_price=155.0,
                unrealized_pnl=0.0
            )
    
    def test_unrealized_pnl_validation(self):
        """Test unrealized P&L calculation validation."""
        # Correct P&L calculation
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0  # 100 * (155 - 150) = 500
        )
        assert position.unrealized_pnl == 500.0
        
        # Incorrect P&L calculation should fail
        with pytest.raises(ValidationError, match="Unrealized P&L mismatch"):
            Position(
                symbol="AAPL",
                quantity=100.0,
                avg_cost=150.0,
                current_price=155.0,
                unrealized_pnl=600.0  # Should be 500
            )
    
    def test_position_properties(self):
        """Test position calculated properties."""
        # Long position
        long_position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0
        )
        
        assert long_position.market_value == 15500.0  # 100 * 155
        assert long_position.total_pnl == 700.0  # 500 + 200
        assert long_position.is_long is True
        assert long_position.is_short is False
        
        # Short position
        short_position = Position(
            symbol="TSLA",
            quantity=-50.0,
            avg_cost=200.0,
            current_price=190.0,
            unrealized_pnl=500.0,  # -50 * (190 - 200) = 500
            realized_pnl=-100.0
        )
        
        assert short_position.market_value == 9500.0  # 50 * 190
        assert short_position.total_pnl == 400.0  # 500 + (-100)
        assert short_position.is_long is False
        assert short_position.is_short is True


class TestPortfolio:
    """Test cases for Portfolio model."""
    
    def test_valid_portfolio(self):
        """Test creation of valid portfolio."""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0
        )
        
        portfolio = Portfolio(
            user_id="user123",
            positions={"AAPL": position},
            cash_balance=10000.0,
            total_value=25500.0,  # 10000 + (100 * 155)
            last_updated=datetime.now()
        )
        
        assert portfolio.user_id == "user123"
        assert len(portfolio.positions) == 1
        assert portfolio.cash_balance == 10000.0
        assert portfolio.total_value == 25500.0
    
    def test_total_value_validation(self):
        """Test total value calculation validation."""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0
        )
        
        # Incorrect total value should fail
        with pytest.raises(ValidationError, match="Total value mismatch"):
            Portfolio(
                user_id="user123",
                positions={"AAPL": position},
                cash_balance=10000.0,
                total_value=20000.0,  # Should be 25500
                last_updated=datetime.now()
            )
    
    def test_position_symbol_key_validation(self):
        """Test that position symbols match dictionary keys."""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0
        )
        
        # Mismatched symbol and key should fail
        with pytest.raises(ValidationError, match="Position symbol AAPL doesn't match key TSLA"):
            Portfolio(
                user_id="user123",
                positions={"TSLA": position},  # Wrong key
                cash_balance=10000.0,
                total_value=25500.0,
                last_updated=datetime.now()
            )
    
    def test_portfolio_properties(self):
        """Test portfolio calculated properties."""
        position1 = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0
        )
        
        position2 = Position(
            symbol="TSLA",
            quantity=50.0,
            avg_cost=200.0,
            current_price=210.0,
            unrealized_pnl=500.0,
            realized_pnl=100.0
        )
        
        portfolio = Portfolio(
            user_id="user123",
            positions={"AAPL": position1, "TSLA": position2},
            cash_balance=5000.0,
            total_value=31000.0,  # 5000 + 15500 + 10500
            last_updated=datetime.now()
        )
        
        assert portfolio.positions_value == 26000.0  # 15500 + 10500
        assert portfolio.total_pnl == 1300.0  # (500+200) + (500+100)
        assert portfolio.unrealized_pnl == 1000.0  # 500 + 500
        assert portfolio.realized_pnl == 300.0  # 200 + 100
    
    def test_portfolio_methods(self):
        """Test portfolio manipulation methods."""
        portfolio = Portfolio(
            user_id="user123",
            positions={},
            cash_balance=10000.0,
            total_value=10000.0,
            last_updated=datetime.now()
        )
        
        # Test adding position
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0
        )
        
        portfolio.add_position(position)
        assert len(portfolio.positions) == 1
        assert portfolio.get_position("AAPL") == position
        assert portfolio.total_value == 25500.0  # Recalculated
        
        # Test removing position
        removed = portfolio.remove_position("AAPL")
        assert removed == position
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 10000.0  # Recalculated
        
        # Test getting non-existent position
        assert portfolio.get_position("TSLA") is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_market_data_edge_cases(self):
        """Test market data edge cases."""
        # Very small prices
        data = MarketData(
            symbol="PENNY",
            timestamp=datetime.now(),
            open=0.001,
            high=0.002,
            low=0.0005,
            close=0.0015,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        assert data.open == 0.001
        
        # Very large volume
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1e12,  # 1 trillion
            exchange=ExchangeType.ROBINHOOD
        )
        assert data.volume == 1e12
    
    def test_trading_signal_edge_cases(self):
        """Test trading signal edge cases."""
        # Signal without target price or stop loss
        signal = TradingSignal(
            symbol="AAPL",
            action=TradingAction.BUY,
            confidence=0.8,
            position_size=0.1,
            timestamp=datetime.now(),
            model_version="v1.0.0"
        )
        assert signal.target_price is None
        assert signal.stop_loss is None
        
        # Very long model version
        long_version = "v" + "1.0.0-" + "a" * 100
        signal = TradingSignal(
            symbol="AAPL",
            action=TradingAction.BUY,
            confidence=0.8,
            position_size=0.1,
            timestamp=datetime.now(),
            model_version=long_version
        )
        assert signal.model_version == long_version
    
    def test_position_edge_cases(self):
        """Test position edge cases."""
        # Very small position
        position = Position(
            symbol="AAPL",
            quantity=0.001,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=0.005  # 0.001 * (155 - 150)
        )
        assert position.quantity == 0.001
        
        # Negative quantity (short position)
        position = Position(
            symbol="AAPL",
            quantity=-100.0,
            avg_cost=150.0,
            current_price=145.0,
            unrealized_pnl=500.0  # -100 * (145 - 150) = 500
        )
        assert position.is_short is True
        assert position.unrealized_pnl == 500.0
    
    def test_portfolio_edge_cases(self):
        """Test portfolio edge cases."""
        # Empty portfolio
        portfolio = Portfolio(
            user_id="user123",
            positions={},
            cash_balance=0.0,
            total_value=0.0,
            last_updated=datetime.now()
        )
        assert portfolio.positions_value == 0.0
        assert portfolio.total_pnl == 0.0
        
        # Portfolio with only cash
        portfolio = Portfolio(
            user_id="user123",
            positions={},
            cash_balance=100000.0,
            total_value=100000.0,
            last_updated=datetime.now()
        )
        assert portfolio.total_value == 100000.0