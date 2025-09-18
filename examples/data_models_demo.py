"""
Demonstration of the AI trading platform data models.
"""
from datetime import datetime, timezone
from src.models import (
    MarketData, ExchangeType,
    TradingSignal, TradingAction,
    Portfolio, Position
)


def main():
    """Demonstrate data model usage."""
    print("=== AI Trading Platform Data Models Demo ===\n")
    
    # 1. Market Data Example
    print("1. Market Data:")
    market_data = MarketData(
        symbol="AAPL",
        timestamp=datetime(2023, 12, 1, 15, 30, tzinfo=timezone.utc),
        open=150.25,
        high=152.10,
        low=149.80,
        close=151.75,
        volume=1000000.0,
        exchange=ExchangeType.ROBINHOOD
    )
    print(f"   Symbol: {market_data.symbol}")
    print(f"   OHLCV: {market_data.open}/{market_data.high}/{market_data.low}/{market_data.close}/{market_data.volume}")
    print(f"   Exchange: {market_data.exchange}")
    print(f"   JSON: {market_data.model_dump_json()}\n")
    
    # 2. Trading Signal Example
    print("2. Trading Signal:")
    signal = TradingSignal(
        symbol="AAPL",
        action=TradingAction.BUY,
        confidence=0.85,
        position_size=0.1,
        target_price=155.0,
        stop_loss=145.0,
        timestamp=datetime.now(timezone.utc),
        model_version="cnn-lstm-v1.2.3"
    )
    print(f"   Signal: {signal.action} {signal.symbol}")
    print(f"   Confidence: {signal.confidence:.2%}")
    print(f"   Position Size: {signal.position_size:.1%}")
    print(f"   Target/Stop: ${signal.target_price}/${signal.stop_loss}")
    print(f"   Actionable: {signal.is_actionable()}")
    print(f"   JSON: {signal.model_dump_json()}\n")
    
    # 3. Position Example
    print("3. Position:")
    position = Position(
        symbol="AAPL",
        quantity=100.0,
        avg_cost=150.0,
        current_price=155.0,
        unrealized_pnl=500.0,  # 100 * (155 - 150)
        realized_pnl=200.0
    )
    print(f"   Position: {position.quantity} shares of {position.symbol}")
    print(f"   Cost Basis: ${position.avg_cost:.2f}")
    print(f"   Current Price: ${position.current_price:.2f}")
    print(f"   Market Value: ${position.market_value:.2f}")
    print(f"   P&L: ${position.unrealized_pnl:.2f} unrealized, ${position.realized_pnl:.2f} realized")
    print(f"   Total P&L: ${position.total_pnl:.2f}")
    print(f"   Position Type: {'Long' if position.is_long else 'Short'}")
    print(f"   JSON: {position.model_dump_json()}\n")
    
    # 4. Portfolio Example
    print("4. Portfolio:")
    portfolio = Portfolio(
        user_id="user123",
        positions={"AAPL": position},
        cash_balance=10000.0,
        total_value=25500.0,  # 10000 + (100 * 155)
        last_updated=datetime.now(timezone.utc)
    )
    print(f"   User: {portfolio.user_id}")
    print(f"   Cash Balance: ${portfolio.cash_balance:.2f}")
    print(f"   Positions Value: ${portfolio.positions_value:.2f}")
    print(f"   Total Value: ${portfolio.total_value:.2f}")
    print(f"   Total P&L: ${portfolio.total_pnl:.2f}")
    print(f"   Number of Positions: {len(portfolio.positions)}")
    
    # Demonstrate portfolio methods
    print("\n   Portfolio Operations:")
    print(f"   - Get AAPL position: {portfolio.get_position('AAPL').symbol if portfolio.get_position('AAPL') else 'Not found'}")
    print(f"   - Get TSLA position: {portfolio.get_position('TSLA') or 'Not found'}")
    
    # Add another position
    tsla_position = Position(
        symbol="TSLA",
        quantity=50.0,
        avg_cost=200.0,
        current_price=210.0,
        unrealized_pnl=500.0,  # 50 * (210 - 200)
        realized_pnl=0.0
    )
    portfolio.add_position(tsla_position)
    print(f"   - After adding TSLA: {len(portfolio.positions)} positions")
    print(f"   - Updated total value: ${portfolio.total_value:.2f}")
    
    print(f"\n   JSON: {portfolio.model_dump_json()}\n")
    
    # 5. Validation Examples
    print("5. Validation Examples:")
    
    try:
        # This should fail - high < low
        invalid_data = MarketData(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=148.0,  # Invalid: high < low
            low=149.0,
            close=148.5,
            volume=1000.0,
            exchange=ExchangeType.ROBINHOOD
        )
    except Exception as e:
        print(f"   ✓ Caught invalid market data: {e}")
    
    try:
        # This should fail - confidence > 1
        invalid_signal = TradingSignal(
            symbol="TEST",
            action=TradingAction.BUY,
            confidence=1.5,  # Invalid: > 1.0
            position_size=0.1,
            timestamp=datetime.now(timezone.utc),
            model_version="test"
        )
    except Exception as e:
        print(f"   ✓ Caught invalid signal: {e}")
    
    try:
        # This should fail - quantity = 0
        invalid_position = Position(
            symbol="TEST",
            quantity=0.0,  # Invalid: cannot be zero
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=0.0
        )
    except Exception as e:
        print(f"   ✓ Caught invalid position: {e}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()