"""
Trading Environment Demo

This example demonstrates how to use the TradingEnvironment for reinforcement learning
training and evaluation. It shows the key features including:

1. Environment setup with realistic market data
2. Action space and observation space usage
3. Risk-adjusted reward calculation
4. Portfolio management and performance tracking
5. Integration with RL training frameworks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.trading_environment import TradingEnvironment, TradingConfig, ActionType


def generate_realistic_market_data(
    symbols: List[str],
    start_date: str = "2023-01-01",
    periods: int = 252,  # One trading year
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic market data with correlated returns and volatility clustering.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    # Define initial prices and characteristics for each symbol
    symbol_params = {
        "AAPL": {"initial_price": 150.0, "volatility": 0.25, "drift": 0.08},
        "GOOGL": {"initial_price": 2500.0, "volatility": 0.30, "drift": 0.10},
        "MSFT": {"initial_price": 300.0, "volatility": 0.22, "drift": 0.07},
        "TSLA": {"initial_price": 200.0, "volatility": 0.45, "drift": 0.15},
        "SPY": {"initial_price": 400.0, "volatility": 0.18, "drift": 0.06},
    }

    data = []

    for symbol in symbols:
        if symbol not in symbol_params:
            # Default parameters for unknown symbols
            params = {"initial_price": 100.0, "volatility": 0.25, "drift": 0.08}
        else:
            params = symbol_params[symbol]

        # Generate correlated returns with volatility clustering
        returns = []
        volatility = params["volatility"]

        for i in range(periods):
            # GARCH-like volatility clustering
            if i > 0:
                volatility = (
                    0.95 * volatility
                    + 0.05 * abs(returns[-1])
                    + 0.01 * np.random.normal(0, 0.1)
                )
                volatility = max(0.05, min(0.8, volatility))  # Bound volatility

            # Generate return with drift and volatility
            daily_return = (params["drift"] / 252) + volatility * np.random.normal(
                0, 1
            ) / np.sqrt(252)
            returns.append(daily_return)

        # Convert returns to prices
        prices = [params["initial_price"]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV data
        for i, date in enumerate(dates):
            price = prices[i]

            # Generate intraday volatility
            intraday_vol = volatility * 0.3  # Intraday volatility is typically lower
            high_factor = 1 + abs(np.random.normal(0, intraday_vol))
            low_factor = 1 - abs(np.random.normal(0, intraday_vol))
            open_factor = 1 + np.random.normal(0, intraday_vol * 0.5)

            high = price * high_factor
            low = price * low_factor
            open_price = price * open_factor

            # Ensure OHLC relationships are valid
            high = max(high, price, open_price)
            low = min(low, price, open_price)

            # Generate volume with some correlation to volatility
            base_volume = (
                1000000 if symbol != "GOOGL" else 500000
            )  # GOOGL typically lower volume
            volume_factor = 1 + abs(returns[i]) * 5  # Higher volume on big moves
            volume = base_volume * volume_factor * (1 + np.random.normal(0, 0.3))
            volume = max(100000, volume)  # Minimum volume

            data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume,
                }
            )

    return pd.DataFrame(data)


def demonstrate_basic_usage():
    """Demonstrate basic TradingEnvironment usage."""
    print("=== Basic TradingEnvironment Usage ===\n")

    # Generate market data
    symbols = ["AAPL", "GOOGL", "MSFT"]
    market_data = generate_realistic_market_data(symbols, periods=100)

    # Create trading configuration
    config = TradingConfig(
        initial_balance=100000.0,
        max_position_size=0.3,  # Max 30% per position
        transaction_cost=0.001,  # 0.1% transaction cost
        slippage=0.0005,  # 0.05% slippage
        lookback_window=20,  # 20-day lookback
        max_drawdown_limit=0.25,  # 25% max drawdown
        reward_scaling=1000.0,
    )

    # Initialize environment
    env = TradingEnvironment(market_data=market_data, config=config, symbols=symbols)

    print(f"Environment initialized with {len(symbols)} symbols")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Max steps: {env.max_steps}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nInitial portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"Initial cash balance: ${info['cash_balance']:,.2f}")

    # Demonstrate different types of actions
    print("\n--- Demonstrating Trading Actions ---")

    # 1. Buy actions
    print("\n1. Executing BUY actions...")
    buy_action = np.array(
        [
            1.0,
            0.2,  # BUY 20% position in AAPL
            1.0,
            0.15,  # BUY 15% position in GOOGL
            1.0,
            0.1,  # BUY 10% position in MSFT
        ]
    )

    obs, reward, terminated, truncated, info = env.step(buy_action)
    print(f"After BUY: Portfolio value = ${info['portfolio_value']:,.2f}")
    print(f"Positions: {info['positions']}")
    print(f"Reward: {reward:.4f}")

    # 2. Hold actions
    print("\n2. Executing HOLD actions...")
    hold_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # HOLD all

    for _ in range(5):  # Hold for 5 steps
        obs, reward, terminated, truncated, info = env.step(hold_action)

    print(f"After HOLD: Portfolio value = ${info['portfolio_value']:,.2f}")
    print(f"Total return: {info['total_return']:.2%}")

    # 3. Sell actions
    print("\n3. Executing SELL actions...")
    sell_action = np.array(
        [
            2.0,
            0.5,  # SELL 50% of AAPL position
            0.0,
            0.0,  # HOLD GOOGL
            2.0,
            1.0,  # SELL all MSFT
        ]
    )

    obs, reward, terminated, truncated, info = env.step(sell_action)
    print(f"After SELL: Portfolio value = ${info['portfolio_value']:,.2f}")
    print(f"Positions: {info['positions']}")

    return env


def demonstrate_performance_tracking():
    """Demonstrate performance tracking and metrics."""
    print("\n=== Performance Tracking Demo ===\n")

    # Create environment
    symbols = ["AAPL", "SPY"]
    market_data = generate_realistic_market_data(symbols, periods=150)
    config = TradingConfig(initial_balance=50000.0, lookback_window=10)
    env = TradingEnvironment(market_data, config, symbols)

    # Run a simple trading strategy
    obs, info = env.reset(seed=123)

    portfolio_values = [info["portfolio_value"]]
    returns = []

    print("Running simple momentum strategy...")

    for step in range(50):
        # Simple momentum strategy: buy when recent returns are positive
        current_prices = env._get_current_prices()

        # Calculate simple momentum signal
        if len(env.returns_history) >= 5:
            recent_return = np.mean(env.returns_history[-5:])
            if recent_return > 0.01:  # Buy signal
                action = np.array([1.0, 0.1, 1.0, 0.1])  # Buy both
            elif recent_return < -0.01:  # Sell signal
                action = np.array([2.0, 0.5, 2.0, 0.5])  # Sell half
            else:  # Hold
                action = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            action = np.array([1.0, 0.2, 1.0, 0.2])  # Initial buy

        obs, reward, terminated, truncated, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        returns.append(info["total_return"])

        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break

    # Calculate and display performance metrics
    metrics = env.get_portfolio_metrics()

    print(f"\n--- Performance Metrics ---")
    print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")

    return portfolio_values, returns


def demonstrate_risk_management():
    """Demonstrate risk management features."""
    print("\n=== Risk Management Demo ===\n")

    # Create high-risk configuration
    config = TradingConfig(
        initial_balance=10000.0,
        max_position_size=0.8,  # Allow large positions
        max_drawdown_limit=0.15,  # Strict drawdown limit
        transaction_cost=0.002,  # Higher transaction costs
        slippage=0.001,  # Higher slippage
    )

    symbols = ["TSLA"]  # High volatility stock
    market_data = generate_realistic_market_data(symbols, periods=80)
    env = TradingEnvironment(market_data, config, symbols)

    obs, info = env.reset(seed=456)

    print("Testing risk management with aggressive trading...")
    print(f"Max drawdown limit: {config.max_drawdown_limit:.1%}")
    print(f"Initial portfolio: ${info['portfolio_value']:,.2f}")

    # Aggressive trading strategy
    for step in range(30):
        # Alternate between large buy and sell positions
        if step % 4 == 0:
            action = np.array([1.0, 0.8])  # Large buy
        elif step % 4 == 2:
            action = np.array([2.0, 0.8])  # Large sell
        else:
            action = np.array([0.0, 0.0])  # Hold

        obs, reward, terminated, truncated, info = env.step(action)

        current_drawdown = info["drawdown"]
        print(
            f"Step {step + 1}: Portfolio=${info['portfolio_value']:,.0f}, "
            f"Drawdown={current_drawdown:.1%}, Reward={reward:.2f}"
        )

        if terminated:
            print(f"\n⚠️  Episode terminated due to risk limits!")
            print(f"Final drawdown: {current_drawdown:.1%}")
            break

        if truncated:
            print(f"\nEpisode completed normally")
            break

    # Show final risk metrics
    final_metrics = env.get_portfolio_metrics()
    print(f"\nFinal Risk Assessment:")
    if final_metrics:
        print(f"Max Drawdown Experienced: {final_metrics['max_drawdown']:.1%}")
        print(f"Volatility: {final_metrics['volatility']:.1%}")
        print(f"Total Trades: {final_metrics['total_trades']}")
    else:
        print("Insufficient data for comprehensive metrics")
        print(f"Total Trades: {len(env.trade_history)}")
        print(f"Final Portfolio Value: ${env.portfolio_value:,.2f}")


def demonstrate_multi_asset_portfolio():
    """Demonstrate multi-asset portfolio management."""
    print("\n=== Multi-Asset Portfolio Demo ===\n")

    # Create diversified portfolio
    symbols = ["AAPL", "GOOGL", "MSFT", "SPY", "TSLA"]
    market_data = generate_realistic_market_data(symbols, periods=120)

    config = TradingConfig(
        initial_balance=200000.0,
        max_position_size=0.25,  # Max 25% per asset
        lookback_window=15,
    )

    env = TradingEnvironment(market_data, config, symbols)
    obs, info = env.reset(seed=789)

    print(f"Managing portfolio with {len(symbols)} assets")
    print(f"Initial balance: ${info['portfolio_value']:,.2f}")

    # Implement simple diversification strategy
    target_weights = {
        "AAPL": 0.20,
        "GOOGL": 0.20,
        "MSFT": 0.20,
        "SPY": 0.25,
        "TSLA": 0.15,
    }

    print(f"\nTarget allocation: {target_weights}")

    # Initial allocation
    print("\nExecuting initial allocation...")
    initial_action = []
    for symbol in symbols:
        initial_action.extend([1.0, target_weights[symbol]])  # BUY to target weight

    obs, reward, terminated, truncated, info = env.step(np.array(initial_action))

    print(f"After initial allocation:")
    current_prices = env._get_current_prices()
    for symbol in symbols:
        position_value = info["positions"][symbol] * current_prices[symbol]
        weight = position_value / info["portfolio_value"]
        print(f"  {symbol}: {weight:.1%} (target: {target_weights[symbol]:.1%})")

    # Rebalancing simulation
    print(f"\nSimulating periodic rebalancing...")

    for rebalance_step in range(10):
        # Hold for several steps
        for _ in range(5):
            hold_action = np.array([0.0, 0.0] * len(symbols))
            obs, reward, terminated, truncated, info = env.step(hold_action)

            if terminated or truncated:
                break

        if terminated or truncated:
            break

        # Check if rebalancing is needed
        current_prices = env._get_current_prices()
        current_weights = {}
        total_positions_value = 0

        for symbol in symbols:
            position_value = info["positions"][symbol] * current_prices[symbol]
            total_positions_value += position_value
            current_weights[symbol] = position_value / info["portfolio_value"]

        # Rebalance if any weight deviates by more than 5%
        needs_rebalancing = any(
            abs(current_weights[symbol] - target_weights[symbol]) > 0.05
            for symbol in symbols
        )

        if needs_rebalancing:
            print(f"\nRebalancing at step {rebalance_step * 5 + 5}:")

            rebalance_action = []
            for symbol in symbols:
                current_weight = current_weights[symbol]
                target_weight = target_weights[symbol]

                if current_weight < target_weight - 0.02:  # Need to buy
                    rebalance_action.extend([1.0, 0.05])  # Small buy
                elif current_weight > target_weight + 0.02:  # Need to sell
                    rebalance_action.extend([2.0, 0.1])  # Small sell
                else:
                    rebalance_action.extend([0.0, 0.0])  # Hold

            obs, reward, terminated, truncated, info = env.step(
                np.array(rebalance_action)
            )

            # Show updated weights
            current_prices = env._get_current_prices()
            for symbol in symbols:
                position_value = info["positions"][symbol] * current_prices[symbol]
                weight = position_value / info["portfolio_value"]
                print(f"  {symbol}: {weight:.1%}")

    # Final portfolio summary
    final_metrics = env.get_portfolio_metrics()
    print(f"\n--- Final Portfolio Summary ---")
    print(f"Final Value: ${final_metrics['final_portfolio_value']:,.2f}")
    print(f"Total Return: {final_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {final_metrics['max_drawdown']:.2%}")


def main():
    """Run all demonstrations."""
    print("🚀 TradingEnvironment Comprehensive Demo\n")
    print("This demo showcases the key features of the TradingEnvironment")
    print("for reinforcement learning in algorithmic trading.\n")

    try:
        # Basic usage
        env = demonstrate_basic_usage()

        # Performance tracking
        portfolio_values, returns = demonstrate_performance_tracking()

        # Risk management
        demonstrate_risk_management()

        # Multi-asset portfolio
        demonstrate_multi_asset_portfolio()

        print("\n🎉 All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Realistic market simulation with OHLCV data")
        print("✓ Transaction costs and slippage modeling")
        print("✓ Risk-adjusted reward functions")
        print("✓ Portfolio state representation")
        print("✓ Dynamic position sizing")
        print("✓ Risk management and drawdown limits")
        print("✓ Multi-asset portfolio management")
        print("✓ Performance metrics calculation")
        print("✓ Gymnasium compatibility")

        print("\nThe TradingEnvironment is ready for RL training!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
