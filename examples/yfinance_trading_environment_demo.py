"""
YFinance Trading Environment Demo

This demo showcases the YFinance trading environment with comprehensive
reward functions and risk metrics for RL training.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml import (
    YFinanceTradingEnvironment, YFinanceConfig, MarketRegime,
    RewardType, RewardConfig, RewardFunctionFactory,
    create_yfinance_environment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_environment():
    """Demonstrate basic environment functionality."""
    print("=" * 60)
    print("YFINANCE TRADING ENVIRONMENT DEMO")
    print("=" * 60)
    
    # Create environment with default configuration
    env = create_yfinance_environment(
        symbols=["SPY", "QQQ", "AAPL", "MSFT"],
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    print(f"Environment created with {len(env.symbols)} symbols")
    print(f"Data range: {env.start_date} to {env.end_date}")
    print(f"Max steps: {env.max_steps}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"Initial market regime: {info['market_regime']}")
    
    return env


def demo_trading_simulation(env, num_steps=100):
    """Demonstrate trading simulation with different strategies."""
    print(f"\n{'='*60}")
    print("TRADING SIMULATION DEMO")
    print(f"{'='*60}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    portfolio_history = []
    regime_history = []
    action_history = []
    reward_history = []
    
    print(f"Starting simulation for {num_steps} steps...")
    
    for step in range(num_steps):
        # Simple trading strategy: buy in bull markets, sell in bear markets
        current_regime = MarketRegime[info['market_regime']]
        
        # Generate actions based on regime
        if current_regime == MarketRegime.BULL_MARKET:
            # Buy strategy: moderate position sizes
            actions = []
            for i in range(len(env.symbols)):
                action_type = 1  # BUY
                position_size = 0.1  # 10% position
                actions.extend([action_type, position_size])
        
        elif current_regime == MarketRegime.BEAR_MARKET:
            # Sell strategy: reduce positions
            actions = []
            for i in range(len(env.symbols)):
                action_type = 2  # SELL
                position_size = 0.5  # Sell 50% of position
                actions.extend([action_type, position_size])
        
        elif current_regime == MarketRegime.HIGH_VOLATILITY:
            # Conservative strategy: hold positions
            actions = []
            for i in range(len(env.symbols)):
                action_type = 0  # HOLD
                position_size = 0.0
                actions.extend([action_type, position_size])
        
        else:  # SIDEWAYS_MARKET
            # Balanced strategy: small positions
            actions = []
            for i in range(len(env.symbols)):
                action_type = np.random.choice([0, 1, 2])  # Random action
                position_size = 0.05  # Small position
                actions.extend([action_type, position_size])
        
        action = np.array(actions, dtype=np.float32)
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record data
        portfolio_history.append(info['portfolio_value'])
        regime_history.append(info['market_regime'])
        action_history.append(action.copy())
        reward_history.append(reward)
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d}: Portfolio=${info['portfolio_value']:8,.0f}, "
                  f"Return={info['total_return']:6.2%}, Regime={info['market_regime']:15s}, "
                  f"Reward={reward:8.4f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Final results
    final_metrics = env.get_portfolio_metrics()
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final Portfolio Value: ${info['portfolio_value']:,.2f}")
    print(f"Total Return: {info['total_return']:.2%}")
    print(f"Total Trades: {info['num_trades']}")
    print(f"Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {final_metrics.get('max_drawdown', 0):.2%}")
    print(f"Volatility: {final_metrics.get('volatility', 0):.2%}")
    
    return {
        'portfolio_history': portfolio_history,
        'regime_history': regime_history,
        'action_history': action_history,
        'reward_history': reward_history,
        'final_metrics': final_metrics
    }


def demo_reward_functions(env):
    """Demonstrate different reward functions."""
    print(f"\n{'='*60}")
    print("REWARD FUNCTIONS DEMO")
    print(f"{'='*60}")
    
    # Create different reward functions
    reward_config = RewardConfig(
        return_weight=0.4,
        sharpe_weight=0.3,
        drawdown_penalty=2.0,
        transaction_cost_penalty=1.0
    )
    
    reward_functions = {
        'Simple Return': RewardFunctionFactory.create_reward_function(
            RewardType.SIMPLE_RETURN, reward_config
        ),
        'Sharpe Ratio': RewardFunctionFactory.create_reward_function(
            RewardType.SHARPE_RATIO, reward_config
        ),
        'Multi-Objective': RewardFunctionFactory.create_reward_function(
            RewardType.MULTI_OBJECTIVE, reward_config
        ),
        'Regime Adaptive': RewardFunctionFactory.create_reward_function(
            RewardType.REGIME_ADAPTIVE, reward_config
        )
    }
    
    # Test with sample data
    portfolio_value = 105000.0
    previous_value = 100000.0
    trade_results = [{'executed': True, 'cost': 50.0, 'slippage': 0.001}]
    market_regime = MarketRegime.BULL_MARKET
    portfolio_state = {
        'drawdown': 0.02,
        'exposure': 0.7,
        'positions': {'AAPL': 100, 'MSFT': 50},
        'current_prices': {'AAPL': 150, 'MSFT': 300},
        'portfolio_value': portfolio_value
    }
    
    print("Testing reward functions with sample data:")
    print(f"Portfolio Value: ${portfolio_value:,.2f} (was ${previous_value:,.2f})")
    print(f"Market Regime: {market_regime.name}")
    print(f"Trade Cost: ${trade_results[0]['cost']:.2f}")
    print(f"Current Drawdown: {portfolio_state['drawdown']:.1%}")
    print(f"Portfolio Exposure: {portfolio_state['exposure']:.1%}")
    print()
    
    for name, reward_func in reward_functions.items():
        reward = reward_func.calculate_reward(
            portfolio_value, previous_value, trade_results,
            market_regime, portfolio_state
        )
        print(f"{name:20s}: {reward:8.4f}")
    
    # Test regime adaptation
    print(f"\n{'Regime Sensitivity Analysis':=^60}")
    adaptive_reward = reward_functions['Regime Adaptive']
    
    for regime in MarketRegime:
        reward = adaptive_reward.calculate_reward(
            portfolio_value, previous_value, trade_results,
            regime, portfolio_state
        )
        print(f"{regime.name:20s}: {reward:8.4f}")


def demo_risk_metrics():
    """Demonstrate risk metrics calculations."""
    print(f"\n{'='*60}")
    print("RISK METRICS DEMO")
    print(f"{'='*60}")
    
    from src.ml.reward_functions import RiskMetricsCalculator, PortfolioConstraints
    
    # Create risk calculator
    config = RewardConfig()
    risk_calc = RiskMetricsCalculator(config)
    constraints = PortfolioConstraints(config)
    
    # Generate sample returns data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    portfolio_values = np.cumprod(1 + returns) * 100000  # Portfolio value series
    
    print("Risk Metrics for Sample Portfolio:")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    
    # Calculate various risk metrics
    var_5 = risk_calc.calculate_var(returns, 0.05)
    cvar_5 = risk_calc.calculate_cvar(returns, 0.05)
    max_dd = risk_calc.calculate_maximum_drawdown(portfolio_values)
    ulcer = risk_calc.calculate_ulcer_index(portfolio_values)
    downside_dev = risk_calc.calculate_downside_deviation(returns)
    
    print(f"{'VaR (5%)':<25} {var_5:<15.4f}")
    print(f"{'CVaR (5%)':<25} {cvar_5:<15.4f}")
    print(f"{'Max Drawdown':<25} {max_dd:<15.2%}")
    print(f"{'Ulcer Index':<25} {ulcer:<15.4f}")
    print(f"{'Downside Deviation':<25} {downside_dev:<15.4f}")
    
    # Test portfolio constraints
    print(f"\n{'Portfolio Constraints Check':=^60}")
    
    positions = {'AAPL': 500, 'MSFT': 200, 'GOOGL': 50}
    current_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500}
    portfolio_value = 400000
    
    # Check various constraints
    constraints_to_check = [
        ('Position Size (AAPL)', lambda: constraints.check_position_size_constraint(
            'AAPL', 500, 150, portfolio_value, 0.2
        )),
        ('Total Exposure', lambda: constraints.check_total_exposure_constraint(
            positions, current_prices, portfolio_value, 0.8
        )),
        ('Concentration', lambda: constraints.check_concentration_constraint(
            positions, current_prices, 0.4
        )),
        ('Drawdown', lambda: constraints.check_drawdown_constraint(
            portfolio_value, 450000, 0.15
        ))
    ]
    
    for constraint_name, check_func in constraints_to_check:
        valid, message = check_func()
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{constraint_name:<25} {status:<10} {message}")


def plot_results(results):
    """Plot simulation results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        ax1.plot(results['portfolio_history'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Cumulative rewards
        cumulative_rewards = np.cumsum(results['reward_history'])
        ax2.plot(cumulative_rewards)
        ax2.set_title('Cumulative Rewards')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Cumulative Reward')
        ax2.grid(True)
        
        # Market regime distribution
        regime_counts = {}
        for regime in results['regime_history']:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        ax3.bar(regime_counts.keys(), regime_counts.values())
        ax3.set_title('Market Regime Distribution')
        ax3.set_xlabel('Market Regime')
        ax3.set_ylabel('Frequency')
        ax3.tick_params(axis='x', rotation=45)
        
        # Reward distribution
        ax4.hist(results['reward_history'], bins=30, alpha=0.7)
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('yfinance_trading_demo_results.png', dpi=300, bbox_inches='tight')
        print(f"\nResults plotted and saved to 'yfinance_trading_demo_results.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")


def main():
    """Run the complete demo."""
    print("YFinance Trading Environment and Reward Functions Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Basic environment
        env = demo_basic_environment()
        
        # Demo 2: Trading simulation
        results = demo_trading_simulation(env, num_steps=50)
        
        # Demo 3: Reward functions
        demo_reward_functions(env)
        
        # Demo 4: Risk metrics
        demo_risk_metrics()
        
        # Plot results
        plot_results(results)
        
        print(f"\n{'='*60}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        # Summary
        print("\nKey Features Demonstrated:")
        print("✓ YFinance data integration with real market data")
        print("✓ Market regime detection and adaptation")
        print("✓ Realistic transaction costs and slippage modeling")
        print("✓ Multi-objective reward functions")
        print("✓ Comprehensive risk metrics (VaR, CVaR, drawdown)")
        print("✓ Portfolio constraints and risk management")
        print("✓ Regime-adaptive reward shaping")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()