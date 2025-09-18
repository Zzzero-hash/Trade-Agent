"""
Portfolio Management System Demo

This example demonstrates the comprehensive portfolio management system
including Modern Portfolio Theory optimization, dynamic rebalancing,
and risk management.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.portfolio import Portfolio, Position
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.portfolio_optimizer import OptimizationMethod, OptimizationConstraints
from src.services.portfolio_rebalancer import RebalancingConfig


def create_sample_portfolio():
    """Create a sample portfolio for demonstration."""
    positions = {
        'AAPL': Position(
            symbol='AAPL',
            quantity=100.0,
            avg_cost=150.0,
            current_price=160.0,
            unrealized_pnl=1000.0
        ),
        'GOOGL': Position(
            symbol='GOOGL',
            quantity=50.0,
            avg_cost=2000.0,
            current_price=2100.0,
            unrealized_pnl=5000.0
        ),
        'MSFT': Position(
            symbol='MSFT',
            quantity=75.0,
            avg_cost=280.0,
            current_price=300.0,
            unrealized_pnl=1500.0
        ),
        'TSLA': Position(
            symbol='TSLA',
            quantity=25.0,
            avg_cost=800.0,
            current_price=850.0,
            unrealized_pnl=1250.0
        )
    }
    
    portfolio = Portfolio(
        user_id='demo_user',
        positions=positions,
        cash_balance=20000.0,
        total_value=184750.0,  # 16000 + 105000 + 22500 + 21250 + 20000
        last_updated=datetime.now(timezone.utc)
    )
    
    return portfolio


def create_sample_market_data():
    """Create sample market data for demonstration."""
    # Generate 1 year of daily data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    # Generate synthetic price data with different characteristics
    np.random.seed(42)
    n_days = len(dates)
    
    data = {}
    base_prices = {'AAPL': 150, 'GOOGL': 2000, 'MSFT': 280, 'TSLA': 800}
    volatilities = {'AAPL': 0.02, 'GOOGL': 0.025, 'MSFT': 0.018, 'TSLA': 0.035}
    drifts = {'AAPL': 0.0003, 'GOOGL': 0.0004, 'MSFT': 0.0002, 'TSLA': 0.0005}
    
    for symbol in symbols:
        # Generate random walk with drift
        returns = np.random.normal(drifts[symbol], volatilities[symbol], n_days)
        prices = base_prices[symbol] * np.exp(np.cumsum(returns))
        data[symbol] = prices
    
    market_data = pd.DataFrame(data, index=dates)
    return market_data


def demonstrate_portfolio_optimization():
    """Demonstrate different portfolio optimization methods."""
    print("=== Portfolio Optimization Demo ===")
    
    # Create sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    expected_returns = pd.Series([0.12, 0.15, 0.10, 0.18], index=symbols)
    
    # Create covariance matrix
    cov_data = np.array([
        [0.04, 0.02, 0.015, 0.025],
        [0.02, 0.06, 0.018, 0.030],
        [0.015, 0.018, 0.03, 0.020],
        [0.025, 0.030, 0.020, 0.08]
    ])
    covariance_matrix = pd.DataFrame(cov_data, index=symbols, columns=symbols)
    
    # Initialize portfolio management service
    service = PortfolioManagementService(risk_free_rate=0.02)
    
    # Test different optimization methods
    methods = [
        OptimizationMethod.MAXIMUM_SHARPE,
        OptimizationMethod.MINIMUM_VARIANCE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.FACTOR_BASED
    ]
    
    for method in methods:
        print(f"\n--- {method.value.replace('_', ' ').title()} Optimization ---")
        
        result = service.optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=method
        )
        
        if result.success:
            print(f"Expected Return: {result.expected_return:.4f}")
            print(f"Volatility: {result.expected_volatility:.4f}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
            print("Optimal Weights:")
            for symbol, weight in result.weights.items():
                print(f"  {symbol}: {weight:.4f} ({weight*100:.1f}%)")
        else:
            print(f"Optimization failed: {result.message}")


def demonstrate_rebalancing():
    """Demonstrate portfolio rebalancing functionality."""
    print("\n=== Portfolio Rebalancing Demo ===")
    
    # Create portfolio and market data
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    # Initialize service
    service = PortfolioManagementService(risk_free_rate=0.02)
    
    # Get current prices
    current_prices = service._get_current_prices(market_data)
    
    # Calculate current weights
    current_weights = service.rebalancer._calculate_current_weights(portfolio, current_prices)
    print("Current Portfolio Weights:")
    for symbol, weight in current_weights.items():
        print(f"  {symbol}: {weight:.4f} ({weight*100:.1f}%)")
    
    # Define target weights (equal weight portfolio)
    target_weights = {symbol: 0.25 for symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']}
    print("\nTarget Portfolio Weights:")
    for symbol, weight in target_weights.items():
        print(f"  {symbol}: {weight:.4f} ({weight*100:.1f}%)")
    
    # Check if rebalancing is needed
    config = RebalancingConfig(
        threshold_method="percentage",
        rebalance_threshold=0.05,  # 5% deviation threshold
        transaction_cost_threshold=0.002
    )
    
    recommendation = service.rebalancer.should_rebalance(
        portfolio=portfolio,
        target_weights=target_weights,
        current_prices=current_prices,
        config=config
    )
    
    print(f"\nRebalancing Recommendation:")
    print(f"  Should Rebalance: {recommendation.should_rebalance}")
    print(f"  Reason: {recommendation.reason}")
    print(f"  Urgency Score: {recommendation.urgency_score:.3f}")
    print(f"  Expected Transaction Costs: ${recommendation.expected_transaction_costs:.2f}")
    print(f"  Expected Benefit: ${recommendation.expected_benefit:.2f}")
    
    if recommendation.should_rebalance:
        print("\nRequired Trades:")
        for symbol, quantity in recommendation.trades_required.items():
            action = "BUY" if quantity > 0 else "SELL"
            print(f"  {action} {abs(quantity):.2f} shares of {symbol}")


def demonstrate_risk_assessment():
    """Demonstrate portfolio risk assessment."""
    print("\n=== Portfolio Risk Assessment Demo ===")
    
    # Create portfolio and market data
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    # Initialize service
    service = PortfolioManagementService(risk_free_rate=0.02)
    
    # Calculate risk metrics
    returns = service._calculate_returns(market_data, lookback_days=252)
    covariance_matrix = service._calculate_covariance_matrix(returns)
    current_prices = service._get_current_prices(market_data)
    
    risk_metrics = service.assess_portfolio_risk(
        portfolio=portfolio,
        current_prices=current_prices,
        covariance_matrix=covariance_matrix,
        confidence_level=0.05
    )
    
    print("Portfolio Risk Metrics:")
    print(f"  Portfolio Volatility: {risk_metrics['portfolio_volatility']:.4f} ({risk_metrics['portfolio_volatility']*100:.2f}%)")
    print(f"  Daily VaR (95%): ${risk_metrics['daily_var']:.2f}")
    print(f"  Expected Shortfall: ${risk_metrics['expected_shortfall']:.2f}")
    print(f"  Concentration Risk: {risk_metrics['concentration_risk']:.4f}")
    
    # Calculate position sizing for a new asset
    print("\nPosition Sizing for New Asset (NVDA):")
    position_size = service.calculate_position_sizing(
        portfolio=portfolio,
        symbol='NVDA',
        expected_return=0.20,
        volatility=0.35,
        correlation_with_portfolio=0.6,
        risk_budget=0.02
    )
    print(f"  Recommended Position Size: {position_size:.4f} ({position_size*100:.2f}% of portfolio)")
    print(f"  Dollar Amount: ${position_size * portfolio.total_value:.2f}")


def demonstrate_complete_workflow():
    """Demonstrate complete portfolio management workflow."""
    print("\n=== Complete Portfolio Management Workflow ===")
    
    # Create portfolio and market data
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    # Initialize service
    service = PortfolioManagementService(risk_free_rate=0.02)
    
    # Set optimization constraints
    constraints = OptimizationConstraints(
        min_weight=0.05,  # Minimum 5% allocation
        max_weight=0.4,   # Maximum 40% allocation
        transaction_cost=0.001
    )
    
    # Set rebalancing configuration
    rebalancing_config = RebalancingConfig(
        threshold_method="percentage",
        rebalance_threshold=0.03,  # 3% threshold
        transaction_cost_threshold=0.002
    )
    
    # Run complete portfolio management
    result = service.manage_portfolio(
        portfolio=portfolio,
        market_data=market_data,
        optimization_method=OptimizationMethod.MAXIMUM_SHARPE,
        lookback_days=100,
        optimization_constraints=constraints,
        rebalancing_config=rebalancing_config
    )
    
    if 'error' not in result:
        print("Portfolio Management Results:")
        
        # Optimization results
        opt_result = result['optimization_result']
        if opt_result.success:
            print(f"\nOptimization ({opt_result.optimization_method.value}):")
            print(f"  Expected Return: {opt_result.expected_return:.4f}")
            print(f"  Volatility: {opt_result.expected_volatility:.4f}")
            print(f"  Sharpe Ratio: {opt_result.sharpe_ratio:.4f}")
            print(f"  Transaction Costs: ${opt_result.transaction_costs:.2f}")
        
        # Rebalancing recommendation
        rebalance_rec = result['rebalance_recommendation']
        print(f"\nRebalancing:")
        print(f"  Should Rebalance: {rebalance_rec.should_rebalance}")
        print(f"  Urgency Score: {rebalance_rec.urgency_score:.3f}")
        
        # Portfolio metrics
        metrics = result['portfolio_metrics']
        if metrics:
            print(f"\nPortfolio Metrics:")
            print(f"  Total Value: ${metrics['total_value']:,.2f}")
            print(f"  Expected Return: {metrics['expected_return']:.4f}")
            print(f"  Volatility: {metrics['volatility']:.4f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Effective Positions: {metrics['effective_positions']:.2f}")
    else:
        print(f"Portfolio management failed: {result['error']}")


def main():
    """Run all portfolio management demonstrations."""
    print("Portfolio Management System Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_portfolio_optimization()
        demonstrate_rebalancing()
        demonstrate_risk_assessment()
        demonstrate_complete_workflow()
        
        print("\n" + "=" * 50)
        print("Portfolio Management Demo Completed Successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()