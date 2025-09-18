"""
Risk Management System Demo

This demo showcases the comprehensive risk management capabilities including:
- Real-time P&L monitoring and drawdown tracking
- Automated risk limit enforcement and position sizing
- Stress testing and scenario analysis capabilities
"""
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.portfolio import Portfolio, Position
from src.models.risk_management import (
    RiskLimit, RiskMetrics, StressTestScenario, PositionSizingRule,
    RiskLimitType, RiskLimitStatus
)
from src.services.risk_manager import RiskManager
from src.services.risk_monitoring_service import (
    RiskMonitoringService, log_alert_callback
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_portfolio() -> Portfolio:
    """Create a sample portfolio for demonstration."""
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            quantity=50.0,
            avg_cost=2000.0,
            current_price=2100.0,
            unrealized_pnl=5000.0,
            realized_pnl=1000.0
        ),
        "TSLA": Position(
            symbol="TSLA",
            quantity=75.0,
            avg_cost=200.0,
            current_price=190.0,
            unrealized_pnl=-750.0,
            realized_pnl=-300.0
        )
    }
    
    # Calculate total value: cash + positions market value
    positions_value = sum(pos.quantity * pos.current_price for pos in positions.values())
    cash_balance = 25000.0
    total_value = cash_balance + positions_value
    
    return Portfolio(
        user_id="demo_user",
        positions=positions,
        cash_balance=cash_balance,
        total_value=total_value,
        last_updated=datetime.now(timezone.utc)
    )


def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for demonstration."""
    # Generate 1 year of daily data
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='D'
    )
    
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price movements
    aapl_returns = np.random.normal(0.0008, 0.02, len(dates))
    googl_returns = np.random.normal(0.0006, 0.025, len(dates))
    tsla_returns = np.random.normal(0.001, 0.035, len(dates))
    
    # Convert to prices
    aapl_prices = 150 * np.exp(np.cumsum(aapl_returns))
    googl_prices = 2000 * np.exp(np.cumsum(googl_returns))
    tsla_prices = 200 * np.exp(np.cumsum(tsla_returns))
    
    return pd.DataFrame({
        'AAPL': aapl_prices,
        'GOOGL': googl_prices,
        'TSLA': tsla_prices
    }, index=dates)


def demonstrate_risk_metrics():
    """Demonstrate risk metrics calculation."""
    print("\n" + "="*60)
    print("RISK METRICS CALCULATION DEMO")
    print("="*60)
    
    # Create risk manager and sample data
    risk_manager = RiskManager(risk_free_rate=0.02)
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    print(f"Portfolio Overview:")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    print(f"  Cash Balance: ${portfolio.cash_balance:,.2f}")
    print(f"  Positions: {len(portfolio.positions)}")
    print(f"  Unrealized P&L: ${portfolio.unrealized_pnl:,.2f}")
    print(f"  Realized P&L: ${portfolio.realized_pnl:,.2f}")
    
    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_risk_metrics(portfolio, market_data)
    
    print(f"\nRisk Metrics:")
    print(f"  Portfolio Volatility: {risk_metrics.portfolio_volatility:.2%}")
    print(f"  Daily VaR (95%): ${risk_metrics.portfolio_var:,.2f}")
    print(f"  Current Drawdown: {risk_metrics.current_drawdown:.2%}")
    print(f"  Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"  Concentration Risk: {risk_metrics.concentration_risk:.3f}")
    print(f"  Leverage: {risk_metrics.leverage:.2f}x")
    print(f"  Daily P&L: ${risk_metrics.daily_pnl:,.2f}")


def demonstrate_risk_limits():
    """Demonstrate risk limit enforcement."""
    print("\n" + "="*60)
    print("RISK LIMIT ENFORCEMENT DEMO")
    print("="*60)
    
    risk_manager = RiskManager()
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    # Add various risk limits
    limits = [
        RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold=0.15,
            warning_threshold=0.10,
            enabled=True
        ),
        RiskLimit(
            limit_type=RiskLimitType.CONCENTRATION,
            threshold=0.40,
            warning_threshold=0.30,
            enabled=True
        ),
        RiskLimit(
            limit_type=RiskLimitType.PORTFOLIO_VAR,
            threshold=5000.0,
            warning_threshold=3000.0,
            enabled=True
        ),
        RiskLimit(
            limit_type=RiskLimitType.POSITION_SIZE,
            threshold=0.25,
            warning_threshold=0.20,
            symbol="AAPL",
            enabled=True
        )
    ]
    
    for limit in limits:
        risk_manager.add_risk_limit(limit)
        print(f"Added {limit.limit_type.value} limit: {limit.threshold}")
    
    # Calculate metrics and check limits
    risk_metrics = risk_manager.calculate_risk_metrics(portfolio, market_data)
    alerts = risk_manager.check_risk_limits(portfolio, risk_metrics)
    
    print(f"\nRisk Limit Check Results:")
    if alerts:
        for alert in alerts:
            status_color = "🔴" if alert.status == RiskLimitStatus.BREACH else "🟡"
            print(f"  {status_color} {alert.message}")
    else:
        print("  ✅ All risk limits within acceptable ranges")


def demonstrate_position_sizing():
    """Demonstrate position sizing enforcement."""
    print("\n" + "="*60)
    print("POSITION SIZING ENFORCEMENT DEMO")
    print("="*60)
    
    risk_manager = RiskManager()
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    # Add position sizing rules
    rule = PositionSizingRule(
        rule_name="conservative_growth",
        max_position_size=0.15,  # Max 15% per position
        volatility_target=0.02,  # Target 2% volatility contribution
        correlation_penalty=0.5,  # 50% penalty for correlated positions
        kelly_fraction=0.25,     # Use 25% of Kelly criterion
        enabled=True
    )
    
    risk_manager.add_position_sizing_rule(rule)
    print(f"Added position sizing rule: {rule.rule_name}")
    print(f"  Max Position Size: {rule.max_position_size:.1%}")
    print(f"  Volatility Target: {rule.volatility_target:.1%}")
    
    # Test position sizing for new positions
    test_positions = [
        ("MSFT", 0.20, 0.12, 0.18),  # symbol, intended_size, expected_return, volatility
        ("NVDA", 0.25, 0.25, 0.35),
        ("SPY", 0.30, 0.08, 0.15)
    ]
    
    print(f"\nPosition Sizing Results:")
    for symbol, intended_size, expected_return, volatility in test_positions:
        adjusted_size = risk_manager.enforce_position_sizing(
            portfolio=portfolio,
            symbol=symbol,
            intended_size=intended_size,
            expected_return=expected_return,
            volatility=volatility,
            market_data=market_data
        )
        
        adjustment = (adjusted_size / intended_size - 1) * 100
        print(f"  {symbol}:")
        print(f"    Intended: {intended_size:.1%} → Adjusted: {adjusted_size:.1%} ({adjustment:+.1f}%)")
        print(f"    Expected Return: {expected_return:.1%}, Volatility: {volatility:.1%}")


def demonstrate_stress_testing():
    """Demonstrate stress testing capabilities."""
    print("\n" + "="*60)
    print("STRESS TESTING DEMO")
    print("="*60)
    
    risk_manager = RiskManager()
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    print(f"Portfolio Value Before Stress: ${portfolio.total_value:,.2f}")
    
    # Get predefined scenarios
    scenarios = risk_manager.get_predefined_stress_scenarios()
    
    print(f"\nRunning {len(scenarios)} stress test scenarios:")
    
    for scenario in scenarios:
        try:
            result = risk_manager.run_stress_test(portfolio, scenario, market_data)
            
            loss_pct = result.loss_percentage * 100
            status = "🔴 SEVERE" if loss_pct > 20 else "🟡 MODERATE" if loss_pct > 10 else "🟢 MILD"
            
            print(f"\n  {scenario.scenario_name}:")
            print(f"    {status} Loss: ${result.total_loss:,.0f} ({loss_pct:.1f}%)")
            print(f"    Portfolio Value After: ${result.portfolio_value_after:,.2f}")
            
            # Show top position impacts
            sorted_impacts = sorted(
                result.position_impacts.items(),
                key=lambda x: x[1]
            )
            print(f"    Worst Affected Positions:")
            for symbol, impact in sorted_impacts[:3]:
                print(f"      {symbol}: ${impact:,.0f}")
                
        except Exception as e:
            print(f"    ❌ {scenario.scenario_name}: Failed ({e})")


async def demonstrate_real_time_monitoring():
    """Demonstrate real-time risk monitoring."""
    print("\n" + "="*60)
    print("REAL-TIME RISK MONITORING DEMO")
    print("="*60)
    
    risk_manager = RiskManager()
    
    # Add some risk limits
    risk_manager.add_risk_limit(RiskLimit(
        limit_type=RiskLimitType.MAX_DRAWDOWN,
        threshold=0.10,
        warning_threshold=0.05,
        enabled=True
    ))
    
    # Create monitoring service with alert callback
    monitoring_service = RiskMonitoringService(
        risk_manager=risk_manager,
        monitoring_interval=2,  # Check every 2 seconds for demo
        alert_callbacks=[log_alert_callback]
    )
    
    # Portfolio and market data providers
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    def portfolio_provider():
        # Simulate portfolio value declining to trigger alerts
        portfolio.total_value *= 0.98  # 2% decline each check
        portfolio.cash_balance = portfolio.total_value - sum(
            pos.market_value for pos in portfolio.positions.values()
        )
        return portfolio
    
    def market_data_provider():
        return market_data
    
    print("Starting real-time monitoring (will run for 10 seconds)...")
    print("Simulating portfolio decline to demonstrate alert system...")
    
    # Start monitoring
    await monitoring_service.start_monitoring(
        portfolio_provider, market_data_provider
    )
    
    # Let it run for a bit
    await asyncio.sleep(10)
    
    # Stop monitoring
    await monitoring_service.stop_monitoring()
    
    # Show monitoring results
    status = monitoring_service.get_current_risk_status()
    print(f"\nMonitoring Results:")
    print(f"  Total Checks: {status['monitoring_stats']['total_checks']}")
    print(f"  Alerts Generated: {status['monitoring_stats']['alerts_generated']}")
    print(f"  Active Alerts: {status['active_alerts_count']}")


def demonstrate_risk_report():
    """Demonstrate comprehensive risk report generation."""
    print("\n" + "="*60)
    print("COMPREHENSIVE RISK REPORT DEMO")
    print("="*60)
    
    risk_manager = RiskManager()
    monitoring_service = RiskMonitoringService(risk_manager)
    
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    # Generate comprehensive report
    report = monitoring_service.generate_risk_report(
        portfolio, market_data, include_stress_tests=True
    )
    
    print("Risk Report Generated:")
    print(f"  Report Timestamp: {report['report_timestamp']}")
    
    # Portfolio summary
    summary = report['portfolio_summary']
    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${summary['total_value']:,.2f}")
    print(f"  Positions: {summary['positions_count']}")
    print(f"  Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    
    # Risk metrics
    metrics = report['current_risk_metrics']
    print(f"\nCurrent Risk Metrics:")
    print(f"  Portfolio Volatility: {metrics['portfolio_volatility']:.2%}")
    print(f"  Daily VaR: ${metrics['portfolio_var']:,.2f}")
    print(f"  Concentration Risk: {metrics['concentration_risk']:.3f}")
    
    # Stress test results
    if 'stress_test_results' in report:
        print(f"\nStress Test Summary:")
        for result in report['stress_test_results']:
            loss_pct = result['loss_percentage'] * 100
            print(f"  {result['scenario_name']}: {loss_pct:.1f}% loss")


async def main():
    """Run all risk management demonstrations."""
    print("🚀 AI Trading Platform - Risk Management System Demo")
    print("This demo showcases comprehensive risk management capabilities")
    
    try:
        # Run demonstrations
        demonstrate_risk_metrics()
        demonstrate_risk_limits()
        demonstrate_position_sizing()
        demonstrate_stress_testing()
        await demonstrate_real_time_monitoring()
        demonstrate_risk_report()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Real-time P&L monitoring and drawdown tracking")
        print("  ✓ Automated risk limit enforcement")
        print("  ✓ Intelligent position sizing with correlation analysis")
        print("  ✓ Comprehensive stress testing scenarios")
        print("  ✓ Real-time monitoring with alert system")
        print("  ✓ Detailed risk reporting and analytics")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())