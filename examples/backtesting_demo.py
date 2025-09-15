"""
Backtesting Framework Demo

This demo shows how to use the comprehensive backtesting framework
with walk-forward analysis, performance attribution, and stress testing.

Requirements: 2.5, 5.7, 9.6
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import backtesting components
from src.models.backtesting import (
    BacktestConfig, BacktestResult, StressTestScenario, BacktestPeriodType
)
from src.services.backtesting_engine import BacktestingEngine
from src.services.data_aggregator import DataAggregator
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService


class BacktestingDemo:
    """Demonstration of backtesting framework capabilities."""
    
    def __init__(self):
        self.setup_services()
    
    def setup_services(self):
        """Initialize required services for backtesting."""
        
        # In a real implementation, these would be properly configured
        # For demo purposes, we'll create mock services
        
        logger.info("Setting up backtesting services...")
        
        # Create service instances (would be dependency injected in real app)
        self.data_aggregator = DataAggregator()
        self.decision_engine = TradingDecisionEngine()
        self.portfolio_service = PortfolioManagementService()
        
        # Create backtesting engine
        self.backtesting_engine = BacktestingEngine(
            data_aggregator=self.data_aggregator,
            decision_engine=self.decision_engine,
            portfolio_service=self.portfolio_service
        )
        
        logger.info("Backtesting services initialized")
    
    def create_sample_config(self) -> BacktestConfig:
        """Create a sample backtesting configuration."""
        
        config = BacktestConfig(
            # Time period
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            
            # Assets to test
            symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            
            # Walk-forward analysis parameters
            training_period_days=252,  # 1 year training
            testing_period_days=63,    # 3 months testing
            period_type=BacktestPeriodType.ROLLING,
            overlap_days=21,           # 1 month overlap
            
            # Portfolio parameters
            initial_balance=100000.0,
            max_position_size=0.25,    # Max 25% per position
            
            # Trading costs
            transaction_cost=0.001,    # 0.1% transaction cost
            slippage=0.0005,          # 0.05% slippage
            
            # Risk management
            max_drawdown_limit=0.15,   # 15% max drawdown
            stop_loss_threshold=0.08,  # 8% stop loss
            
            # Rebalancing
            rebalance_frequency='weekly'
        )
        
        logger.info(f"Created backtest config: {len(config.symbols)} symbols, "
                   f"{config.training_period_days}d training, {config.testing_period_days}d testing")
        
        return config
    
    def create_stress_scenarios(self) -> List[StressTestScenario]:
        """Create stress testing scenarios."""
        
        scenarios = [
            # Market crash scenario (2008-style)
            StressTestScenario(
                name="Market Crash 2008",
                description="Simulates 2008 financial crisis conditions",
                market_shock_magnitude=-0.35,  # 35% market drop
                shock_duration_days=180,       # 6 months
                recovery_duration_days=365,    # 1 year recovery
                volatility_multiplier=2.5,     # 2.5x normal volatility
                correlation_increase=0.3       # Increased correlations
            ),
            
            # COVID-19 style crash and recovery
            StressTestScenario(
                name="Pandemic Shock",
                description="Simulates COVID-19 style market shock",
                market_shock_magnitude=-0.25,  # 25% drop
                shock_duration_days=45,        # 1.5 months
                recovery_duration_days=180,    # 6 months recovery
                volatility_multiplier=3.0,     # 3x volatility
                correlation_increase=0.4       # High correlations
            ),
            
            # Dot-com bubble burst
            StressTestScenario(
                name="Tech Bubble Burst",
                description="Simulates dot-com bubble burst",
                market_shock_magnitude=-0.45,  # 45% drop for tech
                shock_duration_days=365,       # 1 year decline
                recovery_duration_days=730,    # 2 years recovery
                volatility_multiplier=2.0,     # 2x volatility
                correlation_increase=0.2       # Moderate correlation increase
            ),
            
            # High volatility regime
            StressTestScenario(
                name="High Volatility Regime",
                description="Extended period of high volatility",
                market_shock_magnitude=0.0,    # No systematic shock
                shock_duration_days=252,       # 1 year
                recovery_duration_days=0,      # No recovery needed
                volatility_multiplier=2.5,     # 2.5x volatility
                correlation_increase=0.1       # Slight correlation increase
            )
        ]
        
        logger.info(f"Created {len(scenarios)} stress test scenarios")
        return scenarios
    
    async def run_basic_backtest(self) -> BacktestResult:
        """Run a basic backtesting example."""
        
        logger.info("=" * 60)
        logger.info("RUNNING BASIC BACKTEST")
        logger.info("=" * 60)
        
        # Create configuration
        config = self.create_sample_config()
        
        # Progress callback
        def progress_callback(progress: float, message: str):
            logger.info(f"Progress: {progress:.1%} - {message}")
        
        # Run backtest
        logger.info("Starting backtesting execution...")
        result = await self.backtesting_engine.run_backtest(
            config=config,
            progress_callback=progress_callback
        )
        
        # Display results
        self.display_backtest_results(result)
        
        return result
    
    async def run_walk_forward_analysis(self) -> BacktestResult:
        """Run walk-forward analysis with different period types."""
        
        logger.info("=" * 60)
        logger.info("RUNNING WALK-FORWARD ANALYSIS")
        logger.info("=" * 60)
        
        # Test different period types
        period_types = [
            BacktestPeriodType.ROLLING,
            BacktestPeriodType.EXPANDING,
            BacktestPeriodType.FIXED
        ]
        
        results = {}
        
        for period_type in period_types:
            logger.info(f"\nTesting {period_type.value} walk-forward analysis...")
            
            config = self.create_sample_config()
            config.period_type = period_type
            
            result = await self.backtesting_engine.run_backtest(config)
            results[period_type.value] = result
            
            logger.info(f"{period_type.value} Results:")
            logger.info(f"  Total Periods: {result.total_periods}")
            logger.info(f"  Overall Return: {result.overall_metrics.total_return:.2f}%")
            logger.info(f"  Sharpe Ratio: {result.overall_metrics.sharpe_ratio:.3f}")
            logger.info(f"  Max Drawdown: {result.overall_metrics.max_drawdown:.2f}%")
            logger.info(f"  Performance Consistency: {result.performance_consistency:.3f}")
        
        # Compare results
        self.compare_walk_forward_results(results)
        
        return results[BacktestPeriodType.ROLLING.value]  # Return rolling result
    
    async def run_stress_testing(self) -> List[Any]:
        """Run comprehensive stress testing."""
        
        logger.info("=" * 60)
        logger.info("RUNNING STRESS TESTING")
        logger.info("=" * 60)
        
        # Create base configuration
        config = self.create_sample_config()
        
        # Create stress scenarios
        scenarios = self.create_stress_scenarios()
        
        # Run stress tests
        logger.info("Starting stress testing...")
        stress_results = await self.backtesting_engine.run_stress_test(
            config=config,
            scenarios=scenarios
        )
        
        # Display stress test results
        self.display_stress_test_results(stress_results)
        
        return stress_results
    
    def display_backtest_results(self, result: BacktestResult):
        """Display comprehensive backtesting results."""
        
        logger.info("\n" + "=" * 50)
        logger.info("BACKTESTING RESULTS")
        logger.info("=" * 50)
        
        # Execution summary
        logger.info(f"Execution Time: {result.execution_duration:.2f} seconds")
        logger.info(f"Total Periods: {result.total_periods}")
        logger.info(f"Date Range: {result.config.start_date.date()} to {result.config.end_date.date()}")
        logger.info(f"Symbols: {', '.join(result.config.symbols)}")
        
        # Overall performance
        metrics = result.overall_metrics
        logger.info(f"\nOVERALL PERFORMANCE:")
        logger.info(f"  Total Return: {metrics.total_return:.2f}%")
        logger.info(f"  Annualized Return: {metrics.annualized_return:.2f}%")
        logger.info(f"  Volatility: {metrics.volatility:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        logger.info(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
        logger.info(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
        
        # Risk metrics
        logger.info(f"\nRISK METRICS:")
        logger.info(f"  Maximum Drawdown: {metrics.max_drawdown:.2f}%")
        logger.info(f"  Max DD Duration: {metrics.max_drawdown_duration} days")
        logger.info(f"  VaR (95%): {result.var_95:.2f}%")
        logger.info(f"  CVaR (95%): {result.cvar_95:.2f}%")
        
        # Trading metrics
        logger.info(f"\nTRADING METRICS:")
        logger.info(f"  Total Trades: {metrics.total_trades}")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"  Avg Trade Return: {metrics.avg_trade_return:.3f}%")
        
        # Statistical significance
        if metrics.t_statistic is not None and metrics.p_value is not None:
            logger.info(f"\nSTATISTICAL SIGNIFICANCE:")
            logger.info(f"  T-Statistic: {metrics.t_statistic:.3f}")
            logger.info(f"  P-Value: {metrics.p_value:.4f}")
            significance = "Significant" if metrics.p_value < 0.05 else "Not Significant"
            logger.info(f"  Result: {significance} (α = 0.05)")
        
        # Stability metrics
        if result.stability_metrics:
            logger.info(f"\nSTABILITY METRICS:")
            for metric, value in result.stability_metrics.items():
                logger.info(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Performance consistency
        logger.info(f"\nPERFORMANCE CONSISTENCY:")
        logger.info(f"  Consistency Score: {result.performance_consistency:.3f}")
        logger.info(f"  Average Period Return: {result.avg_period_performance:.2f}%")
        
        # Period-by-period summary
        logger.info(f"\nPERIOD SUMMARY:")
        for i, period in enumerate(result.period_results[:5]):  # Show first 5 periods
            logger.info(f"  Period {period.period_id + 1}: "
                       f"{period.performance_metrics.total_return:.2f}% return, "
                       f"{period.performance_metrics.max_drawdown:.2f}% max DD")
        
        if len(result.period_results) > 5:
            logger.info(f"  ... and {len(result.period_results) - 5} more periods")
    
    def display_stress_test_results(self, stress_results: List[Any]):
        """Display stress testing results."""
        
        logger.info("\n" + "=" * 50)
        logger.info("STRESS TESTING RESULTS")
        logger.info("=" * 50)
        
        for result in stress_results:
            logger.info(f"\nScenario: {result.scenario.name}")
            logger.info(f"Description: {result.scenario.description}")
            
            # Scenario parameters
            logger.info(f"Parameters:")
            logger.info(f"  Market Shock: {result.scenario.market_shock_magnitude:.1%}")
            logger.info(f"  Shock Duration: {result.scenario.shock_duration_days} days")
            logger.info(f"  Volatility Multiplier: {result.scenario.volatility_multiplier:.1f}x")
            
            # Performance comparison
            normal = result.normal_metrics
            stressed = result.stressed_metrics
            
            logger.info(f"Performance Impact:")
            logger.info(f"  Normal Return: {normal.total_return:.2f}%")
            logger.info(f"  Stressed Return: {stressed.total_return:.2f}%")
            logger.info(f"  Performance Degradation: {result.performance_degradation:.1%}")
            
            logger.info(f"Risk Impact:")
            logger.info(f"  Normal Max DD: {normal.max_drawdown:.2f}%")
            logger.info(f"  Stressed Max DD: {stressed.max_drawdown:.2f}%")
            logger.info(f"  Worst Case Loss: {result.worst_case_loss:.2f}%")
            
            if result.time_to_recovery:
                logger.info(f"  Recovery Time: {result.time_to_recovery} days")
            
            logger.info(f"Risk Management:")
            logger.info(f"  Max Leverage Used: {result.max_leverage_used:.2f}")
            logger.info(f"  Positions Liquidated: {result.positions_liquidated}")
    
    def compare_walk_forward_results(self, results: Dict[str, BacktestResult]):
        """Compare results from different walk-forward methods."""
        
        logger.info("\n" + "=" * 50)
        logger.info("WALK-FORWARD COMPARISON")
        logger.info("=" * 50)
        
        comparison_data = []
        
        for method, result in results.items():
            comparison_data.append({
                'Method': method,
                'Periods': result.total_periods,
                'Return': f"{result.overall_metrics.total_return:.2f}%",
                'Sharpe': f"{result.overall_metrics.sharpe_ratio:.3f}",
                'Max DD': f"{result.overall_metrics.max_drawdown:.2f}%",
                'Consistency': f"{result.performance_consistency:.3f}",
                'Execution Time': f"{result.execution_duration:.1f}s"
            })
        
        # Display comparison table
        logger.info(f"{'Method':<12} {'Periods':<8} {'Return':<8} {'Sharpe':<8} {'Max DD':<8} {'Consistency':<12} {'Time':<8}")
        logger.info("-" * 70)
        
        for data in comparison_data:
            logger.info(f"{data['Method']:<12} {data['Periods']:<8} {data['Return']:<8} "
                       f"{data['Sharpe']:<8} {data['Max DD']:<8} {data['Consistency']:<12} {data['Execution Time']:<8}")
        
        # Recommendations
        logger.info(f"\nRECOMMENDations:")
        
        # Find best performing method
        best_return = max(results.items(), key=lambda x: x[1].overall_metrics.total_return)
        best_sharpe = max(results.items(), key=lambda x: x[1].overall_metrics.sharpe_ratio)
        best_consistency = max(results.items(), key=lambda x: x[1].performance_consistency)
        
        logger.info(f"  Best Return: {best_return[0]} ({best_return[1].overall_metrics.total_return:.2f}%)")
        logger.info(f"  Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1].overall_metrics.sharpe_ratio:.3f})")
        logger.info(f"  Most Consistent: {best_consistency[0]} (Score: {best_consistency[1].performance_consistency:.3f})")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive backtesting demonstration."""
        
        logger.info("Starting Comprehensive Backtesting Demo")
        logger.info("=" * 60)
        
        try:
            # 1. Basic backtesting
            basic_result = await self.run_basic_backtest()
            
            # 2. Walk-forward analysis
            wf_result = await self.run_walk_forward_analysis()
            
            # 3. Stress testing
            stress_results = await self.run_stress_testing()
            
            # 4. Summary and recommendations
            self.generate_summary_report(basic_result, wf_result, stress_results)
            
            logger.info("\n" + "=" * 60)
            logger.info("DEMO COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise
    
    def generate_summary_report(self, basic_result, wf_result, stress_results):
        """Generate comprehensive summary report."""
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPREHENSIVE SUMMARY REPORT")
        logger.info("=" * 60)
        
        # Strategy performance summary
        logger.info(f"\nSTRATEGY PERFORMANCE SUMMARY:")
        logger.info(f"  Overall Return: {basic_result.overall_metrics.total_return:.2f}%")
        logger.info(f"  Risk-Adjusted Return (Sharpe): {basic_result.overall_metrics.sharpe_ratio:.3f}")
        logger.info(f"  Maximum Drawdown: {basic_result.overall_metrics.max_drawdown:.2f}%")
        logger.info(f"  Performance Consistency: {basic_result.performance_consistency:.3f}")
        
        # Risk assessment
        logger.info(f"\nRISK ASSESSMENT:")
        logger.info(f"  Value at Risk (95%): {basic_result.var_95:.2f}%")
        logger.info(f"  Conditional VaR (95%): {basic_result.cvar_95:.2f}%")
        
        # Stress test summary
        if stress_results:
            worst_scenario = min(stress_results, key=lambda x: x.stressed_metrics.total_return)
            logger.info(f"  Worst Stress Scenario: {worst_scenario.scenario.name}")
            logger.info(f"  Worst Case Return: {worst_scenario.stressed_metrics.total_return:.2f}%")
            logger.info(f"  Worst Case Drawdown: {worst_scenario.worst_case_loss:.2f}%")
        
        # Trading efficiency
        logger.info(f"\nTRADING EFFICIENCY:")
        logger.info(f"  Total Trades: {basic_result.overall_metrics.total_trades}")
        logger.info(f"  Win Rate: {basic_result.overall_metrics.win_rate:.1%}")
        logger.info(f"  Profit Factor: {basic_result.overall_metrics.profit_factor:.2f}")
        
        # Recommendations
        logger.info(f"\nRECOMMENDATIONS:")
        
        # Performance-based recommendations
        if basic_result.overall_metrics.sharpe_ratio > 1.0:
            logger.info("  ✓ Strong risk-adjusted performance (Sharpe > 1.0)")
        elif basic_result.overall_metrics.sharpe_ratio > 0.5:
            logger.info("  ⚠ Moderate risk-adjusted performance (0.5 < Sharpe < 1.0)")
        else:
            logger.info("  ✗ Poor risk-adjusted performance (Sharpe < 0.5)")
        
        # Drawdown-based recommendations
        if basic_result.overall_metrics.max_drawdown < 10:
            logger.info("  ✓ Low maximum drawdown (<10%)")
        elif basic_result.overall_metrics.max_drawdown < 20:
            logger.info("  ⚠ Moderate maximum drawdown (10-20%)")
        else:
            logger.info("  ✗ High maximum drawdown (>20%) - Consider risk management")
        
        # Consistency recommendations
        if basic_result.performance_consistency > 0.7:
            logger.info("  ✓ High performance consistency")
        elif basic_result.performance_consistency > 0.5:
            logger.info("  ⚠ Moderate performance consistency")
        else:
            logger.info("  ✗ Low performance consistency - Strategy may be unstable")
        
        # Statistical significance
        if (basic_result.overall_metrics.p_value is not None and 
            basic_result.overall_metrics.p_value < 0.05):
            logger.info("  ✓ Statistically significant results")
        else:
            logger.info("  ⚠ Results may not be statistically significant")


async def main():
    """Main demo function."""
    
    # Create and run demo
    demo = BacktestingDemo()
    await demo.run_comprehensive_demo()


if __name__ == '__main__':
    # Run the demo
    asyncio.run(main())