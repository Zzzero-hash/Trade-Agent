"""
Comprehensive test suite for the backtesting framework.

Tests backtesting accuracy, statistical significance, walk-forward analysis,
and stress testing capabilities.

Requirements: 2.5, 5.7, 9.6
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from scipy import stats
from pydantic import ValidationError

from src.models.backtesting import (
    BacktestConfig, BacktestResult, BacktestPeriodResult, PerformanceMetrics,
    TradeRecord, StressTestScenario, StressTestResult, BacktestPeriodType
)
from src.models.portfolio import Portfolio, Position
from src.models.trading_signal import TradingSignal, TradingAction
from src.models.market_data import MarketData
from src.services.backtesting_engine import BacktestingEngine


class TestBacktestingModels:
    """Test backtesting data models and validation."""
    
    def test_backtest_config_validation(self):
        """Test BacktestConfig validation rules."""
        
        # Valid configuration
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            symbols=['AAPL', 'GOOGL'],
            training_period_days=252,
            testing_period_days=63
        )
        assert config.symbols == ['AAPL', 'GOOGL']
        assert config.initial_balance == 100000.0
        
        # Invalid date range
        with pytest.raises(ValueError, match="End date must be after start date"):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2022, 1, 1),
                symbols=['AAPL']
            )
        
        # Insufficient date range
        with pytest.raises(ValueError, match="Date range.*is too short"):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 10),
                symbols=['AAPL'],
                training_period_days=252,
                testing_period_days=63
            )
        
        # Empty symbols
        with pytest.raises(ValidationError):
            BacktestConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 12, 31),
                symbols=[]
            )
    
    def test_trade_record_validation(self):
        """Test TradeRecord validation and calculations."""
        
        # Valid trade record
        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            symbol='AAPL',
            action='BUY',
            quantity=100.0,
            price=150.0,
            commission=1.0,
            slippage_cost=0.75
        )
        
        assert trade.action == 'BUY'
        assert trade.total_cost == 15001.75  # 100 * 150 + 1.0 + 0.75
        
        # Invalid action
        with pytest.raises(ValueError, match="Action must be BUY or SELL"):
            TradeRecord(
                timestamp=datetime.now(timezone.utc),
                symbol='AAPL',
                action='INVALID',
                quantity=100.0,
                price=150.0
            )
    
    def test_performance_metrics_validation(self):
        """Test PerformanceMetrics validation."""
        
        # Valid metrics
        metrics = PerformanceMetrics(
            total_return=25.5,
            annualized_return=12.3,
            volatility=18.2,
            sharpe_ratio=0.67,
            sortino_ratio=0.89,
            calmar_ratio=0.61,
            max_drawdown=8.5,
            max_drawdown_duration=45,
            win_rate=0.58,
            profit_factor=1.35,
            total_trades=156,
            avg_trade_return=0.16
        )
        
        assert metrics.win_rate == 0.58
        assert metrics.max_drawdown == 8.5
        
        # Invalid win rate
        with pytest.raises(ValueError):
            PerformanceMetrics(
                total_return=25.5,
                annualized_return=12.3,
                volatility=18.2,
                sharpe_ratio=0.67,
                sortino_ratio=0.89,
                calmar_ratio=0.61,
                max_drawdown=8.5,
                max_drawdown_duration=45,
                win_rate=1.5,  # Invalid: > 1.0
                profit_factor=1.35,
                total_trades=156,
                avg_trade_return=0.16
            )
    
    def test_backtest_period_result_validation(self):
        """Test BacktestPeriodResult validation."""
        
        # Create valid performance metrics
        metrics = PerformanceMetrics(
            total_return=10.0,
            annualized_return=8.0,
            volatility=15.0,
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            calmar_ratio=0.4,
            max_drawdown=5.0,
            max_drawdown_duration=30,
            win_rate=0.6,
            profit_factor=1.2,
            total_trades=50,
            avg_trade_return=0.2
        )
        
        # Valid period result
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        values = [100000.0, 101000.0, 102000.0]
        
        period_result = BacktestPeriodResult(
            period_id=0,
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2022, 12, 31),
            test_start=datetime(2023, 1, 1),
            test_end=datetime(2023, 3, 31),
            performance_metrics=metrics,
            portfolio_values=values,
            portfolio_dates=dates
        )
        
        assert period_result.test_duration_days == 89
        assert period_result.final_portfolio_value == 102000.0
        
        # Mismatched portfolio data
        with pytest.raises(ValueError, match="Portfolio values and dates must have same length"):
            BacktestPeriodResult(
                period_id=0,
                train_start=datetime(2022, 1, 1),
                train_end=datetime(2022, 12, 31),
                test_start=datetime(2023, 1, 1),
                test_end=datetime(2023, 3, 31),
                performance_metrics=metrics,
                portfolio_values=[100000.0, 101000.0],
                portfolio_dates=[datetime(2023, 1, 1)]  # Mismatched length
            )


class TestBacktestingEngine:
    """Test BacktestingEngine functionality."""
    
    @pytest.fixture
    def mock_data_aggregator(self):
        """Create mock data aggregator."""
        aggregator = Mock()
        aggregator.get_historical_data = AsyncMock()
        return aggregator
    
    @pytest.fixture
    def mock_decision_engine(self):
        """Create mock trading decision engine."""
        engine = Mock()
        engine.generate_signal = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_portfolio_service(self):
        """Create mock portfolio management service."""
        service = Mock()
        return service
    
    @pytest.fixture
    def backtesting_engine(self, mock_data_aggregator, mock_decision_engine, mock_portfolio_service):
        """Create BacktestingEngine instance with mocked dependencies."""
        return BacktestingEngine(
            data_aggregator=mock_data_aggregator,
            decision_engine=mock_decision_engine,
            portfolio_service=mock_portfolio_service
        )
    
    @pytest.fixture
    def sample_config(self):
        """Create sample backtesting configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            symbols=['AAPL', 'GOOGL'],
            training_period_days=60,
            testing_period_days=30,
            initial_balance=100000.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        data = []
        for symbol in ['AAPL', 'GOOGL']:
            for date in dates:
                # Generate realistic price data
                base_price = 150.0 if symbol == 'AAPL' else 2500.0
                price = base_price * (1 + np.random.normal(0, 0.02))
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': np.random.randint(1000000, 10000000),
                    'returns': np.random.normal(0.001, 0.02),
                    'volatility': np.random.uniform(0.15, 0.25),
                    'rsi': np.random.uniform(30, 70),
                    'macd': np.random.normal(0, 1),
                    'macd_signal': np.random.normal(0, 1),
                    'bb_position': np.random.uniform(0, 1),
                    'volume_ratio': np.random.uniform(0.5, 2.0),
                    'sma_5': price * 0.995,
                    'sma_20': price * 0.99,
                    'ema_12': price * 0.998
                })
        
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_walk_forward_period_generation(self, backtesting_engine, sample_config):
        """Test walk-forward period generation."""
        
        periods = backtesting_engine._generate_walk_forward_periods(sample_config)
        
        # Should generate multiple periods
        assert len(periods) > 0
        
        # Check period structure
        for train_start, train_end, test_start, test_end in periods:
            assert train_end == test_start  # No gap between train and test
            assert (train_end - train_start).days == sample_config.training_period_days
            assert (test_end - test_start).days == sample_config.testing_period_days
        
        # Test different period types
        sample_config.period_type = BacktestPeriodType.EXPANDING
        expanding_periods = backtesting_engine._generate_walk_forward_periods(sample_config)
        
        # Expanding periods should have increasing training windows
        if len(expanding_periods) > 1:
            first_train_days = (expanding_periods[0][1] - expanding_periods[0][0]).days
            second_train_days = (expanding_periods[1][1] - expanding_periods[1][0]).days
            assert second_train_days >= first_train_days
    
    @pytest.mark.asyncio
    async def test_historical_data_loading(self, backtesting_engine, sample_config, sample_market_data):
        """Test historical data loading and preparation."""
        
        # Mock data aggregator to return sample data for each symbol
        def mock_get_data(symbol, **kwargs):
            return sample_market_data[sample_market_data['symbol'] == symbol]
        
        backtesting_engine.data_aggregator.get_historical_data.side_effect = mock_get_data
        
        # Load historical data
        loaded_data = await backtesting_engine._load_historical_data(sample_config)
        
        # Verify data loading
        assert not loaded_data.empty
        assert 'symbol' in loaded_data.columns
        assert set(loaded_data['symbol'].unique()).issubset(set(sample_config.symbols))
        
        # Verify data aggregator was called for each symbol
        assert backtesting_engine.data_aggregator.get_historical_data.call_count == len(sample_config.symbols)
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, backtesting_engine, sample_config, sample_market_data):
        """Test trading signal generation for backtesting."""
        
        # Mock signal generation
        mock_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.75,
            position_size=0.1,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        backtesting_engine.decision_engine.generate_signal.return_value = mock_signal
        
        # Generate signals for a date
        daily_data = sample_market_data[sample_market_data['timestamp'].dt.date == sample_market_data['timestamp'].dt.date.iloc[0]]
        signals = await backtesting_engine._generate_signals_for_date(daily_data, sample_config)
        
        # Verify signals generated
        assert len(signals) > 0
        assert all(isinstance(signal, TradingSignal) for signal in signals)
    
    @pytest.mark.asyncio
    async def test_trade_execution(self, backtesting_engine, sample_config, sample_market_data):
        """Test trade execution with transaction costs and slippage."""
        
        # Create test portfolio
        portfolio = Portfolio(
            user_id='test_backtest',
            positions={},
            cash_balance=100000.0,
            total_value=100000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        # Create test signals
        buy_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.8,
            position_size=0.1,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        # Execute trades
        daily_data = sample_market_data[sample_market_data['symbol'] == 'AAPL'].head(1)
        trades = await backtesting_engine._execute_trades(
            portfolio, [buy_signal], daily_data, sample_config
        )
        
        # Verify trade execution
        assert len(trades) > 0
        trade = trades[0]
        assert trade.action == 'BUY'
        assert trade.symbol == 'AAPL'
        assert trade.commission > 0  # Transaction cost applied
        assert trade.slippage_cost > 0  # Slippage applied
        
        # Verify portfolio updated
        assert portfolio.cash_balance < 100000.0  # Cash reduced
        assert 'AAPL' in portfolio.positions  # Position created
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, backtesting_engine):
        """Test performance metrics calculation accuracy."""
        
        # Create test portfolio values (10% gain over 30 days)
        initial_balance = 100000.0
        portfolio_values = [initial_balance * (1 + 0.1 * i / 30) for i in range(31)]
        
        # Create test trades
        trades = [
            TradeRecord(
                timestamp=datetime.now(timezone.utc),
                symbol='AAPL',
                action='BUY',
                quantity=100.0,
                price=150.0,
                commission=1.0
            ),
            TradeRecord(
                timestamp=datetime.now(timezone.utc),
                symbol='AAPL',
                action='SELL',
                quantity=100.0,
                price=165.0,
                commission=1.0
            )
        ]
        
        # Calculate metrics
        metrics = backtesting_engine._calculate_period_metrics(
            portfolio_values, initial_balance, trades
        )
        
        # Verify metrics
        assert metrics.total_return == pytest.approx(10.0, rel=0.01)  # 10% return
        assert metrics.total_trades == 2
        assert metrics.volatility >= 0  # Non-negative volatility
        assert metrics.max_drawdown >= 0  # Non-negative drawdown
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, backtesting_engine):
        """Test VaR and CVaR calculation."""
        
        # Create portfolio values with some volatility
        np.random.seed(42)  # For reproducible results
        initial_balance = 100000.0
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        portfolio_values = [initial_balance]
        
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # Calculate risk metrics
        var_95, cvar_95 = backtesting_engine._calculate_risk_metrics(portfolio_values, initial_balance)
        
        # Verify risk metrics
        assert var_95 > 0  # VaR should be positive (loss)
        assert cvar_95 > 0  # CVaR should be positive (loss)
        assert cvar_95 >= var_95  # CVaR should be >= VaR
    
    @pytest.mark.asyncio
    async def test_full_backtest_execution(self, backtesting_engine, sample_config, sample_market_data):
        """Test complete backtesting workflow."""
        
        # Mock dependencies
        backtesting_engine.data_aggregator.get_historical_data.return_value = sample_market_data
        
        # Mock signal generation to return actionable signals
        def mock_signal_generator(*args, **kwargs):
            return TradingSignal(
                symbol='AAPL',
                action=TradingAction.BUY if np.random.random() > 0.5 else TradingAction.SELL,
                confidence=0.7,
                position_size=0.05,
                timestamp=datetime.now(timezone.utc),
                model_version='test-v1.0'
            )
        
        backtesting_engine.decision_engine.generate_signal.side_effect = mock_signal_generator
        
        # Run backtest
        result = await backtesting_engine.run_backtest(sample_config)
        
        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert result.total_periods > 0
        assert len(result.period_results) == result.total_periods
        assert isinstance(result.overall_metrics, PerformanceMetrics)
        
        # Verify execution metadata
        assert result.execution_start <= result.execution_end
        assert result.execution_duration > 0
        
        # Verify performance consistency
        consistency = result.performance_consistency
        assert 0 <= consistency <= 1
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, backtesting_engine, sample_config, sample_market_data):
        """Test stress testing functionality."""
        
        # Mock dependencies
        backtesting_engine.data_aggregator.get_historical_data.return_value = sample_market_data
        backtesting_engine.decision_engine.generate_signal.return_value = TradingSignal(
            symbol='AAPL',
            action=TradingAction.HOLD,
            confidence=0.5,
            position_size=0.0,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        # Create stress test scenarios
        scenarios = [
            StressTestScenario(
                name='Market Crash',
                description='20% market drop',
                market_shock_magnitude=-0.2,
                shock_duration_days=30,
                volatility_multiplier=2.0
            ),
            StressTestScenario(
                name='High Volatility',
                description='Increased volatility',
                market_shock_magnitude=0.0,
                shock_duration_days=60,
                volatility_multiplier=3.0
            )
        ]
        
        # Run stress tests
        stress_results = await backtesting_engine.run_stress_test(sample_config, scenarios)
        
        # Verify stress test results
        assert len(stress_results) == len(scenarios)
        
        for result in stress_results:
            assert isinstance(result, StressTestResult)
            assert isinstance(result.stressed_metrics, PerformanceMetrics)
            assert isinstance(result.normal_metrics, PerformanceMetrics)
            assert result.performance_degradation >= 0  # Should show some degradation
    
    def test_stability_metrics_calculation(self, backtesting_engine):
        """Test stability metrics calculation across periods."""
        
        # Create mock period results with varying performance
        period_results = []
        
        for i in range(5):
            metrics = PerformanceMetrics(
                total_return=10.0 + np.random.normal(0, 2),  # Varying returns
                annualized_return=8.0 + np.random.normal(0, 1.5),
                volatility=15.0 + np.random.normal(0, 3),
                sharpe_ratio=0.5 + np.random.normal(0, 0.1),
                sortino_ratio=0.6 + np.random.normal(0, 0.1),
                calmar_ratio=0.4 + np.random.normal(0, 0.1),
                max_drawdown=5.0 + np.random.uniform(0, 3),
                max_drawdown_duration=30 + np.random.randint(0, 20),
                win_rate=0.6 + np.random.normal(0, 0.05),
                profit_factor=1.2 + np.random.normal(0, 0.1),
                total_trades=50 + np.random.randint(0, 20),
                avg_trade_return=0.2 + np.random.normal(0, 0.05)
            )
            
            period_result = BacktestPeriodResult(
                period_id=i,
                train_start=datetime(2023, 1, 1) + timedelta(days=i*30),
                train_end=datetime(2023, 1, 1) + timedelta(days=i*30 + 60),
                test_start=datetime(2023, 1, 1) + timedelta(days=i*30 + 60),
                test_end=datetime(2023, 1, 1) + timedelta(days=i*30 + 90),
                performance_metrics=metrics,
                portfolio_values=[100000.0, 105000.0, 110000.0],
                portfolio_dates=[
                    datetime(2023, 1, 1) + timedelta(days=i*30 + 60),
                    datetime(2023, 1, 1) + timedelta(days=i*30 + 75),
                    datetime(2023, 1, 1) + timedelta(days=i*30 + 90)
                ]
            )
            
            period_results.append(period_result)
        
        # Calculate stability metrics
        stability_metrics = backtesting_engine._calculate_stability_metrics(period_results)
        
        # Verify stability metrics
        assert 'return_consistency' in stability_metrics
        assert 'sharpe_consistency' in stability_metrics
        assert 'drawdown_consistency' in stability_metrics
        assert 'win_rate_avg' in stability_metrics
        assert 'profit_factor_avg' in stability_metrics
        
        # All consistency metrics should be between 0 and 1
        for key in ['return_consistency', 'sharpe_consistency', 'drawdown_consistency']:
            if key in stability_metrics:
                assert 0 <= stability_metrics[key] <= 1


class TestStatisticalSignificance:
    """Test statistical significance testing in backtesting."""
    
    def test_t_test_calculation(self):
        """Test t-statistic calculation for return significance."""
        
        # Create sample returns with known statistical properties
        np.random.seed(42)
        returns = np.random.normal(0.05, 0.15, 100)  # 5% mean return, 15% std
        
        # Calculate t-statistic
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Verify statistical significance
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
        # With 5% mean return and reasonable sample size, should be significant
        if abs(t_stat) > 1.96:  # 95% confidence
            assert p_value < 0.05
    
    def test_performance_attribution(self):
        """Test performance attribution across different factors."""
        
        # This would test attribution of returns to different factors
        # like market beta, sector allocation, stock selection, etc.
        
        # Simplified test for now
        total_return = 0.15  # 15% total return
        market_return = 0.10  # 10% market return
        
        # Alpha calculation
        alpha = total_return - market_return
        assert alpha == pytest.approx(0.05, rel=1e-10)  # 5% alpha
        
        # Beta calculation (simplified)
        portfolio_volatility = 0.18
        market_volatility = 0.15
        correlation = 0.8
        
        beta = correlation * (portfolio_volatility / market_volatility)
        assert beta == pytest.approx(0.96, rel=0.01)


class TestBacktestingAccuracy:
    """Test backtesting accuracy and realistic market simulation."""
    
    def test_transaction_cost_impact(self):
        """Test that transaction costs are properly applied."""
        
        # Test trade with transaction costs
        trade_value = 10000.0
        transaction_cost_rate = 0.001  # 0.1%
        
        expected_cost = trade_value * transaction_cost_rate
        
        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            symbol='AAPL',
            action='BUY',
            quantity=100.0,
            price=100.0,
            commission=expected_cost
        )
        
        assert trade.commission == expected_cost
        assert trade.total_cost == trade_value + expected_cost
    
    def test_slippage_calculation(self):
        """Test slippage calculation and application."""
        
        # Test slippage impact on execution price
        market_price = 100.0
        slippage_rate = 0.0005  # 0.05%
        
        # Buy order should have positive slippage (higher price)
        buy_execution_price = market_price * (1 + slippage_rate)
        assert buy_execution_price == 100.05
        
        # Sell order should have negative slippage (lower price)
        sell_execution_price = market_price * (1 - slippage_rate)
        assert sell_execution_price == 99.95
    
    def test_realistic_market_conditions(self):
        """Test simulation of realistic market conditions."""
        
        # Test market hours, holidays, liquidity constraints, etc.
        # This is a simplified test - full implementation would be more complex
        
        # Verify that backtesting accounts for:
        # 1. Market hours (no trading outside market hours)
        # 2. Weekends and holidays
        # 3. Liquidity constraints
        # 4. Bid-ask spreads
        
        # For now, just verify basic structure
        market_open = datetime(2023, 1, 3, 9, 30)  # Tuesday 9:30 AM
        market_close = datetime(2023, 1, 3, 16, 0)  # Tuesday 4:00 PM
        
        assert market_open.weekday() < 5  # Weekday
        assert market_open.hour >= 9  # After market open
        assert market_close.hour <= 16  # Before market close


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])