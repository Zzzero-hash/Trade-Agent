"""
Integration tests for backtesting framework with existing system components.

Tests the integration between backtesting engine and other system services.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.models.backtesting import BacktestConfig, BacktestPeriodType
from src.models.trading_signal import TradingSignal, TradingAction
from src.services.backtesting_engine import BacktestingEngine


class TestBacktestingIntegration:
    """Integration tests for backtesting framework."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create realistic sample market data."""
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        
        data = []
        for symbol in ['AAPL', 'GOOGL']:
            base_price = 150.0 if symbol == 'AAPL' else 2500.0
            
            for i, date in enumerate(dates):
                # Generate realistic price movement
                price_change = np.random.normal(0.001, 0.02)  # 0.1% daily drift, 2% volatility
                price = base_price * (1 + price_change * i / len(dates))
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * 0.999,
                    'high': price * 1.005,
                    'low': price * 0.995,
                    'close': price,
                    'volume': np.random.randint(1000000, 5000000),
                    'returns': price_change,
                    'volatility': np.random.uniform(0.15, 0.25),
                    'rsi': np.random.uniform(30, 70),
                    'macd': np.random.normal(0, 1),
                    'macd_signal': np.random.normal(0, 1),
                    'bb_position': np.random.uniform(0, 1),
                    'volume_ratio': np.random.uniform(0.8, 1.2),
                    'sma_5': price * 0.998,
                    'sma_20': price * 0.995,
                    'ema_12': price * 0.997
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for integration testing."""
        
        # Mock data aggregator
        data_aggregator = Mock()
        data_aggregator.get_historical_data = AsyncMock()
        
        # Mock decision engine
        decision_engine = Mock()
        decision_engine.generate_signal = AsyncMock()
        
        # Mock portfolio service
        portfolio_service = Mock()
        
        return data_aggregator, decision_engine, portfolio_service
    
    @pytest.mark.asyncio
    async def test_end_to_end_backtesting_workflow(self, sample_market_data, mock_services):
        """Test complete end-to-end backtesting workflow."""
        
        data_aggregator, decision_engine, portfolio_service = mock_services
        
        # Setup mock data aggregator
        def mock_get_data(symbol, **kwargs):
            return sample_market_data[sample_market_data['symbol'] == symbol].copy()
        
        data_aggregator.get_historical_data.side_effect = mock_get_data
        
        # Setup mock decision engine to generate realistic signals
        def mock_generate_signal(symbol, market_data=None):
            if np.random.random() < 0.3:  # 30% chance of actionable signal
                action = np.random.choice([TradingAction.BUY, TradingAction.SELL])
                return TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=np.random.uniform(0.6, 0.9),
                    position_size=np.random.uniform(0.05, 0.15),
                    timestamp=datetime.now(timezone.utc),
                    model_version='integration-test-v1.0'
                )
            else:
                return TradingSignal(
                    symbol=symbol,
                    action=TradingAction.HOLD,
                    confidence=0.5,
                    position_size=0.0,
                    timestamp=datetime.now(timezone.utc),
                    model_version='integration-test-v1.0'
                )
        
        decision_engine.generate_signal.side_effect = mock_generate_signal
        
        # Create backtesting engine
        engine = BacktestingEngine(
            data_aggregator=data_aggregator,
            decision_engine=decision_engine,
            portfolio_service=portfolio_service
        )
        
        # Create test configuration
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            symbols=['AAPL', 'GOOGL'],
            training_period_days=30,
            testing_period_days=15,
            period_type=BacktestPeriodType.ROLLING,
            initial_balance=100000.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        # Run backtest
        result = await engine.run_backtest(config)
        
        # Verify results
        assert result is not None
        assert result.total_periods > 0
        assert len(result.period_results) == result.total_periods
        assert result.overall_metrics is not None
        
        # Verify data was loaded for each symbol
        assert data_aggregator.get_historical_data.call_count >= len(config.symbols)
        
        # Verify signals were generated
        assert decision_engine.generate_signal.call_count > 0
        
        # Verify performance metrics are reasonable
        assert isinstance(result.overall_metrics.total_return, float)
        assert isinstance(result.overall_metrics.sharpe_ratio, float)
        assert result.overall_metrics.max_drawdown >= 0
        assert result.overall_metrics.total_trades >= 0
        
        # Verify risk metrics
        assert result.var_95 >= 0
        assert result.cvar_95 >= 0
        assert result.cvar_95 >= result.var_95
        
        print(f"Integration test completed successfully:")
        print(f"  Periods processed: {result.total_periods}")
        print(f"  Total return: {result.overall_metrics.total_return:.2f}%")
        print(f"  Sharpe ratio: {result.overall_metrics.sharpe_ratio:.3f}")
        print(f"  Max drawdown: {result.overall_metrics.max_drawdown:.2f}%")
        print(f"  Total trades: {result.overall_metrics.total_trades}")
    
    @pytest.mark.asyncio
    async def test_backtesting_with_different_configurations(self, sample_market_data, mock_services):
        """Test backtesting with different configuration parameters."""
        
        data_aggregator, decision_engine, portfolio_service = mock_services
        
        # Setup mocks
        data_aggregator.get_historical_data.side_effect = lambda symbol, **kwargs: \
            sample_market_data[sample_market_data['symbol'] == symbol].copy()
        
        decision_engine.generate_signal.return_value = TradingSignal(
            symbol='AAPL',
            action=TradingAction.HOLD,
            confidence=0.5,
            position_size=0.0,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        # Create backtesting engine
        engine = BacktestingEngine(
            data_aggregator=data_aggregator,
            decision_engine=decision_engine,
            portfolio_service=portfolio_service
        )
        
        # Test different configurations
        configs = [
            # Conservative configuration
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 2, 28),
                symbols=['AAPL'],
                training_period_days=20,
                testing_period_days=10,
                initial_balance=50000.0,
                max_position_size=0.1,
                transaction_cost=0.002,
                slippage=0.001
            ),
            # Aggressive configuration
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 2, 28),
                symbols=['AAPL', 'GOOGL'],
                training_period_days=15,
                testing_period_days=7,
                initial_balance=200000.0,
                max_position_size=0.3,
                transaction_cost=0.0005,
                slippage=0.0002
            )
        ]
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nTesting configuration {i+1}...")
            result = await engine.run_backtest(config)
            results.append(result)
            
            # Verify each result
            assert result.total_periods > 0
            assert len(result.period_results) == result.total_periods
            
            print(f"  Configuration {i+1} results:")
            print(f"    Periods: {result.total_periods}")
            print(f"    Return: {result.overall_metrics.total_return:.2f}%")
            print(f"    Max DD: {result.overall_metrics.max_drawdown:.2f}%")
        
        # Compare results
        assert len(results) == len(configs)
        
        # Results should be different due to different configurations
        if len(results) > 1:
            # At least some metrics should be different
            different_metrics = False
            for i in range(1, len(results)):
                if (results[i].overall_metrics.total_return != results[0].overall_metrics.total_return or
                    results[i].total_periods != results[0].total_periods):
                    different_metrics = True
                    break
            
            assert different_metrics, "Different configurations should produce different results"
    
    @pytest.mark.asyncio
    async def test_backtesting_error_handling(self, mock_services):
        """Test backtesting error handling and recovery."""
        
        data_aggregator, decision_engine, portfolio_service = mock_services
        
        # Create backtesting engine
        engine = BacktestingEngine(
            data_aggregator=data_aggregator,
            decision_engine=decision_engine,
            portfolio_service=portfolio_service
        )
        
        # Test with no data available
        data_aggregator.get_historical_data.return_value = pd.DataFrame()
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            symbols=['INVALID_SYMBOL'],
            training_period_days=10,
            testing_period_days=5
        )
        
        # Should raise an error when no data is available
        with pytest.raises(ValueError, match="No historical data could be loaded"):
            await engine.run_backtest(config)
        
        # Test with signal generation failure - need sufficient data for periods
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        test_data = []
        
        for date in dates:
            test_data.append({
                'timestamp': date,
                'symbol': 'AAPL',
                'open': 150.0,
                'high': 151.0,
                'low': 149.0,
                'close': 150.5,
                'volume': 1000000,
                'returns': 0.001,
                'volatility': 0.2,
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'bb_position': 0.5,
                'volume_ratio': 1.0,
                'sma_5': 150.0,
                'sma_20': 149.5,
                'ema_12': 150.2
            })
        
        data_aggregator.get_historical_data.return_value = pd.DataFrame(test_data)
        
        # Mock signal generation to raise an exception
        decision_engine.generate_signal.side_effect = Exception("Signal generation failed")
        
        # Should handle signal generation errors gracefully
        result = await engine.run_backtest(config)
        
        # Should still complete but with no trades
        assert result is not None
        assert result.overall_metrics.total_trades == 0
    
    def test_backtesting_engine_statistics(self, mock_services):
        """Test backtesting engine execution statistics tracking."""
        
        data_aggregator, decision_engine, portfolio_service = mock_services
        
        # Create backtesting engine
        engine = BacktestingEngine(
            data_aggregator=data_aggregator,
            decision_engine=decision_engine,
            portfolio_service=portfolio_service
        )
        
        # Check initial statistics
        stats = engine.get_execution_stats()
        
        assert stats['total_backtests'] == 0
        assert stats['total_periods_processed'] == 0
        assert stats['avg_execution_time'] == 0.0
        
        # Statistics should be properly tracked after running backtests
        # (This would be tested in the actual backtest execution tests above)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])