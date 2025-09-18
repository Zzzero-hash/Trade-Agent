"""
Comprehensive tests for TradingEnvironment.

This is a mission-critical system that requires thorough testing of all components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.ml.trading_environment import (
    TradingEnvironment, 
    TradingConfig, 
    ActionType, 
    MarketState
)


class TestTradingConfig:
    """Test TradingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TradingConfig()
        
        assert config.initial_balance == 100000.0
        assert config.max_position_size == 0.2
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.lookback_window == 60
        assert config.max_drawdown_limit == 0.2
        assert config.risk_free_rate == 0.02
        assert config.reward_scaling == 1000.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TradingConfig(
            initial_balance=50000.0,
            max_position_size=0.1,
            transaction_cost=0.002
        )
        
        assert config.initial_balance == 50000.0
        assert config.max_position_size == 0.1
        assert config.transaction_cost == 0.002


class TestTradingEnvironment:
    """Test TradingEnvironment class."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'GOOGL']
        
        data = []
        for symbol in symbols:
            np.random.seed(42 if symbol == 'AAPL' else 43)  # Reproducible data
            
            # Generate realistic price data
            initial_price = 150.0 if symbol == 'AAPL' else 2500.0
            returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
            prices = [initial_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            for i, date in enumerate(dates):
                price = prices[i]
                # Generate OHLC from close price
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = price * (1 + np.random.normal(0, 0.005))
                volume = np.random.uniform(1000000, 5000000)
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': max(high, price, open_price),
                    'low': min(low, price, open_price),
                    'close': price,
                    'volume': volume
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def trading_config(self):
        """Create test trading configuration."""
        return TradingConfig(
            initial_balance=10000.0,
            max_position_size=0.3,
            transaction_cost=0.001,
            slippage=0.0005,
            lookback_window=10,
            max_drawdown_limit=0.15,
            reward_scaling=100.0
        )
    
    @pytest.fixture
    def trading_env(self, sample_market_data, trading_config):
        """Create trading environment for testing."""
        return TradingEnvironment(
            market_data=sample_market_data,
            config=trading_config,
            symbols=['AAPL', 'GOOGL']
        )
    
    def test_environment_initialization(self, trading_env, trading_config):
        """Test environment initialization."""
        assert trading_env.config == trading_config
        assert trading_env.symbols == ['AAPL', 'GOOGL']
        assert trading_env.n_symbols == 2
        assert trading_env.initial_balance == 10000.0
        assert trading_env.current_step == 0
        assert len(trading_env.positions) == 2
        assert all(pos == 0.0 for pos in trading_env.positions.values())
    
    def test_action_space(self, trading_env):
        """Test action space definition."""
        action_space = trading_env.action_space
        
        # Should have 2 values per symbol (action_type, position_size)
        expected_size = 2 * 2  # 2 symbols * 2 values
        assert action_space.shape == (expected_size,)
        
        # Check bounds
        assert np.all(action_space.low == [0.0, 0.0, 0.0, 0.0])
        assert np.all(action_space.high == [2.0, 0.3, 2.0, 0.3])
    
    def test_observation_space(self, trading_env):
        """Test observation space definition."""
        obs_space = trading_env.observation_space
        
        # Market features: 15 features * 2 symbols * 10 lookback = 300
        # Portfolio features: 3 + 2 symbols = 5
        expected_size = 15 * 2 * 10 + 3 + 2
        assert obs_space.shape == (expected_size,)
    
    def test_reset(self, trading_env):
        """Test environment reset functionality."""
        obs, info = trading_env.reset(seed=42)
        
        # Check state reset
        assert trading_env.current_step == trading_env.config.lookback_window
        assert trading_env.cash_balance == trading_env.initial_balance
        assert trading_env.portfolio_value == trading_env.initial_balance
        assert all(pos == 0.0 for pos in trading_env.positions.values())
        
        # Check observation shape
        assert obs.shape == trading_env.observation_space.shape
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        
        # Check info
        assert 'step' in info
        assert 'portfolio_value' in info
        assert 'positions' in info
    
    def test_step_hold_action(self, trading_env):
        """Test step with HOLD actions."""
        obs, info = trading_env.reset(seed=42)
        initial_portfolio_value = trading_env.portfolio_value
        
        # HOLD action for both symbols
        action = np.array([0.0, 0.0, 0.0, 0.0])  # HOLD, 0 size for both symbols
        
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # Portfolio should remain unchanged (except for potential market movements)
        assert trading_env.cash_balance == trading_env.initial_balance
        assert all(pos == 0.0 for pos in trading_env.positions.values())
        assert not terminated
        assert not truncated
    
    def test_step_buy_action(self, trading_env):
        """Test step with BUY actions."""
        obs, info = trading_env.reset(seed=42)
        initial_cash = trading_env.cash_balance
        
        # BUY action for AAPL
        action = np.array([1.0, 0.1, 0.0, 0.0])  # BUY 10% position in AAPL, HOLD GOOGL
        
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # Should have bought AAPL shares
        assert trading_env.cash_balance < initial_cash
        assert trading_env.positions['AAPL'] > 0
        assert trading_env.positions['GOOGL'] == 0
        assert isinstance(reward, float)
    
    def test_step_sell_action(self, trading_env):
        """Test step with SELL actions."""
        obs, info = trading_env.reset(seed=42)
        
        # First buy some shares
        buy_action = np.array([1.0, 0.2, 1.0, 0.1])  # BUY both symbols
        trading_env.step(buy_action)
        
        cash_after_buy = trading_env.cash_balance
        aapl_position = trading_env.positions['AAPL']
        
        # Now sell AAPL
        sell_action = np.array([2.0, 0.5, 0.0, 0.0])  # SELL 50% of AAPL, HOLD GOOGL
        obs, reward, terminated, truncated, info = trading_env.step(sell_action)
        
        # Should have sold some AAPL shares
        assert trading_env.cash_balance > cash_after_buy
        assert trading_env.positions['AAPL'] < aapl_position
        assert trading_env.positions['AAPL'] >= 0  # Should not go negative
    
    def test_transaction_costs(self, trading_env):
        """Test transaction costs are applied correctly."""
        obs, info = trading_env.reset(seed=42)
        initial_cash = trading_env.cash_balance
        
        # Get current price for calculation
        current_prices = trading_env._get_current_prices()
        aapl_price = current_prices['AAPL']
        
        # BUY action
        position_size = 0.1
        action = np.array([1.0, position_size, 0.0, 0.0])
        
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # Calculate expected cost with transaction fees and slippage
        max_shares = (initial_cash * position_size) / aapl_price
        slippage = aapl_price * trading_env.config.slippage
        execution_price = aapl_price + slippage
        gross_cost = max_shares * execution_price
        transaction_cost = gross_cost * trading_env.config.transaction_cost
        expected_total_cost = gross_cost + transaction_cost
        
        # Cash should be reduced by approximately the expected cost
        actual_cost = initial_cash - trading_env.cash_balance
        assert abs(actual_cost - expected_total_cost) < 1.0  # Allow small rounding differences
    
    def test_portfolio_value_calculation(self, trading_env):
        """Test portfolio value calculation."""
        obs, info = trading_env.reset(seed=42)
        
        # Buy some shares
        action = np.array([1.0, 0.2, 1.0, 0.1])
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # Calculate expected portfolio value
        current_prices = trading_env._get_current_prices()
        expected_positions_value = sum(
            trading_env.positions[symbol] * current_prices[symbol]
            for symbol in trading_env.symbols
        )
        expected_portfolio_value = trading_env.cash_balance + expected_positions_value
        
        assert abs(trading_env.portfolio_value - expected_portfolio_value) < 0.01
    
    def test_reward_calculation(self, trading_env):
        """Test reward calculation includes risk-adjusted metrics."""
        obs, info = trading_env.reset(seed=42)
        
        # Take some actions to generate returns
        action = np.array([1.0, 0.1, 1.0, 0.1])
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # Reward should be a finite number
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_termination_conditions(self, trading_env):
        """Test environment termination conditions."""
        obs, info = trading_env.reset(seed=42)
        
        # Simulate large loss to trigger termination
        trading_env.portfolio_value = trading_env.initial_balance * 0.05  # 95% loss
        
        action = np.array([0.0, 0.0, 0.0, 0.0])  # HOLD
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        assert terminated
    
    def test_drawdown_termination(self, trading_env):
        """Test termination due to maximum drawdown."""
        obs, info = trading_env.reset(seed=42)
        
        # Set up scenario where drawdown exceeds limit
        trading_env.max_portfolio_value = 20000.0
        trading_env.portfolio_value = 16000.0  # 20% drawdown
        
        assert trading_env._is_terminated()
    
    def test_observation_consistency(self, trading_env):
        """Test observation consistency and validity."""
        obs, info = trading_env.reset(seed=42)
        
        for _ in range(10):
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            
            # Check observation properties
            assert obs.shape == trading_env.observation_space.shape
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))
            
            if terminated or truncated:
                break
    
    def test_action_parsing(self, trading_env):
        """Test action parsing functionality."""
        action = np.array([1.5, 0.2, 2.1, 0.05])  # Mixed actions with clipping needed
        
        parsed_actions = trading_env._parse_action(action)
        
        assert len(parsed_actions) == 2
        assert parsed_actions[0]['symbol'] == 'AAPL'
        assert parsed_actions[0]['action_type'] == ActionType.BUY  # 1.5 -> 1
        assert parsed_actions[0]['position_size'] == 0.2
        
        assert parsed_actions[1]['symbol'] == 'GOOGL'
        assert parsed_actions[1]['action_type'] == ActionType.SELL  # 2.1 -> 2
        assert parsed_actions[1]['position_size'] == 0.05
    
    def test_market_data_preparation(self, sample_market_data):
        """Test market data preparation and feature calculation."""
        env = TradingEnvironment(sample_market_data)
        
        # Check that technical indicators were calculated
        required_features = [
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'volume_ratio', 'sma_5', 'sma_20', 'ema_12'
        ]
        
        for feature in required_features:
            assert feature in env.market_data.columns
    
    def test_portfolio_metrics(self, trading_env):
        """Test portfolio performance metrics calculation."""
        obs, info = trading_env.reset(seed=42)
        
        # Execute several steps to build history
        for _ in range(20):
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            
            if terminated or truncated:
                break
        
        metrics = trading_env.get_portfolio_metrics()
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'total_trades', 'final_portfolio_value'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
    
    def test_render_functionality(self, trading_env, capsys):
        """Test rendering functionality."""
        obs, info = trading_env.reset(seed=42)
        
        # Test human rendering
        trading_env.render(mode="human")
        captured = capsys.readouterr()
        
        assert "Step:" in captured.out
        assert "Portfolio Value:" in captured.out
        assert "Cash Balance:" in captured.out
    
    def test_edge_cases(self, trading_env):
        """Test edge cases and error conditions."""
        # Test with invalid market data
        with pytest.raises(ValueError):
            invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
            TradingEnvironment(invalid_data)
        
        # Test action bounds
        obs, info = trading_env.reset(seed=42)
        
        # Extreme actions should be clipped
        extreme_action = np.array([10.0, 5.0, -1.0, -0.5])
        parsed = trading_env._parse_action(extreme_action)
        
        assert parsed[0]['action_type'] == ActionType.SELL  # 10.0 -> 2
        assert parsed[0]['position_size'] == trading_env.config.max_position_size  # 5.0 -> max
        assert parsed[1]['action_type'] == ActionType.HOLD  # -1.0 -> 0
        assert parsed[1]['position_size'] == 0.0  # -0.5 -> 0
    
    def test_multi_step_consistency(self, trading_env):
        """Test consistency across multiple steps."""
        obs, info = trading_env.reset(seed=42)
        
        portfolio_values = [trading_env.portfolio_value]
        
        # Execute multiple steps and track portfolio value
        for i in range(30):
            action = np.array([1.0, 0.05, 1.0, 0.05])  # Small consistent buys
            obs, reward, terminated, truncated, info = trading_env.step(action)
            
            portfolio_values.append(trading_env.portfolio_value)
            
            # Portfolio value should always be positive
            assert trading_env.portfolio_value > 0
            
            # Cash + positions should equal portfolio value
            current_prices = trading_env._get_current_prices()
            positions_value = sum(
                trading_env.positions[symbol] * current_prices[symbol]
                for symbol in trading_env.symbols
            )
            expected_value = trading_env.cash_balance + positions_value
            assert abs(trading_env.portfolio_value - expected_value) < 0.01
            
            if terminated or truncated:
                break
    
    def test_dynamic_portfolio_alignment(self, trading_env):
        """Test that environment dynamically aligns to portfolio needs."""
        obs, info = trading_env.reset(seed=42)
        
        # Test different portfolio configurations
        configs = [
            {'symbols': ['AAPL'], 'max_position_size': 0.5},
            {'symbols': ['AAPL', 'GOOGL'], 'max_position_size': 0.2},
        ]
        
        for config in configs:
            # Create environment with different configuration
            test_env = TradingEnvironment(
                market_data=trading_env.market_data,
                config=TradingConfig(max_position_size=config['max_position_size']),
                symbols=config['symbols']
            )
            
            obs, info = test_env.reset(seed=42)
            
            # Verify environment adapts to configuration
            assert test_env.symbols == config['symbols']
            assert test_env.config.max_position_size == config['max_position_size']
            assert test_env.n_symbols == len(config['symbols'])
            
            # Action space should adapt
            expected_action_size = len(config['symbols']) * 2
            assert test_env.action_space.shape == (expected_action_size,)


class TestIntegration:
    """Integration tests for TradingEnvironment with external components."""
    
    def test_gymnasium_compatibility(self, sample_market_data):
        """Test compatibility with Gymnasium interface."""
        env = TradingEnvironment(sample_market_data)
        
        # Test standard Gymnasium interface
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            if terminated or truncated:
                break
    
    def test_stable_baselines3_compatibility(self, sample_market_data):
        """Test compatibility with Stable-Baselines3 interface."""
        env = TradingEnvironment(sample_market_data)
        
        # Test that environment can be used with SB3-style training loop
        obs, info = env.reset()
        
        # Simulate SB3 training loop
        for episode in range(3):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            assert isinstance(episode_reward, (int, float))
    
    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_reproducibility(self, sample_market_data, seed):
        """Test that environment is reproducible with same seed."""
        env1 = TradingEnvironment(sample_market_data)
        env2 = TradingEnvironment(sample_market_data)
        
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        np.testing.assert_array_equal(obs1, obs2)
        
        # Test multiple steps
        for _ in range(10):
            action = env1.action_space.sample()
            
            obs1, reward1, term1, trunc1, info1 = env1.step(action)
            obs2, reward2, term2, trunc2, info2 = env2.step(action)
            
            np.testing.assert_array_almost_equal(obs1, obs2, decimal=6)
            assert abs(reward1 - reward2) < 1e-6
            assert term1 == term2
            assert trunc1 == trunc2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])