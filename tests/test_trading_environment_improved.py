"""
Comprehensive tests for the improved trading environment.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.ml.trading_environment import TradingEnvironment, TradingConfig
from src.ml.environment_factory import TradingEnvironmentFactory
from src.ml.reward_calculator import RiskAdjustedRewardCalculator
from src.ml.portfolio_manager import PortfolioManager
from src.ml.market_features import MarketFeatureCalculator


class TestTradingConfig:
    """Test trading configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = TradingConfig()
        assert config.initial_balance == 100000.0
        assert config.max_position_size == 0.2
    
    def test_invalid_initial_balance(self):
        """Test invalid initial balance validation."""
        with pytest.raises(ValueError, match="Initial balance must be positive"):
            TradingConfig(initial_balance=-1000.0)
    
    def test_invalid_position_size(self):
        """Test invalid position size validation."""
        with pytest.raises(ValueError, match="Max position size must be between 0 and 1"):
            TradingConfig(max_position_size=1.5)
    
    def test_invalid_transaction_cost(self):
        """Test invalid transaction cost validation."""
        with pytest.raises(ValueError, match="Transaction cost cannot be negative"):
            TradingConfig(transaction_cost=-0.01)


class TestPortfolioManager:
    """Test portfolio management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.symbols = ['AAPL', 'GOOGL']
        self.portfolio = PortfolioManager(100000.0, self.symbols)
    
    def test_initialization(self):
        """Test portfolio initialization."""
        assert self.portfolio.cash_balance == 100000.0
        assert self.portfolio.portfolio_value == 100000.0
        assert all(pos == 0.0 for pos in self.portfolio.positions.values())
    
    def test_buy_execution(self):
        """Test buy order execution."""
        result = self.portfolio.execute_trade(
            symbol='AAPL',
            action_type='BUY',
            position_size=0.1,
            current_price=150.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        assert result.executed
        assert result.quantity > 0
        assert self.portfolio.cash_balance < 100000.0
        assert self.portfolio.positions['AAPL'] > 0
    
    def test_sell_execution(self):
        """Test sell order execution."""
        # First buy some shares
        self.portfolio.execute_trade(
            'AAPL', 'BUY', 0.1, 150.0, 0.001, 0.0005
        )
        
        initial_position = self.portfolio.positions['AAPL']
        
        # Then sell half
        result = self.portfolio.execute_trade(
            'AAPL', 'SELL', 0.5, 155.0, 0.001, 0.0005
        )
        
        assert result.executed
        assert result.quantity < 0  # Negative for sell
        assert self.portfolio.positions['AAPL'] < initial_position


class TestRewardCalculator:
    """Test reward calculation strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskAdjustedRewardCalculator()
        self.config = TradingConfig()
    
    def test_positive_return_reward(self):
        """Test reward for positive returns."""
        reward = self.calculator.calculate_reward(
            portfolio_value=110000.0,
            previous_portfolio_value=100000.0,
            initial_balance=100000.0,
            returns_history=[0.05, 0.03, 0.02],
            trade_results=[],
            config=self.config
        )
        
        assert reward > 0  # Should be positive for gains
    
    def test_negative_return_penalty(self):
        """Test penalty for negative returns."""
        reward = self.calculator.calculate_reward(
            portfolio_value=90000.0,
            previous_portfolio_value=100000.0,
            initial_balance=100000.0,
            returns_history=[-0.05, -0.03, -0.02],
            trade_results=[],
            config=self.config
        )
        
        assert reward < 0  # Should be negative for losses


class TestMarketFeatureCalculator:
    """Test market feature calculation."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'AAPL',
            'open': np.random.uniform(140, 160, 100),
            'high': np.random.uniform(150, 170, 100),
            'low': np.random.uniform(130, 150, 100),
            'close': np.random.uniform(140, 160, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
    
    def test_feature_calculation(self):
        """Test feature calculation."""
        features = MarketFeatureCalculator.calculate_features(self.data)
        
        # Check that new columns were added
        feature_columns = MarketFeatureCalculator.get_feature_columns()
        for col in feature_columns:
            assert col in features.columns
        
        # Check that RSI is bounded
        rsi_values = features['rsi'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)


class TestTradingEnvironment:
    """Test trading environment functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'GOOGL']
        
        data = []
        for symbol in symbols:
            for i, date in enumerate(dates):
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': 150.0 + i * 0.1,
                    'high': 155.0 + i * 0.1,
                    'low': 145.0 + i * 0.1,
                    'close': 150.0 + i * 0.1,
                    'volume': 1000000.0
                })
        
        self.market_data = pd.DataFrame(data)
        self.config = TradingConfig(lookback_window=10)
        self.env = TradingEnvironment(
            self.market_data, self.config, symbols
        )
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        assert self.env.n_symbols == 2
        assert self.env.max_steps > 0
        assert self.env.action_space.shape[0] == 4  # 2 symbols * 2 actions each
    
    def test_reset_functionality(self):
        """Test environment reset."""
        obs, info = self.env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == self.env.observation_space.shape
        assert info['portfolio_value'] == self.config.initial_balance
    
    def test_step_functionality(self):
        """Test environment step."""
        self.env.reset(seed=42)
        
        # Create valid action
        action = np.array([1.0, 0.1, 0.0, 0.0])  # Buy AAPL, hold GOOGL
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_termination_conditions(self):
        """Test environment termination conditions."""
        self.env.reset(seed=42)
        
        # Simulate large loss to trigger termination
        self.env.portfolio_value = self.env.initial_balance * 0.05  # 95% loss
        
        action = np.array([0.0, 0.0, 0.0, 0.0])  # Hold all
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        assert terminated  # Should terminate due to large loss


class TestEnvironmentFactory:
    """Test environment factory functionality."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.market_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'AAPL',
            'open': np.random.uniform(140, 160, 50),
            'high': np.random.uniform(150, 170, 50),
            'low': np.random.uniform(130, 150, 50),
            'close': np.random.uniform(140, 160, 50),
            'volume': np.random.uniform(1000000, 5000000, 50)
        })
    
    def test_basic_environment_creation(self):
        """Test basic environment creation."""
        env = TradingEnvironmentFactory.create_basic_environment(
            self.market_data
        )
        
        assert isinstance(env, TradingEnvironment)
        assert env.config.initial_balance == 100000.0
    
    def test_high_frequency_environment(self):
        """Test high-frequency environment creation."""
        env = TradingEnvironmentFactory.create_high_frequency_environment(
            self.market_data
        )
        
        assert env.config.max_position_size == 0.1  # Smaller positions
        assert env.config.lookback_window == 10  # Shorter lookback
    
    def test_conservative_environment(self):
        """Test conservative environment creation."""
        env = TradingEnvironmentFactory.create_conservative_environment(
            self.market_data
        )
        
        assert env.config.max_position_size == 0.05  # Very small positions
        assert env.config.max_drawdown_limit == 0.1  # Strict drawdown limit


if __name__ == "__main__":
    pytest.main([__file__])