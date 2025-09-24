"""Unit tests for trading environment improvements."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.ml.yfinance_trading_environment import YFinanceTradingEnvironment, YFinanceConfig


class TestActionParsing:
    """Test improved action parsing logic."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = YFinanceConfig(
            initial_balance=100000.0,
            max_position_size=0.1,
            transaction_cost=0.001,
            slippage_factor=0.0005
        )
        
        # Mock the environment creation to avoid data loading
        with patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._load_and_prepare_data'):
            self.env = YFinanceTradingEnvironment(
                symbols=['AAPL', 'GOOGL'],
                start_date="2022-01-01",
                end_date="2023-01-01",
                config=self.config
            )
            # Set up minimal required attributes
            self.env.symbols = ['AAPL', 'GOOGL']
            self.env.processed_data = MagicMock()
            self.env.max_steps = 100
    
    def test_action_parsing_buy_signal(self):
        """Test that strong positive actions are interpreted as BUY."""
        # Action array: [action_type, position_size] for each symbol
        action = np.array([0.8, 0.05, -0.2, 0.03])  # BUY AAPL 5%, HOLD GOOGL
        
        parsed_actions = self.env._parse_action(action)
        
        assert len(parsed_actions) == 2
        assert parsed_actions[0]['action_type'] == 1  # BUY
        assert parsed_actions[0]['position_size'] == 0.05
        assert parsed_actions[1]['action_type'] == 0  # HOLD (weak signal)
    
    def test_action_parsing_sell_signal(self):
        """Test that strong negative actions are interpreted as SELL."""
        action = np.array([-0.8, 0.05, 0.5, 0.03])  # SELL AAPL, BUY GOOGL
        
        parsed_actions = self.env._parse_action(action)
        
        assert parsed_actions[0]['action_type'] == 2  # SELL
        assert parsed_actions[1]['action_type'] == 1  # BUY
    
    def test_action_parsing_hold_signal(self):
        """Test that weak signals are interpreted as HOLD."""
        action = np.array([0.1, 0.05, -0.2, 0.03])  # Weak signals
        
        parsed_actions = self.env._parse_action(action)
        
        assert parsed_actions[0]['action_type'] == 0  # HOLD
        assert parsed_actions[1]['action_type'] == 0  # HOLD
    
    def test_action_parsing_small_position_converted_to_hold(self):
        """Test that small position sizes are converted to HOLD."""
        action = np.array([0.8, 0.005, 0.8, 0.02])  # Strong signal but tiny position
        
        parsed_actions = self.env._parse_action(action)
        
        assert parsed_actions[0]['action_type'] == 0  # Converted to HOLD (position too small)
        assert parsed_actions[1]['action_type'] == 1  # BUY (position size OK)
    
    def test_position_size_clipping(self):
        """Test that position sizes are properly clipped."""
        action = np.array([0.8, 0.5, 0.8, -0.1])  # Large and negative position sizes
        
        parsed_actions = self.env._parse_action(action)
        
        assert parsed_actions[0]['position_size'] == 0.1  # Clipped to max
        assert parsed_actions[1]['position_size'] == 0.0  # Clipped to min


class TestRewardCalculation:
    """Test improved reward calculation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = YFinanceConfig(initial_balance=100000.0)
        
        with patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._load_and_prepare_data'):
            self.env = YFinanceTradingEnvironment(
                symbols=['AAPL'],
                start_date="2022-01-01",
                end_date="2023-01-01",
                config=self.config
            )
            # Setup required attributes
            self.env.portfolio_history = [100000.0, 101000.0]  # 1% gain
            self.env.portfolio_value = 101000.0
            self.env.max_portfolio_value = 101000.0
            self.env.returns_history = []
    
    def test_positive_portfolio_return_reward(self):
        """Test that positive portfolio returns generate positive rewards."""
        trade_results = [{'executed': False}]  # No trades
        
        reward = self.env._calculate_reward(trade_results, None)
        
        assert reward > 0, "Positive portfolio return should generate positive reward"
        # 1% return * 100 scaling = 1.0 base reward
        assert abs(reward - 1.0) < 0.1, f"Expected ~1.0 reward, got {reward}"
    
    def test_zero_portfolio_return_gets_exploration_bonus(self):
        """Test that zero returns still get small exploration bonus."""
        self.env.portfolio_history = [100000.0]  # No previous value
        self.env.portfolio_value = 100000.0  # No change
        
        trade_results = [{'executed': False}]
        
        reward = self.env._calculate_reward(trade_results, None)
        
        assert reward > 0, "Should get exploration bonus even with zero return"
        assert reward == 0.001, f"Expected 0.001 exploration bonus, got {reward}"
    
    def test_action_bonus_for_executed_trades(self):
        """Test that executed trades get action bonus."""
        self.env.portfolio_history = [100000.0]
        self.env.portfolio_value = 100000.0
        
        trade_results = [{'executed': True}]  # Trade executed
        
        reward = self.env._calculate_reward(trade_results, None)
        
        # Should get base exploration bonus + action bonus
        expected_reward = 0.001 + 0.001  # exploration + action bonus
        assert abs(reward - expected_reward) < 1e-6, f"Expected {expected_reward}, got {reward}"
    
    def test_negative_portfolio_return_penalty(self):
        """Test that negative returns generate negative rewards."""
        self.env.portfolio_history = [100000.0, 99000.0]  # 1% loss
        self.env.portfolio_value = 99000.0
        
        trade_results = [{'executed': False}]
        
        reward = self.env._calculate_reward(trade_results, None)
        
        assert reward < 0, "Negative portfolio return should generate negative reward"


class TestTradeExecution:
    """Test trade execution improvements."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = YFinanceConfig(
            initial_balance=100000.0,
            max_position_size=0.1,
            transaction_cost=0.001
        )
        
        with patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._load_and_prepare_data'):
            self.env = YFinanceTradingEnvironment(
                symbols=['AAPL'],
                start_date="2022-01-01",
                end_date="2023-01-01",
                config=self.config
            )
            # Setup required attributes
            self.env.symbols = ['AAPL']
            self.env.cash_balance = 100000.0
            self.env.positions = {'AAPL': 0.0}
            self.env.portfolio_value = 100000.0
            self.env.position_ages = {'AAPL': 0}
    
    @patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._get_current_prices')
    @patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._calculate_current_exposure')
    def test_buy_order_execution(self, mock_exposure, mock_prices):
        """Test that valid buy orders are executed."""
        mock_prices.return_value = {'AAPL': 150.0}
        mock_exposure.return_value = 0.0  # No current exposure
        
        actions = [{
            'symbol': 'AAPL',
            'action_type': 1,  # BUY
            'position_size': 0.05  # 5% position
        }]
        
        results = self.env._execute_trades(actions)
        
        assert len(results) == 1
        result = results[0]
        assert result['executed'] is True
        assert result['quantity'] > 0
        assert self.env.cash_balance < 100000.0  # Cash should be reduced
        assert self.env.positions['AAPL'] > 0  # Should have position
    
    @patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._get_current_prices')
    @patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._calculate_current_exposure')
    def test_minimum_trade_size_rejection(self, mock_exposure, mock_prices):
        """Test that trades below minimum size are rejected."""
        mock_prices.return_value = {'AAPL': 150.0}
        mock_exposure.return_value = 0.0
        
        actions = [{
            'symbol': 'AAPL',
            'action_type': 1,  # BUY
            'position_size': 0.0003  # Very small position (30 dollars)
        }]
        
        results = self.env._execute_trades(actions)
        
        assert len(results) == 1
        result = results[0]
        assert result['executed'] is False
        assert result['reason'] == 'min_trade_size'
        assert self.env.cash_balance == 100000.0  # Cash unchanged
    
    def test_hold_action_no_execution(self):
        """Test that HOLD actions don't execute trades."""
        actions = [{
            'symbol': 'AAPL',
            'action_type': 0,  # HOLD
            'position_size': 0.05
        }]
        
        results = self.env._execute_trades(actions)
        
        assert len(results) == 1
        result = results[0]
        assert result['executed'] is False
        assert result['reason'] == 'hold'


class TestEnvironmentIntegration:
    """Integration tests for environment improvements."""
    
    @patch('src.ml.yfinance_trading_environment.YFinanceTradingEnvironment._load_and_prepare_data')
    def test_step_function_with_meaningful_actions(self, mock_load_data):
        """Test that step function processes meaningful actions correctly."""
        config = YFinanceConfig(initial_balance=100000.0)
        env = YFinanceTradingEnvironment(
            symbols=['AAPL', 'GOOGL'],
            start_date="2022-01-01",
            end_date="2023-01-01",
            config=config
        )
        
        # Setup minimal environment state
        env.symbols = ['AAPL', 'GOOGL']
        env.current_step = 0
        env.max_steps = 100
        env.cash_balance = 100000.0
        env.positions = {'AAPL': 0.0, 'GOOGL': 0.0}
        env.portfolio_value = 100000.0
        env.max_portfolio_value = 100000.0
        env.portfolio_history = []
        env.returns_history = []
        env.drawdown_history = []
        env.regime_history = []
        env.trade_history = []
        env.position_ages = {'AAPL': 0, 'GOOGL': 0}
        
        # Mock required methods
        with patch.object(env, '_get_observation', return_value=np.zeros(10)):
            with patch.object(env, '_get_info', return_value={}):
                with patch.object(env, '_get_current_prices', return_value={'AAPL': 150.0, 'GOOGL': 2500.0}):
                    with patch.object(env, '_calculate_current_exposure', return_value=0.0):
                        with patch.object(env, '_detect_market_regime', return_value=None):
                            with patch.object(env, '_update_portfolio_value'):
                                
                                # Test meaningful buy action
                                action = np.array([0.8, 0.05, -0.1, 0.02])  # BUY AAPL 5%, HOLD GOOGL
                                
                                obs, reward, terminated, truncated, info = env.step(action)
                                
                                # Should get some reward (at least exploration bonus)
                                assert reward >= 0.001, f"Expected positive reward, got {reward}"
                                assert not terminated
                                assert not truncated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])