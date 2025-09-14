"""
Tests for Trading Decision Engine

This module tests the trading decision engine's signal generation,
risk management, and position sizing capabilities.
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch

import pytest
import numpy as np
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.trading_decision_engine import (
    TradingDecisionEngine,
    SignalCombiner,
    RiskCalculator,
    PositionSizer,
    SignalComponents,
    RiskMetrics,
    PositionSizingParams,
    RiskLevel
)
from models.trading_signal import TradingSignal, TradingAction
from models.portfolio import Portfolio, Position
from ml.hybrid_model import CNNLSTMHybridModel
from ml.rl_ensemble import EnsembleManager


class TestSignalComponents:
    """Test signal components data structure."""
    
    def test_signal_components_creation(self):
        """Test creating signal components."""
        components = SignalComponents(
            cnn_lstm_classification=np.array([0.6, 0.3, 0.1]),
            cnn_lstm_regression=105.5,
            cnn_lstm_uncertainty=0.15,
            rl_action=np.array([0.2]),
            rl_confidence=0.8,
            market_features={'volatility': 0.25, 'volume': 1000000},
            timestamp=datetime.now(timezone.utc)
        )
        
        assert len(components.cnn_lstm_classification) == 3
        assert components.cnn_lstm_regression == 105.5
        assert components.cnn_lstm_uncertainty == 0.15
        assert components.rl_confidence == 0.8
        assert 'volatility' in components.market_features


class TestRiskMetrics:
    """Test risk metrics calculations."""
    
    def test_risk_metrics_creation(self):
        """Test creating risk metrics."""
        metrics = RiskMetrics(
            volatility=0.25,
            var_95=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            correlation_risk=0.1,
            liquidity_risk=0.05,
            concentration_risk=0.2,
            overall_risk_score=0.3
        )
        
        assert metrics.volatility == 0.25
        assert metrics.var_95 == 0.05
        assert metrics.sharpe_ratio == 1.2
        assert metrics.overall_risk_score == 0.3


class TestPositionSizingParams:
    """Test position sizing parameters."""
    
    def test_default_params(self):
        """Test default position sizing parameters."""
        params = PositionSizingParams()
        
        assert params.max_position_size == 0.2
        assert params.risk_per_trade == 0.02
        assert params.confidence_threshold == 0.6
        assert params.kelly_fraction == 0.25
        assert params.max_leverage == 1.0


class TestSignalCombiner:
    """Test signal combination logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.combiner = SignalCombiner(
            cnn_lstm_weight=0.6,
            rl_weight=0.4,
            uncertainty_penalty=0.1
        )
    
    def test_combiner_initialization(self):
        """Test signal combiner initialization."""
        assert abs(self.combiner.cnn_lstm_weight + self.combiner.rl_weight - 1.0) < 1e-6
        assert self.combiner.uncertainty_penalty == 0.1
    
    def test_combine_buy_signal(self):
        """Test combining signals for BUY decision."""
        components = SignalComponents(
            cnn_lstm_classification=np.array([0.8, 0.15, 0.05]),  # Strong buy
            cnn_lstm_regression=105.0,
            cnn_lstm_uncertainty=0.1,
            rl_action=np.array([0.5]),  # Positive RL action (buy)
            rl_confidence=0.9,
            market_features={},
            timestamp=datetime.now(timezone.utc)
        )
        
        result = self.combiner.combine_signals(components)
        
        assert result['action'] == TradingAction.BUY
        assert result['confidence'] > 0.5
        assert 'probabilities' in result
        assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-5
    
    def test_combine_sell_signal(self):
        """Test combining signals for SELL decision."""
        components = SignalComponents(
            cnn_lstm_classification=np.array([0.1, 0.2, 0.7]),  # Strong sell
            cnn_lstm_regression=95.0,
            cnn_lstm_uncertainty=0.05,
            rl_action=np.array([-0.5]),  # Negative RL action (sell)
            rl_confidence=0.85,
            market_features={},
            timestamp=datetime.now(timezone.utc)
        )
        
        result = self.combiner.combine_signals(components)
        
        assert result['action'] == TradingAction.SELL
        assert result['confidence'] > 0.5
    
    def test_combine_hold_signal(self):
        """Test combining signals for HOLD decision."""
        components = SignalComponents(
            cnn_lstm_classification=np.array([0.3, 0.5, 0.2]),  # Hold signal
            cnn_lstm_regression=100.0,
            cnn_lstm_uncertainty=0.2,
            rl_action=np.array([0.05]),  # Neutral RL action
            rl_confidence=0.6,
            market_features={},
            timestamp=datetime.now(timezone.utc)
        )
        
        result = self.combiner.combine_signals(components)
        
        assert result['action'] == TradingAction.HOLD
    
    def test_uncertainty_penalty(self):
        """Test that high uncertainty reduces confidence."""
        # Low uncertainty case
        low_uncertainty_components = SignalComponents(
            cnn_lstm_classification=np.array([0.8, 0.15, 0.05]),
            cnn_lstm_regression=105.0,
            cnn_lstm_uncertainty=0.05,  # Low uncertainty
            rl_action=np.array([0.5]),
            rl_confidence=0.9,
            market_features={},
            timestamp=datetime.now(timezone.utc)
        )
        
        # High uncertainty case
        high_uncertainty_components = SignalComponents(
            cnn_lstm_classification=np.array([0.8, 0.15, 0.05]),
            cnn_lstm_regression=105.0,
            cnn_lstm_uncertainty=0.5,  # High uncertainty
            rl_action=np.array([0.5]),
            rl_confidence=0.9,
            market_features={},
            timestamp=datetime.now(timezone.utc)
        )
        
        low_result = self.combiner.combine_signals(low_uncertainty_components)
        high_result = self.combiner.combine_signals(high_uncertainty_components)
        
        assert low_result['confidence'] > high_result['confidence']


class TestRiskCalculator:
    """Test risk calculation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator(risk_free_rate=0.02)
    
    def test_risk_calculator_initialization(self):
        """Test risk calculator initialization."""
        assert self.calculator.risk_free_rate == 0.02
    
    def test_calculate_risk_metrics_insufficient_data(self):
        """Test risk calculation with insufficient data."""
        metrics = self.calculator.calculate_risk_metrics(
            symbol="AAPL",
            current_price=150.0,
            price_history=[]
        )
        
        # Should return default metrics
        assert metrics.volatility == 0.2
        assert metrics.var_95 == 0.05
        assert metrics.sharpe_ratio == 0.0
        assert metrics.overall_risk_score == 0.3
    
    def test_calculate_risk_metrics_with_data(self):
        """Test risk calculation with sufficient data."""
        # Generate price history with some volatility
        base_price = 100.0
        price_history = []
        for i in range(50):
            # Add some random walk
            price = base_price + np.random.normal(0, 2) + i * 0.1
            price_history.append(max(price, 1.0))  # Ensure positive prices
        
        metrics = self.calculator.calculate_risk_metrics(
            symbol="AAPL",
            current_price=price_history[-1] + 1.0,
            price_history=price_history
        )
        
        assert metrics.volatility > 0
        assert metrics.var_95 >= 0
        assert metrics.max_drawdown >= 0
        assert 0 <= metrics.overall_risk_score <= 1


class TestPositionSizer:
    """Test position sizing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = PositionSizingParams(
            max_position_size=0.2,
            confidence_threshold=0.6,
            kelly_fraction=0.25
        )
        self.sizer = PositionSizer(self.params)
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization."""
        assert self.sizer.params.max_position_size == 0.2
        assert self.sizer.params.confidence_threshold == 0.6
    
    def test_calculate_position_size_high_confidence(self):
        """Test position sizing with high confidence."""
        risk_metrics = RiskMetrics(
            volatility=0.15,
            var_95=0.03,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            correlation_risk=0.05,
            liquidity_risk=0.02,
            concentration_risk=0.1,
            overall_risk_score=0.2
        )
        
        size = self.sizer.calculate_position_size(
            signal_confidence=0.9,
            risk_metrics=risk_metrics,
            portfolio=None,
            current_price=100.0
        )
        
        assert 0 <= size <= self.params.max_position_size
        assert size > 0  # Should be positive for high confidence
    
    def test_calculate_position_size_low_confidence(self):
        """Test position sizing with low confidence."""
        risk_metrics = RiskMetrics(
            volatility=0.3,
            var_95=0.08,
            sharpe_ratio=0.5,
            max_drawdown=0.2,
            correlation_risk=0.1,
            liquidity_risk=0.05,
            concentration_risk=0.15,
            overall_risk_score=0.4
        )
        
        size = self.sizer.calculate_position_size(
            signal_confidence=0.4,  # Below threshold
            risk_metrics=risk_metrics,
            portfolio=None,
            current_price=100.0
        )
        
        assert size == 0.0  # Should be zero for low confidence


class TestTradingDecisionEngine:
    """Test trading decision engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock CNN+LSTM model
        self.mock_cnn_lstm = Mock(spec=CNNLSTMHybridModel)
        self.mock_cnn_lstm.is_trained = True
        self.mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.6, 0.3, 0.1]]),
            'ensemble_regression': np.array([[105.0]]),
            'regression_uncertainty': np.array([[0.1]])
        }
        
        # Mock RL ensemble
        self.mock_rl_ensemble = Mock(spec=EnsembleManager)
        self.mock_rl_ensemble.predict.return_value = np.array([0.2])
        self.mock_rl_ensemble.agents = [Mock(is_trained=True) for _ in range(3)]
        for agent in self.mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.2]), None)
        
        # Create engine
        self.engine = TradingDecisionEngine(
            cnn_lstm_model=self.mock_cnn_lstm,
            rl_ensemble=self.mock_rl_ensemble,
            model_version="test_v1.0.0"
        )
    
    def test_engine_initialization(self):
        """Test trading decision engine initialization."""
        assert self.engine.cnn_lstm_model == self.mock_cnn_lstm
        assert self.engine.rl_ensemble == self.mock_rl_ensemble
        assert self.engine.model_version == "test_v1.0.0"
        assert self.engine.enable_stop_loss is True
    
    def test_generate_signal_success(self):
        """Test successful signal generation."""
        # Prepare test data
        market_data = np.random.randn(8, 30)  # 8 features, 30 timesteps
        current_price = 100.0
        
        # Generate signal
        signal = self.engine.generate_signal(
            symbol="AAPL",
            market_data=market_data,
            current_price=current_price
        )
        
        # Verify signal
        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "AAPL"
        assert signal.action in [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD]
        assert 0 <= signal.confidence <= 1
        assert 0 <= signal.position_size <= 1
        assert signal.model_version == "test_v1.0.0"
    
    def test_generate_signal_with_portfolio(self):
        """Test signal generation with portfolio context."""
        # Create test portfolio
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_cost=95.0,
            current_price=100.0,
            unrealized_pnl=500.0
        )
        
        portfolio = Portfolio(
            user_id="test_user",
            positions={"AAPL": position},
            cash_balance=10000.0,
            total_value=20000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        market_data = np.random.randn(8, 30)
        
        signal = self.engine.generate_signal(
            symbol="AAPL",
            market_data=market_data,
            current_price=100.0,
            portfolio=portfolio
        )
        
        assert signal is not None
        assert isinstance(signal, TradingSignal)
    
    def test_generate_signal_untrained_model(self):
        """Test signal generation with untrained model."""
        # Set model as untrained
        self.mock_cnn_lstm.is_trained = False
        
        market_data = np.random.randn(8, 30)
        
        signal = self.engine.generate_signal(
            symbol="AAPL",
            market_data=market_data,
            current_price=100.0
        )
        
        # Should return None for untrained model
        assert signal is None
    
    def test_stop_loss_calculation_buy_signal(self):
        """Test stop-loss calculation for BUY signals."""
        stop_loss, target_price = self.engine._calculate_stop_loss_take_profit(
            action=TradingAction.BUY,
            current_price=100.0,
            volatility=0.2
        )
        
        assert stop_loss is not None
        assert target_price is not None
        assert stop_loss < 100.0  # Stop loss below current price for BUY
        assert target_price > 100.0  # Target above current price for BUY
    
    def test_stop_loss_calculation_sell_signal(self):
        """Test stop-loss calculation for SELL signals."""
        stop_loss, target_price = self.engine._calculate_stop_loss_take_profit(
            action=TradingAction.SELL,
            current_price=100.0,
            volatility=0.2
        )
        
        assert stop_loss is not None
        assert target_price is not None
        assert stop_loss > 100.0  # Stop loss above current price for SELL
        assert target_price < 100.0  # Target below current price for SELL
    
    def test_stop_loss_calculation_hold_signal(self):
        """Test stop-loss calculation for HOLD signals."""
        stop_loss, target_price = self.engine._calculate_stop_loss_take_profit(
            action=TradingAction.HOLD,
            current_price=100.0,
            volatility=0.2
        )
        
        assert stop_loss is None
        assert target_price is None
    
    def test_model_status(self):
        """Test model status reporting."""
        status = self.engine.get_model_status()
        
        assert 'cnn_lstm_trained' in status
        assert 'rl_ensemble_agents' in status
        assert 'rl_trained_agents' in status
        assert 'model_version' in status
        assert 'symbols_tracked' in status
        
        assert status['cnn_lstm_trained'] is True
        assert status['rl_ensemble_agents'] == 3
        assert status['model_version'] == "test_v1.0.0"
    
    def test_update_models(self):
        """Test model updating functionality."""
        new_cnn_lstm = Mock(spec=CNNLSTMHybridModel)
        new_rl_ensemble = Mock(spec=EnsembleManager)
        
        self.engine.update_models(
            cnn_lstm_model=new_cnn_lstm,
            rl_ensemble=new_rl_ensemble
        )
        
        assert self.engine.cnn_lstm_model == new_cnn_lstm
        assert self.engine.rl_ensemble == new_rl_ensemble
    
    def test_price_history_management(self):
        """Test price history tracking."""
        symbol = "AAPL"
        
        # Add prices to history
        for price in [100.0, 101.0, 99.5, 102.0]:
            self.engine._update_price_history(symbol, price)
        
        assert symbol in self.engine.price_history
        assert len(self.engine.price_history[symbol]) == 4
        assert self.engine.price_history[symbol][-1] == 102.0
    
    def test_risk_filters(self):
        """Test risk management filters."""
        # Test signal that should pass filters
        good_signal = {
            'action': TradingAction.BUY,
            'confidence': 0.8
        }
        
        good_risk_metrics = RiskMetrics(
            volatility=0.15,
            var_95=0.03,
            sharpe_ratio=1.2,
            max_drawdown=0.1,
            correlation_risk=0.05,
            liquidity_risk=0.02,
            concentration_risk=0.1,
            overall_risk_score=0.3
        )
        
        assert self.engine._passes_risk_filters(good_signal, good_risk_metrics, None)
        
        # Test signal that should fail filters (low confidence)
        bad_signal = {
            'action': TradingAction.BUY,
            'confidence': 0.4  # Below threshold
        }
        
        assert not self.engine._passes_risk_filters(bad_signal, good_risk_metrics, None)


class TestTradingDecisionEngineIntegration:
    """Integration tests for trading decision engine."""
    
    @patch('src.ml.hybrid_model.CNNLSTMHybridModel')
    @patch('src.ml.rl_ensemble.EnsembleManager')
    def test_end_to_end_signal_generation(self, mock_rl_class, mock_cnn_class):
        """Test complete end-to-end signal generation workflow."""
        # Set up mocks
        mock_cnn_lstm = mock_cnn_class.return_value
        mock_cnn_lstm.is_trained = True
        mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.7, 0.2, 0.1]]),
            'ensemble_regression': np.array([[105.0]]),
            'regression_uncertainty': np.array([[0.1]])
        }
        
        mock_rl_ensemble = mock_rl_class.return_value
        mock_rl_ensemble.predict.return_value = np.array([0.3])
        mock_rl_ensemble.agents = [Mock(is_trained=True) for _ in range(3)]
        for agent in mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.3]), None)
        
        # Create engine
        engine = TradingDecisionEngine(
            cnn_lstm_model=mock_cnn_lstm,
            rl_ensemble=mock_rl_ensemble,
            model_version="integration_test_v1.0.0"
        )
        
        # Generate multiple signals
        symbols = ["AAPL", "GOOGL", "MSFT"]
        signals = []
        
        for symbol in symbols:
            market_data = np.random.randn(8, 30)
            current_price = 100.0 + np.random.randn() * 10
            
            signal = engine.generate_signal(
                symbol=symbol,
                market_data=market_data,
                current_price=current_price
            )
            
            if signal:
                signals.append(signal)
        
        # Verify signals
        assert len(signals) > 0
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol in symbols
            assert 0 <= signal.confidence <= 1
            assert 0 <= signal.position_size <= 1
            assert signal.model_version == "integration_test_v1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])