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

from src.services.trading_decision_engine import (
    TradingDecisionEngine,
    SignalCombiner,
    RiskCalculator,
    PositionSizer,
    SignalComponents,
    RiskMetrics,
    PositionSizingParams,
    RiskLevel
)
from src.models.trading_signal import TradingSignal, TradingAction
from src.models.portfolio import Portfolio, Position
from src.ml.hybrid_model import CNNLSTMHybridModel
from src.ml.rl_ensemble import EnsembleManager


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
        assert abs(self.combiner.base_cnn_lstm_weight + self.combiner.base_rl_weight - 1.0) < 1e-6
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
        # Mock CNN+LSTM model with more favorable responses
        self.mock_cnn_lstm = Mock(spec=CNNLSTMHybridModel)
        self.mock_cnn_lstm.is_trained = True
        self.mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.7, 0.2, 0.1]]),  # Stronger signal
            'ensemble_regression': np.array([[105.0]]),
            'regression_uncertainty': np.array([[0.08]])  # Lower uncertainty
        }
        
        # Mock RL ensemble with stronger signal
        self.mock_rl_ensemble = Mock(spec=EnsembleManager)
        self.mock_rl_ensemble.predict.return_value = np.array([0.3])  # Stronger RL signal
        self.mock_rl_ensemble.agents = [Mock(is_trained=True) for _ in range(3)]
        for agent in self.mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.3]), None)
        
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


class TestEnhancedTradingDecisionEngine:
    """Test enhanced trading decision engine functionality with CNN+LSTM integration."""
    
    def setup_method(self):
        """Set up test fixtures for enhanced engine."""
        # Mock enhanced CNN+LSTM model
        self.mock_cnn_lstm = Mock(spec=CNNLSTMHybridModel)
        self.mock_cnn_lstm.is_trained = True
        self.mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.6, 0.3, 0.1]]),
            'ensemble_regression': np.array([[105.0]]),
            'regression_uncertainty': np.array([[0.1]]),
            'ensemble_weights': np.array([0.2, 0.3, 0.3, 0.2])
        }
        
        # Mock enhanced RL ensemble
        self.mock_rl_ensemble = Mock(spec=EnsembleManager)
        self.mock_rl_ensemble.predict.return_value = np.array([0.2])
        self.mock_rl_ensemble.agents = [Mock(is_trained=True) for _ in range(4)]
        for agent in self.mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.2]), None)
        
        # Create enhanced engine
        self.engine = TradingDecisionEngine(
            cnn_lstm_model=self.mock_cnn_lstm,
            rl_ensemble=self.mock_rl_ensemble,
            model_version="enhanced_v2.0.0"
        )
    
    def test_enhanced_cnn_lstm_predictions(self):
        """Test enhanced CNN+LSTM prediction extraction."""
        market_data = np.random.randn(8, 30)
        
        predictions = self.engine._get_cnn_lstm_predictions(market_data)
        
        # Check enhanced prediction components
        assert 'classification_probs' in predictions
        assert 'classification_confidence' in predictions
        assert 'regression_pred' in predictions
        assert 'regression_uncertainty' in predictions
        assert 'confidence_interval_95' in predictions
        assert 'confidence_interval_68' in predictions
        assert 'enhanced_features' in predictions
        
        # Check enhanced features
        enhanced_features = predictions['enhanced_features']
        assert 'price_momentum' in enhanced_features
        assert 'volatility_regime' in enhanced_features
        assert 'trend_strength' in enhanced_features
        assert 'market_regime' in enhanced_features
    
    def test_market_regime_classification(self):
        """Test market regime classification."""
        # Test bullish regime
        bullish_probs = np.array([0.7, 0.2, 0.1])
        regime = self.engine._classify_market_regime(bullish_probs, 0.1)
        assert regime == "bullish"
        
        # Test bearish regime
        bearish_probs = np.array([0.1, 0.2, 0.7])
        regime = self.engine._classify_market_regime(bearish_probs, 0.1)
        assert regime == "bearish"
        
        # Test high uncertainty regime
        uncertain_probs = np.array([0.4, 0.3, 0.3])
        regime = self.engine._classify_market_regime(uncertain_probs, 0.4)
        assert regime == "high_uncertainty"
    
    def test_enhanced_rl_predictions(self):
        """Test enhanced RL predictions with CNN+LSTM integration."""
        market_data = np.random.randn(8, 30)
        cnn_lstm_features = {
            'enhanced_features': {
                'price_momentum': 0.05,
                'volatility_regime': 0.2,
                'trend_strength': 0.3,
                'market_regime': 'bullish'
            },
            'classification_confidence': 0.8,
            'regression_uncertainty': 0.1
        }
        
        predictions = self.engine._get_rl_predictions(
            market_data, None, cnn_lstm_features
        )
        
        # Check enhanced prediction components
        assert 'action' in predictions
        assert 'confidence' in predictions
        assert 'action_interpretation' in predictions
        assert 'ensemble_agreement' in predictions
        assert 'individual_actions' in predictions
    
    def test_rl_action_interpretation(self):
        """Test RL action interpretation."""
        # Test buy action
        buy_action = np.array([0.5])
        interpretation = self.engine._interpret_rl_action(buy_action, 0.8)
        assert interpretation['type'] == 'buy'
        assert interpretation['strength'] > 0
        
        # Test sell action
        sell_action = np.array([-0.5])
        interpretation = self.engine._interpret_rl_action(sell_action, 0.8)
        assert interpretation['type'] == 'sell'
        assert interpretation['strength'] > 0
        
        # Test hold action
        hold_action = np.array([0.05])
        interpretation = self.engine._interpret_rl_action(hold_action, 0.8)
        assert interpretation['type'] == 'hold'
    
    def test_enhanced_signal_combination(self):
        """Test enhanced signal combination with adaptive weighting."""
        # Create enhanced signal combiner
        combiner = SignalCombiner(adaptive_weighting=True)
        
        components = SignalComponents(
            cnn_lstm_classification=np.array([0.7, 0.2, 0.1]),
            cnn_lstm_regression=105.0,
            cnn_lstm_uncertainty=0.1,
            rl_action=np.array([0.3]),
            rl_confidence=0.8,
            market_features={
                'market_regime': 'bullish',
                'rl_ensemble_agreement': 0.9
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        result = combiner.combine_signals(components)
        
        # Check enhanced result components
        assert 'action' in result
        assert 'confidence' in result
        assert 'confidence_breakdown' in result
        assert 'adaptive_weights' in result
        assert 'market_regime' in result
        assert 'signal_strength' in result
        
        # Check confidence breakdown
        breakdown = result['confidence_breakdown']
        assert 'final_confidence' in breakdown
        assert 'uncertainty_penalty' in breakdown
        assert 'regime_confidence_multiplier' in breakdown
    
    def test_enhanced_position_sizing(self):
        """Test enhanced position sizing with CNN+LSTM insights."""
        risk_metrics = RiskMetrics(
            volatility=0.2,
            var_95=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.1,
            correlation_risk=0.05,
            liquidity_risk=0.02,
            concentration_risk=0.1,
            overall_risk_score=0.3
        )
        
        enhanced_signal_data = {
            'market_regime': 'bullish',
            'signal_strength': 0.4,
            'confidence_breakdown': {
                'uncertainty_penalty': 0.1
            },
            'adaptive_weights': {
                'rl_weight': 0.6
            }
        }
        
        size = self.engine.position_sizer.calculate_position_size(
            signal_confidence=0.8,
            risk_metrics=risk_metrics,
            portfolio=None,
            current_price=100.0,
            enhanced_signal_data=enhanced_signal_data
        )
        
        assert 0 <= size <= self.engine.position_sizer.params.max_position_size
        assert size > 0  # Should be positive for good signal
    
    def test_enhanced_stop_loss_calculation(self):
        """Test enhanced stop-loss calculation with CNN+LSTM confidence intervals."""
        cnn_lstm_predictions = {
            'confidence_interval_68': 2.0,  # $2 confidence interval
            'confidence_interval_95': 4.0,  # $4 confidence interval
            'regression_uncertainty': 0.15
        }
        
        combined_signal = {
            'confidence': 0.8,
            'signal_strength': 0.3
        }
        
        stop_loss, target_price = self.engine._calculate_stop_loss_take_profit(
            action=TradingAction.BUY,
            current_price=100.0,
            volatility=0.2,
            cnn_lstm_predictions=cnn_lstm_predictions,
            combined_signal=combined_signal
        )
        
        assert stop_loss is not None
        assert target_price is not None
        assert stop_loss < 100.0  # Stop loss below current price for BUY
        assert target_price > 100.0  # Target above current price for BUY
        
        # Check that confidence intervals influenced the calculation
        # (This is implicit in the calculation, hard to test directly)
    
    def test_enhanced_risk_filters(self):
        """Test enhanced risk filters with CNN+LSTM confidence intervals."""
        # Good signal that should pass
        good_signal = {
            'action': TradingAction.BUY,
            'confidence': 0.8,
            'confidence_breakdown': {
                'uncertainty_penalty': 0.1,
                'regime_confidence_multiplier': 0.9
            },
            'signal_strength': 0.2,
            'market_regime': 'bullish'
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
        
        good_cnn_lstm_predictions = {
            'regression_uncertainty': 0.1,
            'confidence_interval_95': 0.05
        }
        
        assert self.engine._passes_risk_filters(
            good_signal, good_risk_metrics, None, good_cnn_lstm_predictions
        )
        
        # Bad signal with high uncertainty that should fail
        bad_signal = {
            'action': TradingAction.BUY,
            'confidence': 0.8,
            'confidence_breakdown': {
                'uncertainty_penalty': 0.4,  # High uncertainty penalty
                'regime_confidence_multiplier': 0.9
            },
            'signal_strength': 0.2,
            'market_regime': 'high_uncertainty'
        }
        
        assert not self.engine._passes_risk_filters(
            bad_signal, good_risk_metrics, None, good_cnn_lstm_predictions
        )
    
    def test_enhanced_end_to_end_signal_generation(self):
        """Test complete enhanced signal generation workflow."""
        # Set up more favorable mock responses to ensure signal generation
        self.mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.8, 0.15, 0.05]]),  # Strong buy signal
            'ensemble_regression': np.array([[105.0]]),
            'regression_uncertainty': np.array([[0.08]]),  # Low uncertainty
            'ensemble_weights': np.array([0.25, 0.25, 0.25, 0.25])
        }
        
        self.mock_rl_ensemble.predict.return_value = np.array([0.4])  # Positive RL signal
        for agent in self.mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.4]), None)
        
        market_data = np.random.randn(8, 30)
        current_price = 100.0
        
        signal = self.engine.generate_signal(
            symbol="AAPL",
            market_data=market_data,
            current_price=current_price,
            market_features={'volume': 1000000, 'spread': 0.01}
        )
        
        # Should generate a signal with enhanced features
        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "AAPL"
        assert signal.model_version == "enhanced_v2.0.0"
        
        # Verify enhanced calculations were used
        # The signal should have reasonable confidence and position size
        assert signal.confidence > 0.5
        assert signal.position_size > 0.0


class TestEnhancedSignalAccuracy:
    """Test signal generation accuracy and risk controls."""
    
    def setup_method(self):
        """Set up test fixtures for accuracy testing."""
        # Create more realistic mock models
        self.mock_cnn_lstm = Mock(spec=CNNLSTMHybridModel)
        self.mock_cnn_lstm.is_trained = True
        
        self.mock_rl_ensemble = Mock(spec=EnsembleManager)
        self.mock_rl_ensemble.agents = [Mock(is_trained=True) for _ in range(3)]
        
        self.engine = TradingDecisionEngine(
            cnn_lstm_model=self.mock_cnn_lstm,
            rl_ensemble=self.mock_rl_ensemble,
            model_version="accuracy_test_v1.0.0"
        )
    
    def test_signal_consistency(self):
        """Test that similar inputs produce consistent signals."""
        # Set up consistent mock responses
        self.mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.7, 0.2, 0.1]]),
            'ensemble_regression': np.array([[105.0]]),
            'regression_uncertainty': np.array([[0.1]])
        }
        
        self.mock_rl_ensemble.predict.return_value = np.array([0.3])
        for agent in self.mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.3]), None)
        
        # Generate multiple signals with similar data
        signals = []
        for _ in range(5):
            market_data = np.random.randn(8, 30) * 0.1 + 1.0  # Similar data
            signal = self.engine.generate_signal(
                symbol="AAPL",
                market_data=market_data,
                current_price=100.0
            )
            if signal:
                signals.append(signal)
        
        # Check consistency
        if len(signals) > 1:
            actions = [s.action for s in signals]
            confidences = [s.confidence for s in signals]
            
            # Actions should be consistent
            assert len(set(actions)) <= 2  # Allow some variation
            
            # Confidences should be similar
            confidence_std = np.std(confidences)
            assert confidence_std < 0.2  # Not too much variation
    
    def test_risk_control_effectiveness(self):
        """Test that risk controls effectively filter risky signals."""
        # Set up high-risk scenario
        self.mock_cnn_lstm.predict.return_value = {
            'ensemble_classification': np.array([[0.4, 0.3, 0.3]]),  # Uncertain
            'ensemble_regression': np.array([[100.0]]),
            'regression_uncertainty': np.array([[0.5]])  # High uncertainty
        }
        
        self.mock_rl_ensemble.predict.return_value = np.array([0.1])  # Weak signal
        for agent in self.mock_rl_ensemble.agents:
            agent.predict.return_value = (np.array([0.1]), None)
        
        market_data = np.random.randn(8, 30)
        
        signal = self.engine.generate_signal(
            symbol="RISKY",
            market_data=market_data,
            current_price=100.0
        )
        
        # Should be filtered out or have very low position size
        if signal is not None:
            assert signal.position_size < 0.05  # Very small position
        # Or signal could be None (filtered out)
    
    def test_confidence_calibration(self):
        """Test that confidence scores are well-calibrated."""
        test_cases = [
            # (cnn_lstm_probs, cnn_lstm_uncertainty, rl_action, expected_confidence_range)
            ([0.9, 0.05, 0.05], 0.05, 0.8, (0.8, 1.0)),  # High confidence
            ([0.6, 0.3, 0.1], 0.15, 0.3, (0.5, 0.8)),    # Medium confidence
            ([0.4, 0.4, 0.2], 0.3, 0.1, (0.2, 0.6)),     # Low confidence
        ]
        
        for cnn_probs, uncertainty, rl_action, expected_range in test_cases:
            self.mock_cnn_lstm.predict.return_value = {
                'ensemble_classification': np.array([cnn_probs]),
                'ensemble_regression': np.array([[100.0]]),
                'regression_uncertainty': np.array([[uncertainty]])
            }
            
            self.mock_rl_ensemble.predict.return_value = np.array([rl_action])
            for agent in self.mock_rl_ensemble.agents:
                agent.predict.return_value = (np.array([rl_action]), None)
            
            market_data = np.random.randn(8, 30)
            signal = self.engine.generate_signal(
                symbol="TEST",
                market_data=market_data,
                current_price=100.0
            )
            
            if signal is not None:
                min_conf, max_conf = expected_range
                assert min_conf <= signal.confidence <= max_conf, \
                    f"Confidence {signal.confidence} not in expected range {expected_range}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])