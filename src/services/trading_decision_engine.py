"""
Trading Decision Engine

This module implements the core trading decision engine that combines
CNN+LSTM model predictions with RL ensemble outputs to generate
trading signals with confidence scoring, position sizing, and risk management.

Requirements: 2.1, 2.3, 2.7
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from ..models.trading_signal import TradingSignal, TradingAction
from ..models.market_data import MarketData
from ..models.portfolio import Portfolio, Position
from ..ml.hybrid_model import CNNLSTMHybridModel
from ..ml.rl_ensemble import EnsembleManager


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SignalComponents:
    """Components that make up a trading signal."""
    cnn_lstm_classification: np.ndarray  # [buy_prob, hold_prob, sell_prob]
    cnn_lstm_regression: float  # Price prediction
    cnn_lstm_uncertainty: float  # Prediction uncertainty
    rl_action: np.ndarray  # RL ensemble action
    rl_confidence: float  # RL confidence score
    market_features: Dict[str, float]  # Additional market features
    timestamp: datetime


@dataclass
class RiskMetrics:
    """Risk assessment metrics for position sizing."""
    volatility: float
    var_95: float  # Value at Risk (95% confidence)
    sharpe_ratio: float
    max_drawdown: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    overall_risk_score: float


@dataclass
class PositionSizingParams:
    """Parameters for position sizing calculation."""
    max_position_size: float = 0.2  # Maximum 20% of portfolio per position
    risk_per_trade: float = 0.02  # Risk 2% of portfolio per trade
    volatility_lookback: int = 20  # Days for volatility calculation
    confidence_threshold: float = 0.6  # Minimum confidence for trading
    kelly_fraction: float = 0.25  # Kelly criterion fraction
    max_leverage: float = 1.0  # Maximum leverage allowed


class TradingDecisionEngine:
    """
    Core trading decision engine that combines ML models with risk management.
    
    This engine integrates CNN+LSTM predictions with RL ensemble decisions
    to generate trading signals with proper risk management and position sizing.
    """
    
    def __init__(
        self,
        cnn_lstm_model: CNNLSTMHybridModel,
        rl_ensemble: EnsembleManager,
        position_sizing_params: Optional[PositionSizingParams] = None,
        risk_free_rate: float = 0.02,
        enable_stop_loss: bool = True,
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.15,  # 15% take profit
        model_version: str = "v1.0.0"
    ):
        """
        Initialize trading decision engine.
        
        Args:
            cnn_lstm_model: Trained CNN+LSTM hybrid model
            rl_ensemble: Trained RL ensemble manager
            position_sizing_params: Position sizing configuration
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            enable_stop_loss: Whether to enable stop-loss mechanisms
            stop_loss_pct: Stop-loss percentage (e.g., 0.05 = 5%)
            take_profit_pct: Take-profit percentage
            model_version: Version identifier for generated signals
        """
        self.cnn_lstm_model = cnn_lstm_model
        self.rl_ensemble = rl_ensemble
        self.position_sizing_params = position_sizing_params or PositionSizingParams()
        self.risk_free_rate = risk_free_rate
        self.enable_stop_loss = enable_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.model_version = model_version
        
        # Risk management components
        self.risk_calculator = RiskCalculator(risk_free_rate)
        self.position_sizer = PositionSizer(self.position_sizing_params)
        self.signal_combiner = SignalCombiner()
        
        # Historical data for risk calculations
        self.price_history: Dict[str, List[float]] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(
        self,
        symbol: str,
        market_data: np.ndarray,
        current_price: float,
        portfolio: Optional[Portfolio] = None,
        market_features: Optional[Dict[str, float]] = None
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal for a given symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Preprocessed market data for ML models
            current_price: Current market price
            portfolio: Current portfolio state
            market_features: Additional market features
            
        Returns:
            Trading signal or None if no actionable signal
        """
        try:
            # Get predictions from CNN+LSTM model
            cnn_lstm_predictions = self._get_cnn_lstm_predictions(market_data)
            
            # Get predictions from RL ensemble
            rl_predictions = self._get_rl_predictions(market_data, portfolio)
            
            # Combine signal components
            signal_components = SignalComponents(
                cnn_lstm_classification=cnn_lstm_predictions['classification_probs'],
                cnn_lstm_regression=cnn_lstm_predictions['regression_pred'],
                cnn_lstm_uncertainty=cnn_lstm_predictions.get('regression_uncertainty', 0.0),
                rl_action=rl_predictions['action'],
                rl_confidence=rl_predictions['confidence'],
                market_features=market_features or {},
                timestamp=datetime.now(timezone.utc)
            )
            
            # Combine signals to get final decision
            combined_signal = self.signal_combiner.combine_signals(signal_components)
            
            # Calculate risk metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                symbol, current_price, self.price_history.get(symbol, [])
            )
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                signal_confidence=combined_signal['confidence'],
                risk_metrics=risk_metrics,
                portfolio=portfolio,
                current_price=current_price
            )
            
            # Apply risk management filters
            if not self._passes_risk_filters(combined_signal, risk_metrics, portfolio):
                self.logger.info(f"Signal for {symbol} filtered out by risk management")
                return None
            
            # Calculate stop-loss and take-profit levels
            stop_loss, target_price = self._calculate_stop_loss_take_profit(
                action=combined_signal['action'],
                current_price=current_price,
                volatility=risk_metrics.volatility
            )
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                action=combined_signal['action'],
                confidence=combined_signal['confidence'],
                position_size=position_size,
                target_price=target_price,
                stop_loss=stop_loss,
                timestamp=datetime.now(timezone.utc),
                model_version=self.model_version
            )
            
            # Update price history
            self._update_price_history(symbol, current_price)
            
            self.logger.info(f"Generated signal for {symbol}: {signal.action} "
                           f"(confidence: {signal.confidence:.3f}, "
                           f"position_size: {signal.position_size:.3f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _get_cnn_lstm_predictions(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Get predictions from CNN+LSTM model."""
        if not self.cnn_lstm_model.is_trained:
            raise ValueError("CNN+LSTM model is not trained")
        
        # Ensure correct input shape: (batch_size, input_channels, sequence_length)
        if len(market_data.shape) == 2:
            market_data = market_data.reshape(1, market_data.shape[0], market_data.shape[1])
        elif len(market_data.shape) == 3 and market_data.shape[0] != 1:
            market_data = market_data[:1]  # Take first sample if batch
        
        predictions = self.cnn_lstm_model.predict(
            market_data,
            return_uncertainty=True,
            use_ensemble=True
        )
        
        return {
            'classification_probs': predictions['ensemble_classification'][0],  # Remove batch dim
            'regression_pred': predictions['ensemble_regression'][0, 0],  # Remove batch and feature dims
            'regression_uncertainty': predictions['regression_uncertainty'][0, 0] if 'regression_uncertainty' in predictions else 0.0
        }
    
    def _get_rl_predictions(
        self,
        market_data: np.ndarray,
        portfolio: Optional[Portfolio]
    ) -> Dict[str, Any]:
        """Get predictions from RL ensemble."""
        # Prepare observation for RL (flatten market data and add portfolio state)
        market_features = market_data.flatten()
        
        # Add portfolio features if available
        if portfolio is not None:
            portfolio_features = np.array([
                portfolio.total_value,
                portfolio.cash_balance,
                len(portfolio.positions),
                sum(pos.unrealized_pnl for pos in portfolio.positions.values())
            ])
            observation = np.concatenate([market_features, portfolio_features])
        else:
            observation = market_features
        
        # Get ensemble prediction
        action = self.rl_ensemble.predict(observation, deterministic=True)
        
        # Calculate confidence based on ensemble agreement
        individual_actions = []
        for agent in self.rl_ensemble.agents:
            if agent.is_trained:
                individual_action, _ = agent.predict(observation, deterministic=True)
                individual_actions.append(individual_action)
        
        if individual_actions:
            # Calculate confidence as inverse of action variance
            action_variance = np.var(individual_actions, axis=0)
            confidence = 1.0 / (1.0 + np.mean(action_variance))
        else:
            confidence = 0.5  # Default confidence for untrained ensemble
        
        return {
            'action': action,
            'confidence': confidence
        }
    
    def _passes_risk_filters(
        self,
        signal: Dict[str, Any],
        risk_metrics: RiskMetrics,
        portfolio: Optional[Portfolio]
    ) -> bool:
        """Apply risk management filters to trading signals."""
        
        # Minimum confidence filter
        if signal['confidence'] < self.position_sizing_params.confidence_threshold:
            return False
        
        # Maximum risk filter
        if risk_metrics.overall_risk_score > 0.8:  # High risk threshold
            return False
        
        # Portfolio concentration filter
        if portfolio is not None:
            if risk_metrics.concentration_risk > 0.5:  # Too concentrated
                return False
        
        # Volatility filter
        if risk_metrics.volatility > 0.5:  # Extremely high volatility
            return False
        
        # VaR filter
        if risk_metrics.var_95 > 0.1:  # More than 10% VaR
            return False
        
        return True
    
    def _calculate_stop_loss_take_profit(
        self,
        action: TradingAction,
        current_price: float,
        volatility: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop-loss and take-profit levels."""
        if not self.enable_stop_loss or action == TradingAction.HOLD:
            return None, None
        
        # Adjust stop-loss based on volatility
        adjusted_stop_loss_pct = max(
            self.stop_loss_pct,
            volatility * 2  # At least 2x daily volatility
        )
        
        adjusted_take_profit_pct = max(
            self.take_profit_pct,
            volatility * 3  # At least 3x daily volatility
        )
        
        if action == TradingAction.BUY:
            stop_loss = current_price * (1 - adjusted_stop_loss_pct)
            target_price = current_price * (1 + adjusted_take_profit_pct)
        elif action == TradingAction.SELL:
            stop_loss = current_price * (1 + adjusted_stop_loss_pct)
            target_price = current_price * (1 - adjusted_take_profit_pct)
        else:
            stop_loss = None
            target_price = None
        
        return stop_loss, target_price
    
    def _update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for risk calculations."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only recent history (e.g., 252 trading days)
        max_history = 252
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    def update_models(
        self,
        cnn_lstm_model: Optional[CNNLSTMHybridModel] = None,
        rl_ensemble: Optional[EnsembleManager] = None
    ) -> None:
        """Update the underlying ML models."""
        if cnn_lstm_model is not None:
            self.cnn_lstm_model = cnn_lstm_model
            self.logger.info("Updated CNN+LSTM model")
        
        if rl_ensemble is not None:
            self.rl_ensemble = rl_ensemble
            self.logger.info("Updated RL ensemble")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of underlying models."""
        return {
            'cnn_lstm_trained': self.cnn_lstm_model.is_trained,
            'rl_ensemble_agents': len(self.rl_ensemble.agents),
            'rl_trained_agents': sum(1 for agent in self.rl_ensemble.agents if agent.is_trained),
            'model_version': self.model_version,
            'symbols_tracked': len(self.price_history)
        }


class SignalCombiner:
    """Combines CNN+LSTM and RL signals into final trading decision."""
    
    def __init__(
        self,
        cnn_lstm_weight: float = 0.6,
        rl_weight: float = 0.4,
        uncertainty_penalty: float = 0.1
    ):
        """
        Initialize signal combiner.
        
        Args:
            cnn_lstm_weight: Weight for CNN+LSTM predictions
            rl_weight: Weight for RL predictions
            uncertainty_penalty: Penalty factor for high uncertainty
        """
        self.cnn_lstm_weight = cnn_lstm_weight
        self.rl_weight = rl_weight
        self.uncertainty_penalty = uncertainty_penalty
        
        # Ensure weights sum to 1
        total_weight = cnn_lstm_weight + rl_weight
        self.cnn_lstm_weight /= total_weight
        self.rl_weight /= total_weight
    
    def combine_signals(self, components: SignalComponents) -> Dict[str, Any]:
        """
        Combine signal components into final trading decision.
        
        Args:
            components: Signal components from different models
            
        Returns:
            Combined signal with action and confidence
        """
        # Extract CNN+LSTM classification probabilities
        buy_prob, hold_prob, sell_prob = components.cnn_lstm_classification
        
        # Convert RL action to probabilities (assuming continuous action space)
        rl_action_value = components.rl_action[0] if len(components.rl_action) > 0 else 0.0
        
        # Map RL action to buy/hold/sell probabilities
        if rl_action_value > 0.1:  # Buy signal
            rl_buy_prob, rl_hold_prob, rl_sell_prob = 0.8, 0.2, 0.0
        elif rl_action_value < -0.1:  # Sell signal
            rl_buy_prob, rl_hold_prob, rl_sell_prob = 0.0, 0.2, 0.8
        else:  # Hold signal
            rl_buy_prob, rl_hold_prob, rl_sell_prob = 0.1, 0.8, 0.1
        
        # Combine probabilities using weighted average
        combined_buy_prob = (
            self.cnn_lstm_weight * buy_prob +
            self.rl_weight * rl_buy_prob
        )
        combined_hold_prob = (
            self.cnn_lstm_weight * hold_prob +
            self.rl_weight * rl_hold_prob
        )
        combined_sell_prob = (
            self.cnn_lstm_weight * sell_prob +
            self.rl_weight * rl_sell_prob
        )
        
        # Normalize probabilities
        total_prob = combined_buy_prob + combined_hold_prob + combined_sell_prob
        combined_buy_prob /= total_prob
        combined_hold_prob /= total_prob
        combined_sell_prob /= total_prob
        
        # Determine final action
        max_prob = max(combined_buy_prob, combined_hold_prob, combined_sell_prob)
        
        if max_prob == combined_buy_prob:
            action = TradingAction.BUY
        elif max_prob == combined_sell_prob:
            action = TradingAction.SELL
        else:
            action = TradingAction.HOLD
        
        # Calculate combined confidence
        base_confidence = max_prob
        
        # Apply uncertainty penalty
        uncertainty_factor = 1.0 - (components.cnn_lstm_uncertainty * self.uncertainty_penalty)
        uncertainty_factor = max(0.1, uncertainty_factor)  # Minimum 10% confidence
        
        # Combine with RL confidence
        final_confidence = (
            base_confidence * uncertainty_factor * 
            (0.5 + 0.5 * components.rl_confidence)
        )
        
        # Ensure confidence is in valid range
        final_confidence = np.clip(final_confidence, 0.0, 1.0)
        
        return {
            'action': action,
            'confidence': float(final_confidence),
            'probabilities': {
                'buy': float(combined_buy_prob),
                'hold': float(combined_hold_prob),
                'sell': float(combined_sell_prob)
            }
        }


class RiskCalculator:
    """Calculates risk metrics for position sizing and risk management."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize risk calculator."""
        self.risk_free_rate = risk_free_rate
    
    def calculate_risk_metrics(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            price_history: Historical prices
            
        Returns:
            Risk metrics for the symbol
        """
        if len(price_history) < 2:
            # Default risk metrics for insufficient data
            return RiskMetrics(
                volatility=0.2,  # Default 20% volatility
                var_95=0.05,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                correlation_risk=0.0,
                liquidity_risk=0.1,
                concentration_risk=0.0,
                overall_risk_score=0.3
            )
        
        # Calculate returns
        prices = np.array(price_history + [current_price])
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * -1
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            if np.std(excess_returns) > 0 else 0.0
        )
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * -1
        
        # Liquidity risk (based on price volatility)
        liquidity_risk = min(volatility / 0.5, 1.0)  # Normalize to [0, 1]
        
        # Overall risk score (weighted combination)
        overall_risk_score = (
            0.3 * min(volatility / 0.5, 1.0) +
            0.3 * min(var_95 / 0.1, 1.0) +
            0.2 * min(max_drawdown / 0.2, 1.0) +
            0.2 * liquidity_risk
        )
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            correlation_risk=0.0,  # TODO: Implement correlation analysis
            liquidity_risk=liquidity_risk,
            concentration_risk=0.0,  # Will be calculated at portfolio level
            overall_risk_score=overall_risk_score
        )


class PositionSizer:
    """Calculates optimal position sizes based on risk and confidence."""
    
    def __init__(self, params: PositionSizingParams):
        """Initialize position sizer with parameters."""
        self.params = params
    
    def calculate_position_size(
        self,
        signal_confidence: float,
        risk_metrics: RiskMetrics,
        portfolio: Optional[Portfolio],
        current_price: float
    ) -> float:
        """
        Calculate optimal position size.
        
        Args:
            signal_confidence: Confidence in the trading signal
            risk_metrics: Risk metrics for the asset
            portfolio: Current portfolio state
            current_price: Current asset price
            
        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        # Base position size from confidence
        base_size = signal_confidence * self.params.max_position_size
        
        # Risk-adjusted size using Kelly criterion
        if risk_metrics.sharpe_ratio > 0:
            kelly_size = min(
                risk_metrics.sharpe_ratio / (risk_metrics.volatility ** 2),
                self.params.max_position_size
            ) * self.params.kelly_fraction
        else:
            kelly_size = base_size * 0.5  # Reduce size for negative Sharpe
        
        # Volatility adjustment
        volatility_adjustment = max(0.1, 1.0 - risk_metrics.volatility)
        
        # VaR adjustment
        var_adjustment = max(0.1, 1.0 - risk_metrics.var_95 * 10)
        
        # Portfolio concentration adjustment
        concentration_adjustment = 1.0
        if portfolio is not None:
            # Reduce size if portfolio is already concentrated
            concentration_adjustment = max(0.5, 1.0 - risk_metrics.concentration_risk)
        
        # Combine all adjustments
        adjusted_size = min(
            base_size,
            kelly_size
        ) * volatility_adjustment * var_adjustment * concentration_adjustment
        
        # Apply minimum confidence threshold
        if signal_confidence < self.params.confidence_threshold:
            adjusted_size = 0.0
        
        # Ensure size is within bounds
        final_size = np.clip(adjusted_size, 0.0, self.params.max_position_size)
        
        return float(final_size)