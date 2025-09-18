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

from src.models.trading_signal import TradingSignal, TradingAction
from src.models.market_data import MarketData
from src.models.portfolio import Portfolio, Position
from src.ml.hybrid_model import CNNLSTMHybridModel
from src.ml.rl_ensemble import EnsembleManager
from src.ml.decision_auditor import DecisionAuditor, create_decision_auditor


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
        cnn_lstm_model: Optional[CNNLSTMHybridModel] = None,
        rl_ensemble: Optional[EnsembleManager] = None,
        position_sizing_params: Optional[PositionSizingParams] = None,
        decision_auditor: Optional[DecisionAuditor] = None,
        risk_free_rate: float = 0.02,
        enable_stop_loss: bool = True,
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.15,  # 15% take profit
        model_version: str = "v1.0.0",
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

        # Initialize decision auditor
        self.decision_auditor = decision_auditor or create_decision_auditor()


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
        market_features: Optional[Dict[str, float]] = None,
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
            # Get enhanced predictions from CNN+LSTM model
            cnn_lstm_predictions = self._get_cnn_lstm_predictions(market_data)

            # Get enhanced predictions from RL ensemble with CNN+LSTM integration
            rl_predictions = self._get_rl_predictions(
                market_data, portfolio, cnn_lstm_predictions
            )

            # Combine enhanced signal components
            signal_components = SignalComponents(
                cnn_lstm_classification=cnn_lstm_predictions["classification_probs"],
                cnn_lstm_regression=cnn_lstm_predictions["regression_pred"],
                cnn_lstm_uncertainty=cnn_lstm_predictions.get(
                    "regression_uncertainty", 0.0
                ),
                rl_action=rl_predictions["action"],
                rl_confidence=rl_predictions["confidence"],
                market_features={
                    **(market_features or {}),
                    **cnn_lstm_predictions.get("enhanced_features", {}),
                    "rl_ensemble_agreement": rl_predictions.get(
                        "ensemble_agreement", 0.5
                    ),
                    "market_regime": cnn_lstm_predictions.get(
                        "enhanced_features", {}
                    ).get("market_regime", "unknown"),
                },
                timestamp=datetime.now(timezone.utc),
            )

            # Combine signals to get final decision
            combined_signal = self.signal_combiner.combine_signals(signal_components)

            # Calculate risk metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                symbol, current_price, self.price_history.get(symbol, [])
            )

            # Calculate enhanced position size
            position_size = self.position_sizer.calculate_position_size(
                signal_confidence=combined_signal["confidence"],
                risk_metrics=risk_metrics,
                portfolio=portfolio,
                current_price=current_price,
                enhanced_signal_data=combined_signal,
            )

            # Apply enhanced risk management filters
            if not self._passes_risk_filters(
                combined_signal, risk_metrics, portfolio, cnn_lstm_predictions
            ):
                self.logger.info(
                    f"Signal for {symbol} filtered out by enhanced risk management"
                )
                return None

            # Calculate enhanced stop-loss and take-profit levels
            stop_loss, target_price = self._calculate_stop_loss_take_profit(
                action=combined_signal["action"],
                current_price=current_price,
                volatility=risk_metrics.volatility,
                cnn_lstm_predictions=cnn_lstm_predictions,
                combined_signal=combined_signal,
            )

            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                action=combined_signal["action"],
                confidence=combined_signal["confidence"],
                position_size=position_size,
                target_price=target_price,
                stop_loss=stop_loss,
                timestamp=datetime.now(timezone.utc),
                model_version=self.model_version,
            )

            # Update price history
            self._update_price_history(symbol, current_price)

            self.logger.info(
                f"Generated signal for {symbol}: {signal.action} "
                f"(confidence: {signal.confidence:.3f}, "
                f"position_size: {signal.position_size:.3f})"
            )

            # Log the decision with the auditor
            self.decision_auditor.log_decision(
                model=self.cnn_lstm_model,  # The primary model
                input_data=market_data,
                prediction=signal.to_dict(),
                shap_values=None,  # SHAP values would be computed on-demand
                attention_weights=cnn_lstm_predictions.get('attention_weights'),
                confidence_scores={'final_confidence': signal.confidence},
                feature_importance=None,  # Feature importance would be computed on-demand
                ensemble_weights=cnn_lstm_predictions.get('ensemble_weights'),
                metadata={
                    'symbol': symbol,
                    'current_price': current_price,
                    'portfolio_context': portfolio.to_dict() if portfolio else None,
                    'risk_metrics': risk_metrics.__dict__,
                    'signal_components': combined_signal,
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _get_cnn_lstm_predictions(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Get enhanced predictions from CNN+LSTM model with comprehensive uncertainty quantification."""
        if self.cnn_lstm_model is None:
            # Return default predictions for testing
            return {
                "classification_probs": np.array([0.33, 0.33, 0.33]),
                "classification_confidence": 0.5,
                "regression_pred": 10.0,
                "regression_uncertainty": 0.1,
                "confidence_interval_95": 0.2,
                "confidence_interval_68": 0.1,
                "enhanced_features": {
                    "price_momentum": 0.0,
                    "volatility_regime": 0.0,
                    "trend_strength": 0.0,
                    "market_regime": "unknown",
                },
                "ensemble_weights": None,
            }
        
        if not self.cnn_lstm_model.is_trained:
            raise ValueError("CNN+LSTM model is not trained")

        # Ensure correct input shape: (batch_size, input_channels, sequence_length)
        if len(market_data.shape) == 2:
            market_data = market_data.reshape(
                1, market_data.shape[0], market_data.shape[1]
            )
        elif len(market_data.shape) == 3 and market_data.shape[0] != 1:
            market_data = market_data[:1]  # Take first sample if batch

        predictions = self.cnn_lstm_model.predict(
            market_data, return_uncertainty=True, use_ensemble=True
        )

        # Enhanced uncertainty quantification
        classification_probs = predictions["ensemble_classification"][0]
        regression_pred = predictions["ensemble_regression"][0, 0]
        regression_uncertainty = predictions.get(
            "regression_uncertainty", np.array([[0.0]])
        )[0, 0]

        # Calculate classification confidence (entropy-based)
        classification_entropy = -np.sum(
            classification_probs * np.log(classification_probs + 1e-8)
        )
        max_entropy = -np.log(
            1.0 / len(classification_probs)
        )  # Maximum possible entropy
        classification_confidence = 1.0 - (classification_entropy / max_entropy)

        # Calculate prediction confidence intervals using uncertainty
        confidence_interval_95 = (
            1.96 * regression_uncertainty
        )  # 95% confidence interval
        confidence_interval_68 = regression_uncertainty  # 68% confidence interval

        # Enhanced feature extraction for RL integration
        enhanced_features = {
            "price_momentum": float(regression_pred - 100.0),  # Relative to baseline
            "volatility_regime": min(
                regression_uncertainty / 0.1, 1.0
            ),  # Normalized volatility
            "trend_strength": float(
                np.max(classification_probs) - 0.33
            ),  # Above random
            "market_regime": self._classify_market_regime(
                classification_probs, regression_uncertainty
            ),
        }

        return {
            "classification_probs": classification_probs,
            "classification_confidence": classification_confidence,
            "regression_pred": regression_pred,
            "regression_uncertainty": regression_uncertainty,
            "confidence_interval_95": confidence_interval_95,
            "confidence_interval_68": confidence_interval_68,
            "enhanced_features": enhanced_features,
            "ensemble_weights": predictions.get("ensemble_weights", None),
        }

    def _get_rl_predictions(
        self,
        market_data: np.ndarray,
        portfolio: Optional[Portfolio],
        cnn_lstm_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get enhanced predictions from RL ensemble with CNN+LSTM integration."""
        # Handle case where rl_ensemble is None (for testing)
        if self.rl_ensemble is None:
            # Return default predictions for testing
            return {
                "action": np.array([0.0]),  # Hold action
                "confidence": 0.5,
                "action_interpretation": {
                    "type": "hold",
                    "strength": 0.5,
                    "raw_value": 0.0,
                    "confidence_adjusted_strength": 0.25,
                },
                "ensemble_agreement": 0.5,
                "individual_actions": [],
            }
        
        # Prepare enhanced observation for RL
        market_features = market_data.flatten()

        # Add CNN+LSTM enhanced features if available
        enhanced_obs_components = [market_features]

        if cnn_lstm_features is not None:
            # Add CNN+LSTM derived features
            enhanced_features = cnn_lstm_features.get("enhanced_features", {})
            cnn_lstm_obs = np.array(
                [
                    enhanced_features.get("price_momentum", 0.0),
                    enhanced_features.get("volatility_regime", 0.0),
                    enhanced_features.get("trend_strength", 0.0),
                    cnn_lstm_features.get("classification_confidence", 0.0),
                    cnn_lstm_features.get("regression_uncertainty", 0.0),
                ]
            )
            enhanced_obs_components.append(cnn_lstm_obs)

        # Add portfolio features if available
        if portfolio is not None:
            portfolio_features = np.array(
                [
                    portfolio.total_value / 100000.0,  # Normalize
                    portfolio.cash_balance / portfolio.total_value,  # Cash ratio
                    len(portfolio.positions) / 10.0,  # Normalize position count
                    sum(pos.unrealized_pnl for pos in portfolio.positions.values())
                    / portfolio.total_value,  # P&L ratio
                ]
            )
            enhanced_obs_components.append(portfolio_features)

        # Combine all observation components
        observation = np.concatenate(enhanced_obs_components)

        # Get ensemble prediction
        action = self.rl_ensemble.predict(observation, deterministic=True)

        # Enhanced confidence calculation based on ensemble agreement and CNN+LSTM uncertainty
        individual_actions = []
        individual_confidences = []

        for agent in self.rl_ensemble.agents:
            if agent.is_trained:
                individual_action, _ = agent.predict(observation, deterministic=True)
                individual_actions.append(individual_action)

                # Calculate individual agent confidence (simplified)
                action_magnitude = np.abs(individual_action).mean()
                individual_confidences.append(min(action_magnitude * 2, 1.0))

        if individual_actions:
            # Calculate confidence as combination of ensemble agreement and individual confidences
            action_variance = np.var(individual_actions, axis=0)
            ensemble_agreement = 1.0 / (1.0 + np.mean(action_variance))
            avg_individual_confidence = np.mean(individual_confidences)

            # Combine with CNN+LSTM confidence if available
            if cnn_lstm_features is not None:
                cnn_lstm_confidence = cnn_lstm_features.get(
                    "classification_confidence", 0.5
                )
                # Weighted combination: 60% RL ensemble, 40% CNN+LSTM
                confidence = (
                    0.6 * (ensemble_agreement * avg_individual_confidence)
                    + 0.4 * cnn_lstm_confidence
                )
            else:
                confidence = ensemble_agreement * avg_individual_confidence
        else:
            confidence = 0.5  # Default confidence for untrained ensemble

        # Calculate action interpretation for better signal combination
        action_interpretation = self._interpret_rl_action(action, confidence)

        return {
            "action": action,
            "confidence": confidence,
            "action_interpretation": action_interpretation,
            "ensemble_agreement": ensemble_agreement if individual_actions else 0.5,
            "individual_actions": individual_actions,
        }

    def _passes_risk_filters(
        self,
        signal: Dict[str, Any],
        risk_metrics: RiskMetrics,
        portfolio: Optional[Portfolio],
        cnn_lstm_predictions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Apply enhanced risk management filters using CNN+LSTM confidence intervals."""

        # Minimum confidence filter
        if signal["confidence"] < self.position_sizing_params.confidence_threshold:
            self.logger.debug("Signal filtered: confidence below threshold")
            return False

        # Enhanced confidence breakdown filter
        confidence_breakdown = signal.get("confidence_breakdown", {})
        if confidence_breakdown:
            # Check if uncertainty penalty is too high
            uncertainty_penalty = confidence_breakdown.get("uncertainty_penalty", 0.0)
            if uncertainty_penalty > 0.3:  # High uncertainty penalty
                self.logger.debug("Signal filtered: high uncertainty penalty")
                return False

            # Check regime confidence multiplier
            regime_multiplier = confidence_breakdown.get(
                "regime_confidence_multiplier", 1.0
            )
            if regime_multiplier < 0.7:  # Very unfavorable market regime
                self.logger.debug("Signal filtered: unfavorable market regime")
                return False

        # Maximum risk filter
        if risk_metrics.overall_risk_score > 0.8:  # High risk threshold
            self.logger.debug("Signal filtered: overall risk score too high")
            return False

        # Enhanced CNN+LSTM uncertainty filter
        if cnn_lstm_predictions is not None:
            regression_uncertainty = cnn_lstm_predictions.get(
                "regression_uncertainty", 0.0
            )
            if regression_uncertainty > 0.4:  # Very high prediction uncertainty
                self.logger.debug(
                    "Signal filtered: CNN+LSTM prediction uncertainty too high"
                )
                return False

            # Check if confidence interval is too wide
            confidence_interval_95 = cnn_lstm_predictions.get(
                "confidence_interval_95", 0.0
            )
            if confidence_interval_95 > 0.2:  # More than 20% price uncertainty
                self.logger.debug("Signal filtered: confidence interval too wide")
                return False

        # Portfolio concentration filter
        if portfolio is not None:
            if risk_metrics.concentration_risk > 0.5:  # Too concentrated
                self.logger.debug("Signal filtered: portfolio too concentrated")
                return False

        # Volatility filter
        if risk_metrics.volatility > 0.5:  # Extremely high volatility
            self.logger.debug("Signal filtered: volatility too high")
            return False

        # VaR filter
        if risk_metrics.var_95 > 0.1:  # More than 10% VaR
            self.logger.debug("Signal filtered: VaR too high")
            return False

        # Signal strength filter
        signal_strength = signal.get("signal_strength", 0.0)
        if signal_strength < -0.1:  # Negative signal strength (worse than random)
            self.logger.debug("Signal filtered: negative signal strength")
            return False

        # Market regime filter
        market_regime = signal.get("market_regime", "unknown")
        if market_regime == "high_uncertainty" and signal["confidence"] < 0.8:
            # In high uncertainty regimes, require very high confidence
            self.logger.debug(
                "Signal filtered: insufficient confidence for uncertain market regime"
            )
            return False

        return True

    def _calculate_stop_loss_take_profit(
        self,
        action: TradingAction,
        current_price: float,
        volatility: float,
        cnn_lstm_predictions: Optional[Dict[str, Any]] = None,
        combined_signal: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate enhanced stop-loss and take-profit levels using CNN+LSTM confidence intervals."""
        if not self.enable_stop_loss or action == TradingAction.HOLD:
            return None, None

        # Base adjustments using volatility
        base_stop_loss_pct = max(self.stop_loss_pct, volatility * 2)
        base_take_profit_pct = max(self.take_profit_pct, volatility * 3)

        # Enhanced adjustments using CNN+LSTM predictions
        if cnn_lstm_predictions is not None:
            # Use CNN+LSTM confidence intervals for more precise stop-loss
            confidence_interval_68 = cnn_lstm_predictions.get(
                "confidence_interval_68", volatility * current_price
            )
            confidence_interval_95 = cnn_lstm_predictions.get(
                "confidence_interval_95", volatility * current_price * 2
            )

            # Convert confidence intervals to percentages
            ci_68_pct = confidence_interval_68 / current_price
            ci_95_pct = confidence_interval_95 / current_price

            # Adjust stop-loss based on prediction uncertainty
            uncertainty = cnn_lstm_predictions.get("regression_uncertainty", volatility)
            uncertainty_multiplier = 1.0 + (
                uncertainty * 2
            )  # Wider stops for higher uncertainty

            # Use 68% confidence interval for stop-loss (1 standard deviation)
            uncertainty_adjusted_stop_loss = max(
                base_stop_loss_pct, ci_68_pct * uncertainty_multiplier
            )

            # Use 95% confidence interval for take-profit (2 standard deviations)
            uncertainty_adjusted_take_profit = max(
                base_take_profit_pct, ci_95_pct * 0.8
            )  # Slightly tighter than full CI
        else:
            uncertainty_adjusted_stop_loss = base_stop_loss_pct
            uncertainty_adjusted_take_profit = base_take_profit_pct

        # Further adjustments based on signal confidence
        if combined_signal is not None:
            signal_confidence = combined_signal.get("confidence", 0.5)
            signal_strength = combined_signal.get("signal_strength", 0.0)

            # Tighter stops for high-confidence signals
            confidence_adjustment = 0.8 + 0.4 * signal_confidence  # Range: 0.8 to 1.2

            # Adjust based on signal strength
            strength_adjustment = 0.9 + 0.2 * max(
                0, signal_strength
            )  # Range: 0.9 to 1.1

            final_stop_loss_pct = uncertainty_adjusted_stop_loss * confidence_adjustment
            final_take_profit_pct = (
                uncertainty_adjusted_take_profit * strength_adjustment
            )
        else:
            final_stop_loss_pct = uncertainty_adjusted_stop_loss
            final_take_profit_pct = uncertainty_adjusted_take_profit

        # Calculate final prices
        if action == TradingAction.BUY:
            stop_loss = current_price * (1 - final_stop_loss_pct)
            target_price = current_price * (1 + final_take_profit_pct)
        elif action == TradingAction.SELL:
            stop_loss = current_price * (1 + final_stop_loss_pct)
            target_price = current_price * (1 - final_take_profit_pct)
        else:
            stop_loss = None
            target_price = None

        return stop_loss, target_price

    def _classify_market_regime(
        self, classification_probs: np.ndarray, uncertainty: float
    ) -> str:
        """Classify market regime based on CNN+LSTM predictions."""
        buy_prob, hold_prob, sell_prob = classification_probs

        if uncertainty > 0.3:
            return "high_uncertainty"
        elif buy_prob > 0.6:
            return "bullish"
        elif sell_prob > 0.6:
            return "bearish"
        elif hold_prob > 0.6:
            return "sideways"
        else:
            return "mixed"

    def _interpret_rl_action(
        self, action: np.ndarray, confidence: float
    ) -> Dict[str, Any]:
        """Interpret RL action for better signal combination."""
        action_value = action[0] if len(action) > 0 else 0.0

        # Determine action type and strength
        if action_value > 0.2:
            action_type = "buy"
            strength = min(action_value, 1.0)
        elif action_value < -0.2:
            action_type = "sell"
            strength = min(abs(action_value), 1.0)
        else:
            action_type = "hold"
            strength = (
                1.0 - abs(action_value) / 0.2
            )  # Stronger hold for values closer to 0

        return {
            "type": action_type,
            "strength": strength,
            "raw_value": action_value,
            "confidence_adjusted_strength": strength * confidence,
        }

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
        rl_ensemble: Optional[EnsembleManager] = None,
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
        cnn_lstm_trained = (
            self.cnn_lstm_model.is_trained if self.cnn_lstm_model else False
        )
        rl_ensemble_agents = (
            len(self.rl_ensemble.agents) if self.rl_ensemble else 0
        )
        rl_trained_agents = (
            sum(1 for agent in self.rl_ensemble.agents if agent.is_trained)
            if self.rl_ensemble else 0
        )
        
        return {
            "cnn_lstm_trained": cnn_lstm_trained,
            "rl_ensemble_agents": rl_ensemble_agents,
            "rl_trained_agents": rl_trained_agents,
            "model_version": self.model_version,
            "symbols_tracked": len(self.price_history),
        }


    def register_model_version(
        self,
        training_data_hash: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Register the current model version with the decision auditor."""
        self.decision_auditor.register_model_version(
            model=self.cnn_lstm_model,
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics
        )
        self.logger.info(f"Registered model version {self.cnn_lstm_model.version} with the auditor.")


class SignalCombiner:
    """Enhanced signal combiner that integrates CNN+LSTM and RL signals with uncertainty-aware weighting."""

    def __init__(
        self,
        cnn_lstm_weight: float = 0.6,
        rl_weight: float = 0.4,
        uncertainty_penalty: float = 0.15,
        confidence_threshold: float = 0.5,
        adaptive_weighting: bool = True,
    ):
        """
        Initialize enhanced signal combiner.

        Args:
            cnn_lstm_weight: Base weight for CNN+LSTM predictions
            rl_weight: Base weight for RL predictions
            uncertainty_penalty: Penalty factor for high uncertainty
            confidence_threshold: Minimum confidence threshold
            adaptive_weighting: Whether to use adaptive weighting based on model confidence
        """
        self.base_cnn_lstm_weight = cnn_lstm_weight
        self.base_rl_weight = rl_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.confidence_threshold = confidence_threshold
        self.adaptive_weighting = adaptive_weighting

        # Ensure base weights sum to 1
        total_weight = cnn_lstm_weight + rl_weight
        self.base_cnn_lstm_weight /= total_weight
        self.base_rl_weight /= total_weight

    def combine_signals(self, components: SignalComponents) -> Dict[str, Any]:
        """
        Enhanced signal combination with uncertainty-aware weighting and market regime adaptation.

        Args:
            components: Signal components from different models

        Returns:
            Combined signal with action, confidence, and detailed analysis
        """
        # Extract CNN+LSTM classification probabilities
        buy_prob, hold_prob, sell_prob = components.cnn_lstm_classification

        # Get market regime and uncertainty information
        market_regime = components.market_features.get("market_regime", "unknown")
        rl_ensemble_agreement = components.market_features.get(
            "rl_ensemble_agreement", 0.5
        )

        # Adaptive weighting based on model confidence and market regime
        if self.adaptive_weighting:
            cnn_lstm_weight, rl_weight = self._calculate_adaptive_weights(
                components, market_regime, rl_ensemble_agreement
            )
        else:
            cnn_lstm_weight = self.base_cnn_lstm_weight
            rl_weight = self.base_rl_weight

        # Enhanced RL action interpretation
        rl_action_value = (
            components.rl_action[0] if len(components.rl_action) > 0 else 0.0
        )

        # More sophisticated RL action to probability mapping
        rl_probs = self._map_rl_action_to_probabilities(
            rl_action_value, components.rl_confidence
        )
        rl_buy_prob, rl_hold_prob, rl_sell_prob = rl_probs

        # Combine probabilities using adaptive weights
        combined_buy_prob = cnn_lstm_weight * buy_prob + rl_weight * rl_buy_prob
        combined_hold_prob = cnn_lstm_weight * hold_prob + rl_weight * rl_hold_prob
        combined_sell_prob = cnn_lstm_weight * sell_prob + rl_weight * rl_sell_prob

        # Normalize probabilities
        total_prob = combined_buy_prob + combined_hold_prob + combined_sell_prob
        combined_buy_prob /= total_prob
        combined_hold_prob /= total_prob
        combined_sell_prob /= total_prob

        # Determine final action with confidence threshold
        max_prob = max(combined_buy_prob, combined_hold_prob, combined_sell_prob)

        if max_prob == combined_buy_prob and max_prob > self.confidence_threshold:
            action = TradingAction.BUY
        elif max_prob == combined_sell_prob and max_prob > self.confidence_threshold:
            action = TradingAction.SELL
        else:
            action = TradingAction.HOLD

        # Enhanced confidence calculation
        confidence_metrics = self._calculate_enhanced_confidence(
            components, max_prob, cnn_lstm_weight, rl_weight, market_regime
        )

        return {
            "action": action,
            "confidence": confidence_metrics["final_confidence"],
            "probabilities": {
                "buy": float(combined_buy_prob),
                "hold": float(combined_hold_prob),
                "sell": float(combined_sell_prob),
            },
            "confidence_breakdown": confidence_metrics,
            "adaptive_weights": {
                "cnn_lstm_weight": cnn_lstm_weight,
                "rl_weight": rl_weight,
            },
            "market_regime": market_regime,
            "signal_strength": max_prob - 0.33,  # Above random chance
        }

    def _calculate_adaptive_weights(
        self,
        components: SignalComponents,
        market_regime: str,
        rl_ensemble_agreement: float,
    ) -> Tuple[float, float]:
        """Calculate adaptive weights based on model confidence and market conditions."""

        # Base weights
        cnn_lstm_weight = self.base_cnn_lstm_weight
        rl_weight = self.base_rl_weight

        # Adjust based on CNN+LSTM uncertainty
        uncertainty = components.cnn_lstm_uncertainty
        if uncertainty > 0.3:  # High uncertainty
            # Increase RL weight when CNN+LSTM is uncertain
            rl_weight += 0.1
            cnn_lstm_weight -= 0.1
        elif uncertainty < 0.1:  # Low uncertainty
            # Increase CNN+LSTM weight when it's confident
            cnn_lstm_weight += 0.1
            rl_weight -= 0.1

        # Adjust based on RL ensemble agreement
        if rl_ensemble_agreement > 0.8:  # High agreement
            rl_weight += 0.05
            cnn_lstm_weight -= 0.05
        elif rl_ensemble_agreement < 0.5:  # Low agreement
            rl_weight -= 0.05
            cnn_lstm_weight += 0.05

        # Market regime adjustments
        if market_regime == "high_uncertainty":
            # In uncertain markets, rely more on RL's adaptive behavior
            rl_weight += 0.1
            cnn_lstm_weight -= 0.1
        elif market_regime in ["bullish", "bearish"]:
            # In trending markets, CNN+LSTM pattern recognition is valuable
            cnn_lstm_weight += 0.05
            rl_weight -= 0.05

        # Ensure weights are positive and sum to 1
        cnn_lstm_weight = max(0.1, min(0.9, cnn_lstm_weight))
        rl_weight = max(0.1, min(0.9, rl_weight))

        total_weight = cnn_lstm_weight + rl_weight
        cnn_lstm_weight /= total_weight
        rl_weight /= total_weight

        return cnn_lstm_weight, rl_weight

    def _map_rl_action_to_probabilities(
        self, action_value: float, rl_confidence: float
    ) -> Tuple[float, float, float]:
        """Map RL action to buy/hold/sell probabilities with confidence weighting."""

        # Sigmoid-like mapping for smoother transitions
        def sigmoid(x, steepness=5.0):
            return 1.0 / (1.0 + np.exp(-steepness * x))

        # Map action value to probabilities
        if action_value > 0.05:  # Buy signal
            buy_strength = sigmoid(action_value - 0.05)
            buy_prob = 0.4 + 0.5 * buy_strength * rl_confidence
            sell_prob = 0.1 * (1 - buy_strength)
            hold_prob = 1.0 - buy_prob - sell_prob
        elif action_value < -0.05:  # Sell signal
            sell_strength = sigmoid(-action_value - 0.05)
            sell_prob = 0.4 + 0.5 * sell_strength * rl_confidence
            buy_prob = 0.1 * (1 - sell_strength)
            hold_prob = 1.0 - buy_prob - sell_prob
        else:  # Hold signal
            hold_strength = 1.0 - abs(action_value) / 0.05
            hold_prob = 0.5 + 0.4 * hold_strength * rl_confidence
            buy_prob = 0.25 * (1 - hold_strength)
            sell_prob = 0.25 * (1 - hold_strength)

        # Normalize
        total = buy_prob + hold_prob + sell_prob
        return buy_prob / total, hold_prob / total, sell_prob / total

    def _calculate_enhanced_confidence(
        self,
        components: SignalComponents,
        base_confidence: float,
        cnn_lstm_weight: float,
        rl_weight: float,
        market_regime: str,
    ) -> Dict[str, float]:
        """Calculate enhanced confidence with detailed breakdown."""

        # CNN+LSTM confidence components
        cnn_lstm_uncertainty_penalty = (
            components.cnn_lstm_uncertainty * self.uncertainty_penalty
        )
        cnn_lstm_confidence_factor = max(0.1, 1.0 - cnn_lstm_uncertainty_penalty)

        # RL confidence components
        rl_confidence_factor = components.rl_confidence

        # Market regime confidence adjustment
        regime_confidence_multiplier = {
            "bullish": 1.0,
            "bearish": 1.0,
            "sideways": 0.9,
            "mixed": 0.8,
            "high_uncertainty": 0.7,
            "unknown": 0.8,
        }.get(market_regime, 0.8)

        # Ensemble agreement bonus
        ensemble_agreement = components.market_features.get(
            "rl_ensemble_agreement", 0.5
        )
        agreement_bonus = (ensemble_agreement - 0.5) * 0.2  # Up to 10% bonus

        # Combined confidence calculation
        weighted_model_confidence = (
            cnn_lstm_weight * cnn_lstm_confidence_factor
            + rl_weight * rl_confidence_factor
        )

        final_confidence = (
            base_confidence * weighted_model_confidence * regime_confidence_multiplier
            + agreement_bonus
        )

        # Ensure confidence is in valid range
        final_confidence = np.clip(final_confidence, 0.0, 1.0)

        return {
            "final_confidence": float(final_confidence),
            "base_confidence": float(base_confidence),
            "cnn_lstm_confidence_factor": float(cnn_lstm_confidence_factor),
            "rl_confidence_factor": float(rl_confidence_factor),
            "regime_confidence_multiplier": float(regime_confidence_multiplier),
            "agreement_bonus": float(agreement_bonus),
            "uncertainty_penalty": float(cnn_lstm_uncertainty_penalty),
        }


class RiskCalculator:
    """Calculates risk metrics for position sizing and risk management."""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize risk calculator."""
        self.risk_free_rate = risk_free_rate

    def calculate_risk_metrics(
        self, symbol: str, current_price: float, price_history: List[float]
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
                overall_risk_score=0.3,
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
            if np.std(excess_returns) > 0
            else 0.0
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
            0.3 * min(volatility / 0.5, 1.0)
            + 0.3 * min(var_95 / 0.1, 1.0)
            + 0.2 * min(max_drawdown / 0.2, 1.0)
            + 0.2 * liquidity_risk
        )

        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            correlation_risk=0.0,  # TODO: Implement correlation analysis
            liquidity_risk=liquidity_risk,
            concentration_risk=0.0,  # Will be calculated at portfolio level
            overall_risk_score=overall_risk_score,
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
        current_price: float,
        enhanced_signal_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate optimal position size with enhanced CNN+LSTM and RL integration.

        Args:
            signal_confidence: Confidence in the trading signal
            risk_metrics: Risk metrics for the asset
            portfolio: Current portfolio state
            current_price: Current asset price
            enhanced_signal_data: Enhanced signal data from CNN+LSTM and RL models

        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        # Base position size from confidence
        base_size = signal_confidence * self.params.max_position_size

        # Enhanced Kelly criterion with uncertainty adjustment
        if risk_metrics.sharpe_ratio > 0:
            kelly_size = (
                min(
                    risk_metrics.sharpe_ratio / (risk_metrics.volatility**2),
                    self.params.max_position_size,
                )
                * self.params.kelly_fraction
            )
        else:
            kelly_size = base_size * 0.5  # Reduce size for negative Sharpe

        # Enhanced adjustments using CNN+LSTM and RL insights
        adjustments = self._calculate_enhanced_adjustments(
            risk_metrics, portfolio, enhanced_signal_data
        )

        # Combine base sizes with enhanced adjustments
        confidence_adjusted_size = base_size * adjustments["confidence_multiplier"]
        kelly_adjusted_size = kelly_size * adjustments["kelly_multiplier"]

        # Take the more conservative of the two approaches
        conservative_size = min(confidence_adjusted_size, kelly_adjusted_size)

        # Apply all adjustment factors
        final_adjustments = (
            adjustments["volatility_adjustment"]
            * adjustments["var_adjustment"]
            * adjustments["concentration_adjustment"]
            * adjustments["regime_adjustment"]
            * adjustments["uncertainty_adjustment"]
        )

        adjusted_size = conservative_size * final_adjustments

        # Apply minimum confidence threshold
        if signal_confidence < self.params.confidence_threshold:
            adjusted_size = 0.0

        # Ensure size is within bounds
        final_size = np.clip(adjusted_size, 0.0, self.params.max_position_size)

        return float(final_size)

    def _calculate_enhanced_adjustments(
        self,
        risk_metrics: RiskMetrics,
        portfolio: Optional[Portfolio],
        enhanced_signal_data: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate enhanced adjustment factors using CNN+LSTM and RL insights."""

        # Default adjustments
        adjustments = {
            "confidence_multiplier": 1.0,
            "kelly_multiplier": 1.0,
            "volatility_adjustment": max(0.1, 1.0 - risk_metrics.volatility),
            "var_adjustment": max(0.1, 1.0 - risk_metrics.var_95 * 10),
            "concentration_adjustment": 1.0,
            "regime_adjustment": 1.0,
            "uncertainty_adjustment": 1.0,
        }

        # Portfolio concentration adjustment
        if portfolio is not None:
            adjustments["concentration_adjustment"] = max(
                0.5, 1.0 - risk_metrics.concentration_risk
            )

        # Enhanced adjustments using signal data
        if enhanced_signal_data is not None:
            # Market regime adjustment
            market_regime = enhanced_signal_data.get("market_regime", "unknown")
            regime_multipliers = {
                "bullish": 1.1,  # Slightly more aggressive in bull markets
                "bearish": 0.9,  # More conservative in bear markets
                "sideways": 0.8,  # Conservative in sideways markets
                "mixed": 0.7,  # Very conservative in mixed signals
                "high_uncertainty": 0.6,  # Very conservative when uncertain
                "unknown": 0.8,
            }
            adjustments["regime_adjustment"] = regime_multipliers.get(
                market_regime, 0.8
            )

            # CNN+LSTM uncertainty adjustment
            confidence_breakdown = enhanced_signal_data.get("confidence_breakdown", {})
            uncertainty_penalty = confidence_breakdown.get("uncertainty_penalty", 0.0)
            adjustments["uncertainty_adjustment"] = max(
                0.3, 1.0 - uncertainty_penalty * 2
            )

            # Signal strength adjustment
            signal_strength = enhanced_signal_data.get("signal_strength", 0.0)
            if signal_strength > 0.3:  # Strong signal
                adjustments["confidence_multiplier"] = 1.2
            elif signal_strength > 0.1:  # Moderate signal
                adjustments["confidence_multiplier"] = 1.0
            else:  # Weak signal
                adjustments["confidence_multiplier"] = 0.7

            # RL ensemble agreement adjustment
            adaptive_weights = enhanced_signal_data.get("adaptive_weights", {})
            rl_weight = adaptive_weights.get("rl_weight", 0.4)
            if rl_weight > 0.6:  # High RL influence suggests dynamic conditions
                adjustments["kelly_multiplier"] = 1.1  # Slightly more aggressive Kelly
            elif rl_weight < 0.3:  # Low RL influence suggests stable patterns
                adjustments["kelly_multiplier"] = 0.9  # More conservative Kelly

        return adjustments
