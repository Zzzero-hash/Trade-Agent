"""
Comprehensive Reward Functions and Risk Metrics for Trading RL.

This module implements advanced reward functions and risk metrics for trading
reinforcement learning agents, including multi-objective rewards, risk-adjusted
performance metrics, and dynamic reward shaping based on market conditions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.ml.yfinance_trading_environment import MarketRegime

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class RewardType(Enum):
    """Types of reward functions available."""
    SIMPLE_RETURN = "simple_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MULTI_OBJECTIVE = "multi_objective"
    RISK_ADJUSTED = "risk_adjusted"
    REGIME_ADAPTIVE = "regime_adaptive"


@dataclass
class RewardConfig:
    """Configuration for reward function parameters."""
    # Basic reward weights
    return_weight: float = 0.4
    sharpe_weight: float = 0.3
    sortino_weight: float = 0.2
    calmar_weight: float = 0.1
    
    # Risk penalties
    drawdown_penalty: float = 2.0
    volatility_penalty: float = 0.5
    var_penalty: float = 1.0
    cvar_penalty: float = 1.5
    
    # Transaction costs
    transaction_cost_penalty: float = 1.0
    slippage_penalty: float = 0.5
    
    # Risk management bonuses/penalties
    diversification_bonus: float = 0.1
    concentration_penalty: float = 0.2
    leverage_penalty: float = 1.0
    
    # Dynamic parameters
    lookback_window: int = 20
    risk_free_rate: float = 0.02
    confidence_level: float = 0.05  # For VaR/CVaR calculations
    
    # Regime-specific adjustments
    bull_market_multiplier: float = 1.2
    bear_market_multiplier: float = 0.8
    volatile_market_multiplier: float = 0.9
    sideways_market_multiplier: float = 1.0
    
    # Scaling factors
    reward_scaling: float = 1000.0
    risk_scaling: float = 100.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.return_weight < 0:
            raise ValueError("Return weight must be non-negative")
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")


class BaseRewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.returns_history: List[float] = []
        self.portfolio_values: List[float] = []
        self.drawdowns: List[float] = []
        self.trade_costs: List[float] = []
        
    @abstractmethod
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate reward for the current step."""
        pass
    
    def reset(self):
        """Reset reward function state."""
        self.returns_history = []
        self.portfolio_values = []
        self.drawdowns = []
        self.trade_costs = []
    
    def update_history(
        self,
        portfolio_value: float,
        trade_results: List[Dict],
        drawdown: float
    ):
        """Update internal history for reward calculations."""
        self.portfolio_values.append(portfolio_value)
        self.drawdowns.append(drawdown)
        
        # Calculate return
        if len(self.portfolio_values) > 1:
            portfolio_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns_history.append(portfolio_return)
        
        # Calculate trade costs
        total_cost = sum(abs(result.get('cost', 0)) for result in trade_results if result.get('executed', False))
        self.trade_costs.append(total_cost)


class SimpleReturnReward(BaseRewardFunction):
    """Simple return-based reward function."""
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate simple return reward."""
        if previous_value <= 0:
            return 0.0
        
        portfolio_return = (portfolio_value - previous_value) / previous_value
        
        # Apply regime multiplier
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        return portfolio_return * self.config.reward_scaling * regime_multiplier
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-specific multiplier."""
        multipliers = {
            MarketRegime.BULL_MARKET: self.config.bull_market_multiplier,
            MarketRegime.BEAR_MARKET: self.config.bear_market_multiplier,
            MarketRegime.HIGH_VOLATILITY: self.config.volatile_market_multiplier,
            MarketRegime.SIDEWAYS_MARKET: self.config.sideways_market_multiplier
        }
        return multipliers.get(regime, 1.0)


class SharpeRatioReward(BaseRewardFunction):
    """Sharpe ratio-based reward function."""
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate Sharpe ratio reward."""
        self.update_history(portfolio_value, trade_results, portfolio_state.get('drawdown', 0))
        
        if len(self.returns_history) < 2:
            return 0.0
        
        # Calculate rolling Sharpe ratio
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        excess_returns = recent_returns - (self.config.risk_free_rate / 252)
        
        if len(excess_returns) < 2 or np.std(excess_returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Apply regime adjustment
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        return sharpe_ratio * self.config.sharpe_weight * regime_multiplier
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-specific multiplier."""
        return SimpleReturnReward._get_regime_multiplier(self, regime)


class SortinoRatioReward(BaseRewardFunction):
    """Sortino ratio-based reward function (downside deviation)."""
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate Sortino ratio reward."""
        self.update_history(portfolio_value, trade_results, portfolio_state.get('drawdown', 0))
        
        if len(self.returns_history) < 2:
            return 0.0
        
        # Calculate rolling Sortino ratio
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        excess_returns = recent_returns - (self.config.risk_free_rate / 252)
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            # No downside risk, use standard deviation
            if np.std(excess_returns) == 0:
                return 0.0
            sortino_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        
        # Apply regime adjustment
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        return sortino_ratio * self.config.sortino_weight * regime_multiplier
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-specific multiplier."""
        return SimpleReturnReward._get_regime_multiplier(self, regime)


class CalmarRatioReward(BaseRewardFunction):
    """Calmar ratio-based reward function (return/max drawdown)."""
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate Calmar ratio reward."""
        self.update_history(portfolio_value, trade_results, portfolio_state.get('drawdown', 0))
        
        if len(self.returns_history) < 2:
            return 0.0
        
        # Calculate annualized return
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        if len(recent_returns) == 0:
            return 0.0
        
        annualized_return = np.mean(recent_returns) * 252
        
        # Calculate maximum drawdown
        recent_drawdowns = self.drawdowns[-self.config.lookback_window:]
        max_drawdown = max(recent_drawdowns) if recent_drawdowns else 0.0
        
        if max_drawdown == 0:
            return annualized_return * self.config.calmar_weight
        
        calmar_ratio = annualized_return / max_drawdown
        
        # Apply regime adjustment
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        return calmar_ratio * self.config.calmar_weight * regime_multiplier
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-specific multiplier."""
        return SimpleReturnReward._get_regime_multiplier(self, regime)


class MultiObjectiveReward(BaseRewardFunction):
    """Multi-objective reward combining returns, Sharpe ratio, and risk metrics."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.risk_calculator = RiskMetricsCalculator(config)
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate multi-objective reward."""
        self.update_history(portfolio_value, trade_results, portfolio_state.get('drawdown', 0))
        
        # Base return component
        if previous_value <= 0:
            portfolio_return = 0.0
        else:
            portfolio_return = (portfolio_value - previous_value) / previous_value
        
        return_component = portfolio_return * self.config.return_weight
        
        # Risk-adjusted components
        sharpe_component = self._calculate_sharpe_component()
        sortino_component = self._calculate_sortino_component()
        
        # Risk penalties
        drawdown_penalty = self._calculate_drawdown_penalty(portfolio_state.get('drawdown', 0))
        var_penalty = self._calculate_var_penalty()
        cvar_penalty = self._calculate_cvar_penalty()
        
        # Transaction cost penalty
        transaction_penalty = self._calculate_transaction_penalty(trade_results, portfolio_value)
        
        # Portfolio risk penalties
        concentration_penalty = self._calculate_concentration_penalty(portfolio_state)
        leverage_penalty = self._calculate_leverage_penalty(portfolio_state)
        
        # Combine components
        reward = (
            return_component * self.config.reward_scaling +
            sharpe_component +
            sortino_component +
            drawdown_penalty +
            var_penalty +
            cvar_penalty +
            transaction_penalty +
            concentration_penalty +
            leverage_penalty
        )
        
        # Apply regime multiplier
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        return reward * regime_multiplier
    
    def _calculate_sharpe_component(self) -> float:
        """Calculate Sharpe ratio component."""
        if len(self.returns_history) < 2:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        excess_returns = recent_returns - (self.config.risk_free_rate / 252)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe_ratio * self.config.sharpe_weight
    
    def _calculate_sortino_component(self) -> float:
        """Calculate Sortino ratio component."""
        if len(self.returns_history) < 2:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        excess_returns = recent_returns - (self.config.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino_ratio * self.config.sortino_weight
    
    def _calculate_drawdown_penalty(self, current_drawdown: float) -> float:
        """Calculate drawdown penalty."""
        return -current_drawdown * self.config.drawdown_penalty
    
    def _calculate_var_penalty(self) -> float:
        """Calculate Value at Risk penalty."""
        if len(self.returns_history) < 10:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        var = self.risk_calculator.calculate_var(recent_returns, self.config.confidence_level)
        
        return -abs(var) * self.config.var_penalty
    
    def _calculate_cvar_penalty(self) -> float:
        """Calculate Conditional Value at Risk penalty."""
        if len(self.returns_history) < 10:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        cvar = self.risk_calculator.calculate_cvar(recent_returns, self.config.confidence_level)
        
        return -abs(cvar) * self.config.cvar_penalty
    
    def _calculate_transaction_penalty(self, trade_results: List[Dict], portfolio_value: float) -> float:
        """Calculate transaction cost penalty."""
        total_cost = sum(
            abs(result.get('cost', 0)) for result in trade_results 
            if result.get('executed', False)
        )
        
        if portfolio_value <= 0:
            return 0.0
        
        cost_ratio = total_cost / portfolio_value
        return -cost_ratio * self.config.transaction_cost_penalty
    
    def _calculate_concentration_penalty(self, portfolio_state: Dict) -> float:
        """Calculate portfolio concentration penalty."""
        positions = portfolio_state.get('positions', {})
        current_prices = portfolio_state.get('current_prices', {})
        portfolio_value = portfolio_state.get('portfolio_value', 1)
        
        if not positions or not current_prices or portfolio_value <= 0:
            return 0.0
        
        # Calculate position weights
        position_values = [
            positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in positions.keys()
        ]
        
        total_position_value = sum(position_values)
        if total_position_value <= 0:
            return 0.0
        
        weights = [value / total_position_value for value in position_values]
        
        # Calculate Herfindahl-Hirschman Index (concentration measure)
        hhi = sum(w**2 for w in weights)
        
        # Penalty for high concentration (HHI > 0.5 indicates high concentration)
        if hhi > 0.5:
            return -(hhi - 0.5) * self.config.concentration_penalty
        
        return 0.0
    
    def _calculate_leverage_penalty(self, portfolio_state: Dict) -> float:
        """Calculate leverage penalty."""
        exposure = portfolio_state.get('exposure', 0)
        
        # Penalty for excessive leverage (exposure > 1.0)
        if exposure > 1.0:
            return -(exposure - 1.0) * self.config.leverage_penalty
        
        return 0.0
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-specific multiplier."""
        return SimpleReturnReward._get_regime_multiplier(self, regime)


class RegimeAdaptiveReward(MultiObjectiveReward):
    """Regime-adaptive reward that adjusts weights based on market conditions."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.regime_weights = self._initialize_regime_weights()
    
    def _initialize_regime_weights(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Initialize regime-specific weight adjustments."""
        return {
            MarketRegime.BULL_MARKET: {
                'return_weight': 0.5,  # Higher weight on returns
                'sharpe_weight': 0.2,
                'risk_penalty': 0.8,   # Lower risk penalty
            },
            MarketRegime.BEAR_MARKET: {
                'return_weight': 0.2,  # Lower weight on returns
                'sharpe_weight': 0.3,
                'risk_penalty': 1.5,   # Higher risk penalty
            },
            MarketRegime.HIGH_VOLATILITY: {
                'return_weight': 0.3,
                'sharpe_weight': 0.4,  # Higher weight on risk-adjusted returns
                'risk_penalty': 2.0,   # Much higher risk penalty
            },
            MarketRegime.SIDEWAYS_MARKET: {
                'return_weight': 0.4,
                'sharpe_weight': 0.3,
                'risk_penalty': 1.0,   # Standard risk penalty
            }
        }
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_value: float,
        trade_results: List[Dict],
        market_regime: MarketRegime,
        portfolio_state: Dict,
        **kwargs
    ) -> float:
        """Calculate regime-adaptive reward."""
        # Temporarily adjust config weights based on regime
        original_weights = {
            'return_weight': self.config.return_weight,
            'sharpe_weight': self.config.sharpe_weight,
            'drawdown_penalty': self.config.drawdown_penalty,
            'var_penalty': self.config.var_penalty,
            'cvar_penalty': self.config.cvar_penalty
        }
        
        # Apply regime-specific adjustments
        regime_adjustments = self.regime_weights.get(market_regime, {})
        
        self.config.return_weight = regime_adjustments.get('return_weight', original_weights['return_weight'])
        self.config.sharpe_weight = regime_adjustments.get('sharpe_weight', original_weights['sharpe_weight'])
        
        risk_multiplier = regime_adjustments.get('risk_penalty', 1.0)
        self.config.drawdown_penalty = original_weights['drawdown_penalty'] * risk_multiplier
        self.config.var_penalty = original_weights['var_penalty'] * risk_multiplier
        self.config.cvar_penalty = original_weights['cvar_penalty'] * risk_multiplier
        
        # Calculate reward with adjusted weights
        reward = super().calculate_reward(
            portfolio_value, previous_value, trade_results, 
            market_regime, portfolio_state, **kwargs
        )
        
        # Restore original weights
        for key, value in original_weights.items():
            setattr(self.config, key, value)
        
        return reward


class RiskMetricsCalculator:
    """Calculator for various risk metrics."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def calculate_maximum_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        
        return np.max(drawdowns)
    
    def calculate_ulcer_index(self, portfolio_values: np.ndarray) -> float:
        """Calculate Ulcer Index (measure of downside risk)."""
        if len(portfolio_values) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max * 100
        
        return np.sqrt(np.mean(drawdowns**2))
    
    def calculate_downside_deviation(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate downside deviation."""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return np.sqrt(np.mean((downside_returns - target_return)**2))
    
    def calculate_tracking_error(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate tracking error against benchmark."""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0
        
        excess_returns = portfolio_returns - benchmark_returns
        return np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def calculate_information_ratio(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio."""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = self.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(excess_returns) * 252 / tracking_error  # Annualized
    
    def calculate_beta(self, portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio beta."""
        if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 2:
            return 1.0
        
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def calculate_treynor_ratio(self, portfolio_returns: np.ndarray, market_returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Treynor ratio."""
        if len(portfolio_returns) == 0:
            return 0.0
        
        beta = self.calculate_beta(portfolio_returns, market_returns)
        
        if beta == 0:
            return 0.0
        
        excess_return = np.mean(portfolio_returns) * 252 - risk_free_rate
        return excess_return / beta


class RewardFunctionFactory:
    """Factory for creating reward functions."""
    
    @staticmethod
    def create_reward_function(
        reward_type: RewardType,
        config: Optional[RewardConfig] = None
    ) -> BaseRewardFunction:
        """Create a reward function of the specified type."""
        if config is None:
            config = RewardConfig()
        
        reward_functions = {
            RewardType.SIMPLE_RETURN: SimpleReturnReward,
            RewardType.SHARPE_RATIO: SharpeRatioReward,
            RewardType.SORTINO_RATIO: SortinoRatioReward,
            RewardType.CALMAR_RATIO: CalmarRatioReward,
            RewardType.MULTI_OBJECTIVE: MultiObjectiveReward,
            RewardType.RISK_ADJUSTED: MultiObjectiveReward,  # Alias
            RewardType.REGIME_ADAPTIVE: RegimeAdaptiveReward
        }
        
        reward_class = reward_functions.get(reward_type)
        if reward_class is None:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        return reward_class(config)


# Portfolio constraint and risk management functions
class PortfolioConstraints:
    """Portfolio-level constraints and risk management rules."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def check_position_size_constraint(
        self,
        symbol: str,
        proposed_position: float,
        current_price: float,
        portfolio_value: float,
        max_position_size: float = 0.2
    ) -> Tuple[bool, str]:
        """Check if position size violates constraints."""
        position_value = proposed_position * current_price
        position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_weight > max_position_size:
            return False, f"Position size {position_weight:.2%} exceeds limit {max_position_size:.2%}"
        
        return True, "OK"
    
    def check_total_exposure_constraint(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        max_exposure: float = 0.8
    ) -> Tuple[bool, str]:
        """Check if total exposure violates constraints."""
        total_position_value = sum(
            positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in positions.keys()
        )
        
        exposure = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        if exposure > max_exposure:
            return False, f"Total exposure {exposure:.2%} exceeds limit {max_exposure:.2%}"
        
        return True, "OK"
    
    def check_concentration_constraint(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        max_concentration: float = 0.4
    ) -> Tuple[bool, str]:
        """Check if portfolio concentration violates constraints."""
        position_values = [
            positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in positions.keys()
        ]
        
        total_value = sum(position_values)
        if total_value <= 0:
            return True, "OK"
        
        max_weight = max(position_values) / total_value
        
        if max_weight > max_concentration:
            return False, f"Max position weight {max_weight:.2%} exceeds limit {max_concentration:.2%}"
        
        return True, "OK"
    
    def check_drawdown_constraint(
        self,
        current_value: float,
        max_value: float,
        max_drawdown: float = 0.15
    ) -> Tuple[bool, str]:
        """Check if drawdown violates constraints."""
        drawdown = (max_value - current_value) / max_value if max_value > 0 else 0
        
        if drawdown > max_drawdown:
            return False, f"Drawdown {drawdown:.2%} exceeds limit {max_drawdown:.2%}"
        
        return True, "OK"
    
    def apply_stop_loss(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        entry_prices: Dict[str, float],
        stop_loss_threshold: float = 0.05
    ) -> List[Dict]:
        """Apply stop-loss rules and return forced sell orders."""
        forced_sells = []
        
        for symbol in positions.keys():
            if positions[symbol] > 0:  # Long position
                current_price = current_prices.get(symbol, 0)
                entry_price = entry_prices.get(symbol, current_price)
                
                if entry_price > 0:
                    loss = (entry_price - current_price) / entry_price
                    
                    if loss > stop_loss_threshold:
                        forced_sells.append({
                            'symbol': symbol,
                            'action_type': 2,  # SELL
                            'position_size': 1.0,  # Sell entire position
                            'reason': 'stop_loss'
                        })
        
        return forced_sells


# Example usage and testing
if __name__ == "__main__":
    # Test reward functions
    config = RewardConfig()
    
    # Create different reward functions
    simple_reward = RewardFunctionFactory.create_reward_function(RewardType.SIMPLE_RETURN, config)
    multi_reward = RewardFunctionFactory.create_reward_function(RewardType.MULTI_OBJECTIVE, config)
    adaptive_reward = RewardFunctionFactory.create_reward_function(RewardType.REGIME_ADAPTIVE, config)
    
    # Test data
    portfolio_value = 105000.0
    previous_value = 100000.0
    trade_results = [{'executed': True, 'cost': 50.0}]
    market_regime = MarketRegime.BULL_MARKET
    portfolio_state = {
        'drawdown': 0.02,
        'exposure': 0.7,
        'positions': {'AAPL': 100, 'GOOGL': 50},
        'current_prices': {'AAPL': 150, 'GOOGL': 2500},
        'portfolio_value': portfolio_value
    }
    
    # Calculate rewards
    simple_r = simple_reward.calculate_reward(
        portfolio_value, previous_value, trade_results, market_regime, portfolio_state
    )
    
    multi_r = multi_reward.calculate_reward(
        portfolio_value, previous_value, trade_results, market_regime, portfolio_state
    )
    
    adaptive_r = adaptive_reward.calculate_reward(
        portfolio_value, previous_value, trade_results, market_regime, portfolio_state
    )
    
    print(f"Simple Return Reward: {simple_r:.4f}")
    print(f"Multi-Objective Reward: {multi_r:.4f}")
    print(f"Regime-Adaptive Reward: {adaptive_r:.4f}")
    
    # Test risk metrics
    risk_calc = RiskMetricsCalculator(config)
    returns = np.random.normal(0.001, 0.02, 100)  # Simulated returns
    
    var = risk_calc.calculate_var(returns, 0.05)
    cvar = risk_calc.calculate_cvar(returns, 0.05)
    
    print(f"VaR (5%): {var:.4f}")
    print(f"CVaR (5%): {cvar:.4f}")
    
    # Test portfolio constraints
    constraints = PortfolioConstraints(config)
    
    valid, msg = constraints.check_position_size_constraint(
        'AAPL', 1000, 150, 100000, 0.2
    )
    print(f"Position size check: {valid}, {msg}")