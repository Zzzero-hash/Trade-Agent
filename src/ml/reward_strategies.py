"""
Reward Strategy Implementations for Trading Environment.

This module implements the Strategy pattern for different reward calculation
methods, allowing flexible reward function selection and experimentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of reward strategies available."""
    SIMPLE_RETURN = "simple_return"
    SHARPE_RATIO = "sharpe_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    MULTI_OBJECTIVE = "multi_objective"
    SORTINO_RATIO = "sortino_ratio"


@dataclass
class RewardContext:
    """Context information for reward calculation."""
    portfolio_value: float
    previous_portfolio_value: float
    initial_balance: float
    max_portfolio_value: float
    cash_balance: float
    positions: Dict[str, float]
    current_prices: Dict[str, float]
    trade_results: List[Dict]
    returns_history: List[float]
    risk_free_rate: float = 0.02
    transaction_cost_rate: float = 0.001


class RewardStrategy(ABC):
    """Abstract base class for reward calculation strategies."""
    
    @abstractmethod
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate reward based on the current context."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this reward strategy."""
        pass
    
    def get_strategy_info(self) -> Dict[str, str]:
        """Get information about this reward strategy."""
        return {
            'name': self.get_strategy_name(),
            'description': self.__doc__ or "No description available"
        }


class SimpleReturnReward(RewardStrategy):
    """Simple portfolio return-based reward strategy."""
    
    def __init__(self, scaling_factor: float = 1000.0):
        """
        Initialize simple return reward strategy.
        
        Args:
            scaling_factor: Factor to scale rewards for better learning
        """
        self.scaling_factor = scaling_factor
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate reward based on simple portfolio return."""
        if context.previous_portfolio_value <= 0:
            return 0.0
        
        portfolio_return = (
            (context.portfolio_value - context.previous_portfolio_value) / 
            context.previous_portfolio_value
        )
        
        return portfolio_return * self.scaling_factor
    
    def get_strategy_name(self) -> str:
        return "Simple Return"


class SharpeRatioReward(RewardStrategy):
    """Sharpe ratio-based reward strategy."""
    
    def __init__(
        self, 
        scaling_factor: float = 1000.0,
        lookback_window: int = 20,
        min_periods: int = 5
    ):
        """
        Initialize Sharpe ratio reward strategy.
        
        Args:
            scaling_factor: Factor to scale rewards
            lookback_window: Number of periods for Sharpe calculation
            min_periods: Minimum periods required for calculation
        """
        self.scaling_factor = scaling_factor
        self.lookback_window = lookback_window
        self.min_periods = min_periods
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate reward based on Sharpe ratio."""
        if len(context.returns_history) < self.min_periods:
            # Fall back to simple return for early periods
            return SimpleReturnReward(self.scaling_factor).calculate_reward(context)
        
        # Get recent returns
        recent_returns = np.array(context.returns_history[-self.lookback_window:])
        
        # Calculate excess returns
        daily_risk_free_rate = context.risk_free_rate / 252
        excess_returns = recent_returns - daily_risk_free_rate
        
        # Calculate Sharpe ratio
        if np.std(excess_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        else:
            sharpe_ratio = 0.0
        
        # Current period return
        current_return = (
            (context.portfolio_value - context.previous_portfolio_value) / 
            context.previous_portfolio_value
        ) if context.previous_portfolio_value > 0 else 0.0
        
        # Combine current return with Sharpe ratio
        reward = (current_return + sharpe_ratio * 0.1) * self.scaling_factor
        
        return reward
    
    def get_strategy_name(self) -> str:
        return "Sharpe Ratio"


class RiskAdjustedReward(RewardStrategy):
    """Risk-adjusted reward with drawdown penalties."""
    
    def __init__(
        self,
        scaling_factor: float = 1000.0,
        drawdown_penalty: float = 2.0,
        transaction_penalty: float = 1.0,
        volatility_penalty: float = 0.5
    ):
        """
        Initialize risk-adjusted reward strategy.
        
        Args:
            scaling_factor: Factor to scale rewards
            drawdown_penalty: Penalty multiplier for drawdowns
            transaction_penalty: Penalty multiplier for transaction costs
            volatility_penalty: Penalty multiplier for volatility
        """
        self.scaling_factor = scaling_factor
        self.drawdown_penalty = drawdown_penalty
        self.transaction_penalty = transaction_penalty
        self.volatility_penalty = volatility_penalty
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate risk-adjusted reward."""
        # Portfolio return
        portfolio_return = (
            (context.portfolio_value - context.previous_portfolio_value) / 
            context.previous_portfolio_value
        ) if context.previous_portfolio_value > 0 else 0.0
        
        # Drawdown penalty
        current_drawdown = (
            (context.max_portfolio_value - context.portfolio_value) / 
            context.max_portfolio_value
        ) if context.max_portfolio_value > 0 else 0.0
        
        drawdown_component = -current_drawdown * self.drawdown_penalty
        
        # Transaction cost penalty
        transaction_costs = sum(
            abs(result.get('cost', 0)) * context.transaction_cost_rate
            for result in context.trade_results
            if result.get('executed', False)
        )
        
        transaction_component = -(transaction_costs / context.portfolio_value) * self.transaction_penalty
        
        # Volatility penalty (if enough history)
        volatility_component = 0.0
        if len(context.returns_history) >= 10:
            recent_volatility = np.std(context.returns_history[-10:])
            volatility_component = -recent_volatility * self.volatility_penalty
        
        # Combine components
        total_reward = (
            portfolio_return + 
            drawdown_component + 
            transaction_component + 
            volatility_component
        ) * self.scaling_factor
        
        return total_reward
    
    def get_strategy_name(self) -> str:
        return "Risk Adjusted"


class MultiObjectiveReward(RewardStrategy):
    """Multi-objective reward combining multiple metrics."""
    
    def __init__(
        self,
        return_weight: float = 0.6,
        sharpe_weight: float = 0.2,
        drawdown_weight: float = 0.1,
        diversification_weight: float = 0.1,
        scaling_factor: float = 1000.0
    ):
        """
        Initialize multi-objective reward strategy.
        
        Args:
            return_weight: Weight for return component
            sharpe_weight: Weight for Sharpe ratio component
            drawdown_weight: Weight for drawdown component
            diversification_weight: Weight for diversification component
            scaling_factor: Factor to scale rewards
        """
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.diversification_weight = diversification_weight
        self.scaling_factor = scaling_factor
        
        # Normalize weights
        total_weight = sum([return_weight, sharpe_weight, drawdown_weight, diversification_weight])
        self.return_weight /= total_weight
        self.sharpe_weight /= total_weight
        self.drawdown_weight /= total_weight
        self.diversification_weight /= total_weight
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate multi-objective reward."""
        # Return component
        portfolio_return = (
            (context.portfolio_value - context.previous_portfolio_value) / 
            context.previous_portfolio_value
        ) if context.previous_portfolio_value > 0 else 0.0
        
        return_component = portfolio_return * self.return_weight
        
        # Sharpe component
        sharpe_component = 0.0
        if len(context.returns_history) >= 5:
            recent_returns = np.array(context.returns_history[-20:])
            excess_returns = recent_returns - (context.risk_free_rate / 252)
            
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
                sharpe_component = sharpe_ratio * self.sharpe_weight * 0.1
        
        # Drawdown component (penalty)
        current_drawdown = (
            (context.max_portfolio_value - context.portfolio_value) / 
            context.max_portfolio_value
        ) if context.max_portfolio_value > 0 else 0.0
        
        drawdown_component = -current_drawdown * self.drawdown_weight
        
        # Diversification component
        diversification_component = self._calculate_diversification_bonus(context)
        
        # Combine components
        total_reward = (
            return_component + 
            sharpe_component + 
            drawdown_component + 
            diversification_component
        ) * self.scaling_factor
        
        return total_reward
    
    def _calculate_diversification_bonus(self, context: RewardContext) -> float:
        """Calculate diversification bonus based on position distribution."""
        if not context.positions or context.portfolio_value <= 0:
            return 0.0
        
        # Calculate position weights
        position_values = []
        for symbol, quantity in context.positions.items():
            if symbol in context.current_prices:
                position_value = quantity * context.current_prices[symbol]
                position_values.append(position_value)
        
        if not position_values:
            return 0.0
        
        total_position_value = sum(position_values)
        if total_position_value <= 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (lower is more diversified)
        weights = [value / total_position_value for value in position_values]
        hhi = sum(weight ** 2 for weight in weights)
        
        # Convert to diversification bonus (1/n is perfectly diversified)
        n_positions = len([w for w in weights if w > 0.01])  # Positions > 1%
        perfect_diversification = 1.0 / max(n_positions, 1)
        
        # Bonus for being closer to perfect diversification
        diversification_bonus = (perfect_diversification - hhi) * self.diversification_weight
        
        return max(0.0, diversification_bonus)  # Only positive bonus
    
    def get_strategy_name(self) -> str:
        return "Multi-Objective"


class SortinoRatioReward(RewardStrategy):
    """Sortino ratio-based reward (focuses on downside risk)."""
    
    def __init__(
        self,
        scaling_factor: float = 1000.0,
        lookback_window: int = 20,
        min_periods: int = 5,
        target_return: float = 0.0
    ):
        """
        Initialize Sortino ratio reward strategy.
        
        Args:
            scaling_factor: Factor to scale rewards
            lookback_window: Number of periods for calculation
            min_periods: Minimum periods required
            target_return: Target return for Sortino calculation
        """
        self.scaling_factor = scaling_factor
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.target_return = target_return
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate reward based on Sortino ratio."""
        if len(context.returns_history) < self.min_periods:
            return SimpleReturnReward(self.scaling_factor).calculate_reward(context)
        
        # Get recent returns
        recent_returns = np.array(context.returns_history[-self.lookback_window:])
        
        # Calculate excess returns over target
        excess_returns = recent_returns - self.target_return
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        else:
            downside_deviation = 0.0
        
        # Calculate Sortino ratio
        if downside_deviation > 0:
            sortino_ratio = np.mean(excess_returns) / downside_deviation
        else:
            sortino_ratio = np.mean(excess_returns) if np.mean(excess_returns) > 0 else 0.0
        
        # Current period return
        current_return = (
            (context.portfolio_value - context.previous_portfolio_value) / 
            context.previous_portfolio_value
        ) if context.previous_portfolio_value > 0 else 0.0
        
        # Combine current return with Sortino ratio
        reward = (current_return + sortino_ratio * 0.1) * self.scaling_factor
        
        return reward
    
    def get_strategy_name(self) -> str:
        return "Sortino Ratio"


class RewardStrategyFactory:
    """Factory for creating reward strategies."""
    
    _strategies = {
        RewardType.SIMPLE_RETURN: SimpleReturnReward,
        RewardType.SHARPE_RATIO: SharpeRatioReward,
        RewardType.RISK_ADJUSTED: RiskAdjustedReward,
        RewardType.MULTI_OBJECTIVE: MultiObjectiveReward,
        RewardType.SORTINO_RATIO: SortinoRatioReward,
    }
    
    @classmethod
    def create_strategy(
        self, 
        strategy_type: RewardType, 
        **kwargs
    ) -> RewardStrategy:
        """
        Create a reward strategy instance.
        
        Args:
            strategy_type: Type of reward strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            Reward strategy instance
            
        Raises:
            ValueError: If strategy type is not supported
        """
        if strategy_type not in self._strategies:
            raise ValueError(f"Unsupported reward strategy: {strategy_type}")
        
        strategy_class = self._strategies[strategy_type]
        return strategy_class(**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> List[RewardType]:
        """Get list of available reward strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls, strategy_type: RewardType) -> Dict[str, str]:
        """Get information about a specific strategy."""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Create temporary instance to get info
        temp_strategy = cls._strategies[strategy_type]()
        return temp_strategy.get_strategy_info()


# Convenience function for creating strategies
def create_reward_strategy(strategy_name: str, **kwargs) -> RewardStrategy:
    """
    Create reward strategy by name.
    
    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy parameters
        
    Returns:
        Reward strategy instance
    """
    try:
        strategy_type = RewardType(strategy_name.lower())
        return RewardStrategyFactory.create_strategy(strategy_type, **kwargs)
    except ValueError:
        raise ValueError(f"Unknown reward strategy: {strategy_name}")