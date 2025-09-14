"""
Reward calculation strategies for trading environments.
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class RewardCalculator(ABC):
    """Abstract base class for reward calculation strategies."""
    
    @abstractmethod
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        initial_balance: float,
        returns_history: List[float],
        trade_results: List[Dict],
        config: 'TradingConfig'
    ) -> float:
        """Calculate reward for the current step."""
        pass


class RiskAdjustedRewardCalculator(RewardCalculator):
    """Risk-adjusted reward calculator with Sharpe ratio and drawdown penalties."""
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        initial_balance: float,
        returns_history: List[float],
        trade_results: List[Dict],
        config: 'TradingConfig'
    ) -> float:
        """Calculate risk-adjusted reward."""
        # Portfolio return
        if previous_portfolio_value > 0:
            portfolio_return = (
                (portfolio_value - previous_portfolio_value) / 
                previous_portfolio_value
            )
        else:
            portfolio_return = (
                (portfolio_value - initial_balance) / initial_balance
            )
        
        # Sharpe ratio component
        sharpe_ratio = self._calculate_sharpe_ratio(returns_history, config)
        
        # Drawdown penalty
        max_portfolio_value = max(
            [initial_balance] + 
            [initial_balance * (1 + r) for r in returns_history]
        )
        current_drawdown = (
            (max_portfolio_value - portfolio_value) / max_portfolio_value
        )
        drawdown_penalty = -current_drawdown * 2.0
        
        # Transaction cost penalty
        transaction_penalty = self._calculate_transaction_penalty(
            trade_results, portfolio_value, config
        )
        
        # Combine components
        reward = (
            portfolio_return * config.reward_scaling +
            sharpe_ratio * 0.1 +
            drawdown_penalty +
            transaction_penalty
        )
        
        return reward
    
    def _calculate_sharpe_ratio(
        self, 
        returns_history: List[float], 
        config: 'TradingConfig'
    ) -> float:
        """Calculate Sharpe ratio from returns history."""
        if len(returns_history) < 2:
            return 0.0
        
        returns_array = np.array(returns_history[-20:])  # Last 20 returns
        excess_returns = returns_array - (config.risk_free_rate / 252)
        
        if np.std(excess_returns) > 0:
            return np.mean(excess_returns) / np.std(excess_returns)
        return 0.0
    
    def _calculate_transaction_penalty(
        self, 
        trade_results: List[Dict], 
        portfolio_value: float,
        config: 'TradingConfig'
    ) -> float:
        """Calculate penalty for transaction costs."""
        if portfolio_value <= 0:
            return 0.0
        
        total_cost = sum(
            abs(result['cost']) * config.transaction_cost 
            for result in trade_results if result['executed']
        )
        return -total_cost / portfolio_value


class SimpleReturnRewardCalculator(RewardCalculator):
    """Simple return-based reward calculator."""
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        initial_balance: float,
        returns_history: List[float],
        trade_results: List[Dict],
        config: 'TradingConfig'
    ) -> float:
        """Calculate simple return-based reward."""
        if previous_portfolio_value > 0:
            return (
                (portfolio_value - previous_portfolio_value) / 
                previous_portfolio_value * config.reward_scaling
            )
        return 0.0