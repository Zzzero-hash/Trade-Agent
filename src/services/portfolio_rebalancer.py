"""
Dynamic portfolio rebalancing with transaction cost optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.models.portfolio import Portfolio, Position
from src.services.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationMethod, OptimizationConstraints,
    RebalancingConfig, OptimizationResult
)


@dataclass
class RebalanceRecommendation:
    """Recommendation for portfolio rebalancing."""
    should_rebalance: bool
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    weight_deviations: Dict[str, float]
    expected_transaction_costs: float
    expected_benefit: float
    trades_required: Dict[str, float]  # Symbol -> quantity change
    reason: str
    urgency_score: float  # 0-1, higher means more urgent


@dataclass
class RebalanceExecution:
    """Result of rebalancing execution."""
    executed: bool
    trades_executed: Dict[str, float]
    actual_transaction_costs: float
    new_weights: Dict[str, float]
    execution_time: datetime
    slippage: float
    message: str


class PortfolioRebalancer:
    """
    Dynamic portfolio rebalancer with transaction cost optimization.
    """
    
    def __init__(self, optimizer: PortfolioOptimizer):
        """
        Initialize portfolio rebalancer.
        
        Args:
            optimizer: Portfolio optimizer instance
        """
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        self.last_rebalance_time: Optional[datetime] = None
    
    def should_rebalance(
        self,
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        config: RebalancingConfig
    ) -> RebalanceRecommendation:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            portfolio: Current portfolio
            target_weights: Target allocation weights
            current_prices: Current market prices
            config: Rebalancing configuration
        
        Returns:
            RebalanceRecommendation with rebalancing decision and details
        """
        # Calculate current weights
        current_weights = self._calculate_current_weights(portfolio, current_prices)
        
        # Calculate weight deviations
        weight_deviations = {}
        max_deviation = 0.0
        total_deviation = 0.0
        
        for symbol in target_weights:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights[symbol]
            deviation = abs(current_weight - target_weight)
            weight_deviations[symbol] = current_weight - target_weight
            max_deviation = max(max_deviation, deviation)
            total_deviation += deviation
        
        # Check time-based constraints
        time_check = self._check_time_constraints(config)
        if not time_check["can_rebalance"]:
            return RebalanceRecommendation(
                should_rebalance=False,
                target_weights=target_weights,
                current_weights=current_weights,
                weight_deviations=weight_deviations,
                expected_transaction_costs=0.0,
                expected_benefit=0.0,
                trades_required={},
                reason=time_check["reason"],
                urgency_score=0.0
            )
        
        # Calculate expected transaction costs
        trades_required = self._calculate_required_trades(
            portfolio, target_weights, current_prices
        )
        expected_transaction_costs = self._estimate_transaction_costs(
            trades_required, current_prices, config
        )
        
        # Determine if rebalancing is needed based on method
        should_rebalance = False
        reason = ""
        urgency_score = 0.0
        
        if config.threshold_method == "percentage":
            if max_deviation > config.rebalance_threshold:
                should_rebalance = True
                reason = f"Maximum weight deviation ({max_deviation:.3f}) exceeds threshold ({config.rebalance_threshold:.3f})"
                urgency_score = min(max_deviation / config.rebalance_threshold, 1.0)
        
        elif config.threshold_method == "absolute":
            portfolio_value = portfolio.total_value
            max_absolute_deviation = max_deviation * portfolio_value
            threshold_value = config.rebalance_threshold
            
            if max_absolute_deviation > threshold_value:
                should_rebalance = True
                reason = f"Maximum absolute deviation (${max_absolute_deviation:.2f}) exceeds threshold (${threshold_value:.2f})"
                urgency_score = min(max_absolute_deviation / threshold_value, 1.0)
        
        elif config.threshold_method == "volatility":
            # Use volatility-adjusted thresholds (simplified)
            volatility_adjusted_threshold = config.rebalance_threshold * (1 + total_deviation)
            if max_deviation > volatility_adjusted_threshold:
                should_rebalance = True
                reason = f"Volatility-adjusted deviation exceeds threshold"
                urgency_score = min(max_deviation / volatility_adjusted_threshold, 1.0)
        
        # Check if transaction costs are too high
        if should_rebalance and expected_transaction_costs > config.transaction_cost_threshold * portfolio.total_value:
            should_rebalance = False
            reason = f"Transaction costs too high: {expected_transaction_costs:.2f} > {config.transaction_cost_threshold * portfolio.total_value:.2f}"
            urgency_score = 0.0
        
        # Estimate expected benefit from rebalancing
        expected_benefit = self._estimate_rebalancing_benefit(
            current_weights, target_weights, portfolio.total_value
        )
        
        return RebalanceRecommendation(
            should_rebalance=should_rebalance,
            target_weights=target_weights,
            current_weights=current_weights,
            weight_deviations=weight_deviations,
            expected_transaction_costs=expected_transaction_costs,
            expected_benefit=expected_benefit,
            trades_required=trades_required,
            reason=reason,
            urgency_score=urgency_score
        )
    
    def execute_rebalance(
        self,
        portfolio: Portfolio,
        recommendation: RebalanceRecommendation,
        current_prices: Dict[str, float],
        execution_config: Optional[Dict] = None
    ) -> RebalanceExecution:
        """
        Execute portfolio rebalancing.
        
        Args:
            portfolio: Current portfolio
            recommendation: Rebalancing recommendation
            current_prices: Current market prices
            execution_config: Execution configuration (slippage, etc.)
        
        Returns:
            RebalanceExecution with execution results
        """
        if not recommendation.should_rebalance:
            return RebalanceExecution(
                executed=False,
                trades_executed={},
                actual_transaction_costs=0.0,
                new_weights={},
                execution_time=datetime.now(),
                slippage=0.0,
                message="No rebalancing needed"
            )
        
        execution_config = execution_config or {}
        slippage_rate = execution_config.get("slippage_rate", 0.001)
        
        try:
            # Execute trades
            trades_executed = {}
            actual_transaction_costs = 0.0
            total_slippage = 0.0
            
            for symbol, quantity_change in recommendation.trades_required.items():
                if abs(quantity_change) < 0.01:  # Skip very small trades
                    continue
                
                current_price = current_prices[symbol]
                
                # Apply slippage
                if quantity_change > 0:  # Buying
                    execution_price = current_price * (1 + slippage_rate)
                else:  # Selling
                    execution_price = current_price * (1 - slippage_rate)
                
                # Calculate transaction cost
                trade_value = abs(quantity_change * execution_price)
                transaction_cost = trade_value * 0.001  # 0.1% transaction cost
                
                # Update portfolio
                if symbol in portfolio.positions:
                    position = portfolio.positions[symbol]
                    new_quantity = position.quantity + quantity_change
                    
                    if new_quantity == 0:
                        # Close position
                        portfolio.remove_position(symbol)
                    else:
                        # Update position
                        if quantity_change > 0:
                            # Buying more
                            total_cost = position.quantity * position.avg_cost + quantity_change * execution_price
                            new_avg_cost = total_cost / new_quantity
                        else:
                            # Selling some
                            new_avg_cost = position.avg_cost  # Keep same avg cost
                        
                        updated_position = Position(
                            symbol=symbol,
                            quantity=new_quantity,
                            avg_cost=new_avg_cost,
                            current_price=current_price,
                            unrealized_pnl=new_quantity * (current_price - new_avg_cost)
                        )
                        portfolio.add_position(updated_position)
                else:
                    # New position
                    if quantity_change > 0:
                        new_position = Position(
                            symbol=symbol,
                            quantity=quantity_change,
                            avg_cost=execution_price,
                            current_price=current_price,
                            unrealized_pnl=quantity_change * (current_price - execution_price)
                        )
                        portfolio.add_position(new_position)
                
                # Update cash balance
                cash_change = -quantity_change * execution_price - transaction_cost
                portfolio.cash_balance += cash_change
                
                trades_executed[symbol] = quantity_change
                actual_transaction_costs += transaction_cost
                total_slippage += abs(execution_price - current_price) * abs(quantity_change)
            
            # Calculate new weights
            new_weights = self._calculate_current_weights(portfolio, current_prices)
            
            # Update last rebalance time
            self.last_rebalance_time = datetime.now()
            
            return RebalanceExecution(
                executed=True,
                trades_executed=trades_executed,
                actual_transaction_costs=actual_transaction_costs,
                new_weights=new_weights,
                execution_time=self.last_rebalance_time,
                slippage=total_slippage,
                message=f"Rebalancing executed successfully. {len(trades_executed)} trades executed."
            )
            
        except Exception as e:
            self.logger.error(f"Rebalancing execution failed: {e}")
            return RebalanceExecution(
                executed=False,
                trades_executed={},
                actual_transaction_costs=0.0,
                new_weights={},
                execution_time=datetime.now(),
                slippage=0.0,
                message=f"Rebalancing execution failed: {str(e)}"
            )
    
    def optimize_and_rebalance(
        self,
        portfolio: Portfolio,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_prices: Dict[str, float],
        optimization_method: OptimizationMethod = OptimizationMethod.MAXIMUM_SHARPE,
        optimization_constraints: Optional[OptimizationConstraints] = None,
        rebalancing_config: Optional[RebalancingConfig] = None
    ) -> Tuple[OptimizationResult, RebalanceRecommendation, Optional[RebalanceExecution]]:
        """
        Optimize portfolio and execute rebalancing if needed.
        
        Args:
            portfolio: Current portfolio
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix
            current_prices: Current market prices
            optimization_method: Optimization method to use
            optimization_constraints: Optimization constraints
            rebalancing_config: Rebalancing configuration
        
        Returns:
            Tuple of (optimization_result, rebalance_recommendation, rebalance_execution)
        """
        if optimization_constraints is None:
            optimization_constraints = OptimizationConstraints()
        
        if rebalancing_config is None:
            rebalancing_config = RebalancingConfig()
        
        # Get current weights for transaction cost calculation
        current_weights = self._calculate_current_weights(portfolio, current_prices)
        
        # Optimize portfolio
        optimization_result = self.optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=optimization_method,
            constraints=optimization_constraints,
            current_weights=current_weights
        )
        
        if not optimization_result.success:
            return optimization_result, RebalanceRecommendation(
                should_rebalance=False,
                target_weights={},
                current_weights=current_weights,
                weight_deviations={},
                expected_transaction_costs=0.0,
                expected_benefit=0.0,
                trades_required={},
                reason="Optimization failed",
                urgency_score=0.0
            ), None
        
        # Check if rebalancing is needed
        rebalance_recommendation = self.should_rebalance(
            portfolio=portfolio,
            target_weights=optimization_result.weights,
            current_prices=current_prices,
            config=rebalancing_config
        )
        
        # Execute rebalancing if recommended
        rebalance_execution = None
        if rebalance_recommendation.should_rebalance:
            rebalance_execution = self.execute_rebalance(
                portfolio=portfolio,
                recommendation=rebalance_recommendation,
                current_prices=current_prices
            )
        
        return optimization_result, rebalance_recommendation, rebalance_execution
    
    def _calculate_current_weights(
        self,
        portfolio: Portfolio,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        if portfolio.total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in portfolio.positions.items():
            if symbol in current_prices:
                market_value = abs(position.quantity) * current_prices[symbol]
                weights[symbol] = market_value / portfolio.total_value
        
        return weights
    
    def _calculate_required_trades(
        self,
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate required trades to reach target weights."""
        trades = {}
        current_weights = self._calculate_current_weights(portfolio, current_prices)
        
        for symbol in target_weights:
            target_weight = target_weights[symbol]
            current_weight = current_weights.get(symbol, 0.0)
            
            target_value = target_weight * portfolio.total_value
            current_value = current_weight * portfolio.total_value
            
            value_change = target_value - current_value
            
            if symbol in current_prices and current_prices[symbol] > 0:
                quantity_change = value_change / current_prices[symbol]
                if abs(quantity_change) > 0.01:  # Only include meaningful trades
                    trades[symbol] = quantity_change
        
        return trades
    
    def _estimate_transaction_costs(
        self,
        trades: Dict[str, float],
        current_prices: Dict[str, float],
        config: RebalancingConfig
    ) -> float:
        """Estimate transaction costs for trades."""
        total_cost = 0.0
        
        for symbol, quantity in trades.items():
            if symbol in current_prices:
                trade_value = abs(quantity * current_prices[symbol])
                transaction_cost = trade_value * 0.001  # 0.1% transaction cost
                total_cost += transaction_cost
        
        return total_cost
    
    def _estimate_rebalancing_benefit(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """Estimate expected benefit from rebalancing."""
        # Simplified benefit estimation based on weight deviations
        total_deviation = sum(
            abs(current_weights.get(symbol, 0.0) - target_weight)
            for symbol, target_weight in target_weights.items()
        )
        
        # Assume benefit is proportional to deviation reduction
        estimated_benefit = total_deviation * portfolio_value * 0.01  # 1% of deviation value
        
        return estimated_benefit
    
    def _check_time_constraints(self, config: RebalancingConfig) -> Dict[str, any]:
        """Check time-based rebalancing constraints."""
        now = datetime.now()
        
        if self.last_rebalance_time is None:
            return {"can_rebalance": True, "reason": "First rebalancing"}
        
        time_since_last = now - self.last_rebalance_time
        min_interval = timedelta(days=config.min_rebalance_interval)
        max_interval = timedelta(days=config.max_rebalance_interval)
        
        if time_since_last < min_interval:
            return {
                "can_rebalance": False,
                "reason": f"Too soon since last rebalance ({time_since_last.days} days < {config.min_rebalance_interval} days)"
            }
        
        if time_since_last > max_interval:
            return {
                "can_rebalance": True,
                "reason": f"Maximum interval exceeded ({time_since_last.days} days > {config.max_rebalance_interval} days)"
            }
        
        return {"can_rebalance": True, "reason": "Time constraints satisfied"}