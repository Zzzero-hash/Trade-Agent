"""
Advanced portfolio optimization using Modern Portfolio Theory and other strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
from scipy.stats import norm
import logging
from datetime import datetime, timedelta

from src.models.portfolio import Portfolio, Position


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    FACTOR_BASED = "factor_based"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector_weight: Optional[float] = None
    min_positions: Optional[int] = None
    max_positions: Optional[int] = None
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    transaction_cost: float = 0.001
    min_trade_size: float = 100.0


@dataclass
class RebalancingConfig:
    """Configuration for portfolio rebalancing."""
    threshold_method: str = "percentage"  # "percentage", "absolute", "volatility"
    rebalance_threshold: float = 0.05  # 5% deviation
    min_rebalance_interval: int = 7  # days
    max_rebalance_interval: int = 30  # days
    transaction_cost_threshold: float = 0.002  # 0.2%
    consider_tax_implications: bool = False


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_method: OptimizationMethod
    success: bool
    message: str
    transaction_costs: float = 0.0
    turnover: float = 0.0


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer implementing Modern Portfolio Theory
    and other optimization strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MAXIMUM_SHARPE,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            method: Optimization method to use
            constraints: Optimization constraints
            current_weights: Current portfolio weights for transaction cost calculation
        
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        try:
            if method == OptimizationMethod.MEAN_VARIANCE:
                return self._optimize_mean_variance(
                    expected_returns, covariance_matrix, constraints, current_weights
                )
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                return self._optimize_maximum_sharpe(
                    expected_returns, covariance_matrix, constraints, current_weights
                )
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                return self._optimize_minimum_variance(
                    covariance_matrix, constraints, current_weights
                )
            elif method == OptimizationMethod.RISK_PARITY:
                return self._optimize_risk_parity(
                    covariance_matrix, constraints, current_weights
                )
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                return self._optimize_black_litterman(
                    expected_returns, covariance_matrix, constraints, current_weights
                )
            elif method == OptimizationMethod.FACTOR_BASED:
                return self._optimize_factor_based(
                    expected_returns, covariance_matrix, constraints, current_weights
                )
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
                
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=method,
                success=False,
                message=f"Optimization failed: {str(e)}"
            )
    
    def _optimize_maximum_sharpe(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        current_weights: Optional[Dict[str, float]]
    ) -> OptimizationResult:
        """Optimize for maximum Sharpe ratio."""
        n_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Add transaction cost penalty if current weights provided
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in expected_returns.index])
                turnover = np.sum(np.abs(weights - current_w))
                transaction_cost = turnover * constraints.transaction_cost
                sharpe -= transaction_cost
            
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights_dict = dict(zip(expected_returns.index, result.x))
            portfolio_return = np.dot(result.x, expected_returns)
            portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Calculate transaction costs and turnover
            transaction_costs = 0.0
            turnover = 0.0
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in expected_returns.index])
                turnover = np.sum(np.abs(result.x - current_w))
                transaction_costs = turnover * constraints.transaction_cost
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=portfolio_return,
                expected_volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                optimization_method=OptimizationMethod.MAXIMUM_SHARPE,
                success=True,
                message="Optimization successful",
                transaction_costs=transaction_costs,
                turnover=turnover
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.MAXIMUM_SHARPE,
                success=False,
                message=f"Optimization failed: {result.message}"
            )
    
    def _optimize_minimum_variance(
        self,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        current_weights: Optional[Dict[str, float]]
    ) -> OptimizationResult:
        """Optimize for minimum variance."""
        n_assets = len(covariance_matrix)
        
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Add transaction cost penalty
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in covariance_matrix.index])
                turnover = np.sum(np.abs(weights - current_w))
                transaction_cost = turnover * constraints.transaction_cost
                portfolio_vol += transaction_cost
            
            return portfolio_vol
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights_dict = dict(zip(covariance_matrix.index, result.x))
            portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
            
            # Calculate transaction costs and turnover
            transaction_costs = 0.0
            turnover = 0.0
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in covariance_matrix.index])
                turnover = np.sum(np.abs(result.x - current_w))
                transaction_costs = turnover * constraints.transaction_cost
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=0.0,  # Not optimizing for return
                expected_volatility=portfolio_vol,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.MINIMUM_VARIANCE,
                success=True,
                message="Optimization successful",
                transaction_costs=transaction_costs,
                turnover=turnover
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.MINIMUM_VARIANCE,
                success=False,
                message=f"Optimization failed: {result.message}"
            )
    
    def _optimize_risk_parity(
        self,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        current_weights: Optional[Dict[str, float]]
    ) -> OptimizationResult:
        """Optimize for risk parity (equal risk contribution)."""
        n_assets = len(covariance_matrix)
        
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            if portfolio_vol == 0:
                return np.inf
            
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_contrib = portfolio_vol / n_assets
            risk_parity_error = np.sum((contrib - target_contrib) ** 2)
            
            # Add transaction cost penalty
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in covariance_matrix.index])
                turnover = np.sum(np.abs(weights - current_w))
                transaction_cost = turnover * constraints.transaction_cost
                risk_parity_error += transaction_cost * 1000  # Scale transaction cost
            
            return risk_parity_error
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess - equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights_dict = dict(zip(covariance_matrix.index, result.x))
            portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
            
            # Calculate transaction costs and turnover
            transaction_costs = 0.0
            turnover = 0.0
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in covariance_matrix.index])
                turnover = np.sum(np.abs(result.x - current_w))
                transaction_costs = turnover * constraints.transaction_cost
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=0.0,  # Not optimizing for return
                expected_volatility=portfolio_vol,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.RISK_PARITY,
                success=True,
                message="Risk parity optimization successful",
                transaction_costs=transaction_costs,
                turnover=turnover
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.RISK_PARITY,
                success=False,
                message=f"Risk parity optimization failed: {result.message}"
            )
    
    def _optimize_mean_variance(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        current_weights: Optional[Dict[str, float]]
    ) -> OptimizationResult:
        """Optimize using mean-variance optimization with target return."""
        if constraints.target_return is None:
            # Default to maximum Sharpe if no target return specified
            return self._optimize_maximum_sharpe(expected_returns, covariance_matrix, constraints, current_weights)
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Add transaction cost penalty
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in expected_returns.index])
                turnover = np.sum(np.abs(weights - current_w))
                transaction_cost = turnover * constraints.transaction_cost
                portfolio_vol += transaction_cost
            
            return portfolio_vol
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - constraints.target_return}  # Target return
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights_dict = dict(zip(expected_returns.index, result.x))
            portfolio_return = np.dot(result.x, expected_returns)
            portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Calculate transaction costs and turnover
            transaction_costs = 0.0
            turnover = 0.0
            if current_weights is not None:
                current_w = np.array([current_weights.get(symbol, 0.0) for symbol in expected_returns.index])
                turnover = np.sum(np.abs(result.x - current_w))
                transaction_costs = turnover * constraints.transaction_cost
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=portfolio_return,
                expected_volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                optimization_method=OptimizationMethod.MEAN_VARIANCE,
                success=True,
                message="Mean-variance optimization successful",
                transaction_costs=transaction_costs,
                turnover=turnover
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.MEAN_VARIANCE,
                success=False,
                message=f"Mean-variance optimization failed: {result.message}"
            )
    
    def _optimize_black_litterman(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        current_weights: Optional[Dict[str, float]]
    ) -> OptimizationResult:
        """Optimize using Black-Litterman model."""
        # Simplified Black-Litterman implementation
        # In practice, this would require market cap weights and investor views
        
        # Use market cap weights as prior (simplified as equal weights here)
        market_weights = np.array([1.0 / len(expected_returns)] * len(expected_returns))
        
        # Risk aversion parameter (typical value)
        risk_aversion = 3.0
        
        # Implied returns from market weights
        implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        # Combine with expected returns (simplified - no investor views)
        tau = 0.05  # Scaling factor
        bl_returns = pd.Series(implied_returns, index=expected_returns.index)
        
        # Use the Black-Litterman returns for optimization
        result = self._optimize_maximum_sharpe(bl_returns, covariance_matrix, constraints, current_weights)
        
        # Update the optimization method in the result
        if result.success:
            result.optimization_method = OptimizationMethod.BLACK_LITTERMAN
        
        return result
    
    def _optimize_factor_based(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        current_weights: Optional[Dict[str, float]]
    ) -> OptimizationResult:
        """Optimize using factor-based approach."""
        # Simplified factor-based optimization
        # In practice, this would use factor loadings and factor returns
        
        # For now, use a momentum-based factor approach
        # Weight assets based on their expected returns relative to volatility
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        risk_adjusted_returns = expected_returns / individual_vols
        
        # Normalize to create weights
        weights = risk_adjusted_returns / risk_adjusted_returns.sum()
        weights = np.maximum(weights, constraints.min_weight)
        weights = np.minimum(weights, constraints.max_weight)
        weights = weights / weights.sum()  # Renormalize
        
        weights_dict = dict(zip(expected_returns.index, weights))
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate transaction costs and turnover
        transaction_costs = 0.0
        turnover = 0.0
        if current_weights is not None:
            current_w = np.array([current_weights.get(symbol, 0.0) for symbol in expected_returns.index])
            turnover = np.sum(np.abs(weights - current_w))
            transaction_costs = turnover * constraints.transaction_cost
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            optimization_method=OptimizationMethod.FACTOR_BASED,
            success=True,
            message="Factor-based optimization successful",
            transaction_costs=transaction_costs,
            turnover=turnover
        )