"""
Comprehensive portfolio management service integrating optimization and rebalancing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.models.portfolio import Portfolio, Position
from src.services.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationMethod, OptimizationConstraints, OptimizationResult
)
from src.services.portfolio_rebalancer import (
    PortfolioRebalancer, RebalancingConfig, RebalanceRecommendation, RebalanceExecution
)


class PortfolioManagementService:
    """
    Comprehensive portfolio management service that handles optimization,
    rebalancing, and risk management.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio management service.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.optimizer = PortfolioOptimizer(risk_free_rate)
        self.rebalancer = PortfolioRebalancer(self.optimizer)
        self.logger = logging.getLogger(__name__)
    
    def manage_portfolio(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        optimization_method: OptimizationMethod = OptimizationMethod.MAXIMUM_SHARPE,
        lookback_days: int = 252,
        optimization_constraints: Optional[OptimizationConstraints] = None,
        rebalancing_config: Optional[RebalancingConfig] = None
    ) -> Dict:
        """
        Complete portfolio management including optimization and rebalancing.
        
        Args:
            portfolio: Current portfolio
            market_data: Historical market data
            optimization_method: Method for portfolio optimization
            lookback_days: Days of historical data to use
            optimization_constraints: Optimization constraints
            rebalancing_config: Rebalancing configuration
        
        Returns:
            Dictionary with optimization and rebalancing results
        """
        try:
            # Calculate expected returns and covariance matrix
            returns_data = self._calculate_returns(market_data, lookback_days)
            expected_returns = self._estimate_expected_returns(returns_data)
            covariance_matrix = self._calculate_covariance_matrix(returns_data)
            
            # Get current prices
            current_prices = self._get_current_prices(market_data)
            
            # Perform optimization and rebalancing
            optimization_result, rebalance_recommendation, rebalance_execution = (
                self.rebalancer.optimize_and_rebalance(
                    portfolio=portfolio,
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    current_prices=current_prices,
                    optimization_method=optimization_method,
                    optimization_constraints=optimization_constraints,
                    rebalancing_config=rebalancing_config
                )
            )
            
            return {
                "optimization_result": optimization_result,
                "rebalance_recommendation": rebalance_recommendation,
                "rebalance_execution": rebalance_execution,
                "portfolio_metrics": self._calculate_portfolio_metrics(
                    portfolio, current_prices, expected_returns, covariance_matrix
                ),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio management failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now()
            }   
 
    def _calculate_returns(self, market_data: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
        """Calculate returns from market data."""
        if len(market_data) < lookback_days:
            lookback_days = len(market_data)
        
        recent_data = market_data.tail(lookback_days)
        returns = recent_data.pct_change().dropna()
        
        return returns
    
    def _estimate_expected_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Estimate expected returns using historical mean."""
        # Annualize daily returns
        expected_returns = returns_data.mean() * 252
        return expected_returns
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix of returns."""
        # Annualize covariance matrix
        cov_matrix = returns_data.cov() * 252
        return cov_matrix
    
    def _get_current_prices(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Get current prices from market data."""
        latest_prices = market_data.iloc[-1]
        return latest_prices.to_dict()
    
    def _calculate_portfolio_metrics(
        self,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        current_weights = self.rebalancer._calculate_current_weights(portfolio, current_prices)
        
        if not current_weights:
            return {}
        
        # Convert to arrays for calculations
        symbols = list(current_weights.keys())
        weights = np.array([current_weights[symbol] for symbol in symbols])
        returns = np.array([expected_returns.get(symbol, 0.0) for symbol in symbols])
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, returns)
        
        # Portfolio volatility
        portfolio_cov = covariance_matrix.loc[symbols, symbols]
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(portfolio_cov, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.optimizer.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Diversification metrics
        concentration = np.sum(weights ** 2)  # Herfindahl index
        effective_positions = 1 / concentration if concentration > 0 else 0
        
        return {
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe_ratio,
            "concentration": concentration,
            "effective_positions": effective_positions,
            "total_value": portfolio.total_value,
            "cash_balance": portfolio.cash_balance,
            "positions_value": portfolio.positions_value,
            "unrealized_pnl": portfolio.unrealized_pnl,
            "realized_pnl": portfolio.realized_pnl
        }
    
    def calculate_position_sizing(
        self,
        portfolio: Portfolio,
        symbol: str,
        expected_return: float,
        volatility: float,
        correlation_with_portfolio: float = 0.0,
        risk_budget: float = 0.02
    ) -> float:
        """
        Calculate optimal position size based on volatility and correlation.
        
        Args:
            portfolio: Current portfolio
            symbol: Symbol to size
            expected_return: Expected return for the asset
            volatility: Asset volatility
            correlation_with_portfolio: Correlation with existing portfolio
            risk_budget: Risk budget as fraction of portfolio value
        
        Returns:
            Optimal position size as fraction of portfolio value
        """
        if volatility <= 0:
            return 0.0
        
        # Kelly criterion with risk adjustment
        kelly_fraction = expected_return / (volatility ** 2)
        
        # Adjust for correlation with existing portfolio
        correlation_adjustment = 1 - abs(correlation_with_portfolio) * 0.5
        adjusted_kelly = kelly_fraction * correlation_adjustment
        
        # Apply risk budget constraint
        risk_adjusted_size = min(adjusted_kelly, risk_budget / volatility)
        
        # Ensure reasonable bounds
        position_size = max(0.0, min(risk_adjusted_size, 0.1))  # Max 10% position
        
        return position_size
    
    def assess_portfolio_risk(
        self,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        confidence_level: float = 0.05
    ) -> Dict:
        """
        Assess portfolio risk metrics.
        
        Args:
            portfolio: Current portfolio
            current_prices: Current market prices
            covariance_matrix: Covariance matrix of returns
            confidence_level: Confidence level for VaR calculation
        
        Returns:
            Dictionary with risk metrics
        """
        current_weights = self.rebalancer._calculate_current_weights(portfolio, current_prices)
        
        if not current_weights:
            return {}
        
        symbols = list(current_weights.keys())
        weights = np.array([current_weights[symbol] for symbol in symbols])
        
        # Portfolio volatility
        portfolio_cov = covariance_matrix.loc[symbols, symbols]
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(portfolio_cov, weights)))
        
        # Value at Risk (VaR)
        from scipy.stats import norm
        var_multiplier = norm.ppf(confidence_level)
        daily_var = portfolio.total_value * portfolio_vol / np.sqrt(252) * var_multiplier
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = portfolio.total_value * portfolio_vol / np.sqrt(252) * (
            norm.pdf(var_multiplier) / confidence_level
        )
        
        # Maximum drawdown (simplified)
        max_drawdown = portfolio.total_value - portfolio.cash_balance  # Simplified
        
        # Concentration risk
        concentration = np.sum(weights ** 2)
        
        return {
            "portfolio_volatility": portfolio_vol,
            "daily_var": abs(daily_var),
            "expected_shortfall": abs(expected_shortfall),
            "max_drawdown": max_drawdown,
            "concentration_risk": concentration,
            "confidence_level": confidence_level
        }
    
    async def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        action: str
    ) -> bool:
        """
        Update portfolio position (mock implementation for testing).
        
        Args:
            symbol: Trading symbol
            quantity: Quantity of the trade
            price: Execution price
            action: Trade action ('buy' or 'sell')
            
        Returns:
            bool: True if update successful, False otherwise
        """
        # This is a mock implementation for testing purposes
        # In a real implementation, this would update the actual portfolio
        return True