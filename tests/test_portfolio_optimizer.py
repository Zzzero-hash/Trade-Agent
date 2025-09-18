"""
Tests for portfolio optimization functionality.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.services.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationMethod, OptimizationConstraints, OptimizationResult
)


class TestPortfolioOptimizer:
    """Test cases for PortfolioOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create portfolio optimizer instance."""
        return PortfolioOptimizer(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        # Sample expected returns (annualized)
        expected_returns = pd.Series([0.12, 0.15, 0.10, 0.20], index=symbols)
        
        # Sample covariance matrix (annualized)
        cov_data = np.array([
            [0.04, 0.02, 0.015, 0.025],
            [0.02, 0.06, 0.018, 0.030],
            [0.015, 0.018, 0.03, 0.020],
            [0.025, 0.030, 0.020, 0.08]
        ])
        covariance_matrix = pd.DataFrame(cov_data, index=symbols, columns=symbols)
        
        return expected_returns, covariance_matrix
    
    def test_maximum_sharpe_optimization(self, optimizer, sample_data):
        """Test maximum Sharpe ratio optimization."""
        expected_returns, covariance_matrix = sample_data
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MAXIMUM_SHARPE
        )
        
        assert result.success
        assert result.optimization_method == OptimizationMethod.MAXIMUM_SHARPE
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6  # Weights sum to 1
        assert all(w >= 0 for w in result.weights.values())  # Non-negative weights
        assert result.sharpe_ratio > 0
        assert result.expected_return > 0
        assert result.expected_volatility > 0
    
    def test_minimum_variance_optimization(self, optimizer, sample_data):
        """Test minimum variance optimization."""
        expected_returns, covariance_matrix = sample_data
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MINIMUM_VARIANCE
        )
        
        assert result.success
        assert result.optimization_method == OptimizationMethod.MINIMUM_VARIANCE
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
        assert result.expected_volatility > 0
    
    def test_risk_parity_optimization(self, optimizer, sample_data):
        """Test risk parity optimization."""
        expected_returns, covariance_matrix = sample_data
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.RISK_PARITY
        )
        
        assert result.success
        assert result.optimization_method == OptimizationMethod.RISK_PARITY
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
    
    def test_optimization_with_constraints(self, optimizer, sample_data):
        """Test optimization with custom constraints."""
        expected_returns, covariance_matrix = sample_data
        
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.4,
            transaction_cost=0.002
        )
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MAXIMUM_SHARPE,
            constraints=constraints
        )
        
        assert result.success
        assert all(w >= constraints.min_weight - 1e-6 for w in result.weights.values())
        assert all(w <= constraints.max_weight + 1e-6 for w in result.weights.values())
    
    def test_transaction_cost_calculation(self, optimizer, sample_data):
        """Test transaction cost calculation with current weights."""
        expected_returns, covariance_matrix = sample_data
        
        current_weights = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.2, 'TSLA': 0.2}
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MAXIMUM_SHARPE,
            current_weights=current_weights
        )
        
        assert result.success
        assert result.transaction_costs >= 0
        assert result.turnover >= 0
    
    def test_mean_variance_with_target_return(self, optimizer, sample_data):
        """Test mean-variance optimization with target return."""
        expected_returns, covariance_matrix = sample_data
        
        constraints = OptimizationConstraints(target_return=0.12)
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MEAN_VARIANCE,
            constraints=constraints
        )
        
        assert result.success
        assert abs(result.expected_return - constraints.target_return) < 1e-3
    
    def test_factor_based_optimization(self, optimizer, sample_data):
        """Test factor-based optimization."""
        expected_returns, covariance_matrix = sample_data
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.FACTOR_BASED
        )
        
        assert result.success
        assert result.optimization_method == OptimizationMethod.FACTOR_BASED
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    
    def test_black_litterman_optimization(self, optimizer, sample_data):
        """Test Black-Litterman optimization."""
        expected_returns, covariance_matrix = sample_data
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.BLACK_LITTERMAN
        )
        
        assert result.success
        assert result.optimization_method == OptimizationMethod.BLACK_LITTERMAN
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    
    def test_optimization_failure_handling(self, optimizer):
        """Test handling of optimization failures."""
        # Create invalid data (negative covariance matrix)
        symbols = ['A', 'B']
        expected_returns = pd.Series([0.1, 0.1], index=symbols)
        covariance_matrix = pd.DataFrame([[-1, 0], [0, -1]], index=symbols, columns=symbols)
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MAXIMUM_SHARPE
        )
        
        assert not result.success
        assert "failed" in result.message.lower()
    
    def test_empty_portfolio_handling(self, optimizer):
        """Test handling of empty portfolio data."""
        expected_returns = pd.Series([], dtype=float)
        covariance_matrix = pd.DataFrame()
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            method=OptimizationMethod.MAXIMUM_SHARPE
        )
        
        assert not result.success