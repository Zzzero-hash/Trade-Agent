"""
Tests for comprehensive portfolio management service.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.portfolio import Portfolio, Position
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.portfolio_optimizer import OptimizationMethod


class TestPortfolioManagementService:
    """Test cases for PortfolioManagementService."""
    
    @pytest.fixture
    def service(self):
        """Create portfolio management service instance."""
        return PortfolioManagementService(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100.0,
                avg_cost=150.0,
                current_price=160.0,
                unrealized_pnl=1000.0
            ),
            'GOOGL': Position(
                symbol='GOOGL',
                quantity=50.0,
                avg_cost=2000.0,
                current_price=2100.0,
                unrealized_pnl=5000.0
            ),
            'MSFT': Position(
                symbol='MSFT',
                quantity=75.0,
                avg_cost=280.0,
                current_price=300.0,
                unrealized_pnl=1500.0
            )
        }
        
        portfolio = Portfolio(
            user_id='test_user',
            positions=positions,
            cash_balance=15000.0,
            total_value=158500.0,  # 16000 + 105000 + 22500 + 15000
            last_updated=datetime.now(timezone.utc)
        )
        
        return portfolio
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Generate synthetic price data
        np.random.seed(42)
        n_days = len(dates)
        
        data = {}
        for symbol in symbols:
            # Generate random walk with drift
            returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))  # Price series
            data[symbol] = prices
        
        market_data = pd.DataFrame(data, index=dates)
        return market_data
    
    def test_calculate_returns(self, service, sample_market_data):
        """Test returns calculation from market data."""
        returns = service._calculate_returns(sample_market_data, lookback_days=252)
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns.columns) == len(sample_market_data.columns)
        assert len(returns) <= 252
        assert not returns.isnull().all().any()  # No columns with all NaN
    
    def test_estimate_expected_returns(self, service, sample_market_data):
        """Test expected returns estimation."""
        returns = service._calculate_returns(sample_market_data, lookback_days=252)
        expected_returns = service._estimate_expected_returns(returns)
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == len(returns.columns)
        assert all(isinstance(ret, (int, float)) for ret in expected_returns)
    
    def test_calculate_covariance_matrix(self, service, sample_market_data):
        """Test covariance matrix calculation."""
        returns = service._calculate_returns(sample_market_data, lookback_days=252)
        cov_matrix = service._calculate_covariance_matrix(returns)
        
        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape[0] == cov_matrix.shape[1]  # Square matrix
        assert len(cov_matrix) == len(returns.columns)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric matrix
        assert all(cov_matrix.iloc[i, i] > 0 for i in range(len(cov_matrix)))  # Positive diagonal
    
    def test_get_current_prices(self, service, sample_market_data):
        """Test current prices extraction."""
        current_prices = service._get_current_prices(sample_market_data)
        
        assert isinstance(current_prices, dict)
        assert len(current_prices) == len(sample_market_data.columns)
        assert all(price > 0 for price in current_prices.values())
    
    def test_calculate_portfolio_metrics(self, service, sample_portfolio, sample_market_data):
        """Test portfolio metrics calculation."""
        returns = service._calculate_returns(sample_market_data, lookback_days=252)
        expected_returns = service._estimate_expected_returns(returns)
        covariance_matrix = service._calculate_covariance_matrix(returns)
        current_prices = service._get_current_prices(sample_market_data)
        
        metrics = service._calculate_portfolio_metrics(
            portfolio=sample_portfolio,
            current_prices=current_prices,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix
        )
        
        assert isinstance(metrics, dict)
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'concentration' in metrics
        assert 'effective_positions' in metrics
        assert 'total_value' in metrics
        
        # Validate metric values
        assert metrics['total_value'] == sample_portfolio.total_value
        assert metrics['volatility'] >= 0
        assert metrics['concentration'] >= 0
        assert metrics['effective_positions'] >= 0
    
    def test_calculate_position_sizing(self, service, sample_portfolio):
        """Test position sizing calculation."""
        position_size = service.calculate_position_sizing(
            portfolio=sample_portfolio,
            symbol='TSLA',
            expected_return=0.15,
            volatility=0.3,
            correlation_with_portfolio=0.5,
            risk_budget=0.02
        )
        
        assert isinstance(position_size, float)
        assert 0 <= position_size <= 0.1  # Between 0 and 10%
    
    def test_position_sizing_zero_volatility(self, service, sample_portfolio):
        """Test position sizing with zero volatility."""
        position_size = service.calculate_position_sizing(
            portfolio=sample_portfolio,
            symbol='BOND',
            expected_return=0.05,
            volatility=0.0,  # Zero volatility
            risk_budget=0.02
        )
        
        assert position_size == 0.0
    
    def test_assess_portfolio_risk(self, service, sample_portfolio, sample_market_data):
        """Test portfolio risk assessment."""
        returns = service._calculate_returns(sample_market_data, lookback_days=252)
        covariance_matrix = service._calculate_covariance_matrix(returns)
        current_prices = service._get_current_prices(sample_market_data)
        
        risk_metrics = service.assess_portfolio_risk(
            portfolio=sample_portfolio,
            current_prices=current_prices,
            covariance_matrix=covariance_matrix,
            confidence_level=0.05
        )
        
        assert isinstance(risk_metrics, dict)
        assert 'portfolio_volatility' in risk_metrics
        assert 'daily_var' in risk_metrics
        assert 'expected_shortfall' in risk_metrics
        assert 'concentration_risk' in risk_metrics
        
        # Validate risk metric values
        assert risk_metrics['portfolio_volatility'] >= 0
        assert risk_metrics['daily_var'] >= 0
        assert risk_metrics['expected_shortfall'] >= 0
        assert risk_metrics['concentration_risk'] >= 0
    
    def test_manage_portfolio_complete_workflow(self, service, sample_portfolio, sample_market_data):
        """Test complete portfolio management workflow."""
        result = service.manage_portfolio(
            portfolio=sample_portfolio,
            market_data=sample_market_data,
            optimization_method=OptimizationMethod.MAXIMUM_SHARPE,
            lookback_days=100
        )
        
        assert isinstance(result, dict)
        assert 'optimization_result' in result
        assert 'rebalance_recommendation' in result
        assert 'portfolio_metrics' in result
        assert 'timestamp' in result
        
        # Check optimization result
        opt_result = result['optimization_result']
        assert hasattr(opt_result, 'success')
        assert hasattr(opt_result, 'weights')
        assert hasattr(opt_result, 'optimization_method')
        
        # Check portfolio metrics
        metrics = result['portfolio_metrics']
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    def test_manage_portfolio_with_different_methods(self, service, sample_portfolio, sample_market_data):
        """Test portfolio management with different optimization methods."""
        methods = [
            OptimizationMethod.MAXIMUM_SHARPE,
            OptimizationMethod.MINIMUM_VARIANCE,
            OptimizationMethod.RISK_PARITY
        ]
        
        for method in methods:
            result = service.manage_portfolio(
                portfolio=sample_portfolio,
                market_data=sample_market_data,
                optimization_method=method,
                lookback_days=50
            )
            
            assert isinstance(result, dict)
            assert 'optimization_result' in result
            
            opt_result = result['optimization_result']
            if opt_result.success:
                assert opt_result.optimization_method == method
    
    def test_manage_portfolio_error_handling(self, service, sample_portfolio):
        """Test error handling in portfolio management."""
        # Create invalid market data
        invalid_data = pd.DataFrame()
        
        result = service.manage_portfolio(
            portfolio=sample_portfolio,
            market_data=invalid_data,
            lookback_days=100
        )
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'timestamp' in result
    
    def test_empty_portfolio_handling(self, service, sample_market_data):
        """Test handling of empty portfolio."""
        empty_portfolio = Portfolio(
            user_id='test_user',
            positions={},
            cash_balance=10000.0,
            total_value=10000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        result = service.manage_portfolio(
            portfolio=empty_portfolio,
            market_data=sample_market_data,
            lookback_days=100
        )
        
        assert isinstance(result, dict)
        # Should handle empty portfolio gracefully
    
    def test_insufficient_market_data(self, service, sample_portfolio):
        """Test handling of insufficient market data."""
        # Create very limited market data
        limited_data = pd.DataFrame({
            'AAPL': [100, 101, 102],
            'GOOGL': [2000, 2010, 2020]
        })
        
        result = service.manage_portfolio(
            portfolio=sample_portfolio,
            market_data=limited_data,
            lookback_days=100  # More than available data
        )
        
        assert isinstance(result, dict)
        # Should adapt to available data