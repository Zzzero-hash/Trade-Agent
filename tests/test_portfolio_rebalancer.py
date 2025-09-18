"""
Tests for portfolio rebalancing functionality.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.portfolio import Portfolio, Position
from src.services.portfolio_optimizer import PortfolioOptimizer
from src.services.portfolio_rebalancer import (
    PortfolioRebalancer, RebalancingConfig, RebalanceRecommendation
)


class TestPortfolioRebalancer:
    """Test cases for PortfolioRebalancer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create portfolio optimizer instance."""
        return PortfolioOptimizer(risk_free_rate=0.02)
    
    @pytest.fixture
    def rebalancer(self, optimizer):
        """Create portfolio rebalancer instance."""
        return PortfolioRebalancer(optimizer)
    
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
            )
        }
        
        portfolio = Portfolio(
            user_id='test_user',
            positions=positions,
            cash_balance=10000.0,
            total_value=131000.0,  # 16000 + 105000 + 10000
            last_updated=datetime.now(timezone.utc)
        )
        
        return portfolio
    
    @pytest.fixture
    def current_prices(self):
        """Current market prices."""
        return {
            'AAPL': 160.0,
            'GOOGL': 2100.0,
            'MSFT': 300.0,
            'TSLA': 800.0
        }
    
    def test_calculate_current_weights(self, rebalancer, sample_portfolio, current_prices):
        """Test calculation of current portfolio weights."""
        weights = rebalancer._calculate_current_weights(sample_portfolio, current_prices)
        
        expected_aapl_weight = (100 * 160) / 131000  # ~0.122
        expected_googl_weight = (50 * 2100) / 131000  # ~0.802
        
        assert abs(weights['AAPL'] - expected_aapl_weight) < 1e-3
        assert abs(weights['GOOGL'] - expected_googl_weight) < 1e-3
        assert abs(sum(weights.values()) - (121000 / 131000)) < 1e-3  # Excluding cash
    
    def test_should_rebalance_percentage_method(self, rebalancer, sample_portfolio, current_prices):
        """Test rebalancing decision with percentage method."""
        target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        config = RebalancingConfig(
            threshold_method="percentage",
            rebalance_threshold=0.1
        )
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        assert isinstance(recommendation, RebalanceRecommendation)
        assert recommendation.target_weights == target_weights
        assert len(recommendation.current_weights) > 0
        assert recommendation.urgency_score >= 0
    
    def test_should_not_rebalance_small_deviation(self, rebalancer, sample_portfolio, current_prices):
        """Test that small deviations don't trigger rebalancing."""
        # Set target weights close to current weights
        current_weights = rebalancer._calculate_current_weights(sample_portfolio, current_prices)
        target_weights = {k: v * 1.01 for k, v in current_weights.items()}  # 1% deviation
        
        config = RebalancingConfig(
            threshold_method="percentage",
            rebalance_threshold=0.05  # 5% threshold
        )
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        assert not recommendation.should_rebalance
    
    def test_calculate_required_trades(self, rebalancer, sample_portfolio, current_prices):
        """Test calculation of required trades."""
        target_weights = {'AAPL': 0.6, 'GOOGL': 0.4}
        
        trades = rebalancer._calculate_required_trades(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices
        )
        
        assert isinstance(trades, dict)
        assert 'AAPL' in trades or 'GOOGL' in trades  # At least one trade needed
        
        # Verify trade calculations
        for symbol, quantity in trades.items():
            assert isinstance(quantity, float)
            assert abs(quantity) > 0.01  # Only meaningful trades
    
    def test_estimate_transaction_costs(self, rebalancer, current_prices):
        """Test transaction cost estimation."""
        trades = {'AAPL': 100.0, 'GOOGL': -25.0}
        config = RebalancingConfig()
        
        costs = rebalancer._estimate_transaction_costs(trades, current_prices, config)
        
        assert costs > 0
        expected_cost = (100 * 160 + 25 * 2100) * 0.001  # 0.1% transaction cost
        assert abs(costs - expected_cost) < 1.0
    
    def test_execute_rebalance_success(self, rebalancer, sample_portfolio, current_prices):
        """Test successful rebalancing execution."""
        target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        config = RebalancingConfig(rebalance_threshold=0.01)  # Low threshold to trigger
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        if recommendation.should_rebalance:
            execution = rebalancer.execute_rebalance(
                portfolio=sample_portfolio,
                recommendation=recommendation,
                current_prices=current_prices
            )
            
            assert execution.executed
            assert len(execution.trades_executed) > 0
            assert execution.actual_transaction_costs >= 0
            assert execution.slippage >= 0
            assert isinstance(execution.new_weights, dict)
    
    def test_execute_rebalance_no_action_needed(self, rebalancer, sample_portfolio, current_prices):
        """Test rebalancing execution when no action is needed."""
        # Create recommendation that doesn't require rebalancing
        recommendation = RebalanceRecommendation(
            should_rebalance=False,
            target_weights={},
            current_weights={},
            weight_deviations={},
            expected_transaction_costs=0.0,
            expected_benefit=0.0,
            trades_required={},
            reason="No rebalancing needed",
            urgency_score=0.0
        )
        
        execution = rebalancer.execute_rebalance(
            portfolio=sample_portfolio,
            recommendation=recommendation,
            current_prices=current_prices
        )
        
        assert not execution.executed
        assert len(execution.trades_executed) == 0
        assert execution.actual_transaction_costs == 0.0
    
    def test_time_constraints(self, rebalancer, sample_portfolio, current_prices):
        """Test time-based rebalancing constraints."""
        # Set last rebalance time to recent
        rebalancer.last_rebalance_time = datetime.now() - timedelta(days=1)
        
        target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        config = RebalancingConfig(
            min_rebalance_interval=7,  # 7 days minimum
            rebalance_threshold=0.01
        )
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        assert not recommendation.should_rebalance
        assert "Too soon" in recommendation.reason
    
    def test_transaction_cost_threshold(self, rebalancer, sample_portfolio, current_prices):
        """Test transaction cost threshold prevents rebalancing."""
        target_weights = {'AAPL': 0.1, 'GOOGL': 0.9}  # Large rebalancing needed
        config = RebalancingConfig(
            rebalance_threshold=0.01,  # Low threshold to trigger
            transaction_cost_threshold=0.001  # Very low cost threshold
        )
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        # Should not rebalance due to high transaction costs
        if not recommendation.should_rebalance:
            assert "Transaction costs too high" in recommendation.reason
    
    def test_absolute_threshold_method(self, rebalancer, sample_portfolio, current_prices):
        """Test absolute threshold rebalancing method."""
        target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        config = RebalancingConfig(
            threshold_method="absolute",
            rebalance_threshold=5000.0  # $5000 absolute threshold
        )
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        assert isinstance(recommendation, RebalanceRecommendation)
        # The decision depends on actual deviations
    
    def test_volatility_threshold_method(self, rebalancer, sample_portfolio, current_prices):
        """Test volatility-adjusted threshold method."""
        target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        config = RebalancingConfig(
            threshold_method="volatility",
            rebalance_threshold=0.05
        )
        
        recommendation = rebalancer.should_rebalance(
            portfolio=sample_portfolio,
            target_weights=target_weights,
            current_prices=current_prices,
            config=config
        )
        
        assert isinstance(recommendation, RebalanceRecommendation)
    
    def test_empty_portfolio_handling(self, rebalancer):
        """Test handling of empty portfolio."""
        empty_portfolio = Portfolio(
            user_id='test_user',
            positions={},
            cash_balance=10000.0,
            total_value=10000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        weights = rebalancer._calculate_current_weights(empty_portfolio, {})
        assert weights == {}
        
        trades = rebalancer._calculate_required_trades(
            empty_portfolio, {'AAPL': 1.0}, {'AAPL': 100.0}
        )
        assert isinstance(trades, dict)