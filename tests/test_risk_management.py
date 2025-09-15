"""
Comprehensive tests for risk management system.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import asyncio

from src.models.portfolio import Portfolio, Position
from src.models.risk_management import (
    RiskLimit, RiskMetrics, RiskAlert, StressTestScenario, StressTestResult,
    PositionSizingRule, RiskLimitType, RiskLimitStatus
)
from src.services.risk_manager import RiskManager
from src.services.risk_monitoring_service import RiskMonitoringService


class TestRiskModels:
    """Test risk management data models."""
    
    def test_risk_limit_validation(self):
        """Test risk limit validation."""
        # Valid risk limit
        limit = RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold=0.15,
            warning_threshold=0.10,
            enabled=True
        )
        assert limit.threshold == 0.15
        assert limit.warning_threshold == 0.10
        
        # Invalid: warning >= threshold
        with pytest.raises(ValueError):
            RiskLimit(
                limit_type=RiskLimitType.MAX_DRAWDOWN,
                threshold=0.10,
                warning_threshold=0.15,
                enabled=True
            )
    
    def test_risk_metrics_validation(self):
        """Test risk metrics validation."""
        metrics = RiskMetrics(
            portfolio_value=100000.0,
            daily_pnl=1500.0,
            unrealized_pnl=2500.0,
            realized_pnl=500.0,
            max_drawdown=0.08,
            current_drawdown=0.03,
            portfolio_var=2000.0,
            portfolio_volatility=0.15,
            concentration_risk=0.25,
            leverage=1.2,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert metrics.portfolio_value == 100000.0
        assert metrics.timestamp.tzinfo is not None
    
    def test_stress_test_scenario(self):
        """Test stress test scenario model."""
        scenario = StressTestScenario(
            scenario_name="test_crash",
            market_shocks={"AAPL": -0.20, "GOOGL": -0.15},
            correlation_adjustment=1.5,
            volatility_multiplier=2.0,
            description="Test market crash"
        )
        
        assert scenario.scenario_name == "test_crash"
        assert scenario.market_shocks["AAPL"] == -0.20
        assert scenario.correlation_adjustment == 1.5


class TestRiskManager:
    """Test risk manager functionality."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return RiskManager(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100.0,
                avg_cost=150.0,
                current_price=155.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=50.0,
                avg_cost=2000.0,
                current_price=2100.0,
                unrealized_pnl=5000.0,
                realized_pnl=0.0
            )
        }
        
        # Calculate correct total value: cash + positions market value
        # AAPL: 100 * 155 = 15500, GOOGL: 50 * 2100 = 105000
        # Total positions value: 120500, Cash: 10000, Total: 130500
        return Portfolio(
            user_id="test_user",
            positions=positions,
            cash_balance=10000.0,
            total_value=130500.0,
            last_updated=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=100),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic price data
        np.random.seed(42)
        aapl_prices = 150 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        googl_prices = 2000 * np.exp(np.cumsum(np.random.normal(0.0008, 0.025, len(dates))))
        
        return pd.DataFrame({
            'AAPL': aapl_prices,
            'GOOGL': googl_prices
        }, index=dates)
    
    def test_add_risk_limits(self, risk_manager):
        """Test adding risk limits."""
        limit = RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold=0.15,
            warning_threshold=0.10,
            enabled=True
        )
        
        risk_manager.add_risk_limit(limit)
        assert len(risk_manager.risk_limits) == 1
        
        limit_key = f"{RiskLimitType.MAX_DRAWDOWN.value}_portfolio"
        assert limit_key in risk_manager.risk_limits
    
    def test_calculate_risk_metrics(self, risk_manager, sample_portfolio, sample_market_data):
        """Test risk metrics calculation."""
        metrics = risk_manager.calculate_risk_metrics(
            sample_portfolio, sample_market_data
        )
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.portfolio_value == sample_portfolio.total_value
        assert metrics.portfolio_volatility >= 0
        assert metrics.portfolio_var >= 0
        assert metrics.concentration_risk >= 0
        assert metrics.leverage >= 0
    
    def test_check_risk_limits(self, risk_manager, sample_portfolio, sample_market_data):
        """Test risk limit checking."""
        # Add a drawdown limit that should trigger
        limit = RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold=0.05,  # Very low threshold
            warning_threshold=0.02,
            enabled=True
        )
        risk_manager.add_risk_limit(limit)
        
        # Set high water mark to trigger drawdown
        risk_manager.high_water_mark = sample_portfolio.total_value * 1.2
        
        metrics = risk_manager.calculate_risk_metrics(
            sample_portfolio, sample_market_data
        )
        
        alerts = risk_manager.check_risk_limits(sample_portfolio, metrics)
        
        # Should generate alert for drawdown
        assert len(alerts) > 0
        assert any(alert.limit_type == RiskLimitType.MAX_DRAWDOWN for alert in alerts)
    
    def test_position_sizing_enforcement(self, risk_manager, sample_portfolio, sample_market_data):
        """Test position sizing enforcement."""
        # Add position sizing rule
        rule = PositionSizingRule(
            rule_name="conservative",
            max_position_size=0.10,
            volatility_target=0.02,
            correlation_penalty=0.5,
            kelly_fraction=0.25,
            enabled=True
        )
        risk_manager.add_position_sizing_rule(rule)
        
        # Test position sizing
        adjusted_size = risk_manager.enforce_position_sizing(
            portfolio=sample_portfolio,
            symbol="MSFT",
            intended_size=0.20,  # 20% intended
            expected_return=0.12,
            volatility=0.25,
            market_data=sample_market_data
        )
        
        # Should be reduced due to rule constraints
        assert adjusted_size <= 0.10  # Max position size constraint
        assert adjusted_size > 0
    
    def test_stress_testing(self, risk_manager, sample_portfolio, sample_market_data):
        """Test stress testing functionality."""
        scenario = StressTestScenario(
            scenario_name="test_crash",
            market_shocks={"AAPL": -0.20, "GOOGL": -0.15},
            correlation_adjustment=1.0,
            volatility_multiplier=1.0,
            description="Test crash scenario"
        )
        
        result = risk_manager.run_stress_test(
            sample_portfolio, scenario, sample_market_data
        )
        
        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "test_crash"
        assert result.portfolio_value_after < result.portfolio_value_before
        assert result.total_loss < 0  # Should be negative (loss)
        assert "AAPL" in result.position_impacts
        assert "GOOGL" in result.position_impacts
    
    def test_predefined_stress_scenarios(self, risk_manager):
        """Test predefined stress scenarios."""
        scenarios = risk_manager.get_predefined_stress_scenarios()
        
        assert len(scenarios) > 0
        assert any(s.scenario_name == "market_crash_2008" for s in scenarios)
        assert any(s.scenario_name == "covid_crash_2020" for s in scenarios)
        
        # Validate scenario structure
        for scenario in scenarios:
            assert isinstance(scenario.market_shocks, dict)
            assert len(scenario.market_shocks) > 0
            assert scenario.correlation_adjustment > 0
            assert scenario.volatility_multiplier > 0
    
    def test_portfolio_volatility_calculation(self, risk_manager):
        """Test portfolio volatility calculation."""
        # Test with empty weights
        weights = {}
        cov_matrix = pd.DataFrame()
        vol = risk_manager._calculate_portfolio_volatility(weights, cov_matrix)
        assert vol == 0.0
        
        # Test with valid data
        weights = {"AAPL": 0.6, "GOOGL": 0.4}
        cov_matrix = pd.DataFrame({
            "AAPL": [0.04, 0.02],
            "GOOGL": [0.02, 0.06]
        }, index=["AAPL", "GOOGL"])
        
        vol = risk_manager._calculate_portfolio_volatility(weights, cov_matrix)
        assert vol > 0
        assert isinstance(vol, float)
    
    def test_var_calculation(self, risk_manager):
        """Test Value at Risk calculation."""
        portfolio_value = 100000.0
        volatility = 0.15
        confidence_level = 0.05
        
        var = risk_manager._calculate_var(portfolio_value, volatility, confidence_level)
        
        assert var > 0
        assert var < portfolio_value  # VaR should be less than total value
    
    def test_concentration_risk_calculation(self, risk_manager):
        """Test concentration risk calculation."""
        # Highly concentrated portfolio
        concentrated_weights = {"AAPL": 0.8, "GOOGL": 0.2}
        conc_risk_high = risk_manager._calculate_concentration_risk(concentrated_weights)
        
        # Diversified portfolio
        diversified_weights = {"AAPL": 0.25, "GOOGL": 0.25, "MSFT": 0.25, "TSLA": 0.25}
        conc_risk_low = risk_manager._calculate_concentration_risk(diversified_weights)
        
        assert conc_risk_high > conc_risk_low
        assert 0 <= conc_risk_low <= 1
        assert 0 <= conc_risk_high <= 1


class TestRiskMonitoringService:
    """Test risk monitoring service."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return RiskManager()
    
    @pytest.fixture
    def monitoring_service(self, risk_manager):
        """Create monitoring service instance."""
        return RiskMonitoringService(
            risk_manager=risk_manager,
            monitoring_interval=1  # 1 second for testing
        )
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100.0,
                avg_cost=150.0,
                current_price=155.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0
            )
        }
        
        return Portfolio(
            user_id="test_user",
            positions=positions,
            cash_balance=10000.0,
            total_value=25500.0,
            last_updated=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        np.random.seed(42)
        prices = 150 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        return pd.DataFrame({'AAPL': prices}, index=dates)
    
    def test_alert_callbacks(self, monitoring_service):
        """Test alert callback management."""
        callback_called = []
        
        def test_callback(alert, portfolio, metrics):
            callback_called.append(True)
        
        # Add callback
        monitoring_service.add_alert_callback(test_callback)
        assert len(monitoring_service.alert_callbacks) == 1
        
        # Remove callback
        monitoring_service.remove_alert_callback(test_callback)
        assert len(monitoring_service.alert_callbacks) == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitoring_service, sample_portfolio, sample_market_data):
        """Test monitoring start/stop lifecycle."""
        portfolio_provider = lambda: sample_portfolio
        market_data_provider = lambda: sample_market_data
        
        # Start monitoring
        await monitoring_service.start_monitoring(
            portfolio_provider, market_data_provider
        )
        assert monitoring_service.is_monitoring
        
        # Let it run for a short time
        await asyncio.sleep(2)
        
        # Stop monitoring
        await monitoring_service.stop_monitoring()
        assert not monitoring_service.is_monitoring
        
        # Should have collected some metrics
        assert len(monitoring_service.risk_metrics_history) > 0
    
    def test_risk_status_reporting(self, monitoring_service):
        """Test risk status reporting."""
        status = monitoring_service.get_current_risk_status()
        
        assert "status" in status
        assert status["status"] in ["monitoring", "stopped", "no_data"]
    
    def test_data_cleanup(self, monitoring_service):
        """Test old data cleanup."""
        # Add some old metrics
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        old_metrics = RiskMetrics(
            portfolio_value=100000.0,
            daily_pnl=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            portfolio_var=0.0,
            portfolio_volatility=0.0,
            concentration_risk=0.0,
            leverage=1.0,
            timestamp=old_time
        )
        
        monitoring_service.risk_metrics_history.append(old_metrics)
        initial_count = len(monitoring_service.risk_metrics_history)
        
        # Clean up data older than 7 days
        monitoring_service.cleanup_old_data(days=7)
        
        # Old data should be removed
        assert len(monitoring_service.risk_metrics_history) < initial_count
    
    def test_risk_report_generation(self, monitoring_service, sample_portfolio, sample_market_data):
        """Test comprehensive risk report generation."""
        report = monitoring_service.generate_risk_report(
            sample_portfolio, sample_market_data, include_stress_tests=True
        )
        
        assert "report_timestamp" in report
        assert "portfolio_summary" in report
        assert "current_risk_metrics" in report
        assert "monitoring_status" in report
        
        # Check portfolio summary
        portfolio_summary = report["portfolio_summary"]
        assert portfolio_summary["total_value"] == sample_portfolio.total_value
        assert portfolio_summary["positions_count"] == len(sample_portfolio.positions)
    
    def test_data_export(self, monitoring_service, tmp_path):
        """Test risk data export functionality."""
        # Add some test data
        test_metrics = RiskMetrics(
            portfolio_value=100000.0,
            daily_pnl=1000.0,
            unrealized_pnl=2000.0,
            realized_pnl=500.0,
            max_drawdown=0.05,
            current_drawdown=0.02,
            portfolio_var=1500.0,
            portfolio_volatility=0.12,
            concentration_risk=0.3,
            leverage=1.1,
            timestamp=datetime.now(timezone.utc)
        )
        monitoring_service.risk_metrics_history.append(test_metrics)
        
        # Test JSON export
        json_file = tmp_path / "risk_data.json"
        success = monitoring_service.export_risk_data(str(json_file), format="json")
        
        assert success
        assert json_file.exists()
        
        # Test CSV export
        csv_file = tmp_path / "risk_data.csv"
        success = monitoring_service.export_risk_data(str(csv_file), format="csv")
        
        assert success


class TestRiskCalculations:
    """Test specific risk calculation methods."""
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation logic."""
        risk_manager = RiskManager()
        
        # Create portfolio with declining value
        portfolio1 = Portfolio(
            user_id="test",
            positions={},
            cash_balance=100000.0,
            total_value=100000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        # First calculation - sets high water mark
        current_dd1, max_dd1 = risk_manager._calculate_drawdown_metrics(portfolio1)
        assert current_dd1 == 0.0
        assert max_dd1 == 0.0
        assert risk_manager.high_water_mark == 100000.0
        
        # Portfolio declines
        portfolio2 = Portfolio(
            user_id="test",
            positions={},
            cash_balance=90000.0,
            total_value=90000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        current_dd2, max_dd2 = risk_manager._calculate_drawdown_metrics(portfolio2)
        assert current_dd2 == 0.1  # 10% drawdown
        assert max_dd2 == 0.1
        
        # Portfolio recovers partially
        portfolio3 = Portfolio(
            user_id="test",
            positions={},
            cash_balance=95000.0,
            total_value=95000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        current_dd3, max_dd3 = risk_manager._calculate_drawdown_metrics(portfolio3)
        assert current_dd3 == 0.05  # 5% drawdown from peak
        assert max_dd3 == 0.1  # Max drawdown remains 10%
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion position sizing."""
        risk_manager = RiskManager()
        
        # High expected return, low volatility - should suggest larger position
        expected_return = 0.15
        volatility = 0.10
        
        rule = PositionSizingRule(
            rule_name="test",
            max_position_size=0.20,
            volatility_target=0.02,
            correlation_penalty=0.0,
            kelly_fraction=1.0,  # Full Kelly
            enabled=True
        )
        
        # Create dummy portfolio and market data
        portfolio = Portfolio(
            user_id="test",
            positions={},
            cash_balance=100000.0,
            total_value=100000.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        market_data = pd.DataFrame({
            'TEST': [100, 101, 102, 103, 104]
        })
        
        size = risk_manager._calculate_rule_based_size(
            portfolio, "TEST", expected_return, volatility, market_data, rule
        )
        
        # Kelly fraction = expected_return / volatility^2 = 0.15 / 0.01 = 15
        # But should be capped by max_position_size and volatility_target
        assert 0 < size <= rule.max_position_size
    
    def test_correlation_adjustment(self):
        """Test correlation-based position size adjustment."""
        risk_manager = RiskManager()
        
        # Create portfolio with existing position
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100.0,
                avg_cost=150.0,
                current_price=155.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0
            )
        }
        
        portfolio = Portfolio(
            user_id="test",
            positions=positions,
            cash_balance=10000.0,
            total_value=25500.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        # Create correlated market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Highly correlated assets
        base_returns = np.random.normal(0.001, 0.02, 100)
        aapl_prices = 150 * np.exp(np.cumsum(base_returns))
        msft_prices = 250 * np.exp(np.cumsum(base_returns + np.random.normal(0, 0.005, 100)))
        
        market_data = pd.DataFrame({
            'AAPL': aapl_prices,
            'MSFT': msft_prices
        }, index=dates)
        
        # Calculate correlation
        correlation = risk_manager._calculate_symbol_correlation("MSFT", portfolio, market_data)
        
        # Should detect high correlation
        assert correlation > 0.5  # Expecting high correlation due to similar returns


if __name__ == "__main__":
    pytest.main([__file__])