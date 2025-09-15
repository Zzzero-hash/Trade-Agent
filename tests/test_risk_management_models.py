"""
Test suite for risk management models.
"""
import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.models.risk_management import (
    RiskLimit,
    RiskLimitType,
    RiskLimitStatus,
    RiskMetrics,
    RiskAlert,
    RiskAlertBuilder,
    RiskLimitFactory,
    StressTestScenario,
    StressTestResult,
    PositionSizingRule,
    ensure_timezone_aware
)


class TestRiskLimit:
    """Test RiskLimit model validation and functionality."""
    
    def test_valid_risk_limit_creation(self):
        """Test creating a valid risk limit."""
        limit = RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold=0.15,
            warning_threshold=0.10,
            enabled=True
        )
        assert limit.limit_type == RiskLimitType.MAX_DRAWDOWN
        assert limit.threshold == 0.15
        assert limit.warning_threshold == 0.10
        assert limit.enabled is True
    
    def test_warning_threshold_validation(self):
        """Test that warning threshold must be less than threshold."""
        with pytest.raises(ValidationError) as exc_info:
            RiskLimit(
                limit_type=RiskLimitType.MAX_DRAWDOWN,
                threshold=0.10,
                warning_threshold=0.15,  # Invalid: greater than threshold
                enabled=True
            )
        assert "Warning threshold must be less than breach threshold" in str(exc_info.value)


class TestRiskMetrics:
    """Test RiskMetrics model validation and computed properties."""
    
    def test_valid_risk_metrics_creation(self):
        """Test creating valid risk metrics."""
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
        assert metrics.total_pnl == 3000.0  # unrealized + realized
        assert metrics.daily_return_pct == 1.5  # 1500/100000 * 100
        assert not metrics.is_high_risk
    
    def test_high_risk_detection(self):
        """Test high risk state detection."""
        metrics = RiskMetrics(
            portfolio_value=100000.0,
            daily_pnl=-5000.0,
            unrealized_pnl=-2000.0,
            realized_pnl=-1000.0,
            max_drawdown=0.20,
            current_drawdown=0.18,  # High drawdown
            portfolio_var=5000.0,
            portfolio_volatility=0.35,
            concentration_risk=0.45,  # High concentration
            leverage=3.5,  # High leverage
            timestamp=datetime.now(timezone.utc)
        )
        assert metrics.is_high_risk
    
    def test_pnl_validation(self):
        """Test P&L validation against portfolio value."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetrics(
                portfolio_value=100000.0,
                daily_pnl=150000.0,  # Exceeds portfolio value
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
        assert "exceeds portfolio value" in str(exc_info.value)


class TestRiskAlertBuilder:
    """Test RiskAlertBuilder pattern."""
    
    def test_breach_alert_creation(self):
        """Test creating breach alert using builder."""
        alert = (RiskAlertBuilder()
                .with_limit_breach(
                    RiskLimitType.MAX_DRAWDOWN, 
                    0.18, 
                    0.15
                )
                .for_symbol("AAPL")
                .build())
        
        assert alert.status == RiskLimitStatus.BREACH
        assert alert.limit_type == RiskLimitType.MAX_DRAWDOWN
        assert alert.current_value == 0.18
        assert alert.threshold == 0.15
        assert alert.symbol == "AAPL"
        assert "Max Drawdown for AAPL breach" in alert.message
        assert alert.severity_score > 10  # High severity
    
    def test_warning_alert_creation(self):
        """Test creating warning alert using builder."""
        alert = (RiskAlertBuilder()
                .with_warning(
                    RiskLimitType.POSITION_SIZE, 
                    0.12, 
                    0.15
                )
                .build())
        
        assert alert.status == RiskLimitStatus.WARNING
        assert alert.limit_type == RiskLimitType.POSITION_SIZE
        assert alert.severity_score < 10  # Lower severity than breach


class TestRiskLimitFactory:
    """Test RiskLimitFactory pattern."""
    
    def test_conservative_limits_creation(self):
        """Test creating conservative risk limits."""
        limits = RiskLimitFactory.create_conservative_limits()
        
        assert "max_drawdown" in limits
        assert "position_size" in limits
        assert "daily_loss" in limits
        
        # Conservative limits should be stricter
        assert limits["max_drawdown"].threshold == 0.10
        assert limits["position_size"].threshold == 0.05
        assert limits["daily_loss"].threshold == 0.02
    
    def test_aggressive_limits_creation(self):
        """Test creating aggressive risk limits."""
        limits = RiskLimitFactory.create_aggressive_limits()
        
        # Aggressive limits should be more lenient
        assert limits["max_drawdown"].threshold == 0.25
        assert limits["position_size"].threshold == 0.15
        assert limits["daily_loss"].threshold == 0.05


class TestStressTestScenario:
    """Test StressTestScenario validation."""
    
    def test_valid_scenario_creation(self):
        """Test creating valid stress test scenario."""
        scenario = StressTestScenario(
            scenario_name="market_crash_2008",
            market_shocks={"SPY": -0.35, "QQQ": -0.40},
            correlation_adjustment=1.5,
            volatility_multiplier=2.0,
            description="2008 financial crisis simulation"
        )
        assert scenario.scenario_name == "market_crash_2008"
        assert scenario.market_shocks["SPY"] == -0.35
    
    def test_extreme_shock_validation(self):
        """Test validation of extreme market shocks."""
        with pytest.raises(ValidationError) as exc_info:
            StressTestScenario(
                scenario_name="impossible_crash",
                market_shocks={"SPY": -1.5},  # 150% drop - impossible
                description="Impossible scenario"
            )
        assert "exceeds 100%" in str(exc_info.value)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_ensure_timezone_aware(self):
        """Test timezone awareness utility."""
        # Naive datetime
        naive_dt = datetime(2023, 12, 1, 15, 30, 0)
        aware_dt = ensure_timezone_aware(naive_dt)
        assert aware_dt.tzinfo is not None
        assert aware_dt.tzinfo == timezone.utc
        
        # Already aware datetime
        already_aware = datetime.now(timezone.utc)
        result = ensure_timezone_aware(already_aware)
        assert result.tzinfo == timezone.utc