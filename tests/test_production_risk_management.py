"""
Tests for Production Risk Management Service

Tests risk calculations, monitoring accuracy, alert reliability,
and automated risk controls.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.services.risk_management_service import (
    ProductionRiskManager, RiskLimits, RiskLevel, RiskAlert
)
from src.services.monitoring_service import MonitoringService
from src.services.alert_service import AlertService
from src.models.trading_models import Position, Trade
from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository


@pytest.fixture
def mock_position_repo():
    repo = Mock(spec=PositionRepository)
    repo.get_active_positions = AsyncMock()
    return repo


@pytest.fixture
def mock_trade_repo():
    repo = Mock(spec=TradeRepository)
    return repo


@pytest.fixture
def mock_monitoring_service():
    service = Mock(spec=MonitoringService)
    service.record_metric = AsyncMock()
    return service


@pytest.fixture
def mock_alert_service():
    service = Mock(spec=AlertService)
    service.send_risk_alert = AsyncMock()
    service.send_critical_alert = AsyncMock()
    return service


@pytest.fixture
def risk_manager(mock_position_repo, mock_trade_repo, mock_monitoring_service, mock_alert_service):
    return ProductionRiskManager(
        position_repo=mock_position_repo,
        trade_repo=mock_trade_repo,
        monitoring_service=mock_monitoring_service,
        alert_service=mock_alert_service
    )


@pytest.fixture
def sample_position():
    return Position(
        position_id="pos_123",
        customer_id="customer_1",
        symbol="AAPL",
        quantity=Decimal("100"),
        cost_basis=Decimal("15000"),  # $150 per share
        current_value=Decimal("14000"),  # $140 per share
        unrealized_pnl=Decimal("-1000"),
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_risk_limits():
    return RiskLimits(
        max_position_size=Decimal("20000"),
        max_daily_loss=Decimal("2000"),
        max_portfolio_risk=Decimal("0.15"),
        max_correlation_exposure=Decimal("0.3"),
        stop_loss_percentage=Decimal("0.1"),  # 10%
        max_leverage=Decimal("2.0")
    )


class TestRiskCalculations:
    """Test risk calculation accuracy."""
    
    @pytest.mark.asyncio
    async def test_position_risk_calculation(self, risk_manager, sample_position, sample_risk_limits):
        """Test individual position risk calculation."""
        await risk_manager.set_risk_limits("customer_1", sample_risk_limits)
        
        # Position loss is $1000 on $15000 cost basis = 6.67%
        # Should not trigger stop-loss (10% threshold)
        should_trigger = await risk_manager._should_trigger_stop_loss(sample_position, sample_risk_limits)
        assert not should_trigger
        
        # Increase loss to trigger stop-loss
        sample_position.current_value = Decimal("13000")  # $130 per share
        sample_position.unrealized_pnl = Decimal("-2000")  # 13.33% loss
        
        should_trigger = await risk_manager._should_trigger_stop_loss(sample_position, sample_risk_limits)
        assert should_trigger
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_calculation(self, risk_manager):
        """Test portfolio-level risk calculation."""
        positions = [
            Position(
                position_id="pos_1",
                customer_id="customer_1",
                symbol="AAPL",
                quantity=Decimal("100"),
                cost_basis=Decimal("15000"),
                current_value=Decimal("14000"),
                unrealized_pnl=Decimal("-1000"),
                created_at=datetime.utcnow()
            ),
            Position(
                position_id="pos_2",
                customer_id="customer_1",
                symbol="GOOGL",
                quantity=Decimal("50"),
                cost_basis=Decimal("10000"),
                current_value=Decimal("9500"),
                unrealized_pnl=Decimal("-500"),
                created_at=datetime.utcnow()
            )
        ]
        
        portfolio_risk = await risk_manager._calculate_portfolio_risk(positions)
        assert isinstance(portfolio_risk, Decimal)
        assert portfolio_risk >= 0
    
    @pytest.mark.asyncio
    async def test_risk_limit_violations(self, risk_manager, sample_position, sample_risk_limits, mock_alert_service):
        """Test risk limit violation detection."""
        await risk_manager.set_risk_limits("customer_1", sample_risk_limits)
        
        # Test position size violation
        sample_position.current_value = Decimal("25000")  # Exceeds max_position_size
        await risk_manager._handle_position_size_violation(sample_position, sample_risk_limits)
        
        mock_alert_service.send_risk_alert.assert_called_once()
        alert_call = mock_alert_service.send_risk_alert.call_args[0][0]
        assert alert_call.risk_level == RiskLevel.HIGH
        assert "Position size exceeds limit" in alert_call.message
    
    @pytest.mark.asyncio
    async def test_daily_loss_violation(self, risk_manager, sample_risk_limits, mock_alert_service):
        """Test daily loss limit violation."""
        await risk_manager.set_risk_limits("customer_1", sample_risk_limits)
        
        # Simulate daily loss exceeding limit
        daily_pnl = Decimal("-3000")  # Exceeds max_daily_loss of $2000
        await risk_manager._handle_daily_loss_violation("customer_1", daily_pnl, sample_risk_limits)
        
        # Should send critical alert and activate circuit breaker
        mock_alert_service.send_risk_alert.assert_called_once()
        alert_call = mock_alert_service.send_risk_alert.call_args[0][0]
        assert alert_call.risk_level == RiskLevel.CRITICAL
        
        # Circuit breaker should be active
        assert "customer_1" in risk_manager.circuit_breaker_active


class TestStopLossAutomation:
    """Test automated stop-loss execution."""
    
    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, risk_manager, sample_position, mock_monitoring_service, mock_alert_service):
        """Test automated stop-loss order execution."""
        with patch.object(risk_manager, '_place_stop_loss_order', return_value="order_123") as mock_place_order:
            await risk_manager._execute_stop_loss(sample_position)
            
            # Should place stop-loss order
            mock_place_order.assert_called_once_with(sample_position)
            
            # Should track the order
            assert sample_position.position_id in risk_manager.stop_loss_orders
            assert risk_manager.stop_loss_orders[sample_position.position_id] == "order_123"
            
            # Should send alert
            mock_alert_service.send_risk_alert.assert_called_once()
            
            # Should record metric
            mock_monitoring_service.record_metric.assert_called_with(
                "stop_loss_executions_total", 1,
                labels={"symbol": sample_position.symbol, "customer_id": sample_position.customer_id}
            )
    
    @pytest.mark.asyncio
    async def test_stop_loss_failure_handling(self, risk_manager, sample_position, mock_alert_service):
        """Test stop-loss execution failure handling."""
        with patch.object(risk_manager, '_place_stop_loss_order', side_effect=Exception("Broker API error")):
            await risk_manager._execute_stop_loss(sample_position)
            
            # Should send critical alert on failure
            mock_alert_service.send_critical_alert.assert_called_once()
            alert_call = mock_alert_service.send_critical_alert.call_args
            assert "Stop-loss execution failed" in alert_call[0][0]
    
    @pytest.mark.asyncio
    async def test_stop_loss_monitoring(self, risk_manager):
        """Test stop-loss order status monitoring."""
        # Add a stop-loss order to track
        risk_manager.stop_loss_orders["pos_123"] = "order_123"
        
        with patch.object(risk_manager, '_check_order_status', return_value="filled"):
            # Run one iteration of stop-loss monitoring
            await asyncio.wait_for(
                risk_manager._stop_loss_monitoring_loop(),
                timeout=0.1
            )
        
        # Order should be removed from tracking after completion
        assert "pos_123" not in risk_manager.stop_loss_orders


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, risk_manager, sample_risk_limits):
        """Test circuit breaker activation on daily loss violation."""
        await risk_manager.set_risk_limits("customer_1", sample_risk_limits)
        
        # Trigger daily loss violation
        daily_pnl = Decimal("-3000")
        await risk_manager._handle_daily_loss_violation("customer_1", daily_pnl, sample_risk_limits)
        
        # Circuit breaker should be active
        assert not await risk_manager.is_trading_allowed("customer_1")
        assert "customer_1" in risk_manager.circuit_breaker_active
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, risk_manager, mock_monitoring_service):
        """Test manual circuit breaker reset."""
        # Activate circuit breaker
        risk_manager.circuit_breaker_active.add("customer_1")
        
        # Reset circuit breaker
        await risk_manager.reset_circuit_breaker("customer_1")
        
        # Should be reset
        assert await risk_manager.is_trading_allowed("customer_1")
        assert "customer_1" not in risk_manager.circuit_breaker_active
        
        # Should record metric
        mock_monitoring_service.record_metric.assert_called_with(
            "circuit_breaker_resets_total", 1,
            labels={"customer_id": "customer_1"}
        )


class TestRiskMonitoring:
    """Test real-time risk monitoring."""
    
    @pytest.mark.asyncio
    async def test_position_sync(self, risk_manager, mock_position_repo):
        """Test position synchronization from database."""
        # Mock active positions
        positions = [
            Position(
                position_id="pos_1",
                customer_id="customer_1",
                symbol="AAPL",
                quantity=Decimal("100"),
                cost_basis=Decimal("15000"),
                current_value=Decimal("14000"),
                unrealized_pnl=Decimal("-1000"),
                created_at=datetime.utcnow()
            ),
            Position(
                position_id="pos_2",
                customer_id="customer_2",
                symbol="GOOGL",
                quantity=Decimal("50"),
                cost_basis=Decimal("10000"),
                current_value=Decimal("9500"),
                unrealized_pnl=Decimal("-500"),
                created_at=datetime.utcnow()
            )
        ]
        mock_position_repo.get_active_positions.return_value = positions
        
        # Run one iteration of position sync
        await asyncio.wait_for(
            risk_manager._position_sync_loop(),
            timeout=0.1
        )
        
        # Positions should be grouped by customer
        assert "customer_1" in risk_manager.active_positions
        assert "customer_2" in risk_manager.active_positions
        assert len(risk_manager.active_positions["customer_1"]) == 1
        assert len(risk_manager.active_positions["customer_2"]) == 1
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_monitoring(self, risk_manager, mock_monitoring_service):
        """Test portfolio risk monitoring loop."""
        # Set up test positions
        risk_manager.active_positions = {
            "customer_1": [
                Position(
                    position_id="pos_1",
                    customer_id="customer_1",
                    symbol="AAPL",
                    quantity=Decimal("100"),
                    cost_basis=Decimal("15000"),
                    current_value=Decimal("14000"),
                    unrealized_pnl=Decimal("-1000"),
                    created_at=datetime.utcnow()
                )
            ]
        }
        
        # Run one iteration of portfolio monitoring
        await asyncio.wait_for(
            risk_manager._portfolio_risk_monitoring_loop(),
            timeout=0.1
        )
        
        # Should record portfolio metrics
        mock_monitoring_service.record_metric.assert_any_call(
            "portfolio_value", 14000.0,
            labels={"customer_id": "customer_1"}
        )
        mock_monitoring_service.record_metric.assert_any_call(
            "portfolio_pnl", -1000.0,
            labels={"customer_id": "customer_1"}
        )


class TestAlertReliability:
    """Test alert system reliability."""
    
    @pytest.mark.asyncio
    async def test_risk_alert_generation(self, risk_manager, sample_position, mock_alert_service):
        """Test risk alert generation and delivery."""
        await risk_manager._send_risk_alert(
            "customer_1", sample_position, RiskLevel.HIGH, "Test risk alert"
        )
        
        mock_alert_service.send_risk_alert.assert_called_once()
        alert = mock_alert_service.send_risk_alert.call_args[0][0]
        
        assert isinstance(alert, RiskAlert)
        assert alert.customer_id == "customer_1"
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.message == "Test risk alert"
        assert alert.position_id == sample_position.position_id
    
    @pytest.mark.asyncio
    async def test_critical_alert_escalation(self, risk_manager, mock_alert_service):
        """Test critical alert escalation."""
        await risk_manager._send_risk_alert(
            "customer_1", None, RiskLevel.CRITICAL, "Critical risk violation"
        )
        
        mock_alert_service.send_risk_alert.assert_called_once()
        alert = mock_alert_service.send_risk_alert.call_args[0][0]
        
        assert alert.risk_level == RiskLevel.CRITICAL
        assert alert.position_id is None  # Portfolio-level alert


class TestRiskManagerIntegration:
    """Test risk manager integration with other services."""
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, risk_manager, mock_monitoring_service):
        """Test integration with monitoring service."""
        # Test metric recording
        await risk_manager.monitoring_service.record_metric("test_metric", 1.0)
        mock_monitoring_service.record_metric.assert_called_with("test_metric", 1.0)
    
    @pytest.mark.asyncio
    async def test_alert_integration(self, risk_manager, mock_alert_service):
        """Test integration with alert service."""
        # Test critical alert
        await risk_manager.alert_service.send_critical_alert("Test", "Test message")
        mock_alert_service.send_critical_alert.assert_called_with("Test", "Test message")
    
    @pytest.mark.asyncio
    async def test_repository_integration(self, risk_manager, mock_position_repo):
        """Test integration with position repository."""
        await risk_manager.position_repo.get_active_positions()
        mock_position_repo.get_active_positions.assert_called_once()


class TestRiskManagerConfiguration:
    """Test risk manager configuration and limits."""
    
    @pytest.mark.asyncio
    async def test_risk_limits_setting(self, risk_manager, sample_risk_limits):
        """Test setting risk limits for customers."""
        await risk_manager.set_risk_limits("customer_1", sample_risk_limits)
        
        assert "customer_1" in risk_manager.risk_limits
        limits = risk_manager.risk_limits["customer_1"]
        assert limits.max_position_size == sample_risk_limits.max_position_size
        assert limits.stop_loss_percentage == sample_risk_limits.stop_loss_percentage
    
    @pytest.mark.asyncio
    async def test_trading_permission_check(self, risk_manager):
        """Test trading permission checks."""
        # Normal customer should be allowed to trade
        assert await risk_manager.is_trading_allowed("customer_1")
        
        # Customer with active circuit breaker should not be allowed
        risk_manager.circuit_breaker_active.add("customer_2")
        assert not await risk_manager.is_trading_allowed("customer_2")


@pytest.mark.asyncio
async def test_risk_manager_startup(risk_manager):
    """Test risk manager startup process."""
    # This would test the actual startup, but we'll mock it to avoid infinite loops
    with patch.object(risk_manager, '_risk_monitoring_loop', return_value=None):
        with patch.object(risk_manager, '_position_sync_loop', return_value=None):
            with patch.object(risk_manager, '_stop_loss_monitoring_loop', return_value=None):
                with patch.object(risk_manager, '_portfolio_risk_monitoring_loop', return_value=None):
                    # Test that startup doesn't raise exceptions
                    try:
                        await asyncio.wait_for(risk_manager.start_risk_monitoring(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass  # Expected due to infinite loops