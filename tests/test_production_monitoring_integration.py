"""
Production Monitoring Integration Tests

Tests the complete production monitoring system including risk management,
performance monitoring, alerting, and circuit breakers working together.
"""

import pytest
import asyncio
import tempfile
import yaml
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.services.production_monitoring_orchestrator import (
    ProductionMonitoringOrchestrator, create_production_monitoring
)
from src.services.risk_management_service import RiskLimits
from src.services.alert_service import AlertSeverity
from src.models.trading_models import Position
from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository


@pytest.fixture
def test_config():
    """Create test monitoring configuration."""
    return {
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "port": 8001,  # Different port for testing
                "metrics_path": "/metrics",
                "scrape_interval": "15s"
            },
            "grafana": {
                "enabled": False,  # Disable for testing
                "url": "http://localhost:3000",
                "api_key": "test_key",
                "datasource": "prometheus"
            },
            "thresholds": {
                "request_latency_warning": 500,
                "request_latency_critical": 1000,
                "trade_execution_latency_warning": 250,
                "trade_execution_latency_critical": 500,
                "feature_extraction_latency_warning": 50,
                "feature_extraction_latency_critical": 100,
                "error_rate_warning": 2.0,
                "error_rate_critical": 5.0,
                "cpu_usage_warning": 80.0,
                "cpu_usage_critical": 90.0,
                "memory_usage_warning": 85.0,
                "memory_usage_critical": 95.0,
                "disk_usage_warning": 90.0,
                "disk_usage_critical": 95.0,
                "portfolio_drawdown_warning": 10.0,
                "portfolio_drawdown_critical": 15.0,
                "daily_loss_warning": 5000.0,
                "daily_loss_critical": 10000.0
            }
        },
        "risk_management": {
            "risk_check_interval": 1,  # Fast for testing
            "position_sync_interval": 2,
            "portfolio_monitoring_interval": 3,
            "default_limits": {
                "max_position_size": 10000.0,
                "max_daily_loss": 1000.0,
                "max_portfolio_risk": 0.15,
                "stop_loss_percentage": 0.10,
                "max_leverage": 2.0
            },
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 60,
                "success_threshold": 2
            }
        },
        "alerting": {
            "pagerduty": {
                "enabled": False,
                "integration_key": "test_key",
                "api_url": "https://events.pagerduty.com/v2/enqueue"
            },
            "slack": {
                "enabled": False,
                "webhook_url": "https://hooks.slack.com/test",
                "channel": "#test",
                "username": "Test Bot"
            },
            "email": {
                "enabled": False,
                "smtp_host": "localhost",
                "smtp_port": 587,
                "username": "test",
                "password": "test",
                "from_address": "test@example.com"
            }
        }
    }
           


@pytest.fixture
def mock_position_repo():
    """Mock position repository."""
    repo = Mock(spec=PositionRepository)
    repo.get_active_positions = AsyncMock(return_value=[])
    repo.get_position_by_id = AsyncMock(return_value=None)
    repo.update_position = AsyncMock()
    return repo


@pytest.fixture
def mock_trade_repo():
    """Mock trade repository."""
    repo = Mock(spec=TradeRepository)
    repo.get_trades_by_date = AsyncMock(return_value=[])
    repo.create_trade = AsyncMock()
    return repo


@pytest.fixture
async def monitoring_orchestrator(test_config, mock_position_repo, mock_trade_repo):
    """Create monitoring orchestrator for testing."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        orchestrator = ProductionMonitoringOrchestrator(config_path)
        orchestrator.initialize_risk_manager(mock_position_repo, mock_trade_repo)
        yield orchestrator
    finally:
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
        if orchestrator.services_running:
            await orchestrator.stop_monitoring()


class TestProductionMonitoringIntegration:
    """Integration tests for production monitoring system."""
    
    async def test_monitoring_orchestrator_initialization(self, monitoring_orchestrator):
        """Test that monitoring orchestrator initializes correctly."""
        assert monitoring_orchestrator.config is not None
        assert monitoring_orchestrator.alert_service is not None
        assert monitoring_orchestrator.monitoring_service is not None
        assert monitoring_orchestrator.performance_service is not None
        assert monitoring_orchestrator.circuit_breaker_service is not None
        assert monitoring_orchestrator.risk_manager is not None
        assert not monitoring_orchestrator.services_running
    
    async def test_monitoring_service_startup_and_shutdown(self, monitoring_orchestrator):
        """Test monitoring services can start and stop properly."""
        # Start monitoring
        start_task = asyncio.create_task(monitoring_orchestrator.start_monitoring())
        
        # Give it time to start
        await asyncio.sleep(0.1)
        
        assert monitoring_orchestrator.services_running
        assert len(monitoring_orchestrator.monitoring_tasks) > 0
        
        # Stop monitoring
        await monitoring_orchestrator.stop_monitoring()
        
        assert not monitoring_orchestrator.services_running
        assert len(monitoring_orchestrator.monitoring_tasks) == 0
        
        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    async def test_risk_management_integration(self, monitoring_orchestrator, mock_position_repo):
        """Test risk management integration with monitoring."""
        # Setup test position
        test_position = Position(
            id="test_pos_1",
            customer_id="test_customer",
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            current_price=Decimal("140.00"),  # 10 loss
            position_type="long",
            created_at=datetime.utcnow()
        )
        
        mock_position_repo.get_active_positions.return_value = [test_position]
        
        # Configure risk limits
        await monitoring_orchestrator.configure_customer_risk_limits(
            "test_customer",
            {
                "max_position_size": 5000.0,
                "max_daily_loss": 500.0,
                "stop_loss_percentage": 0.05
            }
        )
        
        # Verify risk limits were set
        risk_manager = monitoring_orchestrator.risk_manager
        assert "test_customer" in risk_manager.risk_limits
        
        limits = risk_manager.risk_limits["test_customer"]
        assert limits.max_position_size == 5000.0
        assert limits.max_daily_loss == 500.0
        assert limits.stop_loss_percentage == 0.05
    
    async def test_performance_monitoring_thresholds(self, monitoring_orchestrator):
        """Test performance monitoring threshold configuration."""
        # Start monitoring to configure thresholds
        start_task = asyncio.create_task(monitoring_orchestrator.start_monitoring())
        await asyncio.sleep(0.1)
        
        # Check that thresholds were configured
        perf_service = monitoring_orchestrator.performance_service
        
        # Verify thresholds exist for key operations
        assert "api_requests" in perf_service.operation_thresholds
        assert "trade_execution" in perf_service.operation_thresholds
        assert "feature_extraction" in perf_service.operation_thresholds
        
        # Stop monitoring
        await monitoring_orchestrator.stop_monitoring()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    async def test_circuit_breaker_configuration(self, monitoring_orchestrator):
        """Test circuit breaker configuration."""
        # Start monitoring to configure circuit breakers
        start_task = asyncio.create_task(monitoring_orchestrator.start_monitoring())
        await asyncio.sleep(0.1)
        
        # Check that circuit breakers were created
        cb_service = monitoring_orchestrator.circuit_breaker_service
        
        # Verify circuit breakers exist for external services
        expected_services = [
            "robinhood_api",
            "td_ameritrade_api", 
            "interactive_brokers_api",
            "coinbase_api",
            "oanda_api"
        ]
        
        for service in expected_services:
            assert service in cb_service.circuit_breakers
        
        # Stop monitoring
        await monitoring_orchestrator.stop_monitoring()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    async def test_health_check_system(self, monitoring_orchestrator):
        """Test comprehensive health check system."""
        # Start monitoring
        start_task = asyncio.create_task(monitoring_orchestrator.start_monitoring())
        await asyncio.sleep(0.2)  # Give time for health checks
        
        # Get monitoring status
        status = await monitoring_orchestrator.get_monitoring_status()
        
        assert status["services_running"] is True
        assert "services" in status
        assert "monitoring" in status["services"]
        assert "performance" in status["services"]
        assert "circuit_breakers" in status["services"]
        assert "alerts" in status["services"]
        
        # Stop monitoring
        await monitoring_orchestrator.stop_monitoring()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    async def test_emergency_stop_and_recovery(self, monitoring_orchestrator):
        """Test emergency stop and recovery procedures."""
        # Start monitoring
        start_task = asyncio.create_task(monitoring_orchestrator.start_monitoring())
        await asyncio.sleep(0.1)
        
        # Test emergency stop
        await monitoring_orchestrator.emergency_stop()
        
        # Verify all circuit breakers are open
        cb_health = await monitoring_orchestrator.circuit_breaker_service.health_check()
        assert cb_health["open_circuit_breakers"] == cb_health["total_circuit_breakers"]
        
        # Test emergency recovery
        await monitoring_orchestrator.emergency_recovery()
        
        # Verify circuit breakers are closed
        cb_health = await monitoring_orchestrator.circuit_breaker_service.health_check()
        assert cb_health["open_circuit_breakers"] == 0
        
        # Stop monitoring
        await monitoring_orchestrator.stop_monitoring()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    async def test_alert_integration(self, monitoring_orchestrator):
        """Test alert service integration."""
        alert_service = monitoring_orchestrator.alert_service
        
        # Test sending different types of alerts
        await alert_service.send_warning_alert(
            "Test Warning",
            "This is a test warning alert"
        )
        
        await alert_service.send_error_alert(
            "Test Error", 
            "This is a test error alert"
        )
        
        await alert_service.send_critical_alert(
            "Test Critical",
            "This is a test critical alert"
        )
        
        # Get alert statistics
        stats = await alert_service.get_alert_statistics()
        assert "total_alerts" in stats
        assert stats["total_alerts"] >= 3
    
    async def test_metrics_collection_and_reporting(self, monitoring_orchestrator):
        """Test metrics collection and reporting."""
        monitoring_service = monitoring_orchestrator.monitoring_service
        
        # Record some test metrics
        await monitoring_service.record_metric("test_counter", 1.0)
        await monitoring_service.record_metric("test_gauge", 42.0)
        await monitoring_service.record_histogram("test_histogram", 0.5)
        
        # Get metrics summary
        summary = await monitoring_service.get_metrics_summary()
        assert len(summary) >= 3
        
        # Verify metrics exist
        metric_names = [metric["name"] for metric in summary]
        assert "test_counter" in metric_names
        assert "test_gauge" in metric_names
        assert "test_histogram" in metric_names


class TestProductionMonitoringFactory:
    """Test the factory function for creating monitoring orchestrator."""
    
    async def test_create_production_monitoring_without_repos(self, test_config):
        """Test creating monitoring without repositories."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            orchestrator = await create_production_monitoring(config_path)
            
            assert orchestrator is not None
            assert orchestrator.risk_manager is None  # Should be None without repos
            
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    async def test_create_production_monitoring_with_repos(self, test_config):
        """Test creating monitoring with repositories."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        # Create mock repositories
        position_repo = Mock(spec=PositionRepository)
        trade_repo = Mock(spec=TradeRepository)
        
        try:
            orchestrator = await create_production_monitoring(
                config_path, position_repo, trade_repo
            )
            
            assert orchestrator is not None
            assert orchestrator.risk_manager is not None  # Should be initialized
            
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestProductionMonitoringErrorHandling:
    """Test error handling in production monitoring."""
    
    async def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        with pytest.raises(Exception):
            ProductionMonitoringOrchestrator("nonexistent_config.yaml")
    
    async def test_monitoring_service_failure_recovery(self, monitoring_orchestrator):
        """Test recovery from monitoring service failures."""
        # Mock a service failure
        with patch.object(
            monitoring_orchestrator.monitoring_service,
            'get_metrics_summary',
            side_effect=Exception("Service failure")
        ):
            # Health check should handle the failure gracefully
            await monitoring_orchestrator._perform_health_checks()
            
            # System should still be operational
            status = await monitoring_orchestrator.get_monitoring_status()
            assert "services" in status
    
    async def test_risk_manager_without_initialization(self):
        """Test risk manager operations without proper initialization."""
        orchestrator = ProductionMonitoringOrchestrator()
        
        with pytest.raises(ValueError, match="Risk manager not initialized"):
            await orchestrator.configure_customer_risk_limits("test", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])