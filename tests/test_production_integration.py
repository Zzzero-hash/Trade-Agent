"""
Simple production integration test to verify monitoring orchestrator works.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.services.production_monitoring_orchestrator import (
    ProductionMonitoringOrchestrator,
    create_production_monitoring,
)
from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository


@pytest.fixture
def simple_config():
    """Create minimal test configuration."""
    return {
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "port": 8001,
                "metrics_path": "/metrics",
                "scrape_interval": "15s",
            },
            "grafana": {
                "enabled": False,
                "url": "http://localhost:3000",
                "api_key": "test_key",
                "datasource": "prometheus",
            },
            "thresholds": {
                "request_latency_warning": 500,
                "request_latency_critical": 1000,
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
                "daily_loss_critical": 10000.0,
            },
        },
        "risk_management": {
            "risk_check_interval": 1,
            "position_sync_interval": 2,
            "portfolio_monitoring_interval": 3,
            "default_limits": {
                "max_position_size": 10000.0,
                "max_daily_loss": 1000.0,
                "max_portfolio_risk": 0.15,
                "stop_loss_percentage": 0.10,
                "max_leverage": 2.0,
            },
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 60,
                "success_threshold": 2,
            },
        },
        "alerting": {
            "pagerduty": {
                "enabled": False,
                "integration_key": "test_key",
                "api_url": "https://events.pagerduty.com/v2/enqueue",
            },
            "slack": {
                "enabled": False,
                "webhook_url": "https://hooks.slack.com/test",
                "channel": "#test",
                "username": "Test Bot",
            },
            "email": {
                "enabled": False,
                "smtp_host": "localhost",
                "smtp_port": 587,
                "username": "test",
                "password": "test",
                "from_address": "test@example.com",
            },
        },
    }


def test_orchestrator_creation(simple_config):
    """Test that orchestrator can be created with config."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(simple_config, f)
        config_path = f.name

    try:
        orchestrator = ProductionMonitoringOrchestrator(config_path)

        assert orchestrator is not None
        assert orchestrator.config is not None
        assert orchestrator.alert_service is not None
        assert orchestrator.monitoring_service is not None
        assert orchestrator.performance_service is not None
        assert orchestrator.circuit_breaker_service is not None
        assert not orchestrator.services_running

    finally:
        Path(config_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_orchestrator_with_repositories(simple_config):
    """Test orchestrator with mock repositories."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(simple_config, f)
        config_path = f.name

    # Create mock repositories
    position_repo = Mock(spec=PositionRepository)
    position_repo.get_active_positions = AsyncMock(return_value=[])

    trade_repo = Mock(spec=TradeRepository)
    trade_repo.get_trades_by_date = AsyncMock(return_value=[])

    try:
        orchestrator = await create_production_monitoring(
            config_path, position_repo, trade_repo
        )

        assert orchestrator is not None
        assert orchestrator.risk_manager is not None

    finally:
        Path(config_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_risk_limits_configuration(simple_config):
    """Test configuring risk limits."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(simple_config, f)
        config_path = f.name

    # Create mock repositories
    position_repo = Mock(spec=PositionRepository)
    position_repo.get_active_positions = AsyncMock(return_value=[])

    trade_repo = Mock(spec=TradeRepository)
    trade_repo.get_trades_by_date = AsyncMock(return_value=[])

    try:
        orchestrator = await create_production_monitoring(
            config_path, position_repo, trade_repo
        )

        # Configure risk limits
        await orchestrator.configure_customer_risk_limits(
            "test_customer",
            {
                "max_position_size": 5000.0,
                "max_daily_loss": 500.0,
                "stop_loss_percentage": 0.05,
            },
        )

        # Verify limits were set
        assert "test_customer" in orchestrator.risk_manager.risk_limits
        limits = orchestrator.risk_manager.risk_limits["test_customer"]
        assert limits.max_position_size == 5000.0
        assert limits.max_daily_loss == 500.0
        assert limits.stop_loss_percentage == 0.05

    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
