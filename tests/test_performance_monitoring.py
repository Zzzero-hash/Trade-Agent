"""
Performance Monitoring Tests

Tests for the performance monitoring service functionality.
"""

import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.services.production_monitoring_orchestrator import ProductionMonitoringOrchestrator
from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository


def test_performance_monitoring_creation():
    """Test that performance monitoring can be created."""
    config = {
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "port": 8001,
                "metrics_path": "/metrics",
                "scrape_interval": "15s"
            },
            "grafana": {
                "enabled": False,
                "url": "http://localhost:3000",
                "api_key": "test_key",
                "datasource": "prometheus"
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
                "daily_loss_critical": 10000.0
            }
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
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        orchestrator = ProductionMonitoringOrchestrator(config_path)
        
        # Verify orchestrator components
        assert orchestrator is not None
        assert orchestrator.config is not None
        assert orchestrator.alert_service is not None
        assert orchestrator.monitoring_service is not None
        assert orchestrator.performance_service is not None
        assert orchestrator.circuit_breaker_service is not None
        assert not orchestrator.services_running
        
        # Verify configuration loaded correctly
        assert orchestrator.config["monitoring"]["prometheus"]["port"] == 8001
        assert orchestrator.config["risk_management"]["risk_check_interval"] == 1
        assert orchestrator.config["alerting"]["pagerduty"]["enabled"] is False
        
        print("✓ Performance monitoring orchestrator created successfully")
        
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_risk_management_integration():
    """Test risk management integration."""
    config = {
        "monitoring": {
            "prometheus": {"enabled": True, "port": 8001, "metrics_path": "/metrics", "scrape_interval": "15s"},
            "grafana": {"enabled": False, "url": "http://localhost:3000", "api_key": "test_key", "datasource": "prometheus"},
            "thresholds": {"request_latency_warning": 500, "request_latency_critical": 1000, "error_rate_warning": 2.0, "error_rate_critical": 5.0}
        },
        "risk_management": {
            "risk_check_interval": 1,
            "position_sync_interval": 2,
            "default_limits": {"max_position_size": 10000.0, "max_daily_loss": 1000.0, "max_portfolio_risk": 0.15, "stop_loss_percentage": 0.10, "max_leverage": 2.0},
            "circuit_breaker": {"failure_threshold": 3, "recovery_timeout": 60, "success_threshold": 2}
        },
        "alerting": {
            "pagerduty": {"enabled": False, "integration_key": "test_key", "api_url": "https://events.pagerduty.com/v2/enqueue"},
            "slack": {"enabled": False, "webhook_url": "https://hooks.slack.com/test", "channel": "#test", "username": "Test Bot"},
            "email": {"enabled": False, "smtp_host": "localhost", "smtp_port": 587, "username": "test", "password": "test", "from_address": "test@example.com"}
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    # Create mock repositories
    position_repo = Mock(spec=PositionRepository)
    position_repo.get_active_positions = AsyncMock(return_value=[])
    
    trade_repo = Mock(spec=TradeRepository)
    trade_repo.get_trades_by_date = AsyncMock(return_value=[])
    
    try:
        orchestrator = ProductionMonitoringOrchestrator(config_path)
        orchestrator.initialize_risk_manager(position_repo, trade_repo)
        
        # Verify risk manager was initialized
        assert orchestrator.risk_manager is not None
        assert orchestrator.risk_manager.position_repo == position_repo
        assert orchestrator.risk_manager.trade_repo == trade_repo
        
        print("✓ Risk management integration successful")
        
    finally:
        Path(config_path).unlink(missing_ok=True)


async def test_monitoring_services_async():
    """Test async monitoring service operations."""
    config = {
        "monitoring": {
            "prometheus": {"enabled": True, "port": 8002, "metrics_path": "/metrics", "scrape_interval": "15s"},
            "grafana": {"enabled": False, "url": "http://localhost:3000", "api_key": "test_key", "datasource": "prometheus"},
            "thresholds": {"request_latency_warning": 500, "request_latency_critical": 1000, "error_rate_warning": 2.0, "error_rate_critical": 5.0}
        },
        "risk_management": {
            "risk_check_interval": 1,
            "position_sync_interval": 2,
            "default_limits": {"max_position_size": 10000.0, "max_daily_loss": 1000.0, "max_portfolio_risk": 0.15, "stop_loss_percentage": 0.10, "max_leverage": 2.0},
            "circuit_breaker": {"failure_threshold": 3, "recovery_timeout": 60, "success_threshold": 2}
        },
        "alerting": {
            "pagerduty": {"enabled": False, "integration_key": "test_key", "api_url": "https://events.pagerduty.com/v2/enqueue"},
            "slack": {"enabled": False, "webhook_url": "https://hooks.slack.com/test", "channel": "#test", "username": "Test Bot"},
            "email": {"enabled": False, "smtp_host": "localhost", "smtp_port": 587, "username": "test", "password": "test", "from_address": "test@example.com"}
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        orchestrator = ProductionMonitoringOrchestrator(config_path)
        
        # Test metrics recording
        await orchestrator.monitoring_service.record_metric("test_metric", 42.0)
        
        # Test alert service
        await orchestrator.alert_service.send_warning_alert(
            "Test Alert",
            "This is a test alert for monitoring verification"
        )
        
        # Test getting monitoring status
        status = await orchestrator.get_monitoring_status()
        assert "services_running" in status
        assert "timestamp" in status
        assert "services" in status
        
        print("✓ Async monitoring operations successful")
        
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run synchronous tests
    test_performance_monitoring_creation()
    test_risk_management_integration()
    
    # Run async test
    asyncio.run(test_monitoring_services_async())
    
    print("\n✅ All performance monitoring tests passed!")