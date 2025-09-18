"""
Production Monitoring Orchestrator

Coordinates all monitoring services, manages configuration,
and provides unified monitoring management for production deployment.
"""

import asyncio
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.services.monitoring_service import MonitoringService
from src.services.performance_monitoring_service import PerformanceMonitoringService
from src.services.risk_management_service import ProductionRiskManager
from src.models.risk_models import RiskLimits
from src.services.alert_service import (
    ProductionAlertService, PagerDutyConfig, SlackConfig, EmailConfig
)
from src.services.circuit_breaker_service import CircuitBreakerService
from src.services.grafana_integration_service import (
    GrafanaIntegrationService, GrafanaConfig
)
from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository


class ProductionMonitoringOrchestrator:
    """Orchestrates all production monitoring services."""
    
    def __init__(self, config_path: str = "src/config/production_monitoring.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.alert_service = self._create_alert_service()
        self.monitoring_service = MonitoringService(
            alert_service=self.alert_service,
            metrics_port=self.config["monitoring"]["prometheus"]["port"]
        )
        self.performance_service = PerformanceMonitoringService(
            monitoring_service=self.monitoring_service,
            alert_service=self.alert_service
        )
        self.circuit_breaker_service = CircuitBreakerService(
            monitoring_service=self.monitoring_service,
            alert_service=self.alert_service
        )
        
        # Risk management service (requires repositories)
        self.risk_manager: Optional[ProductionRiskManager] = None
        
        # Grafana integration
        self.grafana_service: Optional[GrafanaIntegrationService] = None
        if self.config["monitoring"]["grafana"]["enabled"]:
            self.grafana_service = self._create_grafana_service()
        
        # Service status
        self.services_running = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Expand environment variables
            config = self._expand_env_vars(config)
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in config."""
        import os
        import re
        
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Replace ${VAR} with environment variable
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            for match in matches:
                env_value = os.getenv(match, "")
                obj = obj.replace(f"${{{match}}}", env_value)
            return obj
        else:
            return obj
    
    def _create_alert_service(self) -> ProductionAlertService:
        """Create alert service with configured channels."""
        alert_config = self.config["alerting"]
        
        # PagerDuty configuration
        pagerduty_config = None
        if alert_config["pagerduty"]["enabled"]:
            pagerduty_config = PagerDutyConfig(
                integration_key=alert_config["pagerduty"]["integration_key"],
                api_url=alert_config["pagerduty"]["api_url"]
            )
        
        # Slack configuration
        slack_config = None
        if alert_config["slack"]["enabled"]:
            slack_config = SlackConfig(
                webhook_url=alert_config["slack"]["webhook_url"],
                channel=alert_config["slack"]["channel"],
                username=alert_config["slack"]["username"]
            )
        
        # Email configuration
        email_config = None
        if alert_config["email"]["enabled"]:
            email_config = EmailConfig(
                smtp_host=alert_config["email"]["smtp_host"],
                smtp_port=alert_config["email"]["smtp_port"],
                username=alert_config["email"]["username"],
                password=alert_config["email"]["password"],
                from_address=alert_config["email"]["from_address"]
            )
        
        return ProductionAlertService(
            pagerduty_config=pagerduty_config,
            slack_config=slack_config,
            email_config=email_config
        )
    
    def _create_grafana_service(self) -> GrafanaIntegrationService:
        """Create Grafana integration service."""
        grafana_config = self.config["monitoring"]["grafana"]
        
        config = GrafanaConfig(
            url=grafana_config["url"],
            api_key=grafana_config["api_key"],
            datasource=grafana_config["datasource"]
        )
        
        return GrafanaIntegrationService(config)
    
    def initialize_risk_manager(
        self,
        position_repo: PositionRepository,
        trade_repo: TradeRepository
    ):
        """Initialize risk manager with repositories."""
        self.risk_manager = ProductionRiskManager(
            position_repo=position_repo,
            trade_repo=trade_repo,
            monitoring_service=self.monitoring_service,
            alert_service=self.alert_service
        )
        
        # Configure risk manager intervals
        risk_config = self.config["risk_management"]
        self.risk_manager.risk_check_interval = risk_config["risk_check_interval"]
        self.risk_manager.position_sync_interval = risk_config["position_sync_interval"]
    
    async def start_monitoring(self):
        """Start all monitoring services."""
        if self.services_running:
            self.logger.warning("Monitoring services already running")
            return
        
        self.logger.info("Starting production monitoring services")
        
        try:
            # Configure performance thresholds
            await self._configure_performance_thresholds()
            
            # Configure circuit breakers
            await self._configure_circuit_breakers()
            
            # Setup Grafana dashboards
            if self.grafana_service:
                await self._setup_grafana_dashboards()
            
            # Start monitoring services
            monitoring_tasks = [
                asyncio.create_task(self.monitoring_service.start_monitoring()),
                asyncio.create_task(self.performance_service.start_monitoring())
            ]
            
            # Start risk manager if initialized
            if self.risk_manager:
                monitoring_tasks.append(
                    asyncio.create_task(self.risk_manager.start_risk_monitoring())
                )
            
            # Start health check task
            monitoring_tasks.append(
                asyncio.create_task(self._health_check_loop())
            )
            
            self.monitoring_tasks = monitoring_tasks
            self.services_running = True
            
            self.logger.info("Production monitoring services started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*monitoring_tasks)
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring services: {e}")
            await self.stop_monitoring()
            raise
    
    async def stop_monitoring(self):
        """Stop all monitoring services."""
        if not self.services_running:
            return
        
        self.logger.info("Stopping production monitoring services")
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Stop individual services
        self.monitoring_service.stop_monitoring()
        
        self.services_running = False
        self.monitoring_tasks = []
        
        self.logger.info("Production monitoring services stopped")
    
    async def _configure_performance_thresholds(self):
        """Configure performance monitoring thresholds."""
        thresholds = self.config["monitoring"]["thresholds"]
        
        # Configure request latency thresholds
        self.performance_service.set_thresholds(
            operation="api_requests",
            latency_threshold_ms=thresholds["request_latency_critical"],
            error_rate_threshold=thresholds["error_rate_critical"] / 100.0
        )
        
        # Configure trade execution thresholds
        self.performance_service.set_thresholds(
            operation="trade_execution",
            latency_threshold_ms=thresholds["trade_execution_latency_critical"],
            error_rate_threshold=thresholds["error_rate_critical"] / 100.0
        )
        
        # Configure feature extraction thresholds
        self.performance_service.set_thresholds(
            operation="feature_extraction",
            latency_threshold_ms=thresholds["feature_extraction_latency_critical"],
            error_rate_threshold=thresholds["error_rate_critical"] / 100.0
        )
        
        self.logger.info("Performance thresholds configured")
    
    async def _configure_circuit_breakers(self):
        """Configure circuit breakers for external services."""
        from src.services.circuit_breaker_service import CircuitBreakerConfig
        
        cb_config = self.config["risk_management"]["circuit_breaker"]
        
        # Default circuit breaker configuration
        default_config = CircuitBreakerConfig(
            failure_threshold=cb_config["failure_threshold"],
            recovery_timeout=cb_config["recovery_timeout"],
            success_threshold=cb_config["success_threshold"]
        )
        
        # Create circuit breakers for external services
        external_services = [
            "robinhood_api",
            "td_ameritrade_api", 
            "interactive_brokers_api",
            "coinbase_api",
            "oanda_api"
        ]
        
        for service in external_services:
            self.circuit_breaker_service.create_circuit_breaker(service, default_config)
        
        self.logger.info("Circuit breakers configured")
    
    async def _setup_grafana_dashboards(self):
        """Setup Grafana dashboards."""
        try:
            async with self.grafana_service as grafana:
                dashboard_uids = await grafana.create_trading_platform_dashboards()
                self.logger.info(f"Created Grafana dashboards: {dashboard_uids}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup Grafana dashboards: {e}")
    
    async def _health_check_loop(self):
        """Continuous health check loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(300)
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        health_status = {
            "timestamp": datetime.utcnow(),
            "overall_healthy": True,
            "services": {}
        }
        
        # Check monitoring service
        try:
            metrics_summary = await self.monitoring_service.get_metrics_summary()
            health_status["services"]["monitoring"] = {
                "healthy": True,
                "metrics_count": len(metrics_summary)
            }
        except Exception as e:
            health_status["services"]["monitoring"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["overall_healthy"] = False
        
        # Check performance service
        try:
            perf_summary = await self.performance_service.get_performance_summary()
            health_status["services"]["performance"] = {
                "healthy": True,
                "operations_tracked": len(perf_summary.get("operations", {}))
            }
        except Exception as e:
            health_status["services"]["performance"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["overall_healthy"] = False
        
        # Check circuit breakers
        try:
            cb_health = await self.circuit_breaker_service.health_check()
            health_status["services"]["circuit_breakers"] = {
                "healthy": cb_health["health_score"] > 0.8,
                "open_breakers": cb_health["open_circuit_breakers"],
                "total_breakers": cb_health["total_circuit_breakers"]
            }
            
            if cb_health["health_score"] <= 0.8:
                health_status["overall_healthy"] = False
                
        except Exception as e:
            health_status["services"]["circuit_breakers"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["overall_healthy"] = False
        
        # Check risk manager if available
        if self.risk_manager:
            try:
                # Check if risk monitoring is active
                active_positions = len(self.risk_manager.active_positions)
                circuit_breakers_active = len(self.risk_manager.circuit_breaker_active)
                
                health_status["services"]["risk_management"] = {
                    "healthy": True,
                    "active_positions": active_positions,
                    "circuit_breakers_active": circuit_breakers_active
                }
                
                if circuit_breakers_active > 0:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                health_status["services"]["risk_management"] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        
        # Record health metrics
        await self.monitoring_service.record_metric(
            "system_health_score",
            1.0 if health_status["overall_healthy"] else 0.0
        )
        
        # Send alert if unhealthy
        if not health_status["overall_healthy"]:
            unhealthy_services = [
                name for name, status in health_status["services"].items()
                if not status["healthy"]
            ]
            
            await self.alert_service.send_error_alert(
                "System Health Check Failed",
                f"Unhealthy services: {', '.join(unhealthy_services)}",
                metadata=health_status
            )
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        status = {
            "services_running": self.services_running,
            "timestamp": datetime.utcnow(),
            "services": {}
        }
        
        if self.services_running:
            # Get monitoring service status
            try:
                status["services"]["monitoring"] = await self.monitoring_service.get_metrics_summary()
            except Exception as e:
                status["services"]["monitoring"] = {"error": str(e)}
            
            # Get performance service status
            try:
                status["services"]["performance"] = await self.performance_service.get_performance_summary()
            except Exception as e:
                status["services"]["performance"] = {"error": str(e)}
            
            # Get circuit breaker status
            try:
                status["services"]["circuit_breakers"] = await self.circuit_breaker_service.health_check()
            except Exception as e:
                status["services"]["circuit_breakers"] = {"error": str(e)}
            
            # Get alert service status
            try:
                status["services"]["alerts"] = await self.alert_service.get_alert_statistics()
            except Exception as e:
                status["services"]["alerts"] = {"error": str(e)}
        
        return status
    
    async def configure_customer_risk_limits(self, customer_id: str, limits: Dict[str, float]):
        """Configure risk limits for a customer."""
        if not self.risk_manager:
            raise ValueError("Risk manager not initialized")
        
        risk_limits = RiskLimits(
            max_position_size=limits.get("max_position_size", 50000.0),
            max_daily_loss=limits.get("max_daily_loss", 5000.0),
            max_portfolio_risk=limits.get("max_portfolio_risk", 0.15),
            max_correlation_exposure=limits.get("max_correlation_exposure", 0.3),
            stop_loss_percentage=limits.get("stop_loss_percentage", 0.1),
            max_leverage=limits.get("max_leverage", 2.0)
        )
        
        await self.risk_manager.set_risk_limits(customer_id, risk_limits)
        self.logger.info(f"Configured risk limits for customer {customer_id}")
    
    async def emergency_stop(self):
        """Emergency stop - activate all circuit breakers."""
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        
        # Activate all circuit breakers
        await self.circuit_breaker_service.force_open_all()
        
        # Send critical alert
        await self.alert_service.send_critical_alert(
            "EMERGENCY STOP ACTIVATED",
            "All trading operations have been halted due to emergency stop activation"
        )
    
    async def emergency_recovery(self):
        """Emergency recovery - reset all circuit breakers."""
        self.logger.warning("EMERGENCY RECOVERY INITIATED")
        
        # Reset all circuit breakers
        await self.circuit_breaker_service.force_close_all()
        
        # Send warning alert
        await self.alert_service.send_warning_alert(
            "EMERGENCY RECOVERY COMPLETED",
            "All trading operations have been restored after emergency recovery"
        )


# Factory function for easy initialization
async def create_production_monitoring(
    config_path: str = "src/config/production_monitoring.yaml",
    position_repo: Optional[PositionRepository] = None,
    trade_repo: Optional[TradeRepository] = None
) -> ProductionMonitoringOrchestrator:
    """Create and configure production monitoring orchestrator."""
    
    orchestrator = ProductionMonitoringOrchestrator(config_path)
    
    # Initialize risk manager if repositories provided
    if position_repo and trade_repo:
        orchestrator.initialize_risk_manager(position_repo, trade_repo)
    
    return orchestrator


if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = await create_production_monitoring()
        
        try:
            await orchestrator.start_monitoring()
        except KeyboardInterrupt:
            await orchestrator.stop_monitoring()
    
    asyncio.run(main())