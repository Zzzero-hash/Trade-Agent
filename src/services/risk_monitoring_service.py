"""
Real-time risk monitoring service with automated alerting and reporting.
"""
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta, timezone
import logging
from collections import defaultdict
import json

from src.models.portfolio import Portfolio
from src.models.risk_management import (
    RiskMetrics, RiskAlert, RiskLimit, StressTestResult,
    RiskLimitType, RiskLimitStatus
)
from src.services.risk_manager import RiskManager


class RiskMonitoringService:
    """
    Real-time risk monitoring service that continuously tracks portfolio risk
    and triggers alerts when limits are breached.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        monitoring_interval: int = 60,  # seconds
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize risk monitoring service.
        
        Args:
            risk_manager: Risk manager instance
            monitoring_interval: Monitoring frequency in seconds
            alert_callbacks: List of callback functions for alerts
        """
        self.risk_manager = risk_manager
        self.monitoring_interval = monitoring_interval
        self.alert_callbacks = alert_callbacks or []
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Risk metrics history
        self.risk_metrics_history: List[RiskMetrics] = []
        self.alert_history: List[RiskAlert] = []
        
        # Performance tracking
        self.monitoring_stats = {
            'total_checks': 0,
            'alerts_generated': 0,
            'last_check_time': None,
            'average_check_duration': 0.0
        }
    
    async def start_monitoring(
        self,
        portfolio_provider: Callable[[], Portfolio],
        market_data_provider: Callable[[], pd.DataFrame]
    ) -> None:
        """
        Start continuous risk monitoring.
        
        Args:
            portfolio_provider: Function that returns current portfolio
            market_data_provider: Function that returns current market data
        """
        if self.is_monitoring:
            self.logger.warning("Risk monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(portfolio_provider, market_data_provider)
        )
        
        self.logger.info(
            f"Started risk monitoring with {self.monitoring_interval}s interval"
        )
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped risk monitoring")
    
    async def _monitoring_loop(
        self,
        portfolio_provider: Callable[[], Portfolio],
        market_data_provider: Callable[[], pd.DataFrame]
    ) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                start_time = datetime.now()
                
                # Get current data
                portfolio = portfolio_provider()
                market_data = market_data_provider()
                
                # Calculate risk metrics
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    portfolio, market_data
                )
                
                # Check risk limits
                alerts = self.risk_manager.check_risk_limits(portfolio, risk_metrics)
                
                # Store metrics and alerts
                self.risk_metrics_history.append(risk_metrics)
                self.alert_history.extend(alerts)
                
                # Trigger alert callbacks
                for alert in alerts:
                    await self._trigger_alert_callbacks(alert, portfolio, risk_metrics)
                
                # Update monitoring statistics
                check_duration = (datetime.now() - start_time).total_seconds()
                self._update_monitoring_stats(check_duration)
                
                # Log monitoring status
                if alerts:
                    self.logger.warning(
                        f"Risk monitoring check completed: {len(alerts)} alerts generated"
                    )
                else:
                    self.logger.debug("Risk monitoring check completed: no alerts")
                
                # Wait for next check
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _trigger_alert_callbacks(
        self,
        alert: RiskAlert,
        portfolio: Portfolio,
        risk_metrics: RiskMetrics
    ) -> None:
        """Trigger all registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert, portfolio, risk_metrics)
                else:
                    callback(alert, portfolio, risk_metrics)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
        self.logger.info("Added alert callback")
    
    def remove_alert_callback(self, callback: Callable) -> None:
        """Remove an alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info("Removed alert callback")
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk monitoring status."""
        if not self.risk_metrics_history:
            return {"status": "no_data", "message": "No risk metrics available"}
        
        latest_metrics = self.risk_metrics_history[-1]
        active_alerts = self.risk_manager.get_active_alerts()
        
        # Categorize alerts by severity
        alert_summary = defaultdict(int)
        for alert in active_alerts:
            alert_summary[alert.status.value] += 1
        
        return {
            "status": "monitoring" if self.is_monitoring else "stopped",
            "latest_metrics": latest_metrics.model_dump(),
            "active_alerts_count": len(active_alerts),
            "alert_summary": dict(alert_summary),
            "monitoring_stats": self.monitoring_stats,
            "last_update": latest_metrics.timestamp.isoformat()
        }
    
    def get_risk_metrics_history(
        self,
        hours: int = 24
    ) -> List[RiskMetrics]:
        """Get risk metrics history for specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.risk_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_alert_history(
        self,
        hours: int = 24,
        status_filter: Optional[RiskLimitStatus] = None
    ) -> List[RiskAlert]:
        """Get alert history for specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        if status_filter:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.status == status_filter
            ]
        
        return filtered_alerts
    
    def generate_risk_report(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        include_stress_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio: Current portfolio
            market_data: Market data for calculations
            include_stress_tests: Whether to include stress test results
        
        Returns:
            Comprehensive risk report
        """
        try:
            # Current risk metrics
            current_metrics = self.risk_manager.calculate_risk_metrics(
                portfolio, market_data
            )
            
            # Active alerts
            active_alerts = self.risk_manager.get_active_alerts()
            
            # Historical performance
            metrics_24h = self.get_risk_metrics_history(24)
            pnl_trend = [m.daily_pnl for m in metrics_24h[-10:]]  # Last 10 checks
            
            # Risk limit status
            risk_limit_status = {}
            for limit_key, limit in self.risk_manager.risk_limits.items():
                current_value = self.risk_manager._get_limit_current_value(
                    limit, portfolio, current_metrics
                )
                if current_value is not None:
                    utilization = current_value / limit.threshold
                    risk_limit_status[limit_key] = {
                        "current_value": current_value,
                        "threshold": limit.threshold,
                        "utilization": utilization,
                        "status": "normal" if utilization < 0.8 else "warning" if utilization < 1.0 else "breach"
                    }
            
            report = {
                "report_timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio_summary": {
                    "total_value": portfolio.total_value,
                    "cash_balance": portfolio.cash_balance,
                    "positions_count": len(portfolio.positions),
                    "unrealized_pnl": portfolio.unrealized_pnl,
                    "realized_pnl": portfolio.realized_pnl
                },
                "current_risk_metrics": current_metrics.model_dump(),
                "active_alerts": [alert.model_dump() for alert in active_alerts],
                "risk_limit_status": risk_limit_status,
                "performance_trends": {
                    "pnl_trend": pnl_trend,
                    "avg_daily_pnl": sum(pnl_trend) / len(pnl_trend) if pnl_trend else 0,
                    "pnl_volatility": pd.Series(pnl_trend).std() if len(pnl_trend) > 1 else 0
                },
                "monitoring_status": {
                    "is_active": self.is_monitoring,
                    "total_checks": self.monitoring_stats['total_checks'],
                    "alerts_generated": self.monitoring_stats['alerts_generated']
                }
            }
            
            # Add stress test results if requested
            if include_stress_tests:
                stress_results = []
                scenarios = self.risk_manager.get_predefined_stress_scenarios()
                
                for scenario in scenarios[:3]:  # Run top 3 scenarios
                    try:
                        result = self.risk_manager.run_stress_test(
                            portfolio, scenario, market_data
                        )
                        stress_results.append(result.model_dump())
                    except Exception as e:
                        self.logger.error(f"Stress test failed for {scenario.scenario_name}: {e}")
                
                report["stress_test_results"] = stress_results
            
            return report
            
        except Exception as e:
            self.logger.error(f"Risk report generation failed: {e}")
            return {
                "error": str(e),
                "report_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def export_risk_data(
        self,
        filepath: str,
        hours: int = 24,
        format: str = "json"
    ) -> bool:
        """
        Export risk monitoring data to file.
        
        Args:
            filepath: Output file path
            hours: Hours of data to export
            format: Export format ('json' or 'csv')
        
        Returns:
            True if export successful, False otherwise
        """
        try:
            metrics_data = self.get_risk_metrics_history(hours)
            alerts_data = self.get_alert_history(hours)
            
            if format.lower() == "json":
                export_data = {
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "risk_metrics": [m.model_dump() for m in metrics_data],
                    "alerts": [a.model_dump() for a in alerts_data],
                    "monitoring_stats": self.monitoring_stats
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == "csv":
                # Export metrics to CSV
                if metrics_data:
                    metrics_df = pd.DataFrame([m.model_dump() for m in metrics_data])
                    metrics_df.to_csv(filepath.replace('.csv', '_metrics.csv'), index=False)
                
                # Export alerts to CSV
                if alerts_data:
                    alerts_df = pd.DataFrame([a.model_dump() for a in alerts_data])
                    alerts_df.to_csv(filepath.replace('.csv', '_alerts.csv'), index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Risk data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Risk data export failed: {e}")
            return False
    
    def _update_monitoring_stats(self, check_duration: float) -> None:
        """Update monitoring statistics."""
        self.monitoring_stats['total_checks'] += 1
        self.monitoring_stats['last_check_time'] = datetime.now(timezone.utc)
        
        # Update average check duration
        current_avg = self.monitoring_stats['average_check_duration']
        total_checks = self.monitoring_stats['total_checks']
        
        new_avg = ((current_avg * (total_checks - 1)) + check_duration) / total_checks
        self.monitoring_stats['average_check_duration'] = new_avg
    
    def cleanup_old_data(self, days: int = 7) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Clean metrics history
        self.risk_metrics_history = [
            metrics for metrics in self.risk_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        # Clean alert history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        self.logger.info(f"Cleaned up risk monitoring data older than {days} days")


# Alert callback implementations

async def email_alert_callback(
    alert: RiskAlert,
    portfolio: Portfolio,
    risk_metrics: RiskMetrics
) -> None:
    """Send email alert (placeholder implementation)."""
    # This would integrate with an email service
    print(f"EMAIL ALERT: {alert.message}")


async def slack_alert_callback(
    alert: RiskAlert,
    portfolio: Portfolio,
    risk_metrics: RiskMetrics
) -> None:
    """Send Slack alert (placeholder implementation)."""
    # This would integrate with Slack API
    print(f"SLACK ALERT: {alert.message}")


def log_alert_callback(
    alert: RiskAlert,
    portfolio: Portfolio,
    risk_metrics: RiskMetrics
) -> None:
    """Log alert to file."""
    logger = logging.getLogger("risk_alerts")
    logger.warning(
        f"RISK ALERT: {alert.limit_type.value} - {alert.message} "
        f"(Portfolio: ${portfolio.total_value:,.2f})"
    )


async def emergency_stop_callback(
    alert: RiskAlert,
    portfolio: Portfolio,
    risk_metrics: RiskMetrics
) -> None:
    """Emergency stop trading on critical alerts."""
    if alert.status == RiskLimitStatus.BREACH:
        # This would integrate with trading system to halt trading
        print(f"EMERGENCY STOP: Trading halted due to {alert.limit_type.value} breach")