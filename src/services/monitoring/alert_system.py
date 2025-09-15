"""
Alert system using Observer pattern for better extensibility.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.models.monitoring import Alert, AlertSeverity
from src.utils.logging import get_logger

logger = get_logger("alert_system")


class AlertObserver(ABC):
    """Abstract observer for alert notifications."""
    
    @abstractmethod
    async def notify(self, alert: Alert) -> None:
        """Handle alert notification."""
        pass


class AlertSubject:
    """Subject that manages alert observers."""
    
    def __init__(self):
        self._observers: List[AlertObserver] = []
        self.alert_history: List[Alert] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=30)
    
    def attach(self, observer: AlertObserver) -> None:
        """Attach an observer."""
        self._observers.append(observer)
        logger.info(f"Alert observer attached: {observer.__class__.__name__}")
    
    def detach(self, observer: AlertObserver) -> None:
        """Detach an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
            logger.info(f"Alert observer detached: {observer.__class__.__name__}")
    
    async def notify_observers(self, alert: Alert) -> None:
        """Notify all observers about an alert."""
        
        # Check cooldown to prevent spam
        if self._is_in_cooldown(alert):
            return
        
        self._update_cooldown(alert)
        self.alert_history.append(alert)
        
        # Notify all observers
        for observer in self._observers:
            try:
                await observer.notify(alert)
            except Exception as e:
                logger.error(f"Error notifying observer {observer.__class__.__name__}: {e}")
        
        logger.info(f"Alert sent: {alert.title} (Severity: {alert.severity.value})")
    
    def _is_in_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period."""
        cooldown_key = f"{alert.model_name}_{alert.metric_name}_{alert.severity.value}"
        now = datetime.now()
        
        if cooldown_key in self.alert_cooldown:
            return now - self.alert_cooldown[cooldown_key] < self.cooldown_period
        
        return False
    
    def _update_cooldown(self, alert: Alert) -> None:
        """Update cooldown timestamp for alert."""
        cooldown_key = f"{alert.model_name}_{alert.metric_name}_{alert.severity.value}"
        self.alert_cooldown[cooldown_key] = datetime.now()


# Concrete Observer Implementations
class EmailAlertObserver(AlertObserver):
    """Email alert notification observer."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    async def notify(self, alert: Alert) -> None:
        """Send alert via email."""
        # Implementation would use aiosmtplib for async email sending
        logger.info(f"EMAIL ALERT: {alert.title} - {alert.message}")


class SlackAlertObserver(AlertObserver):
    """Slack alert notification observer."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def notify(self, alert: Alert) -> None:
        """Send alert via Slack webhook."""
        # Implementation would use aiohttp for async webhook calls
        logger.info(f"SLACK ALERT: {alert.title} - {alert.message}")


class DatabaseAlertObserver(AlertObserver):
    """Database alert storage observer."""
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    async def notify(self, alert: Alert) -> None:
        """Store alert in database."""
        # Implementation would store alert in database
        logger.info(f"DB ALERT STORED: {alert.id}")


class MetricsAlertObserver(AlertObserver):
    """Metrics collection observer."""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
    
    async def notify(self, alert: Alert) -> None:
        """Record alert metrics."""
        self.metrics_collector.increment_counter(
            "alerts_sent_total",
            tags={
                "severity": alert.severity.value,
                "model": alert.model_name or "system"
            }
        )


# Alert Factory for different severity levels
class AlertFactory:
    """Factory for creating different types of alerts."""
    
    @staticmethod
    def create_performance_alert(
        model_name: str,
        metric_name: str,
        current_value: float,
        threshold: float
    ) -> Alert:
        """Create a performance threshold alert."""
        
        severity = AlertSeverity.HIGH if current_value < threshold * 0.8 else AlertSeverity.MEDIUM
        
        return Alert(
            id=f"performance_threshold_{model_name}_{metric_name}_{datetime.now().timestamp()}",
            severity=severity,
            title=f"Performance Threshold Breach: {metric_name.title()}",
            message=f"Model {model_name} {metric_name} ({current_value:.3f}) is below threshold ({threshold})",
            timestamp=datetime.now(),
            model_name=model_name,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )
    
    @staticmethod
    def create_drift_alert(model_name: str, drift_result) -> Alert:
        """Create a drift detection alert."""
        
        return Alert(
            id=f"drift_detection_{model_name}_{drift_result.drift_type.value}_{datetime.now().timestamp()}",
            severity=drift_result.severity,
            title=f"Model Drift Detected: {drift_result.drift_type.value.replace('_', ' ').title()}",
            message=f"Model {model_name} shows {drift_result.drift_type.value} "
                   f"(score: {drift_result.drift_score:.4f}, threshold: {drift_result.threshold})",
            timestamp=drift_result.timestamp,
            model_name=model_name,
            metadata=drift_result.details
        )
    
    @staticmethod
    def create_system_error_alert(model_name: str, error: Exception) -> Alert:
        """Create a system error alert."""
        
        return Alert(
            id=f"monitoring_error_{model_name}_{datetime.now().timestamp()}",
            severity=AlertSeverity.HIGH,
            title="Monitoring System Error",
            message=f"Error in monitoring cycle for model {model_name}: {str(error)}",
            timestamp=datetime.now(),
            model_name=model_name
        )