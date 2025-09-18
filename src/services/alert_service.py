"""
Production Alert Service

Provides comprehensive alerting with PagerDuty integration, escalation procedures,
and multi-channel notification support for production incidents.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from pydantic import BaseModel

from src.models.risk_models import RiskAlert


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"


@dataclass
class AlertRule:
    name: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    escalation_delay: int  # minutes
    max_escalations: int


class Alert(BaseModel):
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class PagerDutyConfig(BaseModel):
    integration_key: str
    api_url: str = "https://events.pagerduty.com/v2/enqueue"
    timeout: int = 30


class SlackConfig(BaseModel):
    webhook_url: str
    channel: str
    username: str = "Trading Platform"
    timeout: int = 30


class EmailConfig(BaseModel):
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    timeout: int = 30


class ProductionAlertService:
    """Production-grade alert service with multiple channels and escalation."""
    
    def __init__(
        self,
        pagerduty_config: Optional[PagerDutyConfig] = None,
        slack_config: Optional[SlackConfig] = None,
        email_config: Optional[EmailConfig] = None
    ):
        self.pagerduty_config = pagerduty_config
        self.slack_config = slack_config
        self.email_config = email_config
        self.logger = logging.getLogger(__name__)
        
        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.escalation_timers: Dict[str, asyncio.Task] = {}
        
        # Alert rules
        self.alert_rules: List[AlertRule] = []
        self._setup_default_rules()
        
        # Rate limiting
        self.alert_counts: Dict[str, List[datetime]] = {}
        self.rate_limit_window = timedelta(minutes=5)
        self.max_alerts_per_window = 10
        
    def _setup_default_rules(self):
        """Setup default alert rules."""
        self.alert_rules = [
            AlertRule(
                name="critical_system_failure",
                condition="severity == 'critical'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.EMAIL],
                escalation_delay=5,
                max_escalations=3
            ),
            AlertRule(
                name="trading_system_error",
                condition="source == 'trading' and severity in ['error', 'critical']",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK],
                escalation_delay=10,
                max_escalations=2
            ),
            AlertRule(
                name="risk_violation",
                condition="source == 'risk_management'",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                escalation_delay=15,
                max_escalations=1
            ),
            AlertRule(
                name="performance_degradation",
                condition="source == 'monitoring' and severity == 'warning'",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
                escalation_delay=30,
                max_escalations=1
            )
        ]
    
    async def send_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send an alert through appropriate channels."""
        alert_id = f"alert_{datetime.utcnow().timestamp()}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Check rate limiting
        if await self._is_rate_limited(alert):
            self.logger.warning(f"Alert rate limited: {title}")
            return alert_id
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Find matching rules and send alerts
        matching_rules = self._find_matching_rules(alert)
        
        for rule in matching_rules:
            await self._send_alert_via_channels(alert, rule.channels)
            
            # Setup escalation if needed
            if rule.max_escalations > 0:
                await self._setup_escalation(alert, rule)
        
        self.logger.info(f"Alert sent: {alert_id} - {title}")
        return alert_id
    
    async def send_critical_alert(self, title: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send a critical alert."""
        return await self.send_alert(title, description, AlertSeverity.CRITICAL, "system", metadata)
    
    async def send_error_alert(self, title: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send an error alert."""
        return await self.send_alert(title, description, AlertSeverity.ERROR, "system", metadata)
    
    async def send_warning_alert(self, title: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send a warning alert."""
        return await self.send_alert(title, description, AlertSeverity.WARNING, "system", metadata)
    
    async def send_risk_alert(self, risk_alert: RiskAlert) -> str:
        """Send a risk management alert."""
        metadata = {
            "customer_id": risk_alert.customer_id,
            "position_id": risk_alert.position_id,
            "current_value": str(risk_alert.current_value) if risk_alert.current_value else None,
            "threshold_value": str(risk_alert.threshold_value) if risk_alert.threshold_value else None
        }
        
        severity_mapping = {
            "low": AlertSeverity.INFO,
            "medium": AlertSeverity.WARNING,
            "high": AlertSeverity.ERROR,
            "critical": AlertSeverity.CRITICAL
        }
        
        severity = severity_mapping.get(risk_alert.risk_level.value, AlertSeverity.WARNING)
        
        return await self.send_alert(
            f"Risk Alert: {risk_alert.alert_type}",
            risk_alert.message,
            severity,
            "risk_management",
            metadata
        )
    
    async def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        now = datetime.utcnow()
        alert_key = f"{alert.source}_{alert.severity.value}"
        
        # Initialize if not exists
        if alert_key not in self.alert_counts:
            self.alert_counts[alert_key] = []
        
        # Remove old entries
        cutoff_time = now - self.rate_limit_window
        self.alert_counts[alert_key] = [
            timestamp for timestamp in self.alert_counts[alert_key]
            if timestamp > cutoff_time
        ]
        
        # Check rate limit
        if len(self.alert_counts[alert_key]) >= self.max_alerts_per_window:
            return True
        
        # Add current alert
        self.alert_counts[alert_key].append(now)
        return False
    
    def _find_matching_rules(self, alert: Alert) -> List[AlertRule]:
        """Find alert rules that match the given alert."""
        matching_rules = []
        
        for rule in self.alert_rules:
            try:
                # Simple condition evaluation (in production, use a proper expression evaluator)
                if self._evaluate_condition(rule.condition, alert):
                    matching_rules.append(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating rule condition {rule.name}: {e}")
        
        return matching_rules
    
    def _evaluate_condition(self, condition: str, alert: Alert) -> bool:
        """Evaluate alert rule condition (simplified implementation)."""
        # This is a simplified implementation. In production, use a proper expression evaluator
        context = {
            "severity": alert.severity.value,
            "source": alert.source,
            "title": alert.title,
            "description": alert.description
        }
        
        try:
            # Replace variables in condition
            for key, value in context.items():
                condition = condition.replace(key, f"'{value}'")
            
            # Evaluate condition (be careful with eval in production!)
            return eval(condition)
        except:
            return False
    
    async def _send_alert_via_channels(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert through specified channels."""
        tasks = []
        
        for channel in channels:
            if channel == AlertChannel.PAGERDUTY and self.pagerduty_config:
                tasks.append(self._send_pagerduty_alert(alert))
            elif channel == AlertChannel.SLACK and self.slack_config:
                tasks.append(self._send_slack_alert(alert))
            elif channel == AlertChannel.EMAIL and self.email_config:
                tasks.append(self._send_email_alert(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty."""
        try:
            payload = {
                "routing_key": self.pagerduty_config.integration_key,
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": alert.title,
                    "source": alert.source,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": {
                        "description": alert.description,
                        "metadata": alert.metadata
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.pagerduty_config.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.pagerduty_config.timeout)
                ) as response:
                    if response.status == 202:
                        self.logger.info(f"PagerDuty alert sent: {alert.alert_id}")
                    else:
                        self.logger.error(f"PagerDuty alert failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack."""
        try:
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "channel": self.slack_config.channel,
                "username": self.slack_config.username,
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            }
                        ],
                        "footer": "Trading Platform Alert",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.slack_config.timeout)
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack alert sent: {alert.alert_id}")
                    else:
                        self.logger.error(f"Slack alert failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email."""
        try:
            # Email implementation would go here
            # For now, just log
            self.logger.info(f"Email alert would be sent: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    async def _setup_escalation(self, alert: Alert, rule: AlertRule):
        """Setup escalation timer for an alert."""
        async def escalate():
            try:
                await asyncio.sleep(rule.escalation_delay * 60)  # Convert to seconds
                
                if alert.alert_id in self.active_alerts and not alert.resolved:
                    escalated_alert = Alert(
                        alert_id=f"{alert.alert_id}_escalated",
                        title=f"ESCALATED: {alert.title}",
                        description=f"Alert not resolved after {rule.escalation_delay} minutes.\n\nOriginal: {alert.description}",
                        severity=AlertSeverity.CRITICAL,
                        source=alert.source,
                        timestamp=datetime.utcnow(),
                        metadata=alert.metadata
                    )
                    
                    # Send escalated alert
                    await self._send_alert_via_channels(escalated_alert, [AlertChannel.PAGERDUTY])
                    
                    self.logger.warning(f"Alert escalated: {alert.alert_id}")
                    
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(f"Escalation error for {alert.alert_id}: {e}")
        
        # Create escalation task
        task = asyncio.create_task(escalate())
        self.escalation_timers[alert.alert_id] = task
    
    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            # Cancel escalation timer
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]
            
            # Send resolution notification to PagerDuty
            if self.pagerduty_config:
                await self._send_pagerduty_resolution(alert, resolution_note)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id}")
    
    async def _send_pagerduty_resolution(self, alert: Alert, resolution_note: Optional[str]):
        """Send alert resolution to PagerDuty."""
        try:
            payload = {
                "routing_key": self.pagerduty_config.integration_key,
                "event_action": "resolve",
                "dedup_key": alert.alert_id
            }
            
            if resolution_note:
                payload["payload"] = {"summary": resolution_note}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.pagerduty_config.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.pagerduty_config.timeout)
                ) as response:
                    if response.status == 202:
                        self.logger.info(f"PagerDuty resolution sent: {alert.alert_id}")
                    else:
                        self.logger.error(f"PagerDuty resolution failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty resolution: {e}")
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    async def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        recent_alerts = [a for a in self.alert_history if a.timestamp > last_24h]
        weekly_alerts = [a for a in self.alert_history if a.timestamp > last_7d]
        
        return {
            "active_alerts": len(self.active_alerts),
            "alerts_last_24h": len(recent_alerts),
            "alerts_last_7d": len(weekly_alerts),
            "critical_alerts_24h": len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            "average_resolution_time_minutes": self._calculate_average_resolution_time(),
            "top_alert_sources": self._get_top_alert_sources(recent_alerts)
        }
    
    def _calculate_average_resolution_time(self) -> float:
        """Calculate average alert resolution time in minutes."""
        resolved_alerts = [a for a in self.alert_history if a.resolved and a.resolved_at]
        
        if not resolved_alerts:
            return 0.0
        
        total_time = sum(
            (alert.resolved_at - alert.timestamp).total_seconds()
            for alert in resolved_alerts
        )
        
        return total_time / len(resolved_alerts) / 60  # Convert to minutes
    
    def _get_top_alert_sources(self, alerts: List[Alert]) -> Dict[str, int]:
        """Get top alert sources by count."""
        source_counts = {}
        for alert in alerts:
            source_counts[alert.source] = source_counts.get(alert.source, 0) + 1
        
        return dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        self.logger.info(f"Removed alert rule: {rule_name}")