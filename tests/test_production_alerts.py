"""
Tests for Production Alert Service

Tests alert reliability, PagerDuty integration, escalation procedures,
and multi-channel notification delivery.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.services.alert_service import (
    ProductionAlertService, Alert, AlertSeverity, AlertChannel, AlertRule,
    PagerDutyConfig, SlackConfig, EmailConfig
)
from src.services.risk_management_service import RiskAlert, RiskLevel


@pytest.fixture
def pagerduty_config():
    return PagerDutyConfig(
        integration_key="test_integration_key",
        api_url="https://events.pagerduty.com/v2/enqueue"
    )


@pytest.fixture
def slack_config():
    return SlackConfig(
        webhook_url="https://hooks.slack.com/test",
        channel="#alerts",
        username="Trading Platform"
    )


@pytest.fixture
def email_config():
    return EmailConfig(
        smtp_host="smtp.test.com",
        smtp_port=587,
        username="alerts@test.com",
        password="test_password",
        from_address="alerts@test.com"
    )


@pytest.fixture
def alert_service(pagerduty_config, slack_config, email_config):
    return ProductionAlertService(
        pagerduty_config=pagerduty_config,
        slack_config=slack_config,
        email_config=email_config
    )


@pytest.fixture
def sample_alert():
    return Alert(
        alert_id="alert_123",
        title="Test Alert",
        description="This is a test alert",
        severity=AlertSeverity.WARNING,
        source="test",
        timestamp=datetime.utcnow(),
        metadata={"test_key": "test_value"}
    )


@pytest.fixture
def sample_risk_alert():
    return RiskAlert(
        alert_id="risk_123",
        customer_id="customer_1",
        risk_level=RiskLevel.HIGH,
        alert_type="position_size_violation",
        message="Position size exceeds limit",
        timestamp=datetime.utcnow(),
        position_id="pos_123",
        current_value=25000.0,
        threshold_value=20000.0
    )


class TestAlertGeneration:
    """Test alert generation and basic functionality."""
    
    @pytest.mark.asyncio
    async def test_send_alert_basic(self, alert_service):
        """Test basic alert sending."""
        alert_id = await alert_service.send_alert(
            title="Test Alert",
            description="Test description",
            severity=AlertSeverity.WARNING,
            source="test"
        )
        
        assert alert_id is not None
        assert alert_id in alert_service.active_alerts
        
        alert = alert_service.active_alerts[alert_id]
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "test"
    
    @pytest.mark.asyncio
    async def test_send_critical_alert(self, alert_service):
        """Test critical alert sending."""
        alert_id = await alert_service.send_critical_alert(
            "Critical System Failure",
            "Database connection lost"
        )
        
        alert = alert_service.active_alerts[alert_id]
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.source == "system"
    
    @pytest.mark.asyncio
    async def test_send_warning_alert(self, alert_service):
        """Test warning alert sending."""
        alert_id = await alert_service.send_warning_alert(
            "Performance Degradation",
            "Response time increased"
        )
        
        alert = alert_service.active_alerts[alert_id]
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "system"
    
    @pytest.mark.asyncio
    async def test_send_risk_alert(self, alert_service, sample_risk_alert):
        """Test risk alert sending."""
        alert_id = await alert_service.send_risk_alert(sample_risk_alert)
        
        alert = alert_service.active_alerts[alert_id]
        assert alert.severity == AlertSeverity.ERROR  # HIGH risk level maps to ERROR
        assert alert.source == "risk_management"
        assert "customer_1" in alert.metadata["customer_id"]


class TestAlertRules:
    """Test alert rule matching and processing."""
    
    def test_alert_rule_matching(self, alert_service, sample_alert):
        """Test alert rule matching logic."""
        # Test critical system failure rule
        sample_alert.severity = AlertSeverity.CRITICAL
        matching_rules = alert_service._find_matching_rules(sample_alert)
        
        # Should match the critical_system_failure rule
        rule_names = [rule.name for rule in matching_rules]
        assert "critical_system_failure" in rule_names
    
    def test_condition_evaluation(self, alert_service):
        """Test alert rule condition evaluation."""
        alert = Alert(
            alert_id="test",
            title="Test",
            description="Test",
            severity=AlertSeverity.CRITICAL,
            source="trading",
            timestamp=datetime.utcnow()
        )
        
        # Test simple condition
        assert alert_service._evaluate_condition("severity == 'critical'", alert)
        assert not alert_service._evaluate_condition("severity == 'warning'", alert)
        
        # Test complex condition
        assert alert_service._evaluate_condition("source == 'trading' and severity in ['error', 'critical']", alert)
    
    def test_custom_alert_rule(self, alert_service):
        """Test adding custom alert rules."""
        custom_rule = AlertRule(
            name="custom_test_rule",
            condition="source == 'custom'",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.SLACK],
            escalation_delay=5,
            max_escalations=1
        )
        
        alert_service.add_alert_rule(custom_rule)
        assert custom_rule in alert_service.alert_rules


class TestRateLimiting:
    """Test alert rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_basic(self, alert_service):
        """Test basic rate limiting."""
        # Send alerts rapidly
        alert_ids = []
        for i in range(15):  # Exceed rate limit of 10
            alert_id = await alert_service.send_alert(
                f"Test Alert {i}",
                "Test description",
                AlertSeverity.WARNING,
                "test"
            )
            alert_ids.append(alert_id)
        
        # Some alerts should be rate limited
        active_count = len([aid for aid in alert_ids if aid in alert_service.active_alerts])
        assert active_count <= 10  # Should not exceed rate limit
    
    @pytest.mark.asyncio
    async def test_rate_limiting_by_source_and_severity(self, alert_service):
        """Test rate limiting is applied per source and severity."""
        # Send warning alerts from test source
        for i in range(12):
            await alert_service.send_alert(
                f"Warning {i}",
                "Test",
                AlertSeverity.WARNING,
                "test"
            )
        
        # Send error alerts from test source (different bucket)
        for i in range(5):
            await alert_service.send_alert(
                f"Error {i}",
                "Test",
                AlertSeverity.ERROR,
                "test"
            )
        
        # Error alerts should not be rate limited
        error_alerts = [
            alert for alert in alert_service.active_alerts.values()
            if alert.severity == AlertSeverity.ERROR
        ]
        assert len(error_alerts) == 5


class TestPagerDutyIntegration:
    """Test PagerDuty integration."""
    
    @pytest.mark.asyncio
    async def test_pagerduty_alert_sending(self, alert_service, sample_alert):
        """Test sending alert to PagerDuty."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 202
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await alert_service._send_pagerduty_alert(sample_alert)
            
            # Should make POST request to PagerDuty
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL
            assert call_args[0][0] == alert_service.pagerduty_config.api_url
            
            # Check payload structure
            payload = call_args[1]['json']
            assert payload['routing_key'] == alert_service.pagerduty_config.integration_key
            assert payload['event_action'] == 'trigger'
            assert payload['dedup_key'] == sample_alert.alert_id
    
    @pytest.mark.asyncio
    async def test_pagerduty_alert_failure(self, alert_service, sample_alert):
        """Test PagerDuty alert failure handling."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 500  # Server error
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Should handle failure gracefully
            await alert_service._send_pagerduty_alert(sample_alert)
            
            # Should not raise exception
            assert True
    
    @pytest.mark.asyncio
    async def test_pagerduty_resolution(self, alert_service, sample_alert):
        """Test sending alert resolution to PagerDuty."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 202
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await alert_service._send_pagerduty_resolution(sample_alert, "Issue resolved")
            
            mock_post.assert_called_once()
            payload = mock_post.call_args[1]['json']
            assert payload['event_action'] == 'resolve'
            assert payload['dedup_key'] == sample_alert.alert_id


class TestSlackIntegration:
    """Test Slack integration."""
    
    @pytest.mark.asyncio
    async def test_slack_alert_sending(self, alert_service, sample_alert):
        """Test sending alert to Slack."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await alert_service._send_slack_alert(sample_alert)
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL
            assert call_args[0][0] == alert_service.slack_config.webhook_url
            
            # Check payload structure
            payload = call_args[1]['json']
            assert payload['channel'] == alert_service.slack_config.channel
            assert payload['username'] == alert_service.slack_config.username
            assert len(payload['attachments']) == 1
            
            attachment = payload['attachments'][0]
            assert attachment['title'] == sample_alert.title
            assert attachment['text'] == sample_alert.description
    
    @pytest.mark.asyncio
    async def test_slack_alert_colors(self, alert_service):
        """Test Slack alert color mapping."""
        test_cases = [
            (AlertSeverity.INFO, "good"),
            (AlertSeverity.WARNING, "warning"),
            (AlertSeverity.ERROR, "danger"),
            (AlertSeverity.CRITICAL, "danger")
        ]
        
        for severity, expected_color in test_cases:
            alert = Alert(
                alert_id="test",
                title="Test",
                description="Test",
                severity=severity,
                source="test",
                timestamp=datetime.utcnow()
            )
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = Mock()
                mock_response.status = 200
                mock_post.return_value.__aenter__.return_value = mock_response
                
                await alert_service._send_slack_alert(alert)
                
                payload = mock_post.call_args[1]['json']
                attachment = payload['attachments'][0]
                assert attachment['color'] == expected_color


class TestEscalation:
    """Test alert escalation procedures."""
    
    @pytest.mark.asyncio
    async def test_escalation_setup(self, alert_service, sample_alert):
        """Test escalation timer setup."""
        rule = AlertRule(
            name="test_escalation",
            condition="severity == 'warning'",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK],
            escalation_delay=1,  # 1 minute
            max_escalations=1
        )
        
        await alert_service._setup_escalation(sample_alert, rule)
        
        # Should create escalation timer
        assert sample_alert.alert_id in alert_service.escalation_timers
        
        # Cleanup
        alert_service.escalation_timers[sample_alert.alert_id].cancel()
    
    @pytest.mark.asyncio
    async def test_escalation_execution(self, alert_service, sample_alert):
        """Test escalation execution."""
        rule = AlertRule(
            name="test_escalation",
            condition="severity == 'warning'",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK],
            escalation_delay=0.01,  # Very short delay for testing
            max_escalations=1
        )
        
        # Add alert to active alerts
        alert_service.active_alerts[sample_alert.alert_id] = sample_alert
        
        with patch.object(alert_service, '_send_alert_via_channels') as mock_send:
            await alert_service._setup_escalation(sample_alert, rule)
            
            # Wait for escalation
            await asyncio.sleep(0.02)
            
            # Should send escalated alert
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_escalation_cancellation_on_resolution(self, alert_service, sample_alert):
        """Test escalation cancellation when alert is resolved."""
        rule = AlertRule(
            name="test_escalation",
            condition="severity == 'warning'",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK],
            escalation_delay=1,
            max_escalations=1
        )
        
        # Add alert and setup escalation
        alert_service.active_alerts[sample_alert.alert_id] = sample_alert
        await alert_service._setup_escalation(sample_alert, rule)
        
        # Resolve alert
        await alert_service.resolve_alert(sample_alert.alert_id)
        
        # Escalation timer should be cancelled
        assert sample_alert.alert_id not in alert_service.escalation_timers


class TestAlertResolution:
    """Test alert resolution functionality."""
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self, alert_service, sample_alert):
        """Test basic alert resolution."""
        # Add alert to active alerts
        alert_service.active_alerts[sample_alert.alert_id] = sample_alert
        
        await alert_service.resolve_alert(sample_alert.alert_id, "Issue fixed")
        
        # Alert should be resolved and removed from active alerts
        assert sample_alert.resolved
        assert sample_alert.resolved_at is not None
        assert sample_alert.alert_id not in alert_service.active_alerts
    
    @pytest.mark.asyncio
    async def test_resolution_with_pagerduty(self, alert_service, sample_alert):
        """Test alert resolution with PagerDuty notification."""
        alert_service.active_alerts[sample_alert.alert_id] = sample_alert
        
        with patch.object(alert_service, '_send_pagerduty_resolution') as mock_pd_resolution:
            await alert_service.resolve_alert(sample_alert.alert_id, "Fixed by team")
            
            mock_pd_resolution.assert_called_once_with(sample_alert, "Fixed by team")


class TestAlertStatistics:
    """Test alert statistics and reporting."""
    
    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_service):
        """Test getting active alerts."""
        # Send some alerts
        await alert_service.send_alert("Alert 1", "Description 1", AlertSeverity.WARNING, "test")
        await alert_service.send_alert("Alert 2", "Description 2", AlertSeverity.ERROR, "test")
        
        active_alerts = await alert_service.get_active_alerts()
        assert len(active_alerts) >= 2
    
    @pytest.mark.asyncio
    async def test_get_alert_history(self, alert_service):
        """Test getting alert history."""
        # Send and resolve an alert
        alert_id = await alert_service.send_alert("Historical Alert", "Description", AlertSeverity.INFO, "test")
        await alert_service.resolve_alert(alert_id)
        
        history = await alert_service.get_alert_history(hours=1)
        assert len(history) >= 1
        
        # Check that resolved alert is in history
        resolved_alerts = [a for a in history if a.resolved]
        assert len(resolved_alerts) >= 1
    
    @pytest.mark.asyncio
    async def test_get_alert_statistics(self, alert_service):
        """Test alert statistics calculation."""
        # Send various alerts
        await alert_service.send_alert("Warning 1", "Desc", AlertSeverity.WARNING, "source1")
        await alert_service.send_alert("Critical 1", "Desc", AlertSeverity.CRITICAL, "source2")
        await alert_service.send_alert("Error 1", "Desc", AlertSeverity.ERROR, "source1")
        
        stats = await alert_service.get_alert_statistics()
        
        assert "active_alerts" in stats
        assert "alerts_last_24h" in stats
        assert "critical_alerts_24h" in stats
        assert "top_alert_sources" in stats
        assert stats["active_alerts"] >= 3
        assert stats["alerts_last_24h"] >= 3
        assert stats["critical_alerts_24h"] >= 1
    
    @pytest.mark.asyncio
    async def test_average_resolution_time(self, alert_service):
        """Test average resolution time calculation."""
        # Send and quickly resolve an alert
        alert_id = await alert_service.send_alert("Quick Alert", "Desc", AlertSeverity.INFO, "test")
        await asyncio.sleep(0.01)  # Small delay
        await alert_service.resolve_alert(alert_id)
        
        stats = await alert_service.get_alert_statistics()
        assert stats["average_resolution_time_minutes"] >= 0


class TestMultiChannelDelivery:
    """Test multi-channel alert delivery."""
    
    @pytest.mark.asyncio
    async def test_multi_channel_delivery(self, alert_service):
        """Test alert delivery through multiple channels."""
        channels = [AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.EMAIL]
        
        with patch.object(alert_service, '_send_pagerduty_alert') as mock_pd:
            with patch.object(alert_service, '_send_slack_alert') as mock_slack:
                with patch.object(alert_service, '_send_email_alert') as mock_email:
                    
                    alert = Alert(
                        alert_id="multi_test",
                        title="Multi-channel Test",
                        description="Test",
                        severity=AlertSeverity.CRITICAL,
                        source="test",
                        timestamp=datetime.utcnow()
                    )
                    
                    await alert_service._send_alert_via_channels(alert, channels)
                    
                    # All channels should be called
                    mock_pd.assert_called_once()
                    mock_slack.assert_called_once()
                    mock_email.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_channel_failure_isolation(self, alert_service):
        """Test that failure in one channel doesn't affect others."""
        channels = [AlertChannel.PAGERDUTY, AlertChannel.SLACK]
        
        with patch.object(alert_service, '_send_pagerduty_alert', side_effect=Exception("PD Error")):
            with patch.object(alert_service, '_send_slack_alert') as mock_slack:
                
                alert = Alert(
                    alert_id="failure_test",
                    title="Failure Test",
                    description="Test",
                    severity=AlertSeverity.ERROR,
                    source="test",
                    timestamp=datetime.utcnow()
                )
                
                # Should not raise exception despite PagerDuty failure
                await alert_service._send_alert_via_channels(alert, channels)
                
                # Slack should still be called
                mock_slack.assert_called_once()


class TestAlertServiceConfiguration:
    """Test alert service configuration and customization."""
    
    def test_alert_service_without_configs(self):
        """Test alert service initialization without external configs."""
        service = ProductionAlertService()
        
        assert service.pagerduty_config is None
        assert service.slack_config is None
        assert service.email_config is None
        assert len(service.alert_rules) > 0  # Should have default rules
    
    def test_custom_rate_limiting(self, alert_service):
        """Test custom rate limiting configuration."""
        # Modify rate limiting settings
        alert_service.max_alerts_per_window = 5
        alert_service.rate_limit_window = timedelta(minutes=1)
        
        # Test that new limits are applied
        assert alert_service.max_alerts_per_window == 5
        assert alert_service.rate_limit_window == timedelta(minutes=1)
    
    def test_alert_rule_management(self, alert_service):
        """Test adding and removing alert rules."""
        initial_count = len(alert_service.alert_rules)
        
        # Add rule
        new_rule = AlertRule(
            name="test_rule",
            condition="source == 'test'",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.SLACK],
            escalation_delay=10,
            max_escalations=1
        )
        alert_service.add_alert_rule(new_rule)
        assert len(alert_service.alert_rules) == initial_count + 1
        
        # Remove rule
        alert_service.remove_alert_rule("test_rule")
        assert len(alert_service.alert_rules) == initial_count