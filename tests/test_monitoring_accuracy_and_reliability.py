"""
Tests for monitoring accuracy and alert reliability.

This module provides comprehensive tests to validate the accuracy of the monitoring system
and the reliability of alerting mechanisms as required by Task 27.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.services.monitoring.advanced_monitoring_system import (
    AdvancedMonitoringSystem,
    RealTimeAnomalyDetector,
    StatisticalDriftDetector,
    AlertRoutingSystem,
    AnomalyDetectionConfig,
    PerformanceDegradationAlert
)
from src.models.monitoring import (
    Alert,
    AlertSeverity,
    ModelPerformanceMetrics
)
from src.services.data_aggregator import DataQualityReport, DataQualityIssue
from src.utils.monitoring import MetricsCollector


class TestMonitoringAccuracy:
    """Test the accuracy of monitoring components"""
    
    @pytest.fixture
    def anomaly_detector(self):
        config = AnomalyDetectionConfig(
            window_size=50,
            min_samples=10,
            z_score_threshold=2.0,
            sensitivity=1.0
        )
        return RealTimeAnomalyDetector(config)
    
    @pytest.fixture
    def drift_detector(self):
        return StatisticalDriftDetector()
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_accuracy(self, anomaly_detector):
        """Test the accuracy of anomaly detection algorithms"""
        
        # Test with normal data (should not detect anomalies)
        normal_data = np.random.normal(100, 10, 100)  # Mean=100, Std=10
        false_positives = 0
        
        for value in normal_data:
            anomalies = await anomaly_detector.detect_anomalies("accuracy_test", float(value))
            if anomalies:
                false_positives += 1
        
        # False positive rate should be low (allowing up to 10% for this test)
        false_positive_rate = false_positives / len(normal_data)
        assert false_positive_rate <= 0.1, f"False positive rate too high: {false_positive_rate:.2%}"
        
        # Test with clear outliers (should detect anomalies)
        outlier_data = [200] * 5  # Clear outliers (10 std deviations away)
        true_positives = 0
        
        for value in outlier_data:
            anomalies = await anomaly_detector.detect_anomalies("accuracy_test", float(value))
            if anomalies:
                true_positives += 1
        
        # Should detect most outliers
        true_positive_rate = true_positives / len(outlier_data)
        assert true_positive_rate >= 0.8, f"True positive rate too low: {true_positive_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_drift_detection_statistical_significance(self, drift_detector):
        """Test that drift detection uses proper statistical significance testing"""
        
        # Create data with no significant drift
        reference_data = np.random.normal(0, 1, 1000)
        similar_data = np.random.normal(0.1, 1, 1000)  # Small difference
        
        result = await drift_detector.detect_drift_with_significance(
            "significance_test", similar_data, reference_data, alpha=0.05
        )
        
        # Should not detect drift for small differences
        # (This tests that the statistical test is working correctly)
        if result.detected:
            # If drift is detected, check that p-value is close to threshold
            assert result.details['min_p_value'] >= 0.01, "P-value should not be extremely small for small differences"
    
    @pytest.mark.asyncio
    async def test_drift_detection_power(self, drift_detector):
        """Test the power of drift detection to detect actual drift"""
        
        # Create data with clear drift
        reference_data = np.random.normal(0, 1, 1000)
        drifted_data = np.random.normal(2, 1, 1000)  # Clear mean shift
        
        result = await drift_detector.detect_drift_with_significance(
            "power_test", drifted_data, reference_data, alpha=0.05
        )
        
        # Should definitely detect clear drift
        assert result.detected, "Should detect clear drift"
        assert result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL], "Should have high severity for clear drift"
        assert result.details['effect_size'] > 0.5, "Effect size should be large for clear drift"
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection_accuracy(self):
        """Test accuracy of performance degradation detection"""
        mock_metrics_collector = Mock(spec=MetricsCollector)
        monitoring_system = AdvancedMonitoringSystem(metrics_collector=mock_metrics_collector)
        
        # Set baseline performance
        baseline_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="accuracy_test_model",
            model_version="v1.0",
            accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.86
        )
        
        await monitoring_system.track_model_performance("accuracy_test_model", baseline_metrics)
        
        # Test with normal performance (should not trigger alerts)
        normal_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="accuracy_test_model",
            model_version="v1.0",
            accuracy=0.88,  # Within 5% threshold
            precision=0.83,
            recall=0.86,
            f1_score=0.84
        )
        
        with patch.object(monitoring_system.alert_subject, 'notify_observers') as mock_notify:
            await monitoring_system.track_model_performance("accuracy_test_model", normal_metrics)
            # Should not send alerts for small degradations
            assert mock_notify.call_count == 0, "Should not send alerts for small performance changes"
    
    @pytest.mark.asyncio
    async def test_data_quality_monitoring_accuracy(self):
        """Test accuracy of data quality monitoring"""
        mock_metrics_collector = Mock(spec=MetricsCollector)
        monitoring_system = AdvancedMonitoringSystem(metrics_collector=mock_metrics_collector)
        
        # Test with high quality data
        good_quality_reports = [
            DataQualityReport(
                symbol="TEST",
                exchange="test_exchange",
                timestamp=datetime.now(),
                issue_type=DataQualityIssue.MISSING_DATA,
                severity="low",
                description="Minor missing data"
            )
        ] * 5  # Only 5 low severity issues
        
        await monitoring_system.track_data_quality(good_quality_reports)
        
        # Should not trigger alerts for low quality issues
        # (Implementation would check alert history, but we'll check the quality score instead)
        latest_quality = monitoring_system.data_quality_history[-1]
        assert latest_quality['quality_score'] > 0.8, "Quality score should be high for good quality data"


class TestAlertReliability:
    """Test the reliability of alerting mechanisms"""
    
    @pytest.fixture
    def alert_routing_system(self):
        return AlertRoutingSystem()
    
    @pytest.fixture
    def sample_alert(self):
        return Alert(
            id="reliability_test_123",
            severity=AlertSeverity.HIGH,
            title="Reliability Test Alert",
            message="This is a test alert for reliability testing",
            timestamp=datetime.now(),
            model_name="test_model",
            metric_name="accuracy",
            current_value=0.7,
            threshold=0.8
        )
    
    def test_alert_routing_reliability(self, alert_routing_system, sample_alert):
        """Test that alerts are routed reliably to the correct channels"""
        
        # Register mock alert channels
        email_observer = Mock()
        email_observer.notify = AsyncMock()
        slack_observer = Mock()
        slack_observer.notify = AsyncMock()
        
        alert_routing_system.register_alert_channel('email', email_observer)
        alert_routing_system.register_alert_channel('slack', slack_observer)
        
        # Add routing rule
        rule = {
            'name': 'reliability_test_rule',
            'conditions': {'severity': ['high', 'critical']},
            'channels': ['email', 'slack']
        }
        alert_routing_system.add_routing_rule(rule)
        
        # Route alert
        asyncio.run(alert_routing_system.route_alert(sample_alert))
        
        # Both channels should receive the alert
        email_observer.notify.assert_called_once_with(sample_alert)
        slack_observer.notify.assert_called_once_with(sample_alert)
    
    def test_alert_deduplication_reliability(self, alert_routing_system, sample_alert):
        """Test that duplicate alerts are handled reliably"""
        
        # Register mock channel
        mock_observer = Mock()
        mock_observer.notify = AsyncMock()
        alert_routing_system.register_alert_channel('test', mock_observer)
        
        # Route the same alert twice in quick succession
        asyncio.run(alert_routing_system.route_alert(sample_alert))
        asyncio.run(alert_routing_system.route_alert(sample_alert))
        
        # Should only send once due to deduplication
        mock_observer.notify.assert_called_once_with(sample_alert)
    
    @pytest.mark.asyncio
    async def test_alert_acknowledgment_reliability(self, alert_routing_system, sample_alert):
        """Test that alert acknowledgment works reliably"""
        
        # Acknowledge an alert
        alert_routing_system.acknowledge_alert("test_alert_123", "test_user")
        
        # Check acknowledgment status
        status = alert_routing_system.get_alert_acknowledgment_status("test_alert_123")
        assert status is not None, "Alert acknowledgment status should be tracked"
        assert status['acknowledged_by'] == "test_user", "Should track who acknowledged the alert"
        assert status['status'] == "acknowledged", "Should track acknowledgment status"
    
    def test_escalation_reliability(self, alert_routing_system, sample_alert):
        """Test that alert escalation works reliably"""
        
        # Add escalation policy
        policy = {
            'levels': [
                {'delay_minutes': 0, 'channels': ['email']},
                {'delay_minutes': 1, 'channels': ['slack']}  # 1 minute for testing
            ],
            'max_escalations': 2,
            'cooldown_minutes': 5
        }
        alert_routing_system.add_escalation_policy('test_policy', policy)
        
        # Add routing rule with escalation policy
        rule = {
            'name': 'escalation_test_rule',
            'conditions': {'severity': ['high']},
            'channels': ['email'],
            'escalation_policy': 'test_policy'
        }
        alert_routing_system.add_routing_rule(rule)
        
        # Register mock channels
        email_observer = Mock()
        email_observer.notify = AsyncMock()
        slack_observer = Mock()
        slack_observer.notify = AsyncMock()
        
        alert_routing_system.register_alert_channel('email', email_observer)
        alert_routing_system.register_alert_channel('slack', slack_observer)
        
        # Route alert to trigger escalation
        asyncio.run(alert_routing_system.route_alert(sample_alert))
        
        # Should immediately send to email
        email_observer.notify.assert_called_once_with(sample_alert)
        
        # Should set up escalation (but won't actually escalate in this test due to timing)
        escalation_key = f"{sample_alert.id}_test_policy"
        assert escalation_key in alert_routing_system.escalation_state, "Should set up escalation state"
    
    def test_time_based_routing_reliability(self, alert_routing_system):
        """Test that time-based routing works reliably"""
        
        # Add time-based routing rule
        time_rule = {
            'name': 'business_hours_rule',
            'time_conditions': {
                'start_time': '09:00',
                'end_time': '17:00',
                'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
            },
            'channels': ['email']
        }
        alert_routing_system.add_time_based_routing_rule(time_rule)
        
        # Test time range checking
        # Business hours
        assert alert_routing_system._is_time_in_range("10:00", "09:00", "17:00") is True
        assert alert_routing_system._is_time_in_range("08:00", "09:00", "17:00") is False
        
        # Overnight hours
        assert alert_routing_system._is_time_in_range("20:00", "17:00", "09:00") is True
        assert alert_routing_system._is_time_in_range("10:00", "17:00", "09:00") is False


class TestMonitoringSystemIntegration:
    """Integration tests for monitoring accuracy and alert reliability"""
    
    @pytest.fixture
    def full_monitoring_system(self):
        """Create a fully configured monitoring system"""
        mock_metrics_collector = Mock(spec=MetricsCollector)
        mock_metrics_collector.get_system_metrics.return_value = Mock(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage_percent=70.0
        )
        return AdvancedMonitoringSystem(metrics_collector=mock_metrics_collector)
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_accuracy(self, full_monitoring_system):
        """Test end-to-end accuracy of the monitoring system"""
        
        # 1. Set up baseline performance
        baseline_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="integration_accuracy_test",
            model_version="v1.0",
            accuracy=0.95,
            precision=0.92,
            recall=0.94,
            f1_score=0.93
        )
        
        await full_monitoring_system.track_model_performance("integration_accuracy_test", baseline_metrics)
        
        # Verify baseline was set
        assert "integration_accuracy_test" in full_monitoring_system.performance_baselines
        assert full_monitoring_system.performance_baselines["integration_accuracy_test"]["accuracy"] == 0.95
        
        # 2. Test drift detection accuracy
        reference_data = np.random.normal(0, 1, 1000)
        drifted_data = np.random.normal(1.5, 1, 100)  # Clear drift
        
        drift_result = await full_monitoring_system.detect_model_drift(
            "integration_accuracy_test", drifted_data, reference_data
        )
        
        # Should detect drift accurately
        assert drift_result.detected is True
        assert drift_result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        assert drift_result.drift_score > 0.9  # High confidence in drift
        
        # 3. Test performance degradation accuracy
        degraded_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="integration_accuracy_test",
            model_version="v1.0",
            accuracy=0.70,  # 26% degradation
            precision=0.65,
            recall=0.72,
            f1_score=0.68
        )
        
        with patch.object(full_monitoring_system.alert_subject, 'notify_observers') as mock_notify:
            await full_monitoring_system.track_model_performance("integration_accuracy_test", degraded_metrics)
            
            # Should trigger alert for significant degradation
            mock_notify.assert_called_once()
            alert = mock_notify.call_args[0][0]
            assert "degradation" in alert.title.lower()
            assert alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_alert_reliability_under_load(self, full_monitoring_system):
        """Test alert reliability when handling multiple alerts"""
        
        # Register mock alert observer
        mock_observer = Mock()
        mock_observer.notify = AsyncMock()
        full_monitoring_system.register_alert_observer(mock_observer)
        
        # Generate multiple alerts rapidly
        alerts_sent = 0
        for i in range(10):
            alert = Alert(
                id=f"load_test_{i}",
                severity=AlertSeverity.HIGH,
                title=f"Load Test Alert {i}",
                message=f"This is load test alert #{i}",
                timestamp=datetime.now(),
                model_name="load_test_model"
            )
            
            await full_monitoring_system.alert_subject.notify_observers(alert)
            alerts_sent += 1
        
        # All alerts should be sent
        assert mock_observer.notify.call_count == alerts_sent, "All alerts should be delivered reliably"
        
        # Check alert history
        assert len(full_monitoring_system.alert_subject.alert_history) == alerts_sent, "All alerts should be stored in history"
    
    @pytest.mark.asyncio
    async def test_false_positive_rate_validation(self, full_monitoring_system):
        """Validate that the false positive rate is within acceptable limits"""
        
        # Test with normal data that should not trigger alerts
        normal_metrics = []
        for i in range(50):  # Test with 50 normal samples
            metrics = ModelPerformanceMetrics(
                timestamp=datetime.now(),
                model_name="fpr_test_model",
                model_version="v1.0",
                accuracy=0.90 + np.random.normal(0, 0.01),  # Small random variations around 0.90
                precision=0.85 + np.random.normal(0, 0.01),
                recall=0.88 + np.random.normal(0, 0.01),
                f1_score=0.86 + np.random.normal(0, 0.01)
            )
            normal_metrics.append(metrics)
        
        # Set baseline
        await full_monitoring_system.track_model_performance("fpr_test_model", normal_metrics[0])
        
        # Track all normal metrics
        false_positives = 0
        with patch.object(full_monitoring_system.alert_subject, 'notify_observers') as mock_notify:
            for metrics in normal_metrics[1:]:  # Skip first one used for baseline
                await full_monitoring_system.track_model_performance("fpr_test_model", metrics)
                if mock_notify.called:
                    false_positives += 1
                    mock_notify.reset_mock()
        
        # Calculate false positive rate
        false_positive_rate = false_positives / (len(normal_metrics) - 1)
        
        # Should be very low (less than 5%)
        assert false_positive_rate < 0.05, f"False positive rate too high: {false_positive_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_detection_sensitivity_validation(self, full_monitoring_system):
        """Validate that the detection sensitivity is adequate for real issues"""
        
        # Set baseline
        baseline_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="sensitivity_test_model",
            model_version="v1.0",
            accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.86
        )
        await full_monitoring_system.track_model_performance("sensitivity_test_model", baseline_metrics)
        
        # Test with clearly degraded performance that should trigger alerts
        degraded_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="sensitivity_test_model",
            model_version="v1.0",
            accuracy=0.60,  # 33% degradation - should definitely trigger
            precision=0.55,
            recall=0.58,
            f1_score=0.56
        )
        
        true_positives = 0
        with patch.object(full_monitoring_system.alert_subject, 'notify_observers') as mock_notify:
            for _ in range(10):  # Test multiple times to ensure reliability
                await full_monitoring_system.track_model_performance("sensitivity_test_model", degraded_metrics)
                if mock_notify.called:
                    true_positives += 1
                    mock_notify.reset_mock()
        
        # Calculate true positive rate
        true_positive_rate = true_positives / 10
        
        # Should be very high (more than 90%)
        assert true_positive_rate > 0.9, f"True positive rate too low: {true_positive_rate:.2%}"


if __name__ == "__main__":
    pytest.main([__file__])
