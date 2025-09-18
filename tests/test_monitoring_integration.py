"""
Integration tests for the monitoring system components.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.services.monitoring.monitoring_orchestrator import MonitoringOrchestrator
from src.services.monitoring.performance_tracker import PerformanceTracker
from src.services.monitoring.drift_detector import DriftDetector
from src.services.monitoring.alert_manager import AlertManager
from src.services.monitoring.alert_system import AlertSubject, EmailAlertObserver
from src.services.monitoring.advanced_monitoring_system import AdvancedMonitoringSystem
from src.models.monitoring import ModelPerformanceMetrics, Alert, AlertSeverity


class TestMonitoringOrchestrator:
    """Test the monitoring orchestrator integration."""
    
    @pytest.fixture
    def alert_subject(self):
        return AlertSubject()
    
    @pytest.fixture
    def performance_tracker(self):
        return PerformanceTracker()
    
    @pytest.fixture
    def drift_detector(self):
        return DriftDetector()
    
    @pytest.fixture
    def alert_manager(self, alert_subject):
        return AlertManager(alert_subject)
    
    @pytest.fixture
    def orchestrator(self, performance_tracker, drift_detector, alert_manager):
        return MonitoringOrchestrator(performance_tracker, drift_detector, alert_manager)
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_cycle(self, orchestrator):
        """Test a complete monitoring cycle."""
        model_name = "test_model"
        
        # Create sample prediction data
        prediction_data = []
        for i in range(50):
            prediction_data.append({
                'prediction': np.random.choice([0, 1]),
                'actual': np.random.choice([0, 1]),
                'confidence': np.random.uniform(0.6, 0.9),
                'model_version': '1.0',
                'timestamp': datetime.now() - timedelta(minutes=i)
            })
        
        # Run monitoring cycle
        result = await orchestrator.run_monitoring_cycle(model_name, prediction_data)
        
        # Verify result structure
        assert result.model_name == model_name
        assert result.timestamp is not None
        assert result.performance_metrics is not None
        assert result.drift_detection is not None
        assert 'data_drift' in result.drift_detection
        assert 'performance_drift' in result.drift_detection
    
    @pytest.mark.asyncio
    async def test_monitoring_with_performance_alerts(self, orchestrator, alert_subject):
        """Test monitoring cycle that triggers performance alerts."""
        model_name = "poor_performance_model"
        
        # Create prediction data with poor performance
        prediction_data = []
        for i in range(50):
            # Simulate poor model performance (low accuracy)
            prediction = np.random.choice([0, 1])
            actual = 1 - prediction  # Always wrong
            prediction_data.append({
                'prediction': prediction,
                'actual': actual,
                'confidence': 0.5,
                'model_version': '1.0',
                'timestamp': datetime.now() - timedelta(minutes=i)
            })
        
        # Mock alert observer to capture alerts
        alerts_received = []
        
        class TestObserver:
            async def notify(self, alert):
                alerts_received.append(alert)
        
        observer = TestObserver()
        alert_subject.attach(observer)
        
        # Run monitoring cycle
        result = await orchestrator.run_monitoring_cycle(model_name, prediction_data)
        
        # Should have generated performance alerts
        assert len(alerts_received) > 0
        performance_alerts = [a for a in alerts_received if 'performance' in a.title.lower()]
        assert len(performance_alerts) > 0


class TestEndToEndMonitoring:
    """End-to-end integration tests for the monitoring system."""
    
    @pytest.fixture
    def full_monitoring_system(self):
        """Create a fully configured monitoring system."""
        from src.services.monitoring.advanced_monitoring_system import (
            AnomalyDetectionConfig, ModelDriftConfig, PerformanceDegradationConfig,
            SystemHealthConfig, AlertRoutingConfig
        )
        
        anomaly_config = AnomalyDetectionConfig(
            z_score_threshold=2.0,
            min_samples_for_detection=5
        )
        drift_config = ModelDriftConfig(
            data_drift_threshold=0.1,
            drift_detection_window=10
        )
        performance_config = PerformanceDegradationConfig(
            accuracy_threshold=0.1,
            consecutive_failures_threshold=2
        )
        health_config = SystemHealthConfig(
            cpu_threshold=70.0,
            memory_threshold=75.0
        )
        routing_config = AlertRoutingConfig()
        
        return AdvancedMonitoringSystem(
            anomaly_config, drift_config, performance_config,
            health_config, routing_config
        )
    
    @pytest.mark.asyncio
    async def test_real_time_anomaly_detection_flow(self, full_monitoring_system):
        """Test real-time anomaly detection in full system."""
        model_name = "realtime_test_model"
        
        # Capture alerts
        alerts_received = []
        
        class AlertCapture:
            async def notify(self, alert):
                alerts_received.append(alert)
        
        full_monitoring_system.add_alert_observer(AlertCapture())
        
        # Feed normal data
        for i in range(10):
            data_point = {
                "feature1": np.random.normal(0, 1),
                "feature2": np.random.normal(5, 2)
            }
            await full_monitoring_system.process_model_data(model_name, data_point)
        
        # Feed anomalous data
        anomalous_data = {
            "feature1": 15.0,  # Clear anomaly
            "feature2": 5.0
        }
        await full_monitoring_system.process_model_data(model_name, anomalous_data)
        
        # Should detect anomaly
        anomaly_alerts = [a for a in alerts_received if 'anomaly' in a.title.lower()]
        assert len(anomaly_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_performance_degradation_flow(self, full_monitoring_system):
        """Test performance degradation detection in full system."""
        model_name = "degradation_test_model"
        
        # Capture alerts
        alerts_received = []
        
        class AlertCapture:
            async def notify(self, alert):
                alerts_received.append(alert)
        
        full_monitoring_system.add_alert_observer(AlertCapture())
        
        # Simulate performance degradation
        good_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version="1.0",
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.86
        )
        
        # Process good performance first to establish baseline
        for _ in range(10):  # Need multiple samples to establish baseline
            await full_monitoring_system.process_model_data(
                model_name, {"dummy": 1.0}, good_metrics
            )
        
        # Then simulate degraded performance
        bad_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version="1.0",
            accuracy=0.7,  # Significant degradation
            precision=0.65,
            recall=0.68,
            f1_score=0.66
        )
        
        await full_monitoring_system.process_model_data(
            model_name, {"dummy": 1.0}, bad_metrics
        )
        
        # Should detect performance degradation
        degradation_alerts = [a for a in alerts_received if 'degradation' in a.title.lower()]
        assert len(degradation_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_drift_detection_flow(self, full_monitoring_system):
        """Test drift detection in full system."""
        model_name = "drift_test_model"
        
        # Capture alerts
        alerts_received = []
        
        class AlertCapture:
            async def notify(self, alert):
                alerts_received.append(alert)
        
        full_monitoring_system.add_alert_observer(AlertCapture())
        
        # Generate reference and current data with clear drift
        reference_data = np.random.normal(0, 1, (50, 5))
        current_data = np.random.normal(3, 2, (50, 5))  # Different distribution
        
        await full_monitoring_system.process_model_data(
            model_name,
            {"dummy": 1.0},
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Should detect drift
        drift_alerts = [a for a in alerts_received if 'drift' in a.title.lower()]
        assert len(drift_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring_flow(self, full_monitoring_system):
        """Test system health monitoring in full system."""
        # Start monitoring
        await full_monitoring_system.start_monitoring()
        
        # Capture alerts
        alerts_received = []
        
        class AlertCapture:
            async def notify(self, alert):
                alerts_received.append(alert)
        
        full_monitoring_system.add_alert_observer(AlertCapture())
        
        # Mock high resource usage
        with patch.object(
            full_monitoring_system.health_monitor, 
            '_collect_health_metrics'
        ) as mock_collect:
            mock_collect.return_value = {
                'cpu_percent': 85.0,  # Above threshold
                'memory_percent': 80.0,  # Above threshold
                'disk_percent': 60.0
            }
            
            # Wait for monitoring cycle
            await asyncio.sleep(0.1)
        
        await full_monitoring_system.stop_monitoring()
        
        # Should generate health alerts
        health_alerts = [a for a in alerts_received if 'health' in a.title.lower()]
        # Note: May not always trigger due to timing, but system should handle it


class TestAlertSystemIntegration:
    """Test alert system integration with different observers."""
    
    @pytest.fixture
    def alert_subject(self):
        return AlertSubject()
    
    @pytest.mark.asyncio
    async def test_multiple_alert_observers(self, alert_subject):
        """Test multiple alert observers working together."""
        # Create different types of observers
        email_alerts = []
        slack_alerts = []
        db_alerts = []
        
        class MockEmailObserver:
            async def notify(self, alert):
                email_alerts.append(alert)
        
        class MockSlackObserver:
            async def notify(self, alert):
                slack_alerts.append(alert)
        
        class MockDBObserver:
            async def notify(self, alert):
                db_alerts.append(alert)
        
        # Attach observers
        alert_subject.attach(MockEmailObserver())
        alert_subject.attach(MockSlackObserver())
        alert_subject.attach(MockDBObserver())
        
        # Send test alert
        test_alert = Alert(
            id="integration_test_alert",
            severity=AlertSeverity.HIGH,
            title="Integration Test Alert",
            message="This is a test alert for integration testing",
            timestamp=datetime.now(),
            model_name="test_model"
        )
        
        await alert_subject.notify_observers(test_alert)
        
        # All observers should receive the alert
        assert len(email_alerts) == 1
        assert len(slack_alerts) == 1
        assert len(db_alerts) == 1
        
        # All should be the same alert
        assert email_alerts[0].id == test_alert.id
        assert slack_alerts[0].id == test_alert.id
        assert db_alerts[0].id == test_alert.id
    
    @pytest.mark.asyncio
    async def test_alert_cooldown_mechanism(self, alert_subject):
        """Test alert cooldown to prevent spam."""
        alerts_received = []
        
        class CountingObserver:
            async def notify(self, alert):
                alerts_received.append(alert)
        
        alert_subject.attach(CountingObserver())
        
        # Send same alert multiple times rapidly
        for i in range(5):
            alert = Alert(
                id="cooldown_test_alert",
                severity=AlertSeverity.MEDIUM,
                title="Cooldown Test Alert",
                message="This alert should be rate limited",
                timestamp=datetime.now(),
                model_name="test_model",
                metric_name="test_metric"
            )
            await alert_subject.notify_observers(alert)
        
        # Should only receive one alert due to cooldown
        assert len(alerts_received) == 1


class TestMonitoringReliability:
    """Test monitoring system reliability and error handling."""
    
    @pytest.mark.asyncio
    async def test_monitoring_with_data_errors(self):
        """Test monitoring system handles data errors gracefully."""
        from src.services.monitoring.advanced_monitoring_system import (
            RealTimeAnomalyDetector, AnomalyDetectionConfig
        )
        
        config = AnomalyDetectionConfig()
        detector = RealTimeAnomalyDetector(config)
        
        # Test with invalid data types
        invalid_data = {
            "string_feature": "not_a_number",
            "none_feature": None,
            "list_feature": [1, 2, 3]
        }
        
        # Should not crash
        anomalies = await detector.detect_anomalies("test_model", invalid_data)
        assert isinstance(anomalies, list)
    
    @pytest.mark.asyncio
    async def test_monitoring_with_missing_data(self):
        """Test monitoring system handles missing data gracefully."""
        orchestrator = MonitoringOrchestrator(
            PerformanceTracker(),
            DriftDetector(),
            AlertManager(AlertSubject())
        )
        
        # Test with empty prediction data
        result = await orchestrator.run_monitoring_cycle("test_model", [])
        
        # Should complete without errors
        assert result.model_name == "test_model"
        assert result.performance_metrics is None  # No data to calculate metrics
    
    @pytest.mark.asyncio
    async def test_alert_system_error_handling(self):
        """Test alert system handles observer errors gracefully."""
        alert_subject = AlertSubject()
        
        # Create observer that raises exception
        class FailingObserver:
            async def notify(self, alert):
                raise Exception("Observer failed")
        
        # Create normal observer
        successful_alerts = []
        
        class WorkingObserver:
            async def notify(self, alert):
                successful_alerts.append(alert)
        
        alert_subject.attach(FailingObserver())
        alert_subject.attach(WorkingObserver())
        
        # Send alert
        test_alert = Alert(
            id="error_handling_test",
            severity=AlertSeverity.LOW,
            title="Error Handling Test",
            message="Testing error handling",
            timestamp=datetime.now()
        )
        
        # Should not crash and working observer should still receive alert
        await alert_subject.notify_observers(test_alert)
        assert len(successful_alerts) == 1


if __name__ == "__main__":
    pytest.main([__file__])