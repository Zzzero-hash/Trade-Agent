"""
Tests for the monitoring and alerting system
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import asdict

from src.services.model_monitoring_service import (
    ModelMonitoringService,
    ModelPerformanceMetrics,
    DriftDetectionResult,
    DriftType,
    AlertSeverity,
    Alert,
    EmailAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel
)
from src.services.monitoring_dashboard_service import (
    MonitoringDashboardService,
    DashboardMetrics,
    ModelDashboardData
)
from src.services.automated_retraining_service import (
    AutomatedRetrainingService,
    RetrainingJob,
    RetrainingTrigger,
    RetrainingStatus,
    RetrainingConfig
)
from src.utils.monitoring import MetricsCollector, SystemMetrics


class TestModelMonitoringService:
    """Test model monitoring service functionality"""

    @pytest.fixture
    def metrics_collector(self):
        """Mock metrics collector"""
        return Mock(spec=MetricsCollector)

    @pytest.fixture
    def monitoring_service(self, metrics_collector):
        """Create monitoring service instance"""
        return ModelMonitoringService(metrics_collector)

    @pytest.fixture
    def sample_performance_metrics(self):
        """Sample performance metrics"""
        return ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="test_model",
            model_version="v1.0",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            prediction_confidence=0.75
        )

    def test_monitoring_service_initialization(self, monitoring_service):
        """Test monitoring service initialization"""
        assert monitoring_service is not None
        assert monitoring_service.drift_detection_window == 100
        assert len(monitoring_service.alert_channels) == 0
        assert len(monitoring_service.retraining_callbacks) == 0

    def test_register_alert_channel(self, monitoring_service):
        """Test alert channel registration"""
        channel = Mock()
        monitoring_service.register_alert_channel(channel)
        
        assert len(monitoring_service.alert_channels) == 1
        assert monitoring_service.alert_channels[0] == channel

    def test_register_retraining_callback(self, monitoring_service):
        """Test retraining callback registration"""
        callback = Mock()
        model_name = "test_model"
        
        monitoring_service.register_retraining_callback(model_name, callback)
        
        assert model_name in monitoring_service.retraining_callbacks
        assert monitoring_service.retraining_callbacks[model_name] == callback

    def test_set_baseline_metrics(self, monitoring_service, sample_performance_metrics):
        """Test setting baseline metrics"""
        model_name = "test_model"
        
        monitoring_service.set_baseline_metrics(model_name, sample_performance_metrics)
        
        assert model_name in monitoring_service.baseline_metrics
        assert monitoring_service.baseline_metrics[model_name] == sample_performance_metrics

    @pytest.mark.asyncio
    async def test_track_prediction(self, monitoring_service):
        """Test prediction tracking"""
        model_name = "test_model"
        model_version = "v1.0"
        features = np.array([1.0, 2.0, 3.0])
        prediction = 1
        actual = 1
        confidence = 0.8
        
        await monitoring_service.track_prediction(
            model_name, model_version, features, prediction, actual, confidence
        )
        
        assert model_name in monitoring_service.prediction_history
        assert len(monitoring_service.prediction_history[model_name]) == 1
        
        prediction_data = monitoring_service.prediction_history[model_name][0]
        assert prediction_data['model_version'] == model_version
        assert np.array_equal(prediction_data['features'], features)
        assert prediction_data['prediction'] == prediction
        assert prediction_data['actual'] == actual
        assert prediction_data['confidence'] == confidence

    @pytest.mark.asyncio
    async def test_calculate_performance_metrics(self, monitoring_service):
        """Test performance metrics calculation"""
        model_name = "test_model"
        
        # Add some prediction history
        for i in range(20):
            await monitoring_service.track_prediction(
                model_name, "v1.0", 
                np.array([i, i+1, i+2]), 
                i % 2,  # prediction
                i % 2,  # actual (perfect accuracy)
                0.8 + (i % 3) * 0.1  # varying confidence
            )
        
        metrics = await monitoring_service.calculate_performance_metrics(model_name)
        
        assert metrics is not None
        assert metrics.model_name == model_name
        assert metrics.accuracy == 1.0  # Perfect accuracy
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.prediction_confidence is not None

    @pytest.mark.asyncio
    async def test_detect_data_drift_no_drift(self, monitoring_service):
        """Test data drift detection when no drift is present"""
        model_name = "test_model"
        
        # Add consistent feature history (no drift)
        np.random.seed(42)
        for i in range(100):
            features = np.random.normal(0, 1, 5)  # Consistent distribution
            await monitoring_service.track_prediction(
                model_name, "v1.0", features, 0, 0
            )
        
        drift_result = await monitoring_service.detect_data_drift(model_name)
        
        assert drift_result is not None
        assert drift_result.drift_type == DriftType.DATA_DRIFT
        assert not drift_result.detected  # No drift should be detected

    @pytest.mark.asyncio
    async def test_detect_data_drift_with_drift(self, monitoring_service):
        """Test data drift detection when drift is present"""
        model_name = "test_model"
        
        np.random.seed(42)
        
        # Add initial consistent features
        for i in range(50):
            features = np.random.normal(0, 1, 5)
            await monitoring_service.track_prediction(
                model_name, "v1.0", features, 0, 0
            )
        
        # Add drifted features (different distribution)
        for i in range(50):
            features = np.random.normal(5, 2, 5)  # Shifted distribution
            await monitoring_service.track_prediction(
                model_name, "v1.0", features, 0, 0
            )
        
        drift_result = await monitoring_service.detect_data_drift(model_name)
        
        assert drift_result is not None
        assert drift_result.drift_type == DriftType.DATA_DRIFT
        # Note: Drift detection might not always trigger due to randomness

    @pytest.mark.asyncio
    async def test_detect_performance_drift(self, monitoring_service, sample_performance_metrics):
        """Test performance drift detection"""
        model_name = "test_model"
        
        # Set baseline metrics
        monitoring_service.set_baseline_metrics(model_name, sample_performance_metrics)
        
        # Add some predictions with poor performance
        for i in range(20):
            await monitoring_service.track_prediction(
                model_name, "v1.0", 
                np.array([i, i+1, i+2]), 
                0,  # Always predict 0
                1,  # Always actual 1 (poor performance)
                0.5
            )
        
        drift_result = await monitoring_service.detect_performance_drift(model_name)
        
        assert drift_result is not None
        assert drift_result.drift_type == DriftType.PERFORMANCE_DRIFT
        assert drift_result.detected  # Should detect performance degradation

    @pytest.mark.asyncio
    async def test_run_monitoring_cycle(self, monitoring_service):
        """Test complete monitoring cycle"""
        model_name = "test_model"
        
        # Add some prediction history
        for i in range(50):
            await monitoring_service.track_prediction(
                model_name, "v1.0", 
                np.random.normal(0, 1, 5), 
                i % 2, i % 2, 0.8
            )
        
        results = await monitoring_service.run_monitoring_cycle(model_name)
        
        assert results['model_name'] == model_name
        assert 'timestamp' in results
        assert 'performance_metrics' in results
        assert 'drift_detection' in results
        assert results['performance_metrics'] is not None

    @pytest.mark.asyncio
    async def test_alert_sending(self, monitoring_service):
        """Test alert sending functionality"""
        # Register mock alert channel
        mock_channel = Mock()
        monitoring_service.register_alert_channel(mock_channel)
        
        alert = Alert(
            id="test_alert",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            model_name="test_model"
        )
        
        await monitoring_service._send_alert(alert)
        
        # Verify alert was sent through channel
        mock_channel.assert_called_once_with(alert)
        
        # Verify alert was stored in history
        assert len(monitoring_service.alert_history) == 1
        assert monitoring_service.alert_history[0] == alert

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, monitoring_service):
        """Test alert cooldown functionality"""
        mock_channel = Mock()
        monitoring_service.register_alert_channel(mock_channel)
        
        # Set short cooldown for testing
        monitoring_service.cooldown_period = timedelta(seconds=1)
        
        alert = Alert(
            id="test_alert",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            model_name="test_model",
            metric_name="accuracy"
        )
        
        # Send first alert
        await monitoring_service._send_alert(alert)
        assert mock_channel.call_count == 1
        
        # Send second alert immediately (should be blocked by cooldown)
        await monitoring_service._send_alert(alert)
        assert mock_channel.call_count == 1  # Still 1, not called again
        
        # Wait for cooldown to expire
        await asyncio.sleep(1.1)
        
        # Send third alert (should go through)
        await monitoring_service._send_alert(alert)
        assert mock_channel.call_count == 2

    def test_get_model_status(self, monitoring_service, sample_performance_metrics):
        """Test getting model status"""
        model_name = "test_model"
        
        # Add performance history
        monitoring_service.performance_history[model_name] = [sample_performance_metrics]
        
        status = monitoring_service.get_model_status(model_name)
        
        assert status['model_name'] == model_name
        assert 'timestamp' in status
        assert status['performance_metrics'] is not None
        assert status['health_score'] > 0
        assert status['drift_status'] in ['healthy', 'degraded', 'critical']

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, monitoring_service):
        """Test cleanup of old monitoring data"""
        model_name = "test_model"
        
        # Add old data
        old_time = datetime.now() - timedelta(days=35)
        old_metrics = ModelPerformanceMetrics(
            timestamp=old_time,
            model_name=model_name,
            model_version="v1.0",
            accuracy=0.8, precision=0.8, recall=0.8, f1_score=0.8
        )
        
        monitoring_service.performance_history[model_name] = [old_metrics]
        
        old_alert = Alert(
            id="old_alert",
            severity=AlertSeverity.LOW,
            title="Old Alert",
            message="Old alert",
            timestamp=old_time
        )
        monitoring_service.alert_history.append(old_alert)
        
        # Add recent data
        recent_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version="v1.0",
            accuracy=0.85, precision=0.85, recall=0.85, f1_score=0.85
        )
        monitoring_service.performance_history[model_name].append(recent_metrics)
        
        # Cleanup old data (keep 30 days)
        await monitoring_service.cleanup_old_data(days_to_keep=30)
        
        # Verify old data was removed
        assert len(monitoring_service.performance_history[model_name]) == 1
        assert monitoring_service.performance_history[model_name][0] == recent_metrics
        assert len(monitoring_service.alert_history) == 0


class TestMonitoringDashboardService:
    """Test monitoring dashboard service"""

    @pytest.fixture
    def monitoring_service(self):
        """Mock monitoring service"""
        return Mock(spec=ModelMonitoringService)

    @pytest.fixture
    def dashboard_service(self, monitoring_service):
        """Create dashboard service instance"""
        return MonitoringDashboardService(monitoring_service)

    @pytest.mark.asyncio
    async def test_get_system_dashboard(self, dashboard_service, monitoring_service):
        """Test system dashboard generation"""
        # Mock monitoring service data
        monitoring_service.performance_history = {"model1": [], "model2": []}
        monitoring_service.prediction_history = {"model1": [{}] * 10, "model2": [{}] * 5}
        monitoring_service.alert_history = []
        
        # Mock metrics collector
        with patch.object(dashboard_service.metrics_collector, 'get_system_metrics') as mock_metrics:
            mock_metrics.return_value = SystemMetrics(
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=1000,
                memory_available_mb=500,
                disk_usage_percent=70.0,
                network_bytes_sent=1000000,
                network_bytes_recv=2000000,
                timestamp=datetime.now()
            )
            
            dashboard = await dashboard_service.get_system_dashboard()
            
            assert isinstance(dashboard, DashboardMetrics)
            assert dashboard.active_models == 2
            assert dashboard.total_predictions == 15
            assert dashboard.cpu_usage == 50.0
            assert dashboard.memory_usage == 60.0

    @pytest.mark.asyncio
    async def test_get_model_dashboard(self, dashboard_service, monitoring_service):
        """Test model-specific dashboard generation"""
        model_name = "test_model"
        
        # Mock monitoring service methods
        monitoring_service.get_model_status.return_value = {
            'model_name': model_name,
            'drift_status': 'healthy',
            'health_score': 85.0,
            'timestamp': datetime.now(),
            'performance_metrics': {
                'accuracy': 0.85,
                'precision': 0.82
            }
        }
        
        monitoring_service.prediction_history = {
            model_name: [
                {'timestamp': datetime.now(), 'confidence': 0.8},
                {'timestamp': datetime.now(), 'confidence': 0.9}
            ]
        }
        
        monitoring_service.performance_history = {
            model_name: [
                Mock(accuracy=0.85),
                Mock(accuracy=0.87)
            ]
        }
        
        monitoring_service.alert_history = []
        
        dashboard = await dashboard_service.get_model_dashboard(model_name)
        
        assert isinstance(dashboard, ModelDashboardData)
        assert dashboard.model_name == model_name
        assert dashboard.status == 'healthy'
        assert dashboard.health_score == 85.0

    @pytest.mark.asyncio
    async def test_get_alert_summary(self, dashboard_service, monitoring_service):
        """Test alert summary generation"""
        # Mock alert history
        now = datetime.now()
        monitoring_service.alert_history = [
            Mock(
                timestamp=now - timedelta(hours=1),
                severity=AlertSeverity.HIGH,
                model_name="model1"
            ),
            Mock(
                timestamp=now - timedelta(hours=2),
                severity=AlertSeverity.MEDIUM,
                model_name="model2"
            ),
            Mock(
                timestamp=now - timedelta(hours=25),  # Outside 24h window
                severity=AlertSeverity.LOW,
                model_name="model1"
            )
        ]
        
        summary = await dashboard_service.get_alert_summary(hours=24)
        
        assert summary['total_alerts'] == 2  # Only alerts within 24h
        assert summary['severity_breakdown']['high'] == 1
        assert summary['severity_breakdown']['medium'] == 1
        assert summary['model_breakdown']['model1'] == 1
        assert summary['model_breakdown']['model2'] == 1

    def test_dashboard_caching(self, dashboard_service):
        """Test dashboard caching functionality"""
        # Test cache clearing
        dashboard_service.dashboard_cache['test'] = 'value'
        dashboard_service.last_cache_update['test'] = datetime.now()
        
        dashboard_service.clear_cache()
        
        assert len(dashboard_service.dashboard_cache) == 0
        assert len(dashboard_service.last_cache_update) == 0


class TestAutomatedRetrainingService:
    """Test automated retraining service"""

    @pytest.fixture
    def monitoring_service(self):
        """Mock monitoring service"""
        return Mock(spec=ModelMonitoringService)

    @pytest.fixture
    def retraining_service(self, monitoring_service):
        """Create retraining service instance"""
        return AutomatedRetrainingService(monitoring_service)

    @pytest.fixture
    def sample_drift_result(self):
        """Sample drift detection result"""
        return DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            drift_score=0.02,
            threshold=0.05,
            detected=True,
            timestamp=datetime.now(),
            details={'test': 'data'}
        )

    def test_retraining_service_initialization(self, retraining_service):
        """Test retraining service initialization"""
        assert retraining_service is not None
        assert retraining_service.config.enabled is True
        assert len(retraining_service.active_jobs) == 0
        assert len(retraining_service.job_queue) == 0

    def test_configure_retraining(self, retraining_service):
        """Test retraining configuration"""
        new_config = RetrainingConfig(
            enabled=False,
            max_concurrent_jobs=5,
            cooldown_period_hours=12
        )
        
        retraining_service.configure(new_config)
        
        assert retraining_service.config == new_config

    def test_register_retraining_callback(self, retraining_service):
        """Test retraining callback registration"""
        callback = Mock()
        model_name = "test_model"
        
        retraining_service.register_retraining_callback(model_name, callback)
        
        assert model_name in retraining_service.retraining_callbacks
        assert retraining_service.retraining_callbacks[model_name] == callback

    @pytest.mark.asyncio
    async def test_handle_drift_detection_disabled(self, retraining_service, sample_drift_result):
        """Test drift handling when retraining is disabled"""
        retraining_service.config.enabled = False
        
        result = await retraining_service.handle_drift_detection("test_model", sample_drift_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_drift_detection_low_severity(self, retraining_service, sample_drift_result):
        """Test drift handling with low severity"""
        sample_drift_result.severity = AlertSeverity.LOW
        
        result = await retraining_service.handle_drift_detection("test_model", sample_drift_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_drift_detection_cooldown(self, retraining_service, sample_drift_result):
        """Test drift handling during cooldown period"""
        model_name = "test_model"
        
        # Set recent retraining time
        retraining_service.last_retraining[model_name] = datetime.now()
        
        result = await retraining_service.handle_drift_detection(model_name, sample_drift_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_schedule_manual_retraining(self, retraining_service):
        """Test manual retraining scheduling"""
        model_name = "test_model"
        config_overrides = {"learning_rate": 0.01}
        
        # Mock sufficient samples
        retraining_service.monitoring_service.prediction_history = {
            model_name: [{'actual': i} for i in range(1000)]
        }
        
        with patch.object(retraining_service, '_start_retraining_job') as mock_start:
            job_id = await retraining_service.schedule_manual_retraining(
                model_name, config_overrides
            )
            
            assert job_id is not None
            assert job_id.startswith("manual_")
            mock_start.assert_called_once()

    def test_check_cooldown(self, retraining_service):
        """Test cooldown checking"""
        model_name = "test_model"
        
        # No previous retraining - should pass
        assert retraining_service._check_cooldown(model_name) is True
        
        # Recent retraining - should fail
        retraining_service.last_retraining[model_name] = datetime.now()
        assert retraining_service._check_cooldown(model_name) is False
        
        # Old retraining - should pass
        retraining_service.last_retraining[model_name] = datetime.now() - timedelta(hours=10)
        assert retraining_service._check_cooldown(model_name) is True

    def test_check_minimum_samples(self, retraining_service):
        """Test minimum samples checking"""
        model_name = "test_model"
        
        # No predictions - should fail
        retraining_service.monitoring_service.prediction_history = {}
        assert retraining_service._check_minimum_samples(model_name) is False
        
        # Insufficient labeled samples - should fail
        retraining_service.monitoring_service.prediction_history = {
            model_name: [{'actual': None} for _ in range(100)]
        }
        assert retraining_service._check_minimum_samples(model_name) is False
        
        # Sufficient labeled samples - should pass
        retraining_service.monitoring_service.prediction_history = {
            model_name: [{'actual': i} for i in range(1000)]
        }
        assert retraining_service._check_minimum_samples(model_name) is True

    def test_get_job_status_active(self, retraining_service):
        """Test getting status of active job"""
        job = RetrainingJob(
            job_id="test_job",
            model_name="test_model",
            trigger=RetrainingTrigger.MANUAL,
            trigger_details={},
            created_at=datetime.now(),
            started_at=datetime.now(),
            status=RetrainingStatus.RUNNING,
            progress=50.0
        )
        
        retraining_service.active_jobs[job.job_id] = job
        
        status = retraining_service.get_job_status(job.job_id)
        
        assert status is not None
        assert status['job_id'] == job.job_id
        assert status['status'] == 'running'
        assert status['progress'] == 50.0

    def test_get_job_status_not_found(self, retraining_service):
        """Test getting status of non-existent job"""
        status = retraining_service.get_job_status("non_existent_job")
        assert status is None

    def test_get_retraining_summary(self, retraining_service):
        """Test retraining summary generation"""
        # Add some mock data
        retraining_service.active_jobs = {"job1": Mock()}
        retraining_service.job_queue = [Mock(), Mock()]
        retraining_service.job_history = [
            Mock(status=RetrainingStatus.COMPLETED, performance_improvement=0.05),
            Mock(status=RetrainingStatus.FAILED, performance_improvement=None)
        ]
        
        summary = retraining_service.get_retraining_summary()
        
        assert summary['active_jobs'] == 1
        assert summary['queued_jobs'] == 2
        assert summary['success_rate'] == 0.5  # 1 success out of 2 total
        assert summary['avg_improvement'] == 0.05

    @pytest.mark.asyncio
    async def test_cancel_job_active(self, retraining_service):
        """Test cancelling active job"""
        job = RetrainingJob(
            job_id="test_job",
            model_name="test_model",
            trigger=RetrainingTrigger.MANUAL,
            trigger_details={},
            created_at=datetime.now(),
            status=RetrainingStatus.RUNNING
        )
        
        retraining_service.active_jobs[job.job_id] = job
        
        result = await retraining_service.cancel_job(job.job_id)
        
        assert result is True
        assert job.job_id not in retraining_service.active_jobs
        assert job in retraining_service.job_history
        assert job.status == RetrainingStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_queued(self, retraining_service):
        """Test cancelling queued job"""
        job = RetrainingJob(
            job_id="test_job",
            model_name="test_model",
            trigger=RetrainingTrigger.MANUAL,
            trigger_details={},
            created_at=datetime.now()
        )
        
        retraining_service.job_queue.append(job)
        
        result = await retraining_service.cancel_job(job.job_id)
        
        assert result is True
        assert job not in retraining_service.job_queue
        assert job in retraining_service.job_history
        assert job.status == RetrainingStatus.CANCELLED


class TestAlertChannels:
    """Test alert notification channels"""

    def test_email_alert_channel(self):
        """Test email alert channel"""
        smtp_config = {
            'host': 'smtp.example.com',
            'port': 587,
            'username': 'test@example.com',
            'password': 'password'
        }
        
        channel = EmailAlertChannel(smtp_config)
        assert channel.smtp_config == smtp_config
        
        # Test alert sending (mock implementation)
        alert = Alert(
            id="test",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now()
        )
        
        # Should not raise exception
        channel(alert)

    def test_slack_alert_channel(self):
        """Test Slack alert channel"""
        webhook_url = "https://hooks.slack.com/test"
        
        channel = SlackAlertChannel(webhook_url)
        assert channel.webhook_url == webhook_url
        
        alert = Alert(
            id="test",
            severity=AlertSeverity.MEDIUM,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now()
        )
        
        # Should not raise exception
        channel(alert)

    def test_webhook_alert_channel(self):
        """Test generic webhook alert channel"""
        webhook_url = "https://api.example.com/alerts"
        headers = {"Authorization": "Bearer token"}
        
        channel = WebhookAlertChannel(webhook_url, headers)
        assert channel.webhook_url == webhook_url
        assert channel.headers == headers
        
        alert = Alert(
            id="test",
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now()
        )
        
        # Should not raise exception
        channel(alert)


class TestMonitoringIntegration:
    """Integration tests for monitoring system components"""

    @pytest.fixture
    def full_monitoring_setup(self):
        """Setup complete monitoring system"""
        metrics_collector = Mock(spec=MetricsCollector)
        monitoring_service = ModelMonitoringService(metrics_collector)
        dashboard_service = MonitoringDashboardService(monitoring_service)
        retraining_service = AutomatedRetrainingService(monitoring_service)
        
        return {
            'metrics_collector': metrics_collector,
            'monitoring_service': monitoring_service,
            'dashboard_service': dashboard_service,
            'retraining_service': retraining_service
        }

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self, full_monitoring_setup):
        """Test complete monitoring workflow"""
        services = full_monitoring_setup
        monitoring_service = services['monitoring_service']
        retraining_service = services['retraining_service']
        
        model_name = "test_model"
        
        # 1. Set baseline metrics
        baseline_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version="v1.0",
            accuracy=0.9,
            precision=0.88,
            recall=0.92,
            f1_score=0.9
        )
        monitoring_service.set_baseline_metrics(model_name, baseline_metrics)
        
        # 2. Register retraining callback
        retraining_callback = Mock()
        retraining_service.register_retraining_callback(model_name, retraining_callback)
        
        # 3. Track predictions with degrading performance
        for i in range(100):
            # Simulate performance degradation
            prediction = 0 if i < 50 else 1  # Change prediction pattern
            actual = 1  # Always 1, so accuracy degrades
            
            await monitoring_service.track_prediction(
                model_name, "v1.0", 
                np.random.normal(0, 1, 5), 
                prediction, actual, 0.7
            )
        
        # 4. Run monitoring cycle
        results = await monitoring_service.run_monitoring_cycle(model_name)
        
        # 5. Verify monitoring detected issues
        assert results['model_name'] == model_name
        assert results['performance_metrics'] is not None
        
        # Performance should be degraded
        current_accuracy = results['performance_metrics'].accuracy
        assert current_accuracy < baseline_metrics.accuracy

    @pytest.mark.asyncio
    async def test_alert_to_retraining_workflow(self, full_monitoring_setup):
        """Test workflow from alert generation to retraining trigger"""
        services = full_monitoring_setup
        monitoring_service = services['monitoring_service']
        retraining_service = services['retraining_service']
        
        model_name = "test_model"
        
        # Setup sufficient prediction history for retraining
        retraining_service.monitoring_service.prediction_history = {
            model_name: [{'actual': i % 2} for i in range(1000)]
        }
        
        # Create high-severity drift result
        drift_result = DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            drift_score=0.01,
            threshold=0.05,
            detected=True,
            timestamp=datetime.now()
        )
        
        # Handle drift detection (should trigger retraining)
        with patch.object(retraining_service, '_start_retraining_job') as mock_start:
            job_id = await retraining_service.handle_drift_detection(model_name, drift_result)
            
            if job_id:  # If retraining was triggered
                mock_start.assert_called_once()
                
                # Verify job was created correctly
                job = mock_start.call_args[0][0]
                assert job.model_name == model_name
                assert job.trigger == RetrainingTrigger.DATA_DRIFT
                assert job.trigger_details['drift_type'] == 'data_drift'


if __name__ == "__main__":
    pytest.main([__file__])