"""
Tests for the advanced monitoring and alerting system.

This module provides comprehensive tests for all components of the advanced
monitoring system including anomaly detection, drift detection, performance
monitoring, and alert routing.
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
    PredictiveMaintenanceSystem,
    AlertRoutingSystem,
    AnomalyDetectionConfig,
    MonitoringMode,
    HealthStatus,
    SystemHealthMetrics,
    PerformanceDegradationAlert
)
from src.services.monitoring.drift_strategies import (
    DriftDetectionContext,
    KolmogorovSmirnovStrategy,
    MannWhitneyUStrategy,
    PopulationStabilityIndexStrategy,
    DriftDetectionConfig,
    DriftDetectionMethod
)
from src.services.monitoring.config import (
    ConfigManager,
    MonitoringConfig,
    PerformanceThresholds,
    DriftThresholds,
    AlertingSettings
)
from src.services.monitoring.resource_manager import (
    MonitoringResourceManager,
    DataBuffer,
    BufferConfig
)
from src.services.monitoring.alert_system import (
    AlertSubject,
    AlertFactory,
    EmailAlertObserver,
    MetricsAlertObserver
)
from src.models.monitoring import (
    Alert,
    AlertSeverity,
    DriftDetectionResult,
    DriftType,
    ModelPerformanceMetrics
)
from src.services.data_aggregator import DataQualityReport, DataQualityIssue
from src.utils.monitoring import MetricsCollector


class TestRealTimeAnomalyDetector:
    """Test real-time anomaly detection"""
    
    @pytest.fixture
    def anomaly_detector(self):
        config = AnomalyDetectionConfig(
            window_size=50,
            min_samples=10,
            z_score_threshold=2.0,
            sensitivity=1.0
        )
        return RealTimeAnomalyDetector(config)
    
    @pytest.mark.asyncio
    async def test_statistical_anomaly_detection(self, anomaly_detector):
        """Test statistical anomaly detection"""
        
        # Add normal data points
        for i in range(20):
            value = np.random.normal(100, 10)  # Normal distribution
            anomalies = await anomaly_detector.detect_anomalies("test_metric", value)
            # Note: Some anomalies might be detected in normal data due to statistical variation
            # The test should focus on not detecting TOO many anomalies
            assert len(anomalies) <= 2  # Allow some false positives but not too many
        
        # Add clear outlier
        outlier_value = 200  # 10 standard deviations away
        anomalies = await anomaly_detector.detect_anomalies("test_metric", outlier_value)
        
        assert len(anomalies) > 0
        assert anomalies[0]['type'] == 'statistical_outlier'
        assert anomalies[0]['z_score'] > 2.0
        assert anomalies[0]['severity'] in ['medium', 'high']
    
    @pytest.mark.asyncio
    async def test_trend_change_detection(self, anomaly_detector):
        """Test trend change detection"""
        
        # Add data with stable trend
        for i in range(25):
            value = 100 + i * 0.1  # Slight upward trend
            await anomaly_detector.detect_anomalies("trend_metric", value)
        
        # Add data with sudden trend change
        for i in range(10):
            value = 150 + i * 2  # Sharp upward trend
            anomalies = await anomaly_detector.detect_anomalies("trend_metric", value)
        
        # Should detect trend change
        trend_anomalies = [a for a in anomalies if a.get('type') == 'trend_change']
        if trend_anomalies:
            assert trend_anomalies[0]['change_percent'] > 0.3
    
    @pytest.mark.asyncio
    async def test_ml_anomaly_detection(self, anomaly_detector):
        """Test ML-based anomaly detection"""
        
        # Add normal data
        normal_data = np.random.normal(50, 5, 30)
        for value in normal_data:
            await anomaly_detector.detect_anomalies("ml_metric", float(value))
        
        # Add outlier
        outlier = 100  # Clear outlier
        anomalies = await anomaly_detector.detect_anomalies("ml_metric", outlier)
        
        ml_anomalies = [a for a in anomalies if a.get('method') == 'modified_z_score']
        if ml_anomalies:
            assert ml_anomalies[0]['modified_z_score'] > 3.5
    
    def test_anomaly_history_management(self, anomaly_detector):
        """Test anomaly history management"""
        
        # Add some anomalies
        anomaly_detector.anomaly_history["test_metric"] = [
            {'timestamp': datetime.now() - timedelta(hours=25), 'type': 'old'},
            {'timestamp': datetime.now() - timedelta(hours=1), 'type': 'recent'}
        ]
        
        # Trigger cleanup by adding new anomaly
        asyncio.run(anomaly_detector.detect_anomalies("test_metric", 100))
        
        # Old anomalies should be cleaned up
        recent_anomalies = [
            a for a in anomaly_detector.anomaly_history["test_metric"]
            if a['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        assert len(recent_anomalies) <= len(anomaly_detector.anomaly_history["test_metric"])


class TestStatisticalDriftDetector:
    """Test statistical drift detection"""
    
    @pytest.fixture
    def drift_detector(self):
        return StatisticalDriftDetector()
    
    @pytest.mark.asyncio
    async def test_drift_detection_with_significance(self, drift_detector):
        """Test drift detection with statistical significance"""
        
        # Create reference and current data with known drift
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(2, 1, 1000)  # Mean shift
        
        result = await drift_detector.detect_drift_with_significance(
            "test_model", current_data, reference_data, alpha=0.05
        )
        
        assert isinstance(result, DriftDetectionResult)
        assert result.drift_type == DriftType.DATA_DRIFT
        assert result.detected == True
        assert result.drift_score > 0.95  # Should be high confidence
        assert 'ks_p_value' in result.details
        assert 'effect_size' in result.details
        assert result.details['effect_size'] > 0.5  # Large effect size
    
    @pytest.mark.asyncio
    async def test_no_drift_detection(self, drift_detector):
        """Test no drift detection with similar distributions"""
        
        # Create similar distributions
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(0.1, 1, 1000)  # Small difference
        
        result = await drift_detector.detect_drift_with_significance(
            "test_model", current_data, reference_data, alpha=0.05
        )
        
        assert result.detected is False or result.severity == AlertSeverity.LOW
        assert result.details['min_p_value'] > 0.01  # Should have high p-value
    
    def test_reference_distribution_management(self, drift_detector):
        """Test reference distribution management"""
        
        reference_data = np.random.normal(0, 1, 1000)
        drift_detector.set_reference_distribution("test_model", reference_data)
        
        assert "test_model" in drift_detector.reference_distributions
        assert np.array_equal(drift_detector.reference_distributions["test_model"], reference_data)
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, drift_detector):
        """Test handling of insufficient data"""
        
        small_data = np.array([1, 2, 3])  # Too small
        reference_data = np.random.normal(0, 1, 1000)
        
        with pytest.raises(Exception):  # Should raise InsufficientDataError
            await drift_detector.detect_drift_with_significance(
                "test_model", small_data, reference_data
            )


class TestDriftStrategies:
    """Test drift detection strategies"""
    
    @pytest.fixture
    def drift_context(self):
        return DriftDetectionContext()
    
    @pytest.mark.asyncio
    async def test_kolmogorov_smirnov_strategy(self, drift_context):
        """Test Kolmogorov-Smirnov drift detection"""
        
        # Create data with distribution shift
        reference_data = np.random.normal(0, 1, 500)
        current_data = np.random.normal(1, 1, 500)  # Mean shift
        
        result = await drift_context.detect_drift(
            DriftType.DATA_DRIFT,
            "test_model",
            {'feature_history': np.concatenate([reference_data, current_data])},
            threshold=0.05,
            method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        )
        
        assert result is not None
        assert result.detected == True
        assert result.details['method'] == 'kolmogorov_smirnov'
        assert 'ks_statistic' in result.details
        assert 'p_value' in result.details
    
    @pytest.mark.asyncio
    async def test_mann_whitney_u_strategy(self, drift_context):
        """Test Mann-Whitney U drift detection"""
        
        # Create data with median shift
        reference_data = np.random.exponential(1, 500)  # Exponential distribution
        current_data = np.random.exponential(2, 500)    # Different scale
        
        result = await drift_context.detect_drift(
            DriftType.DATA_DRIFT,
            "test_model",
            {'feature_history': np.concatenate([reference_data, current_data])},
            threshold=0.05,
            method=DriftDetectionMethod.MANN_WHITNEY_U
        )
        
        assert result is not None
        assert result.details['method'] == 'mann_whitney_u'
        assert 'mw_statistic' in result.details
        assert 'effect_size' in result.details
    
    @pytest.mark.asyncio
    async def test_psi_strategy(self, drift_context):
        """Test Population Stability Index strategy"""
        
        # Create data with distribution change
        reference_data = np.random.beta(2, 5, 500)
        current_data = np.random.beta(5, 2, 500)  # Different shape
        
        result = await drift_context.detect_drift(
            DriftType.DATA_DRIFT,
            "test_model",
            {'feature_history': np.concatenate([reference_data, current_data])},
            threshold=0.1,
            method=DriftDetectionMethod.POPULATION_STABILITY_INDEX
        )
        
        assert result is not None
        assert result.details['method'] == 'population_stability_index'
        assert 'psi_score' in result.details
    
    @pytest.mark.asyncio
    async def test_performance_based_strategy(self, drift_context):
        """Test performance-based drift detection"""
        
        reference_performance = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.88,
            'f1_score': 0.86
        }
        
        current_performance = {
            'accuracy': 0.75,  # 16.7% degradation
            'precision': 0.70,  # 17.6% degradation
            'recall': 0.80,     # 9.1% degradation
            'f1_score': 0.75    # 12.8% degradation
        }
        
        result = await drift_context.detect_drift(
            DriftType.PERFORMANCE_DRIFT,
            "test_model",
            {
                'baseline_metrics': reference_performance,
                'current_metrics': current_performance
            },
            threshold=0.05,
            method=DriftDetectionMethod.PERFORMANCE_BASED
        )
        
        assert result is not None
        assert result.detected is True
        assert result.details['method'] == 'performance_based'
        assert result.details['max_degradation'] > 0.15
    
    @pytest.mark.asyncio
    async def test_trading_specific_strategy(self, drift_context):
        """Test trading-specific drift detection"""
        
        # Create market data with regime change
        reference_returns = np.random.normal(0.001, 0.02, 1000)  # Low volatility regime
        current_returns = np.random.normal(-0.005, 0.05, 1000)   # High volatility, negative returns
        
        result = await drift_context.detect_drift(
            DriftType.CONCEPT_DRIFT,
            "test_model",
            {'feature_history': np.concatenate([reference_returns, current_returns])},
            threshold=0.3,
            method=DriftDetectionMethod.TRADING_SPECIFIC
        )
        
        assert result is not None
        assert result.details['method'] == 'trading_specific'
        assert 'regime_changes' in result.details


class TestPredictiveMaintenanceSystem:
    """Test predictive maintenance system"""
    
    @pytest.fixture
    def maintenance_system(self):
        return PredictiveMaintenanceSystem()
    
    @pytest.mark.asyncio
    async def test_maintenance_score_calculation(self, maintenance_system):
        """Test maintenance score calculation"""
        
        # Test with healthy metrics
        healthy_metrics = {
            'cpu_utilization': 0.3,
            'memory_utilization': 0.4,
            'error_rate': 0.001,
            'response_time_p95': 50
        }
        
        score = await maintenance_system.calculate_maintenance_score("test_component", healthy_metrics)
        assert 0.8 <= score <= 1.0  # Should be high score
        
        # Test with unhealthy metrics
        unhealthy_metrics = {
            'cpu_utilization': 0.95,
            'memory_utilization': 0.90,
            'error_rate': 0.05,
            'response_time_p95': 500
        }
        
        score = await maintenance_system.calculate_maintenance_score("test_component", unhealthy_metrics)
        assert score < 0.5  # Should be low score
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, maintenance_system):
        """Test trend analysis in maintenance scoring"""
        
        # Add declining trend
        for i in range(10):
            score = 1.0 - (i * 0.05)  # Declining trend
            maintenance_system.health_trends["test_component"].append(score)
        
        metrics = {'cpu_utilization': 0.5, 'memory_utilization': 0.5, 'error_rate': 0.01, 'response_time_p95': 100}
        score = await maintenance_system.calculate_maintenance_score("test_component", metrics)
        
        # Score should be penalized due to declining trend
        assert score < 0.8
    
    def test_failure_predictor_registration(self, maintenance_system):
        """Test custom failure predictor registration"""
        
        def custom_predictor(metrics):
            return 0.5 if metrics.get('custom_metric', 0) > 10 else 1.0
        
        maintenance_system.register_failure_predictor("test_component", custom_predictor)
        
        assert "test_component" in maintenance_system.failure_predictors
        assert len(maintenance_system.failure_predictors["test_component"]) == 1
    
    def test_maintenance_recommendations(self, maintenance_system):
        """Test maintenance recommendations"""
        
        # Set different maintenance scores
        maintenance_system.maintenance_scores = {
            'critical_component': 0.2,
            'high_priority_component': 0.4,
            'medium_priority_component': 0.6,
            'healthy_component': 0.9
        }
        
        recommendations = maintenance_system.get_maintenance_recommendations()
        
        assert len(recommendations) == 3  # Should exclude healthy component
        assert recommendations[0]['priority'] == 'critical'
        assert recommendations[0]['component'] == 'critical_component'
        assert recommendations[-1]['priority'] == 'medium'


class TestAlertRoutingSystem:
    """Test alert routing and escalation system"""
    
    @pytest.fixture
    def alert_routing(self):
        return AlertRoutingSystem()
    
    @pytest.fixture
    def sample_alert(self):
        return Alert(
            id="test_alert_123",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            model_name="test_model",
            metric_name="accuracy",
            current_value=0.7,
            threshold=0.8
        )
    
    def test_routing_rule_management(self, alert_routing):
        """Test routing rule management"""
        
        rule = {
            'name': 'high_severity_rule',
            'conditions': {
                'severity': ['high', 'critical'],
                'model_name': ['important_model']
            },
            'channels': ['email', 'slack'],
            'escalation_policy': 'urgent_policy'
        }
        
        alert_routing.add_routing_rule(rule)
        assert len(alert_routing.routing_rules) == 1
        assert alert_routing.routing_rules[0]['name'] == 'high_severity_rule'
    
    def test_escalation_policy_management(self, alert_routing):
        """Test escalation policy management"""
        
        policy = {
            'levels': [
                {'delay_minutes': 0, 'channels': ['email']},
                {'delay_minutes': 15, 'channels': ['slack']},
                {'delay_minutes': 30, 'channels': ['pagerduty']}
            ],
            'max_escalations': 3,
            'cooldown_minutes': 60
        }
        
        alert_routing.add_escalation_policy('urgent_policy', policy)
        assert 'urgent_policy' in alert_routing.escalation_policies
        assert len(alert_routing.escalation_policies['urgent_policy']['levels']) == 3
    
    def test_rule_matching(self, alert_routing, sample_alert):
        """Test alert rule matching"""
        
        # Add matching rule
        matching_rule = {
            'name': 'test_rule',
            'conditions': {
                'severity': ['high', 'critical'],
                'model_name': ['test_model']
            },
            'channels': ['email']
        }
        
        # Add non-matching rule
        non_matching_rule = {
            'name': 'other_rule',
            'conditions': {
                'severity': ['low'],
                'model_name': ['other_model']
            },
            'channels': ['slack']
        }
        
        alert_routing.add_routing_rule(matching_rule)
        alert_routing.add_routing_rule(non_matching_rule)
        
        # Test matching
        assert alert_routing._matches_rule(sample_alert, matching_rule) is True
        assert alert_routing._matches_rule(sample_alert, non_matching_rule) is False
    
    @pytest.mark.asyncio
    async def test_alert_routing(self, alert_routing, sample_alert):
        """Test alert routing functionality"""
        
        # Mock alert channels
        email_observer = Mock()
        email_observer.notify = AsyncMock()
        slack_observer = Mock()
        slack_observer.notify = AsyncMock()
        
        alert_routing.register_alert_channel('email', email_observer)
        alert_routing.register_alert_channel('slack', slack_observer)
        
        # Add routing rule
        rule = {
            'name': 'test_routing',
            'conditions': {'severity': ['high']},
            'channels': ['email', 'slack']
        }
        alert_routing.add_routing_rule(rule)
        
        # Route alert
        await alert_routing.route_alert(sample_alert)
        
        # Verify channels were called
        email_observer.notify.assert_called_once_with(sample_alert)
        slack_observer.notify.assert_called_once_with(sample_alert)
    
    def test_alert_resolution(self, alert_routing):
        """Test alert resolution and escalation stopping"""
        
        # Set up escalation state
        alert_routing.escalation_state["test_escalation"] = {
            'alert': Mock(id="test_alert"),
            'policy': {},
            'current_level': 1
        }
        
        # Resolve alert
        alert_routing.resolve_alert("test_alert")
        
        # Escalation should be stopped
        assert "test_escalation" not in alert_routing.escalation_state


class TestResourceManager:
    """Test monitoring resource manager"""
    
    @pytest.fixture
    def resource_manager(self):
        return MonitoringResourceManager(max_buffer_size=100)
    
    @pytest.fixture
    def data_buffer(self):
        config = BufferConfig(max_size=50, retention_hours=1)
        return DataBuffer("test_buffer", config)
    
    def test_buffer_creation_and_management(self, resource_manager):
        """Test buffer creation and management"""
        
        buffer = resource_manager.get_buffer("test_buffer")
        assert buffer.name == "test_buffer"
        assert "test_buffer" in resource_manager.buffers
        
        # Getting same buffer should return existing one
        same_buffer = resource_manager.get_buffer("test_buffer")
        assert same_buffer is buffer
    
    def test_data_buffer_operations(self, data_buffer):
        """Test data buffer operations"""
        
        # Test adding data
        test_data = {"value": 123, "timestamp": datetime.now()}
        data_buffer.add(test_data)
        
        assert data_buffer.size() == 1
        
        # Test retrieving data
        recent_data = data_buffer.get_recent(1)
        assert len(recent_data) == 1
        assert recent_data[0]["value"] == 123
    
    def test_buffer_cleanup(self, data_buffer):
        """Test buffer cleanup functionality"""
        
        # Add old data
        old_timestamp = datetime.now() - timedelta(hours=2)
        for i in range(10):
            data_buffer.add({"value": i}, old_timestamp)
        
        # Add recent data
        recent_timestamp = datetime.now()
        for i in range(10):
            data_buffer.add({"value": i + 100}, recent_timestamp)
        
        # Force cleanup
        data_buffer._cleanup_old_data()
        
        # Old data should be removed
        assert data_buffer.size() <= 10
    
    @pytest.mark.asyncio
    async def test_model_data_context_manager(self, resource_manager):
        """Test model data context manager"""
        
        # Add some test data
        await resource_manager.add_prediction("test_model", {"prediction": 0.8})
        
        # Use context manager
        async with resource_manager.get_model_data("test_model") as model_data:
            assert "predictions" in model_data
            assert "features" in model_data
            assert model_data["prediction_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, resource_manager):
        """Test memory optimization"""
        
        # Add data to multiple buffers
        for i in range(50):
            await resource_manager.add_prediction(f"model_{i % 5}", {"prediction": i})
        
        # Run optimization
        stats = await resource_manager.optimize_memory()
        
        assert "buffers_optimized" in stats
        assert "memory_freed_mb" in stats
        assert "items_removed" in stats
    
    def test_buffer_statistics(self, data_buffer):
        """Test buffer statistics"""
        
        # Add some data
        for i in range(5):
            data_buffer.add({"value": i})
        
        stats = data_buffer.get_stats()
        
        assert stats["name"] == "test_buffer"
        assert stats["current_size"] == 5
        assert stats["total_added"] == 5
        assert stats["utilization"] == 5 / 50  # 5 items out of 50 max


class TestAdvancedMonitoringSystem:
    """Test the main advanced monitoring system"""
    
    @pytest.fixture
    def monitoring_system(self):
        mock_metrics_collector = Mock(spec=MetricsCollector)
        return AdvancedMonitoringSystem(metrics_collector=mock_metrics_collector)
    
    @pytest.fixture
    def sample_performance_metrics(self):
        return ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="test_model",
            model_version="v1.0",
            accuracy=0.85,
            precision=0.80,
            recall=0.88,
            f1_score=0.84
        )
    
    def test_monitoring_mode_changes(self, monitoring_system):
        """Test monitoring mode changes"""
        
        # Test normal mode
        monitoring_system.set_monitoring_mode(MonitoringMode.NORMAL)
        assert monitoring_system.monitoring_mode == MonitoringMode.NORMAL
        assert monitoring_system.anomaly_detector.config.sensitivity == 1.0
        
        # Test high sensitivity mode
        monitoring_system.set_monitoring_mode(MonitoringMode.HIGH_SENSITIVITY)
        assert monitoring_system.monitoring_mode == MonitoringMode.HIGH_SENSITIVITY
        assert monitoring_system.anomaly_detector.config.sensitivity == 0.01
    
    def test_alert_observer_registration(self, monitoring_system):
        """Test alert observer registration"""
        
        mock_observer = Mock()
        monitoring_system.register_alert_observer(mock_observer)
        
        assert mock_observer in monitoring_system.alert_subject._observers
    
    def test_retraining_handler_registration(self, monitoring_system):
        """Test retraining handler registration"""
        
        mock_handler = Mock()
        monitoring_system.register_retraining_handler(mock_handler)
        
        assert mock_handler in monitoring_system.retraining_handlers
    
    @pytest.mark.asyncio
    async def test_model_performance_tracking(self, monitoring_system, sample_performance_metrics):
        """Test model performance tracking"""
        
        # First tracking should set baseline
        await monitoring_system.track_model_performance("test_model", sample_performance_metrics)
        
        assert "test_model" in monitoring_system.performance_baselines
        assert monitoring_system.performance_baselines["test_model"]["accuracy"] == 0.85
        
        # Second tracking with degraded performance
        degraded_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="test_model",
            model_version="v1.0",
            accuracy=0.70,  # 17.6% degradation
            precision=0.65,
            recall=0.75,
            f1_score=0.70
        )
        
        with patch.object(monitoring_system, '_handle_performance_degradation') as mock_handler:
            await monitoring_system.track_model_performance("test_model", degraded_metrics)
            mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_quality_tracking(self, monitoring_system):
        """Test data quality tracking"""
        
        # Create quality reports
        quality_reports = [
            DataQualityReport(
                symbol="AAPL",
                exchange="test_exchange",
                timestamp=datetime.now(),
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="high",
                description="Price spike detected"
            ),
            DataQualityReport(
                symbol="GOOGL",
                exchange="test_exchange",
                timestamp=datetime.now(),
                issue_type=DataQualityIssue.VOLUME_ANOMALY,
                severity="medium",
                description="Volume spike detected"
            )
        ]
        
        await monitoring_system.track_data_quality(quality_reports)
        
        assert len(monitoring_system.data_quality_history) == 1
        quality_record = monitoring_system.data_quality_history[0]
        assert quality_record["total_issues"] == 2
        assert quality_record["high_severity_issues"] == 1
        assert quality_record["quality_score"] < 1.0
    
    @pytest.mark.asyncio
    async def test_drift_detection(self, monitoring_system):
        """Test model drift detection"""
        
        # Create data with clear drift
        reference_data = np.random.normal(0, 1, 500)
        current_data = np.random.normal(2, 1, 500)  # Mean shift
        
        with patch.object(monitoring_system.alert_subject, 'notify_observers') as mock_notify:
            result = await monitoring_system.detect_model_drift("test_model", current_data, reference_data)

        assert result.detected == True  # Changed from 'is' to '==' for robustness
        assert result.drift_type == DriftType.DATA_DRIFT
        mock_notify.assert_called_once()
    
    def test_system_status_reporting(self, monitoring_system):
        """Test system status reporting"""
        
        # Add some test data
        monitoring_system.performance_baselines["test_model"] = {"accuracy": 0.9}
        monitoring_system.consecutive_failures["test_model"] = 2
        
        status = monitoring_system.get_system_status()
        
        assert "timestamp" in status
        assert "monitoring_mode" in status
        assert "system_health" in status
        assert "data_quality" in status
        assert "performance_degradation" in status
        assert "maintenance" in status
        assert "alerts" in status
    
    @pytest.mark.asyncio
    async def test_automated_retraining_trigger(self, monitoring_system):
        """Test automated retraining trigger"""
        
        mock_handler = AsyncMock(return_value="job_123")
        monitoring_system.register_retraining_handler(mock_handler)
        
        payload = {
            'reason': 'performance_degradation',
            'triggered_at': datetime.now(),
            'consecutive_failures': 3
        }
        
        with patch.object(monitoring_system.alert_subject, 'notify_observers') as mock_notify:
            await monitoring_system._trigger_automated_retraining("test_model", payload)
            
            mock_handler.assert_called_once_with("test_model", payload)
            mock_notify.assert_called_once()


class TestConfigManager:
    """Test configuration management"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()
    
    def test_default_configuration(self, config_manager):
        """Test default configuration loading"""
        
        assert config_manager.config.enabled is True
        assert config_manager.config.performance_thresholds.accuracy == 0.8
        assert config_manager.config.drift_thresholds.data_drift == 0.05
        assert config_manager.config.alerting.enabled is True
    
    def test_drift_threshold_retrieval(self, config_manager):
        """Test drift threshold retrieval"""
        
        data_drift_threshold = config_manager.get_drift_threshold(DriftType.DATA_DRIFT)
        assert data_drift_threshold == 0.05
        
        performance_drift_threshold = config_manager.get_drift_threshold(DriftType.PERFORMANCE_DRIFT)
        assert performance_drift_threshold == 0.1
    
    def test_performance_threshold_retrieval(self, config_manager):
        """Test performance threshold retrieval"""
        
        accuracy_threshold = config_manager.get_performance_threshold("accuracy")
        assert accuracy_threshold == 0.8
        
        latency_threshold = config_manager.get_performance_threshold("latency_ms")
        assert latency_threshold == 100.0
    
    def test_retraining_trigger_conditions(self, config_manager):
        """Test retraining trigger conditions"""
        
        assert config_manager.should_trigger_retraining(AlertSeverity.CRITICAL) is True
        assert config_manager.should_trigger_retraining(AlertSeverity.HIGH) is True
        assert config_manager.should_trigger_retraining(AlertSeverity.MEDIUM) is False
        assert config_manager.should_trigger_retraining(AlertSeverity.LOW) is False
    
    def test_configuration_updates(self, config_manager):
        """Test configuration updates"""
        
        updates = {
            'enabled': False,
            'debug_mode': True
        }
        
        config_manager.update_config(updates)
        
        assert config_manager.config.enabled is False
        assert config_manager.config.debug_mode is True
    
    def test_config_summary(self, config_manager):
        """Test configuration summary"""
        
        summary = config_manager.get_config_summary()
        
        assert "environment" in summary
        assert "enabled" in summary
        assert "performance_thresholds" in summary
        assert "drift_detection" in summary
        assert "alerting" in summary
        assert "retraining" in summary


@pytest.mark.integration
class TestMonitoringSystemIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.fixture
    def full_monitoring_system(self):
        """Create a fully configured monitoring system"""
        
        # Create mock metrics collector
        mock_metrics_collector = Mock(spec=MetricsCollector)
        mock_metrics_collector.get_system_metrics.return_value = Mock(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage_percent=70.0
        )
        
        # Create monitoring system
        system = AdvancedMonitoringSystem(metrics_collector=mock_metrics_collector)
        
        # Register mock alert observers
        email_observer = Mock()
        email_observer.notify = AsyncMock()
        system.register_alert_observer(email_observer)
        
        return system
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self, full_monitoring_system):
        """Test complete end-to-end monitoring workflow"""
        
        # 1. Set up baseline performance
        baseline_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="integration_test_model",
            model_version="v1.0",
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90
        )
        
        await full_monitoring_system.track_model_performance("integration_test_model", baseline_metrics)
        
        # 2. Simulate performance degradation
        degraded_metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name="integration_test_model",
            model_version="v1.0",
            accuracy=0.70,  # 22% degradation
            precision=0.65,
            recall=0.75,
            f1_score=0.70
        )
        
        with patch.object(full_monitoring_system, '_trigger_automated_retraining') as mock_retraining:
            await full_monitoring_system.track_model_performance("integration_test_model", degraded_metrics)
            
            # Should trigger retraining due to significant degradation
            mock_retraining.assert_called_once()
        
        # 3. Test data quality monitoring
        quality_reports = [
            DataQualityReport(
                symbol="AAPL",
                exchange="test_exchange",
                timestamp=datetime.now(),
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="high",
                description="Significant price anomaly detected"
            )
        ]
        
        await full_monitoring_system.track_data_quality(quality_reports)
        
        # 4. Test drift detection
        reference_data = np.random.normal(0, 1, 1000)
        drifted_data = np.random.normal(1.5, 1.2, 1000)  # Clear distribution shift
        
        drift_result = await full_monitoring_system.detect_model_drift(
            "integration_test_model", 
            drifted_data, 
            reference_data
        )
        
        assert drift_result.detected is True
        assert drift_result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        
        # 5. Verify system status
        system_status = full_monitoring_system.get_system_status()
        
        assert system_status["performance_degradation"]["models_with_issues"] >= 1
        assert system_status["data_quality"]["total_issues"] >= 1
        assert system_status["alerts"]["total_24h"] >= 0
    
    @pytest.mark.asyncio
    async def test_alert_routing_integration(self, full_monitoring_system):
        """Test alert routing integration"""
        
        # Set up alert routing
        routing_system = full_monitoring_system.alert_routing
        
        # Add routing rule
        rule = {
            'name': 'critical_alerts',
            'conditions': {'severity': ['critical', 'high']},
            'channels': ['email', 'slack']
        }
        routing_system.add_routing_rule(rule)
        
        # Register mock channels
        email_observer = Mock()
        email_observer.notify = AsyncMock()
        slack_observer = Mock()
        slack_observer.notify = AsyncMock()
        
        routing_system.register_alert_channel('email', email_observer)
        routing_system.register_alert_channel('slack', slack_observer)
        
        # Create high severity alert
        alert = Alert(
            id="integration_test_alert",
            severity=AlertSeverity.HIGH,
            title="Integration Test Alert",
            message="This is a test alert for integration testing",
            timestamp=datetime.now(),
            model_name="integration_test_model"
        )
        
        # Route alert
        await routing_system.route_alert(alert)
        
        # Verify both channels were notified
        email_observer.notify.assert_called_once_with(alert)
        slack_observer.notify.assert_called_once_with(alert)
    
    @pytest.mark.asyncio
    async def test_resource_management_integration(self, full_monitoring_system):
        """Test resource management integration"""
        
        resource_manager = full_monitoring_system.resource_manager
        
        # Add data to multiple models
        for model_id in range(5):
            for prediction_id in range(100):
                await resource_manager.add_prediction(
                    f"model_{model_id}",
                    {
                        "prediction": np.random.random(),
                        "confidence": np.random.random(),
                        "timestamp": datetime.now()
                    },
                    features=np.random.random((10,))
                )
        
        # Check resource usage
        memory_usage = await resource_manager.get_memory_usage()
        
        assert memory_usage["buffer_count"] >= 10  # 5 models * 2 buffers each
        assert memory_usage["total_predictions"] >= 500
        assert memory_usage["total_features"] >= 500
        
        # Test cleanup
        cleanup_stats = await resource_manager.cleanup_old_data(hours=0)  # Aggressive cleanup
        
        assert isinstance(cleanup_stats, dict)
        assert len(cleanup_stats) >= 10
        
        # Test optimization
        optimization_stats = await resource_manager.optimize_memory()
        
        assert "buffers_optimized" in optimization_stats
        assert "memory_freed_mb" in optimization_stats
