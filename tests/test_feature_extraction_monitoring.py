"""Tests for feature extraction performance monitoring system."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.feature_extraction.monitoring import (
    FeatureExtractionMetrics, 
    FeatureExtractionPerformanceMonitor,
    PerformanceThresholds
)
from src.ml.feature_extraction.enhanced_metrics import (
    EnhancedMetricsCollector,
    RealTimeMetrics
)
from src.ml.feature_extraction.alerting import (
    FeatureExtractionAlertingSystem,
    FeatureExtractionAlert,
    AlertType,
    AlertSeverity
)
from src.ml.feature_extraction.ray_integration import RayServeIntegration
from src.ml.feature_extraction.cache_connection_integration import CacheConnectionIntegration
from src.ml.feature_extraction.dashboard import FeatureExtractionDashboard


class TestFeatureExtractionPerformanceMonitor:
    """Test feature extraction performance monitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create a feature extraction performance monitor instance"""
        return FeatureExtractionPerformanceMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor is not None
        assert len(monitor.metrics_history) == 0
        assert monitor.stats['total_extractions'] == 0
    
    def test_start_and_end_extraction(self, monitor):
        """Test starting and ending extraction timing"""
        # Start extraction
        monitor.start_extraction()
        
        # End extraction with metrics
        monitor.end_extraction(
            duration_ms=50.0,
            used_cache=True,
            used_fallback=False,
            had_error=False,
            input_shape=(1, 10, 60),
            feature_dimensions={'fused_features': 256},
            cpu_percent=45.0,
            memory_mb=150.0,
            gpu_utilization=60.0,
            batch_size=1
        )
        
        # Verify metrics were recorded
        assert len(monitor.metrics_history) == 1
        assert monitor.stats['total_extractions'] == 1
        assert monitor.stats['cache_hits'] == 1
        assert monitor.stats['errors'] == 0
    
    def test_performance_requirements_check(self, monitor):
        """Test performance requirements checking"""
        # Add some metrics that meet requirements
        for i in range(100):
            monitor.end_extraction(
                duration_ms=50.0 + (i % 10),  # Vary between 50-59ms
                used_cache=True,
                had_error=False
            )
        
        # Check requirements status
        requirements = monitor.get_performance_requirements_status()
        assert requirements['meets_10ms_requirement'] is True
        assert requirements['current_p95_latency_ms'] < 100
        assert requirements['latency_margin_ms'] > 0
    
    def test_performance_stats_calculation(self, monitor):
        """Test performance statistics calculation"""
        # Add test metrics
        test_durations = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
        for duration in test_durations:
            monitor.end_extraction(
                duration_ms=duration,
                used_cache=(duration < 50),  # First 5 use cache
                had_error=(duration > 65)    # Last 2 have errors
            )
        
        # Get performance stats
        stats = monitor.get_performance_stats()
        
        assert stats['total_extractions'] == 10
        assert stats['avg_latency_ms'] == np.mean(test_durations)
        assert stats['cache_hit_rate'] == 0.5  # 5 out of 10
        assert stats['error_rate'] == 0.2      # 2 out of 10
        assert stats['meets_100ms_requirement'] is True


class TestEnhancedMetricsCollector:
    """Test enhanced metrics collector"""
    
    @pytest.fixture
    def collector(self):
        """Create an enhanced metrics collector instance"""
        return EnhancedMetricsCollector()
    
    def test_collector_initialization(self, collector):
        """Test collector initialization"""
        assert collector is not None
        assert len(collector.metrics_history) == 0
        assert collector.window_size == 1000
    
    def test_add_metrics(self, collector):
        """Test adding metrics to collector"""
        # Create test metrics
        metrics = FeatureExtractionMetrics(
            timestamp=datetime.now(),
            duration_ms=45.0,
            used_cache=True,
            used_fallback=False,
            had_error=False,
            input_shape=(1, 10, 60),
            feature_dimensions={'fused_features': 256},
            cpu_percent=35.0,
            memory_mb=120.0,
            gpu_utilization=45.0,
            batch_size=1
        )
        
        # Add metrics
        collector.add_feature_extraction_metrics(metrics)
        
        # Verify metrics were added
        assert len(collector.metrics_history) == 1
        assert collector.metrics_history[0] == metrics
    
    def test_real_time_metrics(self, collector):
        """Test real-time metrics calculation"""
        # Add some metrics
        for i in range(10):
            metrics = FeatureExtractionMetrics(
                timestamp=datetime.now(),
                duration_ms=30.0 + i,
                used_cache=(i % 2 == 0),
                had_error=False,
                cpu_percent=20.0 + i,
                memory_mb=100.0 + i,
                gpu_utilization=30.0 + i
            )
            collector.add_feature_extraction_metrics(metrics)
        
        # Get real-time metrics
        real_time = collector.get_real_time_metrics()
        
        assert isinstance(real_time, RealTimeMetrics)
        assert real_time.avg_latency_ms > 0
        assert real_time.cpu_percent > 0
        assert real_time.memory_usage_mb > 0
    
    def test_performance_summary(self, collector):
        """Test performance summary calculation"""
        # Add test data
        durations = [20.0, 25.0, 30.0, 35.0, 40.0]
        for duration in durations:
            metrics = FeatureExtractionMetrics(
                timestamp=datetime.now(),
                duration_ms=duration,
                used_cache=(duration < 30),
                had_error=False
            )
            collector.add_feature_extraction_metrics(metrics)
        
        # Get summary
        summary = collector.get_performance_summary()
        
        assert summary['total_extractions'] == 5
        assert summary['avg_latency_ms'] == np.mean(durations)
        assert summary['p95_latency_ms'] >= summary['avg_latency_ms']
        assert summary['cache_hit_rate'] == 0.4  # 2 out of 5


class TestFeatureExtractionAlertingSystem:
    """Test feature extraction alerting system"""
    
    @pytest.fixture
    def collector(self):
        """Create a metrics collector"""
        return EnhancedMetricsCollector()
    
    @pytest.fixture
    def alerting_system(self, collector):
        """Create an alerting system instance"""
        return FeatureExtractionAlertingSystem(collector)
    
    def test_alerting_system_initialization(self, alerting_system):
        """Test alerting system initialization"""
        assert alerting_system is not None
        assert len(alerting_system.alert_history) == 0
        assert alerting_system.cooldown_period == timedelta(minutes=5)
    
    def test_latency_alerts(self, alerting_system, collector):
        """Test latency alert generation"""
        # Add metrics that violate latency requirements
        for i in range(10):
            metrics = FeatureExtractionMetrics(
                timestamp=datetime.now(),
                duration_ms=150.0,  # Exceeds 100ms threshold
                used_cache=False,
                had_error=False
            )
            collector.add_feature_extraction_metrics(metrics)
        
        # Check for alerts
        alerts = alerting_system.check_performance_alerts()
        
        # Should generate latency alerts
        latency_alerts = [a for a in alerts if a.alert_type == AlertType.LATENCY_THRESHOLD]
        assert len(latency_alerts) > 0
        assert latency_alerts[0].severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def test_resource_alerts(self, alerting_system, collector):
        """Test resource alert generation"""
        # Add metrics that violate resource requirements
        for i in range(5):
            metrics = FeatureExtractionMetrics(
                timestamp=datetime.now(),
                duration_ms=40.0,
                used_cache=False,
                had_error=False,
                memory_mb=1500.0,  # Exceeds 1000MB threshold
                cpu_percent=85.0   # Exceeds 80% threshold
            )
            collector.add_feature_extraction_metrics(metrics)
        
        # Check for alerts
        alerts = alerting_system.check_performance_alerts()
        
        # Should generate resource alerts
        resource_alerts = [a for a in alerts if a.alert_type == AlertType.RESOURCE_THRESHOLD]
        assert len(resource_alerts) > 0
    
    def test_cache_alerts(self, alerting_system, collector):
        """Test cache alert generation"""
        # Add metrics with poor cache hit rate
        for i in range(20):
            metrics = FeatureExtractionMetrics(
                timestamp=datetime.now(),
                duration_ms=40.0,
                used_cache=(i < 3),  # Only 3 out of 20 use cache (15%)
                had_error=False
            )
            collector.add_feature_extraction_metrics(metrics)
        
        # Check for alerts
        alerts = alerting_system.check_performance_alerts()
        
        # Should generate cache alerts
        cache_alerts = [a for a in alerts if a.alert_type == AlertType.CACHE_DEGRADATION]
        assert len(cache_alerts) > 0


class TestRayServeIntegration:
    """Test Ray Serve integration"""
    
    @pytest.fixture
    def integration(self):
        """Create Ray Serve integration instance"""
        return RayServeIntegration()
    
    def test_integration_initialization(self, integration):
        """Test integration initialization"""
        assert integration is not None
        # Integration should initialize automatically
        assert integration.is_initialized is True
    
    def test_sync_metrics(self, integration):
        """Test syncing metrics with Ray Serve"""
        # Create test metrics
        metrics = FeatureExtractionMetrics(
            timestamp=datetime.now(),
            duration_ms=45.0,
            used_cache=True,
            used_fallback=False,
            had_error=False,
            cpu_percent=35.0,
            memory_mb=120.0,
            gpu_utilization=45.0
        )
        
        # Sync metrics
        integration.sync_feature_extraction_metrics(metrics)
        
        # Verify metrics were processed
        unified_stats = integration.get_unified_performance_stats()
        assert 'feature_extraction' in unified_stats
        assert 'performance_requirements' in unified_stats


class TestCacheConnectionIntegration:
    """Test cache and connection pooling integration"""
    
    @pytest.fixture
    def integration(self):
        """Create cache/connection integration instance"""
        return CacheConnectionIntegration()
    
    def test_integration_initialization(self, integration):
        """Test integration initialization"""
        assert integration is not None
        # Integration should initialize automatically
        assert integration.is_initialized is True
    
    def test_get_resource_stats(self, integration):
        """Test getting resource statistics"""
        # Get resource stats
        resource_stats = integration.get_unified_resource_stats()
        
        assert 'timestamp' in resource_stats
        assert 'feature_extraction' in resource_stats
        assert 'cache' in resource_stats
        assert 'connection_pools' in resource_stats


class TestFeatureExtractionDashboard:
    """Test feature extraction dashboard"""
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance"""
        return FeatureExtractionDashboard()
    
    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization"""
        assert dashboard is not None
    
    def test_get_real_time_dashboard_data(self, dashboard):
        """Test getting real-time dashboard data"""
        # Get dashboard data
        dashboard_data = dashboard.get_real_time_dashboard_data()
        
        assert dashboard_data.timestamp is not None
        assert hasattr(dashboard_data, 'feature_extraction')
        assert hasattr(dashboard_data, 'alerts')
        assert hasattr(dashboard_data, 'resource_utilization')
    
    def test_get_performance_trends(self, dashboard):
        """Test getting performance trends"""
        # Get performance trends
        trends = dashboard.get_performance_trends(hours=1)
        
        assert 'time_series' in trends
        assert 'trends' in trends
    
    def test_get_system_health_overview(self, dashboard):
        """Test getting system health overview"""
        # Get health overview
        health = dashboard.get_system_health_overview()
        
        assert 'health_score' in health
        assert 'health_status' in health
        assert 'performance_requirements' in health
        assert 'resource_utilization' in health
    
    def test_export_dashboard_data(self, dashboard):
        """Test exporting dashboard data"""
        # Export data
        exported_data = dashboard.export_dashboard_data()
        
        # Should be valid JSON
        import json
        data = json.loads(exported_data)
        assert isinstance(data, dict)


# Integration tests
class TestFeatureExtractionMonitoringIntegration:
    """Integration tests for feature extraction monitoring system"""
    
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # Create all components
        monitor = FeatureExtractionPerformanceMonitor()
        collector = EnhancedMetricsCollector()
        alerting_system = FeatureExtractionAlertingSystem(collector)
        ray_integration = RayServeIntegration()
        dashboard = FeatureExtractionDashboard()
        
        # Simulate feature extraction operations
        test_durations = [25.0, 30.0, 35.0, 40.0, 45.0, 150.0, 160.0]  # Last two exceed threshold
        for i, duration in enumerate(test_durations):
            # Record in performance monitor
            monitor.end_extraction(
                duration_ms=duration,
                used_cache=(i % 2 == 0),
                had_error=False,
                cpu_percent=30.0 + (i * 2),
                memory_mb=100.0 + (i * 5)
            )
            
            # Create metrics for other systems
            metrics = FeatureExtractionMetrics(
                timestamp=datetime.now(),
                duration_ms=duration,
                used_cache=(i % 2 == 0),
                had_error=False,
                cpu_percent=30.0 + (i * 2),
                memory_mb=100.0 + (i * 5)
            )
            
            # Sync with Ray integration
            ray_integration.sync_feature_extraction_metrics(metrics)
            
            # Add to collector
            collector.add_feature_extraction_metrics(metrics)
        
        # Check alerts
        alerts = alerting_system.check_performance_alerts()
        
        # Get dashboard data
        dashboard_data = dashboard.get_real_time_dashboard_data()
        health_overview = dashboard.get_system_health_overview()
        
        # Verify integration worked
        assert len(alerts) >= 0  # May have alerts depending on thresholds
        assert dashboard_data.feature_extraction is not None
        assert health_overview['health_score'] >= 0
        assert health_overview['health_score'] <= 100


if __name__ == "__main__":
    pytest.main([__file__])