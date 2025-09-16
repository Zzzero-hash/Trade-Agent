"""Integration between feature extraction monitoring and Ray Serve monitoring.

This module provides seamless integration between the feature extraction
performance monitoring system and the existing Ray Serve monitoring
infrastructure to provide unified performance visibility.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from src.ml.ray_serve.monitoring import get_metrics_collector as get_ray_metrics_collector
from src.ml.ray_serve.monitoring import PerformanceMonitor as RayPerformanceMonitor
from src.ml.feature_extraction.monitoring import FeatureExtractionMetrics
from src.ml.feature_extraction.enhanced_metrics import EnhancedMetricsCollector
from src.ml.feature_extraction.alerting import FeatureExtractionAlertingSystem

logger = logging.getLogger(__name__)


class RayServeIntegration:
    """Integration between feature extraction monitoring and Ray Serve monitoring"""
    
    def __init__(self):
        """Initialize Ray Serve integration."""
        self.ray_metrics_collector = get_ray_metrics_collector()
        self.ray_performance_monitor = RayPerformanceMonitor()
        self.enhanced_metrics_collector = EnhancedMetricsCollector()
        self.alerting_system = FeatureExtractionAlertingSystem(self.enhanced_metrics_collector)
        
        # Integration state
        self.is_initialized = False
        self.last_sync_time = 0
        self.sync_interval = 10  # Sync every 10 seconds
        
        logger.info("Ray Serve integration for feature extraction monitoring initialized")
    
    def initialize_integration(self) -> None:
        """Initialize the integration between monitoring systems."""
        if self.is_initialized:
            logger.warning("Ray Serve integration already initialized")
            return
        
        # Verify Ray metrics collector is available
        if self.ray_metrics_collector is None:
            logger.warning("Ray metrics collector not available, integration will be limited")
        
        self.is_initialized = True
        logger.info("Ray Serve integration initialized successfully")
    
    def sync_feature_extraction_metrics(self, metrics: FeatureExtractionMetrics) -> None:
        """Sync feature extraction metrics with Ray Serve monitoring.
        
        Args:
            metrics: Feature extraction metrics to sync
        """
        if not self.is_initialized:
            self.initialize_integration()
        
        # Add to enhanced metrics collector
        self.enhanced_metrics_collector.add_feature_extraction_metrics(metrics)
        
        # Sync with Ray metrics collector if available
        if self.ray_metrics_collector:
            try:
                # Record feature extraction latency
                self.ray_metrics_collector.record_histogram(
                    "feature_extraction_latency_seconds",
                    metrics.duration_ms / 1000.0,  # Convert to seconds
                    {"used_cache": str(metrics.used_cache)}
                )
                
                # Record resource usage
                self.ray_metrics_collector.set_gauge(
                    "feature_extraction_memory_mb",
                    metrics.memory_mb,
                    {"batch_size": str(metrics.batch_size)}
                )
                
                if metrics.gpu_utilization > 0:
                    self.ray_metrics_collector.set_gauge(
                        "feature_extraction_gpu_utilization_percent",
                        metrics.gpu_utilization
                    )
                
                # Record errors
                if metrics.had_error:
                    self.ray_metrics_collector.increment_counter(
                        "feature_extraction_errors_total",
                        {"error_type": "feature_extraction"}
                    )
                
                # Record cache usage
                if metrics.used_cache:
                    self.ray_metrics_collector.increment_counter(
                        "feature_extraction_cache_hits_total"
                    )
                else:
                    self.ray_metrics_collector.increment_counter(
                        "feature_extraction_cache_misses_total"
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to sync metrics with Ray collector: {e}")
        
        # Update Ray performance monitor
        self.ray_performance_monitor.record_request(
            metrics.duration_ms,
            success=not metrics.had_error
        )
        
        # Check for alerts
        current_time = time.time()
        if current_time - self.last_sync_time > self.sync_interval:
            self._check_and_send_alerts()
            self.last_sync_time = current_time
    
    def _check_and_send_alerts(self) -> None:
        """Check for alerts and send them through the alerting system."""
        try:
            alerts = self.alerting_system.check_performance_alerts()
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts for feature extraction performance")
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def get_unified_performance_stats(self) -> Dict[str, Any]:
        """Get unified performance statistics from both systems.
        
        Returns:
            Dictionary with unified performance statistics
        """
        # Get feature extraction stats
        feature_stats = self.enhanced_metrics_collector.get_performance_summary()
        
        # Get Ray Serve stats
        ray_stats = self.ray_performance_monitor.get_performance_stats()
        
        # Get performance requirements status
        requirements_status = self.enhanced_metrics_collector.meets_performance_requirements()
        
        # Combine stats
        unified_stats = {
            'timestamp': datetime.now().isoformat(),
            'feature_extraction': feature_stats,
            'ray_serve': ray_stats,
            'performance_requirements': requirements_status,
            'alert_summary': self.alerting_system.get_alert_summary()
        }
        
        return unified_stats
    
    def get_performance_requirements_status(self) -> Dict[str, Any]:
        """Get status of performance requirements from integrated systems.
        
        Returns:
            Dictionary with performance requirement status
        """
        # Get feature extraction requirements status
        feature_requirements = self.enhanced_metrics_collector.meets_performance_requirements()
        
        # Get Ray Serve requirements status
        ray_requirements = self.ray_performance_monitor.check_performance_requirements()
        
        # Combine requirements status
        combined_status = {
            'feature_extraction_meets_10ms': feature_requirements.get('meets_100ms_requirement', False),
            'feature_extraction_avg_latency_ms': feature_requirements.get('current_avg_latency_ms', 0),
            'feature_extraction_p95_latency_ms': feature_requirements.get('current_p95_latency_ms', 0),
            'ray_serve_meets_100ms': ray_requirements.get('meets_100ms_requirement', False),
            'ray_serve_avg_latency_ms': ray_requirements.get('avg_latency_ms', 0),
            'overall_meets_100ms_requirement': (
                feature_requirements.get('meets_100ms_requirement', False) and
                ray_requirements.get('meets_100ms_requirement', False)
            ),
            'latency_margin_ms': min(
                feature_requirements.get('latency_margin_ms', 0),
                ray_requirements.get('latency_margin_ms', 0)
            )
        }
        
        return combined_status
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get combined resource utilization from both systems.
        
        Returns:
            Dictionary with combined resource utilization metrics
        """
        # Get feature extraction resource utilization
        feature_resources = self.enhanced_metrics_collector.get_resource_utilization()
        
        # Get Ray Serve system health
        ray_health = self.ray_performance_monitor.get_system_health() if hasattr(self.ray_performance_monitor, 'get_system_health') else {}
        
        # Combine resource metrics
        combined_resources = {
            'feature_extraction': feature_resources,
            'ray_serve': ray_health,
            'timestamp': datetime.now().isoformat()
        }
        
        return combined_resources
    
    def reset_monitoring(self) -> None:
        """Reset all monitoring systems."""
        # Reset enhanced metrics collector
        self.enhanced_metrics_collector.reset_metrics()
        
        # Reset alerting system
        self.alerting_system.clear_alert_history()
        
        # Reset Ray performance monitor
        if hasattr(self.ray_performance_monitor, 'reset'):
            self.ray_performance_monitor.reset()
        
        logger.info("All monitoring systems reset")
    
    async def start_background_monitoring(self) -> None:
        """Start background monitoring task."""
        async def monitoring_loop():
            while True:
                try:
                    # Check for alerts periodically
                    self._check_and_send_alerts()
                    
                    # Wait for next check
                    await asyncio.sleep(self.sync_interval)
                    
                except Exception as e:
                    logger.error(f"Error in background monitoring loop: {e}")
                    await asyncio.sleep(5)  # Short sleep on error
        
        # Start monitoring loop in background
        asyncio.create_task(monitoring_loop())
        logger.info("Background monitoring started")
    
    def get_monitoring_health(self) -> Dict[str, Any]:
        """Get health status of the monitoring integration.
        
        Returns:
            Dictionary with monitoring health status
        """
        return {
            'is_initialized': self.is_initialized,
            'ray_metrics_available': self.ray_metrics_collector is not None,
            'enhanced_metrics_collector_size': len(self.enhanced_metrics_collector.metrics_history),
            'alert_history_size': len(self.alerting_system.alert_history),
            'last_sync_time': self.last_sync_time,
            'sync_interval': self.sync_interval
        }