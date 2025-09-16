"""Performance monitoring for feature extraction operations.

This module provides comprehensive performance monitoring with
real-time metrics collection, alerting mechanisms, and integration
with existing monitoring systems to ensure <100ms feature extraction
requirement is met.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import numpy as np

from src.utils.monitoring import get_metrics_collector, MetricsCollector
from src.services.monitoring.alert_system import AlertSubject, AlertFactory, AlertSeverity
from src.services.monitoring.config import ConfigManager
from src.ml.ray_serve.monitoring import get_metrics_collector as get_ray_metrics_collector
from src.ml.feature_extraction.metrics import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionMetrics:
    """Metrics for feature extraction performance monitoring"""
    timestamp: datetime
    duration_ms: float
    used_cache: bool = False
    used_fallback: bool = False
    had_error: bool = False
    input_shape: Optional[tuple] = None
    feature_dimensions: Optional[Dict[str, int]] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    batch_size: int = 1


@dataclass
class PerformanceThresholds:
    """Performance thresholds for feature extraction"""
    max_latency_ms: float = 100.0  # <100ms requirement
    warning_latency_ms: float = 50.0  # Warning at 50ms
    critical_latency_ms: float = 80.0  # Critical at 80ms
    min_cache_hit_rate: float = 0.7  # Minimum 70% cache hit rate
    max_error_rate: float = 0.01  # Maximum 1% error rate
    max_memory_mb: float = 1000.0  # Maximum 1GB memory usage


from src.services.monitoring.alert_system import AlertSubject, AlertFactory, AlertSeverity

class FeatureExtractionPerformanceMonitor:
    """Monitors and tracks feature extraction performance with real-time alerting"""
    
    def __init__(self, alert_subject: AlertSubject, config_manager: Optional[ConfigManager] = None):
        """Initialize performance monitor.
        
        Args:
            alert_subject: The central AlertSubject instance for sending alerts.
            config_manager: Configuration manager for monitoring settings
        """
        self.config_manager = config_manager or ConfigManager()
        self.metrics_collector = get_metrics_collector()
        self.ray_metrics_collector = get_ray_metrics_collector()
        self.alert_system = alert_subject  # Use the central AlertSubject
        self.performance_tracker = PerformanceTracker()
        
        # Performance tracking
        self.metrics_history: List[FeatureExtractionMetrics] = []
        self.window_size = 1000  # Keep last 100 metrics
        self.thresholds = PerformanceThresholds()
        
        # Alerting configuration (removed internal cooldown, handled by AlertSubject)
        
        # Performance statistics
        self.stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'fallbacks': 0,
            'errors': 0,
            'total_latency_ms': 0.0,
            'peak_latency_ms': 0.0
        }
        
        logger.info("Feature extraction performance monitor initialized")
    
    def start_extraction(self) -> None:
        """Start timing a feature extraction operation."""
        self.performance_tracker.start_extraction()
    
    def end_extraction(
        self,
        duration_ms: float,
        used_cache: bool = False,
        used_fallback: bool = False,
        had_error: bool = False,
        input_shape: Optional[tuple] = None,
        feature_dimensions: Optional[Dict[str, int]] = None,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        gpu_utilization: float = 0.0,
        batch_size: int = 1
    ) -> None:
        """End timing and record metrics for feature extraction.
        
        Args:
            duration_ms: Duration of extraction in milliseconds
            used_cache: Whether cache was used
            used_fallback: Whether fallback was used
            had_error: Whether an error occurred
            input_shape: Shape of input data
            feature_dimensions: Dimensions of extracted features
            cpu_percent: CPU utilization percentage
            memory_mb: Memory usage in MB
            gpu_utilization: GPU utilization percentage
            batch_size: Size of batch processed
        """
        # Update performance tracker
        self.performance_tracker.end_extraction(used_cache, used_fallback, had_error)
        
        # Create metrics record
        metrics = FeatureExtractionMetrics(
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            used_cache=used_cache,
            used_fallback=used_fallback,
            had_error=had_error,
            input_shape=input_shape,
            feature_dimensions=feature_dimensions,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_utilization=gpu_utilization,
            batch_size=batch_size
        )
        
        # Add to history with window management
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
        
        # Update statistics
        self._update_statistics(metrics)
        
        # Record metrics
        self._record_metrics(metrics)
        
        # Check performance thresholds and send alerts if needed
        self._check_performance_thresholds(metrics)
        
        # Log periodically
        self.stats['total_extractions'] += 1
        if self.stats['total_extractions'] % 100 == 0:
            self._log_performance_summary()
    
    def _update_statistics(self, metrics: FeatureExtractionMetrics) -> None:
        """Update performance statistics.
        
        Args:
            metrics: Feature extraction metrics to update statistics with
        """
        self.stats['total_latency_ms'] += metrics.duration_ms
        self.stats['peak_latency_ms'] = max(self.stats['peak_latency_ms'], metrics.duration_ms)
        
        if metrics.used_cache:
            self.stats['cache_hits'] += 1
        if metrics.used_fallback:
            self.stats['fallbacks'] += 1
        if metrics.had_error:
            self.stats['errors'] += 1
    
    def _record_metrics(self, metrics: FeatureExtractionMetrics) -> None:
        """Record metrics to monitoring systems.
        
        Args:
            metrics: Feature extraction metrics to record
        """
        # Record to general metrics collector
        if self.metrics_collector:
            self.metrics_collector.record_timer(
                "feature_extraction_duration_ms",
                metrics.duration_ms,
                {"used_cache": str(metrics.used_cache)}
            )
            
            self.metrics_collector.set_gauge(
                "feature_extraction_memory_mb",
                metrics.memory_mb
            )
            
            self.metrics_collector.set_gauge(
                "feature_extraction_cpu_percent",
                metrics.cpu_percent
            )
            
            if metrics.gpu_utilization > 0:
                self.metrics_collector.set_gauge(
                    "feature_extraction_gpu_utilization",
                    metrics.gpu_utilization
                )
        
        # Record to Ray metrics collector if available
        if self.ray_metrics_collector:
            self.ray_metrics_collector.record_histogram(
                "feature_extraction_latency_seconds",
                metrics.duration_ms / 1000.0,
                {"model_version": "cnn_lstm"}
            )
            
            if metrics.had_error:
                self.ray_metrics_collector.increment_counter(
                    "feature_extraction_errors_total",
                    {"model_version": "cnn_lstm", "error_type": "feature_extraction"}
                )
    
    def _check_performance_thresholds(self, metrics: FeatureExtractionMetrics) -> None:
        """Check if performance metrics exceed thresholds and send alerts.
        
        Args:
            metrics: Feature extraction metrics to check
        """
        # Check latency threshold
        if metrics.duration_ms > self.thresholds.max_latency_ms:
            self._send_performance_alert(
                "Feature extraction latency exceeded maximum threshold",
                metrics.duration_ms,
                self.thresholds.max_latency_ms,
                AlertSeverity.CRITICAL
            )
        elif metrics.duration_ms > self.thresholds.critical_latency_ms:
            self._send_performance_alert(
                "Feature extraction latency exceeded critical threshold",
                metrics.duration_ms,
                self.thresholds.critical_latency_ms,
                AlertSeverity.HIGH
            )
        elif metrics.duration_ms > self.thresholds.warning_latency_ms:
            self._send_performance_alert(
                "Feature extraction latency exceeded warning threshold",
                metrics.duration_ms,
                self.thresholds.warning_latency_ms,
                AlertSeverity.MEDIUM
            )
        
        # Check resource usage
        if metrics.memory_mb > self.thresholds.max_memory_mb:
            self._send_resource_alert(
                "Feature extraction memory usage exceeded threshold",
                "memory_mb",
                metrics.memory_mb,
                self.thresholds.max_memory_mb,
                AlertSeverity.HIGH
            )
    
    def _send_performance_alert(
        self,
        message: str,
        current_value: float,
        threshold: float,
        severity: AlertSeverity
    ) -> None:
        """Send performance alert.
        
        Args:
            message: Alert message
            current_value: Current metric value
            threshold: Threshold value
            severity: Alert severity
        """
        alert = AlertFactory.create_performance_alert(
            "feature_extraction",
            "latency",
            current_value,
            threshold
        )
        alert.severity = severity
        alert.message = message
        self._send_alert(alert)
    
    def _send_resource_alert(
        self,
        message: str,
        resource_type: str,
        current_value: float,
        threshold: float,
        severity: AlertSeverity
    ) -> None:
        """Send resource usage alert.
        
        Args:
            message: Alert message
            resource_type: Type of resource (memory, cpu, etc.)
            current_value: Current metric value
            threshold: Threshold value
            severity: Alert severity
        """
        alert = AlertFactory.create_performance_alert(
            "feature_extraction",
            resource_type,
            current_value,
            threshold
        )
        alert.severity = severity
        alert.message = message
        self._send_alert(alert)
    
    def _send_alert(self, alert) -> None:
        """Send alert with cooldown management.
        
        Args:
            alert: Alert to send
        """
        # Send through alert system
        asyncio.create_task(self.alert_system.notify_observers(alert))
        
        # Record alert metric
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "feature_extraction_alerts_total",
                tags={
                    "severity": alert.severity.value,
                    "metric": alert.metric_name or "unknown"
                }
            )
        
        logger.warning(f"Performance alert: {alert.title} - {alert.message}")
    
    def _log_performance_summary(self) -> None:
        """Log performance summary statistics."""
        if not self.metrics_history:
            return
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 extractions
        avg_latency = np.mean([m.duration_ms for m in recent_metrics])
        max_latency = np.max([m.duration_ms for m in recent_metrics])
        cache_hit_rate = np.mean([1 if m.used_cache else 0 for m in recent_metrics])
        error_rate = np.mean([1 if m.had_error else 0 for m in recent_metrics])
        
        logger.info("Feature Extraction Performance Summary (Last 100):")
        logger.info(f"  Average Latency: {avg_latency:.2f}ms")
        logger.info(f"  Max Latency: {max_latency:.2f}ms")
        logger.info(f"  Cache Hit Rate: {cache_hit_rate:.2%}")
        logger.info(f"  Error Rate: {error_rate:.2%}")
        logger.info(f"  Total Extractions: {self.stats['total_extractions']}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-1000:]  # Last 1000 extractions
        durations = [m.duration_ms for m in recent_metrics]
        
        stats = {
            'total_extractions': self.stats['total_extractions'],
            'avg_latency_ms': np.mean(durations) if durations else 0,
            'median_latency_ms': np.median(durations) if durations else 0,
            'p95_latency_ms': np.percentile(durations, 95) if len(durations) >= 20 else 0,
            'p99_latency_ms': np.percentile(durations, 99) if len(durations) >= 100 else 0,
            'max_latency_ms': np.max(durations) if durations else 0,
            'min_latency_ms': np.min(durations) if durations else 0,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_extractions'], 1),
            'fallback_rate': self.stats['fallbacks'] / max(self.stats['total_extractions'], 1),
            'error_rate': self.stats['errors'] / max(self.stats['total_extractions'], 1),
            'avg_memory_mb': np.mean([m.memory_mb for m in recent_metrics]),
            'peak_memory_mb': np.max([m.memory_mb for m in recent_metrics]),
            'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
            'peak_cpu_percent': np.max([m.cpu_percent for m in recent_metrics])
        }
        
        # Add performance requirement status
        stats['meets_100ms_requirement'] = stats['p95_latency_ms'] < 100
        stats['meets_50ms_target'] = stats['avg_latency_ms'] < 50
        
        return stats
    
    def get_performance_requirements_status(self) -> Dict[str, Any]:
        """Get status of performance requirements.
        
        Returns:
            Dictionary with performance requirement status
        """
        stats = self.get_performance_stats()
        
        return {
            'meets_10ms_requirement': stats.get('p95_latency_ms', 0) < 100,
            'meets_50ms_target': stats.get('avg_latency_ms', 0) < 50,
            'current_avg_latency_ms': stats.get('avg_latency_ms', 0),
            'current_p95_latency_ms': stats.get('p95_latency_ms', 0),
            'target_latency_ms': 100,
            'warning_latency_ms': 50,
            'latency_margin_ms': 100 - stats.get('p95_latency_ms', 0)
        }
    
    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.metrics_history.clear()
        self.stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'fallbacks': 0,
            'errors': 0,
            'total_latency_ms': 0.0,
            'peak_latency_ms': 0.0
        }
        self.performance_tracker.reset_metrics()
        logger.info("Performance statistics reset")